"""
One sync cycle, assembled from the parts.

THE LOOP
--------
    1. PULL the change feed and update our mirror of the server's metadata.
    2. SCAN the local tree (cheaply, via the hash cache).
    3. RECONCILE local / base / remote into a plan. Pure function, no I/O.
    4. EXECUTE the plan, updating BASE after each action succeeds.

Three properties make this survivable:

IDEMPOTENCE. Every step can be re-run. If we crash between 3 and 4, or halfway
through 4, the next cycle rescans and re-plans from whatever state actually
exists. There is no "resume from step 7" logic anywhere, because the plan is
always derived fresh from the world rather than remembered. This is the single
biggest simplification available in this problem, and it is why the reconciler
is a pure function.

BASE IS WRITTEN LAST. Only after a change is durably on disk (or durably
committed on the server) do we record that the two sides agree about it. Write
BASE first and a crash makes us believe we synced something we did not, which
is how files silently stop syncing forever.

CONFLICTS DO NOT ABORT THE CYCLE. A CAS rejection on one path is normal - it
means someone beat us to it - and it must not stop the other 900 files from
syncing. We record it and it resolves on the next pass, when the feed has
brought us the version we lost to.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
No parallelism (real clients run many transfers at once, with a bounded pool
and per-file ordering). No bandwidth throttling. No partial-file streaming for
files bigger than memory. No encryption. No sharing, permissions or quotas -
shared folders are where this problem gets genuinely harder, because now other
people's ACL changes are also events in your feed.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .applier import LocalFS
from .clock import VersionVector
from .db import ClientDB
from .fsmodel import Entry, Kind, METADATA_DIR, Node
from .merge import (conflicted_copy_path, is_probably_text, three_way_merge)
from .planner import Action, Op, describe, reconcile
from .scanner import scan
from .server import CommitStatus
from .transport import Link, OfflineError, TransientError


@dataclass
class SyncReport:
    replica: str
    plan: List[Action] = field(default_factory=list)
    applied: int = 0
    conflicts: int = 0
    merged: int = 0
    conflicted_copies: List[str] = field(default_factory=list)
    deferred: List[str] = field(default_factory=list)
    offline: bool = False
    bytes_up: int = 0
    bytes_down: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def quiet(self) -> bool:
        return not self.plan and not self.offline


def _op_id(replica: str, action_kind: str, path: str,
           cid: Optional[str], vv: VersionVector) -> str:
    """A deterministic operation id.

    Derived from the intent rather than randomly generated, so that a retry
    after a crash - not just a retry inside one process - produces the same id
    and the server recognises it as a replay. A random uuid would be forgotten
    across a restart, which is precisely when you need it most.
    """
    h = hashlib.blake2b(digest_size=12)
    h.update("|".join([replica, action_kind, path, cid or "-",
                       vv.to_json()]).encode("utf-8"))
    return f"{replica}:{h.hexdigest()}"


class SyncClient:
    def __init__(self, root: str, replica: str, link: Link,
                 allow_text_merge: bool = True,
                 write_conflict_markers: bool = False):
        self.root = os.path.abspath(root)
        self.replica = replica
        self.link = link
        self.fs = LocalFS(self.root)
        self.db = ClientDB(os.path.join(self.fs.meta, "state.db"))
        self.cache = self.db.load_hash_cache()
        self.allow_text_merge = allow_text_merge
        # Writing '<<<<<<<' into a user's file is a developer's habit, not a
        # consumer product's. Off by default: an unmergeable text file becomes
        # a conflicted copy like anything else.
        self.write_conflict_markers = write_conflict_markers

    # -- the cycle --------------------------------------------------------

    def sync_once(self, trust_cache_from_ns: Optional[int] = None) -> SyncReport:
        rep = SyncReport(replica=self.replica)
        up0, down0 = self.link.stats.bytes_up, self.link.stats.bytes_down

        # 1. Bring our mirror of the server forward.
        try:
            self._pull_feed()
        except OfflineError:
            rep.offline = True
            rep.notes.append("offline: working from the last known server state")
        except TransientError as exc:
            rep.notes.append(f"feed unavailable ({exc}); continuing locally")

        # 2 & 3. Look at the world, decide what to do about it.
        local = scan(self.root, self.cache, now_ns=trust_cache_from_ns)
        base = self.db.load_base()
        remote = self.db.load_remote()
        rep.plan = reconcile(local, base, remote, self.replica,
                             allow_text_merge=self.allow_text_merge)

        # 4. Do it.
        for action in rep.plan:
            try:
                self._execute(action, rep)
                rep.applied += 1
            except OfflineError:
                rep.offline = True
                rep.deferred.append(action.path)
            except TransientError as exc:
                rep.deferred.append(action.path)
                rep.notes.append(f"{action.path}: {exc}")

        self.db.save_hash_cache(self.cache)
        if not rep.offline:
            self.link.report_cursor(self.db.cursor)
        rep.bytes_up = self.link.stats.bytes_up - up0
        rep.bytes_down = self.link.stats.bytes_down - down0
        return rep

    def sync_until_quiet(self, max_passes: int = 6) -> List[SyncReport]:
        """Run cycles until nothing is left to do.

        More than one pass is normal and expected. Resolving a conflict
        *creates* a new file (the conflicted copy) which must itself be
        uploaded; a CAS rejection needs a pass to fetch what we lost to and
        another to resolve it. Convergence, not one-shot correctness, is the
        promise this design makes.
        """
        reports = []
        for _ in range(max_passes):
            rep = self.sync_once()
            reports.append(rep)
            if rep.quiet:
                break
        return reports

    # -- step 1 -----------------------------------------------------------

    def _pull_feed(self) -> None:
        while True:
            entries, cursor, truncated = self.link.changes_since(self.db.cursor)
            for entry in entries:
                self.db.put_remote(entry, commit=False)
            self.db.commit()
            # The cursor advances only after the entries it covers are durably
            # stored. Advance it first and a crash here loses changes forever.
            self.db.cursor = cursor
            if not truncated:
                return

    # -- step 4 -----------------------------------------------------------

    def _execute(self, a: Action, rep: SyncReport) -> None:
        handler = {
            Op.PULL: self._do_pull,
            Op.PUSH: self._do_push,
            Op.MKDIR_LOCAL: self._do_mkdir_local,
            Op.MKDIR_REMOTE: self._do_mkdir_remote,
            Op.DELETE_LOCAL: self._do_delete_local,
            Op.DELETE_REMOTE: self._do_delete_remote,
            Op.MOVE_LOCAL: self._do_move_local,
            Op.MOVE_REMOTE: self._do_move_remote,
            Op.CONFLICT_SPLIT: self._do_conflict_split,
            Op.MERGE_TEXT: self._do_merge_text,
            Op.CONVERGE: self._do_converge,
        }[a.op]
        handler(a, rep)

    # -- inbound ----------------------------------------------------------

    def _do_pull(self, a: Action, rep: SyncReport) -> None:
        node = a.remote.node
        data = self.link.download_content(node.content_id)
        self.fs.atomic_write(a.path, data, node.mtime_ns, node.executable)
        self.cache.forget(a.path)
        self.db.put_base(a.remote)        # only now do we claim agreement

    def _do_mkdir_local(self, a: Action, rep: SyncReport) -> None:
        self.fs.mkdir(a.path)
        self.db.put_base(a.remote)

    def _do_delete_local(self, a: Action, rep: SyncReport) -> None:
        if a.local is not None and a.local.kind is Kind.DIR:
            self.fs.remove_dir_if_empty(a.path)
        else:
            self.fs.to_trash(a.path)      # recoverable, never unlinked
        self.cache.forget(a.path)
        self.db.put_base(a.remote)

    def _do_move_local(self, a: Action, rep: SyncReport) -> None:
        if self.fs.exists(a.from_path):
            self.fs.move(a.from_path, a.path)
        else:                              # lost the race; fall back to bytes
            self._do_pull(a, rep)
            return
        self.cache.forget(a.from_path)
        self.cache.forget(a.path)
        self.db.put_base(a.remote)
        # The source path is now empty here. Forget our baseline for it; the
        # next cycle sees local-absent / base-absent / remote-tombstone and
        # converges it quietly, rather than mistaking it for a fresh delete.
        self.db.drop_base(a.from_path)

    # -- outbound ---------------------------------------------------------

    def _next_vv(self, a: Action) -> VersionVector:
        """The vector our new version should carry.

        It must dominate everything we knew about: our baseline and whatever
        the server currently advertises. Then bump our own counter to record
        that this edit originated here.
        """
        known = a.base.vv if a.base else VersionVector.empty()
        if a.remote is not None:
            known = known.merge(a.remote.vv)
        return known.bump(self.replica)

    def _commit(self, node: Node, vv: VersionVector, a: Action,
                rep: SyncReport, kind: str) -> bool:
        op_id = _op_id(self.replica, kind, node.path, node.content_id, vv)
        self.db.journal_add(op_id, kind, {"path": node.path})
        result = self.link.commit(node, vv, op_id)
        if result.status == CommitStatus.CONFLICT:
            # Someone committed between our feed pull and our write. Take
            # their version into our mirror; the next cycle sees a normal
            # three-way conflict and resolves it with the usual machinery.
            rep.conflicts += 1
            self.db.put_remote(result.server_entry)
            self.db.journal_done(op_id)
            rep.notes.append(
                f"{node.path}: server rejected our write (someone else "
                f"committed first) - will reconcile next pass")
            return False
        if result.status == CommitStatus.MISSING_CONTENT:
            rep.notes.append(f"{node.path}: content not uploaded yet")
            self.db.journal_failed(op_id)
            return False
        self.db.put_remote(result.entry)
        self.db.put_base(result.entry)
        self.db.journal_done(op_id)
        return True

    def _do_push(self, a: Action, rep: SyncReport) -> None:
        data = self.fs.read(a.path)
        manifest = self.link.upload_content(data)     # content first...
        node = a.local.with_(content_id=manifest.content_id, size=len(data))
        self._commit(node, self._next_vv(a), a, rep, "push")  # ...metadata after

    def _do_mkdir_remote(self, a: Action, rep: SyncReport) -> None:
        node = Node(a.path, Kind.DIR)
        self._commit(node, self._next_vv(a), a, rep, "mkdir")

    def _do_delete_remote(self, a: Action, rep: SyncReport) -> None:
        node = Node(a.path, Kind.DELETED)      # a tombstone, not a removal
        self._commit(node, self._next_vv(a), a, rep, "delete")

    def _do_move_remote(self, a: Action, rep: SyncReport) -> None:
        # The content is already on the server - that is the whole point of a
        # move - so this is two metadata commits and zero bytes transferred.
        new_node = a.local.with_(path=a.path)
        if not self._commit(new_node, self._next_vv(a), a, rep, "move-to"):
            return
        old_vv = (a.base.vv if a.base else VersionVector.empty())
        if a.remote is not None:
            old_vv = old_vv.merge(a.remote.vv)
        tomb = Node(a.from_path, Kind.DELETED)
        op_id = _op_id(self.replica, "move-from", a.from_path, None,
                       old_vv.bump(self.replica))
        result = self.link.commit(tomb, old_vv.bump(self.replica), op_id)
        if result.status == CommitStatus.OK:
            self.db.put_remote(result.entry)
            self.db.put_base(result.entry)

    # -- conflict handling ------------------------------------------------

    def _do_converge(self, a: Action, rep: SyncReport) -> None:
        if a.remote is not None:
            self.db.put_base(a.remote)

    def _do_conflict_split(self, a: Action, rep: SyncReport) -> None:
        """Keep both versions. Remote keeps the path; ours is moved aside.

        Remote keeps the original name on purpose: every peer resolving this
        same conflict independently makes the same choice, so they all
        converge without another round of negotiation.
        """
        aside = conflicted_copy_path(a.path, self.replica)
        n = 0
        while self.fs.exists(aside):
            n += 1
            aside = conflicted_copy_path(a.path, self.replica, attempt=n)

        if a.local is not None and a.local.exists and self.fs.exists(a.path):
            self.fs.move(a.path, aside)
            self.cache.forget(a.path)
            rep.conflicted_copies.append(aside)

        if a.remote is not None and a.remote.node.exists:
            self._do_pull(a, rep)          # remote version takes the path
        else:
            self.db.put_base(a.remote) if a.remote else None

        # Upload the preserved copy as a brand-new file of its own.
        if self.fs.exists(aside):
            data = self.fs.read(aside)
            manifest = self.link.upload_content(data)
            node = Node(aside, Kind.FILE, manifest.content_id, len(data))
            fresh = Action(Op.PUSH, aside, local=node)
            self._commit(node, VersionVector.empty().bump(self.replica),
                         fresh, rep, "push")

    def _do_merge_text(self, a: Action, rep: SyncReport) -> None:
        """Try a real merge; fall back to keeping both if it does not work."""
        mine = self.fs.read(a.path)
        theirs = self.link.download_content(a.remote.node.content_id)

        ancestor: Optional[bytes] = None
        base_cid = a.base.node.content_id if a.base else None
        if base_cid is not None:
            try:
                # The ancestor is still retrievable because storage is
                # content-addressed: old versions are not overwritten, they
                # are simply no longer referenced by the current metadata.
                ancestor = self.link.download_content(base_cid)
            except (KeyError, TransientError):
                ancestor = None

        if ancestor is None or not all(map(is_probably_text,
                                           (ancestor, mine, theirs))):
            rep.notes.append(f"{a.path}: not mergeable, keeping both versions")
            self._do_conflict_split(a, rep)
            return

        result = three_way_merge(ancestor, mine, theirs,
                                 mine_label=self.replica,
                                 theirs_label=a.remote.modified_by or "remote")
        if not result.clean and not self.write_conflict_markers:
            rep.notes.append(
                f"{a.path}: {result.conflict_regions} overlapping edit(s) - "
                f"cannot merge automatically, keeping both versions")
            self._do_conflict_split(a, rep)
            return

        self.fs.atomic_write(a.path, result.merged)
        self.cache.forget(a.path)
        rep.merged += 1
        rep.notes.append(f"{a.path}: merged both edits automatically")
        manifest = self.link.upload_content(result.merged)
        node = Node(a.path, Kind.FILE, manifest.content_id, len(result.merged))
        # The merged version genuinely supersedes both inputs, so its vector
        # must dominate both - that is exactly what merge().bump() gives us.
        vv = a.base.vv.merge(a.remote.vv).bump(self.replica)
        self._commit(node, vv, a, rep, "push")

    # -- introspection ----------------------------------------------------

    def tree(self) -> Dict[str, str]:
        """Current on-disk state, for asserting that replicas converged."""
        out = {}
        for path, node in scan(self.root, None).items():
            out[path] = node.content_id if node.kind is Kind.FILE else "<dir>"
        return out
