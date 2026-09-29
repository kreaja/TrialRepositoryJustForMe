"""
The reconciler. Everything else in this package exists to serve this file.

THE ONE IDEA
------------
Given a path, you have three pieces of information:

    LOCAL   what is on this disk right now
    BASE    what this machine and the server last agreed on
    REMOTE  what the server says is there now

Two-way comparison (LOCAL vs REMOTE) is not enough, and this is the single
most important thing to understand about file sync. Suppose LOCAL has no
`notes.txt` and REMOTE has one. Two completely different things could have
happened:

    * someone created it on another machine  -> we should download it
    * we deleted it here                     -> we should delete it there

The two situations are indistinguishable from LOCAL and REMOTE alone. BASE
disambiguates them: if BASE has the file, we deleted it; if BASE does not, they
created it. BASE tells you *which side moved*, and "which side moved" is the
entire question.

This is the same three-way structure as a git merge (ours / theirs / merge
base), and for the same reason.

THE DECISION TABLE
------------------
With `local changed` = LOCAL differs from BASE, and `remote changed` = REMOTE's
version vector carries something BASE's does not:

  local    remote   situation                        action
  changed  changed
  -------  -------  -------------------------------  ----------------------
  no       no       in sync                          nothing
  yes      no       we edited / created / deleted    push to server
  no       yes      they edited / created / deleted  pull to disk
  yes      yes      both moved                       -> see below

When both moved, look at where they ended up:

  * identical content        both made the same change (or both deleted).
                             Not a conflict. Adopt the server's vector and
                             move on. Catching this matters more than it
                             sounds: without it, "both machines ran the same
                             formatter" produces a conflicted copy.
  * local deleted, remote edited   RESURRECT. We keep the remote file. A
                             deletion is cheap to redo and an edit is not, so
                             when in doubt, keep bytes.
  * local edited, remote deleted   KEEP LOCAL, and re-create it on the server.
                             Same reasoning, other direction.
  * both edited, text        try a three-way merge (we have the ancestor,
                             which is exactly what a merge needs).
  * otherwise                CONFLICT SPLIT: remote keeps the path, local
                             content is moved aside to a conflicted copy,
                             which is then uploaded as its own file.

MOVE DETECTION
--------------
After the per-path decisions are made we look at the plan as a whole. A delete
of content X at one path plus a create of the same content X at another is
almost certainly a rename, and turning that pair into a single move keeps a
4 GB folder rename from becoming a 4 GB upload. It is a heuristic - copy the
file then delete the original and we will call it a move, which produces the
right final state anyway.

Real clients do better where the OS lets them, by tracking file identity
(inode on Unix, file id on NTFS) so a rename is known rather than inferred.

WHY THE PLAN IS DATA
--------------------
`reconcile` is a pure function: three dictionaries in, a list of actions out.
No disk, no network, no clock. That is what makes this testable - every
scenario in tests/ is a few lines constructing dicts - and it is why the
demo can print the plan before executing it. Sync engines that interleave
decision-making with I/O are notoriously impossible to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .clock import VersionVector
from .fsmodel import Entry, Kind, Node, depth
from .scanner import Snapshot


class Op(str, Enum):
    NOOP = "noop"
    PULL = "pull"                    # download remote content to disk
    PUSH = "push"                    # upload local content to server
    MKDIR_LOCAL = "mkdir-local"
    MKDIR_REMOTE = "mkdir-remote"
    DELETE_LOCAL = "delete-local"
    DELETE_REMOTE = "delete-remote"
    MOVE_LOCAL = "move-local"        # rename on disk, no transfer
    MOVE_REMOTE = "move-remote"      # rename on server, no transfer
    CONFLICT_SPLIT = "conflict-split"
    MERGE_TEXT = "merge-text"
    CONVERGE = "converge"            # metadata only: adopt the server's vector


# Actions are executed in phases. Order is not cosmetic - it is what keeps
# the tree in a legal state at every intermediate step:
#   1 conflicts first, so nothing is overwritten before it is preserved
#   2 local deletes, deepest first (cannot remove a non-empty directory)
#   3 local mkdirs, shallowest first (parents before children)
#   4 local moves (their destination directory now exists)
#   5 downloads, shallowest first (parent directory must exist)
#   6 everything outbound
#   7 metadata-only convergence
PHASE = {
    Op.CONFLICT_SPLIT: 1, Op.MERGE_TEXT: 1,
    Op.DELETE_LOCAL: 2,
    Op.MKDIR_LOCAL: 3,
    Op.MOVE_LOCAL: 4,
    Op.PULL: 5,
    Op.MKDIR_REMOTE: 6, Op.MOVE_REMOTE: 6, Op.PUSH: 6, Op.DELETE_REMOTE: 6,
    Op.CONVERGE: 7, Op.NOOP: 8,
}

# Within a phase, do we want shallow paths first or deep paths first?
DEEPEST_FIRST = {Op.DELETE_LOCAL, Op.DELETE_REMOTE}


@dataclass
class Action:
    op: Op
    path: str
    reason: str = ""
    local: Optional[Node] = None
    base: Optional[Entry] = None
    remote: Optional[Entry] = None
    from_path: Optional[str] = None      # for moves
    conflict_path: Optional[str] = None  # for conflict splits

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        arrow = f" {self.from_path} ->" if self.from_path else ""
        extra = f" (aside: {self.conflict_path})" if self.conflict_path else ""
        return f"{self.op.value:14s}{arrow} {self.path}{extra}  -- {self.reason}"


def _same(a: Optional[Node], b: Optional[Node]) -> bool:
    a_live = a is not None and a.exists
    b_live = b is not None and b.exists
    if not a_live and not b_live:
        return True                     # absent both sides, or both tombstoned
    if a_live != b_live:
        return False
    return a.same_content_as(b)         # type: ignore[union-attr]


def reconcile(local: Snapshot,
              base: Dict[str, Entry],
              remote: Dict[str, Entry],
              replica: str,
              allow_text_merge: bool = True) -> List[Action]:
    """Pure three-way reconciliation. See the module docstring for the table."""
    actions: List[Action] = []
    paths = set(local) | set(base) | set(remote)

    for path in sorted(paths):
        L: Optional[Node] = local.get(path)
        B: Optional[Entry] = base.get(path)
        R: Optional[Entry] = remote.get(path)

        base_node = B.node if B else None
        base_vv = B.vv if B else VersionVector.empty()

        local_changed = not _same(L, base_node)
        # "Remote changed" is a statement about causality, not content: does
        # the server's vector contain an edit our baseline does not know about?
        remote_changed = R is not None and not base_vv.descends_from(R.vv)

        if not local_changed and not remote_changed:
            continue

        # ---- only one side moved: no ambiguity, just carry it across ------

        if local_changed and not remote_changed:
            actions.append(_one_sided_local(path, L, B, R))
            continue

        if remote_changed and not local_changed:
            actions.append(_one_sided_remote(path, L, B, R))
            continue

        # ---- both sides moved --------------------------------------------

        remote_node = R.node if R else None

        # (a) They agree. Not every divergence is a conflict.
        if _same(L, remote_node):
            actions.append(Action(
                Op.CONVERGE, path, local=L, base=B, remote=R,
                reason="both sides reached the same content independently"))
            continue

        local_live = L is not None and L.exists
        remote_live = remote_node is not None and remote_node.exists

        # (b) We deleted, they edited. Keep the bytes.
        if not local_live and remote_live:
            actions.append(Action(
                Op.PULL, path, local=L, base=B, remote=R,
                reason="deleted here but edited elsewhere - restoring "
                       "(a delete is cheap to repeat, an edit is not)"))
            continue

        # (c) They deleted, we edited. Keep the bytes.
        if local_live and not remote_live:
            actions.append(Action(
                Op.PUSH, path, local=L, base=B, remote=R,
                reason="deleted elsewhere but edited here - re-creating "
                       "it on the server"))
            continue

        # (d) Both edited the same file.
        if (allow_text_merge and local_live and remote_live
                and L.kind is Kind.FILE and remote_node.kind is Kind.FILE
                and base_node is not None and base_node.kind is Kind.FILE):
            actions.append(Action(
                Op.MERGE_TEXT, path, local=L, base=B, remote=R,
                reason="concurrent edits - attempting a three-way merge "
                       "against the common ancestor"))
            continue

        # (e) Anything else, including file-vs-directory. Keep both, remote
        #     keeps the name so that every peer converges on the same choice.
        actions.append(Action(
            Op.CONFLICT_SPLIT, path, local=L, base=B, remote=R,
            reason="concurrent changes that cannot be merged - keeping both"))

    actions = detect_moves(actions)
    return order(actions)


def _one_sided_local(path, L, B, R) -> Action:
    """We changed it, nobody else did."""
    if L is None or not L.exists:
        return Action(Op.DELETE_REMOTE, path, local=L, base=B, remote=R,
                      reason="deleted locally; propagating the deletion")
    if L.kind is Kind.DIR:
        if B is None:
            return Action(Op.MKDIR_REMOTE, path, local=L, base=B, remote=R,
                          reason="new local directory")
        return Action(Op.NOOP, path, reason="directory already known")
    return Action(Op.PUSH, path, local=L, base=B, remote=R,
                  reason="created locally" if B is None else "edited locally")


def _one_sided_remote(path, L, B, R) -> Action:
    """They changed it, we did not."""
    node = R.node
    if not node.exists:
        if L is None:
            return Action(Op.CONVERGE, path, local=L, base=B, remote=R,
                          reason="tombstone for a file we do not have")
        return Action(Op.DELETE_LOCAL, path, local=L, base=B, remote=R,
                      reason="deleted elsewhere; removing our copy")
    if node.kind is Kind.DIR:
        if L is not None and L.kind is Kind.DIR:
            return Action(Op.CONVERGE, path, local=L, base=B, remote=R,
                          reason="directory already present")
        return Action(Op.MKDIR_LOCAL, path, local=L, base=B, remote=R,
                      reason="new remote directory")
    return Action(Op.PULL, path, local=L, base=B, remote=R,
                  reason="created elsewhere" if B is None else "edited elsewhere")


def detect_moves(actions: List[Action]) -> List[Action]:
    """Collapse delete+create of identical content into a single rename."""
    out = list(actions)

    def content_of(a: Action, side: str) -> Optional[str]:
        if side == "base":
            return a.base.node.content_id if a.base and a.base.node else None
        if side == "local":
            return a.local.content_id if a.local else None
        return a.remote.node.content_id if a.remote else None

    # Remote-side rename: we deleted A and created B with the same bytes.
    deletes = {content_of(a, "base"): a for a in out
               if a.op is Op.DELETE_REMOTE and content_of(a, "base")}
    for a in list(out):
        # The destination must be genuinely new on BOTH sides. `base is None`
        # says we have never synced this path; `remote is None` says the
        # server has never heard of it either.
        #
        # That second condition is not fussiness, it is load-bearing. A move
        # carries the SOURCE path's version vector to the DESTINATION path. If
        # the destination already has its own history on the server - even
        # just a tombstone from another replica - that vector will not
        # dominate the destination's, the server's compare-and-swap will
        # reject the write, and the next cycle will regenerate exactly the
        # same doomed plan. The client then spins forever, making no progress
        # and never converging. Found by the convergence fuzzer, not by
        # reasoning; this is the kind of bug that only shows up under
        # interleaving.
        if a.op is not Op.PUSH or a.base is not None or a.remote is not None:
            continue
        cid = content_of(a, "local")
        victim = deletes.pop(cid, None)
        if victim is not None and victim in out:
            out.remove(victim)
            out[out.index(a)] = Action(
                Op.MOVE_REMOTE, a.path, from_path=victim.path,
                local=a.local, base=victim.base, remote=victim.remote,
                reason=f"renamed locally from {victim.path} - "
                       f"metadata only, no bytes transferred")

    # Local-side rename: the server tombstoned A and published B, same bytes.
    ldeletes = {content_of(a, "local"): a for a in out
                if a.op is Op.DELETE_LOCAL and content_of(a, "local")}
    for a in list(out):
        if a.op is not Op.PULL or a.base is not None:
            continue        # a local move needs no commit, so no CAS hazard
        cid = content_of(a, "remote")
        victim = ldeletes.pop(cid, None)
        if victim is not None and victim in out:
            out.remove(victim)
            out[out.index(a)] = Action(
                Op.MOVE_LOCAL, a.path, from_path=victim.path,
                local=victim.local, base=a.base, remote=a.remote,
                reason=f"renamed elsewhere from {victim.path} - "
                       f"renaming on disk instead of downloading")
    return out


def order(actions: List[Action]) -> List[Action]:
    """Sort into an order that is legal at every intermediate step."""
    def key(a: Action):
        d = depth(a.path)
        return (PHASE[a.op], -d if a.op in DEEPEST_FIRST else d, a.path)
    return sorted((a for a in actions if a.op is not Op.NOOP), key=key)


def describe(actions: List[Action]) -> str:  # pragma: no cover - cosmetic
    if not actions:
        return "    (nothing to do - all three views agree)"
    return "\n".join("    " + str(a) for a in actions)
