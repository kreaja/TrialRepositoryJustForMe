"""
The cloud side. In-process, but with the same contract a real one has.

WHAT A SYNC SERVER ACTUALLY OWES ITS CLIENTS
--------------------------------------------
Only four things, and they are all about ordering:

1. A TOTAL ORDER on changes, so clients can say "give me everything after
   position N" and get a complete, resumable, incrementally-cheap answer.
   This is Dropbox's /delta, Drive's changes.list, OneDrive's delta query.
   The server is the single serialisation point - it is what makes the whole
   distributed problem tractable.

2. COMPARE-AND-SWAP on every write. A client says "store this version, and I
   believe the current one is X". If the server's version is not X, the write
   is REJECTED, not merged and not clobbered. Someone else got there first,
   and the client must reconcile before trying again. Without CAS you get
   lost updates: two clients read version 5, both write version 6, one edit
   vanishes with no error anywhere.

3. IDEMPOTENCE. Networks fail in the worst possible way: after the server
   committed but before the client heard about it. The client cannot tell
   that from "never arrived", so it must retry, and the retry must not create
   a duplicate. Every mutation carries a client-generated operation id and the
   server remembers the outcome.

4. INTEGRITY. Verify the hash of everything stored and everything served.

Note what the server does NOT do: it does not resolve conflicts. It detects
them and refuses the write. Resolution happens on the client, because only the
client can see both versions' content, and in an end-to-end-encrypted design
the server could not read them anyway.

CONSISTENCY MODEL - the honest version
--------------------------------------
Per file: linearizable. Every version of a given path is totally ordered by
the CAS chain, and no update is ever silently lost.

Across files: NOT atomic. There is no transaction spanning two paths. If you
save a .tex and its .bib together, a peer may briefly see the new .tex with
the old .bib. Every consumer product works this way, because cross-file
transactions across a wide-area network with offline clients would mean
blocking on unavailable peers. What you get instead is EVENTUAL CONSISTENCY:
if everyone stops editing and stays connected long enough, all replicas
converge to the same state. That is a much weaker promise than users imagine
they are getting, and it is the correct one to design against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .clock import VersionVector
from .fsmodel import Entry, Kind, Node
from .hashing import Manifest, content_id


class CommitStatus(str):
    OK = "ok"
    CONFLICT = "conflict"          # your base version is stale; reconcile
    MISSING_CONTENT = "missing"    # upload the manifest/chunks first


@dataclass
class CommitResult:
    status: str
    entry: Optional[Entry] = None      # the server's entry after the call
    server_entry: Optional[Entry] = None   # on CONFLICT: what is actually there
    seq: int = 0


@dataclass
class ServerStats:
    chunk_bytes_uploaded: int = 0
    chunk_bytes_downloaded: int = 0
    chunks_uploaded: int = 0
    chunks_skipped_dedup: int = 0
    commits: int = 0
    conflicts: int = 0
    idempotent_replays: int = 0


class CloudServer:
    def __init__(self) -> None:
        self.chunks: Dict[str, bytes] = {}
        self.manifests: Dict[str, Manifest] = {}
        self.entries: Dict[str, Entry] = {}
        self.entry_seq: Dict[str, int] = {}
        self._seq = 0
        self._op_results: Dict[str, CommitResult] = {}
        self.replica_cursors: Dict[str, int] = {}
        self.stats = ServerStats()

    # -- content plane ---------------------------------------------------

    def missing_chunks(self, cids: List[str]) -> List[str]:
        """The have/need negotiation. The client asks before it uploads.

        This one call is where deduplication actually happens: chunks the
        server already holds - from this user, this file's previous version,
        or (in a global dedup pool) anyone at all - are never sent again.
        """
        missing = [c for c in cids if c not in self.chunks]
        self.stats.chunks_skipped_dedup += len(cids) - len(missing)
        return missing

    def put_chunk(self, cid: str, data: bytes) -> None:
        actual = content_id(data)
        if actual != cid:
            raise ValueError(f"chunk id mismatch: claimed {cid}, got {actual}")
        if cid not in self.chunks:
            self.chunks[cid] = data
            self.stats.chunks_uploaded += 1
            self.stats.chunk_bytes_uploaded += len(data)

    def get_chunk(self, cid: str) -> bytes:
        data = self.chunks[cid]
        self.stats.chunk_bytes_downloaded += len(data)
        return data

    def put_manifest(self, manifest: Manifest) -> None:
        missing = [c for c in manifest.chunks if c not in self.chunks]
        if missing:
            raise ValueError(f"manifest references {len(missing)} absent chunks")
        self.manifests[manifest.content_id] = manifest

    def get_manifest(self, cid: str) -> Manifest:
        return self.manifests[cid]

    def has_content(self, cid: Optional[str]) -> bool:
        return cid is None or cid in self.manifests

    # -- metadata plane --------------------------------------------------

    def commit(self, node: Node, vv: VersionVector, replica: str,
               op_id: str) -> CommitResult:
        """Compare-and-swap a single path.

        The rule, in one line: accept iff the proposed version vector
        descends from the one currently stored.

        * descends and differs -> the client knew about our version and moved
          past it. Accept.
        * equal                -> a duplicate of what we have. Accept as a
          no-op so retries are harmless.
        * concurrent           -> the client did not know about a change we
          already have. REJECT and hand back our entry so it can reconcile.
        * older                -> stale write, e.g. from a client that has
          been asleep. REJECT for the same reason.
        """
        # Idempotence first: a retry of an operation we already processed must
        # return the original outcome, not do the work twice.
        if op_id in self._op_results:
            self.stats.idempotent_replays += 1
            return self._op_results[op_id]

        if node.kind is Kind.FILE and not self.has_content(node.content_id):
            return CommitResult(CommitStatus.MISSING_CONTENT)

        current = self.entries.get(node.path)
        if current is not None and not vv.descends_from(current.vv):
            self.stats.conflicts += 1
            result = CommitResult(CommitStatus.CONFLICT, server_entry=current)
            # Deliberately NOT memoised: a conflict is a fact about the current
            # state, and the client will retry after reconciling with a
            # different (merged) vector under a new op id.
            return result

        self._seq += 1
        entry = Entry(node, vv, replica)
        self.entries[node.path] = entry
        self.entry_seq[node.path] = self._seq
        self.stats.commits += 1
        result = CommitResult(CommitStatus.OK, entry=entry, seq=self._seq)
        self._op_results[op_id] = result
        return result

    def changes_since(self, cursor: int, limit: int = 1000
                      ) -> Tuple[List[Entry], int, bool]:
        """The change feed: everything that moved after `cursor`.

        Two properties worth noticing:

        COALESCING. We index by path, not by history, so a file edited fifty
        times while a client was offline is delivered once, in its final
        state. The feed is a set of current states, not an event log. This is
        what keeps "back from a two-week holiday" cheap.

        AT-LEAST-ONCE, NOT EXACTLY-ONCE. A client can crash after applying a
        change and before saving the new cursor, and will then see it again.
        That is fine, and it is fine *by construction*: applying a change you
        already have is a no-op, because the comparison is on content, not on
        events. Designing every operation to be re-runnable is what lets you
        get away with a cheap delivery guarantee.
        """
        moved = sorted(
            ((seq, path) for path, seq in self.entry_seq.items() if seq > cursor)
        )
        page, truncated = moved[:limit], len(moved) > limit
        entries = [self.entries[p] for _s, p in page]
        new_cursor = page[-1][0] if page else cursor
        return entries, new_cursor, truncated

    def report_cursor(self, replica: str, cursor: int) -> None:
        self.replica_cursors[replica] = cursor

    # -- housekeeping ----------------------------------------------------

    def gc_tombstones(self, known_replicas: Optional[Set[str]] = None) -> int:
        """Drop tombstones every replica has certainly seen.

        You cannot just expire them on a timer. A laptop that has been in a
        drawer for six months still has the file, and still has a BASE entry
        saying it was synced. If the tombstone is gone when that laptop wakes
        up, it looks like a brand-new local file and gets re-uploaded - the
        file rises from the dead. So: collect only below the minimum cursor
        across all replicas we know about. A replica that never comes back
        pins the tombstones forever, which is why real systems also unenroll
        devices that have been silent past some threshold, and accept the
        (now explicit) risk for those.
        """
        replicas = known_replicas or set(self.replica_cursors)
        if not replicas or not replicas <= set(self.replica_cursors):
            return 0
        safe_below = min(self.replica_cursors[r] for r in replicas)
        removed = 0
        for path, entry in list(self.entries.items()):
            if (entry.kind is Kind.DELETED
                    and self.entry_seq[path] <= safe_below):
                del self.entries[path]
                del self.entry_seq[path]
                removed += 1
        return removed

    def snapshot(self) -> Dict[str, Entry]:
        return dict(self.entries)

    def live_paths(self) -> List[str]:
        return sorted(p for p, e in self.entries.items()
                      if e.kind is not Kind.DELETED)
