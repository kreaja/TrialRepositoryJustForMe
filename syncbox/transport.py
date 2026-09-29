"""
A deliberately unreliable link to the server.

Every interesting property of a sync client is a consequence of the network
being bad. If the network were perfect you could write this whole system in an
afternoon. So the link here can be offline, can drop requests, and - the case
that actually teaches you something - can fail *after* the server has already
committed, leaving the client unable to tell success from failure.

THREE THINGS THIS MODULE DEMONSTRATES
-------------------------------------

RESUMABLE UPLOADS WITHOUT CLIENT-SIDE PROGRESS STATE. On retry we simply ask
the server which chunks it is still missing and send those. The server's own
contents are the resume point, so there is no progress file to keep in sync,
nothing to corrupt, and a resume works even from a different machine. This is
why content-addressed chunking and resumability are really the same feature.

RETRY WITH EXPONENTIAL BACKOFF AND JITTER. The jitter is not decoration. If a
provider has an outage and ten million clients retry on identical schedules,
recovery brings a synchronised thundering herd that knocks the service over
again. Randomising the delay spreads the load.

IDEMPOTENT COMMITS. `fail_after_commit` simulates the ambiguous failure. The
client's only correct response is to retry with the same operation id, which
the server recognises and answers from its memo table instead of applying
twice.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .clock import VersionVector
from .fsmodel import Entry, Node
from .hashing import Manifest, build_manifest, reassemble
from .server import CloudServer, CommitResult, CommitStatus


class OfflineError(RuntimeError):
    """No connectivity at all."""


class TransientError(RuntimeError):
    """The request failed, and we cannot tell whether it took effect."""


@dataclass
class LinkStats:
    requests: int = 0
    failures: int = 0
    bytes_up: int = 0
    bytes_down: int = 0
    retries: int = 0


class Link:
    def __init__(self, server: CloudServer, replica: str,
                 failure_rate: float = 0.0, seed: int = 0,
                 max_retries: int = 6):
        self.server = server
        self.replica = replica
        self.online = True
        self.failure_rate = failure_rate
        self.fail_after_commit = False
        self.max_retries = max_retries
        self.rng = random.Random(seed)
        self.stats = LinkStats()

    # -- link conditions --------------------------------------------------

    def go_offline(self) -> None:
        self.online = False

    def go_online(self) -> None:
        self.online = True

    def _hop(self) -> None:
        """One network round trip, which may simply not work."""
        self.stats.requests += 1
        if not self.online:
            raise OfflineError("no connectivity")
        if self.failure_rate and self.rng.random() < self.failure_rate:
            self.stats.failures += 1
            raise TransientError("connection reset")

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential, capped, with full jitter. Returned rather than slept
        so the demo runs instantly; a real client would sleep this long."""
        ceiling = min(60.0, 0.5 * (2 ** attempt))
        return self.rng.uniform(0, ceiling)

    # -- content plane ----------------------------------------------------

    def upload_content(self, data: bytes) -> Manifest:
        """Push a blob, sending only chunks the server lacks. Resumable."""
        manifest, blobs = build_manifest(data)
        for attempt in range(self.max_retries):
            try:
                self._hop()
                # The resume point: whatever survived previous attempts is
                # already on the server and is not in this list.
                missing = self.server.missing_chunks(list(manifest.chunks))
                for cid in missing:
                    self._hop()
                    self.server.put_chunk(cid, blobs[cid])
                    self.stats.bytes_up += len(blobs[cid])
                self._hop()
                self.server.put_manifest(manifest)
                return manifest
            except OfflineError:
                raise
            except TransientError:
                self.stats.retries += 1
                _delay = self._backoff_delay(attempt)
                continue
        raise TransientError("upload failed after retries")

    def download_content(self, cid: str) -> bytes:
        for attempt in range(self.max_retries):
            try:
                self._hop()
                manifest = self.server.get_manifest(cid)
                fetched: Dict[str, bytes] = {}

                def source(chunk_id: str) -> bytes:
                    if chunk_id not in fetched:
                        self._hop()
                        blob = self.server.get_chunk(chunk_id)
                        self.stats.bytes_down += len(blob)
                        fetched[chunk_id] = blob
                    return fetched[chunk_id]

                # reassemble() re-verifies the hash; a partial or corrupted
                # transfer raises rather than silently writing garbage.
                return reassemble(manifest, source)
            except OfflineError:
                raise
            except TransientError:
                self.stats.retries += 1
                self._backoff_delay(attempt)
                continue
        raise TransientError("download failed after retries")

    # -- metadata plane ---------------------------------------------------

    def commit(self, node: Node, vv: VersionVector, op_id: str) -> CommitResult:
        """Commit metadata, retrying with a stable op_id.

        The op_id must be generated once, by the caller, and reused across
        every retry of the same logical intent. Generating a fresh one per
        attempt would defeat the entire mechanism.
        """
        for attempt in range(self.max_retries):
            try:
                self._hop()
                result = self.server.commit(node, vv, self.replica, op_id)
                if self.fail_after_commit:
                    # The worst case: it worked, we will never know.
                    self.fail_after_commit = False
                    self.stats.failures += 1
                    raise TransientError("connection lost after commit")
                return result
            except OfflineError:
                raise
            except TransientError:
                self.stats.retries += 1
                self._backoff_delay(attempt)
                continue
        raise TransientError("commit failed after retries")

    def changes_since(self, cursor: int) -> Tuple[List[Entry], int, bool]:
        for attempt in range(self.max_retries):
            try:
                self._hop()
                return self.server.changes_since(cursor)
            except OfflineError:
                raise
            except TransientError:
                self.stats.retries += 1
                self._backoff_delay(attempt)
                continue
        raise TransientError("feed failed after retries")

    def report_cursor(self, cursor: int) -> None:
        try:
            self._hop()
            self.server.report_cursor(self.replica, cursor)
        except (OfflineError, TransientError):
            pass    # purely advisory, used only for tombstone collection
