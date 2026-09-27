"""
Turning a directory tree into a snapshot - and doing it cheaply.

THE COST PROBLEM
----------------
The honest way to detect local changes is to hash every file every time. For a
100 GB sync folder that is minutes of disk I/O per cycle, so nobody does it.
Instead every real client keeps a cache keyed on cheap metadata:

    (inode, size, mtime_ns, ctime_ns) -> content_id

If those four match what we saw last time, we assume the content is unchanged
and reuse the cached hash. This turns a full scan into a stat() walk.

WHY THAT CACHE IS A LIE (AND WHY WE KEEP IT ANYWAY)
--------------------------------------------------
It is possible to modify a file and end up with the same size and mtime:
  * mtime has coarse resolution on some filesystems, so an edit inside the
    same tick is invisible;
  * a program can set mtime back explicitly (rsync -t, tar extraction, some
    build tools);
  * network filesystems lie about all of it.

So the cache is a heuristic, and clients mitigate it rather than solve it:
  * include ctime (inode change time), which a user cannot set directly;
  * treat a file as suspicious if its mtime is within one timestamp-resolution
    tick of the last scan, and re-hash it;
  * run a full verifying re-hash periodically in the background.
We do the first two here.

WATCHERS VS SCANNING
--------------------
In production you do not poll. You subscribe: inotify on Linux, FSEvents on
macOS, ReadDirectoryChangesW on Windows. But you always keep the scanner too,
because watchers:
  * drop events under load (inotify queues overflow and tell you only that
    they overflowed),
  * miss everything that happened while your process was not running,
  * do not exist on every filesystem (network mounts especially).
So the watcher is an optimisation that tells you *where* to look, and the
periodic scan is the correctness backstop. `WatchShim` below stands in for the
platform-specific part.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from .fsmodel import METADATA_DIR, Kind, Node, is_safe, normalise
from .hashing import hash_file

Snapshot = Dict[str, Node]

# If a file's mtime is this close to "now", we cannot trust that a later write
# in the same tick would be visible, so we re-hash rather than trust the cache.
MTIME_SUSPICION_WINDOW_NS = 2_000_000_000  # 2 seconds


@dataclass
class HashCache:
    """(inode, size, mtime_ns, ctime_ns) -> content_id."""

    entries: Dict[str, Tuple[int, int, int, int, str]] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.entries is None:
            self.entries = {}
        self.hits = 0
        self.misses = 0

    def lookup(self, path: str, st: os.stat_result, now_ns: int) -> Optional[str]:
        rec = self.entries.get(path)
        if rec is None:
            self.misses += 1
            return None
        ino, size, mtime, ctime, cid = rec
        if (ino, size, mtime, ctime) != (st.st_ino, st.st_size,
                                         st.st_mtime_ns, st.st_ctime_ns):
            self.misses += 1
            return None
        if now_ns - st.st_mtime_ns < MTIME_SUSPICION_WINDOW_NS:
            # Too fresh to trust: an edit in the same timestamp tick would be
            # invisible to us. Pay for the hash.
            self.misses += 1
            return None
        self.hits += 1
        return cid

    def store(self, path: str, st: os.stat_result, cid: str) -> None:
        self.entries[path] = (st.st_ino, st.st_size, st.st_mtime_ns,
                              st.st_ctime_ns, cid)

    def forget(self, path: str) -> None:
        self.entries.pop(path, None)


def scan(root: str, cache: Optional[HashCache] = None,
         now_ns: Optional[int] = None) -> Snapshot:
    """Walk `root` and return {relative_path: Node}.

    Symlinks are recorded as nothing at all here - we skip them. That is a
    real product decision, not laziness: following them can loop forever and
    can export files from outside the sync root; storing them faithfully means
    a Windows peer cannot represent them. Dropbox skips them, Drive skips
    them. Whatever you choose, choose it explicitly.
    """
    cache = cache if cache is not None else HashCache()
    now_ns = now_ns if now_ns is not None else time.time_ns()
    out: Snapshot = {}

    for dirpath, dirnames, filenames in os.walk(root):
        # Never descend into our own metadata.
        dirnames[:] = [d for d in dirnames if d != METADATA_DIR]
        rel_dir = normalise(os.path.relpath(dirpath, root))

        for d in list(dirnames):
            rel = normalise(os.path.join(rel_dir, d)) if rel_dir else normalise(d)
            if not is_safe(rel):
                dirnames.remove(d)
                continue
            out[rel] = Node(path=rel, kind=Kind.DIR)

        for f in filenames:
            rel = normalise(os.path.join(rel_dir, f)) if rel_dir else normalise(f)
            if not is_safe(rel):
                continue
            full = os.path.join(dirpath, f)
            try:
                st = os.lstat(full)
            except FileNotFoundError:
                continue                      # raced with a delete; next cycle
            if not os.path.isfile(full) or os.path.islink(full):
                continue                      # skip symlinks, fifos, sockets
            cid = cache.lookup(rel, st, now_ns)
            if cid is None:
                try:
                    cid = hash_file(full)
                except (FileNotFoundError, PermissionError):
                    continue
                cache.store(rel, st, cid)
            out[rel] = Node(path=rel, kind=Kind.FILE, content_id=cid,
                            size=st.st_size, mtime_ns=st.st_mtime_ns,
                            executable=bool(st.st_mode & 0o100))
    return out


class WatchShim:
    """Stand-in for inotify / FSEvents / ReadDirectoryChangesW.

    Real watchers hand you a stream of paths that *might* have changed. The
    two things every client does with that stream are modelled here:

      DEBOUNCE - an application saving a file typically produces a burst of
      events (create temp, write, write, rename over the original). Uploading
      on the first one wastes bandwidth and can capture a half-written file,
      so you wait for the burst to go quiet.

      COALESCE - twelve events for one path become one unit of work.

    `overflowed` models the case that matters most for correctness: the kernel
    queue filled up and events were lost. The only safe response is to fall
    back to a full scan, which is why you can never delete the scanner.
    """

    def __init__(self, debounce_seconds: float = 0.5, capacity: int = 1024):
        self.debounce = debounce_seconds
        self.capacity = capacity
        self._pending: Set[str] = set()
        self.overflowed = False
        self._last_event_at = 0.0

    def notify(self, path: str) -> None:
        if len(self._pending) >= self.capacity:
            self.overflowed = True          # lost events; must rescan fully
            self._pending.clear()
            return
        self._pending.add(normalise(path))
        self._last_event_at = time.monotonic()

    def settled(self) -> bool:
        return (bool(self._pending)
                and time.monotonic() - self._last_event_at >= self.debounce)

    def drain(self) -> Tuple[Set[str], bool]:
        paths, overflow = set(self._pending), self.overflowed
        self._pending.clear()
        self.overflowed = False
        return paths, overflow
