"""
Touching the user's actual files. This is where you get to destroy data.

ATOMIC REPLACEMENT
------------------
Never write into the destination file. If the process dies, or the disk fills,
or the machine loses power halfway through, the user is left with a truncated
file and their original is gone. Instead:

    write to a temporary file in the same filesystem
    fsync the file           (the bytes are actually on the medium)
    rename over the target   (atomic - readers see old or new, never half)
    fsync the directory      (the rename itself is durable)

`os.replace` is atomic on POSIX and on Windows. "In the same filesystem"
matters because a cross-device rename is really a copy plus a delete, which is
not atomic at all - hence the temp directory living inside the sync root.

The two fsyncs are the part everyone skips. Without the first, the rename can
be durable while the contents are not, and you get a correctly-named file full
of zeros after a power cut. Without the second, the rename itself can be lost.

DELETING IS NOT DELETING
------------------------
A sync engine deletes files because a *remote* machine said so. If anything
upstream was wrong - a bug, a bad merge, a compromised account, a user who
did not mean it - the deletion is unrecoverable and it is your fault. So local
deletions go to a trash folder inside the sync metadata directory, and are
aged out later. This is why every one of these products has a "restore deleted
files" feature with a retention window: not as a courtesy, but because the
engine cannot ever be certain enough to actually unlink something.

RENAME CYCLES
-------------
Users do swap two directories. Applying `a -> b` then `b -> a` naively either
clobbers something or fails. Any set of moves decomposes into chains (safe if
you apply them in dependency order) and cycles (which need one temporary name
to break). `apply_moves` below handles both.
"""

from __future__ import annotations

import errno
import os
import shutil
import time
import uuid
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .fsmodel import METADATA_DIR, ancestors


class LocalFS:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.meta = os.path.join(self.root, METADATA_DIR)
        self.tmp = os.path.join(self.meta, "tmp")
        self.trash = os.path.join(self.meta, "trash")
        for d in (self.meta, self.tmp, self.trash):
            os.makedirs(d, exist_ok=True)

    # -- paths ------------------------------------------------------------

    def abs(self, path: str) -> str:
        full = os.path.abspath(os.path.join(self.root, path))
        # Belt and braces: even after path validation, never write outside.
        if full != self.root and not full.startswith(self.root + os.sep):
            raise ValueError(f"path escapes sync root: {path}")
        return full

    def exists(self, path: str) -> bool:
        return os.path.lexists(self.abs(path))

    def read(self, path: str) -> bytes:
        with open(self.abs(path), "rb") as fh:
            return fh.read()

    # -- durable writes ---------------------------------------------------

    def _fsync_dir(self, directory: str) -> None:
        try:
            fd = os.open(directory, os.O_RDONLY)
        except (OSError, AttributeError):    # not available on some platforms
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def ensure_dirs(self, path: str) -> None:
        for anc in ancestors(path):
            full = self.abs(anc)
            if not os.path.isdir(full):
                if os.path.lexists(full):
                    # A file is sitting where a directory needs to be. Get it
                    # out of the way rather than failing the whole cycle.
                    self.to_trash(anc)
                os.makedirs(full, exist_ok=True)

    def mkdir(self, path: str) -> None:
        self.ensure_dirs(path)
        os.makedirs(self.abs(path), exist_ok=True)

    def atomic_write(self, path: str, data: bytes, mtime_ns: int = 0,
                     executable: bool = False) -> None:
        self.ensure_dirs(path)
        target = self.abs(path)
        tmp = os.path.join(self.tmp, uuid.uuid4().hex)
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())          # bytes are on the medium
        if executable:
            os.chmod(tmp, 0o755)
        if mtime_ns:
            os.utime(tmp, ns=(mtime_ns, mtime_ns))
        os.replace(tmp, target)            # atomic swap
        self._fsync_dir(os.path.dirname(target) or self.root)

    # -- removal ----------------------------------------------------------

    def to_trash(self, path: str) -> Optional[str]:
        """Move a path into the trash instead of unlinking it."""
        src = self.abs(path)
        if not os.path.lexists(src):
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = os.path.join(self.trash, f"{stamp}-{uuid.uuid4().hex[:8]}-"
                                        f"{os.path.basename(path)}")
        shutil.move(src, dest)
        return dest

    def remove_dir_if_empty(self, path: str) -> bool:
        full = self.abs(path)
        try:
            os.rmdir(full)
            return True
        except OSError as exc:
            # ENOTEMPTY is the normal case: a peer deleted the directory but
            # this machine has unsynced files in it. Keeping them is right.
            if exc.errno in (errno.ENOTEMPTY, errno.EEXIST, errno.ENOENT):
                return False
            raise

    # -- moves ------------------------------------------------------------

    def move(self, src: str, dst: str) -> None:
        self.ensure_dirs(dst)
        os.replace(self.abs(src), self.abs(dst))

    def apply_moves(self, moves: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Apply (src, dst) renames, breaking cycles with staging names.

        A set of renames is a permutation, and every permutation decomposes
        into chains and cycles.

        CHAINS resolve themselves if you always pick a move whose destination
        nobody still has to vacate - that destination is free right now, so
        the move cannot clobber anything.

        CYCLES have no such move: every destination is occupied by something
        that is itself waiting to move. Breaking one needs exactly one
        temporary name. Move any member of the cycle out to staging, and the
        cycle becomes a chain, which the rule above then unwinds. Put the
        staged file into its destination at the end.

        Returns the moves in the order performed.
        """
        pending = {src: dst for src, dst in moves if src != dst}
        done: List[Tuple[str, str]] = []
        staged: List[Tuple[str, str, str]] = []   # (origin, staging, final dst)

        def drain_chains() -> None:
            progress = True
            while progress:
                progress = False
                for src in list(pending):
                    dst = pending[src]
                    if dst not in pending:      # destination is already free
                        self.move(src, dst)
                        del pending[src]
                        done.append((src, dst))
                        progress = True

        drain_chains()
        while pending:                          # whatever is left is a cycle
            origin = next(iter(pending))
            final_dst = pending.pop(origin)
            staging = os.path.join(METADATA_DIR, "tmp", uuid.uuid4().hex)
            self.move(origin, staging)          # vacate one slot
            staged.append((origin, staging, final_dst))
            drain_chains()                      # the rest is now a chain
        for origin, staging, final_dst in staged:
            self.move(staging, final_dst)
            done.append((origin, final_dst))
        return done

    # -- housekeeping -----------------------------------------------------

    def purge_trash(self, older_than_seconds: float) -> int:
        cutoff = time.time() - older_than_seconds
        removed = 0
        for name in os.listdir(self.trash):
            full = os.path.join(self.trash, name)
            if os.path.getmtime(full) < cutoff:
                shutil.rmtree(full, ignore_errors=True) if os.path.isdir(full) \
                    else os.remove(full)
                removed += 1
        return removed

    def trash_contents(self) -> List[str]:
        return sorted(os.listdir(self.trash))
