"""
The single data type that describes one node in a synced tree.

WHY TOMBSTONES EXIST
--------------------
The naive model is "a path either has a file or it doesn't". That model cannot
express deletion in a distributed system, and here is the failure it produces:

    Machine A deletes notes.txt while offline.
    Machine A comes back, uploads its state.
    The server sees: A no longer has notes.txt, B still does.

Is that a delete on A that must propagate to B, or is it a create on B that
must propagate to A? Absence carries no information. Without a record saying
"this path was deliberately deleted, at this version", the deleted file comes
back from the dead on the next sync - the single most infamous bug class in
this whole domain.

So deletion is not the removal of a record. It is the writing of a *different*
record: a tombstone, with its own version vector, which propagates like any
other change and can itself be superseded (by someone re-creating the file).

Tombstones accumulate, so they are eventually collected - but only once every
replica has demonstrably seen them (see server.gc_tombstones).

PATH NORMALISATION
------------------
Paths are stored as '/'-separated, relative to the sync root, and Unicode
NFC-normalised. That last one is not pedantry. macOS's HFS+ stored filenames
decomposed (NFD): "é" as "e" + combining accent. Linux stores whatever bytes
you hand it. A file called "résumé.pdf" therefore has two different byte
representations that are the same name to a human, and a sync engine that does
not normalise will cheerfully create both, forever, in an infinite loop.
"""

from __future__ import annotations

import posixpath
import unicodedata
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

from .clock import VersionVector


class Kind(str, Enum):
    FILE = "file"
    DIR = "dir"
    DELETED = "deleted"      # a tombstone


@dataclass(frozen=True)
class Node:
    """What we know about one path.

    `content_id` is None for directories and tombstones. `mtime_ns` and `size`
    are advisory only - they are never used to decide who wins, only to make
    scanning cheap and to restore a plausible timestamp after a download.
    """

    path: str
    kind: Kind
    content_id: Optional[str] = None
    size: int = 0
    mtime_ns: int = 0
    executable: bool = False

    def with_(self, **kw) -> "Node":
        return replace(self, **kw)

    @property
    def exists(self) -> bool:
        return self.kind is not Kind.DELETED

    def same_content_as(self, other: Optional["Node"]) -> bool:
        """Do these two describe the same user-visible state?

        Note what is deliberately *not* compared: mtime. If two machines end up
        with byte-identical content, they are in agreement, and manufacturing a
        conflict over a timestamp difference would be user-hostile.
        """
        if other is None:
            return False
        if self.kind is not other.kind:
            return False
        if self.kind is Kind.FILE:
            return (self.content_id == other.content_id
                    and self.executable == other.executable)
        return True


@dataclass(frozen=True)
class Entry:
    """A Node plus its version vector: the unit the server stores and ships."""

    node: Node
    vv: VersionVector
    modified_by: str = ""

    @property
    def path(self) -> str:
        return self.node.path

    @property
    def kind(self) -> Kind:
        return self.node.kind

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        cid = (self.node.content_id or "")[3:11]
        return f"{self.node.path} [{self.node.kind.value} {cid} {self.vv}]"


# --- path handling -------------------------------------------------------

class UnsafePath(ValueError):
    """A path that we refuse to sync."""


def normalise(path: str) -> str:
    """Canonical form of a sync-relative path."""
    path = unicodedata.normalize("NFC", path.replace("\\", "/"))
    path = posixpath.normpath(path).strip("/")
    if path in (".", ""):
        return ""
    return path


def is_safe(path: str) -> bool:
    """Reject anything that could escape the sync root or wedge a platform.

    Every one of these has been a real vulnerability or a real support ticket
    in a real sync product:
      * '..' segments - a malicious peer writes outside your sync folder.
      * absolute paths - same.
      * a leading '.syncbox' - our own metadata, which must never round-trip.
      * NUL and control characters - rejected by most filesystems anyway.
      * Windows reserved device names (CON, PRN, AUX, NUL, COM1..LPT9) -
        creating these on Windows fails or does something surprising, so a
        Linux client can otherwise upload a file that no Windows peer can
        ever materialise. Real clients rename them on the way down.
    """
    if path == "":
        return False
    if path.startswith("/") or ":" in path.split("/")[0][1:2]:
        return False
    parts = path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return False
    if parts[0] == METADATA_DIR:
        return False
    for part in parts:
        if any(ord(c) < 32 for c in part):
            return False
        stem = part.split(".")[0].upper()
        if stem in _WINDOWS_RESERVED:
            return False
        if part.endswith(" ") or part.endswith("."):
            return False    # Windows silently strips these; round-trip hazard
    return True


def check_safe(path: str) -> str:
    p = normalise(path)
    if not is_safe(p):
        raise UnsafePath(path)
    return p


def parent_of(path: str) -> str:
    return posixpath.dirname(path)


def depth(path: str) -> int:
    return path.count("/")


def ancestors(path: str):
    """Yield 'a', 'a/b' for 'a/b/c.txt' - the dirs that must exist first."""
    parts = path.split("/")[:-1]
    for i in range(1, len(parts) + 1):
        yield "/".join(parts[:i])


METADATA_DIR = ".syncbox"

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
