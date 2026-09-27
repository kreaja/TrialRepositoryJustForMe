"""
Conflict policy: what to do when two edits are genuinely concurrent.

There is no clever answer. If two people changed the same JPEG on two machines
with no communication between them, no algorithm can produce "the right"
JPEG. The engineering question is not how to avoid that, it is what to do that
never loses work and never surprises the user.

STRATEGY 1: THE CONFLICTED COPY (the universal default)
-------------------------------------------------------
Keep both. One version stays at the original path; the other is renamed to
something obvious and human-readable:

    budget (conflicted copy from laptop 2026-09-03 14-22-05).xlsx

Dropbox does exactly this. Its virtues are that it is impossible to lose data,
it needs no understanding of the file format, and the user can see what
happened without reading a log. Its vice is that it pushes the work onto the
user, and in a shared folder it can spray copies everywhere.

Note the choice of *which* version keeps the original name. We give it to the
remote version, so that every peer converges on the same file at that path
without further negotiation. Giving it to the local version would mean two
machines each think their own edit is canonical, and they would then conflict
again forever.

STRATEGY 2: FORMAT-AWARE MERGE
------------------------------
When you understand the file type you can often merge without asking. Line-
oriented text is the classic case, and the three-way merge below is the same
one git uses: compare each side against the common ancestor rather than
against each other, so you can tell an insertion from a deletion, and only
flag a real conflict where the two sides touched the same region.

Google Docs goes much further: it does not store documents as files at all,
but as an ordered stream of operations, merged by operational transformation
so that concurrent edits *compose* instead of conflicting. That is why Docs
has no conflicted copies and Drive-syncing a .docx does. The lesson is that
the conflict problem gets easier the more the system knows about the data - and
that a general file-sync engine, by definition, knows nothing.

STRATEGY 3: DON'T HAVE CONFLICTS
--------------------------------
Lock the file while someone has it open (Box and SharePoint offer this), or
make everything append-only, or use CRDTs whose merge is mathematically
guaranteed to converge. All of these trade away either offline editing or
generality, which is why none of them is the default for whole-filesystem sync.
"""

from __future__ import annotations

import datetime as _dt
import posixpath
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Sequence, Tuple


def conflicted_copy_path(path: str, replica: str,
                         when: Optional[_dt.datetime] = None,
                         attempt: int = 0) -> str:
    """Build the 'conflicted copy' name, keeping the extension intact.

    Keeping the extension matters: the copy has to stay openable by the same
    application, or the user cannot compare the two versions.
    """
    when = when or _dt.datetime.now()
    stamp = when.strftime("%Y-%m-%d %H-%M-%S")
    directory, name = posixpath.split(path)
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    suffix = f" (conflicted copy from {replica} {stamp}"
    suffix += f" #{attempt})" if attempt else ")"
    newname = stem + suffix + (("." + ext) if ext else "")
    return posixpath.join(directory, newname) if directory else newname


@dataclass
class MergeResult:
    merged: Optional[bytes]
    clean: bool
    conflict_regions: int = 0


def is_probably_text(data: bytes, sniff: int = 8192) -> bool:
    """Cheap binary sniff: NUL bytes and undecodable UTF-8 mean 'don't try'."""
    head = data[:sniff]
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _edits(base: Sequence[str], other: Sequence[str]
           ) -> List[Tuple[int, int, List[str]]]:
    """Regions of `base` that `other` replaced, as (start, end, replacement)."""
    out = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, base, other).get_opcodes():
        if tag != "equal":
            out.append((i1, i2, list(other[j1:j2])))
    return out


def three_way_merge(base: bytes, mine: bytes, theirs: bytes,
                    mine_label: str = "local",
                    theirs_label: str = "remote") -> MergeResult:
    """Merge two divergent versions using their common ancestor.

    The ancestor is what makes this possible. Comparing `mine` to `theirs`
    alone, a line present in one and absent in the other is ambiguous - was it
    added here or deleted there? Against the ancestor the answer is plain, and
    that single extra input is the difference between a merge that works and
    one that guesses.
    """
    if not all(is_probably_text(b) for b in (base, mine, theirs)):
        return MergeResult(None, False)

    b = base.decode("utf-8").splitlines(keepends=True)
    m = mine.decode("utf-8").splitlines(keepends=True)
    t = theirs.decode("utf-8").splitlines(keepends=True)

    mine_edits, their_edits = _edits(b, m), _edits(b, t)
    out: List[str] = []
    pos = 0
    conflicts = 0
    mi = ti = 0

    while mi < len(mine_edits) or ti < len(their_edits):
        me = mine_edits[mi] if mi < len(mine_edits) else None
        th = their_edits[ti] if ti < len(their_edits) else None

        # Take whichever edit starts earlier; if they overlap, decide.
        if th is None or (me is not None and me[1] <= th[0]):
            out.extend(b[pos:me[0]]); out.extend(me[2]); pos = me[1]; mi += 1
        elif me is None or th[1] <= me[0]:
            out.extend(b[pos:th[0]]); out.extend(th[2]); pos = th[1]; ti += 1
        else:
            # Overlapping regions: both sides touched the same lines.
            start, end = min(me[0], th[0]), max(me[1], th[1])
            out.extend(b[pos:start])
            if me[2] == th[2]:
                out.extend(me[2])          # identical edits: not a conflict
            else:
                conflicts += 1
                out.append(f"<<<<<<< {mine_label}\n")
                out.extend(me[2])
                out.append("=======\n")
                out.extend(th[2])
                out.append(f">>>>>>> {theirs_label}\n")
            pos = end
            mi += 1
            ti += 1

    out.extend(b[pos:])
    return MergeResult("".join(out).encode("utf-8"), conflicts == 0, conflicts)
