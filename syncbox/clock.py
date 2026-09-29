"""
Version vectors: how a distributed system tells "newer" apart from "different".

THE PROBLEM WITH TIMESTAMPS
---------------------------
The obvious way to decide which of two versions of a file wins is to compare
wall-clock modification times and keep the later one. This is wrong, and it is
wrong in a way that silently destroys user data:

  * Clocks disagree. A laptop 40 seconds fast will win every race it enters,
    forever, including races it should have lost.
  * Clocks move backwards (NTP steps, timezone bugs, users setting the date,
    virtual machines resuming from a snapshot).
  * Filesystems have different timestamp resolutions - one second on some
    network filesystems, one nanosecond on ext4. Two edits inside the same
    tick are indistinguishable.
  * Most importantly: "later" is not the question. The question is "did this
    edit happen *knowing about* that one?" Two people editing the same file
    on a plane produce two edits with different timestamps, but neither one
    supersedes the other. Picking the later one throws away real work.

WHAT A VERSION VECTOR IS
------------------------
Each replica (each machine, each client install) has a stable id. For every
file, we keep a map from replica id to a counter:

    {"laptop": 3, "desktop": 1}

meaning "this version of the file incorporates 3 edits originating on laptop
and 1 originating on desktop". When a replica changes a file, it increments
*its own* counter and leaves the others alone.

Now comparison is structural, not chronological. Given vectors A and B:

    A dominates B      every counter in A is >= the matching one in B
                       (and A != B). A is strictly newer; it already contains
                       everything B knows. Safe to overwrite B with A.
    A == B             same version. Nothing to do.
    concurrent         A has a counter B lacks AND B has a counter A lacks.
                       Neither knows about the other. This is a genuine
                       conflict and no amount of cleverness makes it not one.
                       A human (or a policy) has to decide.

That third case is the entire reason cloud sync is hard, and it is exactly
the case timestamps paper over.

COST
----
A version vector is O(number of replicas that ever touched this file). That is
fine for personal sync (a handful of devices) and becomes a real problem at
Google Drive scale with millions of collaborators, which is why large systems
use variants: server-assigned monotonic revisions with a single authoritative
ordering point, dotted version vectors, or per-file "who last wrote" plus a
causal-context set. The reasoning below is the same in all of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping


@dataclass(frozen=True)
class VersionVector:
    """An immutable map replica_id -> counter. Missing key means zero."""

    counters: Mapping[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Normalise: drop zeros so that {} and {"a": 0} are the same vector,
        # and freeze into a plain dict we never mutate in place.
        clean = {k: int(v) for k, v in dict(self.counters or {}).items() if v}
        object.__setattr__(self, "counters", clean)

    # -- construction ----------------------------------------------------

    @staticmethod
    def empty() -> "VersionVector":
        return VersionVector({})

    @staticmethod
    def of(**counters: int) -> "VersionVector":
        return VersionVector(counters)

    # -- basic accessors -------------------------------------------------

    def get(self, replica: str) -> int:
        return self.counters.get(replica, 0)

    def replicas(self) -> Iterable[str]:
        return self.counters.keys()

    def is_empty(self) -> bool:
        return not self.counters

    # -- the three operations that matter --------------------------------

    def bump(self, replica: str) -> "VersionVector":
        """Record one new edit made *by* `replica`."""
        nxt: Dict[str, int] = dict(self.counters)
        nxt[replica] = nxt.get(replica, 0) + 1
        return VersionVector(nxt)

    def merge(self, other: "VersionVector") -> "VersionVector":
        """Least upper bound: a version that knows everything both know.

        Used when we resolve a conflict - the merged result genuinely
        supersedes both inputs, so it must dominate both.
        """
        keys = set(self.counters) | set(other.counters)
        return VersionVector({k: max(self.get(k), other.get(k)) for k in keys})

    def descends_from(self, other: "VersionVector") -> bool:
        """True if self knows everything other knows (self >= other)."""
        return all(self.get(k) >= v for k, v in other.counters.items())

    # -- derived relations -----------------------------------------------

    def dominates(self, other: "VersionVector") -> bool:
        """Strictly newer: knows everything other knows, plus something more."""
        return self.descends_from(other) and self != other

    def concurrent_with(self, other: "VersionVector") -> bool:
        """Neither descends from the other: a true conflict."""
        return not self.descends_from(other) and not other.descends_from(self)

    def compare(self, other: "VersionVector") -> str:
        """Human-readable relation, handy for tracing: one of
        'same' | 'newer' | 'older' | 'concurrent'."""
        if self == other:
            return "same"
        if self.dominates(other):
            return "newer"
        if other.dominates(self):
            return "older"
        return "concurrent"

    # -- serialisation ---------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(self.counters, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_json(text: str | None) -> "VersionVector":
        if not text:
            return VersionVector.empty()
        return VersionVector(json.loads(text))

    def __str__(self) -> str:
        if not self.counters:
            return "{}"
        inner = " ".join(f"{k}:{v}" for k, v in sorted(self.counters.items()))
        return "{" + inner + "}"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"VV{self}"

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.counters.items())))
