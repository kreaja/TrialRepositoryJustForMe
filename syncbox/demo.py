"""
Runnable scenarios. `python -m syncbox.demo` walks through all of them.

Each scenario sets up a situation, prints the plan the reconciler produced and
the reasoning attached to each action, then shows the resulting state on every
replica. The point is to watch the same small decision table produce sensible
behaviour in situations that look, from the outside, like completely different
features.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

from .client import SyncClient, SyncReport
from .fsmodel import Kind
from .planner import describe
from .scanner import scan
from .server import CloudServer
from .transport import Link

BAR = "=" * 78


class World:
    """A server plus some replicas, in a throwaway directory."""

    def __init__(self, *names: str, failure_rate: float = 0.0):
        self.dir = tempfile.mkdtemp(prefix="syncbox-")
        self.server = CloudServer()
        self.clients: Dict[str, SyncClient] = {}
        for i, name in enumerate(names):
            root = os.path.join(self.dir, name)
            os.makedirs(root, exist_ok=True)
            link = Link(self.server, name, failure_rate=failure_rate, seed=i + 1)
            self.clients[name] = SyncClient(root, name, link)

    def __getitem__(self, name: str) -> SyncClient:
        return self.clients[name]

    def write(self, name: str, path: str, text: str) -> None:
        self[name].fs.atomic_write(path, text.encode())

    def read(self, name: str, path: str) -> Optional[str]:
        try:
            return self[name].fs.read(path).decode()
        except FileNotFoundError:
            return None

    def delete(self, name: str, path: str) -> None:
        os.remove(self[name].fs.abs(path))

    def rename(self, name: str, src: str, dst: str) -> None:
        self[name].fs.move(src, dst)

    def sync(self, name: str, label: str = "", passes: int = 4) -> List[SyncReport]:
        reports = self[name].sync_until_quiet(passes)
        head = label or f"{name} syncs"
        print(f"\n  -- {head} " + "-" * max(0, 60 - len(head)))
        printed = False
        for rep in reports:
            if not rep.plan and not rep.notes:
                continue
            printed = True
            print(describe(rep.plan))
            for note in rep.notes:
                print(f"      note: {note}")
        if not printed:
            print("    (nothing to do)")
        return reports

    def show(self, *names: str) -> None:
        names = names or tuple(self.clients)
        print("\n  state:")
        for name in names:
            files = {p: n for p, n in scan(self[name].root).items()
                     if n.kind is Kind.FILE}
            listing = ", ".join(sorted(files)) or "(empty)"
            print(f"    {name:8s} {listing}")

    def converged(self) -> bool:
        trees = [self[n].tree() for n in self.clients]
        return all(t == trees[0] for t in trees)

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def _title(n: int, text: str) -> None:
    print(f"\n{BAR}\n{n}. {text}\n{BAR}")


# ---------------------------------------------------------------------------

def scenario_first_sync(w: World) -> None:
    _title(1, "THE EASY CASE - one machine uploads, another downloads")
    print("""
  Nothing subtle here yet. Laptop creates two files, pushes them; phone pulls
  the change feed and materialises them. Notice that even this simple case
  needs the metadata commit to happen AFTER the content upload, so the server
  never advertises a file whose bytes are not yet fetchable.""")
    w.write("laptop", "notes/todo.txt", "buy milk\ncall mum\n")
    w.write("laptop", "notes/plan.txt", "1. learn SQL\n2. learn TypeScript\n")
    w.sync("laptop")
    w.sync("phone")
    w.show()
    print(f"\n  converged: {w.converged()}")


def scenario_edit_propagation(w: World) -> None:
    _title(2, "SEQUENTIAL EDIT - and how little gets transferred")
    print("""
  Phone appends a line. The file is re-uploaded, but only the chunks that
  actually changed cross the wire - and the server already had the rest.""")
    w.write("phone", "notes/todo.txt", "buy milk\ncall mum\nbook dentist\n")
    reps = w.sync("phone")
    print(f"      bytes uploaded: {sum(r.bytes_up for r in reps)}")
    w.sync("laptop")
    print(f"\n  laptop now reads: {w.read('laptop', 'notes/todo.txt')!r}")


def scenario_rename(w: World) -> None:
    _title(3, "RENAME - the case that punishes naive designs")
    print("""
  A naive engine sees 'a file disappeared and a different file appeared' and
  re-uploads the whole thing. Because identity here is the content hash, the
  planner pairs the delete with the create and emits a single move: two
  metadata commits, zero bytes of content.""")
    big = "x" * 200_000
    w.write("laptop", "archive/big.iso", big)
    w.sync("laptop", "laptop uploads a 200 KB file")
    w.sync("phone")
    w[ "laptop" ].fs.mkdir("vault")
    w.rename("laptop", "archive/big.iso", "vault/big.iso")
    reps = w.sync("laptop", "laptop renames it")
    print(f"      bytes uploaded for the rename: {sum(r.bytes_up for r in reps)}")
    reps = w.sync("phone", "phone applies the rename")
    print(f"      bytes downloaded by phone:     {sum(r.bytes_down for r in reps)}")
    w.show()


def scenario_offline_conflict(w: World) -> None:
    _title(4, "OFFLINE CONCURRENT EDITS - a real conflict")
    print("""
  Both machines go offline and edit the same file differently. Neither edit
  knows about the other, so their version vectors are CONCURRENT - not one
  newer than the other. No timestamp comparison can rescue this; picking the
  'later' one would silently delete somebody's work.

  Policy: the remote version keeps the path (so every peer makes the same
  choice and they converge), and the local version is preserved as a
  conflicted copy which is then uploaded as a file in its own right.""")
    w["laptop"].link.go_offline()
    w["phone"].link.go_offline()
    w.write("laptop", "notes/plan.txt", "1. learn SQL\n2. learn C++\n")
    w.write("phone", "notes/plan.txt", "1. learn SQL\n2. learn C#\n")
    print("\n  (both edited notes/plan.txt while disconnected)")
    w["laptop"].link.go_online()
    w["phone"].link.go_online()
    w.sync("laptop", "laptop reconnects first - it wins the race, uncontested")
    reps = w.sync("phone", "phone reconnects and finds the world changed")
    for rep in reps:
        for copy in rep.conflicted_copies:
            print(f"      preserved as: {copy}")
    w.sync("laptop", "laptop picks up the conflicted copy")
    w.show()
    print(f"\n  converged: {w.converged()}  (nobody lost an edit)")


def scenario_auto_merge(w: World) -> None:
    _title(5, "CONCURRENT EDITS THAT DO NOT ACTUALLY COLLIDE")
    print("""
  Same setup - two offline edits to one file - but this time they touch
  different lines. Because we kept the common ancestor, a three-way merge can
  tell an insertion from a deletion and combine both edits. This is the same
  reason git can merge two branches that touched different parts of a file.

  Note the honesty of the fallback: if the two edits overlap, we do NOT write
  conflict markers into a consumer's document. We keep both versions instead.""")
    w.write("laptop", "notes/shopping.txt", "milk\nbread\neggs\n")
    w.sync("laptop"); w.sync("phone")
    w["laptop"].link.go_offline(); w["phone"].link.go_offline()
    w.write("laptop", "notes/shopping.txt", "milk\nbread\neggs\ncoffee\n")
    w.write("phone", "notes/shopping.txt", "oat milk\nbread\neggs\n")
    w["laptop"].link.go_online(); w["phone"].link.go_online()
    w.sync("laptop", "laptop uploads its version")
    w.sync("phone", "phone merges rather than conflicting")
    w.sync("laptop", "laptop picks up the merged result")
    print(f"\n  merged file on laptop: {w.read('laptop','notes/shopping.txt')!r}")
    print(f"  merged file on phone : {w.read('phone','notes/shopping.txt')!r}")


def scenario_delete_resurrection(w: World) -> None:
    _title(6, "DELETION - and the file that rises from the dead")
    print("""
  Deleting is where the BASE table earns its keep. When the laptop deletes a
  file, the server does not forget it: it stores a TOMBSTONE, a real record
  with its own version vector. The phone's feed carries the tombstone and the
  phone removes its copy.

  If deletion were merely the absence of a record, the phone - which still has
  the file - would look like it had just created it, and would helpfully
  re-upload it. The file would come back. Forever.""")
    w.write("laptop", "notes/temp.txt", "scratch\n")
    w.sync("laptop"); w.sync("phone")
    print(f"\n  phone has it: {w.read('phone','notes/temp.txt')!r}")
    w.delete("laptop", "notes/temp.txt")
    w.sync("laptop", "laptop deletes it")
    tomb = w.server.entries["notes/temp.txt"]
    print(f"      server record is now: {tomb}")
    w.sync("phone", "phone applies the deletion")
    print(f"\n  phone has it: {w.read('phone','notes/temp.txt')!r}")
    trash = w["phone"].fs.trash_contents()
    print(f"  and it is recoverable from the phone's trash: {trash[-1:]}")
    w.sync("phone", "phone syncs again - does the file come back?")
    print(f"  phone still does not have it: {w.read('phone','notes/temp.txt') is None}")


def scenario_delete_vs_edit(w: World) -> None:
    _title(7, "DELETE ON ONE SIDE, EDIT ON THE OTHER")
    print("""
  Genuinely ambiguous: one user decided the file was rubbish, another was busy
  improving it. There is no correct answer, only a defensible policy.

  Ours: keep the bytes. A deletion is trivial for a human to repeat; an edit
  is not. So the edit wins and the file survives.""")
    w.write("laptop", "notes/draft.txt", "chapter one\n")
    w.sync("laptop"); w.sync("phone")
    w["laptop"].link.go_offline(); w["phone"].link.go_offline()
    w.delete("laptop", "notes/draft.txt")
    w.write("phone", "notes/draft.txt", "chapter one\nchapter two\n")
    w["laptop"].link.go_online(); w["phone"].link.go_online()
    w.sync("laptop", "laptop pushes its deletion")
    w.sync("phone", "phone has an edit to a file the server says is deleted")
    w.sync("laptop", "laptop learns the file is back")
    print(f"\n  laptop: {w.read('laptop','notes/draft.txt')!r}")
    print(f"  phone : {w.read('phone','notes/draft.txt')!r}")


def scenario_lost_ack(w: World) -> None:
    _title(8, "THE AMBIGUOUS FAILURE - did the write land or not?")
    print("""
  The nastiest failure mode in any networked system: the server commits, then
  the connection dies before the acknowledgement arrives. The client cannot
  distinguish this from 'the request never arrived', so it must retry - and
  the retry must not apply the change twice.

  Fix: the client derives the operation id from the intent (replica, path,
  content, version) rather than generating a random one. A retry - even after
  a crash and restart - produces the same id, and the server answers from its
  memo table instead of committing again.""")
    w.write("laptop", "notes/receipt.txt", "important\n")
    w["laptop"].link.fail_after_commit = True
    before = w.server.stats.commits
    w.sync("laptop", "laptop pushes; the acknowledgement is lost")
    after = w.server.stats.commits
    print(f"\n      server commits performed: {after - before}")
    print(f"      idempotent replays served: {w.server.stats.idempotent_replays}")
    print(f"      versions of the file on the server: "
          f"{w.server.entries['notes/receipt.txt'].vv}  (one edit, not two)")


def scenario_three_machines(w3: World) -> None:
    _title(11, "THREE MACHINES, FLAKY NETWORK, EVERYBODY EDITING")
    print("""
  The convergence property, stated properly: if all replicas stop editing and
  are given enough connectivity, they all end up with identical trees. Not
  'immediately', not 'atomically' - eventually. That is the real promise every
  one of these products makes, and it is much weaker than users assume.

  Here three machines edit overlapping files on a link that drops 25% of
  requests, then everybody syncs until quiet.""")
    for name in ("desk", "laptop", "phone"):
        w3.write(name, "shared/log.txt", f"line from {name}\n")
        w3.write(name, f"private/{name}.txt", f"notes belonging to {name}\n")
    for _round in range(3):
        for name in ("desk", "laptop", "phone"):
            w3[name].sync_until_quiet(6)
    trees = {n: set(w3[n].tree()) for n in ("desk", "laptop", "phone")}
    print("\n  final file lists:")
    for name, files in trees.items():
        print(f"    {name:8s} {len(files)} entries")
    for name, files in trees.items():
        for f in sorted(files):
            if "conflicted" in f:
                print(f"      {name}: kept {f}")
                break
    print(f"\n  all three replicas identical: {w3.converged()}")
    st = w3.server.stats
    print(f"  server: {st.commits} commits, {st.conflicts} CAS rejections, "
          f"{st.chunks_skipped_dedup} chunks skipped by dedup")


def scenario_cas_race(w: World) -> None:
    _title(9, "THE RACE - why the server must refuse writes, not merge them")
    print("""
  Everything so far has been polite: one machine at a time. Now interleave.
  The phone pulls the feed, decides what to do... and while it is deciding,
  the laptop commits. The phone's plan is now built on a stale view.

  This is the lost-update problem. If the server accepted both writes on a
  last-one-wins basis, the laptop's edit would be silently gone - no error,
  no conflicted copy, no trace. Compare-and-swap turns that silent loss into
  a loud rejection, which the client then resolves with the same three-way
  machinery as any other conflict.""")
    from .planner import reconcile
    from .scanner import scan as _scan
    from .client import SyncReport

    w.write("laptop", "notes/race.txt", "the laptop's version\n")
    w.write("phone", "notes/race.txt", "the phone's version\n")

    phone = w["phone"]
    phone._pull_feed()                    # phone reads the world...
    plan = reconcile(_scan(phone.root, phone.cache), phone.db.load_base(),
                     phone.db.load_remote(), phone.replica)
    print("\n  phone's plan, built against the world as it was:")
    print(describe([a for a in plan if a.path.endswith("race.txt")]))

    w.sync("laptop", "...but the laptop commits first")

    rep = SyncReport(replica="phone")
    for action in plan:
        phone._execute(action, rep)
    print("\n  -- phone now executes its stale plan " + "-" * 24)
    for note in rep.notes:
        print(f"      note: {note}")
    print(f"      CAS rejections so far: {w.server.stats.conflicts}")

    w.sync("phone", "phone reconciles properly on the next pass")
    w.sync("laptop", "laptop picks up whatever survived")
    both = sorted(p for p in w["laptop"].tree() if "race" in p)
    print("\n  files surviving:")
    for f in both:
        print(f"    {f}  ->  {w.read('laptop', f)!r}")
    print(f"\n  neither edit was lost: {len(both) == 2}")


def scenario_tombstone_gc(w: World) -> None:
    _title(10, "GARBAGE COLLECTING TOMBSTONES - carefully")
    print("""
  Tombstones cannot accumulate forever, but you cannot expire them on a timer
  either. A laptop that has been in a drawer for six months still has the file
  and still believes it is synced; if the tombstone is gone when it wakes up,
  the file looks new and gets re-uploaded - resurrection, again.

  So collection is gated on the MINIMUM CURSOR across all known replicas: drop
  only what everybody has demonstrably seen.""")
    for name in w.clients:
        w[name].sync_once()
    tombs = sorted(p for p, e in w.server.entries.items()
                   if e.kind is Kind.DELETED)
    print(f"\n  tombstones held : {tombs}")
    print(f"  replica cursors : {w.server.replica_cursors}")

    everyone = set(w.server.replica_cursors) | {"drawer-laptop"}
    removed = w.server.gc_tombstones(known_replicas=everyone)
    print(f"\n  a device we have never heard from is enrolled ('drawer-laptop')")
    print(f"  collected: {removed}   <- nothing, because it might still have the files")

    w.server.report_cursor("drawer-laptop", 0)
    removed = w.server.gc_tombstones(known_replicas=everyone)
    print(f"\n  it checks in, but at cursor 0 - still far behind")
    print(f"  collected: {removed}   <- still nothing")

    w.server.report_cursor("drawer-laptop", w.server._seq)
    removed = w.server.gc_tombstones(known_replicas=everyone)
    print(f"\n  it finally catches up to the head of the feed")
    print(f"  collected: {removed}   <- now it is safe")
    print(f"  tombstones remaining: "
          f"{[p for p, e in w.server.entries.items() if e.kind is Kind.DELETED]}")


def main() -> None:
    print(BAR)
    print("syncbox - how cloud file sync actually works")
    print(BAR)

    w = World("laptop", "phone")
    try:
        scenario_first_sync(w)
        scenario_edit_propagation(w)
        scenario_rename(w)
        scenario_offline_conflict(w)
        scenario_auto_merge(w)
        scenario_delete_resurrection(w)
        scenario_delete_vs_edit(w)
        scenario_lost_ack(w)
        scenario_cas_race(w)
        scenario_tombstone_gc(w)
    finally:
        w.cleanup()

    w3 = World("desk", "laptop", "phone", failure_rate=0.25)
    try:
        scenario_three_machines(w3)
    finally:
        w3.cleanup()

    print(f"\n{BAR}\ndone.\n{BAR}")


if __name__ == "__main__":
    main()
