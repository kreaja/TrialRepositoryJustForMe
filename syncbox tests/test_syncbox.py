"""Tests for syncbox. Run: python -m unittest discover -s tests -t ."""

import os
import random
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syncbox.applier import LocalFS
from syncbox.client import SyncClient
from syncbox.clock import VersionVector as V
from syncbox.fsmodel import Entry, Kind, Node, is_safe, normalise
from syncbox.hashing import build_manifest, content_id, reassemble, similarity
from syncbox.merge import conflicted_copy_path, three_way_merge
from syncbox.planner import Op, reconcile
from syncbox.scanner import HashCache, scan
from syncbox.server import CloudServer, CommitStatus
from syncbox.transport import Link, OfflineError


# --- helpers -------------------------------------------------------------

def f(path, cid, kind=Kind.FILE):
    return Node(path, kind, cid if kind is Kind.FILE else None, 1, 0)


def e(path, cid, vv, kind=Kind.FILE):
    return Entry(f(path, cid, kind), vv)


def ops(actions):
    return {(a.op, a.path) for a in actions}


class Harness:
    """A server plus named clients in a temp directory."""

    def __init__(self, *names, failure_rate=0.0):
        self.dir = tempfile.mkdtemp(prefix="syncbox-test-")
        self.server = CloudServer()
        self.clients = {}
        for i, name in enumerate(names):
            root = os.path.join(self.dir, name)
            os.makedirs(root, exist_ok=True)
            self.clients[name] = SyncClient(
                root, name,
                Link(self.server, name, failure_rate=failure_rate, seed=i + 1))

    def __getitem__(self, name):
        return self.clients[name]

    def write(self, name, path, text):
        self[name].fs.atomic_write(path, text.encode())

    def read(self, name, path):
        try:
            return self[name].fs.read(path).decode()
        except FileNotFoundError:
            return None

    def rm(self, name, path):
        os.remove(self[name].fs.abs(path))

    def settle(self, rounds=4):
        for _ in range(rounds):
            for name in self.clients:
                self[name].sync_until_quiet(6)

    def trees(self):
        return {n: self[n].tree() for n in self.clients}

    def converged(self):
        trees = list(self.trees().values())
        return all(t == trees[0] for t in trees)

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# --- version vectors -----------------------------------------------------

class TestVersionVectors(unittest.TestCase):
    def test_zero_counters_are_normalised_away(self):
        self.assertEqual(V.of(a=0), V.empty())

    def test_bump_only_moves_own_counter(self):
        self.assertEqual(V.of(a=1, b=2).bump("a"), V.of(a=2, b=2))

    def test_dominance_and_concurrency(self):
        a, b = V.of(x=2, y=1), V.of(x=1, y=1)
        self.assertTrue(a.dominates(b))
        self.assertFalse(b.dominates(a))
        self.assertFalse(a.concurrent_with(b))
        c = V.of(x=1, y=2)
        self.assertTrue(a.concurrent_with(c))
        self.assertEqual(a.compare(c), "concurrent")

    def test_merge_is_least_upper_bound(self):
        a, b = V.of(x=2, y=1), V.of(x=1, y=3)
        m = a.merge(b)
        self.assertEqual(m, V.of(x=2, y=3))
        self.assertTrue(m.descends_from(a) and m.descends_from(b))

    def test_a_later_timestamp_is_not_a_later_version(self):
        """The whole point: concurrency is structural, not chronological."""
        laptop_edit = V.empty().bump("laptop")
        phone_edit = V.empty().bump("phone")
        self.assertTrue(laptop_edit.concurrent_with(phone_edit))

    def test_json_round_trip(self):
        v = V.of(a=3, b=1)
        self.assertEqual(V.from_json(v.to_json()), v)
        self.assertEqual(V.from_json(None), V.empty())


# --- hashing and chunking ------------------------------------------------

class TestHashing(unittest.TestCase):
    def setUp(self):
        rng = random.Random(7)
        self.data = bytes(rng.getrandbits(8) for _ in range(200_000))

    def test_manifest_round_trip_verifies(self):
        m, blobs = build_manifest(self.data)
        self.assertEqual(reassemble(m, lambda c: blobs[c]), self.data)

    def test_corrupted_chunk_is_detected(self):
        m, blobs = build_manifest(self.data)
        bad = dict(blobs)
        victim = m.chunks[1]
        bad[victim] = b"x" * len(bad[victim])
        with self.assertRaises(Exception):
            reassemble(m, lambda c: bad[c])

    def test_boundaries_resynchronise_after_an_insert(self):
        edited = self.data[:90_000] + b"inserted!" + self.data[90_000:]
        shared, total, _ = similarity(self.data, edited)
        # A fixed-block scheme would reuse nothing after the insertion point.
        self.assertGreater(shared / total, 0.8)

    def test_identical_content_has_identical_id(self):
        self.assertEqual(content_id(b"abc"), content_id(b"abc"))
        self.assertNotEqual(content_id(b"abc"), content_id(b"abd"))


# --- paths ---------------------------------------------------------------

class TestPaths(unittest.TestCase):
    def test_traversal_rejected(self):
        self.assertFalse(is_safe(normalise("../../etc/passwd")))

    def test_metadata_dir_rejected(self):
        self.assertFalse(is_safe(normalise(".syncbox/state.db")))

    def test_windows_reserved_names_rejected(self):
        self.assertFalse(is_safe("CON.txt"))
        self.assertFalse(is_safe("dir/PRN"))
        self.assertTrue(is_safe("dir/console.txt"))

    def test_unicode_normalisation(self):
        import unicodedata
        self.assertEqual(normalise(unicodedata.normalize("NFD", "café.txt")),
                         normalise("café.txt"))


# --- the decision table --------------------------------------------------

class TestPlanner(unittest.TestCase):
    """Every row of the table in planner.py."""

    def test_all_agree_is_a_no_op(self):
        base = {"a": e("a", "b2:1", V.of(p=1))}
        self.assertEqual(reconcile({"a": f("a", "b2:1")}, base, dict(base), "p"),
                         [])

    def test_local_only_change_pushes(self):
        base = {"a": e("a", "b2:1", V.of(p=1))}
        acts = reconcile({"a": f("a", "b2:2")}, base, dict(base), "p")
        self.assertEqual(ops(acts), {(Op.PUSH, "a")})

    def test_remote_only_change_pulls(self):
        base = {"a": e("a", "b2:1", V.of(p=1))}
        remote = {"a": e("a", "b2:2", V.of(p=1, q=1))}
        acts = reconcile({"a": f("a", "b2:1")}, base, remote, "p")
        self.assertEqual(ops(acts), {(Op.PULL, "a")})

    def test_base_distinguishes_our_delete_from_their_create(self):
        """The single most important behaviour in the whole engine."""
        present = {"a": e("a", "b2:1", V.of(p=1))}
        # BASE has it -> we deleted it -> propagate the delete
        acts = reconcile({}, present, dict(present), "p")
        self.assertEqual(ops(acts), {(Op.DELETE_REMOTE, "a")})
        # BASE lacks it -> they created it -> download it
        acts = reconcile({}, {}, present, "p")
        self.assertEqual(ops(acts), {(Op.PULL, "a")})

    def test_identical_concurrent_edits_are_not_a_conflict(self):
        base = {"a": e("a", "b2:old", V.of(p=1))}
        remote = {"a": e("a", "b2:new", V.of(p=1, q=1))}
        acts = reconcile({"a": f("a", "b2:new")}, base, remote, "p")
        self.assertEqual(ops(acts), {(Op.CONVERGE, "a")})

    def test_delete_here_edit_there_resurrects(self):
        base = {"a": e("a", "b2:old", V.of(p=1))}
        remote = {"a": e("a", "b2:new", V.of(p=1, q=1))}
        acts = reconcile({}, base, remote, "p")
        self.assertEqual(ops(acts), {(Op.PULL, "a")})

    def test_edit_here_delete_there_keeps_the_edit(self):
        base = {"a": e("a", "b2:old", V.of(p=1))}
        remote = {"a": Entry(Node("a", Kind.DELETED), V.of(p=1, q=1))}
        acts = reconcile({"a": f("a", "b2:mine")}, base, remote, "p")
        self.assertEqual(ops(acts), {(Op.PUSH, "a")})

    def test_both_deleted_converges_quietly(self):
        base = {"a": e("a", "b2:old", V.of(p=1))}
        remote = {"a": Entry(Node("a", Kind.DELETED), V.of(p=1, q=1))}
        acts = reconcile({}, base, remote, "p")
        self.assertEqual(ops(acts), {(Op.CONVERGE, "a")})

    def test_divergent_binary_edits_split(self):
        base = {"a": e("a", "b2:old", V.of(p=1))}
        remote = {"a": e("a", "b2:theirs", V.of(p=1, q=1))}
        acts = reconcile({"a": f("a", "b2:mine")}, base, remote, "p",
                         allow_text_merge=False)
        self.assertEqual(ops(acts), {(Op.CONFLICT_SPLIT, "a")})

    def test_rename_collapses_to_a_move(self):
        base = {"old/x": e("old/x", "b2:same", V.of(p=1))}
        acts = reconcile({"new/x": f("new/x", "b2:same"),
                          "new": f("new", None, Kind.DIR)},
                         base, dict(base), "p")
        move = [a for a in acts if a.op is Op.MOVE_REMOTE]
        self.assertEqual(len(move), 1)
        self.assertEqual((move[0].from_path, move[0].path), ("old/x", "new/x"))

    def test_move_detection_skips_destinations_with_server_history(self):
        """Regression, found by the convergence fuzzer.

        Replica `three` had deleted b.txt locally and had a file at
        dir/e.txt with b.txt's old content - so it looked like a rename. But
        the server already held a tombstone for dir/e.txt from other
        replicas. Collapsing this into a move would send b.txt's version
        vector to dir/e.txt, where it cannot dominate {one:1, two:1}. The
        CAS rejects it, the next cycle produces the identical plan, and the
        client stalls forever without ever converging.
        """
        same = "b2:same"
        base = {"b.txt": e("b.txt", same, V.of(three=1, two=2))}
        remote = {"b.txt": e("b.txt", same, V.of(three=1, two=2)),
                  "dir/e.txt": Entry(Node("dir/e.txt", Kind.DELETED),
                                     V.of(one=1, two=1))}
        local = {"dir": f("dir", None, Kind.DIR),
                 "dir/e.txt": f("dir/e.txt", same)}
        acts = reconcile(local, base, remote, "three")
        self.assertNotIn(Op.MOVE_REMOTE, {a.op for a in acts})
        self.assertIn((Op.DELETE_REMOTE, "b.txt"), ops(acts))
        self.assertIn((Op.PUSH, "dir/e.txt"), ops(acts))
        # And the push must carry a vector that can actually be accepted.
        push = next(a for a in acts if a.op is Op.PUSH)
        proposed = (V.empty().merge(push.remote.vv).bump("three"))
        self.assertTrue(proposed.descends_from(V.of(one=1, two=1)))

    def test_genuine_rename_still_collapses(self):
        """The guard above must not break the case move detection exists for."""
        base = {"old/x": e("old/x", "b2:same", V.of(p=1))}
        acts = reconcile({"new": f("new", None, Kind.DIR),
                          "new/x": f("new/x", "b2:same")},
                         base, dict(base), "p")
        self.assertIn(Op.MOVE_REMOTE, {a.op for a in acts})

    def test_deletes_are_ordered_deepest_first(self):
        base = {p: e(p, "b2:1", V.of(p=1)) for p in ("d", "d/e", "d/e/f.txt")}
        acts = reconcile({}, base, dict(base), "p")
        paths = [a.path for a in acts if a.op is Op.DELETE_REMOTE]
        self.assertEqual(paths, ["d/e/f.txt", "d/e", "d"])

    def test_downloads_are_ordered_shallowest_first(self):
        remote = {"d": e("d", None, V.of(q=1), Kind.DIR),
                  "d/e": e("d/e", None, V.of(q=1), Kind.DIR),
                  "d/e/f.txt": e("d/e/f.txt", "b2:1", V.of(q=1))}
        acts = reconcile({}, {}, remote, "p")
        self.assertEqual([a.path for a in acts], ["d", "d/e", "d/e/f.txt"])


# --- merging -------------------------------------------------------------

class TestMerge(unittest.TestCase):
    def test_disjoint_edits_merge_cleanly(self):
        r = three_way_merge(b"a\nb\nc\n", b"a\nb\nc\nd\n", b"A\nb\nc\n")
        self.assertTrue(r.clean)
        self.assertEqual(r.merged, b"A\nb\nc\nd\n")

    def test_overlapping_edits_do_not_merge(self):
        r = three_way_merge(b"a\nb\n", b"a\nX\n", b"a\nY\n")
        self.assertFalse(r.clean)
        self.assertEqual(r.conflict_regions, 1)

    def test_identical_edits_are_not_a_conflict(self):
        r = three_way_merge(b"a\nb\n", b"a\nZ\n", b"a\nZ\n")
        self.assertTrue(r.clean)

    def test_binary_is_refused(self):
        self.assertIsNone(three_way_merge(b"\x00\x01", b"\x00\x02",
                                          b"\x00\x03").merged)

    def test_conflicted_name_keeps_the_extension(self):
        p = conflicted_copy_path("a/b/report.tar.gz", "laptop")
        self.assertTrue(p.startswith("a/b/report.tar (conflicted copy from laptop"))
        self.assertTrue(p.endswith(".gz"))


# --- filesystem ----------------------------------------------------------

class TestLocalFS(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.fs = LocalFS(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_atomic_write_creates_parents(self):
        self.fs.atomic_write("x/y/z.txt", b"hello")
        self.assertEqual(self.fs.read("x/y/z.txt"), b"hello")

    def test_no_temp_files_left_behind(self):
        self.fs.atomic_write("a.txt", b"1")
        self.assertEqual(os.listdir(self.fs.tmp), [])

    def test_escape_attempt_raises(self):
        with self.assertRaises(ValueError):
            self.fs.abs("../../outside")

    def test_delete_goes_to_trash(self):
        self.fs.atomic_write("a.txt", b"precious")
        self.fs.to_trash("a.txt")
        self.assertFalse(self.fs.exists("a.txt"))
        self.assertEqual(len(self.fs.trash_contents()), 1)

    def test_two_cycle_swap(self):
        self.fs.atomic_write("a", b"A")
        self.fs.atomic_write("b", b"B")
        self.fs.apply_moves([("a", "b"), ("b", "a")])
        self.assertEqual((self.fs.read("a"), self.fs.read("b")), (b"B", b"A"))

    def test_three_cycle_rotate(self):
        for n in "abc":
            self.fs.atomic_write(n, n.encode())
        self.fs.apply_moves([("a", "b"), ("b", "c"), ("c", "a")])
        self.assertEqual((self.fs.read("a"), self.fs.read("b"),
                          self.fs.read("c")), (b"c", b"a", b"b"))

    def test_chain_moves(self):
        self.fs.atomic_write("x", b"X")
        self.fs.atomic_write("y", b"Y")
        self.fs.apply_moves([("x", "y"), ("y", "z")])
        self.assertEqual((self.fs.read("y"), self.fs.read("z")), (b"X", b"Y"))
        self.assertFalse(self.fs.exists("x"))


# --- scanner -------------------------------------------------------------

class TestScanner(unittest.TestCase):
    def test_cache_avoids_rehashing_unchanged_files(self):
        d = tempfile.mkdtemp()
        try:
            for i in range(5):
                with open(os.path.join(d, f"f{i}.txt"), "w") as fh:
                    fh.write("x" * 100)
            future = 10 ** 19
            cache = HashCache()
            first = scan(d, cache, now_ns=future)
            self.assertEqual(cache.misses, 5)
            cache.hits = cache.misses = 0
            second = scan(d, cache, now_ns=future)
            self.assertEqual((cache.hits, cache.misses), (5, 0))
            self.assertEqual(first, second)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_metadata_directory_is_never_scanned(self):
        d = tempfile.mkdtemp()
        try:
            LocalFS(d).atomic_write("real.txt", b"x")
            self.assertEqual(sorted(scan(d)), ["real.txt"])
        finally:
            shutil.rmtree(d, ignore_errors=True)


# --- server --------------------------------------------------------------

class TestServer(unittest.TestCase):
    def setUp(self):
        self.s = CloudServer()
        m, blobs = build_manifest(b"payload")
        for c in self.s.missing_chunks(list(m.chunks)):
            self.s.put_chunk(c, blobs[c])
        self.s.put_manifest(m)
        self.node = Node("a", Kind.FILE, m.content_id, 7, 0)

    def test_compare_and_swap_rejects_concurrent_writes(self):
        self.assertEqual(self.s.commit(self.node, V.of(p=1), "p", "o1").status,
                         CommitStatus.OK)
        r = self.s.commit(self.node, V.of(q=1), "q", "o2")
        self.assertEqual(r.status, CommitStatus.CONFLICT)
        self.assertEqual(r.server_entry.vv, V.of(p=1))

    def test_retry_with_same_op_id_is_idempotent(self):
        self.s.commit(self.node, V.of(p=1), "p", "o1")
        before = self.s.stats.commits
        self.s.commit(self.node, V.of(p=1), "p", "o1")
        self.assertEqual(self.s.stats.commits, before)
        self.assertEqual(self.s.stats.idempotent_replays, 1)

    def test_content_must_exist_before_metadata(self):
        ghost = Node("b", Kind.FILE, "b2:nonexistent", 1, 0)
        self.assertEqual(self.s.commit(ghost, V.of(p=1), "p", "o3").status,
                         CommitStatus.MISSING_CONTENT)

    def test_feed_coalesces_repeated_edits(self):
        for i in range(5):
            self.s.commit(self.node, V.of(p=i + 1), "p", f"op{i}")
        entries, cursor, _ = self.s.changes_since(0)
        self.assertEqual(len(entries), 1)          # one path, latest state only
        self.assertEqual(self.s.changes_since(cursor)[0], [])

    def test_chunk_id_is_verified_on_upload(self):
        with self.assertRaises(ValueError):
            self.s.put_chunk("b2:lies", b"different bytes")

    def test_tombstones_pinned_by_a_lagging_replica(self):
        self.s.commit(self.node, V.of(p=1), "p", "o1")
        self.s.commit(Node("a", Kind.DELETED), V.of(p=2), "p", "o2")
        self.s.report_cursor("p", self.s._seq)
        self.s.report_cursor("slow", 0)
        self.assertEqual(self.s.gc_tombstones(), 0)
        self.s.report_cursor("slow", self.s._seq)
        self.assertEqual(self.s.gc_tombstones(), 1)


# --- end to end ----------------------------------------------------------

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.h = Harness("laptop", "phone")

    def tearDown(self):
        self.h.close()

    def test_files_propagate(self):
        self.h.write("laptop", "a/b.txt", "hello")
        self.h.settle()
        self.assertEqual(self.h.read("phone", "a/b.txt"), "hello")
        self.assertTrue(self.h.converged())

    def test_rename_transfers_no_content(self):
        self.h.write("laptop", "big.bin", "z" * 100_000)
        self.h.settle()
        before = self.h.server.stats.chunk_bytes_uploaded
        self.h["laptop"].fs.move("big.bin", "renamed.bin")
        self.h.settle()
        self.assertEqual(self.h.server.stats.chunk_bytes_uploaded, before)
        self.assertTrue(self.h["phone"].fs.exists("renamed.bin"))
        self.assertFalse(self.h["phone"].fs.exists("big.bin"))

    def test_offline_edits_conflict_and_keep_both(self):
        self.h.write("laptop", "p.txt", "base\n")
        self.h.settle()
        self.h["laptop"].link.go_offline()
        self.h["phone"].link.go_offline()
        self.h.write("laptop", "p.txt", "laptop version\n")
        self.h.write("phone", "p.txt", "phone version\n")
        self.h["laptop"].link.go_online()
        self.h["phone"].link.go_online()
        self.h.settle()
        self.assertTrue(self.h.converged())
        contents = {self.h.read("laptop", p)
                    for p in self.h["laptop"].tree() if p.endswith(".txt")}
        self.assertIn("laptop version\n", contents)
        self.assertIn("phone version\n", contents)

    def test_disjoint_offline_edits_merge(self):
        self.h.write("laptop", "list.txt", "a\nb\nc\n")
        self.h.settle()
        self.h["laptop"].link.go_offline()
        self.h["phone"].link.go_offline()
        self.h.write("laptop", "list.txt", "a\nb\nc\nd\n")
        self.h.write("phone", "list.txt", "A\nb\nc\n")
        self.h["laptop"].link.go_online()
        self.h["phone"].link.go_online()
        self.h.settle()
        self.assertTrue(self.h.converged())
        self.assertEqual(self.h.read("laptop", "list.txt"), "A\nb\nc\nd\n")
        self.assertEqual(len(self.h["laptop"].tree()), 1)   # no conflicted copy

    def test_deleted_file_does_not_resurrect(self):
        self.h.write("laptop", "gone.txt", "x")
        self.h.settle()
        self.h.rm("laptop", "gone.txt")
        self.h.settle()
        self.assertIsNone(self.h.read("phone", "gone.txt"))
        self.h.settle()                        # and again, for good measure
        self.h.settle()
        self.assertIsNone(self.h.read("phone", "gone.txt"))
        self.assertIsNone(self.h.read("laptop", "gone.txt"))

    def test_edit_beats_delete(self):
        self.h.write("laptop", "d.txt", "one\n")
        self.h.settle()
        self.h["laptop"].link.go_offline()
        self.h["phone"].link.go_offline()
        self.h.rm("laptop", "d.txt")
        self.h.write("phone", "d.txt", "one\ntwo\n")
        self.h["laptop"].link.go_online()
        self.h["phone"].link.go_online()
        self.h.settle()
        self.assertEqual(self.h.read("laptop", "d.txt"), "one\ntwo\n")
        self.assertTrue(self.h.converged())

    def test_lost_acknowledgement_does_not_duplicate(self):
        self.h.write("laptop", "r.txt", "once")
        self.h["laptop"].link.fail_after_commit = True
        self.h.settle()
        self.assertEqual(self.h.server.entries["r.txt"].vv, V.of(laptop=1))
        self.assertGreaterEqual(self.h.server.stats.idempotent_replays, 1)

    def test_work_queued_while_offline_is_delivered_later(self):
        self.h["laptop"].link.go_offline()
        self.h.write("laptop", "queued.txt", "written on a plane")
        rep = self.h["laptop"].sync_once()
        self.assertTrue(rep.offline)
        self.assertIsNone(self.h.read("phone", "queued.txt"))
        self.h["laptop"].link.go_online()
        self.h.settle()
        self.assertEqual(self.h.read("phone", "queued.txt"), "written on a plane")

    def test_flaky_network_still_converges(self):
        h = Harness("a", "b", failure_rate=0.3)
        try:
            for i in range(6):
                h.write("a", f"f{i}.txt", f"content {i}")
            h.settle(rounds=8)
            self.assertTrue(h.converged())
            self.assertEqual(len(h["b"].tree()), 6)
        finally:
            h.close()


# --- the strongest test we have -----------------------------------------

class TestConvergenceFuzz(unittest.TestCase):
    """Random edits on three replicas with random disconnections.

    Convergence is the only property this system actually promises, so it is
    the property worth fuzzing. We make no claim about *what* the final tree
    contains - that depends on conflict policy - only that all three replicas
    agree on it once the noise stops.
    """

    PATHS = ["a.txt", "b.txt", "dir/c.txt", "dir/d.txt",
             "dir/deep/e.txt", "dir/deep/f.bin", "g.txt"]

    def run_one(self, seed):
        rng = random.Random(seed)
        h = Harness("one", "two", "three", failure_rate=0.15)
        try:
            for _step in range(80):
                name = rng.choice(list(h.clients))
                client = h[name]
                action = rng.choices(
                    ["write", "delete", "rename", "sync", "toggle"],
                    weights=[5, 2, 1, 6, 2])[0]
                path = rng.choice(self.PATHS)
                try:
                    if action == "write":
                        h.write(name, path, f"{name}-{rng.randrange(1000)}\n")
                    elif action == "delete" and client.fs.exists(path):
                        os.remove(client.fs.abs(path))
                    elif action == "rename" and client.fs.exists(path):
                        dest = rng.choice(self.PATHS)
                        if dest != path and not client.fs.exists(dest):
                            client.fs.move(path, dest)
                    elif action == "sync":
                        client.sync_once()
                    elif action == "toggle":
                        (client.link.go_offline() if client.link.online
                         else client.link.go_online())
                except (OfflineError, OSError):
                    pass
            # Everyone reconnects and syncs until the dust settles.
            for c in h.clients.values():
                c.link.go_online()
                c.link.failure_rate = 0.0
            h.settle(rounds=8)
            trees = h.trees()
            self.assertTrue(
                h.converged(),
                f"seed {seed} diverged:\n" +
                "\n".join(f"  {n}: {sorted(t)}" for n, t in trees.items()))
        finally:
            h.close()

    def test_converges_over_many_seeds(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                self.run_one(seed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
