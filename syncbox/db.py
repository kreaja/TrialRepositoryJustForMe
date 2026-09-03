"""
The client's durable memory. Three tables carry the whole algorithm.

BASE  - "what this machine and the server last agreed on"
        This is the third leg of the three-way merge and the most important
        table in the system. Without it you can compare LOCAL against REMOTE
        and see they differ, but you cannot tell *which side moved*, so you
        cannot tell a create from a delete or an edit from a stale copy.
        Everything downstream depends on BASE being written only after a
        change has actually been applied and durably stored.

REMOTE - a mirror of the server's metadata, brought forward by the change
        feed. Keeping it locally means a sync cycle needs one incremental
        request, not a full listing of the user's entire Drive.

JOURNAL - intents. Written *before* we act, cleared after. If the process is
        killed mid-upload, restart replays from here. Each entry carries an
        operation id so a retry after an ambiguous failure ("did the server
        get it before the connection dropped?") is idempotent rather than a
        duplicate.

The ordering discipline matters more than the schema:

    apply the change  ->  fsync  ->  update BASE  ->  clear JOURNAL

Do it in any other order and a crash in the gap either loses a change or
re-applies one. This is the same write-ahead-log reasoning a database uses,
for the same reason.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, Iterable, List, Optional, Tuple

from .clock import VersionVector
from .fsmodel import Entry, Kind, Node
from .scanner import HashCache

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS base (
    path        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    content_id  TEXT,
    size        INTEGER NOT NULL DEFAULT 0,
    mtime_ns    INTEGER NOT NULL DEFAULT 0,
    executable  INTEGER NOT NULL DEFAULT 0,
    vv          TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS remote (
    path        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    content_id  TEXT,
    size        INTEGER NOT NULL DEFAULT 0,
    mtime_ns    INTEGER NOT NULL DEFAULT 0,
    executable  INTEGER NOT NULL DEFAULT 0,
    vv          TEXT NOT NULL DEFAULT '{}',
    modified_by TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS hashcache (
    path     TEXT PRIMARY KEY,
    ino      INTEGER NOT NULL,
    size     INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    ctime_ns INTEGER NOT NULL,
    cid      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS journal (
    op_id    TEXT PRIMARY KEY,
    seq      INTEGER,
    kind     TEXT NOT NULL,
    payload  TEXT NOT NULL,
    state    TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS journal_state ON journal(state, seq);
"""


class ClientDB:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        # WAL so a crash mid-write leaves a recoverable database rather than a
        # truncated one; NORMAL rather than FULL because we fsync the payload
        # files ourselves and can replay the journal.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- meta ------------------------------------------------------------

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?",
                                (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        self.conn.commit()

    @property
    def cursor(self) -> int:
        return int(self.get_meta("cursor", "0"))

    @cursor.setter
    def cursor(self, value: int) -> None:
        self.set_meta("cursor", str(value))

    # -- BASE / REMOTE ---------------------------------------------------

    def _load(self, table: str) -> Dict[str, Entry]:
        out: Dict[str, Entry] = {}
        for r in self.conn.execute(f"SELECT * FROM {table}"):
            node = Node(path=r["path"], kind=Kind(r["kind"]),
                        content_id=r["content_id"], size=r["size"],
                        mtime_ns=r["mtime_ns"], executable=bool(r["executable"]))
            mod = r["modified_by"] if "modified_by" in r.keys() else ""
            out[r["path"]] = Entry(node, VersionVector.from_json(r["vv"]), mod)
        return out

    def load_base(self) -> Dict[str, Entry]:
        return self._load("base")

    def load_remote(self) -> Dict[str, Entry]:
        return self._load("remote")

    def _put(self, table: str, entry: Entry, commit: bool = True) -> None:
        n = entry.node
        cols = "path,kind,content_id,size,mtime_ns,executable,vv"
        vals = [n.path, n.kind.value, n.content_id, n.size, n.mtime_ns,
                int(n.executable), entry.vv.to_json()]
        if table == "remote":
            cols += ",modified_by"
            vals.append(entry.modified_by)
        marks = ",".join("?" * len(vals))
        updates = ",".join(f"{c}=excluded.{c}" for c in cols.split(",")[1:])
        self.conn.execute(
            f"INSERT INTO {table}({cols}) VALUES({marks}) "
            f"ON CONFLICT(path) DO UPDATE SET {updates}", vals)
        if commit:
            self.conn.commit()

    def put_base(self, entry: Entry, commit: bool = True) -> None:
        self._put("base", entry, commit)

    def put_remote(self, entry: Entry, commit: bool = True) -> None:
        self._put("remote", entry, commit)

    def drop_base(self, path: str) -> None:
        self.conn.execute("DELETE FROM base WHERE path=?", (path,))
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    # -- hash cache ------------------------------------------------------

    def load_hash_cache(self) -> HashCache:
        cache = HashCache()
        for r in self.conn.execute("SELECT * FROM hashcache"):
            cache.entries[r["path"]] = (r["ino"], r["size"], r["mtime_ns"],
                                        r["ctime_ns"], r["cid"])
        return cache

    def save_hash_cache(self, cache: HashCache) -> None:
        self.conn.execute("DELETE FROM hashcache")
        self.conn.executemany(
            "INSERT INTO hashcache(path,ino,size,mtime_ns,ctime_ns,cid) "
            "VALUES(?,?,?,?,?,?)",
            [(p, *rec) for p, rec in cache.entries.items()])
        self.conn.commit()

    # -- journal ---------------------------------------------------------

    def journal_add(self, op_id: str, kind: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO journal(op_id,kind,payload,state) "
            "VALUES(?,?,?, 'pending')", (op_id, kind, json.dumps(payload)))
        self.conn.commit()

    def journal_pending(self) -> List[Tuple[str, str, dict, int]]:
        rows = self.conn.execute(
            "SELECT op_id,kind,payload,attempts FROM journal "
            "WHERE state='pending' ORDER BY rowid").fetchall()
        return [(r["op_id"], r["kind"], json.loads(r["payload"]), r["attempts"])
                for r in rows]

    def journal_done(self, op_id: str) -> None:
        self.conn.execute("DELETE FROM journal WHERE op_id=?", (op_id,))
        self.conn.commit()

    def journal_failed(self, op_id: str) -> None:
        self.conn.execute(
            "UPDATE journal SET attempts=attempts+1 WHERE op_id=?", (op_id,))
        self.conn.commit()

    def journal_size(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) c FROM journal WHERE state='pending'"
        ).fetchone()["c"]
