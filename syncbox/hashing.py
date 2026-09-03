"""
Content addressing and content-defined chunking.

TWO IDEAS, BOTH LOAD-BEARING
----------------------------

1. CONTENT ADDRESSING. A file is identified by the hash of its bytes, not by
   its name or its timestamp. This buys you an enormous amount:
     * Comparing two versions across a network is one 16-byte comparison
       instead of a byte-by-byte transfer.
     * "Did this file really change, or did something just touch its mtime?"
       becomes answerable.
     * Renames become free: the content id is unchanged, only the path moved,
       so you transfer nothing.
     * Deduplication across users falls out for free. If ten thousand people
       sync the same PDF, the provider stores one copy.

2. CONTENT-DEFINED CHUNKING (CDC). If you split a file into fixed 4 MB blocks
   and someone inserts one byte at the front, every single block changes and
   you re-upload the whole file. So instead of cutting at fixed offsets, cut
   at offsets *chosen by the content itself*: slide a rolling hash over the
   bytes and cut wherever the low bits of the hash happen to be zero. Insert a
   byte at the front and only the chunk containing it changes; every later
   boundary re-synchronises within one chunk. This is why syncing a 2 GB
   virtual machine image after a small edit uploads megabytes, not gigabytes.
   It is the same trick behind rsync, borg, restic, and every serious backup
   tool.

The rolling hash here is a "gear hash": one table lookup and one shift per
byte, which is about as cheap as this gets.

WHICH HASH?
-----------
BLAKE2b, truncated to 128 bits. Cryptographic strength matters here even
though this is not obviously a security context: if an attacker can produce
two different files with the same content id, they can make your client
silently serve the wrong bytes to another user of the same dedup pool. Use a
real hash. (Dropbox uses SHA-256 over 4 MB blocks; git famously used SHA-1 and
has been migrating away from it ever since.)
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from typing import Iterator, List, Tuple

# --- content ids ---------------------------------------------------------

DIGEST_BYTES = 16
EMPTY_ID = "b2:" + hashlib.blake2b(b"", digest_size=DIGEST_BYTES).hexdigest()


def content_id(data: bytes) -> str:
    """Stable identity of a blob of bytes."""
    return "b2:" + hashlib.blake2b(data, digest_size=DIGEST_BYTES).hexdigest()


def hash_file(path: str, bufsize: int = 1 << 20) -> str:
    """Content id of a file on disk, streaming so we never load it all."""
    h = hashlib.blake2b(digest_size=DIGEST_BYTES)
    with open(path, "rb") as fh:
        while True:
            block = fh.read(bufsize)
            if not block:
                break
            h.update(block)
    return "b2:" + h.hexdigest()


# --- content-defined chunking -------------------------------------------

# A fixed, deterministic table of 256 random 64-bit values. Deterministic
# matters: every client must cut at the same boundaries or dedup does nothing.
_gear_rng = random.Random(0xC0FFEE)
_GEAR: List[int] = [_gear_rng.getrandbits(64) for _ in range(256)]
_MASK64 = (1 << 64) - 1

# Target ~8 KiB chunks. Small enough that the demo shows real dedup, large
# enough that per-chunk overhead is not silly. Production systems use
# 256 KiB - 4 MiB; the trade-off is metadata size against transfer savings.
MIN_CHUNK = 2 * 1024
AVG_CHUNK_BITS = 13          # 1 << 13 == 8192
MAX_CHUNK = 64 * 1024
# We test the HIGH bits of the rolling hash, not the low ones. With a
# left-shifting gear hash, bit k of the hash has only absorbed the last k+1
# bytes, so the low bits see a uselessly short window. The top bits see the
# last 64 bytes and mix them properly. (This is what FastCDC does.)
_CUT_MASK = ((1 << AVG_CHUNK_BITS) - 1) << (64 - AVG_CHUNK_BITS)


def chunk_bytes(data: bytes) -> Iterator[Tuple[int, bytes]]:
    """Split `data` into content-defined chunks, yielding (offset, chunk).

    The invariant that makes this useful: the boundary positions depend only
    on a small window of surrounding bytes, so a local edit perturbs only a
    local set of chunks.
    """
    n = len(data)
    if n == 0:
        return
    start = 0
    i = 0
    h = 0
    while i < n:
        h = ((h << 1) + _GEAR[data[i]]) & _MASK64
        i += 1
        size = i - start
        if size < MIN_CHUNK:
            continue
        if (h & _CUT_MASK) == 0 or size >= MAX_CHUNK:
            yield start, data[start:i]
            start = i
            h = 0
    if start < n:
        yield start, data[start:]


@dataclass(frozen=True)
class Manifest:
    """How to rebuild a file: its content id plus the chunks, in order."""

    content_id: str
    size: int
    chunks: Tuple[str, ...]      # chunk content ids, in file order

    def to_json_obj(self) -> dict:
        return {"content_id": self.content_id, "size": self.size,
                "chunks": list(self.chunks)}

    @staticmethod
    def from_json_obj(obj: dict) -> "Manifest":
        return Manifest(obj["content_id"], obj["size"], tuple(obj["chunks"]))


def build_manifest(data: bytes) -> Tuple[Manifest, dict]:
    """Return (manifest, {chunk_id: chunk_bytes}) for a blob."""
    chunks: List[str] = []
    blobs: dict = {}
    for _offset, chunk in chunk_bytes(data):
        cid = content_id(chunk)
        chunks.append(cid)
        blobs[cid] = chunk
    return Manifest(content_id(data), len(data), tuple(chunks)), blobs


def reassemble(manifest: Manifest, chunk_source) -> bytes:
    """Rebuild the file from chunks, verifying the result.

    The verification is not paranoia. A corrupted chunk, a truncated transfer
    or a hash collision in a lookup table would otherwise write silent garbage
    into the user's file. Sync software that does not verify what it writes
    will, eventually, eat somebody's thesis.
    """
    out = b"".join(chunk_source(cid) for cid in manifest.chunks)
    got = content_id(out)
    if got != manifest.content_id:
        raise IntegrityError(
            f"reassembled content id {got} != expected {manifest.content_id}")
    return out


class IntegrityError(Exception):
    """Raised when reconstructed bytes do not match their advertised hash."""


def similarity(a: bytes, b: bytes) -> Tuple[int, int, int]:
    """(shared_chunks, total_chunks_b, bytes_that_would_transfer).

    Used by the demo to show what delta sync actually saves.
    """
    _, ablobs = build_manifest(a)
    mb, bblobs = build_manifest(b)
    have = set(ablobs)
    need = [cid for cid in mb.chunks if cid not in have]
    return (len(mb.chunks) - len(need), len(mb.chunks),
            sum(len(bblobs[c]) for c in set(need)))
