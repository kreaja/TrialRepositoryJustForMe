"""
syncbox - a small but honest file-synchronisation engine.

This package exists to answer one question: how do Dropbox, Google Drive,
OneDrive and Box actually stay consistent when several machines edit the same
tree, go offline, come back, and disagree?

The short answer, and the shape of this package:

    clock.py      version vectors - how we tell "newer" from "different"
    hashing.py    content addressing + content-defined chunking (delta sync)
    fsmodel.py    the one data type that describes a node in the tree
    scanner.py    turning a directory into a snapshot, cheaply
    db.py         the client's durable memory: BASE, REMOTE mirror, journal
    server.py     the "cloud": blob store + metadata with compare-and-swap
    transport.py  a deliberately unreliable network
    planner.py    the reconciler - the heart of the whole thing
    applier.py    making filesystem changes safely and in a legal order
    merge.py      conflict policy: conflicted copies and 3-way text merge
    client.py     one sync cycle, assembled from the parts above
    demo.py       narrated scenarios you can actually run

Nothing here talks to a real network or a real cloud. The "server" is an
in-process object. That is deliberate: every hard part of this problem is
present anyway, and you can single-step through all of it.
"""

__all__ = ["clock", "hashing", "fsmodel", "scanner", "db", "server",
           "transport", "planner", "applier", "merge", "client"]
__version__ = "0.1.0"
