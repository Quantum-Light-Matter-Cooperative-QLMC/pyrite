"""Transparent compression for checkpoint pickles (TODO P2 #8).

Checkpoints (``checkpoints/<material>/{line,brem}.pkl``) are multi-GB per-material
pickles once a sweep has filled in every config -- slow to write on the GPU box,
slow to ``scp`` back to the laptop, and slow to load in the viz notebook.

Writes use zstandard (``compression.zstd``, stdlib on 3.14, no new dependency).
Measured on a 360-case hopg store (17.0 MB raw pickle): zstd-3 compresses at
435 MB/s to 7.9 MB, where gzip-6 manages 34 MB/s to 11.3 MB -- 13x the write
throughput, 30% smaller, and ~2x the read throughput (715 vs 339 MB/s). A single
``cxr remote pull`` runs three full compress passes (remote ``cxr slim`` dump,
then both local component dumps in ``_checkpoint_store.save``), so the codec, not
the wire, dominated pull time on gzip.

zstd's window also absorbs the cross-record duplication of the shared ``E_grid``
/ ``E_grid_brem`` arrays (~48% of raw bytes, only 4 distinct grids per material)
that gzip's 32 KB window could not reach: deduplicating those grids by reference
before the dump shrinks the raw pickle 17.0 -> 8.8 MB but the zstd output only
7.88 -> 7.85 MB. That is why records still store their grid inline -- the codec
already collects that win, with no schema change.

Backward compatibility is the load-side contract: every checkpoint on disk
predating this module is a plain pickle, and every one written between gzip
adoption and zstd is gzip. There is no filename/extension change to signal the
format (every path stays ``<stem>.pkl``), so :func:`load` sniffs the magic
header and picks the matching reader -- no migration step, no version field, and
all three generations sit side by side in one directory indistinguishably to
every caller.
"""

import gzip
import io
import pickle
from typing import IO, Any

# stdlib since 3.14 (the version this project pins); ty's typeshed lags.
from compression import zstd  # ty: ignore[unresolved-import]

_GZIP_MAGIC = b"\x1f\x8b"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

#: Default zstd level. Level 3 is zstd's own default: within 1% of the size of
#: level 10 on checkpoint payloads for 4x the throughput, and it is what a live
#: sweep pays on every finished config.
DEFAULT_LEVEL = 3
#: Level requested by callers asking for the smallest artifact (``cxr remote
#: pull --level9``). Level 19 buys ~4% over level 3 at ~11 MB/s -- opt-in only.
MAX_LEVEL = 19
#: Accepted ``--compresslevel`` bounds (zstd's regular, non-ultra range).
LEVEL_RANGE = (1, 22)

_WRITE_BUFFER = 1 << 20


def dump(obj: Any, path: str, *, compresslevel: int | None = None) -> None:
    """Pickle ``obj`` to ``path``, zstd-compressed.

    ``compresslevel`` is a zstd level in :data:`LEVEL_RANGE`; ``None`` uses
    :data:`DEFAULT_LEVEL`. Callers own atomicity (write to a ``.tmp`` sibling,
    then ``os.replace``); this function just does the encode+write.
    """
    with open(path, "wb") as raw:
        dump_stream(obj, raw, compresslevel=compresslevel)


def dump_stream(
    obj: Any, stream: IO[bytes] | io.RawIOBase, *, compresslevel: int | None = None
) -> None:
    """Encode ``obj`` into an already-open binary ``stream`` (does not close it).

    Lets ``cxr slim -o -`` push straight down a pipe, so the box's compress pass
    overlaps the ssh transfer instead of landing a whole temp artifact on box
    disk first.

    zstd frames carry no timestamp or filename, so equivalent dumps are
    byte-identical without the header pinning gzip needed.
    """
    level = DEFAULT_LEVEL if compresslevel is None else compresslevel
    with zstd.ZstdFile(stream, "wb", level=level) as compressed:
        # pickle.dump emits many small writes; buffering them keeps the
        # compressor working on MB-scale blocks instead of per-array fragments.
        with io.BufferedWriter(compressed, buffer_size=_WRITE_BUFFER) as buffered:
            pickle.dump(obj, buffered, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: str) -> Any:
    """Unpickle ``path``, transparently handling the current zstd format, the
    previous gzip one, and legacy plain-pickle checkpoints.

    Sniffs the magic header rather than trusting the extension (all three
    formats live under the same ``<stem>.pkl`` name), so a checkpoint left on
    disk by any earlier version -- laptop or box -- keeps loading with no
    conversion step.
    """
    with open(path, "rb") as f:
        head = f.read(4)
    if head[:4] == _ZSTD_MAGIC:
        with zstd.ZstdFile(path, "rb") as f:
            return pickle.load(f)
    if head[:2] == _GZIP_MAGIC:
        with gzip.GzipFile(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)
