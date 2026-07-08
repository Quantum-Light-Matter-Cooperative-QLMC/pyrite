"""Transparent gzip (de)compression for checkpoint pickles (TODO P2 #8).

Checkpoints (``checkpoints/<material>.pkl``) are multi-GB per-material pickles
once a sweep has filled in every config -- slow to write on the GPU box, slow
to ``scp`` back to the laptop, and slow to load in the viz notebook. gzip
(stdlib, no new dependency) typically halves or better the on-disk size of
these float-array-heavy pickles at a modest CPU cost, which is a straight win
for both disk footprint and (especially) `dev/remote.py` transfer time.

Backward compatibility is the load-side contract: every checkpoint written
before this change is a plain (uncompressed) pickle, both on the laptop
(``checkpoints/``, ``checkpoints/archive/``) and on the ``qlmc`` box. There is
no filename/extension change to signal the format (every path stays
``<stem>.pkl``), so :func:`load` sniffs the two-byte gzip magic header
(``\\x1f\\x8b``) and picks the matching reader -- no migration step, no
version field, and old + new checkpoints sit side by side in the same
directory indistinguishably to every caller.
"""

import gzip
import pickle
from typing import Any

_GZIP_MAGIC = b"\x1f\x8b"


def dump(obj: Any, path: str, *, compresslevel: int = 6) -> None:
    """Pickle ``obj`` to ``path``, gzip-compressed.

    ``compresslevel=6`` is gzip's default -- most of the size reduction for a
    fraction of the CPU time of the max level 9, which matters here since
    checkpoints are saved after every finished config during a live sweep.
    Callers own atomicity (write to a ``.tmp`` sibling, then ``os.replace``);
    this function just does the encode+write.
    """
    with gzip.GzipFile(path, "wb", compresslevel=compresslevel) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: str) -> Any:
    """Unpickle ``path``, transparently handling both the current
    gzip-compressed format and legacy plain-pickle checkpoints.

    Sniffs the first two bytes for the gzip magic header rather than trusting
    the extension (both formats live under the same ``<stem>.pkl`` name), so a
    pre-compression checkpoint left on disk (laptop or box) keeps loading with
    no conversion step.
    """
    with open(path, "rb") as f:
        head = f.read(2)
    if head == _GZIP_MAGIC:
        with gzip.GzipFile(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)
