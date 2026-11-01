"""Device-kernel dispatch policy for the CXR CXR spectrum.

One module owns these switches so every route reads them through the same
object. Both accumulation routes consult them, so a test that forces the
generic path off a CUDA fast path patches this module and nothing else::

    monkeypatch.setattr(_policy, "_USE_JIT_COHERENT_STREAM", False)

Readers therefore access them as ``_policy._USE_JIT_...`` rather than binding
the values at import.
"""

_USE_JIT_CXR_REDUCTION = True
_USE_JIT_COHERENT_REDUCTION = True
_USE_JIT_COHERENT_STREAM = True
_JIT_CXR_BATCH_TARGET = 400_000
_JIT_COHERENT_PAIR_TARGET = 1_000_000
# Incoherent fused-reduction launches group whole ``1_000_000``-pair reduction
# units up to this many (segment, reflection) pairs, using at most this share
# of the device headroom at this many bytes of temporaries per pair (#906).
_JIT_CXR_LAUNCH_PAIR_TARGET = 8_000_000
_JIT_CXR_LAUNCH_HEADROOM_SHARE = 0.25
_JIT_CXR_LAUNCH_BYTES_PER_PAIR = 256
