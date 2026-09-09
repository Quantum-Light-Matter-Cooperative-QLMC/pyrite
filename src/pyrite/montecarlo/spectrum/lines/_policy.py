"""Device-kernel dispatch policy for the CXR line spectrum.

One module owns the five switches so every route reads them through the same
object. Both accumulation routes consult them, so a test that forces the
generic path off a CUDA fast path patches this module and nothing else::

    monkeypatch.setattr(_policy, "_USE_JIT_COHERENT_STREAM", False)

Readers therefore access them as ``_policy._USE_JIT_...`` rather than binding
the values at import.
"""

_USE_JIT_LINE_REDUCTION = True
_USE_JIT_COHERENT_REDUCTION = True
_USE_JIT_COHERENT_STREAM = True
_JIT_LINE_BATCH_TARGET = 400_000
_JIT_COHERENT_PAIR_TARGET = 1_000_000
