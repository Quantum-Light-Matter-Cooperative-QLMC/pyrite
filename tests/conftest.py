"""Import pyarrow FIRST: on Windows, importing pyrite (numpy/scipy DLLs) before
pyarrow leaves pyarrow's arrow DLLs binding against the wrong runtime, and the
altair/pandas tests then hard-crash (native fault in DataFrame construction).
Loading pyarrow here, before any test module imports pyrite, fixes the order.
Harmless when pyarrow is absent (the altair tests skip without it)."""

import os

# Pin the test session to the CPU/NumPy backend (see pyproject.toml: "Tests
# stay CPU-only/fast"). Much of the suite asserts bit-exact fp64 numerics and
# CPU resource-policy behavior that accelerator fp32 paths (CUDA/ROCm/SYCL)
# cannot reproduce, so an ambient PYRITE_MC_BACKEND from a GPU-equipped dev
# box (e.g. sycl on Intel) must not leak into the session. Both spellings are
# set to keep env_value conflict warnings quiet. Set PYRITE_TEST_BACKEND (or
# its CXR_TEST_BACKEND alias) to run the suite against another backend on
# purpose; hardware-gated tests scrub this pin from subprocesses they spawn.
_TEST_BACKEND = os.environ.get("PYRITE_TEST_BACKEND") or os.environ.get("CXR_TEST_BACKEND") or "cpu"
os.environ["PYRITE_MC_BACKEND"] = _TEST_BACKEND
os.environ["CXR_MC_BACKEND"] = _TEST_BACKEND

try:
    import pyarrow  # noqa: F401
except ImportError:
    pass
