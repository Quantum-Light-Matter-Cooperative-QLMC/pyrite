"""Import pyarrow FIRST: on Windows, importing cxr_mc (numpy/scipy DLLs) before
pyarrow leaves pyarrow's arrow DLLs binding against the wrong runtime, and the
altair/pandas tests then hard-crash (native fault in DataFrame construction).
Loading pyarrow here, before any test module imports cxr_mc, fixes the order.
Harmless when pyarrow is absent (the altair tests skip without it)."""

try:
    import pyarrow  # noqa: F401
except ImportError:
    pass
