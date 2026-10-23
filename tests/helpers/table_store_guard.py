"""Audit table-store writes, including direct NumPy/file writes and helpers."""

import os
from pathlib import Path


class TableStoreWriteGuard:
    """Refuse changes to xsgen/tables outside temporary directories."""

    def __init__(self, allowed_roots):
        self.allowed_roots = tuple(Path(root).resolve() for root in allowed_roots)

    def __call__(self, event, args):
        if event == "open":
            flags = args[2]
            if not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                return
            paths = args[:1]
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "shutil.rmtree"}:
            paths = args[:1]
        elif event == "os.rename":
            paths = args[:2]
        else:
            return
        for value in paths:
            if not isinstance(value, (str, bytes, os.PathLike)):
                continue
            path = Path(os.fsdecode(value)).resolve()
            parts = path.parts
            if not any(
                a == "xsgen" and b == "tables" for a, b in zip(parts, parts[1:], strict=False)
            ):
                continue
            if not any(path.is_relative_to(root) for root in self.allowed_roots):
                raise AssertionError(
                    f"test attempted to write outside a temporary table store: {path}"
                )
