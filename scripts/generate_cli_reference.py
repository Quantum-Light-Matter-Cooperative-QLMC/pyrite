#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`cxr_mc.devtools.cli_reference`."""

from cxr_mc.devtools.cli_reference import build_reference, main

__all__ = ["build_reference", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
