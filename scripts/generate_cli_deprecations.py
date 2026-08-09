#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`cxr_mc.devtools.cli_deprecations`."""

from cxr_mc.devtools.cli_deprecations import build_deprecations, main

__all__ = ["build_deprecations", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
