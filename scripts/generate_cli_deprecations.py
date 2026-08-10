#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`pyrite.devtools.cli_deprecations`."""

from pyrite.devtools.cli_deprecations import build_deprecations, main

__all__ = ["build_deprecations", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
