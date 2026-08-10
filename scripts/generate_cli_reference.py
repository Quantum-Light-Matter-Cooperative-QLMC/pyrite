#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`pyrite.devtools.cli_reference`."""

from pyrite.devtools.cli_reference import build_reference, main

__all__ = ["build_reference", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
