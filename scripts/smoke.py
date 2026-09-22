#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`pyrite.devtools.smoke`."""

from pyrite.devtools.smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
