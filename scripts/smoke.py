#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`cxr_mc.devtools.smoke`."""

from __future__ import annotations

from cxr_mc.devtools.smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
