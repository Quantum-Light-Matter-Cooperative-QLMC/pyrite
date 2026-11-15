"""Domain-free internal helpers: file I/O, hashing, environment, formatting.

A leaf package: modules here import only the standard library and third-party
packages, never another ``pyrite`` module (import-linter contract
``util-imports-no-pyrite``). Nothing here is public API.
"""
