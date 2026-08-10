"""Thin ``python -m`` entry-point shims the remote box invokes on a uv-synced
checkout (``python -m pyrite._entry.scan``, ``python -m pyrite._entry.reproduce_zhai``).

Each module is a minimal ``__main__`` wrapper -- the heavy logic lives elsewhere
(pyrite.runs.scan, src/pyrite/apps/anchor_figures.py) so the spawn/forkserver worker re-import
stays cheap. Prefer the installed CLI (``cxr run [PROFILE] -m <material>``) for local use.
"""
