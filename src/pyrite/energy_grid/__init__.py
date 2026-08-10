"""Photon-energy-grid derivation, encoding, and the ``cxr energy-grid`` group.

The package init stays lightweight: derivation submodules
(``apply``/``bounds``/``defaults``/``derive``/``golden``/``job``/``provenance``)
and the case-grid ``encoding`` helpers import without pulling Click or the remote
layer, so the Monte Carlo hot path is unaffected. The heavy Click command group
lives in :mod:`pyrite.energy_grid._command` and is exposed lazily as
``command`` via ``__getattr__`` so ``cxr`` startup and scan workers stay cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrite.energy_grid._command import command as command

__all__ = ["command"]


def __getattr__(name: str):
    if name == "command":
        from pyrite.energy_grid._command import command

        return command
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
