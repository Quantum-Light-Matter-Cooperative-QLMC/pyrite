"""Photon-energy-grid derivation, encoding, and artifact application.

The package stays lightweight and Click-free: derivation submodules
(``apply``/``bounds``/``defaults``/``derive``/``golden``/``job``/``provenance``)
and the case-grid ``encoding`` helpers import without pulling Click or the remote
layer, so the Monte Carlo hot path is unaffected.

The ``pyrite energy-grid`` command group lives in
:mod:`pyrite.cli.commands.energy_grid` with every other CLI surface. It used to
sit here as ``_command`` and be re-exported lazily, which put this package and
``cli`` in an import cycle; nothing in this package reaches up to it now.
"""

from __future__ import annotations
