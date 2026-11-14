"""Free-electron limit of the #1148 zone PXR modulation check.

A free plane-wave electron (no transverse potential, one occupied plane wave)
has no transverse transition, so the finite-slab amplitude must reduce to the
plane-wave reference even where a Bragg pole lies a few tens of eV above the
free root. Without the pre-crystal path of the diffracted free wave the check
returned 2.24 at the (-1 -1 3) minus-in peak (fresh-context verifier, #1148).

Validation: cr-planar-zone-pxr-modulation
"""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location(
        "channeling_zone_pxr_modulation", ROOT / "checks" / "channeling_zone_pxr_modulation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def free_model(check):
    helper = runpy.run_path(str(ROOT / "checks" / "channeling_detector_components.py"))
    case = helper["_profile_case"]("channeling_demo", "silicon")
    gamma = 1.0 + float(case["E0_keV"]) / 510.99895
    orders = np.arange(-16, 17)
    period = 1.920113109
    pw = orders * 2 * np.pi / period
    energies = check.HBARC**2 / (2 * gamma * check.ME) * pw**2
    order = np.argsort(energies)
    rho = np.zeros((len(orders), len(orders)), complex)
    rho[0, 0] = 1.0  # the p = 0 plane wave, first in energy order
    zone = dict(
        rho=rho[None],
        energies=energies[order][None],
        vectors=np.eye(len(orders))[:, order][None].astype(complex),
        orders=orders,
        kappa=np.zeros(1),
        period=period,
        gamma=gamma,
        depths=1,
    )
    return check.Model(case, zone), helper, case


@pytest.mark.slow
def test_free_electron_reproduces_plane_wave_near_a_bragg_pole(check, free_model):
    model, helper, case = free_model
    centres = dict(check._centres(model, helper, case))
    n = centres["(-1 -1 3) minus-in peak"]
    hkl = (1, 1, -3)
    g = dict(model.reflections)[hkl]
    base = model.native(n, g, hkl)
    assert base is not None and 0 < base["pole"] < 60.0  # pole inside the window
    part = model.modulated(n, g, base)
    assert part["estimate"] == pytest.approx(part["free_born"], rel=1e-2)
    assert part["estimate"] == pytest.approx(base["native"], rel=1e-2)
