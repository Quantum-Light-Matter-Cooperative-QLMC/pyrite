"""Nonempty pair/annihilation coupling in a production-prepared filter scene."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite import api
from pyrite.montecarlo.runner.photon_aperture import prepare_photon_case
from pyrite.montecarlo.transport.photon_coupling import (
    coupled_ledger_residual_eV,
    track_coupled_histories,
)
from pyrite.montecarlo.transport.photon_histories import POSITRON
from tests.montecarlo.test_photon_production_cuda import cuda as cuda
from tests.montecarlo.test_photon_reentry_cuda import _filtered_scene

pytestmark = pytest.mark.slow


def test_cuda_filter_born_pairs_release_rest_once_through_full_queue(cuda, monkeypatch):
    from pyrite.montecarlo.transport import photon_navigation as nav
    from pyrite.montecarlo.transport.pair_production import PAIR_THRESHOLD_EV

    scene = _filtered_scene()
    # A 20 µm filter contains the <8.1 keV pair daughters, making this an
    # annihilation-return probe rather than a charged escape realization.
    scene = replace(scene, filters=(replace(scene.filter_plates[0], thickness_mm=0.02),))
    context = prepare_photon_case(api.build_case(scene))
    box = context["boxes"][0]

    def pair_only(element, energy):
        energy = np.asarray(energy)
        return {
            name: np.where(energy > PAIR_THRESHOLD_EV, 1000.0, 0.0)
            if name == "pair_nuclear"
            else np.zeros_like(energy)
            for name in nav.PHOTON_CHANNEL_ORDER
        }

    # Control only rates: final states, electron/positron transport, vacancies,
    # annihilation and returned photons remain real, on the selected CUDA core.
    monkeypatch.setattr(nav, "photon_cross_sections_ang2", pair_only)
    n = 2
    out = track_coupled_histories(
        np.tile(box.center_ang, (n, 1)),
        np.tile(box.axes[2], (n, 1)),
        np.full(n, 1.03e6),
        np.ones(n),
        np.arange(n),
        np.zeros(n, np.int64),
        context["layers"],
        boxes=context["boxes"],
        width_ang=context["width"],
        height_ang=context["height"],
        seed=341_907,
        photon_cutoff_eV=1000.0,
        charged_stage=context["stage"],
        backend="cuda",
    )
    first = out.charged["round"] == 0
    assert np.count_nonzero(first & (out.charged["species"] == POSITRON)) == n
    assert np.all(out.charged["fate"][first] == "launched")
    assert out.rounds > 0 and out.ledger["complete"].all()
    np.testing.assert_allclose(out.ledger["positron_rest_eV"], 0.0, atol=1e-8)
    assert np.count_nonzero(out.exits["energy_eV"] > 4e5) >= 2 * n
    np.testing.assert_allclose(coupled_ledger_residual_eV(out.ledger), 0.0, atol=1e-7)
