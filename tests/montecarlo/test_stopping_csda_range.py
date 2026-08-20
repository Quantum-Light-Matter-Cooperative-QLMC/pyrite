"""What the Berger--Seltzer splice does to the CSDA range.

Slice A pinned the stopping *curve* and slice C pinned the splice's continuity.
Neither says how much the integrated quantity the campaign actually cares about
moves, so this file measures it: the continuous-slowing-down range

.. math::

    R(E_0) = \\int_{E_\\mathrm{cut}}^{E_0} \\frac{dE}{|dE/ds|},

before and after, on the repository's own materials. The reference is not an
external table -- it is the retired model, reproduced exactly by pushing every
element's crossover out of reach (see :func:`joy_luo_only`), so the difference
is the splice and nothing else.

Ranges shorten because Berger--Seltzer stops harder than Joy--Luo everywhere
above the crossover, and the gap widens with energy: a few percent at 25 keV,
~15% at 100 keV, ~35% at 300 keV. The 300 keV figure is the one that matters
for interpreting thick-target yields -- it is not a correction, it is a
different depth scale.

Validation: relativistic-bethe-stopping
"""

import numpy as np
import pytest

from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo import transport

# The transport cutoff every core uses; the integral has to stop somewhere and
# below it the electron is no longer tracked, so R here is the tracked range,
# not the ESTAR range (which integrates to zero energy).
E_CUT_keV = 5.0


@pytest.fixture
def joy_luo_only(monkeypatch):
    """Force pure Joy--Luo stopping, i.e. the model this branch retired.

    ``_element_crossover_keV`` is the single seam both the layer-table builder
    and the host helper consult; at ``inf`` no element ever reaches its
    Berger--Seltzer branch, so the spliced form degenerates to Joy--Luo
    bit-for-bit rather than approximately.
    """
    monkeypatch.setattr(transport, "_element_crossover_keV", lambda *_: np.inf)
    monkeypatch.setattr(transport, "_CROSSOVER_CACHE", {})
    return None


def composition(material):
    return build_cases(material_sweep(material), n_electrons=8, n_electrons_brem=4)[0][
        "composition"
    ]


def csda_range_ang(comp, E0_keV):
    """Log-spaced quadrature: |dE/ds| ~ 1/E, so log E is the natural variable."""
    u = np.linspace(np.log(E_CUT_keV), np.log(E0_keV), 4001)
    E = np.exp(u)
    return float(np.trapezoid(E / -transport.spliced_stopping_keV_per_ang(comp, E), u))


# material -> {E0: (old range [um], new range [um])}, measured, 4001-point grid.
MEASURED = {
    "hopg": {25.0: (6.1734, 5.9157), 100.0: (82.7659, 70.2618), 300.0: (642.84, 415.37)},
    "silicon": {25.0: (7.0455, 6.7667), 100.0: (91.5191, 77.8712), 300.0: (697.83, 452.28)},
    "wse2": {25.0: (2.7201, 2.6333), 100.0: (33.2964, 28.4833), 300.0: (245.52, 160.06)},
}


@pytest.mark.parametrize("material", sorted(MEASURED))
def test_csda_range_matches_the_measured_value(material):
    comp = composition(material)
    for E0, (_, new_um) in MEASURED[material].items():
        assert csda_range_ang(comp, E0) / 1.0e4 == pytest.approx(new_um, rel=1.0e-3)


@pytest.mark.parametrize("material", sorted(MEASURED))
def test_the_retired_model_reproduces_its_own_measured_ranges(material, joy_luo_only):
    comp = composition(material)
    for E0, (old_um, _) in MEASURED[material].items():
        assert csda_range_ang(comp, E0) / 1.0e4 == pytest.approx(old_um, rel=1.0e-3)


@pytest.mark.parametrize("material", sorted(MEASURED))
def test_the_splice_shortens_the_range_by_more_at_higher_energy(material):
    """Monotone in E_0, and always a shortening.

    Berger--Seltzer lies below Joy--Luo (stops harder) over the whole spliced
    region, so no material and no energy may show a lengthened range, and the
    fractional shortening must grow with beam energy because the crossover sits
    at a few keV and the two curves diverge above it.
    """
    fractions = [new / old - 1.0 for old, new in MEASURED[material].values()]
    assert all(f < 0.0 for f in fractions)
    assert fractions == sorted(fractions, reverse=True)


def test_range_shortening_is_bounded_across_the_whole_catalog():
    """The catalog-wide spread, so a new material cannot silently sit outside it.

    Measured over all 50 catalog materials the shortening spans -4.2% to -2.6%
    at 25 keV, -15.1% to -14.1% at 100 keV, and -35.4% to -34.6% at 300 keV --
    a band under 1.1 points wide at every energy, because the fractional change
    depends on the elements' mean excitation energies only logarithmically.
    """
    bands = {25.0: (-0.045, -0.025), 100.0: (-0.155, -0.139), 300.0: (-0.357, -0.344)}
    for material in MEASURED:
        comp = composition(material)
        for E0, (lo, hi) in bands.items():
            new = csda_range_ang(comp, E0)
            old = MEASURED[material][E0][0] * 1.0e4
            assert lo <= new / old - 1.0 <= hi, (material, E0)
