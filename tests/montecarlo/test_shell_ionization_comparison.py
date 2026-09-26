"""Pinned Bote--Salvat transcription and EEDL shell comparison anchors.

Validation: eedl-shell-ionization-comparison
"""

import numpy as np
import pytest

from pyrite.validation.shell_ionization import (
    BoteSalvatUnavailableError,
    compare_shells,
    load_bote_salvat,
    parse_bote_salvat,
)


@pytest.fixture(scope="module")
def parameters():
    try:
        return load_bote_salvat()
    except BoteSalvatUnavailableError as exc:
        pytest.skip(str(exc))


def test_parser_reads_shell_vectors_and_row_major_fit_matrices():
    text = '''const BoteSalvatElectron = (
BoteSalvatElementDatum(1,[1,2],[3,4],[5 6 7 8; 9 10 11 12],[13,14],[15 16 17 18 19; 20 21 22 23 24]),
BoteSalvatElementDatum(2,[2],[3],[4 5 6 7],[8],[9 10 11 12 13]),
)
"""'''
    parsed = parse_bote_salvat(text)
    assert tuple(parsed) == (1, 2)
    assert parsed[1].shells == ("K", "L1")
    np.testing.assert_array_equal(parsed[1].g, [[5, 6, 7, 8], [9, 10, 11, 12]])
    np.testing.assert_array_equal(parsed[1].a, [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24]])
    assert parsed[2].edge_eV[0] == 8


# xion.f values transcribed in BoteSalvatICX.jl test/xione.jl at the pinned
# 8520cf5 commit: (Z, shell, incident eV, cross section cm^2).
_XION = (
    (12, "L3", 7.71792e01, 3.85315e-18),
    (12, "L3", 1.29569e04, 1.24124e-18),
    (23, "L1", 1.53993e04, 2.63509e-20),
    (67, "K", 1.63117e05, 1.36404e-23),
    (78, "M5", 2.98538e03, 2.02808e-20),
)


@pytest.mark.parametrize(("z", "shell", "energy_eV", "reference_cm2"), _XION)
def test_transcription_matches_xion_f(parameters, z, shell, energy_eV, reference_cm2):
    value = parameters[z].cross_section_cm2(shell, energy_eV)
    assert value == pytest.approx(reference_cm2, rel=0.01)


@pytest.mark.parametrize(
    ("element", "z", "shell", "bounds"),
    [
        ("Si", 14, "K", ((0.86, 0.90), (0.90, 0.95), (0.91, 0.96), (1.01, 1.06))),
        ("W", 74, "L3", ((0.21, 0.25), (0.35, 0.39), (0.70, 0.75), (0.91, 0.95))),
    ],
)
def test_eedl_over_bote_salvat_committed_band(parameters, element, z, shell, bounds):
    overvoltages = (1.1, 2.0, 10.0, 100.0)
    result = next(
        r for r in compare_shells(parameters, element, z, overvoltages) if r.shell == shell
    )
    for ratio, (low, high) in zip(result.eedl_edge_ratio, bounds, strict=True):
        assert low <= ratio <= high
    assert result.neighboring_node_ratio.shape == (4, 2)
