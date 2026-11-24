"""The committed a = 4.125 Kephart records are pinned to the default margin (#522).

Recorded evidence only: the runs are remote (hours long). The test checks that
the recorded margin is the production default of each plane and that the
declared detector-weighted lengths lie in the +-20% band of the comparison. It
does not test a physical dechanneling boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyrite.materials.channeling import planar_profile
from pyrite.montecarlo.channeling import (
    DECHANNELING_CRITICAL_ANGLE_MULTIPLE,
    dechanneling_margin_eV,
)

RECORDS = Path(__file__).parents[3] / "docs/validation/check-records/planar_fiber_runs"
PLANE = {"si110": (1, 1, 0), "si100": (1, 0, 0)}
CASES = ("si110_17", "si100_17", "si100_54", "si110_54")


def _margin(case: str) -> float:
    return float(dechanneling_margin_eV(planar_profile("silicon", PLANE[case[:5]], 293.15)))


@pytest.mark.parametrize("case", CASES)
def test_production_record_uses_default_margin_and_passes(case):
    assert DECHANNELING_CRITICAL_ANGLE_MULTIPLE == 4.125
    run = json.loads((RECORDS / f"dm-a-scan/run-{case}-a4.125.json").read_text())["runs"][0]
    assert run["margin_eV"] == pytest.approx(_margin(case), rel=1e-9)
    assert (run["max_order"], run["n_cells"], run["dz_ang"]) == (64, 32, 15.625)
    for entry in run["cases"]:
        assert 0.8 <= entry["detector_ratio"]["bound"] <= 1.2


@pytest.mark.parametrize("case", CASES)
def test_fibre_record_uses_default_margin_and_passes(case):
    report = json.loads((RECORDS / f"a-scan/report-{case}-a4.125.json").read_text())
    assert report["info"]["margin_eV"] == pytest.approx(_margin(case), rel=1e-9)
    for entry in report["occupation"].values():
        assert 0.8 <= entry["detector"]["ratio"] <= 1.2
