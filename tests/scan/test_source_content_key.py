"""``api.simulate`` and sweeps derive one source content key per case (#191).

Both producers hash the cross-section tables the case itself reads; the
dataset's union of table markers is ``dataset_identity``'s concern only.
"""

from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.campaign.profiles import case_content_key
from pyrite.campaign.sweep import Sweep as LegacySweep
from pyrite.campaign.sweep import build_cases
from pyrite.detectors import EnergyBins
from pyrite.montecarlo.runner import case_table_markers
from pyrite.runs import scan


def _inputs():
    beam = pr.Beam(energy_keV=30.0)
    target = pr.Slab("hopg", thickness_ang=1_000.0, tilt_deg=30.0)
    detector = pr.Detector(
        energy_bins=EnergyBins(
            line=np.array([1_000.0, 1_050.0]),
            brem=np.array([1_000.0, 1_050.0]),
        )
    )
    numerics = pr.Numerics(n_electrons=1, n_electrons_brem=1, bremsstrahlung_model="eedl")
    return beam, target, detector, numerics


def _api_digest(monkeypatch) -> tuple[dict, str]:
    beam, target, detector, numerics = _inputs()

    def fake_run_case(case, *, transport_core):
        return {
            "E_grid": np.array([1.0, 2.0]),
            "spec": np.array([3.0, 4.0]),
            "spec_characteristic": np.array([0.25, 0.5]),
            "E_grid_brem": np.array([1.0, 2.0]),
            "brem_wide": np.array([5.0, 6.0]),
            "brem": np.array([5.0, 6.0]),
        }

    monkeypatch.setattr(api, "run_case", fake_run_case)
    result = pr.simulate(beam, target, detector, numerics=numerics)
    return result.case, result.provenance["identity_digest"]


def _sweep_case():
    beam, target, detector, _ = _inputs()
    return build_cases(
        LegacySweep(material="hopg", beam=beam, target=target, detector=detector),
        n_electrons=1,
        n_electrons_brem=1,
        bremsstrahlung_model="eedl",
    )[0]


def test_api_and_sweep_derive_one_key_for_one_case(monkeypatch):
    api_case, api_digest = _api_digest(monkeypatch)
    sweep_case = _sweep_case()
    assert sweep_case == api_case

    assert api.source_content_key_fn()(sweep_case) == api_digest


def test_sweep_key_ignores_unrelated_dataset_table_markers(monkeypatch):
    """A dataset union carrying tables the case never reads leaves its key alone."""
    _, api_digest = _api_digest(monkeypatch)
    case = _sweep_case()
    own = case_table_markers(case)
    assert own
    union = {**own, "unrelated-table": "f" * 64}

    assert case_content_key(case, xsgen_tables=union) != api_digest
    assert api.source_content_key_fn()(case) == api_digest


def test_dataset_identity_keeps_the_full_dataset_markers():
    """The dataset marks every table its runs may read; one case reads a subset."""
    args = Namespace(
        catalog_profile="standard",
        performance_profile=None,
        fidelity="full",
        quick=False,
    )
    _, _, identity, _ = scan._resolved_run(args, "hopg")
    case = _sweep_case()
    dataset = identity["resolved_parameters"]["xsgen_tables"]
    own = case_table_markers(case)

    assert own.items() < dataset.items()
    assert api.source_content_key_fn()(case) != case_content_key(case, xsgen_tables=dataset)


@pytest.mark.parametrize("producer", ["api", "sweep"])
def test_a_table_marker_change_rekeys_cases_that_read_it(monkeypatch, producer):
    import pyrite.montecarlo.runner.case_tables as case_tables

    def keyed(digest):
        monkeypatch.setattr(
            case_tables,
            "_case_stopping_table_records",
            lambda case: [SimpleNamespace(key="sbethe-hopg", digest=digest)],
        )
        if producer == "api":
            return _api_digest(monkeypatch)[1]
        return api.source_content_key_fn()(_sweep_case())

    assert keyed("a" * 64) != keyed("b" * 64)


def test_a_case_reading_no_table_keeps_its_unmarked_key(monkeypatch):
    import pyrite.montecarlo.runner.case_tables as case_tables

    for name in (
        "_case_stopping_table_records",
        "_case_elastic_table_records",
        "_case_bremslib_table_records",
    ):
        monkeypatch.setattr(case_tables, name, lambda case: [])
    case = _sweep_case()

    assert case_table_markers(case) == {}
    assert api.source_content_key_fn()(case) == case_content_key(case)
