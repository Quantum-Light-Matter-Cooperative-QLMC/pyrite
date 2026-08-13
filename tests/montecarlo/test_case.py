"""Typed simulation-case schema and legacy-serialization compatibility."""

import pickle
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.montecarlo import Case


def _legacy_case(**sweep_kwargs):
    sweep = Sweep(
        material="mose2",
        beam=BeamSpec(energy_keV=30.0),
        tilt_deg=5.0,
        **sweep_kwargs,
    )
    case = build_cases(sweep, n_electrons=2, n_electrons_brem=1)[0]
    return case.to_dict() if isinstance(case, Case) else case


def test_case_round_trips_legacy_mapping_byte_for_byte():
    legacy = _legacy_case()

    case = Case(**legacy)

    assert list(case) == list(legacy)
    assert pickle.dumps(case.to_dict(), protocol=5) == pickle.dumps(legacy, protocol=5)


def test_case_process_pickle_preserves_absent_keys():
    case = Case(**_legacy_case())

    restored = pickle.loads(pickle.dumps(case, protocol=5))

    assert restored.to_dict().keys() == case.to_dict().keys()
    assert pickle.dumps(restored.to_dict(), protocol=5) == pickle.dumps(case.to_dict(), protocol=5)


def test_case_round_trips_conditional_divergence_keys_in_legacy_order():
    sweep = Sweep(
        material="mose2",
        beam=BeamSpec(energy_keV=30.0),
        tilt_deg=5.0,
        tilt_azim_deg=180.0,
        groove_spacing_ang=2.0e4,
    )
    legacy = build_cases(
        sweep,
        n_electrons=2,
        n_electrons_brem=1,
        coherent_emission=True,
        xray_dispersion="refractive",
    )[0]

    case = Case(**legacy)

    assert list(case.to_dict()) == list(legacy)
    assert case.to_dict()["coherent_emission"] is True
    assert case.to_dict()["xray_dispersion"] == "refractive"


def test_case_is_frozen():
    case = Case(**_legacy_case())

    with pytest.raises(FrozenInstanceError):
        case.seed = 7  # ty: ignore[invalid-assignment]


def test_case_rejects_missing_required_field():
    payload = _legacy_case()
    del payload["crystal"]

    with pytest.raises(TypeError, match="crystal"):
        Case(**payload)


def test_case_rejects_unknown_field():
    payload: dict[str, Any] = _legacy_case()
    payload["unexpected_key"] = True

    with pytest.raises(TypeError, match="unexpected_key"):
        Case(**payload)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"Ne": 0}, "Ne must be a positive integer"),
        ({"E0_keV": -1.0}, "E0_keV must be finite and positive"),
        ({"crystal_width_mm": None}, "crystal_width_mm and crystal_height_mm"),
        ({"coherent_emission": False}, "coherent_emission must be absent or True"),
        ({"xray_dispersion": "vacuum"}, "xray_dispersion must be absent or 'refractive'"),
    ],
)
def test_case_validates_physical_and_divergence_invariants(change, message):
    with pytest.raises(ValueError, match=message):
        Case(**{**_legacy_case(), **change})


def test_run_case_accepts_typed_case_and_legacy_mapping(monkeypatch):
    import pyrite.montecarlo.runner as runner

    case = Case(**_legacy_case())
    seen = []
    monkeypatch.setattr(runner, "_transport_case", lambda payload, *args, **kwargs: seen.append(payload))
    monkeypatch.setattr(runner, "_spectrum_case", lambda payload, *args, **kwargs: {"case": payload})

    typed = runner.run_case(case)
    legacy = runner.run_case(case.to_dict())

    assert typed["case"] is case
    assert legacy["case"] == case.to_dict()
    assert seen == [case, case.to_dict()]
