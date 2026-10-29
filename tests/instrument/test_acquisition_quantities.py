"""Semantic acquisition exposure, edges and threshold stay numeric (#755)."""

import pickle
from dataclasses import asdict, fields, replace

import numpy as np
import pint
import pytest

from pyrite.instrument import Acquisition, NativeAcquisition
from pyrite.instrument.observation import acquisition_payload, payload_digest
from pyrite.units import Q_, ureg

EDGES = (1000.0, 2000.0, 4000.0)


def _payload(acquisition: Acquisition) -> str:
    return payload_digest(acquisition_payload(acquisition, rep_rate_hz=5000, bunch_charge_pc=1))


def test_semantic_spellings_match_numeric_fields_and_identity() -> None:
    semantic = Acquisition(
        exposure=Q_(2000, "ms"),
        measured_edges=Q_([1, 2, 4], "keV"),
        hit_threshold=Q_(0.5, "keV"),
        mode="poisson",
        seed=3,
    )
    numeric = Acquisition(2.0, EDGES, 500.0, "poisson", 3)
    assert semantic == numeric
    assert asdict(semantic) == asdict(numeric)
    assert _payload(semantic) == _payload(numeric)
    assert pickle.loads(pickle.dumps(semantic)) == numeric
    assert replace(semantic) == numeric
    names = {item.name for item in fields(semantic)}
    assert names == {"exposure_s", "measured_edges_eV", "hit_threshold_eV", "mode", "seed"}
    assert semantic.exposure == Q_(2, "s")
    assert semantic.hit_threshold == Q_(500, "eV")
    np.testing.assert_array_equal(semantic.measured_edges.to("eV").magnitude, EDGES)
    assert semantic.measured_edges.units == ureg.eV
    assert isinstance(semantic.measured_edges_eV, tuple)


def test_legacy_calls_and_defaults_are_unchanged() -> None:
    assert Acquisition(1.0, EDGES).hit_threshold_eV == 0.0
    legacy = Acquisition(exposure_s=1 * ureg.ms, measured_edges_eV=Q_([1, 2], "keV"))
    assert (legacy.exposure_s, legacy.measured_edges_eV) == (1e-3, (1000.0, 2000.0))
    with pytest.raises(TypeError, match="requires exposure_s or exposure"):
        Acquisition(measured_edges_eV=EDGES)
    with pytest.raises(TypeError, match="requires measured_edges_eV or measured_edges"):
        Acquisition(1.0)


def test_replacement_and_readback_are_detached() -> None:
    acquisition = Acquisition(1.0, EDGES)
    changed = (
        acquisition.with_exposure(Q_(3, "s"))
        .with_measured_edges(Q_([2, 3], "keV"))
        .with_hit_threshold(Q_(100, "eV"))
    )
    assert changed == Acquisition(3.0, (2000.0, 3000.0), 100.0)
    assert acquisition == Acquisition(1.0, EDGES)
    edges = changed.measured_edges
    edges.magnitude[:] = 0
    edges.ito("keV")
    np.testing.assert_array_equal(changed.measured_edges.magnitude, [2000.0, 3000.0])
    source = Q_(np.array([1.0, 2.0]), "keV")
    owned = Acquisition(exposure=Q_(1, "s"), measured_edges=source)
    source.magnitude[:] = 9
    assert owned.measured_edges_eV == (1000.0, 2000.0)


def test_external_registry_converts_to_project_registry() -> None:
    external = pint.UnitRegistry()
    acquisition = Acquisition(
        exposure=external.Quantity(2, "s"), measured_edges=external.Quantity([1, 2], "keV")
    )
    assert acquisition.measured_edges_eV == (1000.0, 2000.0)
    assert acquisition.exposure._REGISTRY is ureg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exposure_s": 1.0, "exposure": Q_(1, "s"), "measured_edges_eV": EDGES},
        {"exposure_s": 1.0, "measured_edges_eV": EDGES, "measured_edges": Q_(EDGES, "eV")},
        {
            "exposure_s": 1.0,
            "measured_edges_eV": EDGES,
            "hit_threshold_eV": 0.0,
            "hit_threshold": Q_(0, "eV"),
        },
    ],
)
def test_dual_spellings_conflict(kwargs) -> None:
    with pytest.raises(TypeError, match="mutually exclusive"):
        Acquisition(**kwargs)


@pytest.mark.parametrize(
    "keyword, value, error",
    [
        ("exposure", 1.0, TypeError),
        ("exposure", "1 s", TypeError),
        ("exposure", Q_([1.0], "s"), TypeError),
        ("exposure", Q_(1, "eV"), ValueError),
        ("exposure", Q_(0, "s"), ValueError),
        ("exposure", Q_(np.nan, "s"), ValueError),
        ("measured_edges", EDGES, TypeError),
        ("measured_edges", Q_(1, "keV"), TypeError),
        ("measured_edges", Q_([[1, 2]], "keV"), TypeError),
        ("measured_edges", Q_([1 + 1j, 2], "keV"), TypeError),
        ("measured_edges", Q_([1, 2], "s"), ValueError),
        ("measured_edges", Q_([1, np.inf], "keV"), ValueError),
        ("measured_edges", Q_([2, 1], "keV"), ValueError),
        ("hit_threshold", 0.0, TypeError),
        ("hit_threshold", Q_(-1, "eV"), ValueError),
        ("hit_threshold", Q_(1, "dimensionless"), ValueError),
    ],
)
def test_semantic_spellings_refuse_invalid_values(keyword, value, error) -> None:
    kwargs = {"exposure": Q_(1, "s"), "measured_edges": Q_(EDGES, "eV"), keyword: value}
    with pytest.raises(error, match="Acquisition"):
        Acquisition(**kwargs)


def test_uniform_quantity_bounds_match_numeric_edges() -> None:
    semantic = Acquisition.uniform(
        exposure=Q_(1, "s"),
        minimum=Q_(1, "keV"),
        maximum=Q_(3, "keV"),
        bin_width=Q_(500, "eV"),
        hit_threshold=Q_(0.2, "keV"),
    )
    numeric = Acquisition.uniform(
        exposure_s=1.0,
        minimum_eV=1000.0,
        maximum_eV=3000.0,
        bin_width_eV=500.0,
        hit_threshold_eV=200.0,
    )
    assert semantic == numeric
    with pytest.raises(TypeError, match="mutually exclusive"):
        Acquisition.uniform(
            exposure_s=1.0, minimum_eV=0, minimum=Q_(0, "eV"), maximum_eV=2, bin_width_eV=1
        )
    with pytest.raises(TypeError, match="requires bin_width_eV or bin_width"):
        Acquisition.uniform(exposure_s=1.0, minimum_eV=0, maximum_eV=2)
    with pytest.raises(TypeError, match="Acquisition.uniform.maximum"):
        Acquisition.uniform(exposure_s=1.0, minimum_eV=0, maximum=2.0, bin_width_eV=1)


def test_native_request_quantities() -> None:
    semantic = Acquisition.native(exposure=Q_(2, "ms"), hit_threshold=Q_(1, "keV"))
    numeric = Acquisition.native(exposure_s=0.002, hit_threshold_eV=1000.0)
    assert semantic == numeric == NativeAcquisition(0.002, 1000.0)
    assert asdict(semantic) == asdict(numeric)
    assert semantic.exposure == Q_(0.002, "s")
    assert semantic.hit_threshold == Q_(1000, "eV")
    assert semantic.with_exposure(Q_(1, "s")) == NativeAcquisition(1.0, 1000.0)
    assert semantic.with_hit_threshold(Q_(0, "eV")).hit_threshold_eV == 0.0
    with pytest.raises(TypeError, match="NativeAcquisition.exposure_s and exposure"):
        NativeAcquisition(1.0, exposure=Q_(1, "s"))
    with pytest.raises(ValueError, match="NativeAcquisition.exposure"):
        Acquisition.native(exposure=Q_(-1, "s"))
