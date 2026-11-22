"""Detector-cone splitting as a case option (#379): policy, guardrails, weighted monitor.

Validation: detector-cone-variance-reduction
"""

import math
import warnings

import numpy as np
import pytest

from pyrite._cone_splitting import (
    CONE_SPLIT_MODEL,
    ConeSplitLowGainWarning,
    ConeSplitting,
    split_transport_core,
)
from pyrite._precision import Precision
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.adaptive import (
    case_measure,
    case_transport_core,
    transport_case_adaptive,
    without_wall_clock,
)
from pyrite.montecarlo.runner.transport_options import split_transport_kwargs
from pyrite.montecarlo.transport.cone_split import ConeSplit, history_sums
from tests.montecarlo.runner.test_adaptive_stopping import _case

# 30 keV: a wide cone (1/gamma0 ~ 0.94 rad) and a high target weight make copies
# frequent in a few dozen histories; the low-gain warning is expected here.
POLICY = ConeSplitting(cone_factor=1.0, min_energy_fraction=0.0, target_weight=0.05)


def _base():
    return {**_case(n_families=2, energy_bins=None), "elastic_model": "elsepa"}


def _split(policy=POLICY):
    return {**_base(), "cone_split": policy.to_dict()}


def _settings(n, chunk):
    return Precision(
        target_rse=0.01,
        min_histories=n,
        max_histories=n,
        block_histories=chunk,
        observables=("cxr", "brem"),
    )


def _monitor(case, n, chunk):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        out = transport_case_adaptive(
            case, _settings(n, chunk), transport_core="auto", block_consumer=lambda block: None
        )
    return out["adaptive_sampling"]


def test_policy_defaults_round_trip_and_build():
    policy = ConeSplitting()
    assert policy.to_dict() == {
        "cone_factor": 8.0,
        "min_energy_fraction": 0.4,
        "angle_scale": 1.0,
        "target_weight": 1e-3,
        "max_copies_per_electron": 1024,
        "model": CONE_SPLIT_MODEL,
    }
    assert ConeSplitting.from_dict(policy.to_dict()) == policy
    cone = ConeSplit((0.0, 0.0, 2.0), **policy.transport_parameters(5000.0))
    gamma = 1.0 + 5000.0 / 510.99895
    assert cone.n_hat == (0.0, 0.0, 1.0)
    assert cone.half_angle_rad == pytest.approx(8.0 / gamma)
    assert cone.min_energy_keV == pytest.approx(2000.0)
    assert cone.angle_scale == 1.0 and cone.target_weight == 1e-3
    disabled = ConeSplitting(angle_scale=None).transport_parameters(5000.0)
    assert math.isinf(disabled["angle_scale"])


@pytest.mark.parametrize(
    "field, value",
    [
        ("cone_factor", 0.0),
        ("target_weight", -1.0),
        ("angle_scale", math.inf),
        ("min_energy_fraction", 1.0),
        ("max_copies_per_electron", 0),
        ("max_copies_per_electron", True),
        ("model", "other"),
    ],
)
def test_policy_validates_fields(field, value):
    with pytest.raises(ValueError):
        ConeSplitting(**{field: value})


def test_policy_rejects_unknown_fields_and_wide_cones():
    with pytest.raises(ValueError, match="unknown cone split"):
        ConeSplitting.from_dict({"cone": 1.0})
    with pytest.raises(ValueError, match="pi/2"):
        ConeSplitting().half_angle_rad(30.0)


@pytest.mark.parametrize(
    "key, value",
    [
        ("coherent_emission", True),
        ("temporal_profile", True),
        ("channeling", {"model": "planar"}),
        ("groove_spacing_ang", 1e4),
        ("gdf_source", {"path": "x"}),
        ("secondary_threshold_eV", 1e4),
        ("pair_production_model", "penelope-2024"),
        ("positron_transport", True),
        ("photon_transport", {"model": "x"}),
        ("transition_radiation", {"model": "two-beam"}),
        ("line_roulette", {"floor": 0.1}),
        ("transport_mode", "mixed"),
    ],
)
def test_case_guardrails_refuse(key, value):
    with pytest.raises(ValueError, match="detector-cone splitting"):
        POLICY.validate_case({**_base(), key: value})


def test_case_guardrails_need_elsepa():
    case = {k: v for k, v in _base().items() if k != "elastic_model"}
    with pytest.raises(ValueError, match="elsepa"):
        POLICY.validate_case(case)


def test_only_the_per_electron_core_has_a_split_port():
    assert split_transport_core("auto") == split_transport_core("per-electron") == "per-electron"
    for core in ("cuda", "lockstep"):
        with pytest.raises(ValueError, match="per-electron CPU"):
            split_transport_core(core)
        with pytest.raises(ValueError, match="per-electron CPU"):
            case_transport_core(_split(), core)
    assert case_transport_core(_split()) == "per-electron"


def test_option_off_adds_no_transport_keyword_and_low_energy_warns():
    assert split_transport_kwargs(_base()) == {}
    with pytest.warns(ConeSplitLowGainWarning, match="near-unit gain"):
        kwargs = split_transport_kwargs(_split())
    assert kwargs["_cone_split"].half_angle_rad == pytest.approx(1.0 / (1.0 + 30.0 / 510.99895))


def test_capture_refuses_split_cases():
    case = {**_split(), "Ne": 4, "Ne_brem": 4}
    with pytest.raises(ValueError, match="trajectory capture"):
        runner._transport_case(case, trajectory_capture=object())


def test_copy_strata_fold_weights_copies_over_primaries():
    """A linear row tally folds to sum_c w_c x_c / N; copies of excluded parents drop."""
    from pyrite.montecarlo.runner.cone_split_strata import copy_strata, fold_copies

    ids = np.array([0, 0, 1, 2, 2, 2, 3])
    length = np.arange(1.0, 8.0)
    weight = np.array([0.25, 0.5, 0.25, 0.125])
    parent = np.array([0, 1, 3, 1])  # block-local parents
    segments = {
        "Ne": 2,
        "cone_split": {
            "copies": {"elec_id": ids, "L_ang": length, "Ne": 4},
            "weight": weight,
            "parent": parent,
            "electron_start": 2,
        },
    }

    def score(rows, n):
        return np.array([rows["L_ang"].sum() / n])

    per_copy = np.bincount(ids, weights=length)
    # Global parents 2, 3, 5, 3: only those below N = 4 are scored.
    keep = 2 + parent < 4
    expected = (weight[keep] * per_copy[keep]).sum() / 4
    np.testing.assert_allclose(fold_copies(segments, 4, score), [expected], rtol=1e-15)
    # scale = W n_c / N with n_c = N = 4; parent 5 (weight 0.25) is dropped.
    assert sorted(scale for scale, _, _ in copy_strata(segments, 4)) == [0.125, 0.25, 0.5]
    assert fold_copies({"Ne": 2}, 2, score) == 0.0


def test_split_spectrum_adds_weighted_copies():
    n = 24
    case = {**_split(), "Ne": n, "Ne_brem": n}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        transport = runner._transport_case(case)
        split = runner._spectrum_case(case, transport)
        segs = {k: v for k, v in transport["segs"].items() if k != "cone_split"}
        primaries = runner._spectrum_case(case, {**transport, "segs": segs})
    assert transport["segs"]["cone_split"]["weight"].size > 0
    for key in ("brem_wide", "spec"):
        assert np.all(np.isfinite(split[key]))
        assert split[key].sum() > primaries[key].sum()


def test_monitor_scores_primary_plus_weighted_copies_per_history():
    n = 24
    case = {**_split(), "Ne": n, "Ne_brem": n}
    settings = _settings(n, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        segs = runner._transport_case(case)["segs"]
    record = segs["cone_split"]
    parent = record["parent"]
    assert parent.size > 0 and record["killed"].any()
    measure = case_measure(case, settings)
    got = measure(0, n, segs)
    plain = {k: v for k, v in segs.items() if k != "cone_split"}
    primary = measure(0, n, plain)
    copies = measure(0, parent.size, record["copies"])
    for name in ("cxr", "brem"):
        expected = history_sums(primary[name], copies[name], parent, record["weight"], n)
        np.testing.assert_array_equal(got[name], expected)
        assert np.any(got[name] != primary[name])


def test_split_monitor_is_chunk_invariant():
    """Copies are keyed by (seed, electron, flight): blocks of 8 equal one block of 48."""
    case = _split()
    one = _monitor(case, 48, 48)
    blocks = _monitor(case, 48, 8)
    assert one["realized_electrons"] == blocks["realized_electrons"] == 48
    # Only the chunk hint inside ``precision`` differs.
    for record in (one, blocks):
        record.pop("precision")
    assert without_wall_clock(one) == without_wall_clock(blocks)
    analog = _monitor(_base(), 48, 48)
    assert analog["statistics"]["cxr"]["mean"] != one["statistics"]["cxr"]["mean"]
    for name in ("cxr", "brem"):
        assert one["statistics"][name]["figure_of_merit"] is not None


def test_split_refuses_observation_directions():
    from pyrite.montecarlo.runner.adaptive import validate_directions

    validate_directions(_split(), None)
    with pytest.raises(ValueError, match="single detector direction"):
        validate_directions(_split(), [(0.0, 0.0, 1.0)])


def test_numerics_lowers_cone_split_into_case_identity():
    from pyrite.campaign.lowering import _with_run_options
    from pyrite.config import Numerics
    from pyrite.montecarlo.case import _ABSENT, Case

    case = Case(**_base())
    assert "cone_split" not in case
    (lowered,) = _with_run_options([case], Numerics(cone_split=POLICY))
    assert lowered["cone_split"] == POLICY.to_dict()
    (unchanged,) = _with_run_options([case], Numerics())
    assert dict(unchanged) == dict(case)
    with pytest.raises(TypeError, match="ConeSplitting"):
        Numerics(cone_split=POLICY.to_dict())
    # Case validation runs the guardrails on replace().
    with pytest.raises(ValueError, match="elsepa"):
        _with_run_options(
            [Case(**{**_base(), "elastic_model": _ABSENT})], Numerics(cone_split=POLICY)
        )


@pytest.mark.parametrize(
    "guard, hinted",
    [("max_electron_share", True), ("effective_sample_size", True), ("stability", False)],
)
def test_statistics_limited_names_split_when_few_histories_dominate(guard, hinted):
    from pyrite.montecarlo.runner.adaptive import _split_hint

    passes = {"max_electron_share": True, "effective_sample_size": True, "stability": True}
    statistics = {"statistics": {"cxr": {"passes": {**passes, guard: False}}}}
    assert bool(_split_hint(_base(), statistics)) is hinted
    assert _split_hint(_split(), statistics) == ""


def test_weight_ladder_is_unbiased_and_lands_on_powers_of_two():
    from pyrite.montecarlo.transport.cone_split import split_stream_keys, weight_ladder

    rng = np.random.default_rng(7)
    n = 200_000
    weights = np.exp(rng.uniform(np.log(1e-4), np.log(0.9), n))
    keys = split_stream_keys(np.arange(n, dtype=np.uint64))
    flights = rng.integers(0, 10_000, n)
    kept, ladder = weight_ladder(weights, keys, flights)
    assert np.all(ladder >= weights) and np.all(ladder < 2.0 * weights)
    np.testing.assert_array_equal(np.log2(ladder), np.round(np.log2(ladder)))
    assert kept.mean() > 0.5
    # E[W 1{kept}] = w, per history and in aggregate.
    score = np.where(kept, ladder, 0.0)
    se = score.std(ddof=1) / np.sqrt(n)
    assert abs(score.mean() - weights.mean()) < 5.0 * se
    # Exact powers of two are kept surely at their own weight.
    exact = np.array([0.25, 2.0**-10, 1.0])
    kept, ladder = weight_ladder(exact, keys[:3], flights[:3])
    assert kept.all()
    np.testing.assert_array_equal(ladder, exact)
    # The draw is keyed: the same (key, flight) gives the same decision.
    again = weight_ladder(weights[:100], keys[:100], flights[:100])
    np.testing.assert_array_equal(
        again[0], weight_ladder(weights[:100], keys[:100], flights[:100])[0]
    )


def test_public_option_copies_carry_power_of_two_weights():
    n = 24
    case = {**_split(), "Ne": n, "Ne_brem": n}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        record = runner._transport_case(case)["segs"]["cone_split"]
    weight = record["weight"]
    assert weight.size > 0
    np.testing.assert_array_equal(np.log2(weight), np.round(np.log2(weight)))


def test_copy_audit_folds_line_mass_onto_parents():
    """Line-yield statistics read m_i + sum_c W m_c per primary; sums add at W."""
    from pyrite.montecarlo.runner.emission_adapters import _fold_copy_audit

    record = {
        "weight": np.array([0.25, 0.25, 0.5]),
        "parent": np.array([1, 0, 1]),
        "electron_start": 2,
    }
    caller = {"n_electrons": 5, "line_mass": 10.0, "electron_mass": np.ones(5)}
    stratum = {"elec_id": np.array([0, 0, 1])}  # rows of copies 0 and 1 (W = 0.25)
    audit = {"line_mass": 4.0, "mass_above": 1.0, "electron_mass": np.array([3.0, 1.0, 0.0])}
    _fold_copy_audit(caller, audit, record, stratum)
    assert caller["line_mass"] == 11.0 and caller["mass_above"] == 0.25
    # Copy 0 -> global parent 3, copy 1 -> global parent 2.
    np.testing.assert_array_equal(caller["electron_mass"], [1.0, 1.0, 1.25, 1.75, 1.0])
