"""Detector-cone splitting as a case option (#379): policy, guardrails, weighted monitor.

Validation: detector-cone-variance-reduction
"""

import math
import os
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


def test_pool_workers_run_split_cases_on_the_per_electron_core(monkeypatch):
    """The pool's automatic lockstep pin (``_worker_init``) yields to a split case."""
    from pyrite.montecarlo.runner.scheduling import _pool_worker

    case = {**_split(), "Ne": 24, "Ne_brem": 24}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        direct = runner._transport_case(case)["segs"]
        assert direct["cone_split"]["weight"].size > 0
        monkeypatch.setattr(runner, "_WORKER_INHERITED_CORE", "")
        monkeypatch.setenv("PYRITE_MC_TRANSPORT_CORE", "lockstep")
        pooled = _pool_worker(case, transport=True)(case)["segs"]
    assert os.environ["PYRITE_MC_TRANSPORT_CORE"] == "lockstep"
    np.testing.assert_array_equal(pooled["L_ang"], direct["L_ang"])
    np.testing.assert_array_equal(pooled["cone_split"]["weight"], direct["cone_split"]["weight"])


def test_brem_repair_replays_the_split_copies():
    case = {**_split(), "Ne": 24, "Ne_brem": 24}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        live = runner.run_case(case)
        repaired = runner._brem_for_case(case, live["E_grid_brem"])
    assert live["cone_split"]["copies"] > 0
    np.testing.assert_allclose(repaired, live["brem_wide"], rtol=1e-12)


def test_selection_continuum_folds_the_split_copies():
    """The selector's continuum share nets against copy-folded line yields (#379)."""
    from pyrite.montecarlo.runner.lattice_share import SelectionContinuum

    case = {**_split(), "Ne": 24, "Ne_brem": 24}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        segs = runner._transport_case(case)["segs"]
    assert segs["cone_split"]["weight"].size > 0
    primaries = {k: v for k, v in segs.items() if k != "cone_split"}
    n_hat = np.array([0.0, 0.0, -1.0])

    def background(segments):
        return SelectionContinuum(case, segments, n_hat, None, (1e3, 2e4), Ne=24).background(None)

    split, analog = background(segs), background(primaries)
    assert split > analog > 0.0


def test_case_stores_one_canonical_split_record():
    """A partial record normalizes to its full policy, model included (#379)."""
    from pyrite.campaign.profiles import case_content_key
    from pyrite.montecarlo.case import Case

    partial = {"cone_factor": 1.0, "min_energy_fraction": 0.0, "target_weight": 0.05}
    sparse = Case(**{**_base(), "cone_split": partial})
    full = Case(**{**_base(), "cone_split": POLICY.to_dict()})
    assert sparse["cone_split"] == POLICY.to_dict()
    assert sparse["cone_split"]["model"] == CONE_SPLIT_MODEL
    assert case_content_key(sparse) == case_content_key(full)


def test_policy_accepts_numpy_scalars_as_python_numbers():
    policy = ConeSplitting(
        cone_factor=np.float32(4.0),
        target_weight=np.float64(1e-3),
        max_copies_per_electron=np.int64(8),
    )
    assert type(policy.cone_factor) is float and type(policy.max_copies_per_electron) is int
    assert policy == ConeSplitting(cone_factor=4.0, max_copies_per_electron=8)


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


def _spectrum_yields(case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        out = runner.run_case(case)
    yields = {
        "line": np.trapezoid(out["spec"], out["E_grid"]),
        "brem": np.trapezoid(out["brem_wide"], out.get("E_grid_brem", out["E_grid"])),
    }
    if out.get("spec_characteristic") is not None:
        yields["characteristic"] = np.trapezoid(out["spec_characteristic"], out["E_grid"])
    return yields, out.get("cone_split")


@pytest.mark.slow
def test_split_spectra_agree_with_analog_through_the_production_run():
    """Anchor: seeded public-option spectra match analog within seed-level error.

    Each seed is one full ``run_case`` of the same case with and without the
    ``cone_split`` key; copies are active on every split seed. Seeds are fixed,
    so the |z| bound is a deterministic regression pin, not a flaky test.
    """
    seeds, n = range(6), 32
    analog, split, copies = [], [], 0
    for seed in seeds:
        base = {**_base(), "Ne": n, "Ne_brem": n, "seed": seed}
        analog.append(_spectrum_yields(base)[0])
        values, record = _spectrum_yields({**base, "cone_split": POLICY.to_dict()})
        split.append(values)
        assert record is not None and record["copies"] > 0
        copies += record["copies"]
    assert copies > len(seeds)
    for name in analog[0]:
        a = np.array([row[name] for row in analog])
        b = np.array([row[name] for row in split])
        se = math.hypot(a.std(ddof=1), b.std(ddof=1)) / math.sqrt(len(seeds))
        assert a.mean() > 0.0 and se > 0.0
        assert abs(b.mean() - a.mean()) / se < 3.0, name


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
    admissible = {**_base(), "E0_keV": 5000.0}  # the default 8/gamma0 cone fits
    assert bool(_split_hint(admissible, statistics)) is hinted
    assert _split_hint(_split(), statistics) == ""
    # Never named where the default option would refuse the case.
    assert _split_hint({**admissible, "elastic_model": "mott"}, statistics) == ""
    assert _split_hint(_base(), statistics) == ""  # 30 keV: half-angle past pi/2


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


def test_grid_populations_carry_copy_lines_at_their_weight():
    from pyrite.montecarlo.runner.cone_split_strata import copy_strata
    from pyrite.montecarlo.runner.cxr_grid import _direction_populations

    n = 24
    case = {**_split(), "Ne": n, "Ne_brem": n}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        segs = runner._transport_case(case)["segs"]
    E_grid, _, _, n_hat, groove = runner._case_geometry(case)
    band = (float(E_grid[0]), float(E_grid[-1]))
    args = (case, n_hat, n, None, groove, *band, None)

    def mass(populations):
        return sum(float(np.sum(p.weight * p.width_eV)) for p in populations)

    plain = {k: v for k, v in segs.items() if k != "cone_split"}
    full = mass(_direction_populations(args[0], segs, *args[1:]))
    primaries = mass(_direction_populations(args[0], plain, *args[1:]))
    expected = 0.0
    for scale, rows, n_copies in copy_strata(segs, n):
        alone = mass(_direction_populations(args[0], rows, *args[1:]))
        expected += scale * n / n_copies * alone
    assert expected > 0.0
    np.testing.assert_allclose(full, primaries + expected, rtol=1e-10)


@pytest.mark.parametrize("blocks", [None, 3])
def test_continuum_folds_copies_exactly_once(monkeypatch, blocks):
    """Copies add once: not again at the native re-entry, nor per electron block."""
    from pyrite.montecarlo.runner import emission
    from pyrite.montecarlo.runner.electron_blocks import iter_electron_blocks

    if blocks is not None:  # the GPU headroom/OOM branch of score_in_electron_blocks

        def blocked(score, segments):
            return sum(score(block) for block in iter_electron_blocks(segments, blocks))

        monkeypatch.setattr(emission, "score_in_electron_blocks", blocked)
    from pyrite.montecarlo.runner.emission_adapters import EmissionBatch, select_continuum

    n = 24
    case = {**_split(), "Ne": n, "Ne_brem": n}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        transport = runner._transport_case(case)
    segs, record = transport["segs"], transport["segs"]["cone_split"]
    continuum = select_continuum(case)
    E_brem = np.asarray(transport["E_brem"], dtype=float)

    def brem(segments, limit):
        batch = EmissionBatch(
            runner._segments_on_device(segments), E_brem, case, transport["n_hat"],
            case.get("abs_layers"), transport["groove"], electron_limit=limit,
        )  # fmt: skip
        return np.asarray(continuum.evaluate(batch, "brem_wide").values, dtype=float)

    added = brem(segs, n) - brem({k: v for k, v in segs.items() if k != "cone_split"}, n)
    # Independent: every copy scored on its own population, weighted, over N.
    copies = record["copies"]
    expected = np.zeros_like(added)
    ids = np.asarray(copies["elec_id"])
    for c, w in enumerate(record["weight"]):
        alone = {**copies, "elec_id": np.where(ids == c, 0, 1)}
        expected += w * brem(alone, 1) / n
    np.testing.assert_allclose(added, expected, rtol=1e-10, atol=1e-30)


def test_retained_block_join_keeps_every_split_record():
    """Chunks of 8 join to the one-chunk record and spectrum (#361 with splitting)."""
    case = _split()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        one, blocks = (
            transport_case_adaptive(case, _settings(48, chunk), transport_core="auto")
            for chunk in (48, 8)
        )
        spectra = [runner._spectrum_case(case, out) for out in (one, blocks)]
    a, b = one["segs"]["cone_split"], blocks["segs"]["cone_split"]
    for key in ("parent", "weight", "flight", "killed"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["copies"]["Ne"] == b["copies"]["Ne"] == a["weight"].size > 0
    for key in ("brem_wide", "spec"):
        np.testing.assert_allclose(spectra[1][key], spectra[0][key], rtol=1e-12)


def test_split_provenance_reaches_spectrum_output_and_numerics():
    from pyrite.config import Numerics

    n = 24
    case = {**_split(), "Ne": n, "Ne_brem": n}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        transport = runner._transport_case(case)
        out = runner._spectrum_case(case, transport)
    record = out["cone_split"]
    weights = transport["segs"]["cone_split"]["weight"]
    assert record["model"] == CONE_SPLIT_MODEL and record["primaries"] == n
    assert (
        record["copies"] == weights.size == sum(s["copies"] for s in record["copy_weight_strata"])
    )
    assert record["half_angle_rad"] == pytest.approx(POLICY.half_angle_rad(case["E0_keV"]))
    assert record["primaries_killed"] == int(transport["segs"]["cone_split"]["killed"].sum())
    described = Numerics(cone_split=POLICY).describe_numerics()["transport"]["effective"]
    assert described["cone_split"] == POLICY.to_dict()
    assert "cone_split" not in Numerics().describe_numerics()["transport"]["effective"]


def test_streamed_split_spectrum_matches_the_retained_path():
    """Blocks of 8 stream the same weighted spectra and provenance as one join."""
    from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed

    n = 24
    case = {**_split(), "Ne": n, "Ne_brem": n}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        retained = runner._spectrum_case(case, runner._transport_case(case))
        streamed = run_case_streamed(case, 8, transport_core="per-electron")
    # Each block's strata carry their own W n_c / N factor: summation order only.
    for key in ("spec", "brem_wide", "spec_characteristic"):
        atol = 1e-10 * float(np.max(np.abs(retained[key])))
        np.testing.assert_allclose(streamed[key], retained[key], rtol=1e-12, atol=atol, err_msg=key)
    assert streamed["cone_split"] == retained["cone_split"]


def test_independent_stopping_with_split_matches_fixed_pair_and_batch_means():
    """#366 brem-prefix stopping on a split case: realized pair equals the fixed run.

    Copies of primaries past the brem freeze drop from brem with their parents;
    every batch-means batch folds only its own primaries' copies.
    """
    settings = Precision(
        target_rse=1.0,
        min_histories=48,
        max_histories=64,
        observables=("cxr", "brem"),
        brem_target_rse=1.0,
        brem_min_histories=24,
        batch_means_band_eV=(1000.0, 10000.0),
        max_history_share=1.0,
        min_effective_histories=0.0,
        max_variance_of_variance=1.0,
        stability_checks=0,
    )
    case = {**_split(), "E_cut_lines_keV": 29.0, "E_cut_brem_keV": 1.0}
    requested = {
        **case,
        "Ne": settings.max_histories,
        "Ne_brem": settings.max_histories,
        "adaptive_precision": settings.to_dict(),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        transport = runner._transport_case(requested)
        output = runner._spectrum_case(requested, transport)
        record = output["adaptive_sampling"]
        n, nb = record["realized_electrons"], record["realized_electrons_brem"]
        assert nb < n
        fixed = {**case, "Ne": n, "Ne_brem": nb}
        reference_transport = runner._transport_case(fixed)
        reference = runner._spectrum_case(fixed, reference_transport)
    split = transport["segs"]["cone_split"]
    parents = split["electron_start"] + split["parent"]
    # Copies on both sides of the brem prefix exercise the drop.
    assert np.any(parents < nb) and np.any(parents >= nb)
    np.testing.assert_array_equal(
        split["parent"], reference_transport["segs"]["cone_split"]["parent"]
    )
    np.testing.assert_array_equal(
        split["weight"], reference_transport["segs"]["cone_split"]["weight"]
    )
    for key in ("spec", "spec_characteristic", "brem", "brem_wide", "E_grid"):
        np.testing.assert_array_equal(output[key], reference[key], err_msg=key)
    batches = record["batch_means"]
    for key, axis in (("spec", "E_grid"), ("brem_wide", "E_grid_brem")):
        energies = np.asarray(output[axis])
        mask = (energies >= 1000) & (energies <= 10000)
        expected = output[key][mask]
        # Strata rescaling per batch reassociates the copy sums.
        tolerance = 64 * np.finfo(expected.dtype).eps
        np.testing.assert_allclose(
            batches[key]["mean"], expected, rtol=tolerance, atol=tolerance * np.abs(expected).max()
        )


@pytest.mark.parametrize("route", ["checkpoint", "shard"])
def test_split_record_survives_checkpoint_and_resume_shards(tmp_path, route):
    """A resumed sweep reloads split spectra and copy provenance unchanged."""
    from pyrite.checkpoints import _checkpoint_store as store
    from pyrite.results.store import store_result

    n = 24
    case = {**_split(), "Ne": n, "Ne_brem": n}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConeSplitLowGainWarning)
        out = runner._spectrum_case(case, runner._transport_case(case))
    results = {}
    store_result(results, case, out)
    name, E0 = case["name"], case["E0_keV"]
    if route == "checkpoint":
        store.save("split", tmp_path, results)
        loaded = store.load("split", tmp_path)
    else:
        store.save_part("split", tmp_path, name, results[name])
        loaded = store.load_parts("split", tmp_path)
    record, saved = loaded[name][E0], results[name][E0]
    assert record["cone_split"] == saved["cone_split"]
    assert record["cone_split"]["copies"] > 0
    assert record["case"]["cone_split"] == POLICY.to_dict()
    for key in ("spec", "brem_wide"):
        np.testing.assert_array_equal(record[key], saved[key], err_msg=key)
