"""Precision boundaries and paired equivalence must survive degenerate samples."""

import ast
import math
import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def methods():
    return runpy.run_path(str(ROOT / "checks/equal_precision/paired_equivalence.py"))


def test_zero_discordance_has_positive_exact_bound_not_zero_uncertainty():
    result = methods()["paired_interval"](
        np.zeros(1000), np.zeros(1000), bound=1, margin=0.01, comparisons=32
    )
    assert 0 < result["difference_ci"][1] < 0.01
    assert result["difference_ci"][0] == -result["difference_ci"][1]
    assert result["status"] == "passed"
    assert result["mismatch_probability_upper"] == pytest.approx(
        -math.expm1(math.log(0.05 / 64) / 1000)
    )


def test_rare_counts_cannot_pass_a_tight_margin_on_no_observed_difference():
    result = methods()["paired_interval"](
        np.zeros(25000), np.zeros(25000), bound=30, margin=0.00022, comparisons=32
    )
    assert result["status"] == "unresolved"
    assert result["difference_ci"][1] > 0.00022


def test_paired_interval_uses_observed_covariance_and_rejects_injected_bias():
    reference = np.tile([0.0, 1.0], 5000)
    same = methods()["paired_interval"](reference, reference, bound=1, margin=0.005, comparisons=32)
    biased = methods()["paired_interval"](
        reference, np.clip(reference + 0.1, 0, 1), bound=1, margin=0.005, comparisons=32
    )
    assert same["status"] == "passed"
    assert same["paired_standard_error"] == 0
    assert same["reference_standard_error"] > 0
    assert same["reference_rse"] == pytest.approx(same["reference_standard_error"] / 0.5)
    assert biased["status"] == "failed"


@pytest.mark.parametrize("candidate,bound", [([np.nan, 0], 1), ([2.0, 0], 1)])
def test_bad_paired_scores_are_rejected(candidate, bound):
    with pytest.raises(ValueError):
        methods()["paired_interval"]([0, 0], candidate, bound=bound, margin=0.1, comparisons=1)


def test_exact_nonzero_mismatch_upper_uses_correct_binomial_tail():
    upper = methods()["mismatch_upper"](1, 2, 0.01)
    assert upper == pytest.approx(math.sqrt(0.99))


def test_quotient32_keeps_subtraction_precise_for_a_sub_ulp_cell():
    generate = runpy.run_path(str(ROOT / "checks/equal_precision/precision_source.py"))[
        "precision_source"
    ]
    source = generate(
        (ROOT / "src/pyrite/montecarlo/transport/_jit_radiative_device.py").read_text(
            encoding="utf-8"
        ),
        "rad-quotient32-24",
    )
    tree = ast.parse(source)
    cell = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rad_cell_rate"
    )
    cell.decorator_list = []
    namespace = {"np": np, "xp": np}
    exec(compile(ast.Module(body=[cell], type_ignores=[]), "<quotient-cell>", "exec"), namespace)
    left, right = np.float64(1e6), np.float64(1e6 + 0.001)
    actual = namespace["_rad_cell_rate"](left, right, np.float32(1.25), np.float32(2.5))
    nodes, weights = np.polynomial.legendre.leggauss(16)
    t = (nodes + 1) / 2
    expected = (
        (right - left) / 2 * np.sum(weights * (1.25 + 1.25 * t) / (left + (right - left) * t))
    )
    assert actual.dtype == np.dtype("float32")
    assert actual == pytest.approx(expected, rel=4e-7)
    assignments = [
        n
        for n in ast.walk(cell)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "x"
    ]
    assert isinstance(assignments[0].value, ast.BinOp)
    assert all(
        isinstance(n, ast.Call) for n in [assignments[0].value.left, assignments[0].value.right]
    )


def test_primary_pairing_requires_real_counter_provenance():
    import copy

    functions = methods()
    record = dict(
        seed=62001, histories=100, primary_coupling=functions["coupling_record"](62001, 100)
    )
    functions["validate_coupling"](record, copy.deepcopy(record))
    other = copy.deepcopy(record)
    other["primary_coupling"]["transport_keys_sha256"] = "wrong"
    with pytest.raises(ValueError, match="provenance"):
        functions["validate_coupling"](record, other)
    both = copy.deepcopy(record)
    both["primary_coupling"]["sources"] = {}
    with pytest.raises(ValueError, match="source mismatch"):
        functions["validate_coupling"](both, copy.deepcopy(both))


def test_rare_bernoulli_interval_coverage_smoke():
    rng = np.random.default_rng(510620)
    interval = methods()["paired_interval"]
    covered = 0
    for _ in range(400):
        result = interval(
            np.zeros(500), rng.binomial(1, 0.03, 500), bound=1, margin=0.01, comparisons=1
        )
        low, high = result["difference_ci"]
        covered += low <= 0.03 <= high
    assert covered >= 380  # fixed seed; conservative check, not a coverage proof


@pytest.mark.parametrize("mutation", [None, "pair", "pilot", "score", "incomplete", "controls"])
def test_equivalence_loader_requires_complete_fixed_design_and_pairing(
    tmp_path, monkeypatch, mutation
):
    import hashlib
    import json
    import sys
    import types

    module = types.ModuleType("paired_equivalence")
    module.__dict__.update(methods())
    monkeypatch.setitem(sys.modules, "paired_equivalence", module)
    names = ["si", "w"]
    variants = ["fp64", "rad-quotient32-24"]
    protocol = dict(
        equivalence=dict(
            reference=variants[0],
            candidate=variants[1],
            observables={
                s: {"transmitted": dict(kind="indicator", bound=1, margin=1)} for s in names
            },
            elements=["Si", "W"],
            comparisons=2,
            alpha=0.05,
        ),
        heldout=dict(seeds=[62001], histories={"si": 2, "w": 2}),
        pilot=dict(scenes=names),
        source_sha256={variants[0]: "reference", variants[1]: "candidate"},
        invariants=dict(
            max_energy_ledger_error_keV=1e-7,
            max_photon_debit_error_keV=1e-9,
            max_direction_norm_error=1e-10,
        ),
    )
    frozen = tmp_path / "protocol.json"
    frozen.write_text(json.dumps(protocol))
    protocol["preflight"] = dict(histories=64, seed=60901)
    frozen.write_text(json.dumps(protocol))
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    for v in variants:
        (preflight / f"probe-{v}.json").write_text(
            json.dumps(
                dict(
                    source=dict(variant_sha256=protocol["source_sha256"][v]),
                    reports=[
                        dict(
                            element=e,
                            screening_passed=True,
                            finite=True,
                            nonnegative=True,
                            photon_support=True,
                        )
                        for e in ["Si", "W"]
                    ],
                )
            )
        )
    data = tmp_path / "data"
    data.mkdir()
    (data / "preflight-gate.json").write_text(json.dumps(dict(eligible=variants)))
    for scene in names:
        for variant in variants:
            if mutation == "incomplete" and scene == "w" and variant == variants[1]:
                continue
            p = data / f"{scene}-{variant}.json"
            values = [0.0, 2.0] if mutation == "score" and variant == variants[1] else [0.0, 1.0]
            np.savez_compressed(p.with_suffix(".npz"), transmitted=values)
            coupling = module.coupling_record(62001, 2)
            if mutation == "pair" and variant == variants[1]:
                coupling["ordered_primary_range"] = [1, 3]
            record = dict(
                schema="pyrite.equal-precision.precision-sample.v1",
                study_phase="pilot"
                if mutation == "pilot" and variant == variants[1]
                else "heldout",
                precision_protocol_sha256=digest,
                scene=dict(id=scene),
                precision_variant=variant,
                seed=62001,
                histories=2,
                precision_source=dict(
                    variant_sha256=protocol["source_sha256"][variant],
                    reference_sha256=protocol["source_sha256"][variants[0]],
                ),
                warmup=dict(histories=2, same_scene_tables_numerics=True, device_synchronized=True),
                cxr_execution_audit=dict(enabled=False, calls=[]),
                raw_scores_sha256=hashlib.sha256(p.with_suffix(".npz").read_bytes()).hexdigest(),
                primary_coupling=coupling,
            )
            fixed = dict(
                manifest_sha256="frozen-manifest",
                tables=[],
                max_de_frac=0.05,
                cutoff_eV=1000,
                batch_histories=5000,
                scratch_mib=4096,
                probe_electrons=0,
                core="cuda",
            )
            record.update(fixed)
            if mutation == "controls":
                record["max_de_frac"] = 0.1
            (preflight / f"{scene}-64-60901-fp64.json").write_text(
                json.dumps(
                    dict(
                        scene=dict(id=scene),
                        precision_protocol_sha256=digest,
                        study_phase="preflight",
                        **fixed,
                    )
                )
            )
            p.write_text(json.dumps(record))
            block = dict(
                primaries=2,
                energy_ledger_max_error_keV=0.0,
                photon_debit_max_error_keV=0.0,
                direction_norm_max_error=0.0,
            )
            p.with_suffix(".audit.json").write_text(json.dumps(dict(blocks=[block, block])))
    target = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze",
            str(data),
            "--protocol",
            str(frozen),
            "--preflight",
            str(preflight),
            "--output",
            str(target),
        ],
    )
    main = runpy.run_path(str(ROOT / "checks/equal_precision/analyze_equivalence.py"))["main"]
    if mutation in ["pair", "pilot", "score", "controls"]:
        with pytest.raises(ValueError):
            main()
    else:
        main()
        result = json.loads(target.read_text())
        assert result["qualified_observable_equivalence"] == (mutation is None)
        assert result["complete_fixed_design"] == (mutation is None)
