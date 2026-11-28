"""Freeze #1041 numerical comparisons without running histories or qualifying timings.

This is a design manifest, not a submission plan. Runtime identities, refined
tables, replay support and pilot-based confirmation populations remain required.
"""

import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
CONE = "cone_source_photons_ge10kev"
CONTRASTS = ("nominal:L3", "L1:L2", "L2:L3")
FAMILIES = ("step", "cutoff", "table", "detector")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pilot_files(archives, source, inventory):
    """Verify original archives and retain hashes of every v3 planning input pair."""
    wanted = set(source["pilot"])
    if len(wanted) != len(source["pilot"]):
        raise ValueError("duplicate v3 pilot inputs")
    names = wanted | {str(Path(name).with_suffix(".npz")) for name in wanted}
    found = {}
    for archive in inventory["archives"]:
        if archive["filename"] not in {f"precision-{job}.tar.gz" for job in inventory["v2_jobs"]}:
            continue
        path = archives / archive["filename"]
        if digest(path) != archive["sha256"]:
            raise ValueError(f"archive checksum mismatch: {path}")
        with tarfile.open(path, "r|gz") as bundle:
            for member in bundle:
                if member.name not in names:
                    continue
                if not member.isfile() or member.name in found:
                    raise ValueError(f"invalid/duplicate pilot member: {member.name}")
                stream = bundle.extractfile(member)
                assert stream is not None
                with stream:
                    found[member.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if set(found) != names:
        raise ValueError(f"missing v3 planning inputs: {sorted(names - found.keys())}")
    for name in wanted:
        if int(Path(name).stem.rsplit("-", 1)[1]) not in (101, 102, 103):
            raise ValueError("planning inputs must be the prior seeds 101-103")
    return dict(sorted(found.items()))


def controls(engine, algorithm, thickness):
    """Primitive ladders; unsupported independent refinements stay in the family."""
    out = []

    def add(name, family, levels, support="available", reason=None, scope="both"):
        out.append(
            dict(
                id=name,
                family=family,
                levels=levels,
                support=support,
                reason=reason,
                scope=scope,
                nominal=levels[0],
            )
        )

    halves = [1, 2, 4, 8]
    add("electron_cutoff_keV", "cutoff", [10 / n for n in halves])
    add(
        "radiative_separator_eV",
        "cutoff",
        [1000 / n for n in halves],
        "unsupported" if engine == "geant4" else "available",
        "No audited independent radiative separator override" if engine == "geant4" else None,
    )
    if engine == "pyrite":
        add("max_dE_frac", "step", [0.05 / n for n in halves])
        if algorithm == "mixed":
            for name in ("C1", "C2"):
                add(name, "step", [0.05 / n for n in halves])
            add("max_step_um", "step", [thickness / (10 * n) for n in halves])
        add(
            "elsepa_energy_density",
            "table",
            halves,
            "unresolved",
            "Independently generated nested tables and generator evidence not staged",
        )
        for name in ("sbethe_energy_density", "elsepa_angular_inverse_cdf", "bremslib_source_mesh"):
            add(name, "table", halves, "unsupported", "No independent refinement/bound established")
        add(
            "moment_policy",
            "table",
            [
                dict(
                    rtol=2e-4 / n,
                    atols=[1e-10 / n, 1e-14 / n, 1e-10 / n],
                    max_nodes=32768,
                    probe_count=8191,
                )
                for n in halves
            ],
        )
        add(
            "cone_angular_order",
            "detector",
            [4, 8, 16, 32],
            "unresolved",
            "Same-track replay and frozen finest energy rule required",
            "cone",
        )
        add(
            "cone_energy_order",
            "detector",
            [4, 8, 16, 32],
            "unresolved",
            "Independent energy-rule override and same-track replay required",
            "cone",
        )
        add(
            "cone_point_approximation",
            "detector",
            ["point", None, None, "integrated-L3"],
            "unresolved",
            "Direct point versus integrated-L3 same-track contrast required",
            "cone",
        )
    elif engine == "geant4":
        add("step_dRoverRange", "step", [0.2 / n for n in halves])
        add("step_final_um", "step", [1000 / n for n in halves])
        if algorithm != "ss":
            start = 0.08 if algorithm == "gs" else 0.04
            add("msc_range_factor", "step", [start / n for n in halves])
        add("bins_per_decade", "table", [7 * n for n in halves])
        add(
            "cone_geometry_normalization",
            "detector",
            [None] * 4,
            "unresolved",
            "Discrete cone boundary and per-incident normalization audit required",
            "cone",
        )
    else:
        for name in ("C1", "C2"):
            add(name, "step", [0.05 / n for n in halves])
        add("dsmax_cm", "step", [thickness * 1e-4 / (10 * n) for n in halves])
        add(
            "native_table_density",
            "table",
            halves,
            "unsupported",
            "No independent native density control/effective-grid readback",
        )
        add(
            "cone_tally",
            "detector",
            [None] * 4,
            "unsupported",
            "No implemented matched cone tally",
            "cone",
        )
    return out


def arms():
    result = []
    for algorithm in ("ss", "mixed"):
        for backend, precision in (("cpu", "fp64"), ("cuda", "fp64"), ("cuda", "mixed")):
            result.append(
                dict(
                    id=f"pyrite-{algorithm}-{backend}-{precision}",
                    engine="pyrite",
                    algorithm=algorithm,
                    backend=backend,
                    precision=precision,
                    straggling=False,
                )
            )
    for algorithm in ("ss", "wvi", "gs"):
        result.append(
            dict(
                id=f"geant4-{algorithm}-cpu",
                engine="geant4",
                algorithm=algorithm,
                backend="cpu",
                precision="engine-native",
                straggling=False,
            )
        )
    result.append(
        dict(
            id="penred-cpu-straggling",
            engine="penred",
            algorithm="mixed",
            backend="cpu",
            precision="engine-native",
            straggling=True,
        )
    )
    return result


def scales_and_membership(source, spec):
    """Keep the union of frozen headline members; choose one common prior scale."""
    rows = []
    for scene in spec["scenes"]:
        for workload in ("transport", "cone"):
            items = [
                i
                for i in source["items"]
                if i["scene"] == scene["id"] and i["set"] == workload and i["target_rse"] == 0.02
            ]
            if not items:
                raise ValueError("missing frozen headline membership")
            members = set().union(*(i["per_observable_histories"] for i in items))
            if workload == "cone":
                members.add(CONE)  # Rare/over-budget cones never disappear.
            for observable in sorted(members):
                arm = "pyrite-continuous" if workload == "cone" else "pyrite-discrete"
                key = f"{scene['id']}|{arm}|{observable}"
                scale = source["cv"][key]["mean"]
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError(f"nonpositive/nonfinite prior scale: {key}")
                rows.append(
                    dict(
                        scene=scene["id"],
                        workload=workload,
                        observable=observable,
                        scale=scale,
                        source_cv_key=key,
                        prior_n=source["cv"][key]["n"],
                    )
                )
    return rows


def make_manifest(files):
    source_path = HERE / "precision-plan-v3.json"
    source = json.loads(source_path.read_text())
    spec = json.loads((HERE / "scenes.json").read_text())
    expected_files = set(source["pilot"]) | {
        str(Path(p).with_suffix(".npz")) for p in source["pilot"]
    }
    if set(files) != expected_files or any(
        not isinstance(h, str) or len(h) != 64 or set(h) - set("0123456789abcdef")
        for h in files.values()
    ):
        raise ValueError("need all verified pilot JSON/score hashes")
    scales = scales_and_membership(source, spec)
    configurations, expected = [], []
    for scene in spec["scenes"]:
        for arm in arms():
            ladder = controls(arm["engine"], arm["algorithm"], scene["thickness_um"])
            config_id = f"{scene['id']}|{arm['id']}"
            configurations.append(
                dict(
                    id=config_id,
                    scene=scene["id"],
                    arm=arm,
                    controls=ladder,
                    candidate="L0 of this new design; old non-L0 timings unqualified",
                    runtime_identity=None,
                    effective_parameter_audit=None,
                    source_binary_table_hashes=None,
                    submission_ready=False,
                )
            )
            for scale in (s for s in scales if s["scene"] == scene["id"]):
                active = [c for c in ladder if c["scope"] == "both" or scale["workload"] == "cone"]
                counts = {f: sum(c["family"] == f for c in active) for f in FAMILIES}
                total = scale["scale"] * 0.02 / 3
                base = dict(
                    config=config_id,
                    observable=scale["observable"],
                    workload=scale["workload"],
                    total_allowance=total,
                )
                for control in active:
                    # Point approximation is one direct test, not fictitious intermediate levels.
                    contrasts = (
                        ("nominal:L3",)
                        if control["id"] == "cone_point_approximation"
                        else CONTRASTS
                    )
                    for contrast in contrasts:
                        expected.append(
                            base
                            | dict(
                                control=control["id"],
                                contrast=contrast,
                                allowance=total / 4 / counts[control["family"]],
                                support=control["support"],
                                reason=control["reason"],
                            )
                        )
                expected.append(
                    base
                    | dict(
                        control="all-fine",
                        contrast="nominal:all-L3",
                        allowance=total,
                        support="unresolved",
                        reason="Requires every primitive, retained identities and interaction evidence",
                    )
                )
    for index, row in enumerate(expected):
        row["id"] = f"comparison-{index:05d}"
    multiplicity = len(expected)
    return dict(
        schema="pyrite.geant4-performance.convergence-design.v1",
        issue=1041,
        purpose="accuracy-only",
        headline=False,
        numerical_gate="unmeasured",
        submission_ready=False,
        target_rse=0.02,
        covers_target_rse=[0.02, 0.05],
        scene_family="six-scene-W-19.3; not the 44-case-W-19.261 matrix",
        scene_specification={k: spec[k] for k in ("scenes", "materials", "bounds", "cone")},
        fixed_settings=dict(
            cone_transmission=1,
            gs_step_limit="safety-plus",
            gs_skin=3,
            wentzel_theta_split_rad=0.15,
            penred_WCC="beam energy; delta rays suppressed",
            penred_bounds_override="energy_loss_fluctuations=true; separate matched family required",
            angular_rule="n_mu=q, n_phi=2q; hold energy order at L3",
            energy_rule="2*q_E Gauss nodes per edge-split panel; hold angular order at L3",
            point_comparison="same histories, point*solid_angle versus integrated angular/energy L3",
        ),
        input_sha256={
            p.name: digest(p)
            for p in (
                source_path,
                HERE / "scenes.json",
                HERE / "evidence-inventory-v1.json",
                Path(__file__),
            )
        },
        pilot_files_sha256=files,
        scales=scales,
        configurations=configurations,
        expected_rows=expected,
        multiplicity=multiplicity,
        family_alpha=0.05,
        row_alpha=0.05 / multiplicity,
        interval_method="two-sided Bonferroni normal/delta; diagnostic eligibility required",
        normal_critical_value=NormalDist().inv_cdf(1 - 0.05 / (2 * multiplicity)),
        scale_rule="Prior v3 PyRITE discrete transport / continuous cone pooled mean; common to all arms",
        membership_rule="Union of prior v3 headline members across arms; cone always retained",
        seed_policy=dict(
            pilot=[301, 302, 303],
            confirmation=[401, 402, 403],
            pairing="independent unless primary IDs, zero scores and substreams are verified",
            shard_map=None,
            penred_pair_range=[0, 1000],
            submission_blocker="Freeze collision-checked per-shard map before submission",
        ),
        resources=dict(
            pilot_cpu_core_hours=64,
            confirmation_cpu_core_hours=256,
            pilot_gpu_hours=2,
            confirmation_gpu_hours=8,
            max_cpus=8,
            max_concurrent_cpu_shards=4,
            max_shard_minutes=30,
            gpu_allocations_at_once=1,
            pilot_histories_per_seed=[20000, 200000, 2000000],
            confirmation_histories=None,
            forecast=None,
            memory_request=None,
        ),
        eligibility=dict(
            vov_max=0.01,
            weighted_effective_count_min=1000,
            discrete_nonzero_min=1000,
            stochastic_paired_discordances_min=1000,
            continuous_tail_slope_min=3,
            tail_run_histories_min=50000,
            shift_over_se_trigger=0.1,
            variance_safety_factor=1.25,
            repeats=3,
            min_histories_per_repeat=20000,
        ),
        blockers=[
            "No measured convergence or eligible confirmation evidence",
            "Actual per-configuration source/binary/table identities and effective settings not frozen",
            "Unsupported/unresolved controls retained in M and family allocations",
            "PenRed requires a separately frozen straggling-on cross-engine family",
            "CPU/CUDA and fp64/mixed arms need their own parity/numerical evidence",
            "No resource forecasts, memory requests, confirmation N or per-shard seed map",
            "Raw input hashes identify unpublished backup; durable raw publication remains open",
        ],
    )


def confirmation_count(variance, allowance, multiplicity):
    """Pilot forecast per arm (independent variance sum) or per paired population."""
    if not all(math.isfinite(x) and x > 0 for x in (variance, allowance)):
        raise ValueError("positive finite pilot variance and allowance required")
    if type(multiplicity) is not int or multiplicity < 1:
        raise ValueError("positive integer multiplicity required")
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * multiplicity))
    total = math.ceil(1.25 * 4 * z**2 * variance / allowance**2)
    return 3 * max(20000, math.ceil(total / 3))


def write_manifest(path, result):
    """Refuse changes to an existing freeze; compact the explicit row lists."""
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError(
                "refusing to overwrite a different frozen design; use a new version/path"
            )
        return
    fields = []
    for key, value in sorted(result.items()):
        if key in ("configurations", "expected_rows"):
            encoded = (
                "[\n"
                + ",\n".join(
                    "  " + json.dumps(row, allow_nan=False, sort_keys=True) for row in value
                )
                + "\n ]"
            )
        else:
            encoded = json.dumps(value, allow_nan=False, sort_keys=True, indent=1)
        fields.append(" " + json.dumps(key) + ": " + encoded)
    path.write_text("{\n" + ",\n".join(fields) + "\n}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archives",
        type=Path,
        help="Checksum-verified prior planning archive directory; needed to freeze",
    )
    parser.add_argument("--output", type=Path, default=HERE / "convergence-design-v1.json")
    parser.add_argument(
        "--check", action="store_true", help="Check frozen design without archives or runs"
    )
    args = parser.parse_args()
    if args.check:
        frozen = json.loads(args.output.read_text())
        result = make_manifest(frozen["pilot_files_sha256"])
        if result != frozen:
            raise ValueError(
                "frozen design differs; version/amend before confirmation, never shrink M"
            )
    else:
        if args.archives is None:
            parser.error("--archives required to freeze the original planning file hashes")
        source = json.loads((HERE / "precision-plan-v3.json").read_text())
        inventory = json.loads((HERE / "evidence-inventory-v1.json").read_text())
        result = make_manifest(pilot_files(args.archives, source, inventory))
        write_manifest(args.output, result)
    print(
        json.dumps(
            dict(
                multiplicity=result["multiplicity"],
                row_alpha=result["row_alpha"],
                submission_ready=False,
                headline=False,
            )
        )
    )


if __name__ == "__main__":
    main()
