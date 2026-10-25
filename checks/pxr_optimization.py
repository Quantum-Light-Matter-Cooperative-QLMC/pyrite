"""Bounded fixed-window PXR/CBS comparison for issue #682.

Run search/validation on the lab worker, through the remote-job workflow.
A preflight evaluates only one four-history case. No numerical defaults change.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

import pyrite as pr
from pyrite.campaign.profiles import case_content_key
from pyrite.execution import LocalExecutor
from pyrite.optimization import (
    Constraint,
    FixedWindowYield,
    Integer,
    Objective,
    OptimizationStudy,
    OptunaOptimizer,
    ParameterSpace,
    TrialState,
)
from pyrite.optimization.identity import study_definition_digest
from pyrite.optimization.study import PyriteEvaluator
from pyrite.results import records

SCORER = FixedWindowYield(820.0, 900.0)
TILTS = (41, 43, 45, 47, 49)
THICKNESSES = (250, 500, 750)
GRID = [
    dict(zip(("target.tilt_deg", "target.thickness_ang"), p, strict=True))
    for p in product(TILTS, THICKNESSES)
]
BUDGET = len(GRID)
MAX_RELATIVE_GRID_CHANGE = 0.02
MAX_RELATIVE_SEM = 0.05


def scene(histories: int = 32, step_eV: float = 2.0) -> pr.Scene:
    """Ideal graphite slab, catalog basal reflections, one central direction.

    HOPG catalog cut (001), mosaic disabled, 30 keV point beam, no coherent
    bunch term. The 90-degree scalar detector supplies a direction, not an
    angular integral. No detector response or acceptance factor enters SCORER.
    ELSEPA transport and continuous stopping are held fixed across methods.
    """
    return pr.Scene(
        pr.Beam(energy_keV=30, transverse=pr.PointSpot(), longitudinal=pr.PointBunch()),
        pr.Slab(
            "hopg",
            thickness_ang=500,
            tilt_deg=45,
            tilt_azim_deg=180,
            footprint=pr.Footprint(5, 5),
            mosaic=False,
        ),
        pr.Detector(observation_angle_deg=90),
        physics=pr.Physics(
            emission="incoherent",
            brem_source=None,
            elastic_model="elsepa",
            inelastic_model="continuous",
            radiative_model="uncoupled",
            bremsstrahlung_model="eedl",
        ),
        numerics=pr.Numerics(
            cxr_trials=histories,
            brem_trials=histories,
            transport_core="per-electron",
            energy_bins=pr.EnergyBins(
                cxr=np.linspace(750, 1000, round(250 / step_eV) + 1),
                brem=np.array([750.0, 1000.0]),
            ),
            cxr_reflections=pr.CxrReflections(n_families=1),
        ),
    )


def study(base: pr.Scene) -> OptimizationStudy:
    """Finite 15-point domain; a 500 Å fabrication cap is a design constraint."""
    return OptimizationStudy(
        base,
        ParameterSpace(
            {
                "target.tilt_deg": Integer(41, 49, step=2),
                "target.thickness_ang": Integer(250, 750, step=250),
            }
        ),
        (Objective(SCORER.name, "maximize", SCORER.units),),
        (Constraint("thickness_ang", "<=", 500, "angstrom"),),
    )


class ComparisonOptimizer(OptunaOptimizer):
    """Use the same durable adapter for TPE and predetermined baselines.

    Baseline suggestions are a pure function of seed and trial number, so
    restart cannot repeat the beginning of a random stream. TPE is reseeded per
    proposal too, using its full completed history. Algorithm choice is included
    in the caller's evaluation_id. This is one coordinator, never distributed
    ask/tell. Grid order is fixed and has exactly BUDGET attempted evaluations.
    """

    def __init__(self, definition, method, seed, storage, name):
        super().__init__(
            definition.parameters,
            definition.objectives,
            constraints=definition.constraints,
            seed=seed,
            storage=storage,
            study_name=name,
        )
        if method not in {"tpe", "random", "grid"}:
            raise ValueError("method must be tpe, random or grid")
        self.method, self.seed = method, seed
        self.current = None

    def ask(self):
        import optuna

        number = len(self.study.trials)
        if self.method == "grid":
            if number >= BUDGET:
                raise ValueError("grid budget exhausted")
            sampler = optuna.samplers.PartialFixedSampler(
                GRID[number], optuna.samplers.RandomSampler(seed=self.seed + number)
            )
        elif self.method == "random":
            sampler = optuna.samplers.RandomSampler(seed=self.seed + number)
        else:
            sampler = optuna.samplers.TPESampler(
                seed=self.seed + number,
                constraints_func=lambda trial: trial.user_attrs.get(
                    "pyrite_constraint_violations", (float("inf"),)
                ),
            )
        self.study.sampler = sampler
        proposal = super().ask()
        self.current = proposal.token
        return proposal


class CountingExecutor:
    """Observe actual submissions while preserving the existing executor seam."""

    def __init__(self, delegate=None):
        self.delegate = delegate if delegate is not None else LocalExecutor()
        self.submitted = 0

    def execute(self, cases, **options):
        self.submitted += len(cases)
        return self.delegate.execute(cases, **options)


class EvidenceEvaluator(PyriteEvaluator):
    """Commit scalar provenance before the optimizer commits its observation."""

    def __init__(self, root, label, seed, optimizer=None, executor=None):
        self.optimizer = optimizer
        self.execution = CountingExecutor(executor)
        self.evidence = {}
        super().__init__(
            self.score_record,
            executor=self.execution,
            base_seed=seed,
            run_options={
                "checkpoint_path": str(root / "checkpoints" / label),
                "max_workers": 0,
            },
        )

    def score_record(self, result):
        (record,) = records(result)
        case = record["case"]
        self.evidence = {
            "case": case.to_dict(),
            "case_key": case_content_key(case),
            "checkpoint_path": self.run_options["checkpoint_path"],
        }
        return {**SCORER(result), "thickness_ang": case["thickness_ang"]}

    def __call__(self, candidate, number):
        start, before = perf_counter(), self.execution.submitted
        self.evidence = {}
        try:
            return super().__call__(candidate, number)
        finally:
            self.evidence.update(
                elapsed_seconds=perf_counter() - start,
                submitted_cases=self.execution.submitted - before,
            )
            if self.optimizer is not None:
                self.optimizer.current.set_user_attr("pxr_evidence", self.evidence)


def search(root: Path, seeds: tuple[int, ...], histories: int = 32) -> list[dict]:
    """Equal attempted budgets, including failed/infeasible candidates."""
    root.mkdir(parents=True, exist_ok=True)
    definition = study(scene(histories))
    (base_case,) = pr.Sweep(base=definition.base).cases()
    resolved_key = case_content_key(base_case)
    reports = []
    for seed, method in product(seeds, ("tpe", "random", "grid")):
        label = f"{method}-{seed}"
        optimizer = ComparisonOptimizer(
            definition, method, seed, f"sqlite:///{root / 'studies.sqlite3'}", label
        )
        evaluator = EvidenceEvaluator(root, label, 10_000 * seed, optimizer)
        identity = f"pxr-demo-v1:{method}:{seed}:{resolved_key}:{SCORER.evaluation_id}"
        result = definition.run(evaluator, optimizer, max_trials=BUDGET, evaluation_id=identity)
        trials = optimizer.study.get_trials()
        reports.append(
            {
                "label": label,
                "evaluation_id": identity,
                "definition_digest": study_definition_digest(
                    definition.base,
                    definition.parameters,
                    definition.objectives,
                    definition.constraints,
                    evaluation_id=identity,
                    base_seed=evaluator.base_seed,
                ),
                "budget": BUDGET,
                "histories_per_trial": histories,
                "failed": sum(t.state is TrialState.FAILED for t in result.trials),
                "infeasible": sum(t.state is TrialState.INFEASIBLE for t in result.trials),
                "best": asdict(result.best_trial) if result.best_trial is not None else None,
                "trials": [
                    asdict(t) | {"evidence": trials[t.number].user_attrs.get("pxr_evidence")}
                    for t in result.trials
                ],
            }
        )
        write_report(root / "search.json", reports)
    return reports


def write_report(path: Path, payload) -> None:
    """Atomic scalar report; full arrays stay in the existing checkpoints/CAS."""
    from pyrite import __version__
    from pyrite._backend import BACKEND, REAL
    from pyrite.api import source_git_state

    sha, dirty = source_git_state()
    stamp_path = Path(__file__).resolve().parents[1] / ".pyrite-sync"
    stamp = {}
    if stamp_path.is_file():
        stamp = dict(line.split(": ", 1) for line in stamp_path.read_text().splitlines())
    report = {
        "schema": 1,
        "pyrite_version": __version__,
        "git_sha": sha,
        "git_dirty": dirty,
        "synced_code": {
            key: stamp[key]
            for key in ("code_digest", "code_revision", "code_dirty", "code_tables_digest")
            if key in stamp
        },
        "python": platform.python_version(),
        "machine": platform.platform(),
        "backend": BACKEND.name,
        "dtype": str(REAL),
        "units": SCORER.units,
        "payload": payload,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate(root: Path, reports: list[dict], replicates: int = 4, histories: int = 512):
    """Independent fixed-seed finalists on paired 1 eV / 0.5 eV node grids.

    SEM is sample standard deviation / sqrt(replicates), not a CI or total
    numerical/model uncertainty. Paired grid differences separate energy-grid
    sensitivity from Monte Carlo noise. A scalar direction has no angular
    quadrature; these results make no detector-acceptance convergence claim.
    """
    if replicates < 2 or histories < 1:
        raise ValueError("validation requires >=2 replicates and positive histories")
    output = []
    for index, report in enumerate(reports):
        best = report["best"]
        if best is None:
            output.append({"label": report["label"], "status": "no feasible finalist"})
            write_report(root / "validation.json", output)
            continue
        evaluated = []
        for replicate, step in product(range(replicates), (1.0, 0.5)):
            # Disjoint from search seeds and between methods/finalists.
            seed = 1_000_000_000 + index * 10_000 + replicate
            definition = study(scene(histories, step))
            candidate = definition.scene(best["parameters"])
            evaluator = EvidenceEvaluator(
                root, f"validate-{report['label']}-{replicate}-{step}", seed
            )
            values = evaluator(candidate, 0)
            evaluated.append(
                {
                    "replicate": replicate,
                    "step_eV": step,
                    **values,
                    "feasible": values["thickness_ang"] <= 500,
                    "evidence": evaluator.evidence,
                }
            )
        fine = np.array([r[SCORER.name] for r in evaluated if r["step_eV"] == 0.5])
        coarse = np.array([r[SCORER.name] for r in evaluated if r["step_eV"] == 1.0])
        mean = float(fine.mean())
        sem = float(fine.std(ddof=1) / np.sqrt(replicates))
        grid_difference = float(np.max(np.abs(fine - coarse)))
        output.append(
            {
                "label": report["label"],
                "parameters": best["parameters"],
                "histories_per_replicate": histories,
                "replicates": replicates,
                "mean": mean,
                "sem": sem,
                "max_paired_grid_difference": grid_difference,
                "precision_pass": mean > 0
                and sem <= MAX_RELATIVE_SEM * mean
                and grid_difference <= MAX_RELATIVE_GRID_CHANGE * mean,
                "relative_sem_limit": MAX_RELATIVE_SEM,
                "relative_grid_change_limit": MAX_RELATIVE_GRID_CHANGE,
                "evaluations": evaluated,
            }
        )
        write_report(root / "validation.json", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "search", "validate"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "preflight":
        evaluator = EvidenceEvaluator(args.output, "preflight", 21)
        values = evaluator(scene(4), 0)
        if values[SCORER.name] <= 0:
            raise RuntimeError("preflight requires a nonzero usable PXR/CBS signal")
        write_report(args.output / "preflight.json", {**values, "evidence": evaluator.evidence})
    elif args.phase == "search":
        search(args.output, (21, 22, 23))
    else:
        reports = json.loads((args.output / "search.json").read_text())["payload"]
        validate(args.output, reports)


if __name__ == "__main__":
    main()
