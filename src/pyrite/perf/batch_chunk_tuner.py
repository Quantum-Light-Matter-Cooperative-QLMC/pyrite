"""Opt-in two-variable Optuna tuner for CUDA electron batches and CXR chunks.

Run a user-provided ``make_scene()`` factory in isolated worker processes.
The unit of comparison is a complete, uncached ``run_case`` (transport plus
spectrum), not Python startup, kernel compilation, or checkpoint I/O.

The tuning API is intentionally separate from the physical OptimizationStudy:
execution batching is not a physical scene parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

_SCHEMA = "pyrite-batch-chunk-tuner-v1"
_BATCH_ENV = "PYRITE_MC_ELECTRON_BATCH_MAX"
_CHUNK_ENV = "PYRITE_MC_CXR_CHUNK"
_OUTPUTS = ("E_grid", "spec", "brem", "E_grid_brem", "brem_wide")


def _choices(text: str) -> tuple[int, ...]:
    """Parse candidate lists without silently truncating invalid numbers."""
    try:
        result = tuple(int(item.strip()) for item in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated positive integers") from exc
    if not result or any(n <= 0 for n in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("candidate sizes must be positive and unique")
    return result


def _study_identity(
    source: bytes,
    *,
    batch_choices: Sequence[int],
    chunk_choices: Sequence[int],
    warmups: int,
    repeats: int,
    rtol: float,
    atol: float,
    baseline: tuple[int, int],
    transport_core: str,
    backend: str,
    precision: str,
    device: str,
    version: str,
) -> str:
    """Reject persistence collisions from changed workload or hardware."""
    payload = dict(
        schema=_SCHEMA,
        source_sha256=hashlib.sha256(source).hexdigest(),
        batch_choices=list(batch_choices),
        chunk_choices=list(chunk_choices),
        warmups=warmups,
        repeats=repeats,
        rtol=rtol,
        atol=atol,
        baseline=list(baseline),
        transport_core=transport_core,
        backend=backend,
        precision=precision,
        device=device,
        pyrite_version=version,
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_scene(path: Path) -> Any:
    from pyrite.campaign.model import Scene

    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_pyrite_tuner_workload", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load scene factory {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "make_scene", None)
    if not callable(factory):
        raise ValueError(f"{path} must define make_scene() -> Scene")
    scene = factory()
    if not isinstance(scene, Scene):
        raise TypeError("make_scene() must return pyrite.Scene")
    if scene.numerics.cxr_chunk is not None:
        raise ValueError("make_scene must leave numerics.cxr_chunk=None so tuning controls it")
    if scene.numerics.execution is not None and scene.numerics.execution.cxr_chunk is not None:
        raise ValueError("make_scene must leave execution.cxr_chunk=None")
    return scene


def _benchmark_worker(
    scene_path: Path,
    *,
    warmups: int,
    repeats: int,
    transport_core: str,
    output_path: Path,
    result_path: Path,
) -> None:
    """One process, one (batch, chunk) pair. Timing excludes warmup/compilation."""
    from pyrite import __version__
    from pyrite._backend import BACKEND, _to_cpu
    from pyrite.campaign.model import Sweep
    from pyrite.montecarlo.runner import run_case
    from pyrite.montecarlo.runner.scheduling import case_runtime_plan

    if BACKEND.name != "cuda":
        raise RuntimeError(f"CUDA required, resolved backend={BACKEND.name!r}")
    scene = _load_scene(scene_path)
    cases = Sweep(base=scene).cases()
    if len(cases) != 1:
        raise ValueError(f"benchmark requires exactly one concrete case, got {len(cases)}")
    (case,) = cases
    if case.get("spec_chunk") not in (None, 0):
        raise ValueError("lowered case has explicit spec_chunk; worker override would be ignored")
    plan = case_runtime_plan(case)
    samples: list[float] = []
    output: Mapping[str, Any] | None = None
    for index in range(warmups + repeats):
        BACKEND.synchronize()
        start = perf_counter()
        output = run_case(case, transport_core=transport_core, keep_segments_on_device=True)
        BACKEND.synchronize()
        elapsed = perf_counter() - start
        if index >= warmups:
            samples.append(elapsed)
    assert output is not None
    arrays = {
        key: np.asarray(_to_cpu(output[key])) for key in _OUTPUTS if output.get(key) is not None
    }
    np.savez_compressed(
        output_path,
        allow_pickle=False,
        **arrays,
        n_segments=np.array(output["n_segments"], dtype=np.int64),
        eta=np.array(output["eta"], dtype=np.float64),
        hit_frac=np.array(output["hit_frac"], dtype=np.float64),
    )
    result_path.write_text(
        json.dumps(
            dict(
                samples_s=samples,
                median_s=statistics.median(samples),
                requested_batch=int(os.environ[_BATCH_ENV]),
                requested_cxr_chunk=int(os.environ[_CHUNK_ENV]),
                admitted_cxr_chunk=plan["spec_chunk"],
                backend=BACKEND.name,
                device=BACKEND.device.name,
                precision=os.environ.get("PYRITE_FP64", "0"),
                pyrite_version=__version__,
            ),
            indent=2,
        )
    )


def _equivalent(candidate: Path, reference: Path, *, rtol: float, atol: float) -> bool:
    """Require same trajectories plus spectra within declared numeric tolerance."""
    with (
        np.load(reference, allow_pickle=False) as ref,
        np.load(candidate, allow_pickle=False) as cur,
    ):
        if set(ref.files) != set(cur.files):
            return False
        for key in ref.files:
            a, b = ref[key], cur[key]
            if a.shape != b.shape:
                return False
            if key in {"n_segments", "E_grid", "E_grid_brem"}:
                if not np.array_equal(a, b):
                    return False
            elif not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False):
                return False
    return True


def _run_subprocess(
    scene: Path,
    *,
    batch: int,
    chunk: int,
    warmups: int,
    repeats: int,
    transport_core: str,
    timeout: float,
    output_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env[_BATCH_ENV] = str(batch)
    env[_CHUNK_ENV] = str(chunk)
    env["PYRITE_MC_BACKEND"] = "cuda"  # no CPU fallback hidden in timings
    env["PYRITE_MC_TRANSPORT_CORE"] = "cuda"  # ignore inherited CPU pins
    command = [
        sys.executable,
        "-m",
        "pyrite.perf.batch_chunk_tuner",
        "--worker",
        "--scene",
        str(scene),
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
        "--transport-core",
        transport_core,
        "--output",
        str(output_path),
        "--result",
        str(result_path),
    ]
    completed = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=timeout, check=False
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2500:]
        raise RuntimeError(f"benchmark subprocess failed ({completed.returncode}): {detail}")
    return json.loads(result_path.read_text())


def tune(
    scene_path: Path,
    *,
    batches: tuple[int, ...],
    chunks: tuple[int, ...],
    trials: int,
    warmups: int = 1,
    repeats: int = 3,
    storage: str = "sqlite:///pyrite-batch-chunk.sqlite3",
    study_name: str = "batch-chunk",
    baseline: tuple[int, int] = (1024, 1024),
    transport_core: str = "cuda",
    timeout: float = 1800.0,
    rtol: float = 1e-4,
    atol: float = 1e-12,
    seed: int = 42,
) -> Any:
    """Jointly tune transport electron batch and line-spectrum segment chunk.

    ``trials`` is the TOTAL requested Optuna trial count across resumes.
    Runtime is the median uncached run_case wall-clock time after JIT warmup,
    excluding subprocess startup, scene lowering and checkpoint I/O.
    """
    if min(trials, warmups, repeats) < 1 or timeout <= 0:
        raise ValueError("trials, warmups, repeats and timeout must be positive")
    if rtol < 0 or atol < 0 or not np.isfinite([rtol, atol]).all():
        raise ValueError("comparison tolerances must be finite and nonnegative")
    if transport_core != "cuda":
        raise ValueError("initial tuner requires CUDA transport_core='cuda'")
    if not all((batches, chunks)) or min((*batches, *chunks, *baseline)) < 1:
        raise ValueError("all candidate and baseline sizes must be positive")
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("install pyrite-mc[optimization] / uv sync --extra optimization") from exc

    scene = Path(scene_path).expanduser().resolve(strict=True)
    # Validate workload once BEFORE allocating a GPU context in the worker.
    if not scene.is_file() or scene.suffix != ".py":
        raise ValueError("--scene must point to a Python file defining make_scene()")
    with tempfile.TemporaryDirectory(prefix="pyrite-tune-") as tmp:
        tmp_path = Path(tmp)
        ref_path, ref_json = tmp_path / "ref.npz", tmp_path / "ref.json"
        reference = _run_subprocess(
            scene,
            batch=baseline[0],
            chunk=baseline[1],
            warmups=warmups,
            repeats=repeats,
            transport_core=transport_core,
            timeout=timeout,
            output_path=ref_path,
            result_path=ref_json,
        )
        digest = _study_identity(
            scene.read_bytes(),
            batch_choices=batches,
            chunk_choices=chunks,
            warmups=warmups,
            repeats=repeats,
            rtol=rtol,
            atol=atol,
            baseline=baseline,
            transport_core=transport_core,
            backend=reference["backend"],
            precision=reference["precision"],
            device=reference["device"],
            version=reference["pyrite_version"],
        )
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )
        if study.user_attrs.get(_SCHEMA) not in (None, digest):
            raise ValueError("stored tuner study differs in workload, hardware or tuning settings")
        if study.user_attrs.get(_SCHEMA) is None:
            if study.trials:
                raise ValueError("existing study has no compatible tuner identity")
            study.set_user_attr(_SCHEMA, digest)
        print(
            f"Reference batch={baseline[0]} chunk={baseline[1]} "
            f"median={reference['median_s']:.4f}s on {reference['device']}",
            flush=True,
        )

        def objective(trial: Any) -> float:
            batch = trial.suggest_categorical("electron_batch", list(batches))
            chunk = trial.suggest_categorical("cxr_chunk", list(chunks))
            out_path = tmp_path / f"trial-{trial.number}.npz"
            json_path = tmp_path / f"trial-{trial.number}.json"
            try:
                measurement = _run_subprocess(
                    scene,
                    batch=batch,
                    chunk=chunk,
                    warmups=warmups,
                    repeats=repeats,
                    transport_core=transport_core,
                    timeout=timeout,
                    output_path=out_path,
                    result_path=json_path,
                )
                if measurement["device"] != reference["device"]:
                    raise ValueError("worker device changed during tuning")
                if measurement["precision"] != reference["precision"]:
                    raise ValueError("worker numerical precision changed")
                if not _equivalent(out_path, ref_path, rtol=rtol, atol=atol):
                    raise ValueError("candidate spectra or transport totals differ from reference")
                trial.set_user_attr("timings_s", measurement["samples_s"])
                trial.set_user_attr("admitted_cxr_chunk", measurement["admitted_cxr_chunk"])
                trial.set_user_attr("reference_median_s", reference["median_s"])
                print(
                    f"trial {trial.number}: batch={batch} chunk={chunk} "
                    f"admitted_chunk={measurement['admitted_cxr_chunk']} "
                    f"median={measurement['median_s']:.4f}s",
                    flush=True,
                )
                return float(measurement["median_s"])
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                trial.set_user_attr("rejection_reason", str(exc))
                print(f"trial {trial.number}: rejected: {exc}", flush=True)
                raise optuna.TrialPruned(str(exc)) from exc

        remaining = max(0, trials - len(study.trials))
        if remaining:
            study.optimize(objective, n_trials=remaining)
        valid = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not valid:
            print("No numerically valid candidate completed; see Optuna trial diagnostics.")
            return study
        best = min(valid, key=lambda t: t.value)
        print(
            f"Best batch={best.params['electron_batch']} "
            f"chunk={best.params['cxr_chunk']} "
            f"median={best.value:.4f}s "
            f"(reference {reference['median_s']:.4f}s)",
            flush=True,
        )
        if best.value >= reference["median_s"]:
            print("No candidate beat the measured reference; retaining the baseline settings.")
            chosen_batch, chosen_chunk = baseline
        else:
            chosen_batch, chosen_chunk = (best.params["electron_batch"], best.params["cxr_chunk"])
        print("Apply to an existing workload with:")
        print(f"  export {_BATCH_ENV}={chosen_batch}")
        print(f"  export {_CHUNK_ENV}={chosen_chunk}")
        return study


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=Path, required=True, help="Python file defining make_scene()"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--batches", type=_choices, default=(1024, 2048, 4096, 8192))
    parser.add_argument("--chunks", type=_choices, default=(256, 512, 1024, 2048, 4096))
    parser.add_argument(
        "--trials", type=int, default=16, help="Total Optuna budget including prior trials"
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--storage", default="sqlite:///pyrite-batch-chunk.sqlite3")
    parser.add_argument("--study-name", default="batch-chunk")
    parser.add_argument("--baseline-batch", type=int, default=1024)
    parser.add_argument("--baseline-chunk", type=int, default=1024)
    parser.add_argument("--transport-core", choices=("cuda",), default="cuda")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-12)
    options = parser.parse_args(argv)
    if options.worker:
        if options.output is None or options.result is None:
            parser.error("internal worker requires --output and --result")
        _benchmark_worker(
            options.scene,
            warmups=options.warmups,
            repeats=options.repeats,
            transport_core=options.transport_core,
            output_path=options.output,
            result_path=options.result,
        )
        return
    tune(
        options.scene,
        batches=options.batches,
        chunks=options.chunks,
        trials=options.trials,
        warmups=options.warmups,
        repeats=options.repeats,
        storage=options.storage,
        study_name=options.study_name,
        baseline=(options.baseline_batch, options.baseline_chunk),
        transport_core=options.transport_core,
        timeout=options.timeout,
        rtol=options.rtol,
        atol=options.atol,
    )


if __name__ == "__main__":
    main()
