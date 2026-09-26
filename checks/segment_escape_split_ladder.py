"""Split-segment ladder for the segment-averaged photon escape (issue #176).

Each transport segment is re-scored as ``k`` collinear pieces of length
``L/k`` that keep the segment's direction, energies and electron id; piece
start ages advance by ``L/(k beta)``. Emission cross sections are therefore
unchanged and only where along the segment escape is evaluated moves. An
estimator that integrates escape exactly along the segment is invariant in
``k``. One that samples escape at the midpoint drifts with ``k`` (Jensen).

Scored per ``k`` on the same trajectories (paired over seeds):

* ``spec_characteristic`` -- total and per-peak yield (hopg: C K, 277 eV);
* ``brem_wide`` integrated over its grid;
* ``spec`` -- incoherent PXR/CBS line total and per-peak yield. Since issue
  #181 this route also scores the segment-mean escape, so its escape bias
  should be flat in ``k``; before #181 its drift measured the midpoint bias.
  Splitting also shortens each piece's formation time, which broadens every
  line; peak windows can shed tail yield for that reason alone. A transparent
  control (escape paths zeroed) carries the broadening only;
  ``spec_escape_only`` is the absorbed-over-transparent ratio and isolates the
  escape bias.

``ratio`` is the seed mean of ``yield(k) / yield(1)``; ``se`` its standard
error over seeds. ``--models mott,elsepa`` also reports the ELSEPA-minus-Mott
C K difference per ``k``, the spurious -15.7 % of the #89 revalidation.

Validation: segment-escape-average

Run (GPU; use the remote box):
  uv run python checks/segment_escape_split_ladder.py --quick
  uv run python checks/segment_escape_split_ladder.py --output REPORT.json
"""

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from soft_deflection_line_sensitivity import line_metrics, peak_windows  # noqa: E402

from pyrite._backend import REAL, xp  # noqa: E402
from pyrite.campaign.config import material_sweep  # noqa: E402
from pyrite.campaign.sweep import build_cases  # noqa: E402
from pyrite.montecarlo import runner  # noqa: E402
from pyrite.montecarlo.spectrum import brem, characteristic  # noqa: E402
from pyrite.montecarlo.spectrum.lines import _batched as lines_batched  # noqa: E402
from pyrite.montecarlo.spectrum.lines import _kernels  # noqa: E402
from pyrite.montecarlo.spectrum.lines import _per_hkl as lines_per_hkl  # noqa: E402
from pyrite.montecarlo.spectrum.lines import _setup as lines_setup  # noqa: E402

MATERIAL = "hopg"
E0_KEV = 100.0
THICKNESS_ANG = 20000.0
TILT_DEG = 30.0
SPLITS = (1, 8, 32)
FULL = dict(Ne=20000, seeds=(1, 2, 3))
QUICK = dict(Ne=300, seeds=(1, 2))
M_E_KEV = 510.99895


def split_segments(segs, k):
    """Return ``segs`` with every row replaced by ``k`` collinear pieces."""
    if k == 1:
        return segs
    if "E_end_keV" in segs or "t_end_ang" in segs:
        raise NotImplementedError("split ladder covers frozen (left-endpoint) rows only")
    L = xp.asarray(segs["L_ang"])
    n = int(L.size)
    out = dict(segs)
    for key, value in segs.items():
        if key.startswith(("vacuum_", "initial_")) or not hasattr(value, "shape"):
            continue
        if value.shape[:1] == (n,):
            out[key] = xp.repeat(xp.asarray(value), k, axis=0)
    v = xp.asarray(segs["v_hat"])
    E = xp.asarray(segs["E_keV"], dtype=L.dtype)
    gamma = 1.0 + E / M_E_KEV
    beta = xp.sqrt(1.0 - 1.0 / gamma**2)
    j = xp.arange(k, dtype=L.dtype)
    # Piece j spans [j L/k, (j+1) L/k] from the segment start.
    offset = (-0.5 + (j + 0.5) / k)[None, :] * L[:, None]  # (n, k)
    r_mid = xp.asarray(segs["r_mid"])
    out["r_mid"] = (r_mid[:, None, :] + offset[..., None] * v[:, None, :]).reshape(-1, 3)
    out["L_ang"] = xp.repeat(L / k, k)
    age = (j / k)[None, :] * (L / beta)[:, None]
    for key in ("t_ang", "t_start_ang"):
        if key in segs:
            out[key] = (xp.asarray(segs[key], dtype=L.dtype)[:, None] + age).reshape(-1)
    return out


@contextmanager
def split_after_clip(k):
    """Split inside every consumer, after its own cutoff clip.

    Consumers re-clip terminal flights at their own energy floor from each
    row's start energy. Pieces repeat the parent's energy, so splitting before
    the clip would keep pieces the parent's clip drops.
    """
    targets = (characteristic, brem, lines_setup)
    original = _kernels._clip_segments_to_cutoff

    def clip_then_split(segments, E_cut_keV, composition, layers=None):
        return split_segments(original(segments, E_cut_keV, composition, layers), k)

    for module in targets:
        module._clip_segments_to_cutoff = clip_then_split
    try:
        yield
    finally:
        for module in targets:
            module._clip_segments_to_cutoff = original


@contextmanager
def transparent_lines():
    """Zero the PXR/CBS escape paths: the line control without absorption.

    Its drift with ``k`` is the formation-time broadening alone, so the
    absorbed-over-transparent ratio isolates the escape bias. Zeroes both the
    midpoint distance (coherent and flight-grouped reductions) and the
    segment-mean escape pieces (incoherent route, issue #181).
    """
    patches = [
        (lines_batched, "_segment_escape_distance"),
        (lines_batched, "_escape_length"),
        (lines_per_hkl, "_segment_escape_distance"),
        (lines_setup, "segment_escape_pieces"),
    ]
    saved = [getattr(module, name) for module, name in patches]
    real_pieces = lines_setup.segment_escape_pieces

    def zero_distance(segments, n_hat, *, xp):
        return xp.zeros(xp.asarray(segments["L_ang"]).shape, dtype=REAL)

    def zero_length(z_mid, thickness, n_z):
        return xp.zeros_like(xp.asarray(z_mid, dtype=REAL))

    def zero_pieces(*args, **kwargs):
        fraction, start, end = real_pieces(*args, **kwargs)
        return fraction, xp.zeros_like(start), xp.zeros_like(end)

    lines_batched._segment_escape_distance = zero_distance
    lines_batched._escape_length = zero_length
    lines_per_hkl._segment_escape_distance = zero_distance
    lines_setup.segment_escape_pieces = zero_pieces
    try:
        yield
    finally:
        for (module, name), value in zip(patches, saved, strict=True):
            setattr(module, name, value)


def _cases(model, cfg):
    sweep = material_sweep(
        MATERIAL, energy_keV=[E0_KEV], thickness_ang=[THICKNESS_ANG], tilt_deg=[TILT_DEG]
    )
    return build_cases(sweep, cfg["Ne"], cfg["Ne"], elastic_model=model)[0]


def run(cfg, models, splits, line_control=True):
    raw = {}  # (model, k) -> list over seeds of spectrum outputs
    grid = None
    mean_seg = {}
    for model in models:
        case0 = _cases(model, cfg)
        lengths = []
        for seed in cfg["seeds"]:
            case = dict(case0)
            case["seed"] = seed
            t0 = time.perf_counter()
            tp = runner._transport_case(case)
            if grid is None:
                grid = tp["E_grid"]
            tp["E_grid"] = grid
            base = tp["segs"]
            lengths.append(float(np.mean(base["L_ang"])))
            for k in splits:
                t1 = time.perf_counter()
                with split_after_clip(k):
                    out = runner._spectrum_case(case, tp)
                    if line_control:
                        with transparent_lines():
                            clear = runner._spectrum_case(case, tp)
                        out["spec_transparent"] = clear["spec"]
                raw.setdefault((model, k), []).append(out)
                print(
                    f"{model} seed={seed} k={k}: {base['L_ang'].size * k} rows, "
                    f"spectrum {time.perf_counter() - t1:.1f} s "
                    f"(transport+ {time.perf_counter() - t0:.1f} s)",
                    flush=True,
                )
        mean_seg[model] = float(np.mean(lengths))

    E = np.asarray(grid)
    report = {
        "material": MATERIAL,
        "E0_keV": E0_KEV,
        "thickness_ang": THICKNESS_ANG,
        "tilt_deg": TILT_DEG,
        "Ne": cfg["Ne"],
        "seeds": list(cfg["seeds"]),
        "mean_segment_ang": mean_seg,
        "models": {},
    }
    for model in models:
        base_runs = raw[(model, 1)]
        keys = ("spec", "spec_characteristic") + (("spec_transparent",) if line_control else ())
        windows, peak_labels = {}, {}
        for key in ("spec", "spec_characteristic"):
            mean_base = np.mean([np.asarray(o[key]) for o in base_runs], axis=0)
            windows[key] = peak_windows(E, mean_base)
            peak_labels[key] = {
                name: f"{key}.{name}@{m['E_eV']:.0f}eV"
                for name, m in line_metrics(E, mean_base, windows[key]).items()
                if name.startswith("peak")
            }
        if line_control:
            # The control is scored in the absorbed spectrum's windows.
            windows["spec_transparent"] = windows["spec"]
            peak_labels["spec_transparent"] = {
                name: label.replace("spec.", "spec_transparent.", 1)
                for name, label in peak_labels["spec"].items()
            }
        per_k = {}
        for k in splits:
            metrics = {}
            for key in keys:
                for o in raw[(model, k)]:
                    m = line_metrics(E, np.asarray(o[key]), windows[key])
                    metrics.setdefault(f"{key}.total", []).append(m["total"])
                    for name, label in peak_labels[key].items():
                        metrics.setdefault(label, []).append(m[name]["yield"])
            grid_b = np.asarray(raw[(model, k)][0]["E_grid_brem"])
            metrics["brem.total"] = [
                float(np.trapezoid(np.asarray(o["brem_wide"]), grid_b)) for o in raw[(model, k)]
            ]
            per_k[k] = metrics
        if line_control:
            # Escape-only line drift: absorbed over transparent, paired per seed.
            for label in [x for x in per_k[1] if x.startswith("spec.")]:
                clear = label.replace("spec.", "spec_transparent.", 1)
                for k in splits:
                    per_k[k][label.replace("spec.", "spec_escape_only.", 1)] = list(
                        np.asarray(per_k[k][label]) / np.asarray(per_k[k][clear])
                    )
        rows = {}
        for label in per_k[splits[0]]:
            base_v = np.asarray(per_k[1][label], dtype=float)
            row = {}
            for k in splits:
                v = np.asarray(per_k[k][label], dtype=float)
                ratio = v / base_v
                row[str(k)] = {
                    "mean": float(np.mean(v)),
                    "ratio": float(np.mean(ratio)),
                    "se": float(np.std(ratio, ddof=1) / np.sqrt(ratio.size))
                    if ratio.size > 1
                    else float("nan"),
                }
            rows[label] = row
        report["models"][model] = rows

    if "mott" in models and "elsepa" in models:
        cross = {}
        for k in splits:
            m = np.mean(report["models"]["mott"]["spec_characteristic.total"][str(k)]["mean"])
            e = np.mean(report["models"]["elsepa"]["spec_characteristic.total"][str(k)]["mean"])
            cross[str(k)] = float((e - m) / m)
        report["elsepa_minus_mott_characteristic"] = cross
    return report


def _print(report, splits):
    print(
        f"\n{report['material']} E0={report['E0_keV']:g} keV t={report['thickness_ang']:g} A "
        f"tilt={report['tilt_deg']:g} deg, Ne={report['Ne']} x {len(report['seeds'])} seeds"
    )
    print("mean segment length [A]:", report["mean_segment_ang"])
    for model, rows in report["models"].items():
        print(f"\n[{model}]")
        head = "".join(f"{'k=' + str(k):>14}{'ratio':>9}{'se':>8}" for k in splits)
        print(f"{'metric':38}{head}")
        for label, row in rows.items():
            cells = "".join(
                f"{row[str(k)]['mean']:14.5g}{row[str(k)]['ratio']:9.4f}{row[str(k)]['se']:8.4f}"
                for k in splits
            )
            print(f"{label:38}{cells}")
    if "elsepa_minus_mott_characteristic" in report:
        print("\nELSEPA - Mott characteristic total (rel):")
        for k, v in report["elsepa_minus_mott_characteristic"].items():
            print(f"  k={k}: {v:+.2%}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--models", default="mott,elsepa")
    parser.add_argument("--splits", default=",".join(str(k) for k in SPLITS))
    parser.add_argument(
        "--no-line-control", action="store_true", help="skip the transparent PXR/CBS control"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    splits = tuple(int(k) for k in args.splits.split(","))
    if splits[0] != 1:
        parser.error("--splits must start with 1 (the production reference)")
    models = tuple(args.models.split(","))
    report = run(QUICK if args.quick else FULL, models, splits, not args.no_line_control)
    _print(report, splits)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
