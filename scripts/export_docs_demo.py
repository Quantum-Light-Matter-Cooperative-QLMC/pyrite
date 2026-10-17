"""Export the documentation results demo data; run with ``uv run python scripts/export_docs_demo.py``.

Writes ``docs/demo/public/results_demo.json``, the only input of the browser-only
marimo demo ``docs/demo/results_app.py``, plus the README preview image
``docs/_static/demo/results-demo-preview.png``.

Spectra come from the cached Zhai et al. (2025) Fig. 1c reproduction: 1 mm HOPG
bulk at every anchor beam energy and the 29 nm film at the top energy. Populate
that cache first with ``pyrite run --preset zhai`` (``--remote`` runs it on the
box; pull the cache back before exporting). This script never reruns it, and it
refuses a stale or missing cache. Trajectories are a small fresh transport run
of the same two samples at the top beam energy, normal incidence, azimuth 180 deg.
"""

import argparse
import base64
import datetime as dt
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = ROOT / "docs" / "demo" / "public" / "results_demo.json"
PREVIEW_OUT = ROOT / "docs" / "_static" / "demo" / "results-demo-preview.png"
APP = ROOT / "docs" / "demo" / "results_app.py"
INVENTORY = ROOT / "docs" / "validation" / "domain-inventories.md"

SCHEMA = "pyrite-docs-demo/1"
# Ledger claims the demo leans on; their statuses are read from the inventory at
# export time so the page never claims more than the ledger does.
CLAIMS = ("line-energy-dispersion", "closed-form-flux", "enhancement-bulk-film")
SIGNIFICANT = 4
AZIMUTH_DEG = 180.0
TILT_DEG = 0.0
SECONDARY_THRESHOLD_EV = 1000.0
DEPTH_MARGIN = 1.3


def _round(values) -> list:
    """Round to ``SIGNIFICANT`` digits; NaN becomes JSON ``null``."""
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    out = np.zeros_like(array)
    nonzero = finite & (array != 0.0)
    if nonzero.any():
        exponent = np.floor(np.log10(np.abs(array[nonzero])))
        scale = 10.0 ** (SIGNIFICANT - 1 - exponent)
        out[nonzero] = np.round(array[nonzero] * scale) / scale
    return [float(v) if ok else None for v, ok in zip(out.tolist(), finite.tolist(), strict=True)]


def _ledger_status(claim: str) -> str:
    match = re.search(
        rf"^\| \[`{re.escape(claim)}`\].*\| `([a-z-]+)` \|\s*$",
        INVENTORY.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise SystemExit(f"ledger claim {claim!r} not found in {INVENTORY.relative_to(ROOT)}")
    return match.group(1)


def _git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def export_spectra(anchor, model) -> dict:
    """Detector-scale Fig. 1c spectra in photons / (eV s nA)."""
    from pyrite.materials.crystal import crystal_absorption_length_ang
    from pyrite.validation.anchor_figures import line_energy_eV, theory_line_energies
    from pyrite.validation.feranchuk_spence import escape_limited_length_ang

    scale = anchor.domega_sr * anchor.per_nA
    lines = theory_line_energies(anchor)
    bulk = [
        {
            "E0_keV": float(E0),
            "line_eV": float(lines[E0]),
            "source": _round(model[E0]["spec"] * scale),
            "line": _round(model[E0]["spec_det"] * scale),
            "brem": _round(model[E0]["brem_det"] * scale),
        }
        for E0 in anchor.energies_keV
    ]
    film = model["film"]
    E_top = float(film["E0_keV"])
    film_series = {
        "E0_keV": E_top,
        "line_eV": float(lines[E_top]),
        "source": _round(film["spec"] * scale),
        "line": _round(film["spec_det"] * scale),
    }
    # Same quantities as anchor_figures.figure_enhancement.
    E_cxr = line_energy_eV(anchor, E_top)
    L_abs = crystal_absorption_length_ang(anchor.crystal, E_cxr)
    n_z = np.cos(anchor.theta_obs_rad)
    ceiling = escape_limited_length_ang(anchor.thick_bulk_ang, L_abs, n_z) / (
        escape_limited_length_ang(anchor.thick_film_ang, L_abs, n_z)
    )
    E_grid = anchor.E_grid
    return {
        "units": "photons / (eV s nA)",
        "E_eV": {"start": float(E_grid[0]), "step": anchor.de_eV, "n": int(E_grid.size)},
        "detector": {
            "theta_obs_deg": float(anchor.detector.observation_angle_deg),
            "solid_angle_sr": float(anchor.domega_sr),
        },
        "bulk": {
            "label": "1 mm bulk",
            "thickness_nm": anchor.thick_bulk_ang / 10.0,
            "series": bulk,
        },
        "film": {
            "label": f"{anchor.thick_film_ang / 10.0:g} nm film",
            "thickness_nm": anchor.thick_film_ang / 10.0,
            "series": [film_series],
        },
        "enhancement": {
            "E0_keV": E_top,
            "mc_peak_ratio": float(model[E_top]["spec_det"].max() / film["spec_det"].max()),
            "no_transport_ceiling": float(ceiling),
        },
    }


def _polyline(trace: dict) -> dict:
    """Collapse Plotly ``start, end, NaN`` segment triples into joined polylines.

    Consecutive segments of one track share an endpoint, so dropping the
    separator and the repeated start roughly thirds the vertex count. Each
    remaining vertex keeps the energy of the segment that starts there.
    """
    xyz = np.column_stack([np.asarray(trace[k], dtype=float) for k in "xyz"])
    color = np.asarray(trace["line"]["color"], dtype=float)
    # Transport records steps interleaved across electrons; a stable sort on the
    # track id (customdata column 3) restores each track's step order.
    order = np.argsort(np.asarray(trace["customdata"], dtype=float)[0::3, 3], kind="stable")
    starts, ends, energy = xyz[0::3][order], xyz[1::3][order], color[0::3][order]
    points, colors = [], []
    for i in range(len(starts)):
        if i == 0 or not np.allclose(starts[i], ends[i - 1], rtol=0.0, atol=1e-9):
            if i:
                points.append(ends[i - 1])
                colors.append(energy[i - 1])
                points.append(np.full(3, np.nan))
                colors.append(np.nan)
        points.append(starts[i])
        colors.append(energy[i])
    if len(starts):
        points.append(ends[-1])
        colors.append(energy[-1])
    xyz = np.asarray(points).reshape(-1, 3)
    trace = dict(trace, x=_round(xyz[:, 0]), y=_round(xyz[:, 1]), z=_round(xyz[:, 2]))
    trace["line"] = dict(trace["line"], color=_round(colors))
    return trace


def _decode(value):
    """Undo Plotly 6's base64 typed-array encoding inside ``to_plotly_json`` output."""
    if isinstance(value, dict):
        if "bdata" in value and "dtype" in value:
            array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
            if "shape" in value:
                array = array.reshape([int(n) for n in str(value["shape"]).split(",")])
            return array
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_decode(v) for v in value]
    return value


def _compact_trace(trace: dict) -> dict:
    name = str(trace.get("name", ""))
    tracks = trace.get("type") == "scatter3d" and name.endswith("tracks")
    if tracks:
        trace = _polyline(trace)
    trace = {k: v for k, v in trace.items() if k not in ("customdata", "hovertemplate", "uid")}
    if tracks:
        generation = 0 if name == "electron tracks" else int(name.split()[1])
        trace["meta"] = {"generation": generation}
    else:
        trace["hoverinfo"] = "skip"
    return json.loads(json.dumps(trace, default=_jsonable))


def export_trajectory(thickness_ang: float, E0_keV: float, *, ne: int, seed: int) -> dict:
    """One small transport run, exported as the trace app's 3D Plotly figure."""
    from pyrite.api import build_configured_cases
    from pyrite.campaign.config import default_settings, trajectory_sweep
    from pyrite.campaign.inelastic_cases import covered_shell_materials
    from pyrite.plots.plotly.trajectories import (
        trajectory_volume_data,
        trajectory_volume_figure_from_data,
    )

    sweep = trajectory_sweep(
        "hopg",
        energies=(E0_keV,),
        tilts=(TILT_DEG,),
        thickness_ang=thickness_ang,
        azim_deg=AZIMUTH_DEG,
    )
    case = build_configured_cases(sweep, default_settings())[0]
    if "hopg" in covered_shell_materials():
        case = dict(
            case,
            energy_model="midpoint",
            inelastic_model="shell-soft-hard",
            inelastic_cutoff_eV=50.0,
            secondary_threshold_eV=SECONDARY_THRESHOLD_EV,
        )
    data = trajectory_volume_data(case, Ne=ne, seed=seed)
    # A thick target dwarfs the micron-scale cascade; draw only the penetrated
    # depth and mark the cut instead of the real exit face.
    depth = float(np.nanmax(np.asarray(data["end_xyz"])[:, 2]))
    cut = DEPTH_MARGIN * depth < float(data["thick"])
    if cut:
        data = dict(data, thick=DEPTH_MARGIN * depth)
    fig = _decode(trajectory_volume_figure_from_data(case, data).to_plotly_json())
    if cut:
        for trace in fig["data"]:
            if trace.get("name") == "exit face":
                trace["name"] = "view cut (sample continues)"
                trace["line"] = dict(trace.get("line", {}), dash="dot")
    layout = {k: v for k, v in fig["layout"].items() if k not in ("template", "uirevision")}
    return {
        "E0_keV": float(case["E0_keV"]),
        "tilt_deg": float(case["tilt_deg"]),
        "azim_deg": float(case["tilt_azim_deg"]),
        "thickness_nm": thickness_ang / 10.0,
        "depth_cut": cut,
        "n_primaries": int(ne),
        "seed": int(seed),
        "inelastic_model": case.get("inelastic_model", "continuous"),
        "traces": [_compact_trace(trace) for trace in fig["data"]],
        "layout": json.loads(json.dumps(layout, default=_jsonable)),
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _round(value) if value.dtype.kind == "f" else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def model_markers() -> dict[str, str]:
    from pyrite.materials.photon_cross_sections import ATTENUATION_MODEL
    from pyrite.montecarlo.spectrum.brem import BREMSSTRAHLUNG_MODEL
    from pyrite.montecarlo.spectrum.cxr._setup import CXR_ESCAPE_MODEL
    from pyrite.montecarlo.transport.stopping import STOPPING_MODEL

    return {
        "CXR_ESCAPE_MODEL": CXR_ESCAPE_MODEL,
        "BREMSSTRAHLUNG_MODEL": BREMSSTRAHLUNG_MODEL,
        "STOPPING_MODEL": STOPPING_MODEL,
        "ATTENUATION_MODEL": ATTENUATION_MODEL,
    }


def write_preview(data: dict, path: Path) -> None:
    """Render the README still from the demo's own figure builders."""
    spec = importlib.util.spec_from_file_location("results_app", APP)
    assert spec is not None and spec.loader is not None
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    path.parent.mkdir(parents=True, exist_ok=True)
    spectrum = app.spectrum_figure(data, "bulk")
    spectrum.update_layout(width=760, height=440)
    tracks = app.trajectory_figure(data, "bulk")
    tracks.update_layout(width=760, height=440, title=None, showlegend=False)
    import plotly.io as pio

    # Two panels side by side: write each still, then join them.
    left = pio.to_image(spectrum, format="png", scale=1)
    right = pio.to_image(tracks, format="png", scale=1)
    from io import BytesIO

    import matplotlib.image as mpimg

    images = [mpimg.imread(BytesIO(blob), format="png") for blob in (left, right)]
    height = max(image.shape[0] for image in images)
    padded = [
        np.pad(image, ((0, height - image.shape[0]), (0, 0), (0, 0)), constant_values=1.0)
        for image in images
    ]
    mpimg.imsave(path, np.concatenate(padded, axis=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, help="Zhai cache (default: workspace cache)")
    parser.add_argument("--ne", type=int, default=20_000, help="cached spectra electrons")
    parser.add_argument("--ne-brem", type=int, default=200, help="cached brem electrons")
    parser.add_argument("--traj-bulk", type=int, default=40, help="bulk primaries to draw")
    parser.add_argument("--traj-film", type=int, default=200, help="film primaries to draw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DATA_OUT)
    parser.add_argument("--preview", type=Path, default=PREVIEW_OUT)
    parser.add_argument("--no-preview", action="store_true", help="skip the README image")
    args = parser.parse_args(argv)

    import pyrite
    from pyrite.validation.anchor_figures import ZhaiAnchor, ZhaiCacheMiss, cached_model_spectra

    anchor = ZhaiAnchor()
    try:
        model, _, cache_path = cached_model_spectra(
            anchor, ne=args.ne, ne_brem=args.ne_brem, cache_dir=args.cache_dir, cache_only=True
        )
    except ZhaiCacheMiss as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    E_top = float(anchor.energies_keV[-1])
    data = {
        "schema": SCHEMA,
        "provenance": {
            "pyrite_version": pyrite.__version__,
            "git_commit": _git_commit(),
            "exported": dt.date.today().isoformat(),
            "spectra_cache": cache_path.name,
            "spectra_electrons": args.ne,
            "model_markers": model_markers(),
            "validation": [{"id": c, "status": _ledger_status(c)} for c in CLAIMS],
        },
        "spectra": export_spectra(anchor, model),
        "trajectories": {
            "bulk": export_trajectory(
                anchor.thick_bulk_ang, E_top, ne=args.traj_bulk, seed=args.seed
            ),
            "film": export_trajectory(
                anchor.thick_film_ang, E_top, ne=args.traj_film, seed=args.seed
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")
    if not args.no_preview:
        write_preview(data, args.preview)
        print(f"wrote {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
