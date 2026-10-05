"""Tracks-only viewer cost probe; decision evidence, not a physics validation.

Run each arm/trial in a fresh process, with the same captured HDF5 input::

    uv run --with pyvista --with kaleido python checks/trajectory_viewer_benchmark.py \
        /tmp/capture.h5 --backend pyvista --output /tmp/pv.json

``--replicate`` repeats complete captured showers with distinct track/history
IDs. This diagnoses rendering resource scaling; it is NOT a new transport run.
Reports first and repeated camera renders separately. Interactive browser
latency, trame delivery, remote GPUs and out-of-core loading are not measured.
"""

import argparse
import gc
import importlib.metadata
import json
import platform
import resource
from pathlib import Path
from time import perf_counter

import numpy as np

from pyrite.montecarlo.trajectories import read_trajectory_artifact
from pyrite.plots._frames import C_ANG_PER_FS


def _generate_small(root: Path) -> None:
    """Generate four eight-primary CPU probes; never a production sweep."""
    from pyrite.instrument import FilterPlate, PixelGrid, PlanarDetector, PlanarPose
    from pyrite.instrument.scene import scene_payload
    from pyrite.materials import CATALOG
    from pyrite.montecarlo import simulate_trajectories
    from pyrite.montecarlo.trajectories import write_trajectory_artifact
    from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

    composition = CATALOG.crystal("silicon").composition
    table = resolve_catalog_table("silicon").arrays()
    scene = scene_payload(
        (
            FilterPlate(
                "silicon",
                0.02,
                (5.0, 5.0),
                PlanarPose.from_observation(10.0, 90.0),
                name="Si window",
            ),
        ),
        PlanarDetector(
            PlanarPose.from_observation(30.0, 90.0),
            pixels=PixelGrid(shape=(256, 256), pitch_mm=(0.055, 0.055)),
        ),
    )
    root.mkdir(parents=True, exist_ok=True)
    for energy, thickness in ((20.0, 2e4), (200.0, 2e6)):
        for mode in ("continuous", "shell-soft-hard"):
            kwargs = (
                {}
                if mode == "continuous"
                else dict(
                    stopping_tables=[table],
                    inelastic_model=mode,
                    inelastic_cutoff_eV=50.0,
                    inelastic_materials=["silicon"],
                    secondary_threshold_eV=1000.0,
                )
            )
            transport = simulate_trajectories(
                energy,
                8,
                thickness,
                composition=composition,
                E_cut_keV=2.0,
                seed=11,
                energy_model="midpoint",
                transport_core="per-electron",
                **kwargs,
            )
            case = dict(
                name=f"Si-{energy:g}keV-{mode}",
                crystal="silicon",
                composition=composition,
                E0_keV=energy,
                thickness_ang=thickness,
                seed=11,
                Ne=8,
                tilt_deg=0.0,
                tilt_azim_deg=0.0,
                theta_obs_rad=np.pi / 2,
                beam_fwhm_mm=None,
                crystal_width_mm=None,
                crystal_height_mm=None,
                inelastic_model=mode,
                energy_model="midpoint",
            )
            if mode != "continuous":
                case.update(secondary_threshold_eV=1000.0, inelastic_cutoff_eV=50.0)
            path = write_trajectory_artifact(
                root / f"{energy:g}-{mode}.h5",
                transport,
                case=case,
                scene=scene,
                settings=dict(transport_core="per-electron", E_cut_keV=2.0),
            )
            print(f"{path}: {len(transport['L_ang'])} segments")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, nargs="?")
    parser.add_argument("--generate-small", type=Path, metavar="DIR")
    parser.add_argument("--backend", choices=("plotly", "pyvista"))
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    if args.generate_small:
        if args.artifact or args.backend or args.output or args.replicate != 1 or args.image:
            parser.error("--generate-small cannot be combined with benchmark arguments")
        _generate_small(args.generate_small)
        return
    if args.artifact is None or args.backend is None or args.output is None:
        parser.error("benchmarking requires ARTIFACT, --backend and --output")
    if args.replicate < 1:
        parser.error("--replicate must be positive")
    start = perf_counter()
    artifact = read_trajectory_artifact(args.artifact)
    case = artifact.case
    transport = artifact.transport
    count = len(transport["L_ang"])
    ids = np.asarray(transport.get("track_id", transport["electron_id"]))
    parents = np.asarray(transport.get("parent_id", np.full(count, -1)))
    track_span = int(ids.max(initial=-1)) + 1
    history_span = int(np.asarray(transport["electron_id"]).max(initial=-1)) + 1
    half = 0.5 * transport["L_ang"][:, None] * transport["v_hat"]
    data = {
        "start_xyz": transport["r_mid"] - half,
        "end_xyz": transport["r_mid"] + half,
        "E": transport["E_start_keV"],
        "t_fs": transport["t_start_ang"] / C_ANG_PER_FS,
        "elec_id": transport["electron_id"],
        "track_id": ids,
        "parent_id": parents,
        "generation": np.asarray(transport.get("generation", np.zeros(count, dtype=int))),
    }
    if args.replicate > 1:
        data = {
            key: np.tile(value, (args.replicate, 1) if value.ndim == 2 else args.replicate)
            for key, value in data.items()
        }
        offsets = np.repeat(np.arange(args.replicate), count)
        data["track_id"] += offsets * track_span
        data["elec_id"] += offsets * history_span
        data["parent_id"] += np.where(data["parent_id"] >= 0, offsets * track_span, 0)
    del transport, artifact, half
    gc.collect()
    prepared = perf_counter()
    report = {
        "artifact": str(args.artifact.resolve()),
        "case": case,
        "backend": args.backend,
        "replicate": args.replicate,
        "synthetic_scaling": args.replicate != 1,
        "segments": len(data["E"]),
        "tracks": len(np.unique(data["track_id"])),
        "input_bytes": args.artifact.stat().st_size,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "plotly": importlib.metadata.version("plotly"),
        "prepare_s": prepared - start,
        "render_size": [800, 600],
    }
    generations = np.unique(data["generation"])
    if args.backend == "plotly":
        import plotly.graph_objects as go

        from pyrite.plots.plotly.trajectories import _tracks_trace, visible_trajectory_data

        traces = []
        for generation in generations:
            rows = data["generation"] == generation
            subset = {key: value[rows] for key, value in data.items()}
            traces.append(_tracks_trace(case, subset, "angstrom", generation=int(generation)))
        figure = go.Figure(traces)
        figure.update_layout(width=800, height=600, scene_aspectmode="data", showlegend=False)
        built = perf_counter()
        serialized = figure.to_json()
        report["serialize_s"] = perf_counter() - built
        report["serialized_bytes"] = len(serialized.encode())
        del serialized
        times = []
        for index in range(3):
            figure.update_layout(scene_camera_eye=dict(x=1.25, y=1.25 + index * 0.1, z=1.25))
            tick = perf_counter()
            png = figure.to_image(format="png", width=800, height=600)
            times.append(perf_counter() - tick)
        tick = perf_counter()
        filtered = visible_trajectory_data(
            data, max_generation=0, min_energy_keV=case["E0_keV"] / 2
        )
        report["filter_s"] = perf_counter() - tick
        report["filter_cells"] = len(filtered["E"])
        report["kaleido"] = importlib.metadata.version("kaleido")
        report["renderer"] = "Kaleido Chrome; each image export includes browser lifecycle"
    else:
        import pyvista as pv

        points = np.stack((data["start_xyz"], data["end_xyz"]), axis=1).reshape(-1, 3)
        lines = np.column_stack(
            (np.full(len(data["E"]), 2), np.arange(2 * len(data["E"])).reshape(-1, 2))
        )
        mesh = pv.PolyData(points, lines=lines)
        for field, key in (
            ("energy", "E"),
            ("generation", "generation"),
            ("track_id", "track_id"),
            ("parent_id", "parent_id"),
        ):
            mesh.cell_data[field] = data[key]
        pv.OFF_SCREEN = True
        plotter = pv.Plotter(window_size=(800, 600))
        for generation in generations:
            subset = mesh.extract_cells(mesh.cell_data["generation"] == generation)
            plotter.add_mesh(
                subset,
                scalars="energy",
                cmap="turbo",
                clim=(0.0, case["E0_keV"]),
                line_width=4 if generation == 0 else 2,
                opacity=1 if generation == 0 else 0.65,
            )
        plotter.camera_position = "iso"
        built = perf_counter()
        times = []
        for index in range(3):
            plotter.camera.azimuth = index * 5
            tick = perf_counter()
            image = plotter.screenshot(return_img=True)
            times.append(perf_counter() - tick)
        tick = perf_counter()
        filtered = mesh.extract_cells(
            (mesh.cell_data["generation"] == 0) & (mesh.cell_data["energy"] >= case["E0_keV"] / 2)
        )
        report["filter_s"] = perf_counter() - tick
        report["filter_cells"] = filtered.n_cells
        report["pyvista"] = pv.__version__
        report["vtk"] = importlib.metadata.version("vtk")
        renderer = pv.Report()
        report["renderer"] = str(renderer)
        # Encoding cost is distinct from rendering and from trame network delivery.
        from io import BytesIO

        from PIL import Image

        tick = perf_counter()
        stream = BytesIO()
        Image.fromarray(image).save(stream, format="PNG")
        report["png_encode_s"] = perf_counter() - tick
        png = stream.getvalue()
        plotter.close()
    report["build_s"] = built - prepared
    report["first_render_s"] = times[0]
    report["repeat_camera_render_s"] = times[1:]
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report["png_bytes"] = len(png)
    if args.image:
        args.image.write_bytes(png)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("backend", "segments", "build_s", "first_render_s", "peak_rss_mib")
            }
        )
    )


if __name__ == "__main__":
    main()
