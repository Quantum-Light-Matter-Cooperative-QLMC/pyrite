"""Opt-in transport-trajectory artifacts: schema, run capture, and VTK export.

Capture must be a read-only side channel: the HDF5 artifact reproduces the
exact ``simulate_trajectories`` mapping a case transported, and a captured run's
spectra are bit-for-bit those of the same seeded run without capture.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from pyrite.materials import CATALOG
from pyrite.montecarlo import runner, simulate_trajectories
from pyrite.montecarlo import trajectories as traj
from pyrite.montecarlo.geometry import tilted_geometry
from pyrite.montecarlo.groove import GrooveSpec
from pyrite.montecarlo.trajectories import (
    TrajectoryArtifactError,
    TrajectoryArtifactExistsError,
    TrajectoryCapture,
    preflight_capture,
    read_trajectory_artifact,
    write_trajectory_artifact,
)
from pyrite.montecarlo.trajectory_export import export_segments_vtp

_SI = dict(element="Si", n_atoms_per_ang3=0.05)


def _case(name="hopg 0.1um tilt 0/az 0", E0_keV=30.0, seed=42):
    E_grid = (100.0, 200.0, 10.0)
    return dict(
        name=name,
        crystal="hopg",
        composition=CATALOG.crystal("hopg").composition,
        hkl_list=[(0, 0, 2)],
        B_ang2=0.8,
        E0_keV=float(E0_keV),
        thickness_ang=1e4,
        E_grid=E_grid,
        E_grid_line=E_grid,
        E_grid_brem=(0.0, 500.0, 50.0),
        theta_obs_rad=np.deg2rad(90.0),
        dtheta_obs_rad=np.deg2rad(2.0),
        tilt_deg=0.0,
        tilt_azim_deg=0.0,
        domega_sr=1e-4,
        beam_uvw=(0, 0, 1),
        mosaic_fwhm_rad=None,
        mosaic_mc_fwhm_rad=None,
        mosaic_mc_nodes=1,
        abs_layers=None,
        layer_radiators=None,
        brem_file=None,
        Ne=10,
        Ne_brem=5,
        seed=seed,
        spec_chunk=None,
        brem_chunk=None,
    )


def _assert_same_tree(actual, expected, path="transport"):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), path
        assert actual.dtype == expected.dtype, path
        assert actual.shape == expected.shape, path
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert list(actual) == list(expected), path
        for key in expected:
            _assert_same_tree(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected), path
        assert len(actual) == len(expected), path
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_same_tree(a, e, f"{path}/{index}")
    else:
        assert type(actual) is type(expected), path
        assert actual == expected, path


def _grooved_transport():
    tilt = np.deg2rad(60.0)
    beam, _ = tilted_geometry(np.deg2rad(40.0), tilt, 0.0)
    return simulate_trajectories(
        10.0,
        50,
        5000.0,
        seed=1,
        beam_dir=beam,
        groove=GrooveSpec(spacing_ang=500.0, depth_ang=400.0, tilt_polar_rad=tilt),
        tilt_polar_rad=tilt,
        **_SI,
    )


# ---- schema round trip ------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"energy_model": "midpoint", "max_dE_frac": 0.05},
        {"energy_model": "midpoint", "straggling": True},
        {"beam_fwhm_mm": 1e-4, "bunch_length_fs": 10.0, "collect_diagnostics": True},
    ],
    ids=["frozen", "midpoint-substeps", "straggling", "beam-diagnostics"],
)
def test_artifact_round_trips_the_complete_transport_result(tmp_path, kwargs):
    segs = simulate_trajectories(
        20.0, 20, 2000.0, seed=3, transport_core="lockstep", **_SI, **kwargs
    )
    case = _case()

    path = write_trajectory_artifact(
        tmp_path / "a.h5", segs, case=case, provenance={"stem": "hopg", "parameter_sha256": "p"}
    )
    artifact = read_trajectory_artifact(path)

    _assert_same_tree(artifact.transport, segs)
    assert artifact.case["name"] == case["name"]
    assert artifact.provenance == {"stem": "hopg", "parameter_sha256": "p"}
    assert artifact.attrs["parameter_sha256"] == "p"
    assert artifact.attrs["case_sha256"] == traj.case_digest(case)
    assert artifact.attrs["segment_count"] == segs["L_ang"].size
    assert artifact.attrs["fields"] == list(segs)
    assert artifact.units["r_mid"] == "angstrom"
    assert artifact.units["E_start_keV"] == "keV"
    assert artifact.units["electron_id"] == "1"
    assert artifact.units["n_backscattered"] == "1"


def test_grooved_vacuum_legs_round_trip(tmp_path):
    segs = _grooved_transport()
    assert segs["vacuum_start_ang"].shape[0] > 0

    artifact = read_trajectory_artifact(
        write_trajectory_artifact(tmp_path / "g.h5", segs, case=_case())
    )

    _assert_same_tree(artifact.transport, segs)


def test_shell_secondary_result_round_trips(tmp_path):
    from pyrite.materials import CATALOG
    from pyrite.montecarlo import shell_configuration
    from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

    if not shell_configuration._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    segs = simulate_trajectories(
        20.0,
        20,
        2.0e4,
        composition=CATALOG.crystal("silicon").composition,
        E_cut_keV=2.0,
        seed=11,
        energy_model="midpoint",
        stopping_tables=[resolve_catalog_table("silicon").arrays()],
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        inelastic_materials=["silicon"],
        secondary_threshold_eV=1000.0,
        transport_core="per-electron",
    )
    assert {"track_id", "secondary_tracks", "secondaries", "inelastic"} <= set(segs)

    artifact = read_trajectory_artifact(
        write_trajectory_artifact(tmp_path / "s.h5", segs, case=_case())
    )

    _assert_same_tree(artifact.transport, segs)


def test_coupled_radiative_result_round_trips(tmp_path):
    from dataclasses import replace

    from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
    from tests.helpers.bremslib import synthetic_bremslib_arrays

    table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6)
    table = replace(
        table,
        scaled_sdcs_mb=table.scaled_sdcs_mb * 1e5,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * 1e5,
    )
    segs = simulate_trajectories(
        E0_keV=60.0,
        Ne=20,
        thickness_ang=4_000.0,
        composition=[("C", 0.1)],
        E_cut_keV=10.0,
        seed=42,
        energy_model="midpoint",
        transport_core="lockstep",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables={"C": table},
    )
    assert "hard_radiative_direction" in segs

    artifact = read_trajectory_artifact(
        write_trajectory_artifact(tmp_path / "r.h5", segs, case=_case())
    )

    _assert_same_tree(artifact.transport, segs)


def test_aliases_are_hard_links_not_copies(tmp_path):
    segs = simulate_trajectories(20.0, 5, 2000.0, seed=0, **_SI)
    path = write_trajectory_artifact(tmp_path / "a.h5", segs, case=_case())

    with h5py.File(path) as handle:
        transport = handle["transport"]
        assert transport["E_keV"] == transport["E_start_keV"]
        assert transport["t_ang"] == transport["t_start_ang"]
        assert transport["elec_id"] == transport["electron_id"]


def test_nested_optional_fields_and_empty_arrays_round_trip(tmp_path):
    # Shape of the shell/secondary/radiative metadata without their tables.
    transport = {
        "electron_id": np.empty(0, dtype=np.int64),
        "L_ang": np.empty(0),
        "flags": np.array([True, False]),
        "inelastic": {"model": "shell-soft-hard", "cutoff_eV": 50.0, "materials": ["si"]},
        "radiative": {"bremslib_tables": (("Si", 14, "k", "d"),)},
        "secondaries": {"counts_per_generation": [{"n": 3}], "threshold_eV": 1e3},
        "secondary_tracks": {"parent_id": np.array([-1, 0], dtype=np.int64)},
        "stopping_tables": ((np.arange(3.0), np.arange(3.0) ** 2),),
        "transport_diagnostics": {"p50": None, "p90": np.float32(0.5)},
        "n_layers": 1,
        "Ne": np.int64(4),
    }

    artifact = read_trajectory_artifact(
        write_trajectory_artifact(tmp_path / "n.h5", transport, case=_case())
    )

    _assert_same_tree(artifact.transport, transport)


def test_unsupported_values_fail_before_publishing(tmp_path):
    with pytest.raises(TypeError, match="unsupported type"):
        write_trajectory_artifact(tmp_path / "x.h5", {"bad": object()}, case=_case())
    assert list(tmp_path.iterdir()) == []


def test_device_arrays_download_in_bounded_blocks(tmp_path, monkeypatch):
    class FakeDevice:
        """Minimal device-array stand-in: slicing stays on device."""

        def __init__(self, host):
            self.host = host
            self.shape, self.dtype, self.size = host.shape, host.dtype, host.size

        def __getitem__(self, index):
            return FakeDevice(self.host[index])

    def _download(value):
        downloads.append(value.host.shape[0])
        return value.host.copy()

    downloads = []
    monkeypatch.setattr(traj, "is_device_array", lambda value: isinstance(value, FakeDevice))
    monkeypatch.setattr(traj, "_to_cpu", _download)
    monkeypatch.setattr(traj, "_DEVICE_BLOCK_BYTES", 10 * 3 * 8)
    host = np.arange(95 * 3, dtype=np.float64).reshape(95, 3)

    path = write_trajectory_artifact(tmp_path / "d.h5", {"r_mid": FakeDevice(host)}, case=_case())

    assert max(downloads) == 10 and sum(downloads) == 95
    np.testing.assert_array_equal(read_trajectory_artifact(path).transport["r_mid"], host)


def test_cuda_resident_run_captures_device_segments(tmp_path):
    """GPU-path check: resident CUDA segments are captured without a host copy."""
    from pyrite import _backend

    cupy = pytest.importorskip("cupy")
    if not _backend.is_device_array(cupy.zeros(1)):
        pytest.skip("selected backend is not CUDA")
    case = _case()
    capture = TrajectoryCapture(root=str(tmp_path))

    transport = runner._transport_case(
        case, transport_core="cuda", keep_segments_on_device=True, trajectory_capture=capture
    )

    artifact = read_trajectory_artifact(capture.path_for(case))
    assert artifact.settings["segments_on_device"] is True
    np.testing.assert_array_equal(
        artifact.transport["r_mid"], _backend._to_cpu(transport["segs"]["r_mid"])
    )


# ---- completion and overwrite ---------------------------------------------


def test_interrupted_write_leaves_nothing_readable(tmp_path, monkeypatch):
    real = traj._write_node
    calls = []

    def _fail_late(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(traj, "_write_node", _fail_late)
    segs = simulate_trajectories(20.0, 5, 2000.0, seed=0, **_SI)

    with pytest.raises(KeyboardInterrupt):
        write_trajectory_artifact(tmp_path / "a.h5", segs, case=_case())

    assert list(tmp_path.iterdir()) == []


def test_incomplete_or_foreign_files_are_rejected(tmp_path):
    incomplete = tmp_path / "i.h5"
    with h5py.File(incomplete, "w") as handle:
        handle.attrs["format"] = traj.FORMAT
        handle.attrs["schema_version"] = traj.SCHEMA_VERSION
        handle.attrs["complete"] = False
    foreign = tmp_path / "f.h5"
    with h5py.File(foreign, "w"):
        pass
    future = tmp_path / "v.h5"
    with h5py.File(future, "w") as handle:
        handle.attrs["format"] = traj.FORMAT
        handle.attrs["schema_version"] = traj.SCHEMA_VERSION + 1
        handle.attrs["complete"] = True

    with pytest.raises(TrajectoryArtifactError, match="incomplete"):
        read_trajectory_artifact(incomplete)
    with pytest.raises(TrajectoryArtifactError, match="not a PyRITE"):
        read_trajectory_artifact(foreign)
    with pytest.raises(TrajectoryArtifactError, match="schema_version"):
        read_trajectory_artifact(future)


def test_existing_artifact_requires_explicit_overwrite(tmp_path):
    first = simulate_trajectories(20.0, 5, 2000.0, seed=0, **_SI)
    second = simulate_trajectories(20.0, 5, 2000.0, seed=1, **_SI)
    path = write_trajectory_artifact(tmp_path / "a.h5", first, case=_case())

    with pytest.raises(TrajectoryArtifactExistsError):
        write_trajectory_artifact(path, second, case=_case())
    write_trajectory_artifact(path, second, case=_case(), overwrite=True)

    _assert_same_tree(read_trajectory_artifact(path).transport, second)


def test_artifact_paths_are_deterministic_and_collision_free():
    a = traj.artifact_relpath({"name": "hopg ne=10/5", "E0_keV": 30.0})
    b = traj.artifact_relpath({"name": "hopg ne=10_5", "E0_keV": 30.0})

    assert a == traj.artifact_relpath({"name": "hopg ne=10/5", "E0_keV": 30})
    assert a.parent != b.parent
    assert a.name == "E0_30keV.h5"
    assert "/" not in a.parent.name


# ---- preflight -------------------------------------------------------------


def _seed(capture, case, provenance=None):
    segs = simulate_trajectories(20.0, 2, 2000.0, seed=0, **_SI)
    return write_trajectory_artifact(
        capture.path_for(case), segs, case=case, provenance=provenance or capture.provenance
    )


def test_preflight_refuses_to_replace_without_overwrite(tmp_path):
    case = _case()
    capture = TrajectoryCapture(root=str(tmp_path))
    _seed(capture, case)

    with pytest.raises(TrajectoryArtifactExistsError, match="--overwrite-trajectories"):
        preflight_capture(capture, [case], [case])

    plan = preflight_capture(TrajectoryCapture(root=str(tmp_path), overwrite=True), [case], [case])
    assert (plan.to_write, plan.replaced) == (1, 1)


def test_preflight_counts_cached_cases_and_removes_stale_partials(tmp_path):
    kept, missing, todo = _case("a"), _case("b"), _case("c")
    capture = TrajectoryCapture(root=str(tmp_path), provenance={"parameter_sha256": "p"})
    _seed(capture, kept)
    partial = capture.path_for(todo)
    partial = partial.with_name(partial.name + traj.PARTIAL_SUFFIX)
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"torn")

    plan = preflight_capture(capture, [kept, missing, todo], [todo])

    assert (plan.to_write, plan.kept, plan.missing_cached, plan.replaced) == (1, 1, 1, 0)
    assert not partial.exists()


@pytest.mark.parametrize("stale", ["case", "parameters"])
def test_preflight_rejects_a_cached_artifact_of_different_physics(tmp_path, stale):
    case = _case()
    capture = TrajectoryCapture(root=str(tmp_path), provenance={"parameter_sha256": "new"})
    if stale == "case":
        _seed(capture, {**case, "seed": case["seed"] + 1} | {"name": case["name"]})
    else:
        _seed(capture, case, provenance={"parameter_sha256": "old"})

    with pytest.raises(TrajectoryArtifactError, match="different physics"):
        preflight_capture(capture, [case], [])


# ---- run pipeline ----------------------------------------------------------


def test_run_case_capture_matches_transport_and_leaves_spectra_unchanged(tmp_path):
    case = _case()
    capture = TrajectoryCapture(root=str(tmp_path))

    plain = runner.run_case(case)
    captured = runner.run_case(case, trajectory_capture=capture)

    for key in ("spec", "spec_characteristic", "brem", "brem_wide"):
        np.testing.assert_array_equal(captured[key], plain[key])
    artifact = read_trajectory_artifact(capture.path_for(case))
    transport = runner._transport_case(case)["segs"]
    _assert_same_tree(artifact.transport, transport)
    assert artifact.settings["Ne_transport"] == max(case["Ne"], case["Ne_brem"])
    assert artifact.settings["E_cut_by_electrons"].shape == (artifact.settings["Ne_transport"],)
    assert artifact.attrs["seed"] == case["seed"]


def test_run_without_capture_never_passes_the_option(monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "_transport_case", lambda *a, **kw: seen.append(kw) or {})
    monkeypatch.setattr(runner, "_spectrum_case", lambda *a, **kw: {})

    runner.run_case(_case())

    assert seen == [
        {"transport_core": "auto", "keep_segments_on_device": False, "trajectory_capture": None}
    ]


def test_cpu_worker_pool_writes_one_artifact_per_case(tmp_path):
    cases = [_case("a"), _case("b", E0_keV=20.0)]
    capture = TrajectoryCapture(root=str(tmp_path))

    runner.run_cases(cases, max_workers=2, progress=False, engine="cpu", trajectory_capture=capture)

    for case in cases:
        artifact = read_trajectory_artifact(capture.path_for(case))
        assert artifact.case["name"] == case["name"]
        assert artifact.attrs["E0_keV"] == case["E0_keV"]


def test_run_sweep_captures_new_cases_only_and_keeps_checkpoints_identical(tmp_path, capsys):
    from pyrite.runs.run import run_sweep

    cases = [_case("a"), _case("b")]
    plain_results, captured_results = {}, {}
    run_sweep(
        cases,
        plain_results,
        checkpoint_path=str(tmp_path / "plain" / "hopg"),
        progress=False,
        max_workers=0,
    )
    capture = TrajectoryCapture(root=str(tmp_path / "traj"))
    run_sweep(
        cases[:1],
        captured_results,
        checkpoint_path=str(tmp_path / "cap" / "hopg"),
        progress=False,
        max_workers=0,
        trajectory_capture=capture,
    )
    # Resume: case "a" is cached and must not be re-transported or re-captured.
    run_sweep(
        cases,
        captured_results,
        checkpoint_path=str(tmp_path / "cap" / "hopg"),
        progress=False,
        max_workers=0,
        trajectory_capture=capture,
    )

    out = capsys.readouterr().out
    assert "1 already captured" in out
    for name in ("a", "b"):
        assert capture.path_for(_case(name)).is_file()
        for key in ("spec", "brem"):
            np.testing.assert_array_equal(
                captured_results[name][30.0][key], plain_results[name][30.0][key]
            )
    assert not any(path.name.startswith("E0_") for path in tmp_path.joinpath("plain").rglob("*"))


# ---- export ----------------------------------------------------------------


def _read_vtp(path):
    """Minimal appended-raw VTK PolyData reader (UInt64 headers, little endian)."""
    data = Path(path).read_bytes()
    head, _, tail = data.partition(b'<AppendedData encoding="raw">')
    raw = tail[tail.index(b"_") + 1 :]
    root = ET.fromstring(head + b"</VTKFile>")
    types = {
        "Float64": "<f8",
        "Float32": "<f4",
        "Int64": "<i8",
        "Int32": "<i4",
        "Int16": "<i2",
        "Int8": "i1",
        "UInt8": "u1",
    }
    arrays = {}
    for node in root.iter("DataArray"):
        offset = int(node.get("offset"))
        size = int(np.frombuffer(raw[offset : offset + 8], "<u8")[0])
        values = np.frombuffer(raw[offset + 8 : offset + 8 + size], types[node.get("type")])
        comps = int(node.get("NumberOfComponents", "1"))
        arrays[node.get("Name")] = values.reshape(-1, comps) if comps > 1 else values
    piece = root.find(".//Piece")
    return piece, arrays


def test_vtp_export_keeps_geometry_and_electron_association(tmp_path, monkeypatch):
    import pyrite.montecarlo.trajectory_export as export

    monkeypatch.setattr(export, "_BLOCK_ROWS", 7)  # force blocks straddling the vacuum rows
    segs = _grooved_transport()
    path = write_trajectory_artifact(tmp_path / "g.h5", segs, case=_case())
    n_seg, n_vac = segs["L_ang"].size, segs["vacuum_start_ang"].shape[0]

    summary = export_segments_vtp(path, tmp_path / "g.vtp")
    piece, arrays = _read_vtp(tmp_path / "g.vtp")

    assert summary["cells"] == int(piece.get("NumberOfLines")) == n_seg + n_vac
    points = arrays["Points"]
    starts, ends = points[0::2], points[1::2]
    np.testing.assert_allclose(0.5 * (starts + ends)[:n_seg], segs["r_mid"])
    np.testing.assert_allclose(np.linalg.norm(ends - starts, axis=1)[:n_seg], segs["L_ang"])
    np.testing.assert_array_equal(starts[n_seg:], segs["vacuum_start_ang"])
    np.testing.assert_array_equal(ends[n_seg:], segs["vacuum_end_ang"])
    np.testing.assert_array_equal(
        arrays["electron_id"], np.concatenate([segs["electron_id"], segs["vacuum_elec_id"]])
    )
    np.testing.assert_array_equal(arrays["is_vacuum"], np.repeat([0, 1], [n_seg, n_vac]))
    np.testing.assert_array_equal(arrays["connectivity"], np.arange(2 * (n_seg + n_vac)))
    np.testing.assert_array_equal(arrays["offsets"], np.arange(2, 2 * (n_seg + n_vac) + 1, 2))
    assert "E_keV" not in arrays and "initial_r_ang" not in arrays


def test_vtp_export_fills_row_only_fields_on_vacuum_legs(tmp_path):
    segs = _grooved_transport()
    n_seg = segs["L_ang"].size
    path = write_trajectory_artifact(tmp_path / "g.h5", segs, case=_case())

    export_segments_vtp(path, tmp_path / "g.vtp")
    _, arrays = _read_vtp(tmp_path / "g.vtp")
    no_vacuum = export_segments_vtp(path, tmp_path / "n.vtp", include_vacuum=False)

    assert (arrays["layer"][n_seg:] == -1).all()
    assert no_vacuum["cells"] == n_seg and "is_vacuum" not in no_vacuum["fields"]


def test_vtp_opens_in_vtk(tmp_path):
    pyvista = pytest.importorskip("pyvista")
    segs = simulate_trajectories(20.0, 20, 2000.0, seed=3, energy_model="midpoint", **_SI)
    path = write_trajectory_artifact(tmp_path / "m.h5", segs, case=_case())
    export_segments_vtp(path, tmp_path / "m.vtp")

    mesh = pyvista.read(tmp_path / "m.vtp")

    assert mesh.n_cells == segs["L_ang"].size
    np.testing.assert_array_equal(mesh.cell_data["electron_id"], segs["electron_id"])
    np.testing.assert_array_equal(mesh.cell_data["event_kind"], segs["event_kind"])


def test_export_cli_writes_beside_artifacts_and_refuses_to_clobber(tmp_path):
    from pyrite.cli.commands.trajectories import command

    segs = simulate_trajectories(20.0, 5, 2000.0, seed=0, **_SI)
    write_trajectory_artifact(tmp_path / "run" / "a" / "E0_20keV.h5", segs, case=_case())
    cli = CliRunner()

    first = cli.invoke(command, [str(tmp_path / "run")])
    again = cli.invoke(command, [str(tmp_path / "run")])
    forced = cli.invoke(command, [str(tmp_path / "run"), "--overwrite"])

    assert first.exit_code == 0, first.output
    assert (tmp_path / "run" / "a" / "E0_20keV.vtp").is_file()
    assert again.exit_code == 1 and "--overwrite" in again.output
    assert forced.exit_code == 0
