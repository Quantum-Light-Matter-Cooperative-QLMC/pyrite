"""#510 PenRed arm: generated inputs, tally parsing and schema records, without PenRed."""

import json
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from pyrite.perf.comparison import comparison_record
from pyrite.perf.penred import (
    MATERIAL_PHOTON_EABS_EV,
    config_text,
    geometry_text,
    mass_fractions,
    parse_tally,
    primary_scores,
)

ROOT = Path(__file__).resolve().parents[2] / "checks/equal_precision"


@pytest.fixture(scope="module")
def adapter():
    return runpy.run_path(str(ROOT / "penred_adapter.py"))


def config(**overrides):
    kwargs = dict(
        composition=[("Si", 0.02205), ("O", 0.0441)],
        density_g_cm3=2.2,
        energy_keV=800.0,
        thickness_cm=0.05,
        width_cm=1.0,
        angle_deg=75.0,
        histories=11,
        seed_pair=7,
        electron_eabs_eV=1e4,
        wcr_eV=1e3,
    )
    kwargs.update(overrides)
    text = config_text(**kwargs)
    return dict(line.split(" ", 1) for line in text.splitlines())


def test_mass_fractions_follow_stoichiometry():
    (z_si, w_si), (z_o, w_o) = mass_fractions([("Si", 0.02205), ("O", 0.0441)])
    assert (z_si, z_o) == (14, 8)
    assert w_si + w_o == pytest.approx(1.0)
    assert w_si == pytest.approx(28.085 / (28.085 + 2 * 15.999))


def test_config_keeps_photons_unsimulated_and_suppresses_delta_rays():
    keys = config()
    assert keys["simulation/threads"] == "1"
    assert keys["sources/generic/source1/nhist"] == "11"
    # Material photon EABS below the 10 keV score threshold stacks every scored
    # photon; the body EABS above E0 absorbs it before transport.
    assert float(keys["materials/slab/eabs/gamma"]) == MATERIAL_PHOTON_EABS_EV < 1e4
    assert float(keys["geometry/eabs/1/gamma"]) > 800e3
    assert float(keys["materials/slab/WCC"]) == 800e3
    assert float(keys["materials/slab/WCR"]) == 1e3
    assert float(keys["materials/slab/eabs/electron"]) == 1e4
    assert keys["tallies/primary/type"] == '"PYRITE_PRIMARY"'
    assert float(keys["tallies/primary/zmax"]) == 0.05
    # Oblique source aims at the origin from just upstream of the entrance face.
    x = float(keys["sources/generic/source1/spatial/position/x"])
    z = float(keys["sources/generic/source1/spatial/position/z"])
    assert z < 0 and x / z == pytest.approx(np.tan(np.radians(75.0)))
    assert float(keys["sources/generic/source1/direction/theta"]) == 75.0
    with pytest.raises(ValueError):
        config(electron_eabs_eV=800e3)


def test_geometry_bounds_the_slab_by_six_planes():
    text = geometry_text(0.01, 1.0)
    assert text.count("SURFACE (") == 12  # six definitions, six body references
    assert "Z-SHIFT=(+1.000000000000000E-02,   0)" in text
    assert "X-SHIFT=(-5.000000000000000E-01,   0)" in text
    assert "Y-SHIFT=(+5.000000000000000E-01,   0)" in text
    sides = [line[-3:-1] for line in text.splitlines() if "SIDE POINTER" in line]
    assert sides == ["+1", "-1", "+1", "-1", "+1", "-1"]
    with pytest.raises(ValueError):
        geometry_text(0.0, 1.0)


TALLY = """\
T 1 5 650000
G 1 20000
G 1 60000
S 1 2 3
T 2 4 400000
T 3 7 9000
G 3 200000
G 3 5000
T 4 6 300000
"""


def test_parse_tally_groups_photons_by_primary():
    parsed = parse_tally(TALLY, 4)
    assert parsed["ids"].tolist() == [0, 1, 2, 3]
    assert parsed["unscored_secondaries"] == 1
    scores = primary_scores(parsed, 4, kind="source")
    assert scores["transmitted"].tolist() == [1, 0, 0, 0]
    assert scores["backscattered"].tolist() == [0, 1, 0, 0]
    assert scores["stopped"].tolist() == [0, 0, 1, 0]
    assert scores["side_exited"].tolist() == [0, 0, 0, 1]
    assert scores["transmitted_energy_keV_per_primary"].tolist() == [650, 0, 0, 0]
    # The 5 keV photon is below every scored band.
    assert scores["photons_ge10kev"].tolist() == [2, 0, 1, 0]
    assert scores["photons_10_50kev"].tolist() == [1, 0, 0, 0]
    assert scores["photons_50_100kev"].tolist() == [1, 0, 0, 0]
    assert scores["photons_ge100kev"].tolist() == [0, 0, 1, 0]
    record = comparison_record(
        case_id="c",
        engine="penred",
        arm="cpu",
        status="ok",
        n_primaries=4,
        energy_keV=800.0,
        seed=7,
        scores=scores,
    )
    assert record["moments"]["photons_ge10kev"]["mean"] == 0.75
    transport = primary_scores(parsed, 4, kind="transport")
    assert not any(name.startswith("photons") for name in transport)
    comparison_record(
        case_id="c",
        engine="penred",
        arm="cpu",
        status="ok",
        n_primaries=4,
        energy_keV=800.0,
        seed=7,
        scores=transport,
        kind="transport",
    )


@pytest.mark.parametrize(
    "text",
    [
        TALLY + "T 4 5 1\n",  # duplicate terminal
        TALLY.replace("T 4 6 300000\n", ""),  # missing primary
        TALLY + "G 9 20000\n",  # photon of no primary
        TALLY + "X 1\n",  # unknown record
        TALLY.replace("T 2 4", "T 2 3"),  # unknown terminal
    ],
)
def test_parse_tally_rejects_broken_records(text):
    with pytest.raises(ValueError):
        parse_tally(text, 4)


def test_adapter_scene_support_and_log_timings(adapter):
    matrix = json.loads((ROOT / "matrix.json").read_text())
    cases = {case["id"]: case for case in matrix["cases"]}
    assert adapter["case_unsupported"](cases["silicon-800kev-tr0.3-0deg-straggling"]) == []
    assert adapter["case_unsupported"](cases["silicon-800kev-tr0.3-0deg-spot100um"]) == [
        "beam spot_rms_mm=0.1"
    ]
    assert adapter["case_arms"](cases["silicon-800kev-tr0.3-0deg-spot100um"]) == []
    assert adapter["case_arms"](cases["tungsten-800kev-tr3-75deg-oblique"]) == [
        "source",
        "transport",
    ]
    assert "secondary_transport" in adapter["case_unsupported"](
        cases["tungsten-800kev-tr0.3-0deg-secondaries"]
    )
    timings, seeds = adapter["parse_log_timings"](
        "Initialization processing time: 7.000000E+00 s\n"
        "Thread 0: Starting simulation of source source1 with initial history 0 and seeds 61 17.\n"
        "Simulation real time:   2.0000E+00 s\nSimulation user time:   1.5733E+00 s\n"
    )
    assert timings == {"initialization": 7.0, "simulation_real": 2.0, "simulation_user": 1.5733}
    assert seeds == [61, 17]


FAKE_PEN_MAIN = textwrap.dedent(
    """\
    import re, sys
    from pathlib import Path
    text = Path(sys.argv[1]).read_text()
    n = int(re.search(r"nhist (\\d+)", text).group(1))
    # Like PenRed 1.15.2: the last sampled particle of the request is not simulated.
    rows = [f"T {h} {5 if h % 2 else 4} 700000\\nG {h} 20000" for h in range(1, n)]
    Path("primary-th0-pyrite-primary.dat").write_text("\\n".join(rows) + "\\n")
    Path("slab.mat").write_text("material")
    Path("simulation.log").write_text("Simulation user time:   1.0E-01 s\\n")
    """
)


def test_adapter_requests_one_extra_history(tmp_path):
    fake = tmp_path / "pen_main"
    fake.write_text(f"#!{sys.executable}\n" + FAKE_PEN_MAIN)
    fake.chmod(0o755)
    output = tmp_path / "out" / "record.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "penred_adapter.py"),
            "silicon-800kev-tr0.3-75deg-oblique",
            "--histories",
            "6",
            "--seed",
            "3",
            "--executable",
            str(fake),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    record = json.loads(output.read_text())
    assert record["status"] == "ok" and record["engine"] == "penred"
    assert record["provenance"]["requested_histories"] == 7
    assert record["moments"]["transmitted"]["mean"] == 0.5
    assert record["moments"]["photons_ge10kev"]["mean"] == 1.0
    assert record["provenance"]["timings_s"]["simulation_user"] == 0.1
    assert (output.with_suffix(".npz")).exists()


def test_prepare_rejects_unpinned_source(tmp_path):
    for name in [
        "src/tallies/generic/includes/genericTallies.hh",
        "src/tallies/generic/source/genericTallies.cpp",
    ]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("not pinned")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "prepare_penred.py"), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0 and "does not match pinned PenRed" in completed.stderr
