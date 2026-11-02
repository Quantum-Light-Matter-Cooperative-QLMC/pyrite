"""Capture/replay transport metadata retains numeric arrays and table identities."""

import importlib.util
import json
from pathlib import Path

import numpy as np


def test_nested_capture_metadata_roundtrip(tmp_path):
    path = Path(__file__).parents[3] / "checks/brem_primary_band_benchmark.py"
    spec = importlib.util.spec_from_file_location("brem_band_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = {
        "stopping_tables": (
            {"energy": np.array([1.0, 2.0]), "values": np.array([[3, 4]], dtype=np.int32)},
        ),
        "radiative": {"bremslib_tables": (("C", 6, "key", "digest"),)},
        "Ne": np.int64(7),
        "thickness_ang": np.float32(200.0),
    }
    target = tmp_path / "metadata.json"
    module.write_json(target, value)
    restored = module.metadata_decode(json.loads(target.read_text()))
    assert isinstance(restored["stopping_tables"], tuple)
    for key in ("energy", "values"):
        np.testing.assert_array_equal(
            restored["stopping_tables"][0][key], value["stopping_tables"][0][key]
        )
        assert restored["stopping_tables"][0][key].dtype == value["stopping_tables"][0][key].dtype
    assert restored["radiative"] == value["radiative"]
    assert restored["Ne"] == 7 and restored["thickness_ang"] == 200.0
