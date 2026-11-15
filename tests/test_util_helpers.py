"""Pin the domain-free helpers in ``pyrite._util`` (issue #1029).

The canonical-JSON digest feeds persisted identities (observation payloads,
photon observation identities, managed CIF names, axial settings), so its bytes
are pinned against a literal digest computed with the pre-consolidation code.
"""

import json
import os

import pytest

from pyrite._util.hashing import canonical_json_bytes, canonical_json_digest, file_sha256
from pyrite._util.io import atomic_write_json, atomic_write_text

_PAYLOAD = {"b": [1, 2.5, None], "a": {"z": "é", "y": True}, "e_keV": 1e-3}
#: sha256(json.dumps(_PAYLOAD, sort_keys=True, separators=(",", ":")).encode())
_PAYLOAD_SHA256 = "22492eb58f884760634dd198d88c144069cd95c5e7d13edea16e1588f07b2617"


def test_canonical_json_digest_is_pinned():
    assert canonical_json_bytes(_PAYLOAD) == (
        b'{"a":{"y":true,"z":"\\u00e9"},"b":[1,2.5,null],"e_keV":0.001}'
    )
    assert canonical_json_digest(_PAYLOAD) == _PAYLOAD_SHA256
    assert canonical_json_digest(_PAYLOAD, allow_nan=True) == _PAYLOAD_SHA256


def test_observation_payload_digest_is_unchanged_and_tolerates_nan():
    from pyrite.instrument.observation import payload_digest

    assert payload_digest(_PAYLOAD) == _PAYLOAD_SHA256
    payload_digest({"x": float("nan")})  # historical behavior: no ValueError
    with pytest.raises(ValueError):
        canonical_json_digest({"x": float("nan")})


def test_file_sha256_streams_whole_file(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"pyrite\n" * 100_000)  # spans several read blocks
    assert file_sha256(path) == "eec93aaed4ebcdbbe6d2a4b802ebab40606a7aee3afd83f6c434aafe388f1486"
    assert file_sha256(str(path)) == file_sha256(path)


def test_atomic_write_text_creates_parent_and_replaces(tmp_path):
    target = tmp_path / "nested" / "state.toml"
    atomic_write_text(target, "a = 1\n")
    atomic_write_text(str(target), "a = 2\n")
    assert target.read_text(encoding="utf-8") == "a = 2\n"
    assert sorted(os.listdir(target.parent)) == ["state.toml"]


def test_atomic_write_text_leaves_target_and_no_temporary_on_failure(tmp_path):
    target = tmp_path / "state.txt"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(UnicodeEncodeError):
        atomic_write_text(target, "é", encoding="ascii")
    assert target.read_text(encoding="utf-8") == "old"
    assert sorted(os.listdir(tmp_path)) == ["state.txt"]


def test_atomic_write_json_matches_indented_dump(tmp_path):
    target = tmp_path / "rows.json"
    atomic_write_json(target, [{"E_keV": 1.0}])
    assert target.read_text(encoding="utf-8") == json.dumps([{"E_keV": 1.0}], indent=2)


def test_paths_reexports_the_shared_writer():
    from pyrite import paths

    assert paths.atomic_write_text is atomic_write_text
