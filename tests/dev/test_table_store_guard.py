"""The suite cannot replace production-key fixtures in a user table store."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("operation", ["store", "save", "copy", "remove", "replace", "rmtree"])
def test_direct_table_writes_outside_the_tmp_store_are_blocked(tmp_path, operation):
    # A fake user home keeps even a broken guard harmless to the real store.
    user = tmp_path / "fake-home" / "xsgen/tables"
    user.mkdir(parents=True)
    victim = user / "production-key.npz"
    victim.write_bytes(b"verified production bytes")
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "stub.npz"
    source.write_bytes(b"five-angle fixture")
    script = """
import os, shutil, sys
from pathlib import Path
import numpy as np
from tests.helpers.table_store_guard import TableStoreWriteGuard
allowed, victim, source, operation = sys.argv[1:]
sys.addaudithook(TableStoreWriteGuard([allowed]))
try:
    if operation == 'store':
        from pyrite.xsgen.store import ElementTarget, TableRequest, store
        os.environ['PYRITE_HOME'] = str(Path(victim).parents[2])
        store(TableRequest('elsepa', 'test', ElementTarget(14), 'elastic_dcs'), {'mu': np.arange(5)})
    elif operation == 'save':
        np.savez_compressed(victim, mu=np.linspace(0, 1, 5))
    elif operation == 'copy':
        shutil.copyfile(source, victim)
    elif operation == 'remove':
        Path(victim).unlink()
    elif operation == 'rmtree':
        shutil.rmtree(Path(victim).parent)
    else:
        Path(source).replace(victim)
except AssertionError as exc:
    assert 'outside a temporary table store' in str(exc)
else:
    raise AssertionError('guard allowed a real-store write')
np.savez_compressed(Path(allowed) / 'ok.npz', mu=np.linspace(0, 1, 5))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(allowed), str(victim), str(source), operation],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert victim.read_bytes() == b"verified production bytes"
    assert (allowed / "ok.npz").is_file()
