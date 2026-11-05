"""Execute the rendered allocation guard before and after a catalog edit (#489)."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from pyrite.remote import config, scripts
from pyrite.remote.physics_identity import expected_inventory
from tests.remote._remote_test_helpers import _bash_or_skip


@pytest.mark.slow
def test_profile_edit_between_slices_fails_before_scan_and_preserves_job_identity(
    monkeypatch, tmp_path
):
    bash = _bash_or_skip(tmp_path)
    profile = "sub_100keV"
    expected = expected_inventory(["hopg"], False, "full", profile)
    stem = expected["inventory"]["hopg/default"]["stem"]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(expected["inventory"]))
    # Resolve with the parent's synthetic table fixtures, then run the real
    # verifier entry point in the rendered shell. No table/GPU setup in children.
    verifier = (
        "import json; from pyrite.remote import physics_identity as p; "
        f"p.resolve_inventory = lambda request: json.load(open({str(inventory)!r})); p.main()"
    )
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    metadata = f"slurm_job_id: 100\nstems: {stem}\n"
    (jobdir / "meta").write_text(metadata)
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = sync ]; then exit 0; fi\n'
        'case "$*" in *"tables verify"*) exit 0 ;; esac\n'
        'case "$*" in *"pyrite.remote.physics_identity"*) shift 5; '
        f'exec {shlex.quote(sys.executable)} -c {shlex.quote(verifier)} "$@" ;; esac\n'
        f'echo scan >> "{tmp_path}/calls"\n'
        f'echo \'{{"cached_cases":0,"completed_new_cases":1}}\' > "{jobdir}/progress/hopg.json"\n'
        "exit 75\n"
    )
    fake_uv.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text("#!/bin/sh\necho 4242\n")
    (bin_dir / "sbatch").chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "REMOTE_UV", str(fake_uv))
    script = scripts._chunked_queue_script(
        "j",
        ["hopg"],
        False,
        None,
        10,
        catalog_profile=profile,
        expected_physics=expected,
    )
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    first = subprocess.run([bash, "-c", script], capture_output=True, text=True, env=env)
    assert first.returncode == 0, (first.stderr, (jobdir / "log").read_text())
    assert (jobdir / "state").read_text().startswith("queued slice 2")
    definition = Path(os.environ["PYRITE_USER_CATALOG"]) / "profiles" / f"{profile}.toml"
    original = definition.read_text()
    assert "30.0, 40.0" in original
    definition.write_text(original.replace("30.0, 40.0", "31.0, 40.0"))
    current = expected_inventory(["hopg"], False, "full", profile)["inventory"]
    assert current["hopg/default"]["stem"] != stem
    inventory.write_text(json.dumps(current))
    second = subprocess.run([bash, "-c", script], capture_output=True, text=True, env=env)
    assert second.returncode == 1
    assert (jobdir / "state").read_text().startswith("FAILED (physics identity)")
    log = (jobdir / "log").read_text()
    assert "hopg/default" in log and "sync, and resubmit" in log
    assert (tmp_path / "calls").read_text() == "scan\n"
    assert f"stems: {stem}\n" in (jobdir / "meta").read_text()
    assert (jobdir / "meta").read_text().count("slurm_job_id:") == 2
