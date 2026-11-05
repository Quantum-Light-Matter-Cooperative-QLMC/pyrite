"""A chunk chain recomputes once per material, then resumes its own work (#495)."""

import os
import subprocess

import pytest

from pyrite.remote import config, scripts
from tests.remote._remote_test_helpers import _bash_or_skip


@pytest.mark.parametrize("first_exit", [75, 143], ids=["budget", "prelimit"])
def test_recompute_resumes_each_material_after_its_first_slice(monkeypatch, tmp_path, first_exit):
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    (jobdir / "meta").write_text("slurm_job_id: 100\n")
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = sync ]; then exit 0; fi\n'
        'case "$*" in *"tables verify"*) exit 0 ;; esac\n'
        f'echo "$*" >> "{tmp_path}/calls"\n'
        'case "$*" in *"-m hopg"*) m=hopg ;; *) m=hbn ;; esac\n'
        f'count_file="{tmp_path}/$m.count"\n'
        'count=$(cat "$count_file" 2>/dev/null || echo 0)\n'
        # Model persisted progress: restarting clears it unless checkpoint resume
        # is requested; old shared-cache records must stay unavailable.
        'case " $* " in *" --recompute "*) ;; *) exit 9 ;; esac\n'
        'case " $* " in *" --resume-checkpoint "*) ;; *) count=0 ;; esac\n'
        'count=$((count + 1))\necho "$count" > "$count_file"\n'
        f'echo "{{\\"cached_cases\\":0,\\"completed_new_cases\\":$count}}" > "{jobdir}/progress/$m.json"\n'
        '[ "$count" -ge 2 ] && exit 0\n'
        + ('kill -USR1 "$(ps -o ppid= -p "$PPID")"\n' if first_exit == 143 else "")
        + f"exit {first_exit}\n"
    )
    fake_uv.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text("#!/bin/sh\necho 4242\n")
    (bin_dir / "sbatch").chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "REMOTE_UV", str(fake_uv))
    script = scripts._chunked_queue_script("j", ["hopg", "hbn"], False, None, 10, recompute=True)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    for _ in range(3 if first_exit == 143 else 2):
        result = subprocess.run([bash, "-c", script], capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
    assert (jobdir / "state").read_text().startswith("done [2/2]")
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == 4
    for material in ("hopg", "hbn"):
        member_calls = [call for call in calls if f"-m {material}" in call]
        assert "--resume-checkpoint" not in member_calls[0]
        assert "--resume-checkpoint" in member_calls[1]
