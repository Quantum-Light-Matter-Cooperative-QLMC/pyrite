"""CLI checks for ``pyrite config setup``: hardware detection and config-store writes."""

from pathlib import Path

from pyrite import cli
from pyrite.cli.commands import backend_setup
from pyrite.console import config as _config

# ---- hardware detection (mocked subprocess/tool presence, no real hardware) ----


def test_detect_nvidia_via_tool_on_path(monkeypatch):
    monkeypatch.setattr(
        backend_setup.shutil,
        "which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )
    assert backend_setup.detect_nvidia() == backend_setup.DetectionResult(
        "nvidia", "cuda", "nvidia-smi on PATH"
    )


def test_detect_nvidia_via_device_node_when_tool_absent(monkeypatch):
    monkeypatch.setattr(backend_setup.shutil, "which", lambda name: None)
    monkeypatch.setattr(backend_setup, "_has_device_nodes", lambda pattern: pattern == "nvidia*")
    assert backend_setup.detect_nvidia() == backend_setup.DetectionResult(
        "nvidia", "cuda", "/dev/nvidia* device node"
    )


def test_detect_nvidia_absent(monkeypatch):
    monkeypatch.setattr(backend_setup.shutil, "which", lambda name: None)
    monkeypatch.setattr(backend_setup, "_has_device_nodes", lambda pattern: False)
    assert backend_setup.detect_nvidia() is None


def test_detect_amd_via_rocm_smi(monkeypatch):
    monkeypatch.setattr(
        backend_setup.shutil,
        "which",
        lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None,
    )
    assert backend_setup.detect_amd() == backend_setup.DetectionResult(
        "amd", "rocm", "rocm-smi on PATH"
    )


def test_detect_amd_via_rocminfo_when_rocm_smi_absent(monkeypatch):
    monkeypatch.setattr(
        backend_setup.shutil,
        "which",
        lambda name: "/usr/bin/rocminfo" if name == "rocminfo" else None,
    )
    assert backend_setup.detect_amd() == backend_setup.DetectionResult(
        "amd", "rocm", "rocminfo on PATH"
    )


def test_detect_amd_via_device_node_when_tools_absent(monkeypatch):
    monkeypatch.setattr(backend_setup.shutil, "which", lambda name: None)
    monkeypatch.setattr(backend_setup, "_has_device_nodes", lambda pattern: pattern == "kfd")
    assert backend_setup.detect_amd() == backend_setup.DetectionResult(
        "amd", "rocm", "/dev/kfd device node"
    )


def test_detect_amd_absent(monkeypatch):
    monkeypatch.setattr(backend_setup.shutil, "which", lambda name: None)
    monkeypatch.setattr(backend_setup, "_has_device_nodes", lambda pattern: False)
    assert backend_setup.detect_amd() is None


def test_detect_intel_via_sycl_ls(monkeypatch):
    monkeypatch.setattr(
        backend_setup.shutil,
        "which",
        lambda name: "/usr/bin/sycl-ls" if name == "sycl-ls" else None,
    )
    monkeypatch.setattr(
        backend_setup, "_run_text", lambda cmd: "[opencl:gpu] Intel(R) UHD Graphics\n"
    )
    assert backend_setup.detect_intel() == backend_setup.DetectionResult(
        "intel", "sycl", "sycl-ls output"
    )


def test_detect_intel_via_clinfo_when_sycl_ls_absent(monkeypatch):
    monkeypatch.setattr(
        backend_setup.shutil, "which", lambda name: "/usr/bin/clinfo" if name == "clinfo" else None
    )
    monkeypatch.setattr(
        backend_setup, "_run_text", lambda cmd: "Platform Vendor: Intel(R) Corporation\n"
    )
    assert backend_setup.detect_intel() == backend_setup.DetectionResult(
        "intel", "sycl", "clinfo output"
    )


def test_detect_intel_via_lspci_vga_when_sycl_tools_absent(monkeypatch):
    monkeypatch.setattr(
        backend_setup.shutil, "which", lambda name: "/usr/bin/lspci" if name == "lspci" else None
    )
    monkeypatch.setattr(
        backend_setup,
        "_run_text",
        lambda cmd: "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics\n",
    )
    assert backend_setup.detect_intel() == backend_setup.DetectionResult(
        "intel", "sycl", "lspci VGA controller"
    )


def test_detect_intel_ignores_non_intel_lspci_vga_line(monkeypatch):
    monkeypatch.setattr(
        backend_setup.shutil, "which", lambda name: "/usr/bin/lspci" if name == "lspci" else None
    )
    monkeypatch.setattr(
        backend_setup,
        "_run_text",
        lambda cmd: "00:02.0 VGA compatible controller: NVIDIA Corporation GA104\n",
    )
    assert backend_setup.detect_intel() is None


def test_detect_intel_absent_when_nothing_reports_intel(monkeypatch):
    monkeypatch.setattr(backend_setup.shutil, "which", lambda name: None)
    assert backend_setup.detect_intel() is None


def test_detect_all_orders_nvidia_amd_intel_and_drops_missing(monkeypatch):
    monkeypatch.setattr(
        backend_setup, "detect_nvidia", lambda: backend_setup.DetectionResult("nvidia", "cuda", "x")
    )
    monkeypatch.setattr(backend_setup, "detect_amd", lambda: None)
    monkeypatch.setattr(
        backend_setup, "detect_intel", lambda: backend_setup.DetectionResult("intel", "sycl", "y")
    )
    assert [r.vendor for r in backend_setup.detect_all()] == ["nvidia", "intel"]


def test_detect_all_empty_when_nothing_detected(monkeypatch):
    monkeypatch.setattr(backend_setup, "detect_nvidia", lambda: None)
    monkeypatch.setattr(backend_setup, "detect_amd", lambda: None)
    monkeypatch.setattr(backend_setup, "detect_intel", lambda: None)
    assert backend_setup.detect_all() == []


# ---- .env read/write: create, append, overwrite, preserve unrelated content ----


def test_read_existing_backend_missing_file_returns_none(tmp_path):
    assert backend_setup.read_existing_backend(tmp_path / ".env") is None


def test_read_existing_backend_ignores_comments_and_unrelated_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nMP_API_KEY_ENV=abc\nPYRITE_MC_BACKEND=cuda\n")
    assert backend_setup.read_existing_backend(env) == "cuda"


# ---- `pyrite setup` CLI integration ----


def _isolate_env(monkeypatch, tmp_path) -> Path:
    env_path = tmp_path / ".env"
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.delenv("PYRITE_MC_BACKEND", raising=False)
    monkeypatch.setattr(backend_setup, "_env_path", lambda: env_path)
    return env_path


def test_setup_no_env_non_interactive_defaults_to_cpu(monkeypatch, tmp_path, capsys):
    env_path = _isolate_env(monkeypatch, tmp_path)
    monkeypatch.setattr(backend_setup, "_interactive", lambda: False)
    monkeypatch.setattr(backend_setup, "detect_all", lambda: [])

    assert cli.main(["config", "setup"]) is None

    assert not env_path.exists()
    assert _config.resolve("backend.default").value == "cpu"
    assert "backend.default=cpu" in capsys.readouterr().out


def test_setup_no_gpu_prints_no_install_instructions(monkeypatch, tmp_path, capsys):
    _isolate_env(monkeypatch, tmp_path)
    monkeypatch.setattr(backend_setup, "_interactive", lambda: False)
    monkeypatch.setattr(backend_setup, "detect_all", lambda: [])

    assert cli.main(["config", "setup"]) is None

    assert "pixi install" not in capsys.readouterr().out


def test_setup_detected_gpu_non_interactive_defaults_to_cpu_with_stderr_hint(
    monkeypatch, tmp_path, capsys
):
    env_path = _isolate_env(monkeypatch, tmp_path)
    monkeypatch.setattr(backend_setup, "_interactive", lambda: False)
    monkeypatch.setattr(
        backend_setup,
        "detect_all",
        lambda: [backend_setup.DetectionResult("nvidia", "cuda", "nvidia-smi on PATH")],
    )

    assert cli.main(["config", "setup"]) is None

    assert not env_path.exists()
    assert _config.resolve("backend.default").value == "cpu"
    captured = capsys.readouterr()
    assert "non-interactive" in captured.err
    assert "pyrite config setup -y" in captured.err


def test_setup_yes_flag_accepts_top_detected_backend_without_prompt(monkeypatch, tmp_path, capsys):
    env_path = _isolate_env(monkeypatch, tmp_path)
    monkeypatch.setattr(backend_setup, "_interactive", lambda: True)
    monkeypatch.setattr(
        backend_setup,
        "detect_all",
        lambda: [backend_setup.DetectionResult("nvidia", "cuda", "nvidia-smi on PATH")],
    )

    assert cli.main(["config", "setup", "-y"]) is None

    assert not env_path.exists()
    assert _config.resolve("backend.default").value == "cuda"
    assert "pixi install -e nvidia" in capsys.readouterr().out


def test_setup_interactive_prompt_accept_writes_detected_backend(monkeypatch, tmp_path, capsys):
    env_path = _isolate_env(monkeypatch, tmp_path)
    monkeypatch.setattr(backend_setup, "_interactive", lambda: True)
    monkeypatch.setattr(
        backend_setup,
        "detect_all",
        lambda: [backend_setup.DetectionResult("amd", "rocm", "rocm-smi on PATH")],
    )
    monkeypatch.setattr(backend_setup.click, "confirm", lambda *args, **kwargs: True)

    assert cli.main(["config", "setup"]) is None

    assert not env_path.exists()
    assert _config.resolve("backend.default").value == "rocm"
    assert "CUPY_INSTALL_USE_HIP=1 pixi install -e amd" in capsys.readouterr().out


def test_setup_interactive_prompt_decline_defaults_to_cpu(monkeypatch, tmp_path):
    env_path = _isolate_env(monkeypatch, tmp_path)
    monkeypatch.setattr(backend_setup, "_interactive", lambda: True)
    monkeypatch.setattr(
        backend_setup,
        "detect_all",
        lambda: [backend_setup.DetectionResult("amd", "rocm", "rocm-smi on PATH")],
    )
    monkeypatch.setattr(backend_setup.click, "confirm", lambda *args, **kwargs: False)

    assert cli.main(["config", "setup"]) is None

    assert not env_path.exists()
    assert _config.resolve("backend.default").value == "cpu"


def test_setup_already_set_is_a_no_op(monkeypatch, tmp_path, capsys):
    env_path = _isolate_env(monkeypatch, tmp_path)
    _config.set_stored("backend.default", "cuda")
    calls: list[bool] = []
    monkeypatch.setattr(backend_setup, "detect_all", lambda: calls.append(True) or [])

    assert cli.main(["config", "setup"]) is None

    assert not env_path.exists()
    assert _config.resolve("backend.default").value == "cuda"
    assert calls == []
    assert "already set" in capsys.readouterr().out


def test_setup_force_reruns_detection_and_overwrites(monkeypatch, tmp_path):
    env_path = _isolate_env(monkeypatch, tmp_path)
    _config.set_stored("backend.default", "cuda")
    monkeypatch.setattr(backend_setup, "_interactive", lambda: False)
    monkeypatch.setattr(backend_setup, "detect_all", lambda: [])

    assert cli.main(["config", "setup", "--force"]) is None

    assert not env_path.exists()
    assert _config.resolve("backend.default").value == "cpu"


def test_setup_preserves_unrelated_env_content(monkeypatch, tmp_path):
    env_path = _isolate_env(monkeypatch, tmp_path)
    env_path.write_text("MP_API_KEY_ENV=abc\n")
    monkeypatch.setattr(backend_setup, "_interactive", lambda: False)
    monkeypatch.setattr(backend_setup, "detect_all", lambda: [])

    assert cli.main(["config", "setup"]) is None

    assert env_path.read_text() == "MP_API_KEY_ENV=abc\n"
    assert _config.resolve("backend.default").value == "cpu"


def test_repo_root_resolves_a_checkout_with_pyproject_toml():
    root = backend_setup._repo_root()
    assert (root / "pyproject.toml").is_file()
