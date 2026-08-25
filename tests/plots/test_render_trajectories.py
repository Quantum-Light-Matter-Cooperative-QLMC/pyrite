"""Cache-key and offscreen-render regression checks for
``plots.plotly.render``."""

import os

import pytest

from pyrite.plots.plotly.render import (
    cached_render_path,
    prune_render_cache,
    render_cache_dir,
    render_cache_key,
    render_reveal_animation,
)


def _key(**overrides):
    kwargs = {
        "material_name": "hopg",
        "case_params": {"E0_keV": 30.0, "tilt_deg": 0.0},
        "Ne": 40,
        "seed": 0,
        "realistic": False,
        "beam_fwhm_mm": None,
        "n_frames": 60,
        "fps": 12,
        "camera": None,
    }
    kwargs.update(overrides)
    return render_cache_key(**kwargs)


def test_render_cache_key_stable_for_same_inputs():
    assert _key() == _key()


def test_render_cache_key_changes_with_Ne():
    assert _key() != _key(Ne=41)


def test_render_cache_key_changes_with_n_frames():
    assert _key() != _key(n_frames=24)


def test_render_cache_key_changes_with_camera():
    assert _key() != _key(camera={"eye": {"x": 1.0, "y": 1.0, "z": 1.0}})


def test_render_cache_key_changes_with_case_params():
    assert _key() != _key(case_params={"E0_keV": 50.0, "tilt_deg": 0.0})


def test_render_cache_dir_honors_xdg_cache_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    out = render_cache_dir()
    assert out == tmp_path / "pyrite" / "viewer-renders"
    assert out.is_dir()


def test_cached_render_path_uses_cache_dir_and_key(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    key = _key()
    path = cached_render_path(key, suffix=".gif")
    assert path == render_cache_dir() / f"{key}.gif"


def test_prune_render_cache_keeps_newest_files(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache_dir = render_cache_dir()
    paths = []
    for i in range(5):
        p = cache_dir / f"render-{i}.mp4"
        p.write_bytes(b"x")
        # Stagger mtimes so ordering is deterministic regardless of filesystem
        # timestamp resolution.
        os.utime(p, (i * 100, i * 100))
        paths.append(p)

    prune_render_cache(keep=2)

    remaining = {p.name for p in cache_dir.iterdir()}
    assert remaining == {"render-3.mp4", "render-4.mp4"}


def test_render_reveal_animation_raises_clear_error_when_deps_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("imageio.v3", "kaleido"):
            raise ImportError(f"no module named {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="viz-render"):
        render_reveal_animation(None, {"start_xyz": [], "end_xyz": []}, "out.mp4")
