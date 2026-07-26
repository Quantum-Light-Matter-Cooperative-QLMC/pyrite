"""Offscreen prerendered-video path for the penetration-tab reveal animation.

Complements the interactive Plotly animation in ``plotly_trajectories`` with a
fixed-camera video (arbitrary frame count, no payload ceiling) for the viewer's
"Render" button. Reuses an already-transported ``data`` dict from
:func:`cxr_mc.plots.plotly_trajectories.trajectory_volume_data` -- no new Monte
Carlo transport, no new physics, so no validation-ledger marker is needed here.

Heavy deps (``kaleido``, ``imageio``, ``imageio-ffmpeg``) are optional (extra
``viz-render``) and imported lazily inside :func:`render_reveal_animation` so
importing this module never requires them.
"""

import hashlib
import json
import os
from pathlib import Path

from cxr_mc.plots.plotly_trajectories import (
    dataset_t_max,
    frame_reveal_fs,
    trajectory_volume_figure_from_data,
)

_RENDER_SALT = "v1"  # bump on any change to the cache-key inputs or render format


def render_reveal_animation(
    rec_or_case,
    data,
    out_path,
    *,
    realistic=False,
    beam_fwhm_mm=None,
    n_frames=60,
    fps=12,
    width=960,
    height=560,
    camera=None,
    progress_cb=None,
):
    """Render the reveal animation for ``data`` to a fixed-camera video file.

    Per frame ``k`` in ``range(n_frames)``: derive a reveal cutoff via
    :func:`frame_reveal_fs` against :func:`dataset_t_max`, build the static
    figure via :func:`trajectory_volume_figure_from_data`, pin the camera (if
    ``camera`` given) via ``fig.update_layout(scene_camera=camera)``, and export
    a PNG frame via kaleido. Frames are stitched into ``out_path``: ``.gif``
    writes an animated GIF (no ffmpeg needed), any other suffix (default usage:
    ``.mp4``) writes h264 video via ``imageio-ffmpeg``.

    ``progress_cb(k, n_frames)``, if given, is called once per frame after that
    frame's PNG has been rendered.

    Returns ``Path(out_path)``. Raises ``RuntimeError`` naming the
    ``viz-render`` install extra if kaleido/imageio are not installed.
    """
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError(
            "render_reveal_animation requires the optional 'viz-render' extra "
            "(kaleido, imageio, imageio-ffmpeg). Install with "
            '`uv sync --extra viz-render` or `pip install "cxr-mc[viz-render]"`.'
        ) from exc
    try:
        import kaleido  # noqa: F401  (import-only availability check)
    except ImportError as exc:
        raise RuntimeError(
            "render_reveal_animation requires the optional 'viz-render' extra "
            "(kaleido, imageio, imageio-ffmpeg). Install with "
            '`uv sync --extra viz-render` or `pip install "cxr-mc[viz-render]"`.'
        ) from exc

    out_path = Path(out_path)
    t_max = dataset_t_max(data)
    frames = []
    for k in range(n_frames):
        cutoff = frame_reveal_fs(k, t_max, n_frames)
        fig = trajectory_volume_figure_from_data(
            rec_or_case,
            data,
            realistic=realistic,
            beam_fwhm_mm=beam_fwhm_mm,
            reveal_until_fs=cutoff,
        )
        if camera is not None:
            fig.update_layout(scene_camera=camera)
        png_bytes = fig.to_image(format="png", width=width, height=height)
        frames.append(iio.imread(png_bytes, extension=".png"))
        if progress_cb is not None:
            progress_cb(k, n_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".gif":
        iio.imwrite(out_path, frames, extension=".gif", fps=fps, loop=0)
    else:
        iio.imwrite(out_path, frames, extension=".mp4", fps=fps, codec="libx264")
    return out_path


def _json_default(obj):
    """``json.dumps`` default hook: stable ``repr``-based text for floats/other
    non-JSON-native values so the cache key is exact and reproducible."""
    return repr(obj)


def render_cache_key(
    material_name,
    case_params,
    Ne,
    seed,
    realistic,
    beam_fwhm_mm,
    n_frames,
    fps,
    camera,
):
    """Stable sha256 hex digest over the full render parameter set, prefixed
    with :data:`_RENDER_SALT` so a format/logic change invalidates old cache
    entries without a collision. Canonical JSON dump (``sort_keys=True``) makes
    the key independent of caller argument/dict-insertion order."""
    payload = {
        "salt": _RENDER_SALT,
        "material_name": material_name,
        "case_params": case_params,
        "Ne": Ne,
        "seed": seed,
        "realistic": realistic,
        "beam_fwhm_mm": beam_fwhm_mm,
        "n_frames": n_frames,
        "fps": fps,
        "camera": camera,
    }
    blob = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render_cache_dir():
    """``~/.cache/cxr-mc/viewer-renders`` (honoring ``XDG_CACHE_HOME`` if set),
    created if missing."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    out = root / "cxr-mc" / "viewer-renders"
    out.mkdir(parents=True, exist_ok=True)
    return out


def cached_render_path(key, suffix=".mp4"):
    """Path a render for ``key`` should live at under :func:`render_cache_dir`."""
    return render_cache_dir() / f"{key}{suffix}"


def prune_render_cache(keep=20):
    """Delete oldest-mtime files under :func:`render_cache_dir` beyond the
    newest ``keep``, bounding cache disk usage."""
    cache_dir = render_cache_dir()
    files = [p for p in cache_dir.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in files[keep:]:
        stale.unlink(missing_ok=True)
