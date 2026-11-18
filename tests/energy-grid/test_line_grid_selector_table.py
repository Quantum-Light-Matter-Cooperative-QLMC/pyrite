"""The line-grid selector table in ``profile-settings.md`` matches the resolver (#1265).

Each row below is rendered into the documentation table and is also the rule
the resolver must follow: every (bandwidth, resolution, quadrature, windows)
combination, on the incoherent (measured default) and coherent/``max_dE_frac``
(kinematic default) builders, resolves as the first matching row says, and its
fallback carries the resolution unchanged.
"""

import itertools
from pathlib import Path

import pytest

from pyrite._cxr_grid_policy import (
    KINEMATIC_BANDWIDTH_POLICY,
    RESONANCE_BANDWIDTH_POLICY,
    kinematic_fallback_payload,
    resolve_cxr_grid_policy,
)
from pyrite.montecarlo.runner.cxr_grid import _auditable_policy

KC, RP = KINEMATIC_BANDWIDTH_POLICY, RESONANCE_BANDWIDTH_POLICY
OS, NYQ, LOC = "sinc-oversampled", "sinc-nyquist", "resonance-local"
UNSET = None
ANY_QUADRATURE = ("node", "bin-mean")
DOC = Path(__file__).parents[2] / "docs" / "repo-design" / "profile-settings.md"

#: (bandwidth set, resolution set, quadrature set, windows, axis, fallback).
#: ``axis`` None refuses; ``"default"`` bandwidth is the builder's default.
ROWS = (
    ((UNSET, KC), (UNSET, NYQ), ANY_QUADRATURE, True, (KC, NYQ), "none"),
    ((UNSET, RP), (LOC,), ("bin-mean",), False, (RP, LOC), "refuses"),
    ((UNSET,), (UNSET, OS, NYQ), ANY_QUADRATURE, False, ("default", "as chosen"), "widens"),
    ((RP,), (UNSET, OS, NYQ), ANY_QUADRATURE, False, (RP, "as chosen"), "refuses"),
    ((KC,), (UNSET, OS, NYQ), ANY_QUADRATURE, False, (KC, "as chosen"), "none"),
)

_FALLBACK_TEXT = {
    "none": "none (already `kinematic-ceiling`)",
    "refuses": "refuses",
    "widens": "widens toward `kinematic-ceiling`",
}


def _cells(values):
    return ", ".join("unset" if v is UNSET else f"`{v}`" for v in values)


def _axis_text(axis):
    bandwidth, resolution = axis
    bandwidth = "default¹" if bandwidth == "default" else f"`{bandwidth}`"
    resolution = "as chosen²" if resolution == "as chosen" else f"`{resolution}`"
    return f"{bandwidth} / {resolution}"


def render_table():
    """The Markdown table ``profile-settings.md`` must contain verbatim."""
    lines = [
        "| `bandwidth` | `resolution` | `quadrature` | `windows` | Axis | Fallback³ |",
        "|---|---|---|---|---|---|",
    ]
    for bandwidths, resolutions, quadratures, windows, axis, fallback in ROWS:
        quadrature = "any" if quadratures == ANY_QUADRATURE else _cells(quadratures)
        lines.append(
            f"| {_cells(bandwidths)} | {_cells(resolutions)} | {quadrature} | "
            f"{'on' if windows else 'off'} | {_axis_text(axis)} | {_FALLBACK_TEXT[fallback]} |"
        )
    return "\n".join(lines)


def _expected(bandwidth, resolution, quadrature, windows):
    for bandwidths, resolutions, quadratures, row_windows, axis, fallback in ROWS:
        if (
            bandwidth in bandwidths
            and resolution in resolutions
            and quadrature in quadratures
            and windows == row_windows
        ):
            return axis, fallback
    return None, None


def test_profile_settings_contains_the_rendered_selector_table():
    assert render_table() in DOC.read_text(encoding="utf-8")


COMBINATIONS = list(
    itertools.product(
        (RP, KC), (UNSET, RP, KC), (UNSET, OS, NYQ, LOC), ANY_QUADRATURE, (False, True)
    )
)


@pytest.mark.parametrize(
    ("builder", "bandwidth", "resolution", "quadrature", "windows"), COMBINATIONS
)
def test_every_combination_resolves_and_falls_back_as_its_row_says(
    builder, bandwidth, resolution, quadrature, windows
):
    """Validation: line-grid-resonance-bandwidth"""
    per_call = {"quadrature": quadrature, "windows": windows}
    if bandwidth is not UNSET:
        per_call["bandwidth"] = bandwidth
    if resolution is not UNSET:
        per_call["resolution"] = resolution
    axis, fallback = _expected(bandwidth, resolution, quadrature, windows)
    if axis is None:
        with pytest.raises(ValueError):
            resolve_cxr_grid_policy(
                start_eV=10.0, stop_eV=1000.0, bandwidth_policy=builder, per_call=per_call
            )
        return
    payload = resolve_cxr_grid_policy(
        start_eV=10.0, stop_eV=1000.0, bandwidth_policy=builder, per_call=per_call
    ).payload()
    want_bandwidth = builder if axis[0] == "default" else axis[0]
    want_resolution = (resolution or OS) if axis[1] == "as chosen" else axis[1]
    assert payload["bandwidth"]["policy"] == want_bandwidth
    assert payload["resolution"]["policy"] == want_resolution

    unaudited = {"name": "case", "channeling": {}}
    widened = kinematic_fallback_payload(payload)
    if want_bandwidth == KC:
        assert widened is None
        assert _auditable_policy(unaudited, payload) is payload
    elif fallback == "widens":
        assert widened["bandwidth"]["policy"] == KC
        assert widened["resolution"] == payload["resolution"]
        assert widened.get("quadrature") == payload.get("quadrature")
        assert _auditable_policy(unaudited, payload) == widened
    else:
        assert fallback == "refuses"
        assert widened is None
        with pytest.raises(ValueError):
            _auditable_policy(unaudited, payload)


def test_deprecated_sinc_shape_is_the_bin_mean_oversampled_step():
    with pytest.warns(DeprecationWarning, match="sinc-shape"):
        alias = resolve_cxr_grid_policy(
            start_eV=10.0,
            stop_eV=1000.0,
            per_call={"resolution": "sinc-shape", "quadrature": "bin-mean"},
        )
    canonical = resolve_cxr_grid_policy(
        start_eV=10.0, stop_eV=1000.0, per_call={"quadrature": "bin-mean"}
    )
    assert alias.payload()["resolution"] == canonical.payload()["resolution"]
    with pytest.raises(ValueError, match="needs bin-mean"):
        resolve_cxr_grid_policy(
            start_eV=10.0, stop_eV=1000.0, per_call={"resolution": "sinc-shape"}
        )
