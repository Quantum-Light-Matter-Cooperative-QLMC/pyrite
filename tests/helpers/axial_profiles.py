"""User-layer profiles for the admitted axial fibre cases (#624).

Each profile names the packaged ``point`` detector and a named target with
``cut = {surface_hkl = [0, 0, 1]}`` and ``footprint = "unbounded"`` (plus
``allow_normal_incidence`` for si-fibre/B). Refusal variants change exactly one
envelope property. No profile declares a continuum grid.
"""

from contextlib import contextmanager

from pyrite.materials import catalog as catalog_module
from pyrite.materials.catalog import load_material_catalog

LINES, BREM = 4, 6
TILT = {"A": (0.088, 225.0), "B": (0.0, 0.0)}


def _target_toml(key, *, thickness_ang=3000.0, footprint='footprint = "unbounded"'):
    tilt, azimuth = TILT[key]
    return (
        'kind = "slab"\nmaterial = "silicon"\n'
        f"thickness_ang = {thickness_ang}\ntilt_deg = {tilt}\ntilt_azim_deg = {azimuth}\n"
        f"allow_normal_incidence = {'true' if tilt == 0.0 else 'false'}\n"
        f"{footprint}\ncut = {{surface_hkl = [0, 0, 1]}}\n"
    )


def _profile_toml(target, detector='"point"'):
    return f"""thickness_ang = {{values = [3000.0]}}
energy_keV = {{values = [3500.0]}}
tilt_deg = {{values = [5.0]}}
tilt_azim_deg = {{values = [0.0]}}
materials = ["silicon"]
cxr_histories = {{values = [{LINES}]}}
brem_histories = {{values = [{BREM}]}}
beam = "example"
detector = {detector}
emission = "incoherent"
energy_model = "midpoint"
radiative_model = "uncoupled"
E_grid_line = {{linspace = {{start = 800.0, stop = 10000.0, num = 47}}}}

[targets]
silicon = "{target}"

[channeling]
model = "axial"
axial = {{version = 2, axis_uvw = [0, 0, 1]}}
"""


@contextmanager
def axial_profiles(layer):
    """Write the profiles into user ``layer`` and refresh the default catalog."""
    (layer / "targets").mkdir(exist_ok=True)
    files = {
        "targets/si_fibre_a.toml": _target_toml("A"),
        "targets/si_fibre_b.toml": _target_toml("B"),
        "targets/si_fibre_b_thin.toml": _target_toml("B", thickness_ang=2000.0),
        "targets/si_fibre_b_finite.toml": _target_toml("B", footprint=""),
        "profiles/axial_fibre_a.toml": _profile_toml("si_fibre_a"),
        "profiles/axial_fibre_b.toml": _profile_toml("si_fibre_b"),
        "profiles/axial_fibre_thin.toml": _profile_toml("si_fibre_b_thin"),
        "profiles/axial_fibre_finite.toml": _profile_toml("si_fibre_b_finite"),
        "profiles/axial_fibre_aperture.toml": _profile_toml("si_fibre_b", '"forward"'),
        "profiles/axial_fibre_off_axis.toml": _profile_toml(
            "si_fibre_b", '{kind = "point", observation_angle_deg = 1.0}'
        ),
    }
    for relative, text in files.items():
        (layer / relative).write_text(text)
    previous = catalog_module._get_default_catalog()
    catalog_module._refresh_default_catalog(load_material_catalog())
    try:
        yield layer
    finally:
        catalog_module._refresh_default_catalog(previous)
