"""
sweep.py
============

Define a parameter sweep and expand it into the per-case dicts that
``montecarlo.run_case`` consumes.

The driver notebook sets ONE :class:`Sweep`. Every physical parameter accepts
**either a single value (fixed) or a sequence/array (swept)**; :func:`build_cases`
takes the Cartesian product over whatever is swept. So

    Sweep(tilt_deg=30.0,               tilt_azim_deg=0.0)          # one case geometry
    Sweep(tilt_deg=np.linspace(1,36,14), tilt_azim_deg=[0,9,18])   # 14*3 geometries

are both valid and need no other code changes.

Crystallography (composition, dominant reflections, zone axis, B-factor, default
energy grid) is looked up per material; the detector geometry defaults to the
2x2 Timepix3 quad. Only ``crystallography`` and the lightweight material
registry are imported here (no GPU), so this module is cheap to import and test.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from . import materials as _materials
from .materials import Layer, ScalarOrSeq
from .materials.crystal import CRYSTALS, dominant_reflections
from .materials.registry import pm as _pm

_CRYSTAL_PARAMS = _materials.CRYSTAL_PARAMS
MATERIAL_LABELS = _materials.MATERIAL_LABELS
pm = _pm

# ---- Timepix3 quad geometry (fixed hardware) --------------------------------
TIMEPIX3_PIXEL_PITCH_M = 55e-6
TIMEPIX3_CHIP_WIDTH_M = 256 * TIMEPIX3_PIXEL_PITCH_M
TIMEPIX3_DISTANCE_M = 0.4
TIMEPIX3_DTHETA_OBS_DEG = float(
    np.degrees(2 * np.arctan((TIMEPIX3_CHIP_WIDTH_M / 2) / TIMEPIX3_DISTANCE_M))
)
TIMEPIX3_DOMEGA_SR = float(TIMEPIX3_CHIP_WIDTH_M**2 / TIMEPIX3_DISTANCE_M**2)


def fmt_thickness(t_ang):
    """Compact human thickness label from Angstroms: 316A / 31.6nm / 17um / 1mm."""
    if t_ang < 1e2:
        return f"{t_ang:g}A"
    if t_ang < 1e4:
        return f"{t_ang / 10:g}nm"
    if t_ang < 1e7:
        return f"{t_ang / 1e4:g}um"
    return f"{t_ang / 1e7:g}mm"


# amorphous substrate number densities [1/Ang^3], from bulk mass density:
#   n_formula = rho[g/cc] * 0.602214 / M[g/mol], then * per-element stoichiometry
_SUBSTRATE_COMP = {
    "sio2": [("Si", 0.02205), ("O", 0.04410)],  # fused silica, rho=2.20, M=60.08
}


def substrate_composition(substrate):
    """Number-density composition [(element, n_per_Ang3), ...] for a substrate.
    Amorphous presets ('sio2') come from bulk density; a crystalline
    substrate already in CRYSTALS (e.g. 'silicon') uses its unit-cell density."""
    if substrate.lower() in _SUBSTRATE_COMP:
        return [(el, n) for el, n in _SUBSTRATE_COMP[substrate.lower()]]
    if substrate in CRYSTALS:
        from collections import Counter

        info = CRYSTALS[substrate]
        counts = Counter(el for el, _ in info["basis"])
        return [(el, c / info["V_cell"]) for el, c in counts.items()]
    raise ValueError(
        f"unknown substrate {substrate!r}; use one of {list(_SUBSTRATE_COMP)} "
        f"or a crystal key in {list(CRYSTALS)}"
    )


def stack_layers(film_composition, film_thickness_ang, stack):
    """Absorber stack [(z_top, z_bot, composition), ...] for a film at the
    entrance face (z=0..t_film) followed by each :class:`Layer` in ``stack``,
    boundaries accumulating downward. Attach as a case's ``abs_layers`` so
    emitted lines/brem are attenuated by the WHOLE stack (each crystalline
    layer still RADIATES via its own layer_radiator). Beam enters the film
    side; with positive tilt (toward-detector, Zhai convention; front exit)
    the lower layers sit BEHIND the emission and do not attenuate -- they
    bite the back-exit / transmission geometry. See
    docs/multilayer-materials.md."""
    t_f = float(film_thickness_ang)
    layers = [(0.0, t_f, [(el, float(n)) for el, n in film_composition])]
    z = t_f
    for lay in stack:
        t = float(lay.thickness_ang)
        layers.append((z, z + t, substrate_composition(lay.material)))
        z += t
    return layers


def film_on_substrate_layers(
    film_composition, film_thickness_ang, substrate, substrate_thickness_ang
):
    """The 2-layer special case of :func:`stack_layers`: one film on one
    substrate (kept as the stable public name for that common stack)."""
    return stack_layers(
        film_composition,
        film_thickness_ang,
        (Layer(substrate, substrate_thickness_ang),),
    )


def _radiator(cp, *, beam_uvw=None, azimuth_rad=None):
    """Coherent-radiator dict from a :func:`crystal_params` result ``cp``: crystal,
    hkl_list, B_ang2, and beam_uvw (``cp``'s own default unless overridden). The
    ``azimuth_rad`` key is included only when given -- a bare substrate radiator
    carries no azimuth of its own (:func:`substrate_radiator`); only a stack/film
    use of a radiator (:func:`layer_radiator`, :func:`build_cases`) does. The one
    constructor behind all three radiator-dict call sites."""
    rad: dict[str, Any] = dict(
        crystal=cp["crystal"],
        hkl_list=cp["hkl_list"],
        B_ang2=cp["B_ang2"],
        beam_uvw=cp["beam_uvw"] if beam_uvw is None else tuple(beam_uvw),
    )
    if azimuth_rad is not None:
        rad["azimuth_rad"] = float(azimuth_rad)
    return rad


def substrate_radiator(substrate, n_families=4):
    """Coherent-radiation crystal params for a substrate, or None if it radiates
    no lines. A CRYSTALLINE substrate (a CRYSTALS key, e.g. 'silicon') returns
    {crystal, hkl_list, B_ang2, beam_uvw} (from crystal_params) so it emits its
    own PXR/CBS; an AMORPHOUS preset ('sio2') returns None (it only
    absorbs + brems). This is the per-layer-radiation half of the multilayer
    feature -- the absorber stack (film_on_substrate_layers) is the other half.
    See docs/multilayer-materials.md."""
    if substrate.lower() in _SUBSTRATE_COMP:
        return None  # amorphous: no coherent lines
    if substrate in CRYSTALS:
        return _radiator(crystal_params(substrate, n_families))
    raise ValueError(
        f"unknown substrate {substrate!r}; use one of {list(_SUBSTRATE_COMP)} "
        f"or a crystal key in {list(CRYSTALS)}"
    )


def layer_radiator(layer: "Layer", n_families: int = 4):
    """Coherent radiator params for one stack :class:`Layer`, or None if the
    layer is amorphous. Same dict as :func:`substrate_radiator` plus the
    per-layer orientation: ``beam_uvw`` (overridden if the Layer sets one) and
    ``azimuth_rad`` (the Layer's in-plane rotation, radians)."""
    rad = substrate_radiator(layer.material, n_families)
    if rad is None:
        return None
    if layer.beam_uvw is not None:
        rad["beam_uvw"] = tuple(layer.beam_uvw)
    rad["azimuth_rad"] = float(np.deg2rad(layer.azimuth_deg))
    return rad


def crystal_params(material: str, n_families: int = 4) -> dict[str, Any]:
    """Fixed crystallography for a material: composition, the dominant
    reflections, the beam zone axis [uvw], the (isotropic) B-factor, and a
    sensible default photon-energy grid. Override the grid via Sweep.e_grid_eV."""
    if material not in _CRYSTAL_PARAMS:
        raise ValueError(f"unknown material {material!r} (have {list(_CRYSTAL_PARAMS)})")
    grid = _CRYSTAL_PARAMS[material]
    B_ang2 = grid["B_ang2"]
    hkl_list = grid.get("hkl_list")
    if hkl_list is None:
        hkl_list = dominant_reflections(material, n_families=n_families, B_ang2=B_ang2)
    else:
        hkl_list = list(hkl_list)  # fresh copy per call, like the derived path
    return dict(
        crystal=material,
        composition=substrate_composition(material),
        hkl_list=hkl_list,
        beam_uvw=grid["beam_uvw"],
        B_ang2=B_ang2,
        E_grid=grid["E_grid"],
    )


@dataclass
class Sweep:
    """One parameter sweep.

    Each of ``thickness_ang``, ``energy_keV``, ``tilt_deg``,
    ``tilt_azim_deg``, ``crystal_width_mm``, and ``crystal_height_mm`` is either
    a single number (fixed) or a sequence/array (swept); build_cases() takes the
    product. The transverse dimensions must be both ``None`` (the legacy
    infinite slab) or both strictly positive full dimensions in mm. The
    remaining fields are fixed setup that rarely changes per run.
    """

    material: str  # required: no default, so a Sweep can't silently load MoSe2
    thickness_ang: ScalarOrSeq = 2e4
    energy_keV: ScalarOrSeq = (30.0, 45.0, 60.0)
    tilt_deg: ScalarOrSeq = 30.0
    tilt_azim_deg: ScalarOrSeq = 0.0
    crystal_width_mm: ScalarOrSeq | None = None
    crystal_height_mm: ScalarOrSeq | None = None
    # fixed setup (single values) ------------------------------------------
    theta_obs_deg: float = 90.0
    n_families: int = 4
    # two independent photon-energy grids (None -> per-material defaults):
    #   E_grid_line : fine + NARROW; where the coherent lines are evaluated (the
    #       expensive sinc^2). Lines are kinematically capped at a few keV, so it
    #       need not extend past ~4 keV.
    #   E_grid_brem : coarse + WIDE; where the smooth bremsstrahlung is evaluated
    #       (cheap). Extend to 20-40 keV / the beam energy to model the full
    #       measured spectrum without inflating the line cost. Default spans the
    #       line start up to the highest beam energy at a 50 eV step.
    E_grid_line: np.ndarray | None = None
    E_grid_brem: np.ndarray | None = None
    e_grid_eV: np.ndarray | None = None  # deprecated: alias for E_grid_line
    dtheta_obs_deg: float | None = None  # None -> Timepix3 default
    domega_sr: float | None = None  # None -> Timepix3 default
    beam_uvw: tuple | None = None  # None -> per-material default
    # GPU memory knobs: segments per matmul in the spectrum / brem kernels (None
    # -> 40000 / 20000 defaults). Lower them (e.g. 4000) to cap peak GPU memory on
    # a busy or shared device; the cost is only a little extra loop overhead.
    spec_chunk: int | None = None
    brem_chunk: int | None = None
    # crystal mosaicity (the INITIAL ANALYTIC broadening; switchable per run).
    #   mosaic=False (default) -> OFF: perfect crystal, an exact no-op.
    #   mosaic=True            -> apply the per-crystal mosaic_fwhm_deg from
    #       crystal_structures.toml. Crystals without a value (diamond, silicon, the
    #       TMDs) stay perfect even with mosaic=True, so this is the "optional,
    #       excluding crystals which lack such data" switch -- HOPG has a value (0.8
    #       deg) and so DOES broaden when mosaic=True.
    #   mosaic_fwhm_deg        -> override the per-crystal value for ALL crystals in
    #       the sweep (e.g. HOPG ZYA 0.4 / ZYB 0.8 / ZYH 3.5), or supply one for a
    #       crystal that has none. Ignored unless mosaic=True.
    mosaic: bool = False
    mosaic_fwhm_deg: float | None = None
    # how the mosaic broadening (when mosaic=True) is applied:
    #   "analytic" (default) -> the cheap energy-shift Gaussian added in quadrature
    #       at detector convolution (results.store_result); the intrinsic spectrum is
    #       untouched, so one record re-broadens to any grade (plot_mosaic_comparison).
    #   "mc"                  -> the EXACT per-orientation average INSIDE mc_spectrum
    #       (broadens PXR+CBS, captures the amplitude variation + asymmetric lineshape
    #       and the mosaic yield change); the analytic term is then suppressed so the
    #       broadening is not double-counted. Costs mosaic_nodes**2 x the line hot loop
    #       (serial under CuPy). See docs/crystal-mosaicity.md.
    mosaic_route: str = "analytic"
    mosaic_nodes: int = 5  # Gauss-Hermite nodes/tilt-axis for mosaic_route="mc" (K=nodes^2)
    # film-on-substrate stack (optional; multilayer feature, docs/multilayer-materials.md).
    # substrate=None -> free-standing film (unchanged). Otherwise each case gets an
    # abs_layers stack that drives BOTH multilayer electron transport (substrate
    # backscatter into the film + substrate bremsstrahlung) AND cross-stack
    # self-absorption of the film's lines/brem. Every CRYSTALLINE layer radiates its
    # own coherent PXR/CBS lines (the film, and a crystalline substrate e.g.
    # "silicon" or "sapphire"), summed incoherently; an amorphous substrate ("sio2") adds
    # only brem + absorption.
    substrate: str | None = None  # "sio2" | a crystal key e.g. "silicon"/"sapphire"
    substrate_thickness_ang: float = 5e6  # 0.5 mm default
    # general N-layer stack under the film (mutually exclusive with substrate=;
    # substrate="x" is sugar for stack=(Layer("x", substrate_thickness_ang),)).
    # Each Layer carries its own thickness + orientation (beam_uvw, azimuth_deg),
    # so e.g. a few-layer film / thin a-SiO2 / thick crystalline Si device stack
    # is stack=(Layer("sio2", 2850), Layer("silicon", 5e6)).
    stack: Sequence[Layer] | None = None


def _seq(x):
    """Normalize a scalar-or-sequence into a 1-D float array, order preserved."""
    return np.atleast_1d(np.asarray(x, dtype=float))


def build_cases(sweep: Sweep, n_electrons=450, n_electrons_brem=100):
    """Expand a :class:`Sweep` into a list of run_case dicts (the Cartesian
    product over the swept thickness / tilt / azimuth / optional footprint, each
    crossed with every beam energy). ``crystal_width_mm`` and
    ``crystal_height_mm`` are full dimensions: both are ``None`` for the legacy
    infinite slab, otherwise both must be positive. Returns the ``cases`` list; preview it with
    :func:`geometry_table`."""
    cp = crystal_params(sweep.material, sweep.n_families)
    # line grid: fine + narrow (per-material default, E_grid_line, or the
    # deprecated e_grid_eV alias). brem grid: coarse + wide -- each case spans
    # up to that case's beam energy because brem cuts off at the particle energy.
    # E_grid_brem overrides the start/spacing, not the per-energy upper limit.
    line_src = sweep.E_grid_line if sweep.E_grid_line is not None else sweep.e_grid_eV
    line_grid = cp["E_grid"] if line_src is None else np.asarray(line_src, float)
    energies = _seq(sweep.energy_keV)
    if sweep.E_grid_brem is not None:
        brem_grid = np.asarray(sweep.E_grid_brem, float)
    else:
        brem_grid = np.arange(float(line_grid[0]), float(energies.max()) * 1e3 + 50.0, 50.0)  # type: ignore[reportIndexIssue,reportArgumentType]

    dtheta = TIMEPIX3_DTHETA_OBS_DEG if sweep.dtheta_obs_deg is None else sweep.dtheta_obs_deg
    domega = TIMEPIX3_DOMEGA_SR if sweep.domega_sr is None else sweep.domega_sr
    beam_uvw = cp["beam_uvw"] if sweep.beam_uvw is None else sweep.beam_uvw
    label = MATERIAL_LABELS.get(sweep.material, sweep.material)
    width_src, height_src = sweep.crystal_width_mm, sweep.crystal_height_mm
    if width_src is None and height_src is None:
        footprints = [(None, None)]
    elif width_src is None or height_src is None:
        raise ValueError("crystal_width_mm and crystal_height_mm must be supplied together")
    else:
        widths, heights = _seq(width_src), _seq(height_src)
        if not (
            np.all(np.isfinite(widths) & (widths > 0.0))
            and np.all(np.isfinite(heights) & (heights > 0.0))
        ):
            raise ValueError("crystal_width_mm and crystal_height_mm must be finite and positive")
        footprints = list(product(widths, heights))

    # crystal mosaicity (analytic, optional): None unless the run enables it AND the
    # crystal has a mosaic_fwhm_deg (or the Sweep overrides it). None -> perfect
    # crystal, so store_result adds no mosaic term (exact no-op).
    mosaic_deg = None
    if sweep.mosaic:
        mosaic_deg = (
            sweep.mosaic_fwhm_deg
            if sweep.mosaic_fwhm_deg is not None
            else CRYSTALS[cp["crystal"]].get("mosaic_fwhm_deg")
        )
    mosaic_fwhm_rad = float(np.deg2rad(mosaic_deg)) if mosaic_deg else None
    if sweep.mosaic_route not in ("analytic", "mc"):
        raise ValueError(f"mosaic_route must be 'analytic' or 'mc', got {sweep.mosaic_route!r}")
    # exact MC route: do the broadening INSIDE mc_spectrum and turn the analytic
    # store_result term OFF (mutually exclusive -- applying both double-counts).
    # Both are None when there is no mosaic to apply.
    mosaic_mc = mosaic_fwhm_rad is not None and sweep.mosaic_route == "mc"
    mosaic_analytic_rad = None if mosaic_mc else mosaic_fwhm_rad
    mosaic_mc_rad = mosaic_fwhm_rad if mosaic_mc else None

    def _triple(g):
        """(start, stop, step) so np.arange(*triple) reproduces grid g."""
        step = float(g[1] - g[0])
        return (float(g[0]), float(g[-1]) + step, step)

    line_triple = _triple(line_grid)
    brem_start = float(brem_grid[0])
    brem_step = float(brem_grid[1] - brem_grid[0])

    # normalize the substrate sugar onto the general stack (mutually exclusive)
    stack = sweep.stack
    if sweep.substrate is not None:
        if stack is not None:
            raise ValueError("give either substrate= or stack=, not both")
        stack = (Layer(sweep.substrate, sweep.substrate_thickness_ang),)

    cases = []
    for i_c, (thickness, tilt, azim, (width, height)) in enumerate(
        product(
            _seq(sweep.thickness_ang),
            _seq(sweep.tilt_deg),
            _seq(sweep.tilt_azim_deg),
            footprints,
        )
    ):
        name = f"{label} {fmt_thickness(thickness)} pol={tilt:g} az={azim:g}"
        abs_layers = None
        layer_radiators = None
        if stack is not None:
            name = f"{name} on {'+'.join(lay.material for lay in stack)}"
            abs_layers = stack_layers(cp["composition"], thickness, stack)
            # per-layer coherent radiators, aligned with abs_layers: the film (its
            # own crystal params) then one per stack Layer (crystal params + the
            # Layer's own orientation if crystalline, None if amorphous). This is
            # what lets a crystalline substrate emit its own PXR/CBS lines
            # (per-layer radiation, slice 3).
            layer_radiators = [
                # stack azimuths are relative to the film -> azimuth_rad=0.0
                _radiator(cp, beam_uvw=beam_uvw, azimuth_rad=0.0),
                *(layer_radiator(lay, sweep.n_families) for lay in stack),
            ]
        if width is not None:
            name = f"{name} footprint={width:g}x{height:g}mm"
        for i_e, E0 in enumerate(energies):
            cases.append(
                dict(
                    name=name,
                    crystal=cp["crystal"],
                    composition=cp["composition"],
                    hkl_list=cp["hkl_list"],
                    B_ang2=cp["B_ang2"],
                    E0_keV=float(E0),
                    thickness_ang=float(thickness),
                    crystal_width_mm=None if width is None else float(width),
                    crystal_height_mm=None if height is None else float(height),
                    E_grid=line_triple,  # legacy key (== line grid)
                    E_grid_line=line_triple,
                    E_grid_brem=(brem_start, float(E0) * 1e3 + brem_step, brem_step),
                    theta_obs_rad=np.deg2rad(sweep.theta_obs_deg),
                    tilt_deg=float(tilt),
                    tilt_azim_deg=float(azim),
                    beam_uvw=beam_uvw,
                    mosaic_fwhm_rad=mosaic_analytic_rad,  # analytic term (None if route="mc")
                    mosaic_mc_fwhm_rad=mosaic_mc_rad,  # exact MC route (None if route="analytic")
                    mosaic_mc_nodes=sweep.mosaic_nodes,
                    abs_layers=abs_layers,  # None -> single slab; else film-on-substrate stack
                    layer_radiators=layer_radiators,  # per-layer coherent radiators (None -> slab)
                    brem_file=None,
                    Ne=n_electrons,
                    Ne_brem=n_electrons_brem,
                    seed=1000 * i_c + 10 * i_e + 1,
                    spec_chunk=sweep.spec_chunk,  # GPU rows/matmul (None -> run_case default)
                    brem_chunk=sweep.brem_chunk,
                    # used downstream (detector model / unit scaling); ignored by run_case:
                    dtheta_obs_rad=np.deg2rad(dtheta),
                    domega_sr=domega,
                )
            )
    return cases


def geometry_table(cases):
    """A one-row-per-config DataFrame summarizing the geometry of a case list,
    for a quick sanity check before running."""
    import pandas as pd

    def _grid(triple):
        """'0.05-4 keV @ 3 eV' label from a (start, stop, step) grid triple."""
        if triple is None:
            return "-"
        s0, s1, ds = triple
        return f"{s0 / 1e3:g}-{(s1 - ds) / 1e3:g} keV @ {ds:g} eV"

    rows, seen = [], set()
    for c in cases:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        same = [k for k in cases if k["name"] == c["name"]]
        rows.append(
            {
                "config": c["name"],
                "beam_uvw": c["beam_uvw"],
                "refl": len(c["hkl_list"]),
                "t [um]": c["thickness_ang"] / 1e4,
                "width [mm]": c.get("crystal_width_mm"),
                "height [mm]": c.get("crystal_height_mm"),
                "polar [deg]": round(c["tilt_deg"], 2),
                "azim [deg]": round(c["tilt_azim_deg"], 2),
                "energies [keV]": [k["E0_keV"] for k in same],
                "line grid": _grid(c.get("E_grid_line", c["E_grid"])),
                "brem grid": _grid(c.get("E_grid_brem")),
                "theta_obs [deg]": round(np.degrees(c["theta_obs_rad"]), 1),
                "dOmega [sr]": c["domega_sr"],
            }
        )
    return pd.DataFrame(rows)
