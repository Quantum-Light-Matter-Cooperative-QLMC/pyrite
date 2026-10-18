"""Beam phase-space case keys for ``build_cases``.

The legacy flat beam keys join a case only where they diverge from the
point-bunch/default-source configuration, and declarative transverse and
longitudinal policies resolve per case point, so default case payloads and
checkpoint matching stay bit-for-bit.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from ..materials import CATALOG
from ..montecarlo.bunch_front import validate_front_source
from ..montecarlo.transverse import resolve_transverse_distribution
from .longitudinal import resolve_longitudinal_distribution

if TYPE_CHECKING:
    from ..montecarlo.gdf import GDFBeam
    from .geometry import LoweredTarget, ResolvedCut
    from .sweep import BeamSpec, Sweep


def _gdf_source_keys(beam: BeamSpec, gdf: GDFBeam) -> dict[str, Any]:
    """The case's GPT source record: path, selection, normalization and content hash."""
    return {
        "path": str(beam.gdf_path),
        "time_s": gdf.time_s,
        "tolerance_s": beam.gdf_time_tolerance_s,
        "normalization": beam.gdf_normalization,
        "z_origin_m": beam.gdf_z_origin_m,
        "sha256": gdf.sha256,
        **({"shape_only": True} if beam.gdf_shape_only else {}),
        **(
            {
                "screen_position_m": gdf.screen_position_m,
                "screen_tolerance_m": beam.gdf_screen_tolerance_m,
            }
            if gdf.screen_position_m is not None
            else {}
        ),
    }


def _reject_mixed_beam_policies(b: BeamSpec) -> None:
    """Refuse a declarative beam policy mixed with its legacy fields."""
    if b.longitudinal is not None and (
        b.bunch_length_fs is not None or b.long_offsets_fs is not None or b.long_shape != "gaussian"
    ):
        raise ValueError(
            "longitudinal policy is incompatible with legacy bunch_length_fs, "
            "long_shape, and long_offsets_fs fields"
        )
    if b.transverse is not None and (
        b.transverse_fwhm_x_mm is not None or b.transverse_fwhm_y_mm is not None
    ):
        # Spot FWHM, divergence and emittance are three numbers for two
        # independent moments plus a correlation. Resolving that by precedence
        # would hand back a plausible beam that is not the requested one, so it
        # is an error. Note the FWHM fields DEFAULT to a 1 mm spot: a transverse
        # policy has to clear them explicitly (BeamSpec.with_transverse does).
        raise ValueError(
            "transverse policy is incompatible with the legacy transverse_fwhm_x_mm "
            "and transverse_fwhm_y_mm fields; clear them "
            "(they default to a 1 mm spot) or use BeamSpec.with_transverse"
        )


def beam_case_keys(sweep: Sweep, gdf: GDFBeam | None) -> dict[str, Any]:
    """Beam phase-space fields threaded onto every case (constant across the product).

    The legacy ``beam_fwhm_mm`` key stays == the x-plane FWHM so a default
    isotropic spot is bit-for-bit; the elliptical ``beam_fwhm_y_mm`` and the
    longitudinal bunch / pulse-rate keys join ONLY when they actually diverge
    from the point-bunch/default-source configuration, so existing case
    payloads and checkpoint matching remain stable.
    """
    assert sweep.target is not None
    b = sweep.beam
    front = validate_front_source(
        b.bunch_front_tilt,
        transverse=b.transverse,
        energy_spread=b.energy_spread_frac,
        source=b.source,
        longitudinal=b.longitudinal,
        offsets=b.long_offsets_fs,
        shape=b.long_shape,
        groove=sweep.target._entrance_face(),
    )
    fwhm_x = None if b.transverse_fwhm_x_mm is None else float(b.transverse_fwhm_x_mm)
    fwhm_y = None if b.transverse_fwhm_y_mm is None else float(b.transverse_fwhm_y_mm)
    beam_case: dict[str, Any] = {"beam_fwhm_mm": fwhm_x}
    if front is not None:
        beam_case["bunch_front_tilt"] = asdict(front)
    if gdf is not None:
        beam_case["gdf_source"] = _gdf_source_keys(b, gdf)
    if b.bunch_charge_pc != 1.0 or b.rep_rate_hz != 5000.0:
        beam_case["bunch_charge_pc"] = float(b.bunch_charge_pc)
        beam_case["rep_rate_hz"] = float(b.rep_rate_hz)
    if gdf is not None and b.gdf_normalization == "gdf_charge":
        assert gdf.absolute_charge_c is not None
        beam_case["bunch_charge_pc"] = gdf.absolute_charge_c * 1e12
        beam_case["rep_rate_hz"] = float(b.rep_rate_hz)
    if fwhm_y != fwhm_x:
        beam_case["beam_fwhm_y_mm"] = fwhm_y
    if b.energy_spread_frac is not None:
        beam_case["energy_spread_frac"] = float(b.energy_spread_frac)
    _reject_mixed_beam_policies(b)
    if b.bunch_length_fs is not None or b.long_offsets_fs is not None:
        beam_case["long_shape"] = b.long_shape
        if b.bunch_length_fs is not None:
            beam_case["bunch_length_fs"] = float(b.bunch_length_fs)
        if b.long_offsets_fs is not None:
            beam_case["long_offsets_fs"] = tuple(float(x) for x in b.long_offsets_fs)
    return beam_case


def beam_distribution_keys(
    sweep: Sweep, cp: dict, orientation: ResolvedCut, geometry: LoweredTarget, energy_keV: float
) -> dict[str, Any]:
    """Resolve declarative longitudinal/transverse policies at one case point.

    They resolve only once material, energy, geometry, and catalog reflection
    provenance are known; sampled offsets never enter case identity.
    """
    assert sweep.target is not None and sweep.detector is not None
    b = sweep.beam
    keys: dict[str, Any] = {}
    resolved_transverse = None
    if b.transverse is not None:
        resolved_transverse = asdict(
            resolve_transverse_distribution(b.transverse, energy_keV=energy_keV)
        )
    if b.longitudinal is not None:
        if b.longitudinal.kind != "gaussian" and orientation.recip_miscut_rad is not None:
            raise ValueError(
                "targeted longitudinal modes assume g parallel to the surface "
                "normal; a miscut CrystalCut is unsupported"
            )
        crystal_spec = CATALOG.crystal(cp["crystal"])
        keys["longitudinal_distribution"] = asdict(
            resolve_longitudinal_distribution(
                b.longitudinal,
                material=sweep.target.material,
                crystal=cp["crystal"],
                lattice=crystal_spec.lattice,
                hkl_list=cp["hkl_list"],
                surface_hkl=orientation.surface_hkl,
                beam_uvw=orientation.beam_uvw,
                energy_keV=energy_keV,
                theta_obs_deg=float(sweep.detector.observation_angle_deg),
                tilt_deg=geometry.tilt_deg,
            )
        )
    if resolved_transverse is not None:
        keys["transverse_distribution"] = resolved_transverse
    return keys
