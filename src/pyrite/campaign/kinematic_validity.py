"""Case-local photon Born diagnostics, separate from transport identity."""

import math
import warnings
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np

from ..materials.crystal import (
    CRYSTALS,
    HBARC_EV_ANG,
    KINEMATIC_DYN_LIMIT,
    KINEMATIC_EXTINCTION_RATIO_LIMIT,
    beta_from_Ee,
    crystal_absorption_length_ang,
    kinematic_validity,
    reciprocal_g_vector,
    reflection_coupling_tables,
)
from ..montecarlo.geometry import _orientation_R, tilted_geometry


class KinematicValidityWarning(UserWarning):
    """A resolved reflection trips the conservative photon Born screen."""


@lru_cache(maxsize=4096)
def _radiator_optics(
    crystal,
    hkls,
    B_ang2,
    energy_keV,
    theta,
    tilt,
    tilt_azim,
    beam_uvw,
    surface_hkl,
    azimuth,
    miscut,
):
    """Central-ray vacuum screen, using Zhai (2025) SI Eqs. (3) and (9).

    Assumes a homogeneous radiator with one incident energy/direction. Uses
    the existing audit's positive-g resonance and plus-g detuning convention;
    no refractive or domain-size correction. Zero positive resonances leaves
    only explicit skipped-reflection records. Thickness-independent optical inputs are reused across a thickness sweep.
    Scalar thresholds and length limits are applied by _radiator_audit.

    Validation: kinematic-validity-envelope
    """
    if not hkls:
        return ()
    lattice = CRYSTALS[crystal]["lattice"]
    g = np.array([reciprocal_g_vector(hkl, lattice)[0] for hkl in hkls])
    rotation = _orientation_R(lattice, beam_uvw, azimuth, miscut, surface_hkl)
    if rotation is not None:
        g = g @ rotation.T
    beam, direction = tilted_geometry(theta, tilt, tilt_azim)
    beta = beta_from_Ee(energy_keV * 1e3)
    # Same positive-g resonance and plus-g detuning as the offline audit.
    # Vacuum, central incident ray only: not a segment-level validity proof.
    omega = beta * (g @ beam) / (1.0 - beta * float(beam @ direction))
    energies = HBARC_EV_ANG * omega
    positive = energies > 0
    indices = np.flatnonzero(positive)
    coupling = np.full(len(hkls), np.nan)
    absorption = np.full(len(hkls), np.nan)
    if indices.size:
        # One optical-data read per element for all resonances. The existing
        # library table is hkl x energy; its diagonal pairs each reflection
        # with its own central-ray resonance, without energy-grid interpolation.
        real, imag, _, _ = reflection_coupling_tables(
            crystal,
            [hkls[i] for i in indices],
            energies[positive],
            B_ang2,
            True,
        )
        coupling[positive] = np.hypot(real.diagonal(), imag.diagonal())
        absorption[positive] = crystal_absorption_length_ang(crystal, energies[positive])
    detuning = np.einsum("ij,ij->i", g, g) + 2.0 * omega * (g @ direction)
    records = []
    for i, hkl in enumerate(hkls):
        record = {"hkl": hkl}
        if not positive[i]:
            record["status"] = "no_positive_resonance"
        elif not np.isfinite(coupling[i]) or np.isnan(absorption[i]):
            record.update(status="outside_optical_data", photon_energy_eV=float(energies[i]))
        else:
            record.update(
                status="evaluated",
                photon_energy_eV=float(energies[i]),
                chi_abs=float(coupling[i]),
                detuning_inv_ang2=float(detuning[i]),
                absorption_length_ang=float(absorption[i]),
            )
        records.append(record)
    # Strict JSON: infinite diagnostics use null; decision flags retain meaning.
    return tuple(records)


def _radiator_audit(
    crystal,
    hkls,
    B_ang2,
    energy_keV,
    thickness_ang,
    theta,
    tilt,
    tilt_azim,
    beam_uvw,
    surface_hkl,
    azimuth,
    miscut,
):
    """Apply the ledgered scalar photon screen to cached optical inputs.

    Source, assumptions and limits: materials.crystal.kinematic_validity.
    Only extinction_ratio depends on thickness; reuse does not alter optics.

    Validation: kinematic-validity-envelope
    """
    optics = _radiator_optics(
        crystal,
        hkls,
        B_ang2,
        energy_keV,
        theta,
        tilt,
        tilt_azim,
        beam_uvw,
        surface_hkl,
        azimuth,
        miscut,
    )
    records = []
    for optical in optics:
        record = dict(optical)
        if optical["status"] == "evaluated":
            values = kinematic_validity(
                optical["photon_energy_eV"],
                optical["chi_abs"],
                optical["detuning_inv_ang2"],
                thickness_ang,
                optical["absorption_length_ang"],
            )
            record.update(
                thickness_ang=thickness_ang,
                **{
                    key: None if isinstance(value, float) and not math.isfinite(value) else value
                    for key, value in values.items()
                },
            )
        records.append(record)
    return records


def case_kinematic_validity(case: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate selected central-ray reflections for every crystalline layer.

    Reports a vacuum screening envelope, without claiming validity over the
    detector acceptance, energy spread, mosaic domains or scattered flights.
    Positive/negative reflections with no positive resonance remain visible.
    Does not warn or alter case payloads; safe for provenance/cache readers.
    """
    radiators = case.get("layer_radiators")
    if radiators is None:
        layers = [(None, case, case["thickness_ang"])]
    else:
        layers = [
            (index, radiator, case["abs_layers"][index][1] - case["abs_layers"][index][0])
            for index, radiator in enumerate(radiators)
            if radiator is not None
        ]
    reflections = []
    for layer, radiator, thickness in layers:
        hkls = tuple(dict.fromkeys(tuple(hkl) for hkl in radiator["hkl_list"]))
        records = _radiator_audit(
            radiator["crystal"],
            hkls,
            float(radiator["B_ang2"]),
            float(case["E0_keV"]),
            float(thickness),
            float(case["theta_obs_rad"]),
            float(np.deg2rad(case.get("tilt_deg", 0.0))),
            float(np.deg2rad(case.get("tilt_azim_deg", 0.0))),
            _indices(radiator.get("beam_uvw")),
            _indices(radiator.get("surface_hkl")),
            float(radiator.get("azimuth_rad", case.get("azimuth_rad", 0.0))),
            _indices(radiator.get("recip_miscut_rad", case.get("recip_miscut_rad"))),
        )
        reflections.extend(
            {"layer": layer, "crystal": radiator["crystal"], **record, "hkl": list(record["hkl"])}
            for record in records
        )
    return {
        "model": "central-ray-vacuum-born-screen-v1",
        "dyn_limit": KINEMATIC_DYN_LIMIT,
        "extinction_ratio_limit": KINEMATIC_EXTINCTION_RATIO_LIMIT,
        "reflections": reflections,
    }


def _indices(value):
    return None if value is None else tuple(value)


def warn_kinematic_validity(case: Mapping[str, Any]) -> None:
    """Warn once per selected case/layer/reflection in this lowering call."""
    audit = case_kinematic_validity(case)
    for reflection in audit["reflections"]:
        reasons = []
        if reflection.get("photon_mixing"):
            reasons.append(f"DYN={reflection['dyn']} >= {audit['dyn_limit']}")
        if reflection.get("extinction_reachable"):
            reasons.append("extinction scale reachable (Bragg envelope)")
        if reflection.get("photon_mixing") and reflection.get("extinction_reachable"):
            warnings.warn(
                f"kinematic PXR/CBS screen: {case['name']} at {case['E0_keV']:g} keV, "
                f"layer={reflection['layer']}, {reflection['crystal']} "
                f"hkl={tuple(reflection['hkl'])}: {'; '.join(reasons)}; "
                "first-order spectra may require dynamical diffraction",
                KinematicValidityWarning,
                stacklevel=2,
            )
