"""Resolve a singleton catalog pixel scene without checkpoint or cache I/O."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np

from .observation import filter_from_config, physical_detector_from_config


def _one(values, label, command_name):
    values = np.asarray(values)
    if values.size != 1:
        raise ValueError(
            f"{command_name} requires profile {label!r} to resolve one value; "
            "use a profile with singleton thickness, energy, polar, and azimuth grids"
        )
    return float(values.item())


def resolve_pixel_scene(
    catalog, material, profile_name, detector_id=None, *, command_name="material simulate"
):
    """Resolve one profile case without constructing a Sweep or checkpoint."""
    from pyrite.campaign.longitudinal import LongitudinalDistribution
    from pyrite.campaign.model import Beam, Numerics
    from pyrite.campaign.sweep import beam_replace, target_from_flat
    from pyrite.detectors import EnergyBins
    from pyrite.instrument import PixelScorer
    from pyrite.montecarlo.transverse import TransverseDistribution

    spec = catalog.material(material)
    membership = catalog.profile_materials(profile_name)
    if membership is not None and material not in membership:
        raise ValueError(f"material {material!r} is not a member of profile {profile_name!r}")
    scan = spec.scan
    energy = _one(scan.energy_keV, profile_name, command_name)
    beam = Beam(energy_keV=energy)
    fields = catalog.profile_beam(profile_name)
    if fields:
        changes = dict(fields)
        if (longitudinal := changes.get("longitudinal")) is not None:
            if not isinstance(longitudinal, Mapping):
                raise TypeError("profile longitudinal policy must be a mapping")
            changes["longitudinal"] = LongitudinalDistribution(
                **cast(dict[str, Any], dict(longitudinal))
            )
        if (transverse := changes.get("transverse")) is not None:
            if not isinstance(transverse, Mapping):
                raise TypeError("profile transverse policy must be a mapping")
            changes["transverse"] = TransverseDistribution(**cast(dict[str, Any], dict(transverse)))
            changes.setdefault("transverse_fwhm_x_mm", None)
            changes.setdefault("transverse_fwhm_y_mm", None)
        if changes.get("source") == "gpt_gdf":
            changes.setdefault("transverse_fwhm_x_mm", None)
            changes.setdefault("transverse_fwhm_y_mm", None)
        beam = beam_replace(beam, **changes)
    target = target_from_flat(
        spec.crystal_key,
        thickness_ang=_one(scan.thickness_ang, profile_name, command_name),
        tilt_deg=_one(scan.tilt_deg, profile_name, command_name),
        tilt_azim_deg=_one(scan.tilt_azim_deg, profile_name, command_name),
        substrate=spec.substrate,
        stack=spec.stack or None,
    )
    detectors = catalog.profile_detector_set(profile_name)
    pixel_ids = [name for name, row in detectors.items() if "distance_mm" in row]
    if detector_id is None and len(pixel_ids) == 1:
        detector_id = pixel_ids[0]
    if detector_id is None:
        raise ValueError(
            f"{command_name} requires one pixel detector or --detector ID; "
            f"available pixel detectors: {', '.join(pixel_ids) or 'none'}"
        )
    if detector_id not in pixel_ids:
        raise ValueError(
            f"detector {detector_id!r} is not a pixel detector in profile {profile_name!r}; "
            f"available: {', '.join(pixel_ids) or 'none'}"
        )
    physical = detectors[detector_id]
    from pyrite.campaign.observation import resolve_profile_observation

    observation = resolve_profile_observation(catalog, profile_name, detector_id=detector_id)
    detector = (
        physical_detector_from_config(physical) if observation is None else observation.detector
    )
    scorer_row = cast("dict[str, Any] | None", physical.get("scorer"))
    scorer = (
        observation.scorer
        if observation is not None
        else PixelScorer()
        if scorer_row is None
        else PixelScorer(
            angular_shape=tuple(scorer_row["angular_shape"]),
            reconstruction=scorer_row.get("reconstruction", "nearest_tile"),
        )
    )
    detector = replace(
        detector,
        energy_bins=EnergyBins(
            line=scan.E_grid_line,
            line_by_energy=scan.E_grid_line_by_energy,
            brem=scan.E_grid_brem,
        ),
    )
    filters = tuple(
        filter_from_config(row) for row in catalog.profile_filters.get(profile_name, ())
    )
    transport = catalog.profile_numerics(profile_name) or {}
    straggling = transport.get("straggling", False)
    if not isinstance(straggling, bool):
        raise TypeError("profile straggling must be a bool")
    energy_model = transport.get("energy_model", "frozen")
    if energy_model not in {"frozen", "midpoint"}:
        raise ValueError("profile energy_model must be 'frozen' or 'midpoint'")
    max_dE_frac = transport.get("max_dE_frac", 0.0)
    if isinstance(max_dE_frac, bool) or not isinstance(max_dE_frac, (int, float)):
        raise TypeError("profile max_dE_frac must be a number")
    n_electrons = (
        450 if scan.n_electrons is None else int(_one(scan.n_electrons, profile_name, command_name))
    )
    n_electrons_brem = (
        100
        if scan.n_electrons_brem is None
        else int(_one(scan.n_electrons_brem, profile_name, command_name))
    )
    return (
        beam,
        target,
        detector,
        filters,
        scorer,
        None if observation is None else observation.acquisition,
        Numerics(
            n_electrons=n_electrons,
            n_electrons_brem=n_electrons_brem,
            straggling=straggling,
            energy_model=cast(Literal["frozen", "midpoint"], energy_model),
            max_dE_frac=float(max_dE_frac),
            inelastic_model=cast(
                Literal["auto", "continuous", "shell-soft-hard"],
                transport.get("inelastic_model", "auto"),
            ),
            inelastic_cutoff_eV=cast(float | None, transport.get("inelastic_cutoff_eV")),
            secondary_threshold_eV=cast(float | None, transport.get("secondary_threshold_eV")),
            elastic_model=cast(Literal["mott", "elsepa"], transport.get("elastic_model", "elsepa")),
            bremsstrahlung_model=cast(
                Literal["auto", "eedl", "bremslib"], transport.get("bremsstrahlung_model", "auto")
            ),
            radiative_model=cast(
                Literal["uncoupled", "bremslib-soft-hard"],
                transport.get("radiative_model", "uncoupled"),
            ),
            radiative_cutoff_eV=cast(float | None, transport.get("radiative_cutoff_eV")),
            pair_production_model=cast(
                Literal["penelope-2024"] | None, transport.get("pair_production_model")
            ),
            positron_transport=bool(transport.get("positron_transport", False)),
            atomic_electron_deflection=cast(
                Literal["kawrakow", "none"],
                transport.get("atomic_electron_deflection", "kawrakow"),
            ),
            precision=catalog.profile_precision(profile_name),
        ),
        catalog.profile_emission(profile_name) or "incoherent",
        detector_id,
    )
