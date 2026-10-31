"""Resolve a singleton catalog pixel scene without checkpoint or cache I/O."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from .._energy_grid_encoding import encode_energy_grid
from .._numerics import _emission_selection
from .observation import filter_from_config, physical_detector_from_config

if TYPE_CHECKING:
    from .._numerics import Numerics, Physics
    from ..observations.spec import Observation
    from .model import Scene


def _one(values, label, command_name):
    values = np.asarray(values)
    if values.size != 1:
        raise ValueError(
            f"{command_name} requires profile {label!r} to resolve one value; "
            "use a profile with singleton thickness, energy, polar, and azimuth grids"
        )
    return float(values.item())


def _band_at_energy(grid, energy_keV):
    """Stop a uniform catalog band one step past the beam energy (#256).

    The public ``EnergyBins.brem`` this scene carries is exact (#815), so the
    catalog band is resolved against its single beam energy here.
    """
    encoded = encode_energy_grid(grid)
    if not isinstance(encoded, tuple):
        return grid
    start, _, step = encoded
    return np.arange(start, float(energy_keV) * 1e3 + step, step)


def resolve_pixel_scene(
    catalog, material, profile_name, detector_id=None, *, command_name="run --ephemeral"
):
    """Resolve one profile case without constructing a Sweep or checkpoint."""
    from pyrite.campaign.longitudinal import LongitudinalDistribution
    from pyrite.campaign.model import Beam, Numerics, Physics
    from pyrite.campaign.sweep import BeamSpec, beam_replace, target_from_flat
    from pyrite.config import AngularSampling as PixelScorer
    from pyrite.detectors import EnergyBins
    from pyrite.montecarlo.transverse import TransverseDistribution

    spec = catalog.material(material)
    membership = catalog.profile_materials(profile_name)
    if membership is not None and material not in membership:
        raise ValueError(f"material {material!r} is not a member of profile {profile_name!r}")
    scan = spec.scan
    energy = _one(scan.energy_keV, profile_name, command_name)
    beam = BeamSpec(energy_keV=energy)
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
    beam = Beam.from_spec(beam)
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
        scorer=scorer,
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
            energy_bins=EnergyBins(
                cxr=scan.E_grid_line,
                cxr_by_energy=scan.E_grid_line_by_energy,
                brem=_declared_brem_band(scan, energy),
            ),
            cxr_trials=n_electrons,
            brem_trials=n_electrons_brem,
            energy_model=cast(Literal["frozen", "midpoint"], energy_model),
            max_dE_frac=float(max_dE_frac),
            channeling_basis_size=int(transport.get("channeling_basis_size", 16)),
            channeling_bloch_cells=int(transport.get("channeling_bloch_cells", 16)),
            channeling_dz_ang=float(transport.get("channeling_dz_ang", 250.0)),
            channeling_zone_max_ang=cast(float | None, transport.get("channeling_zone_max_ang")),
            precision=catalog.profile_precision(profile_name),
        ),
        Physics(
            emission=_emission_selection(catalog.profile_emission(profile_name) or "incoherent"),
            straggling=straggling,
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
            channeling_model=cast(
                Literal["off", "planar", "axial"], transport.get("channeling_model", "off")
            ),
            channeling_plane=cast(tuple[int, int, int] | None, transport.get("channeling_plane")),
            channeling_axial=cast(Mapping[str, object] | None, transport.get("channeling_axial")),
            transition_radiation=cast(
                Literal["off", "history", "bunch", "two-beam"],
                transport.get("transition_radiation", "off"),
            ),
            transition_lattice_origin_ang=cast(
                tuple[float, float, float] | None, transport.get("transition_lattice_origin_ang")
            ),
            positron_transport=bool(transport.get("positron_transport", False)),
            atomic_electron_deflection=cast(
                Literal["kawrakow", "none"],
                transport.get("atomic_electron_deflection", "kawrakow"),
            ),
            shell_ionization_model=cast(
                Literal["eedl", "bote-salvat"],
                transport.get("shell_ionization_model", "eedl"),
            ),
        ),
        detector_id,
    )


@dataclass(frozen=True)
class ProfileSimulation:
    """Internal named pixel-run adapter with independent physical/observation inputs.

    Catalog tables and legacy tuple consumers remain unchanged; execution
    still passes through the shared simulation resolver.
    """

    scene: "Scene"
    physics: "Physics"
    numerics: "Numerics"
    observation: "Observation"
    detector_id: str


def compose_pixel_scene(components: tuple[Any, ...]) -> ProfileSimulation:
    """Split a resolved singleton profile into Scene, models, and Observation.

    Detector response, acquisition, and numerical scoring belong to Observation,
    not to the physical scene. This is value-preserving for historical profiles.
    """
    from ..instrument import PlanarDetector
    from ..observations import Observation
    from .model import Scene

    beam, target, detector, filters, scorer, acquisition, numerics, physics, detector_id = (
        components
    )
    if not isinstance(detector, PlanarDetector):
        raise TypeError("pixel profile composition requires a PlanarDetector")
    physical = replace(detector, response=None, scorer=None)
    return ProfileSimulation(
        scene=Scene(beam, target, physical, filters=filters),
        physics=physics,
        numerics=numerics,
        observation=Observation(
            detector="detector",
            response=detector.response,
            acquisition=acquisition,
            sampling=scorer,
            name=detector_id,
        ),
        detector_id=detector_id,
    )


def _declared_brem_band(scan: Any, energy: float) -> Any:
    """The declared continuum band at ``energy``; ``None`` resolves the automatic grid (#278)."""
    return _band_at_energy(scan.E_grid_brem, energy) if scan.E_grid_brem_declared else None
