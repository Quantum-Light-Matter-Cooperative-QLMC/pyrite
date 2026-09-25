"""Fidelity presets and reproducible dataset identities.

Profiles are policies applied after catalog material lookup.  ``full`` is an
exact compatibility policy: it leaves catalog grids unchanged and retains the
production electron counts.  ``survey`` is provisional and intentionally
smaller.  Every resolved run can be serialized into a deterministic identity,
so differently resolved variants never silently resume into one dataset.
"""

import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .._numerics import (
    PROFILE_NUMERICS_KEYS,
    SAMPLING_KEYS,
    Convergence,
    Numerics,
    electron_counts,
    validate_profile_numerics,
)
from ..detectors import EnergyBins
from ..montecarlo.case import Case
from ..montecarlo.spectrum import BREMSSTRAHLUNG_MODEL, CHARACTERISTIC_MODEL
from ..montecarlo.spectrum.brem_bremslib import BREMSSTRAHLUNG_BREMSLIB_MODEL
from ..montecarlo.transport import STOPPING_MODEL
from ..results import EmissionMode, Settings
from .sweep import (
    Sweep,
    beam_replace,
    build_cases,
    crystal_params,
    target_flat_fields,
    target_replace,
)

FIDELITY_NAMES = ("full", "survey")
DATASET_IDENTITY_SCHEMA = "cxr.dataset-identity.v1"
CASE_CONTENT_KEY_SCHEMA = "cxr.case-content-key.v1"
CURRENT_IDENTITY_VERSION = 1

NUMERICS_GROUPS = (
    (
        "sampling",
        (
            ("n_electrons", "line electrons"),
            ("n_electrons_brem", "bremsstrahlung electrons"),
        ),
    ),
    (
        "convergence",
        (
            ("n_families", "reflection families"),
            ("max_reflections", "maximum reflections"),
            ("mosaic_nodes", "mosaic nodes"),
            ("mosaic_route", "mosaic route"),
        ),
    ),
    (
        "transport",
        (
            ("energy_model", "energy model"),
            ("max_dE_frac", "maximum fractional energy loss"),
            ("straggling", "straggling"),
            ("inelastic_model", "inelastic model"),
            ("inelastic_cutoff_eV", "inelastic cutoff (eV)"),
            ("elastic_model", "elastic model"),
        ),
    ),
)


@dataclass(frozen=True)
class NumericsResolution:
    """Explicit and effective result-affecting numerics with provenance."""

    fidelity: str
    explicit: Mapping[str, object]
    effective: Mapping[str, object]
    sources: Mapping[str, str]

    def groups(self) -> list[dict[str, object]]:
        """Return stable, user-facing groups for CLI and JSON output."""
        return [
            {
                "name": group,
                "fields": [
                    {
                        "key": key,
                        "label": label,
                        "explicit": self.explicit.get(key),
                        "effective": self.effective[key],
                        "source": self.sources[key],
                    }
                    for key, label in fields
                ],
            }
            for group, fields in NUMERICS_GROUPS
        ]


# Case-dict fields excluded from the per-case content key. EVERYTHING else in a
# resolved ``build_cases`` case dict determines the stored spec/brem arrays and
# stays in the key (rule when unsure: keep -- over-inclusion only loses reuse,
# under-inclusion serves wrong numbers). These excluded fields are label / perf /
# post-processing / flux-scale only:
#   - ``name`` and profile/variant fields: presentation/provenance labels.
#   - ``domega_sr``: detector solid angle. ``store_result`` folds it into the
#     derived ``scale`` scalar (``domega_sr * PER_NA``); the spec/brem arrays are
#     untouched. Recomputed per requester on reuse, so a reused blob's arrays are
#     correct for any solid angle.
#   - ``rep_rate_hz`` / ``bunch_charge_pc``: pulse cadence/charge. ``store_result``
#     folds them into the derived ``source_current_na`` scalar only; never into
#     the arrays.
#   - ``beam_current_na``: legacy reporting fallback, never a kernel input.
#   - ``mosaic_fwhm_rad``: analytic post-processing broadening in
#     ``store_result``. The MC route remains keyed by ``mosaic_mc_fwhm_rad``.
#   - ``spec_chunk`` / ``brem_chunk``: GPU batch sizes. Chunk-invariant per the
#     goldens, and only ever set on ``-p/--perf`` runs -- which bypass the cache
#     entirely (perf implies ``--no-cache``) -- so the cache never mixes chunks.
# ``catalog_profile`` / ``variant`` / ``fidelity`` are NOT case-dict fields today
# -- the profile name never reaches a case, which is exactly what lets two
# differently-named profiles' shared case hash equal and reuse one blob. They are
# listed anyway as belt-and-suspenders: should any future path stamp a label onto
# a case dict, it must still never perturb the content key.
_CONTENT_KEY_DENYLIST = frozenset(
    {
        "name",
        "domega_sr",
        "beam_current_na",
        "rep_rate_hz",
        "bunch_charge_pc",
        "mosaic_fwhm_rad",
        "spec_chunk",
        "brem_chunk",
        "catalog_profile",
        "variant",
        "fidelity",
    }
)


def _centered_sample(values: Any, limit: int | None) -> Any:
    """Keep at most ``limit`` values from a centered, narrower input span."""
    if limit is None:
        return values
    array = np.atleast_1d(np.asarray(values))
    if array.size <= limit:
        return values
    indices = np.linspace(0.2 * (array.size - 1), 0.8 * (array.size - 1), limit)
    indices = np.unique(np.rint(indices).astype(int))
    return array[indices]


def _coarsen_grid(values: Any, factor: int, span_fraction: float) -> np.ndarray:
    """Crop a uniform/nonuniform grid around its center, then decimate it."""
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if array.size <= 2:
        return array.copy()
    keep = max(2, int(np.ceil(array.size * span_fraction)))
    start = (array.size - keep) // 2
    cropped = array[start : start + keep]
    coarse = cropped[::factor]
    if coarse[-1] != cropped[-1]:
        coarse = np.append(coarse, cropped[-1])
    return coarse


@dataclass(frozen=True)
class FidelityPreset:
    """Independent settings and grid-reduction policy for one fidelity preset."""

    name: str
    n_electrons: int
    n_electrons_brem: int
    max_energies: int | None = None
    max_thicknesses: int | None = None
    max_tilts: int | None = None
    max_azimuths: int | None = None
    n_families: int | None = None
    max_reflections: int | None = None
    photon_grid_stride: int = 1
    photon_grid_span_fraction: float = 1.0
    provisional: bool = False
    # Emission policy (tri-state). A policy, not a grid reduction, so it rides
    # the profile like n_electrons. "incoherent" (default) is the incoherent
    # path; "coherent"/"both" enable the phased kernel.
    emission: EmissionMode = "incoherent"

    @property
    def coherent_emission(self) -> bool:
        """Derived: whether this preset's emission policy runs the coherent
        kernel. Kept so ``build_cases(coherent_emission=)`` and other
        transport-side readers are untouched by the tri-state rename."""
        return self.emission in {"coherent", "both"}

    def apply_settings(self, settings: Settings) -> Settings:
        """Return settings with this preset's transport counts resolved."""
        return replace(
            settings,
            n_electrons=self.n_electrons,
            n_electrons_brem=self.n_electrons_brem,
            emission=self.emission,
        )

    def apply_sweep(self, sweep: Sweep) -> Sweep:
        """Return catalog sweep reduced according to this preset."""
        if self.name == "full":
            return sweep
        energies = _centered_sample(sweep.beam.energy_keV, self.max_energies)
        energy_values = {float(value) for value in np.atleast_1d(energies)}
        assert sweep.detector is not None
        bins = sweep.detector.energy_bins
        by_energy = bins.line_by_energy
        if by_energy is not None:
            by_energy = {
                float(energy): _coarsen_grid(
                    grid, self.photon_grid_stride, self.photon_grid_span_fraction
                )
                for energy, grid in by_energy.items()
                if float(energy) in energy_values
            }
        line_grid = (
            None
            if bins.line is None
            else _coarsen_grid(
                bins.line,
                self.photon_grid_stride,
                self.photon_grid_span_fraction,
            )
        )
        brem_grid = (
            None
            if bins.brem is None
            else _coarsen_grid(
                bins.brem,
                self.photon_grid_stride,
                self.photon_grid_span_fraction,
            )
        )
        assert sweep.target is not None
        geometry = target_flat_fields(sweep.target)
        return replace(
            sweep,
            beam=beam_replace(sweep.beam, energy_keV=energies),
            target=target_replace(
                sweep.target,
                thickness_ang=_centered_sample(geometry["thickness_ang"], self.max_thicknesses),
                tilt_deg=_centered_sample(geometry["tilt_deg"], self.max_tilts),
                tilt_azim_deg=_centered_sample(geometry["tilt_azim_deg"], self.max_azimuths),
            ),
            n_families=self.n_families or sweep.n_families,
            max_reflections=self.max_reflections,
            detector=replace(
                sweep.detector,
                energy_bins=EnergyBins(line=line_grid, line_by_energy=by_energy, brem=brem_grid),
            ),
        )


_FIDELITY_PRESETS = {
    "full": FidelityPreset("full", n_electrons=300, n_electrons_brem=150),
    "survey": FidelityPreset(
        "survey",
        n_electrons=60,
        n_electrons_brem=30,
        max_energies=2,
        max_thicknesses=3,
        max_tilts=5,
        max_azimuths=2,
        n_families=2,
        max_reflections=4,
        photon_grid_stride=4,
        photon_grid_span_fraction=0.7,
        provisional=True,
    ),
}


def get_fidelity_preset(name: str = "full") -> FidelityPreset:
    """Return named fidelity preset or raise a boundary-friendly ``ValueError``."""
    try:
        return _FIDELITY_PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown fidelity preset {name!r} (choose from {FIDELITY_NAMES})"
        ) from None


def resolve_numerics(
    explicit: Mapping[str, object] | None = None,
    *,
    fidelity: str = "full",
    overrides: Mapping[str, object] | None = None,
) -> NumericsResolution:
    """Resolve run overrides over profile, fidelity, then built-in numerics."""
    preset = get_fidelity_preset(fidelity)
    convergence = Convergence()
    numerics = Numerics(convergence=convergence)
    effective: dict[str, object] = {
        "n_electrons": preset.n_electrons,
        "n_electrons_brem": preset.n_electrons_brem,
        "n_families": (convergence.n_families if preset.n_families is None else preset.n_families),
        "max_reflections": preset.max_reflections,
        "mosaic_nodes": convergence.mosaic_nodes,
        "mosaic_route": convergence.mosaic_route,
        "straggling": numerics.straggling,
        "energy_model": numerics.energy_model,
        "max_dE_frac": numerics.max_dE_frac,
        "inelastic_model": numerics.inelastic_model,
        "inelastic_cutoff_eV": numerics.inelastic_cutoff_eV,
        "elastic_model": numerics.elastic_model,
    }
    sources = {
        key: (
            "fidelity"
            if key in SAMPLING_KEYS
            or (key == "n_families" and preset.n_families is not None)
            or (key == "max_reflections" and preset.max_reflections is not None)
            else "built-in"
        )
        for key in PROFILE_NUMERICS_KEYS
    }
    normalized: dict[str, object] = {}
    for key, value in dict(explicit or {}).items():
        if key not in PROFILE_NUMERICS_KEYS:
            continue
        if key in SAMPLING_KEYS:
            counts = electron_counts(value)
            value = counts[0] if len(counts) == 1 else list(counts)
        normalized[key] = value
        effective[key] = value
        sources[key] = "profile"
    run_overrides = dict(overrides or {})
    validate_profile_numerics({**normalized, **run_overrides})
    for key, value in run_overrides.items():
        if key in PROFILE_NUMERICS_KEYS:
            effective[key] = value
            sources[key] = "run"
    return NumericsResolution(
        fidelity=fidelity,
        explicit=normalized,
        effective=effective,
        sources=sources,
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, np.ndarray):
        # A non-object dtype's `.tolist()` is already a plain nested list of
        # int/float/bool -- never a dataclass or Mapping -- so re-walking each
        # element through `_jsonable` is pure overhead. That walk dominated
        # `pyrite prune`/`pyrite checkpoint gc` runtime (minutes) once an
        # energy-grid case field fell back to its full uncompressed array.
        if value.dtype != object:
            return value.tolist()
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _bremsstrahlung_identity_marker(model: Literal["eedl", "bremslib"]) -> str:
    """Return the selected continuum generation; reject unsupported run models."""
    if model == "eedl":
        return BREMSSTRAHLUNG_MODEL
    if model == "bremslib":
        return BREMSSTRAHLUNG_BREMSLIB_MODEL
    raise ValueError(f"bremsstrahlung_model must be 'eedl' or 'bremslib'; got {model!r}")


def _identity_v1(
    material: str,
    fidelity: str,
    settings: Settings,
    sweep: Sweep,
    *,
    variant: str | None = None,
    catalog_profile: str = "standard",
    xsgen_tables: Mapping[str, str] | None = None,
    bremsstrahlung_model: Literal["eedl", "bremslib"] = "eedl",
) -> dict[str, Any]:
    """Return profile plus exact resolved parameters and stable SHA-256 digest."""
    assert sweep.target is not None
    crystallography = crystal_params(sweep.target.material, sweep.n_families)
    reflections = crystallography["hkl_list"]
    if sweep.max_reflections is not None:
        reflections = reflections[: sweep.max_reflections]
    resolved = {
        "material": material,
        "fidelity": fidelity,
        "variant": variant,
        "settings": _jsonable(settings),
        "sweep": _jsonable(sweep),
        "crystallography": {
            "crystal": crystallography["crystal"],
            "hkl_list": _jsonable(reflections),
            "beam_uvw": _jsonable(
                sweep.beam_uvw if sweep.beam_uvw is not None else crystallography["beam_uvw"]
            ),
            "surface_hkl": _jsonable(
                None if sweep.beam_uvw is not None else crystallography["surface_hkl"]
            ),
        },
    }
    # Hash compatibility: the P2.4 electron-count sweep grids join the payload
    # only when actually set, so pre-existing runs keep their historical
    # parameter_sha256 (and therefore their checkpoint identity) bit-for-bit.
    sweep_payload = resolved["sweep"]
    # Flat legacy projection of the geometry, for the same reason as the beam and
    # detector projections below: geometry lives on ``Sweep.target`` in code, but
    # the hashed payload keeps the HISTORICAL flat keys -- ``thickness_ang``,
    # ``tilt_deg``, the footprint pair, the substrate sugar -- so every
    # pre-existing ``parameter_sha256`` (and its checkpoint stem) stays
    # bit-for-bit.
    sweep_payload.pop("target", None)
    sweep_payload.update(_jsonable(target_flat_fields(sweep.target)))
    detector_payload = sweep_payload.pop("detector")
    sweep_payload["theta_obs_deg"] = detector_payload["observation_angle_deg"]
    sweep_payload["dtheta_obs_deg"] = detector_payload["polar_acceptance_deg"]
    sweep_payload["domega_sr"] = detector_payload["solid_angle_sr"]
    energy_bins = detector_payload.pop("energy_bins")
    sweep_payload["E_grid_line"] = energy_bins["line"]
    sweep_payload["E_grid_line_by_energy"] = energy_bins["line_by_energy"]
    sweep_payload["E_grid_brem"] = energy_bins["brem"]
    sweep_payload["e_grid_eV"] = None
    # Response is applied only to response-free source arrays at read time. It must not
    # fork transport/checkpoint identity, which is deliberately reusable by
    # several response models and is the future multi-detector seam.
    detector_payload.pop("response", None)
    reserved_detector = {
        key: value
        for key, value in detector_payload.items()
        if key not in {"observation_angle_deg", "polar_acceptance_deg", "solid_angle_sr"}
        and value is not None
    }
    if reserved_detector:
        sweep_payload["detector"] = reserved_detector
    # Flat legacy projection of the beam (decision 6). The beam lives on
    # ``Sweep.beam`` in code, but the hashed payload keeps the HISTORICAL flat
    # top-level keys -- ``energy_keV`` and ``beam_fwhm_mm`` at their old
    # positions -- so every pre-existing ``parameter_sha256`` (and therefore its
    # checkpoint stem) stays bit-for-bit. New beam fields (elliptical y, the
    # longitudinal bunch, rep-rate/charge, future emittance) join the hash ONLY
    # when they diverge from their inert defaults -- the same compatibility rule
    # already used for ``n_electrons`` and ``catalog_profile`` below.
    beam_payload = sweep_payload.pop("beam")
    sweep_payload["energy_keV"] = beam_payload["energy_keV"]
    fwhm_x = beam_payload["transverse_fwhm_x_mm"]
    fwhm_y = beam_payload["transverse_fwhm_y_mm"]
    sweep_payload["beam_fwhm_mm"] = fwhm_x  # legacy isotropic key == x-plane FWHM
    if fwhm_y != fwhm_x:
        sweep_payload["transverse_fwhm_y_mm"] = fwhm_y
    if beam_payload["bunch_length_fs"] is not None:
        sweep_payload["bunch_length_fs"] = beam_payload["bunch_length_fs"]
    if beam_payload["long_shape"] != "gaussian":
        sweep_payload["long_shape"] = beam_payload["long_shape"]
    if beam_payload["long_offsets_fs"] is not None:
        sweep_payload["long_offsets_fs"] = beam_payload["long_offsets_fs"]
    if beam_payload["longitudinal"] is not None:
        sweep_payload["longitudinal"] = beam_payload["longitudinal"]
        cases = build_cases(
            sweep,
            n_electrons=settings.n_electrons,
            n_electrons_brem=settings.n_electrons_brem,
            coherent_emission=settings.coherent_emission,
        )
        serialized: dict[str, Any] = {}
        for case in cases:
            resolution = case["longitudinal_distribution"]
            key = json.dumps(_jsonable(resolution), sort_keys=True, separators=(",", ":"))
            serialized.setdefault(key, _jsonable(resolution))
        resolved["longitudinal_resolutions"] = list(serialized.values())
    if beam_payload["transverse"] is not None:
        sweep_payload["transverse"] = beam_payload["transverse"]
    if beam_payload["rep_rate_hz"] != 5000.0:
        sweep_payload["rep_rate_hz"] = beam_payload["rep_rate_hz"]
    if beam_payload["bunch_charge_pc"] != 1.0:
        sweep_payload["bunch_charge_pc"] = beam_payload["bunch_charge_pc"]
    if beam_payload["divergence_mrad"] is not None:
        sweep_payload["divergence_mrad"] = beam_payload["divergence_mrad"]
    if beam_payload["energy_spread_frac"] is not None:
        sweep_payload["energy_spread_frac"] = beam_payload["energy_spread_frac"]
    for key in ("n_electrons", "n_electrons_brem", "line_grid_policy"):
        if sweep_payload.get(key) is None:
            sweep_payload.pop(key, None)
    # Same compatibility rule for catalog_profile (Phase 3 decision 2): only
    # join the hashed payload when it diverges from "standard", so every
    # pre-existing standard-profile identity keeps its historical digest.
    if catalog_profile != "standard":
        resolved["catalog_profile"] = catalog_profile
    # emission (run-affecting) follows the same divergence-only rule: it is a NEW
    # Settings field, so hashing it unconditionally would perturb every
    # pre-existing digest. Strip it from the serialized settings and re-add it at
    # the top level only when it diverges from "incoherent" -- an incoherent
    # run's parameter_sha256 stays bit-for-bit, while "coherent" and "both" each
    # get a distinct digest (and checkpoint stem) so the three modes never
    # collide or resume into one another. Clean rename of the old
    # coherent_emission=True key: no legacy back-compat branch, so pre-existing
    # coherent checkpoints are intentionally orphaned (rev-and-re-run). The bunch
    # fields (bunch_length_fs etc.) keep their own divergence rule above.
    settings_payload = resolved["settings"]
    if isinstance(settings_payload, Mapping):
        emission = str(settings_payload.pop("emission", "incoherent"))
        straggling = bool(settings_payload.pop("straggling", False))
        energy_model = str(settings_payload.pop("energy_model", "frozen"))
        max_dE_frac = float(settings_payload.pop("max_dE_frac", 0.0))
        inelastic_model = str(settings_payload.pop("inelastic_model", "continuous"))
        inelastic_cutoff_eV = settings_payload.pop("inelastic_cutoff_eV", None)
        elastic_model = str(settings_payload.pop("elastic_model", "mott"))
    else:  # pragma: no cover - settings is always a jsonable Mapping here
        emission = str(getattr(settings, "emission", "incoherent"))
        straggling = bool(getattr(settings, "straggling", False))
        energy_model = str(getattr(settings, "energy_model", "frozen"))
        max_dE_frac = float(getattr(settings, "max_dE_frac", 0.0))
        inelastic_model = str(getattr(settings, "inelastic_model", "continuous"))
        inelastic_cutoff_eV = getattr(settings, "inelastic_cutoff_eV", None)
        elastic_model = str(getattr(settings, "elastic_model", "mott"))
    if emission != "incoherent":
        resolved["emission"] = emission
    transport_numerics = {}
    if straggling:
        transport_numerics["straggling"] = True
    if energy_model != "frozen":
        transport_numerics["energy_model"] = energy_model
    if max_dE_frac != 0.0:
        transport_numerics["max_dE_frac"] = max_dE_frac
    # Divergence-only like the three keys above: continuous stopping (the
    # default) leaves every existing digest unchanged.
    if inelastic_model != "continuous":
        transport_numerics["inelastic_model"] = inelastic_model
        transport_numerics["inelastic_cutoff_eV"] = float(cast(float, inelastic_cutoff_eV))
    if elastic_model != "mott":
        transport_numerics["elastic_model"] = elastic_model
    if transport_numerics:
        resolved["transport_numerics"] = transport_numerics
    # The in-medium photon dispersion is unconditional physics now, not an opt-in
    # model, so it no longer earns a divergence-only key. Every digest minted
    # before that change was computed under the retired vacuum k = omega
    # kinematics, though, so the payload carries a permanent line-kinematics
    # generation marker. It is a CONSTANT, not a selector: it perturbs every
    # digest exactly once, orphaning the vacuum-era checkpoints (rev-and-re-run,
    # matching how the emission rename was handled) instead of letting them
    # resume into a run that computes different numbers.
    resolved["line_kinematics"] = "in-medium"
    # Same rule for the electron collision-stopping model. Production resolves
    # one shell- and density-effect-corrected SBETHE table per material; the
    # former Joy--Luo/Berger--Seltzer splice remains reference-only. The model
    # marker separates every splice-era record, while ``xsgen_tables`` below
    # separates different generated inputs and manifests within this model.
    resolved["stopping_model"] = STOPPING_MODEL
    # Characteristic line production is unconditional and changes the stored
    # line arrays, so the exact EEDL/xraydb model generation must separate
    # checkpoints from pre-characteristic and future database generations.
    resolved["characteristic_model"] = CHARACTERISTIC_MODEL
    # The continuum now defaults to evaluated EEDL MF=23/527 + MF=26/527
    # instead of the historical analytic Bethe--Heitler approximation.
    resolved["bremsstrahlung_model"] = _bremsstrahlung_identity_marker(bremsstrahlung_model)
    # Externally generated cross-section tables (issue #161), as table key ->
    # provenance-manifest digest. A *divergence-only* key, like `emission` and
    # `transport_numerics` above and unlike the four model constants: a run
    # that resolves no xsgen table omits it entirely and keeps its historical
    # digest bit-for-bit. The four constants each perturbed every digest once,
    # deliberately, because the physics they name changed for every run; no
    # A participating table's manifest digest covers the Fortran
    # source, the compiler, the deck and the model parameters, so regenerating
    # a table under different settings re-keys the run rather than resuming
    # into checkpoints computed from the old one.
    if xsgen_tables:
        resolved["xsgen_tables"] = {str(key): str(value) for key, value in xsgen_tables.items()}
    encoded = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": DATASET_IDENTITY_SCHEMA,
        "identity_version": 1,
        "material": material,
        "fidelity": fidelity,
        "variant": variant,
        "catalog_profile": catalog_profile,
        "parameter_sha256": hashlib.sha256(encoded).hexdigest(),
        "resolved_parameters": resolved,
    }


# Version 2 is deliberately not registered: it is the deferred canonical
# full-payload/recompute boundary from RFC Change 3 step 3.
IDENTITY_MIGRATIONS = {1: _identity_v1}


def dataset_identity(
    material: str,
    fidelity: str,
    settings: Settings,
    sweep: Sweep,
    *,
    variant: str | None = None,
    catalog_profile: str = "standard",
    identity_version: int = CURRENT_IDENTITY_VERSION,
    xsgen_tables: Mapping[str, str] | None = None,
    bremsstrahlung_model: Literal["eedl", "bremslib"] = "eedl",
) -> dict[str, Any]:
    """Resolve a dataset identity through its explicit versioned algorithm.

    ``xsgen_tables`` maps table key to provenance-manifest digest for every
    externally generated cross-section table this run reads, from
    :func:`pyrite.xsgen.store.identity_markers`. Omitted or empty leaves the
    digest exactly as it was before issue #161.

    ``bremsstrahlung_model`` selects the physics-generation marker. The
    default preserves existing EEDL checkpoint identities.
    """
    try:
        migration = IDENTITY_MIGRATIONS[identity_version]
    except KeyError, TypeError:
        raise ValueError(f"unsupported dataset identity version: {identity_version!r}") from None
    return migration(
        material,
        fidelity,
        settings,
        sweep,
        variant=variant,
        catalog_profile=catalog_profile,
        xsgen_tables=xsgen_tables,
        bremsstrahlung_model=bremsstrahlung_model,
    )


def normalize_dataset_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an artifact identity, treating an absent version as legacy v1."""
    version = identity.get("identity_version", 1)
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in IDENTITY_MIGRATIONS
    ):
        raise ValueError(f"unsupported dataset identity version: {version!r}")
    normalized = dict(identity)
    normalized["identity_version"] = version
    return normalized


def case_content_key(
    case: Case | Mapping[str, Any],
    *,
    xsgen_tables: Mapping[str, str] | None = None,
    bremsstrahlung_model: Literal["eedl", "bremslib"] = "eedl",
) -> str:
    """Content-addressable key for one :func:`pyrite.campaign.sweep.build_cases` case.

    A canonical SHA-256 over the resolved case mapping minus
    :data:`_CONTENT_KEY_DENYLIST`. Because the case is already fully
    resolved -- every physics-relevant field baked in by ``build_cases`` -- and
    carries NO profile name, two cases requested by two differently-named
    profiles that describe the same physics hash to the same key and share one
    cached blob. That single profile-name exclusion is the whole cross-profile
    reuse feature (see :mod:`pyrite.runs.run`'s content-addressable store).

    ``seed`` deliberately stays in the key: it is derived from a case's grid
    position, so two profiles only share a key when their shared case also shares
    a seed (i.e. their grids align, e.g. one energy grid a prefix of the other) --
    exactly the case where the stored arrays are bit-identical. A shared physics
    case whose seed differs simply gets a distinct key and recomputes; the store
    never serves a mismatched-seed result.

    The stopping and characteristic-radiation markers, plus the selected
    bremsstrahlung model marker, join the payload alongside the case. Without
    them a blob from an earlier physics/data generation could be served silently.

    ``xsgen_tables`` is the same thing for externally generated cross-section
    tables (issue #161), and has to be here as well as in
    :func:`dataset_identity` rather than instead of it: the two gate different
    caches. The dataset identity gates the checkpoint *stem*, while this key
    gates the content-addressable blob store, and a run whose stem moved would
    otherwise still be served a blob computed from the superseded table. Unlike
    the three constants it is conditional, so a run reading no xsgen table
    keeps its existing content key and every stored blob stays reachable.
    """
    payload = {
        "schema": CASE_CONTENT_KEY_SCHEMA,
        "stopping_model": STOPPING_MODEL,
        "characteristic_model": CHARACTERISTIC_MODEL,
        "bremsstrahlung_model": _bremsstrahlung_identity_marker(bremsstrahlung_model),
        "case": _jsonable(
            {key: value for key, value in case.items() if key not in _CONTENT_KEY_DENYLIST}
        ),
    }
    if xsgen_tables:
        payload["xsgen_tables"] = {str(key): str(value) for key, value in xsgen_tables.items()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def variant_stem(identity: Mapping[str, Any], *, canonical_full: bool = False) -> str:
    """Map identity to active checkpoint stem.

    Canonical ``full`` retains historical ``<material>`` storage and ``--quick``
    keeps the ``<material>_quick`` stem. Every other resolved variant uses a
    readable ``<material>@<label>-<digest>`` stem (the "@-stem" scheme,
    2026-07-29): the label names the run's ``catalog_profile`` when non-standard
    (so a named-profile checkpoint finally carries its profile name), else the
    fidelity. The 12-hex ``parameter_sha256`` prefix keeps two runs that share a
    label but differ in parameters -- a coherent vs incoherent run, a re-run
    after the backing profile was edited -- from ever colliding on disk.

    Older ``<material>--<fidelity>-<digest>`` stems written before this scheme
    are still resolved on read (dual-read; see :data:`_VARIANT_STEM_RE` and
    :func:`identity_from_stem`); only newly written stems use ``@``.
    """
    material = str(identity["material"])
    if canonical_full and identity["fidelity"] == "full" and identity.get("variant") is None:
        return material
    if identity.get("variant") == "quick":
        return f"{material}_quick"
    return f"{material}@{_stem_label(identity)}-{str(identity['parameter_sha256'])[:12]}"


def _stem_label(identity: Mapping[str, Any]) -> str:
    """Readable @-stem label: an explicit ``variant``, else a non-standard
    ``catalog_profile`` name, else the fidelity."""
    variant = identity.get("variant")
    if variant:
        return str(variant)
    catalog_profile = identity.get("catalog_profile", "standard")
    if catalog_profile and catalog_profile != "standard":
        return str(catalog_profile)
    return str(identity["fidelity"])


def named_profile_identity(
    material: str, fidelity: str = "full", *, catalog_profile: str = "standard"
) -> dict[str, Any]:
    """Resolve identity for an unmodified named material/fidelity pair."""
    # Local import avoids config -> profiles -> config import cycle.
    from .config import default_settings, material_sweep

    return dataset_identity(
        material,
        fidelity,
        default_settings(fidelity),
        material_sweep(material, fidelity=fidelity, catalog_profile=catalog_profile),
        catalog_profile=catalog_profile,
    )


def named_profile_stem(
    material: str, fidelity: str = "full", *, catalog_profile: str = "standard"
) -> str:
    """Checkpoint stem for an unmodified named material/profile pair."""
    return variant_stem(
        named_profile_identity(material, fidelity, catalog_profile=catalog_profile),
        canonical_full=fidelity == "full" and catalog_profile == "standard",
    )


def high_energy_floor_identity(
    material: str, floor_kev: float, fidelity: str = "full", *, catalog_profile: str = "standard"
) -> dict[str, Any]:
    """Resolve identity for a material/profile pair with its energy_keV grid
    floored at ``floor_kev`` for compatibility with legacy remote job records.
    Mirrors ``named_profile_identity`` so remote orchestration can predict the
    same non-canonical stem the runner will write."""
    # Local import avoids config -> profiles -> config import cycle.
    from .config import default_settings, material_sweep

    sweep = material_sweep(material, fidelity=fidelity, catalog_profile=catalog_profile)
    energies = np.asarray(sweep.beam.energy_keV, dtype=float)
    kept = energies[energies >= floor_kev]
    if kept.size == 0:
        raise SystemExit(
            f"{material}: no energies >= {floor_kev} keV in its grid (high-energy "
            "floor); lower --high-energy-min-kev or drop this material"
        )
    sweep = replace(sweep, beam=beam_replace(sweep.beam, energy_keV=kept))
    return dataset_identity(
        material, fidelity, default_settings(fidelity), sweep, catalog_profile=catalog_profile
    )


def high_energy_floor_stem(
    material: str, floor_kev: float, fidelity: str = "full", *, catalog_profile: str = "standard"
) -> str:
    """Checkpoint stem for a high-energy-floored material/profile pair."""
    return variant_stem(
        high_energy_floor_identity(material, floor_kev, fidelity, catalog_profile=catalog_profile),
        canonical_full=False,
    )


# Matches both stem schemes so old and new checkpoints resolve side by side
# (dual-read): the legacy ``<material>--<fidelity>-<digest>`` form (``fidelity``
# group set) and the current ``<material>@<label>-<digest>`` @-stem form
# (``label`` group set). ``material`` and ``label`` are lazy so the trailing
# ``-<12 hex>`` digest anchors the split unambiguously.
_VARIANT_STEM_RE = re.compile(
    r"^(?P<material>.+?)"
    r"(?:--(?P<fidelity>full|survey)|@(?P<label>[^@]+?))"
    r"-(?P<digest>[0-9a-f]{12})$"
)


def _catalog_profile_candidates() -> tuple[str, ...]:
    """Catalog profiles a hashed variant stem could belong to: ``standard``
    first (the common case), then every named profile in materials.toml.
    Local import breaks the config/profiles import cycle (same pattern as
    :func:`named_profile_identity`)."""
    from ..materials import CATALOG

    return ("standard", *CATALOG.profile_names)


def _recompute_candidates(
    groups: Mapping[str, str | None],
) -> Any:
    """``(fidelity, catalog_profile)`` pairs to try when recomputing a
    sidecar-less variant stem's identity. The digest -- not the stem text --
    decides the match, so this only narrows the search using whatever the stem
    names: a legacy ``--<fidelity>-`` stem fixes the fidelity (profile unknown,
    try every candidate); an ``@<label>-`` stem's label is either a fidelity
    name (a standard-profile run) or a ``catalog_profile`` name (fidelity
    unknown, try each)."""
    fidelity = groups.get("fidelity")
    if fidelity is not None:
        for catalog_profile in _catalog_profile_candidates():
            yield fidelity, catalog_profile
        return
    label = groups.get("label")
    if label in FIDELITY_NAMES:
        yield label, "standard"
        return
    for fidelity_name in FIDELITY_NAMES:
        yield fidelity_name, label


def _sidecar_identity(stem: str, root: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Return the resolved ``dataset_identity`` recorded in a stem's
    ``meta.json`` sidecar at run time (:func:`pyrite.runs.run._manifest_save`), or
    ``None`` when no sidecar/identity is present. This is authoritative: it
    survives edits to the backing named profile after the run, which a live
    recompute against the *current* catalog would not (the digest would no
    longer match). The sidecar layout is a storage contract shared with the
    checkpoint and remote owners; resolving its path directly keeps campaign
    identity parsing below those drivers."""
    manifest = Path(root) / stem / "meta.json"
    if not manifest.is_file():
        return None
    try:
        with manifest.open() as handle:
            identity = json.load(handle).get("dataset_identity")
    except OSError, ValueError, TypeError:
        return None
    return normalize_dataset_identity(identity) if isinstance(identity, dict) else None


def identity_from_stem(
    stem: str, root: str | os.PathLike[str] | None = None
) -> dict[str, Any] | None:
    """Reconstruct resolved identity from a checkpoint stem.

    When ``root`` is given and the stem's ``meta.json`` sidecar records a
    ``dataset_identity``, that recorded identity is authoritative and returned
    directly. Reading the sidecar (the established pattern in
    :func:`archive._dataset_identity` and ``_remote/lifecycle``) stays correct
    even after the backing named profile is edited post-run -- a live recompute
    against the current catalog would silently fail to match in that case.

    Only stems with no such sidecar fall back to recomputing: the stem's digest
    commits to a resolved parameter set but the stem text names it only
    partially, so the identity is recomputed under each plausible
    ``(fidelity, catalog_profile)`` pair (see :func:`_recompute_candidates`)
    until the digests agree. A material that is not a member of a candidate
    profile raises inside ``material_sweep`` -- that candidate is simply
    skipped. A run whose digest depends on state not reconstructable from the
    catalog alone (a coherent run, an edited profile) never matches here and
    relies on its sidecar instead."""
    if root is not None:
        recorded = _sidecar_identity(stem, root)
        if recorded is not None:
            return recorded
    match = _VARIANT_STEM_RE.fullmatch(stem)
    if match is not None:
        groups = match.groupdict()
        for fidelity, catalog_profile in _recompute_candidates(groups):
            try:
                identity = named_profile_identity(
                    groups["material"], fidelity, catalog_profile=catalog_profile
                )
            except KeyError, ValueError:
                continue
            if identity["parameter_sha256"].startswith(groups["digest"]):
                return identity
        return None
    try:
        identity = named_profile_identity(stem, "full")
    except ValueError:
        return None
    return identity if named_profile_stem(stem, "full") == stem else None
