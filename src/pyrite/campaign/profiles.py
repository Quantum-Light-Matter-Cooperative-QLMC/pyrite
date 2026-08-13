"""Fidelity presets and reproducible dataset identities.

Profiles are policies applied after catalog material lookup.  ``full`` is an
exact compatibility policy: it leaves catalog grids unchanged and retains the
production electron counts.  ``survey`` is provisional and intentionally
smaller.  Every resolved run can be serialized into a deterministic identity,
so differently resolved variants never silently resume into one dataset.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..montecarlo.case import Case
from ..results import EmissionMode, Settings
from .sweep import Sweep, beam_replace, build_cases, crystal_params

FIDELITY_NAMES = ("full", "survey")
DATASET_IDENTITY_SCHEMA = "cxr.dataset-identity.v1"
CASE_CONTENT_KEY_SCHEMA = "cxr.case-content-key.v1"
CURRENT_IDENTITY_VERSION = 1

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
        by_energy = sweep.E_grid_line_by_energy
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
            if sweep.E_grid_line is None
            else _coarsen_grid(
                sweep.E_grid_line,
                self.photon_grid_stride,
                self.photon_grid_span_fraction,
            )
        )
        brem_grid = (
            None
            if sweep.E_grid_brem is None
            else _coarsen_grid(
                sweep.E_grid_brem,
                self.photon_grid_stride,
                self.photon_grid_span_fraction,
            )
        )
        return replace(
            sweep,
            beam=beam_replace(sweep.beam, energy_keV=energies),
            thickness_ang=_centered_sample(sweep.thickness_ang, self.max_thicknesses),
            tilt_deg=_centered_sample(sweep.tilt_deg, self.max_tilts),
            tilt_azim_deg=_centered_sample(sweep.tilt_azim_deg, self.max_azimuths),
            n_families=self.n_families or sweep.n_families,
            max_reflections=self.max_reflections,
            E_grid_line=line_grid,
            E_grid_line_by_energy=by_energy,
            E_grid_brem=brem_grid,
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


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, np.ndarray):
        # A non-object dtype's `.tolist()` is already a plain nested list of
        # int/float/bool -- never a dataclass or Mapping -- so re-walking each
        # element through `_jsonable` is pure overhead. That walk dominated
        # `cxr prune`/`cxr checkpoint gc` runtime (minutes) once an
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


def _identity_v1(
    material: str,
    fidelity: str,
    settings: Settings,
    sweep: Sweep,
    *,
    variant: str | None = None,
    catalog_profile: str = "standard",
) -> dict[str, Any]:
    """Return profile plus exact resolved parameters and stable SHA-256 digest."""
    crystallography = crystal_params(sweep.material, sweep.n_families)
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
    detector_payload = sweep_payload.pop("detector")
    sweep_payload["theta_obs_deg"] = detector_payload["observation_angle_deg"]
    sweep_payload["dtheta_obs_deg"] = detector_payload["polar_acceptance_deg"]
    sweep_payload["domega_sr"] = detector_payload["solid_angle_sr"]
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
    for key in ("n_electrons", "n_electrons_brem"):
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
        dispersion = str(settings_payload.pop("xray_dispersion", "vacuum"))
    else:  # pragma: no cover - settings is always a jsonable Mapping here
        emission = str(getattr(settings, "emission", "incoherent"))
        dispersion = str(getattr(settings, "xray_dispersion", "vacuum"))
    if emission != "incoherent":
        resolved["emission"] = emission
    # xray_dispersion (run-affecting) follows the same divergence-only rule, for
    # the same reason: hashing it unconditionally would perturb every existing
    # digest, while dropping it would let a refractive run resume into its
    # vacuum twin's checkpoint.
    if dispersion != "vacuum":
        resolved["xray_dispersion"] = dispersion
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
) -> dict[str, Any]:
    """Resolve a dataset identity through its explicit versioned algorithm."""
    try:
        migration = IDENTITY_MIGRATIONS[identity_version]
    except (KeyError, TypeError):
        raise ValueError(f"unsupported dataset identity version: {identity_version!r}") from None
    return migration(
        material,
        fidelity,
        settings,
        sweep,
        variant=variant,
        catalog_profile=catalog_profile,
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


def case_content_key(case: Case | Mapping[str, Any]) -> str:
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
    """
    payload = {
        "schema": CASE_CONTENT_KEY_SCHEMA,
        "case": _jsonable(
            {key: value for key, value in case.items() if key not in _CONTENT_KEY_DENYLIST}
        ),
    }
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
    longer match). Same sidecar read already relied on by
    :func:`archive._dataset_identity` and the remote lifecycle. Local import of
    ``_checkpoint_store`` mirrors the deferred-import pattern used elsewhere in
    this module."""
    from ..checkpoints import _checkpoint_store

    manifest = _checkpoint_store.manifest_path(stem, root)
    if not manifest.is_file():
        return None
    try:
        with manifest.open() as handle:
            identity = json.load(handle).get("dataset_identity")
    except (OSError, ValueError, TypeError):
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
            except (KeyError, ValueError):
                continue
            if identity["parameter_sha256"].startswith(groups["digest"]):
                return identity
        return None
    try:
        identity = named_profile_identity(stem, "full")
    except ValueError:
        return None
    return identity if named_profile_stem(stem, "full") == stem else None
