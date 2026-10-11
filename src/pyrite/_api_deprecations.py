"""Deprecation warnings and registries for renamed or now-required Python API spellings."""

import warnings
from dataclasses import dataclass
from functools import wraps
from inspect import signature

from ._module_deprecations import _window


@dataclass(frozen=True)
class ApiDeprecation:
    """One deprecated Python API spelling and its canonical replacement."""

    spelling: str
    replacement: str
    deprecated_in: str
    remove_in: str


def _api_entry(spelling: str, replacement: str, *, since: str) -> ApiDeprecation:
    return ApiDeprecation(spelling, replacement, since, _window(since))


#: Keyed by the deprecated spelling as a user writes it (issue #451). Each row is
#: held to the shipping version by ``tests/test_deprecation_schedule.py``.
API_DEPRECATIONS: dict[str, ApiDeprecation] = {
    entry.spelling: entry
    for entry in (
        _api_entry(
            "Beam() without transverse=",
            "Beam(transverse=TransverseSpot(fwhm_x_mm=1.0)) or Beam(transverse=PointSpot())",
            since="0.12.0",
        ),
        _api_entry(
            "Beam() without longitudinal=",
            "Beam(longitudinal=PointBunch()) or Beam(longitudinal=BunchLength(rms_fs=...))",
            since="0.12.0",
        ),
        _api_entry(
            "Slab() without footprint=",
            "Slab(footprint=Footprint(5.0, 5.0)) or Slab(footprint=None) for an infinite slab",
            since="0.12.0",
        ),
        _api_entry(
            "Stack() without footprint=",
            "Stack(footprint=Footprint(5.0, 5.0)) or Stack(footprint=None) for infinite extent",
            since="0.12.0",
        ),
        _api_entry(
            'Physics(emission="both")',
            'Physics(emission=("incoherent", "coherent"))',
            since="0.13.1",
        ),
        _api_entry("pyrite.Point", "pyrite.PointSpot", since="0.12.0"),
        _api_entry("simulate(pixel_scorer=)", "PlanarDetector(scorer=...)", since="0.12.0"),
        _api_entry("Scene(pixel_scorer=)", "PlanarDetector(scorer=...)", since="0.12.0"),
        _api_entry(
            "Numerics(backend=)",
            "PYRITE_MC_BACKEND set before importing pyrite (or `pyrite config setup`)",
            since="0.12.0",
        ),
        _api_entry("pyrite.campaign.beam.Point", "pyrite.campaign.beam.PointSpot", since="0.12.0"),
    )
}


# Public policy ownership changes (issue #452), D7 two-minor window.
for _name in (
    "straggling",
    "inelastic_model",
    "inelastic_cutoff_eV",
    "secondary_threshold_eV",
    "elastic_model",
    "bremsstrahlung_model",
    "radiative_model",
    "radiative_cutoff_eV",
    "pair_production_model",
    "positron_transport",
    "atomic_electron_deflection",
    "emission",
    "brem_source",
):
    API_DEPRECATIONS[f"Numerics({_name}=)"] = _api_entry(
        f"Numerics({_name}=)", f"Physics({_name}=...)", since="0.12.0"
    )
for _old, _new in (
    ("Convergence", "CxrReflections"),
    ("Numerics(convergence=)", "Numerics(cxr_reflections=...)"),
    ("simulate(emission=)", "simulate(physics=Physics(emission=...))"),
    ("simulate(brem_source=)", "simulate(physics=Physics(brem_source=...))"),
    ("Scene(emission=)", "Scene(physics=Physics(emission=...))"),
    ("Scene(brem_source=)", "Scene(physics=Physics(brem_source=...))"),
    ("Detector(energy_bins=)", "Numerics(energy_bins=...)"),
    ("PlanarDetector(energy_bins=)", "Numerics(energy_bins=...)"),
):
    API_DEPRECATIONS[_old] = _api_entry(_old, _new, since="0.12.0")


def warn_api(spelling: str, *, stacklevel: int = 3) -> None:
    """Warn that registered *spelling* is deprecated, naming its replacement."""
    entry = API_DEPRECATIONS[spelling]
    # An omitted argument becomes an error at the target; a renamed name is removed.
    outcome = "an error" if " without " in entry.spelling else "removed"
    warnings.warn(
        f"{entry.spelling} is deprecated and will be {outcome} in {entry.remove_in}; "
        f"use {entry.replacement} instead",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def _hide_compat_parameters(initialize, original) -> None:
    """Keep internal copy-state keywords out of public signatures and API docs."""
    original_signature = signature(original)
    initialize.__signature__ = original_signature.replace(
        parameters=[p for p in original_signature.parameters.values() if not p.name.startswith("_")]
    )


def deprecated_energy_bins(cls):
    """Warn on legacy detector-grid input, preserving explicitness on copies."""
    original = cls.__init__
    parameters = signature(original)

    @wraps(original)
    def initialize(self, *args, **kwargs):
        supplied = parameters.bind_partial(self, *args, **kwargs).arguments
        if "_energy_bins_explicit" not in supplied:
            explicit = "energy_bins" in supplied
            if explicit:
                warn_api(f"{cls.__name__}(energy_bins=)", stacklevel=3)
            kwargs["_energy_bins_explicit"] = explicit
        original(self, *args, **kwargs)

    _hide_compat_parameters(initialize, original)
    cls.__init__ = initialize
    return cls
