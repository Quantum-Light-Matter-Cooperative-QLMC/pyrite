"""Resolved profile numerics and their stable CLI/JSON presentation."""

from collections.abc import Mapping
from dataclasses import dataclass

from .._numerics import CHANNELING_KEYS

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
            ("secondary_threshold_eV", "secondary threshold (eV)"),
            ("elastic_model", "elastic model"),
            ("bremsstrahlung_model", "bremsstrahlung model"),
            ("radiative_model", "radiative model"),
            ("radiative_cutoff_eV", "radiative cutoff (eV)"),
            ("pair_production_model", "pair production model"),
            ("positron_transport", "positron transport"),
            ("atomic_electron_deflection", "atomic-electron deflection"),
            *((key, key.replace("_", " ")) for key in CHANNELING_KEYS),
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
