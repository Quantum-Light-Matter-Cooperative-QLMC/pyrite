"""Field-ownership inventory for the existing Physics and Numerics schemas.

Descriptive metadata only, not a second validator, resolver, or serialization.
"""

from types import MappingProxyType

PHYSICAL_MODEL = "physical-model"
NUMERICAL_APPROXIMATION = "numerical-approximation"
SAMPLING_CONVERGENCE = "sampling-convergence"
KERNEL_TUNING = "kernel-tuning"
SCHEDULER_RESOURCE = "scheduler-resource"

PHYSICS_FIELD_OWNERS = MappingProxyType(
    {
        "emission": PHYSICAL_MODEL,
        "brem_source": PHYSICAL_MODEL,
        "straggling": PHYSICAL_MODEL,
        "inelastic_model": PHYSICAL_MODEL,
        "inelastic_cutoff_eV": PHYSICAL_MODEL,
        "secondary_threshold_eV": PHYSICAL_MODEL,
        "elastic_model": PHYSICAL_MODEL,
        "bremsstrahlung_model": PHYSICAL_MODEL,
        "radiative_model": PHYSICAL_MODEL,
        "radiative_cutoff_eV": PHYSICAL_MODEL,
        "pair_production_model": PHYSICAL_MODEL,
        "positron_transport": PHYSICAL_MODEL,
        "atomic_electron_deflection": PHYSICAL_MODEL,
        "channeling_model": PHYSICAL_MODEL,
        "channeling_plane": PHYSICAL_MODEL,
        "channeling_axial": PHYSICAL_MODEL,
        "transition_radiation": PHYSICAL_MODEL,
        "transition_lattice_origin_ang": PHYSICAL_MODEL,
        "shell_ionization_model": PHYSICAL_MODEL,
        "photon_transport": PHYSICAL_MODEL,
    }
)

NUMERICS_FIELD_OWNERS = MappingProxyType(
    {
        "cxr_trials": SAMPLING_CONVERGENCE,
        "brem_trials": SAMPLING_CONVERGENCE,
        "cxr_chunk": KERNEL_TUNING,
        "brem_chunk": KERNEL_TUNING,
        "transport_core": KERNEL_TUNING,
        "energy_model": NUMERICAL_APPROXIMATION,
        "max_dE_frac": NUMERICAL_APPROXIMATION,
        "cxr_reflections": SAMPLING_CONVERGENCE,
        "precision": SAMPLING_CONVERGENCE,
        "sampling": SAMPLING_CONVERGENCE,
        "seed": SAMPLING_CONVERGENCE,
        "energy_bins": NUMERICAL_APPROXIMATION,
        "transport": NUMERICAL_APPROXIMATION,
        "radiation": NUMERICAL_APPROXIMATION,
        "execution": KERNEL_TUNING,
        "channeling_basis_size": NUMERICAL_APPROXIMATION,
        "channeling_bloch_cells": NUMERICAL_APPROXIMATION,
        "channeling_dz_ang": NUMERICAL_APPROXIMATION,
        "channeling_zone_max_ang": NUMERICAL_APPROXIMATION,
    }
)
# Scheduler/resource settings are owned by pyrite.execution, not these fields.
