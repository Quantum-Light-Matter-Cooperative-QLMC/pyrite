"""The CPU transport specializations generated from ``cores``.

Split from :mod:`pyrite.montecarlo.transport.cores` to keep that module's kernel
body within its line budget. Compilation is lazy; only a call compiles a core.
"""

from functools import cache

from .cores import make_cpu_transport_core

_transport_core_ungrooved = make_cpu_transport_core()
_transport_core_ungrooved_lut = make_cpu_transport_core(lut=True)
_transport_core_grooved = make_cpu_transport_core(grooved=True)
_transport_core_ungrooved_perelectron = make_cpu_transport_core(per_electron=True)
_transport_core_ungrooved_perelectron_lut = make_cpu_transport_core(per_electron=True, lut=True)
# Opt-in shell soft/hard inelastic specializations (never selected by default).
_transport_core_ungrooved_inelastic = make_cpu_transport_core(inelastic=True)
_transport_core_ungrooved_lut_inelastic = make_cpu_transport_core(lut=True, inelastic=True)
_transport_core_ungrooved_perelectron_inelastic = make_cpu_transport_core(
    per_electron=True, inelastic=True
)
_transport_core_ungrooved_perelectron_lut_inelastic = make_cpu_transport_core(
    per_electron=True, lut=True, inelastic=True
)
# Opt-in coupled-radiative specializations: exact lockstep, and the exact
# per-electron reference of the CUDA kernel, each with either collision mode.
_transport_core_ungrooved_radiative = make_cpu_transport_core(radiative=True)
_transport_core_ungrooved_inelastic_radiative = make_cpu_transport_core(
    inelastic=True, radiative=True
)
_transport_core_ungrooved_perelectron_radiative = make_cpu_transport_core(
    per_electron=True, radiative=True
)
_transport_core_ungrooved_perelectron_inelastic_radiative = make_cpu_transport_core(
    per_electron=True, inelastic=True, radiative=True
)


@cache
def cone_split_core(*, inelastic, radiative):
    """The exact per-electron CPU core with detector-cone splitting (#203 prototype).

    Validation: detector-cone-variance-reduction
    """
    return make_cpu_transport_core(
        per_electron=True, inelastic=inelastic, radiative=radiative, cone_split=True
    )


@cache
def mixed_elastic_core(*, inelastic, radiative):
    """The exact per-electron CPU core with mixed elastic scattering (#894).

    Validation: mixed-elastic-hinge
    """
    return make_cpu_transport_core(
        per_electron=True, inelastic=bool(inelastic), radiative=bool(radiative), mixed=True
    )


def exact_ungrooved_core(*, per_electron, inelastic, radiative):
    """The exact (non-LUT) ungrooved CPU core for a collision/radiative mode."""
    return {
        (False, False, False): _transport_core_ungrooved,
        (False, True, False): _transport_core_ungrooved_inelastic,
        (False, False, True): _transport_core_ungrooved_radiative,
        (False, True, True): _transport_core_ungrooved_inelastic_radiative,
        (True, False, False): _transport_core_ungrooved_perelectron,
        (True, True, False): _transport_core_ungrooved_perelectron_inelastic,
        (True, False, True): _transport_core_ungrooved_perelectron_radiative,
        (True, True, True): _transport_core_ungrooved_perelectron_inelastic_radiative,
    }[(bool(per_electron), bool(inelastic), bool(radiative))]
