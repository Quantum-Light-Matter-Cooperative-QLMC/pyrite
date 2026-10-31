"""Physical bunch population and distinct-pair Monte Carlo normalization.

Validation: coherent-physical-bunch-population
"""

import warnings

import numpy as np
from scipy.constants import elementary_charge

from ..._backend import _to_cpu, xp

# v2 (#370): finite footprints pair offset-free fields with F_z F_perp, and
# tilted analytic spots carry the face-arrival delay in t0. v3 (#365): the
# default flat omission certifies finite-footprint rows by F_z F_perp, not F_z.
# v4 (#465): linear bunch-front timing shifts the joint transverse factor,
# including analytic averaging and omission bounds on infinite slabs.
COHERENT_POPULATION_MODEL = "physical-distinct-pairs-v4"


class CoherentSamplingError(ValueError):
    """A physical coherent estimate is unresolved by its incident sample."""

    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics


class CoherentSamplingWarning(UserWarning):
    """A physical coherent estimate has finite negative bins, returned unclipped."""

    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics


def physical_bunch_electrons(case):
    """Convert bunch charge in pC to physical electrons; default is 1 pC.

    Source: SI elementary charge, N = Q/e. Zero charge has zero detected
    flux; its per-incident-electron spectrum retains the self term.
    Validation: coherent-physical-bunch-population
    """
    charge = float(case.get("bunch_charge_pc", 1.0))
    if not np.isfinite(charge) or charge < 0.0:
        raise ValueError("bunch_charge_pc must be finite and nonnegative")
    population = charge * 1e-12 / elementary_charge
    if not np.isfinite(population) or 0.0 < population < 1.0:
        raise ValueError("a nonzero coherent bunch must contain at least one physical electron")
    return population


def pair_scale(physical_electrons, incident_samples):
    """Raw cross-pair weight (N-1)/(M-1), before final division by M.

    Source: independent-electron ensemble power per physical electron,
    E|S|² + (N-1) F |E S|². Estimate the second moment by G/M and
    |E S|² by (P-G)/(M(M-1)), where G=sum|S_e|², P=|sum S_e|².
    Assumptions: equally weighted iid incident trajectories, missed entries
    have zero field. Infinite slabs pair complete fields (F=1). A finite
    footprint pairs offset-free fields and averages independent Gaussian
    arrival times and a recorded Gaussian spot analytically, F=F_z F_perp.
    Limits: N=1 gives the self term; identical aligned fields give N|S|²;
    M=N with no offsets recovers the historical coherent sum. No self pairs
    enter the excess.
    Validation: coherent-physical-bunch-population
    """
    population = float(physical_electrons)
    if not np.isfinite(population) or population < 0 or 0 < population < 1:
        raise ValueError("physical_electrons must be zero or finite and at least one")
    if incident_samples < 1:
        raise CoherentSamplingError("coherent emission needs a positive incident sample count")
    if population <= 1:
        return 0.0
    if incident_samples < 2:
        raise CoherentSamplingError(
            "physical coherent cross-electron power requires at least two incident samples; "
            "increase --cxr-trials"
        )
    return (population - 1.0) / (incident_samples - 1)


def mixed_row_power(st, grouped, flat, factor):
    """Raw row estimator; historical sampled-bunch semantics if N is absent.

    Source: the distinct-pair expansion in pair_scale. Assumes grouped and
    flat powers describe the same row fields and factor is their supported
    independent arrival-time average (or one for complete sampled fields).
    F=0 and N=1 retain the grouped self term; aligned fields yield N after
    final division by the incident sample count.
    Validation: coherent-physical-bunch-population
    """
    if st.request.physical_electrons is None:
        return (1.0 - factor) * grouped + factor * flat
    scale = pair_scale(st.request.physical_electrons, st.Ne)
    return grouped + scale * factor * (flat - grouped)


def undersampling_ratio(physical_electrons, incident_samples):
    """Cross-pair weight (N-1)/(M-1) a failed estimate carried, or ``None``.

    Its noise relative to the self term scales with this weight, so a value
    above one means the sample cannot resolve an excess near zero.
    """
    if physical_electrons is None or incident_samples is None or incident_samples < 2:
        return None
    if physical_electrons <= 1:
        return None
    return pair_scale(physical_electrons, incident_samples)


def require_resolved_power(
    power, *, energy_eV=None, incident_samples=None, physical_electrons=None, quantity="power"
):
    """Refuse nonfinite estimates; warn on finite negative bins, never clip.

    Source: exact intensity is a squared field norm and nonnegative.
    Assumes an unclipped signed Monte Carlo estimate; zero power passes.
    Finite negative bins emit :class:`CoherentSamplingWarning` and the signed
    estimate is returned unchanged: the refusal is suspended (#462) because
    it rejected whole spectra over undersampled tail bins far from the band
    of interest. This is a necessary check, not a statistical convergence
    certificate.
    Validation: coherent-physical-bunch-population
    """
    finite = xp.isfinite(power)
    negative = finite & (power < 0)
    invalid = ~finite | negative
    if not bool(_to_cpu(xp.any(invalid))):
        return
    # Reduce on the active backend; never copy the full failed spectrum to CPU.
    nonfinite_count = int(_to_cpu(xp.count_nonzero(~finite)))
    negative_count = int(_to_cpu(xp.count_nonzero(negative)))
    first = int(_to_cpu(xp.argmax(invalid)))
    minimum = (
        float(_to_cpu(xp.min(xp.where(finite, power, xp.inf))))
        if nonfinite_count < power.size
        else None
    )
    diagnostics = {
        "quantity": quantity,
        "total_count": int(power.size),
        "negative_count": negative_count,
        "nonfinite_count": nonfinite_count,
        "minimum_finite_raw_power": minimum,
        "first_invalid_index": first,
        "incident_samples": int(incident_samples) if incident_samples is not None else None,
        "physical_electrons": float(physical_electrons) if physical_electrons is not None else None,
    }
    if energy_eV is not None:
        diagnostics["first_invalid_energy_eV"] = float(_to_cpu(energy_eV[first]))
    if nonfinite_count:
        remedy = "check numerical inputs and arithmetic"
    else:
        remedy = "increase --cxr-trials and check sampling convergence"
        ratio = undersampling_ratio(physical_electrons, incident_samples)
        if ratio is not None:
            diagnostics["pair_weight"] = ratio
            if ratio > 1.0:
                # Cross-pair noise ~ (N-1)/(M-1) x self term, so trials must
                # approach N before an excess near zero resolves.
                remedy = (
                    f"the cross-electron excess is weighted by (N-1)/(M-1) = {ratio:.3g} "
                    f"(N = {physical_electrons:.4g} bunch electrons, M = {incident_samples} "
                    "trials), so its sampling noise swamps a near-zero coherent excess; "
                    "remove sampled offsets (transverse=PointSpot(), "
                    "longitudinal=PointBunch()), lower bunch_charge_pc until N <= M, use a "
                    "finite Slab footprint with a Gaussian LongitudinalDistribution so offsets are averaged "
                    "analytically, or raise --cxr-trials toward N"
                )
    if not nonfinite_count:
        warnings.warn(
            CoherentSamplingWarning(
                "physical coherent power is unresolved: negative pair estimate; "
                f"{quantity}: {negative_count} finite negative of {power.size}; "
                f"first negative index {first}; {remedy}; negative bins are "
                "returned unclipped",
                diagnostics=diagnostics,
            ),
            stacklevel=2,
        )
        return
    raise CoherentSamplingError(
        "physical coherent power is unresolved: negative or nonfinite pair estimate; "
        f"{quantity}: {negative_count} finite negative, {nonfinite_count} nonfinite "
        f"of {power.size}; first invalid index {first}; "
        f"{remedy}; no clipping is applied",
        diagnostics=diagnostics,
    )
