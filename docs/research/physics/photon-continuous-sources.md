---
orphan: true
---

# Continuous BremsLib source sampling for photon histories (#341)

This is a standalone, CPU-only *weighted mean-response* source adapter in
`montecarlo/transport/photon_sources.py`. It **does not** enable the public photon
transport mode, replace the existing default bremsstrahlung estimator, or imply
validation of a thick-target MeV cascade. Use the existing `PhotonAperture`,
`track_photon_histories`/`track_coupled_histories`, and `detected_spectrum` as the
reference scoring path. No source attenuation is applied before tracking.

## Differential source measure and importance weights

For source segment `i` of length `L_i` Å, representative electron energy
`T_i` keV (`E_repr_keV` if present, otherwise `E_keV`), and unit velocity
`v_i`, approximate its unattenuated BremsLib emission measure by

$$
    dN_i = L_i\,dE\,d\Omega\sum_{a\in\mathrm{comp}(i)}
    n_a\,10^{16}\, \frac{d^2\sigma_a(T_i,E,\hat v_i\cdot\hat n)}{dE\,d\Omega}.
$$

Here `n_a` is atoms/Å³; BremsLib `d²σ` is cm²/(eV sr);
`10^16 Å²/cm²` makes `dN_i` dimensionless (photons per input electron
track). This is the *same representative-energy/midpoint approximation* used
by `mc_brem_spectrum`, but with no Beer–Lambert factor and no averaging over
energy or emission direction before sampling.

Choose the segment with proposal probability
`p_i = L_i (E_hi,i - E_lo)/A` where
`A = Σ_i L_i (E_hi,i - E_lo)`, `E_hi,i = min(E_max, 1000*T_i)`,
restricted to segments with positive support. Conditioned on `i`, draw
`E ~ Uniform(E_lo, E_hi,i)`, `Omega ~ Uniform(4π)`, and
`r ~ Uniform(r_mid - L_i v_i/2, r_mid + L_i v_i/2)`.
For `M` independent trials the unnormalized-per-primary history weight is

$$
   w = \frac{4\pi}{M}\frac{L_i(E_{hi,i}-E_{lo})}{p_i}
   \sum_a n_a\,10^{16}\frac{d^2\sigma_a}{dE\,d\Omega}
   = \frac{4\pi A}{M} j_i(E,\Omega).
$$

The second equality reduces rounding and simplifies computation. **Do not
divide by the number of primary electrons here:** the existing
`detected_spectrum(histories, edges_eV, n_primaries)` divides exactly once,
after multiplying detector hits by their source weights. A finite aperture
produces an *aperture-integrated* count/eV/electron spectrum, not photons per
sr; any detector-solid-angle estimator needs independent derivation and
convergence checks before comparison with the directional production path.

### Bremsstrahlung defensive-mixture proposal

The uniform proposal above is the generic `proposal="uniform"` path and the
exact constant-emissivity check. `sample_bremslib_continuum` uses
`proposal="bremsstrahlung"` instead. At MeV the BremsLib source is roughly
`1/E` in photon energy and concentrated within `1/γ` of the electron
direction. Under the uniform proposal, the few trials that land in the
forward cone at low energy then carry most of the estimate. In the #341
10 MeV W full-track check, 5 of 400 batches held 56 % of one redistributed
band, so the batch standard error was not a usable gate.

The bremsstrahlung proposal draws segment `i` with probability
`p_i = L_i ln(E_hi,i/E_lo) / Σ_j L_j ln(E_hi,j/E_lo)`. Given `i`, it draws
energy and direction from defensive mixtures (Owen and Zhou 2000):

$$
q_E(E) = \frac{a}{E\ln(E_{hi,i}/E_{lo})} + \frac{1-a}{E_{hi,i}-E_{lo}},\qquad
q_\Omega(\mu) = \frac{1-b}{4\pi} + b\,\frac{1-\beta_i^2}{4\pi(1-\beta_i\mu)^2},
$$

with `a = b = 0.75`. Here `μ` is the cosine to the segment direction, and
`β_i` comes from the segment's representative kinetic energy. The forward
term is sampled by inverting its CDF on `[-1, 1]`,
`μ = (1 - 1/X)/β` with `X = 2βu/(1-β²) + 1/(1+β)`, and the azimuth is
uniform about the segment. Position stays uniform along the segment. The
weight is

$$
w = \frac{L_i\, j_i(E,\Omega)}{M\,p_i\,q_E(E)\,q_\Omega(\mu)}.
$$

Both densities are positive over the whole supported energy interval and
the full sphere, so the estimator stays unbiased. The uniform and isotropic
shares bound the energy and angle importance factors at `1/(1-a)` and
`1/(1-b)` times those of the uniform proposal. The `(1-βμ)^{-2}` shape is a
Sauter-like importance density only, never a model of the emission. The
observation identity records both proposal names, and the observation model
marker is `bremslib-continuum-photon-only-cpu.v2`.

Proposal factorization does **not** assume factorization of the physical
emissivity. The BremsLib angle-energy cross section is evaluated jointly at
each drawn `(E, direction)`. Since `Omega` covers the entire sphere, initially
off-detector photons can scatter into an aperture. Source positions are
uniform along one segment; spatially changing target composition belongs in
explicit `segment_layer` labels, not an inferred midpoint lookup.

## Usage and boundaries

```python
from pyrite.montecarlo.transport.photon_sources import sample_bremslib_continuum
from pyrite.montecarlo.transport.photon_histories import track_photon_histories, detected_spectrum

sources = sample_bremslib_continuum(
    segments,
    [composition],
    bremslib_tables,
    n_samples=100_000,
    n_primaries=Ne,
    seed=42,
    cutoff_eV=1_000,
    ceiling_eV=100_000,
)
histories = track_photon_histories(
    *sources.track_args,
    layers,
    seed=43,
    photon_cutoff_eV=1_000,
    electron_cutoff_eV=1_000,
    source_kind=sources.source_kind,
    aperture=aperture,
)
counts_per_eV_per_primary = detected_spectrum(histories, energy_bin_edges, Ne)
```

The inputs are already transported electron *segments*, not a depth-integrated
or detector-direction spectrum. `tables` are the prepared per-element BremsLib
tables. No implicit GPU-to-host copy or missing-table/isotropic fallback is
permitted in this reference; energies outside BremsLib coverage raise. The
physical electron's radiative losses must have been modeled when generating
these tracks. The adapter generates **weighted scoring histories**, not analog
emission events, so its incident-energy ledger is only a bookkeeping ledger
for the sampled response. Do not add a histogram of physical hard photons from
these same segments or subtract their energy from the primaries again.

A photon-only call is a scattering/fluorescence reference. Full MeV response
also needs photon-born charged descendant transport via the coupled queue,
with positron and charged vacancy handling still incomplete. Coherent PXR/CBS,
characteristic-source sampling, production mode identity/provenance,
post-collision pixels/filters, and remote benchmarks are separate slices.

## Evidence and status

`tests/montecarlo/test_photon_sources.py` exercises constant-rate analytic
normalization, full-sphere directions, sampled segment positions, an explicit
energy–angle correlation, deterministic replay, empty/invalid input, a
BremsLib synthetic-table check against its angle-integrated SDCS, and the
ordinary-hit denominator of the tracker. Fresh-context verification independently
derived the joint proposal measure, importance weight and unit conversion;
`photon-continuous-source` is `rederived` for both the uniform proposal and the
bremsstrahlung mixture (2026-10-24). See the
[independent derivation](../../validation/radiation-physics/photon-continuous-source.md).
This is not a Geant4/PENELOPE full-track anchor. The feature remains an
experimental reference pending production integration and full-track evidence.
