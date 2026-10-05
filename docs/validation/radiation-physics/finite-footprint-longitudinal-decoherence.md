# Finite-footprint longitudinal decoherence

## Independent derivation (fresh context, 2026-10-05)

This section was written before inspecting implementation bodies or the previous
write-up. Source: the Gaussian characteristic function and conditional expectation;
no external paper equation is claimed. The ledger and `mc_spectrum` signature/docstring
specify a per-electron photon density in photons per eV per sr, with a Gaussian RMS
arrival duration `longitudinal_rms_fs` and finite transverse crystal geometry.

Condition on the entire sampled transverse/transport realization. Let $S_e$ be the
complex field of electron $e$ for one photon energy, polarization, reflection, and
mosaic orientation, including all of its coherently summed segments, transverse
phase, hit/miss history, escape path and attenuation. Let $t_e$ be independent
Gaussian arrival times of variance $\sigma_t^2$, independent of that conditioned
realization. Equal means are permitted: a common timing origin cancels in intensity.
Use physical angular frequency $\Omega=E/\hbar=c\omega$, where the code's $\omega$
has inverse-angstrom units. The total field is

$$
\mathcal A=\sum_e S_e\exp(i\Omega t_e).
$$

The Gaussian characteristic function follows by completing the square in its
normalized Fourier integral:

$$
\phi(\Omega)=\mathbb E[\exp(i\Omega t_e)]
=\exp(i\Omega\bar t)\exp(-\Omega^2\sigma_t^2/2).
$$

For distinct electrons, independence gives

$$
\mathbb E[\exp(i\Omega(t_e-t_f))]=\lvert\phi(\Omega)\rvert^2
=\exp(-\Omega^2\sigma_t^2)=F_z,\qquad e\ne f.
$$

The diagonal factor is exactly one. Expanding the square therefore yields

$$
\boxed{\mathbb E[\lvert\mathcal A\rvert^2\mid\{S_e\}]
=(1-F_z)\sum_e\lvert S_e\rvert^2
+F_z\left\lvert\sum_e S_e\right\rvert^2,
\qquad F_z=\exp[-(\omega c\sigma_t)^2].}
$$

The same operation applies separately to each polarization, reflection and mosaic
orientation, followed by their established intensity sum and per-electron
normalization. It does not introduce cross-reflection interference.

Cheap filters pass: $\omega c\sigma_t$ is dimensionless, $F_z$ lies in $[0,1]$,
and the result is a convex combination of nonnegative intensities with unchanged
units. It is even in frequency and unchanged by reversing the phase convention.
At $\sigma_t=0$ it is the fully coherent sampled finite-footprint field; as
$\lvert\omega c\sigma_t\rvert\to\infty$ it is the grouped intra-electron floor.
For one electron both endpoints are $\lvert S_1\rvert^2$, so every bunch duration
is invariant. For equal in-phase fields the unnormalized result is

$$
\lvert S\rvert^2[N+N(N-1)F_z].
$$

No averaging over transverse positions has occurred. Replacing $S_e$ by an
offset-free field, applying a transverse characteristic function, or dropping
finite escape geometry would change the conditional quantity. Correlation of
arrival time with transverse position, transport, or another electron's timing
invalidates the scalar cross-term factor; its general replacement is the
conditional pair characteristic function. Macro-electron charge weighting is a
separate normalization claim.

## Comparison with implementation

The previous implementation-context derivation and filters agreed with the
independent expression above; they supplied no independent certification. This
revision preserves their conditional scope, assumptions and regression anchors,
and distinguishes physical time from the code's length-valued timing coordinates.

`mc_spectrum` forwards the request into `_setup.py`. For a finite width and height
with a nonzero sampled offset population, setup computes `sigma_z_ang` from the
supplied duration using `C_ANG_PER_FS` and evaluates

```text
finite_footprint_F_host = np.exp(-((omega_host * sigma_z_ang) ** 2))
```

This is precisely the squared characteristic function, with no residual factor
of one half. `seg_r_geom = seg_r` retains the sampled transverse coordinate.
`d_all_geom = seg_t_mid - _matvec3(seg_r, n_hat_d)` removes only the length-valued
arrival offset that the original `d_all` contains as `seg_t0`. The reciprocal
phase still uses `seg_r_geom`; finite-prism escape and formation attenuation
remain attached to the original segments in both reductions. Missed electrons
have zero field and stay in the population normalization.

`_row_decoherence_factor` returns this Gaussian factor directly for the finite
footprint. `_coherent_electron_grouped_row` groups segment fields by electron
before squaring and summing. Both generic routes use the identical coefficients,
phase, formation integral and escape arrays for grouped and flat terms. Their
blend is literal:

```text
spec[:] += ((1.0 - F_row) * grouped_total + F_row * flat_total) * wm
```

The reflection/mosaic weight is applied after blending; rows remain incoherent.
Finalization divides by `Ne` once, preserving per-electron density units.

Static CUDA comparison also matches: per-hkl and batched row-reduction routes
pass each electron's original lines to `_coherent_jit_grouped_row`, accumulate
its squared field into the grouped buffer, and blend against the flat kernel
result. The streaming route retains row planes, squares them for the flat
term, replays original segment blocks with electron group boundaries for the
grouped term, blends row by row, then weights and sums. No CUDA runtime parity
was exercised by this independent verifier.

Exact zero duration is rejected by the public positive-duration input contract;
the short-bunch statement is its limit. Analytic averaging is activated by a
nonzero sampled timing or transverse-offset population. An artificially all-zero
offset population with a positive duration does not activate it. This verifies
the documented finite-footprint branch, not a broader duration-only activation
policy or a transverse ensemble average.

## Numerical evidence and anchors

- Existing CPU anchor:
  `tests/montecarlo/test_coherent_emission.py::test_finite_footprint_partially_coherent_longitudinal_blend`.
  Passed independently (one selected test). It constructs flat and grouped
  reference spectra from decoherence-inactive calls retaining original finite
  positions and computes the Gaussian weight outside the implementation.
- Existing CUDA anchor:
  `tests/montecarlo/test_xray_dispersion_cuda.py::test_coherent_decoherence_gaussian_bunch_finite_footprint_uses_jit_blend`.
  Inspected; not run. GPU evidence remains the owning task's responsibility.
- An independent product Gauss-Hermite integral with 28 nodes per timing
  dimension checks three unequal complex fields against the closed expression
  at dimensionless RMS phases zero, 0.3, one and three.
- A standalone one-electron finite-prism probe compares the offset-active
  spectrum against the offset-free reference for durations spanning the short,
  partial and long regimes; results are recorded below.

## Adjudication

`rederived`: the independent conditional expectation matches the CPU formula
and the inspected CUDA dispatch/reduction structure. Units, limits and phase
conventions pass. No divergent factor, sign, exponent or unit was found in the
implementation. Independent transverse averaging, correlated timing and physical
charge/macroparticle weighting remain outside this claim. Human sign-off remains
pending. Suggested owning ledger edit: `filtered` to `rederived`, recording this
fresh-context verification and keeping the existing anchors.

The quadrature's maximum absolute difference was $1.14\times10^{-13}$ for
intensities between 9.74 and 12.52. The one-electron spectrum's maximum
peak-scaled difference was zero at $10^{-8}$ fs and $6.94\times10^{-16}$ at
$10^{-3}$ and 200 fs. These probes used host float64 and no GPU.
