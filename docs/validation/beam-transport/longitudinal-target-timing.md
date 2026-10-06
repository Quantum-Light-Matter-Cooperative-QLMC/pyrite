# Longitudinal target-line timing

**Validation id:** `longitudinal-target-timing`

**Code:** `longitudinal.py::resolve_longitudinal_distribution`

**Status:** rederived in fresh context on 2026-07-29

## Claim and scope

For the task-approved positive reciprocal-harmonic branch, a catalog-pinned basal reflection sets the photon energy used to derive the microbunch period and Gaussian RMS duration. The validated campaign geometry is `theta_obs_deg=90`, `tilt_deg=45`; materials are HOPG and h-BN; electron kinetic energies are 30 and 100 keV; the selected reflection is positive `(0, 0, 2)`; retained longitudinal coherence is `eta=0.9`.

This result is conditional on the positive-harmonic convention. It does not resolve the separate `line-energy-dispersion` discrepancy concerning the mapping between reciprocal-harmonic sign and the spatial Fourier convention.

**2026-10-05 (#338) reconciliation.** That convention is now settled. The line on $+\mathbf g$ takes momentum transfer $-\mathbf g$, so the positive member $\mathbf v\cdot\mathbf g>0$ radiates at $+\mathbf v\cdot\mathbf g/(1-\hat{\mathbf n}\cdot\mathbf v)$. That is the branch used here, so this result no longer rests on an open sign. The Friedel-mate coupling fix in `line-energy-dispersion` changes amplitudes, not line energies, and does not enter this timing.

## Sources and provenance

- Line dispersion: Zhai 2025 Eq. (10), $E_\gamma=\hbar\,\mathbf v\cdot\mathbf g/ (1-\mathbf v\cdot\hat{\mathbf n}/c)$, using the task-approved positive branch.
- Longitudinal form factor: Fourier transform of a centered Gaussian time distribution.
- Electron speed: relativistic kinetic-energy relation $\gamma=1+K/(m_ec^2)$, $\beta=\sqrt{1-\gamma^{-2}}$.
- Lattice constants: bundled catalog CIFs, HOPG $c=6.711$ Å and h-BN $c=6.661$ Å. Therefore $d_{002}=c/2$.
- Independent numerical evaluation used SciPy CODATA constants: $hc=12398.419843320025$ eV Å, $h=4.135667696923859$ eV fs, $\hbar=0.6582119569509067$ eV fs, and $m_ec^2=510998.9506917532$ eV.

## Independent derivation

For a basal `(002)` reflection,

$$
d_{002}=\frac{c_{\rm lattice}}{2},
\qquad
g_{002}=\frac{2\pi}{d_{002}}.
$$

At observation angle $90^\circ$, the Doppler denominator is unity. At crystal tilt $45^\circ$, the positive basal reciprocal vector has projection $g_{002}\cos45^\circ$ along the electron velocity. Thus

$$
k_\gamma
=\frac{\beta g_{002}\cos45^\circ}
       {1-\beta\cos90^\circ}
=\frac{\beta g_{002}}{\sqrt{2}}
\quad [\mathrm{\AA}^{-1}],
$$

$$
E_\gamma=\hbar c k_\gamma
=\frac{hc\,\beta}{\sqrt{2}\,d_{002}}.
$$

Here $k_\gamma$ is the photon angular wavenumber in inverse angstroms. It is not the temporal angular frequency. The latter is

$$
\Omega=\frac{E_\gamma}{\hbar}=c k_\gamma
\quad [\mathrm{rad/fs}].
$$

Consequently,

$$
T=\frac{2\pi}{\Omega}=\frac{h}{E_\gamma}.
$$

For a centered Gaussian time distribution of RMS width $\sigma_t$,

$$
F(\Omega)=\exp\left(-\frac{\Omega^2\sigma_t^2}{2}\right),
\qquad
|F(\Omega)|^2=\exp(-\Omega^2\sigma_t^2).
$$

Solving $|F|^2=\eta$ gives

$$
\sigma_t=\frac{\sqrt{-\ln\eta}}{\Omega}.
$$

## Independent numeric evaluation

| Material | $K$ (keV) | $d_{002}$ (Å) | $\beta$ | $k_\gamma$ ($\AA^{-1}$) | $E_\gamma$ (eV) | $T$ (fs) | $\Omega$ (rad/fs) | $\sigma_t$ (as) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HOPG | 30 | 3.3555 | 0.3283761762 | 0.4347897215 | 857.9574287 | 0.004820364692 | 1303.466793 | 0.249022720 |
| HOPG | 100 | 3.3555 | 0.5482208617 | 0.7258772501 | 1432.3516594 | 0.002887327054 | 2176.125250 | 0.149160921 |
| h-BN | 30 | 3.3305 | 0.3283761762 | 0.4380534185 | 864.3975835 | 0.004784450785 | 1313.251111 | 0.247167387 |
| h-BN | 100 | 3.3305 | 0.5482208617 | 0.7313259609 | 1443.1034359 | 0.002865815155 | 2192.460074 | 0.148049604 |

## Implementation comparison

The implementation independently selects positive catalog-pinned `(002)` for both materials and evaluates the same expressions:

| Material | $K$ (keV) | Implemented $E_\gamma$ (eV) | Implemented $T$ (fs) | Implemented $\Omega$ (rad/fs) | Implemented $\sigma_t$ (as) |
|---|---:|---:|---:|---:|---:|
| HOPG | 30 | 857.9574289698 | 0.004820364674 | 1303.466798054 | 0.249022719 |
| HOPG | 100 | 1432.3516597386 | 0.002887327043 | 2176.125258159 | 0.149160920 |
| h-BN | 30 | 864.3975838187 | 0.004784450767 | 1313.251115709 | 0.247167386 |
| h-BN | 100 | 1443.1034361966 | 0.002865815144 | 2192.460082196 | 0.148049604 |

The differences from the independent table are at most a few parts in $10^9$, caused by rounded implementation constants: `HC_EV_ANG=12398.4198`, `HBARC_EV_ANG=1973.269804`, and `M_E_EV=510998.95`. Because `HC_EV_ANG` and `HBARC_EV_ANG` are rounded independently, $2\pi/\lambda$ reconstructed from the stored wavelength differs from the directly evaluated $k_\gamma$ by $3.19347\times10^{-9}$ relative. This is numerically negligible but is an exact internal-constants inconsistency.

The implementation now names the inverse-angstrom quantity `k_gamma_inv_ang` and reserves $\Omega$ for temporal angular frequency.

## Filters and limiting cases

- Units pass: $\hbar c\,k_\gamma$ is energy; $h/E_\gamma$ is time; $\sqrt{-\ln\eta}/\Omega$ is time.
- $\eta\to1$ gives $\sigma_t\to0$.
- $\eta\to0^+$ gives $\sigma_t\to\infty$.
- $\beta\to0$ gives $E_\gamma\to0$ and $T\to\infty$.
- One spacing period gives $\Omega T=2\pi$.
- The positive `(002)` sign and $45^\circ$ projection give positive energy; the implementation rejects the negative basal family member.

## Anchor and verdict

Anchor: `tests/montecarlo/test_longitudinal_profiles.py::test_target_timing_uses_pinned_basal_reflection_and_h_over_e`. Its four parameterized cases pass. The test pins reflection selection, independent absolute-energy values, $T=h/E_\gamma$, the `eta=0.9` form factor, and one-period spacing.

**Verdict:** `rederived`. The implementation matches the independent derivation and four numeric campaign points, conditional on the approved positive reciprocal-harmonic branch. This verdict does not clear the upstream `line-energy-dispersion` sign discrepancy.
