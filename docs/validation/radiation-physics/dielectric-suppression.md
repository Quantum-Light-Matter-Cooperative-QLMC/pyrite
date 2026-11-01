# `dielectric-suppression`

## Scope and source

Validation: `dielectric-suppression`. This is a fresh-context verification. Before reading any implementation body it used only three inputs: the ledger row, the derivation docstrings of the listed owners, and two sources. Those sources are M. L. Ter-Mikaelian, *High-Energy Electromagnetic Processes in Condensed Media* (Wiley, 1972), and the PDG review *Passage of Particles Through Matter* (dielectric suppression of soft bremsstrahlung; plasma energy $\hbar\omega_p=28.816\sqrt{\rho\langle Z/A\rangle}$ eV). The claim covers several parts. Coupled BremsLib transport multiplies the BremsLib DCS by the leading Ter-Mikaelian factor in four places: the soft radiative stopping, the hard-event rate, the hard photon-energy sampler, and the expected-value continuum scorer. It uses one plasma energy per layer for all four. Each linear $\chi$ cell is integrated exactly, and a vanishing plasma energy recovers the unsuppressed partition exactly.

The BremsLib interpolation itself, its $1/\beta^2$ handling, and the photon-node union belong to [`bremslib-radiative-partition`](bremslib-radiative-partition.md) and are not re-verified here. This document takes the interpolated cell-node values $\chi_i$ as given.

## Cheap filters

| Check | Result |
|---|---|
| Units | $F$ is a ratio of $({\rm eV})^2$, so it is dimensionless. $a=\gamma\hbar\omega_p$ is in eV. Cell zeroth moment: mb. First moment: mb eV. $C=10^{-27}Z^2$ cm$^2$/mb gives cm$^2$ and eV cm$^2$. Pass. |
| Vacuum $\hbar\omega_p=0$ | $F\equiv1$. The code takes a separate branch that calls the original linear-cell formulas, so the result is bit-identical. Pass. |
| $k\gg a$ | $F\to1$, recovering the unsuppressed moments up to $O(a^2/k^2)$. Pass. |
| $k\ll a$ | $F\approx k^2/a^2$, so $\chi F/k\to\chi k/a^2$ and the hard rate is finite as $k_c\to0$. Pass. |
| Sign and range | $0<F\le1$, increasing in $k$, decreasing in $T$ and $\hbar\omega_p$. Pass. |
| Convention for $\gamma$ | PDG writes $k_p=\gamma\hbar\omega_p$ with $\gamma=E/m_ec^2$ ($E$ the total energy), which equals $1+T/m_ec^2$ for kinetic $T$. The docstrings agree. Pass. |

## Independent derivation

### Suppression factor and plasma energy

In a medium of electron density $n_e$, the photon acquires the Drude permittivity $\epsilon(\omega)=1-\omega_p^2/\omega^2$ with

$$
\hbar\omega_p=\hbar\sqrt{\frac{n_e e^2}{\epsilon_0 m_e}}.
$$

The longitudinal formation length of a photon of energy $k$ from an electron of Lorentz factor $\gamma$ is $\ell_f\simeq 2\gamma^2\hbar c/[k(1+\gamma^2\omega_p^2/\omega^2)]$ in the soft-photon limit. Ter-Mikaelian's leading-order result is that the vacuum soft-photon spectrum is reduced by the ratio of the medium to vacuum formation lengths:

$$
F(T,k)=\frac{k^2}{k^2+a^2},\qquad a=\gamma\hbar\omega_p,\qquad \gamma=1+\frac{T}{m_ec^2}.
$$

PDG gives the same form ($k_p=\gamma\hbar\omega_p$; suppression for $k\lesssim k_p$). The $n_e$ here counts all $Z$ electrons per atom: the PDG $28.816\sqrt{\rho\langle Z/A\rangle}$ eV uses $\langle Z/A\rangle$. For a compound the density is $n_e=\sum_i n_iZ_i$, a property of the whole layer, so every element of the layer uses the same $a$. $F$ depends only on $(T,k)$, not on photon angle, so at fixed $(T,k)$ it scales the DDCS without changing its conditional angular density.

### Suppressed moments

With $C=10^{-27}Z^2$ cm$^2$/mb and the BremsLib scaled SDCS $\chi=(k/Z^2)\,d\sigma/dk$ (in mb, with the $1/\beta^2$ already applied by the partition):

$$
S_s=C\int_0^{k_c}\chi F\,dk,\qquad
\sigma_h=C\int_{k_c}^{T}\frac{\chi F}{k}\,dk,\qquad
S_h=C\int_{k_c}^{T}\chi F\,dk.
$$

On one cell $[k_0,k_1]$, $\chi(k)=c_0+c_1k$ with $c_1=(\chi_1-\chi_0)/(k_1-k_0)$ and $c_0=\chi_0-c_1k_0$. Write $x=k/a$. The three primitives

$$
\int\frac{k}{k^2+a^2}dk=\tfrac12\ln(1+x^2),\qquad
\int\frac{k^2}{k^2+a^2}dk=a\,(x-\arctan x),
$$

$$
\int\frac{k^3}{k^2+a^2}dk=a^2\,\tfrac12\bigl[x^2-\ln(1+x^2)\bigr]
$$

were confirmed by symbolic differentiation (SymPy). With $\Delta C=\tfrac12[\ln(1+v^2)-\ln(1+u^2)]$, $\Delta A=(v-u)-(\arctan v-\arctan u)$, $\Delta B=\tfrac12(v^2-u^2)-\Delta C$, $u=k_0/a$, $v=k_1/a$:

$$
\boxed{\int_{k_0}^{k_1}\frac{\chi F}{k}dk=c_0\,\Delta C+c_1a\,\Delta A,\qquad
\int_{k_0}^{k_1}\chi F\,dk=a\bigl(c_0\,\Delta A+c_1a\,\Delta B\bigr).}
$$

These formulas are exact for $k_0=0$ too: the zeroth moment is finite because $F/k\sim k/a^2$.

### Small-$x$ series

As $x\to0$ the differences $x-\arctan x$ and $x^2-\ln(1+x^2)$ cancel catastrophically. The Maclaurin series (SymPy) are

$$
\tfrac12\ln(1+x^2)=\sum_{n\ge0}(-1)^n\frac{x^{2n+2}}{2n+2},\quad
x-\arctan x=\sum_{n\ge0}(-1)^n\frac{x^{2n+3}}{2n+3},\quad
\tfrac12[x^2-\ln(1+x^2)]=\sum_{n\ge0}(-1)^n\frac{x^{2n+4}}{2n+4}.
$$

Truncating after six terms ($n=0..5$) leaves a leading omitted term of relative size $\tfrac{2}{14}x^{12}\approx1.4\times10^{-13}$ at $x=0.1$, below the declared $10^{-12}$. Because the series are evaluated as differences of monomials, $v^p-u^p$, they need no primitive subtraction.

### Limits

- $a\to0$: $\Delta C\to\ln(k_1/k_0)$, $a\Delta A\to k_1-k_0$, $a^2\Delta B\to\tfrac12(k_1^2-k_0^2)$. This gives $c_0\ln(k_1/k_0)+c_1(k_1-k_0)$ and $\tfrac12(\chi_0+\chi_1)(k_1-k_0)$, the vacuum cell moments.
- $S_s+S_h=C\int_0^T\chi F\,dk$ does not depend on $k_c$ for a fixed interpolated $\chi$ and fixed $a$.
- Omission bound: for nonnegative weights $w$, the fraction omitted, $\int w(1-F)\sigma/\int w\sigma$, is at most $\sup(1-F)$. Because $F$ rises with $k$ and falls with $T$ and $\hbar\omega_p$, that supremum is $1-F(T_{\max},k_{\min},\hbar\omega_{p,\max})$.

## Source-to-code comparison

| Derived quantity | Code | Comparison |
|---|---|---|
| $a=(1+T/m_ec^2)\hbar\omega_p$, $m_ec^2=510998.95$ eV | `dielectric_radiative.py::dielectric_scale_eV`; device `_rad_moment` and the sampler inline the same expression with `F64_ELECTRON_REST_EV` | identical |
| $F=k^2/(k^2+a^2)$ | `dielectric_suppression.py::ter_mikaelian_factor` writes it as `square(k/hypot(k, a))`; `brem_bremslib.py::dielectric_scoring_factor` does the same on the (segment, photon) grid; `brem_jit_kernel.py::_bremslib_or_fallback_weighted_scalar` uses float32 `k2/(k2+scale*scale)` | identical; float32 differs by at most $1.1\times10^{-7}$ relative |
| Cell moments, closed form | `dielectric_cell_moments`: `delta_c`, `delta_a`, `delta_b`; `rate = intercept*delta_c + slope*a*delta_a`; `first = a*(intercept*delta_a + slope*a*delta_b)` | term for term identical to the boxed result |
| Small-$x$ branch ($v<0.1$) | six-term loop, `p = 2n+2`; zeroth $\sum(-1)^n[c_0(v^p-u^p)/p+c_1a(v^{p+1}-u^{p+1})/(p+1)]$; first $a\sum(-1)^n[c_0(v^{p+1}-u^{p+1})/(p+1)+c_1a(v^{p+2}-u^{p+2})/(p+2)]$ | matches the series above; the device `_rad_dielectric_cell` is a line-by-line twin |
| Vacuum limit | `scale_eV == 0.0` branch calls the original `_cell_moments` / `_linear_cell` / `_rad_cell_rate` arithmetic, and the zero-start cell contributes only its first moment | exact, bit for bit |
| Partition assembly | `build_radiative_partition`, `radiative_moments_scalar`, `_rad_moment`: each cell's suppressed first moment goes to soft when $k_1\le k_c$; otherwise its zeroth moment goes to the rate and its first moment to the hard first moment | matches the definitions |
| Sampler | `RadiativePartition.sample_photon_energy`, `sample_hard_photon_energy_scalar` and the device sampler bisect the *suppressed* partial zeroth moment, with $\chi$ linear inside the selected cell | inverts the same CDF as the rate |
| Layer plasma energy | `pack_radiative_layer_tables(..., dielectric=True)` computes `plasma_energy_eV(MediumSpec(layer composition))` per layer, with $n_e=\sum_in_iZ_i$; `radiative_core_args` always passes `dielectric=True`; the cores pass `rad_plasma[L]` to both moments and sampler; the CUDA path flattens the same array | one $\hbar\omega_p$ per layer, shared by all elements, same value in all four uses |
| Scorer plasma energy | `brem_events.py::_scoring_plasma_eV` recomputes `plasma_energy_eV` from the scored layer's composition and refuses a value absent from the transport record `plasma_energy_eV_by_layer` (rtol $10^{-10}$); `brem.py` multiplies the BremsLib DDCS by `dielectric_scoring_factor(seg_E, E_grid, plasma_eV)`; strict mode refuses the EEDL/BH fallback | same $F$ and same $\hbar\omega_p$ as transport |
| Omission bound | `dielectric_omission_bound` returns `1 - ter_mikaelian_factor(Tmax, kmin, max plasma)` | matches |

## Numeric checks (independent of implementation helpers)

All references below are mpmath adaptive quadrature at 30–40 digits of the integrand $\chi F/k$ or $\chi F$, written in a scratch script with no PyRITE integration helpers.

- **Cell moments.** $a\in\{0,\ 31.05,\ 80,\ 1853.95,\ 15735.6\}$ eV; the last two are 31.05 eV at 30 MeV and 80 eV at 100 MeV. Cells range from $[0,10^{-3}]$ eV to $[10^5,3\times10^7]$ eV, with three slope patterns. Worst relative error was $2\times10^{-9}$, reached only for the narrow cell $[2999,3001]$ eV. The vacuum branch shows the same $1.6$–$2.3\times10^{-10}$ there. The cause is cancellation between $c_0\ln(k_1/k_0)$ and $c_1(k_1-k_0)$ when $c_0\gg\chi$: the arithmetic of the original cell formula, not anything specific to the dielectric factor. Every other cell agreed to $\lesssim10^{-13}$.
- **Series/closed-form seam.** At $v=0.1\pm10^{-12}$ the error is $\le2\times10^{-13}$ on both sides. A cell straddling the seam, $[0.09a,0.11a]$, gives $10^{-13}$. A very narrow straddling cell, $[0.0999a,0.1001a]$, gives $1.4\times10^{-9}$, the same narrow-cell conditioning as above.
- **Vacuum and $a\to0$.** At $a=0$ the result equals the original formula bit for bit. At $a=10^{-6}$ eV the non-vacuum branch agrees with the vacuum result to $6\times10^{-16}$.
- **Plasma energy.** For Si at $n=0.0499388$ Å$^{-3}$ the code gives `plasma_energy_eV` $=31.04849$ eV. SciPy CODATA constants give $\hbar\sqrt{n_ee^2/\epsilon_0m_e}=31.04849$ eV, and the PDG $28.816\sqrt{\rho Z/A}$ gives 31.04855 eV, which differs only by the rounding of 28.816. A two-element mixture matches the manual $\sum n_iZ_i$ evaluation exactly.
- **Factor parity.** Over $T\in\{0,\,40\ {\rm keV},\,3,\,100\ {\rm MeV}\}$ and $k\in[1,10^6]$ eV, `ter_mikaelian_factor` and `dielectric_scoring_factor` agree with $k^2/(k^2+a^2)$ to $2.2\times10^{-16}$.
- **Full partition.** On the synthetic Si table of `tests/montecarlo/cascades/test_dielectric_radiative.py` I took $T\in\{30,72,100\}$ MeV, $\hbar\omega_p\in\{0,31.0486\}$ eV and $k_c\in\{0.5,10,1000\}$ keV. Integrating the partition's own piecewise-linear $\chi$ nodes with mpmath reproduces $S_s$, $\sigma_h$ and $S_h$ to $\le2\times10^{-14}$. $S_s+S_h$ is cutoff independent to $2.2\times10^{-16}$. The $\chi$ nodes are bit-identical with and without suppression, so the factor is never baked into the interpolant. At 100 MeV, $k_c=1$ keV, the soft first moment falls to 0.002225 of its vacuum value.
- **Sampler.** At $T=50$ MeV, $k_c=1$ keV, $\hbar\omega_p=31.0486$ eV, the independent suppressed CDF evaluated at sampled $k$ for $u\in\{0,0.1,0.37,0.5,0.9,0.999\}$ returns $u$ to $\le7\times10^{-16}$.
- **Anchors.** `test_dielectric_suppression.py`, `test_dielectric_radiative.py` and `test_dielectric_runner_scorer.py`: 50 passed (2026-10-23, CPU). CUDA parity was not re-run here.

## Findings

- No divergent factor, sign, exponent, unit or convention was found.
- The narrow-cell cancellation (relative error up to about $10^{-9}$ on cells about $10^{-3}$ of their position wide) is inherited from the pre-existing vacuum cell formula. It is negligible against any total moment, but anyone tightening host/device parity tolerances on single cells should know about it.
- The device code computes only the soft first moment and the hard rate. The hard first moment is not needed on the device, so this is not a gap.
- The physics scope stays as ledgered. This is the leading high-$\gamma$ Ter-Mikaelian form for an isotropic bulk medium, applied at all energies. At low $\gamma$ it is only indicative. The plasma energy counts all $Z$ electrons as free, which is the PDG convention. The verification checks transcription and consistency; it is not an external benchmark of the suppressed spectrum.

## Verdict

`rederived`. The factor, plasma energy, $\gamma$ convention, exact linear-cell antiderivatives, six-term small-$x$ series, and vacuum branch all match the independent derivation. So does the use of one per-layer $\hbar\omega_p$ in transport moments, sampler and continuum scorer. Human sign-off remains pending.
