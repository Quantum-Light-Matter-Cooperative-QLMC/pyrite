# Electron transport — independent re-derivation

- **id**: `electron-transport`
- **anchor**: `montecarlo/transport.py::simulate_trajectories` (kernels `_dEds_keV_per_ang`, `_dEds_compound_scalar`, `_sigma_browning_cm2`, `_scatter_rates_sr_scalar`, `_alpha_sr_joy`, `_sample_cos_theta_from_alpha`, `_alpha_from_first_moment`, `_rotate_direction_scalar`)
- **source**: Joy & Luo, *Scanning* **11**, 176 (1989); Browning et al., *J. Appl. Phys.* **76**, 2016 (1994); Bishop/Joy screened-Rutherford screening parameter; NIST SRD 64 Mott transport cross sections
- **verifier**: fresh context (did not write the implementation)

Process note: the four closed forms below (Joy–Luo stopping power, the Browning total-elastic fit, the Bishop/Joy screening parameter, and the relativistic screened-Rutherford total cross section) were written from the cited literature and the ledger row before the corresponding function bodies were read; the first-moment identity and the inverse-CDF angle sampler were derived from the screened-Rutherford differential cross section on paper. The numeric reference column in §5 is an independently coded ICRU-37/Berger–Seltzer Bethe stopping power, not a repo helper.

## 1. Intended quantity

Transport $N_e$ independent electrons of initial kinetic energy $E_0$ through a layered slab as a condensed-history single-elastic-scattering Monte Carlo: sample an elastic free path, advance in a straight line, apply continuous slowing-down over the flight, then sample one elastic deflection. The output is the set of straight radiating segments (midpoint, direction, length, energy, flight clock) consumed by the line and bremsstrahlung kernels.

Units at the physics boundary: energies keV, lengths Å, number densities $\AA^{-3}$, cross sections cm$^2$, macroscopic cross sections cm$^{-1}$.

## 2. Independent derivation

### 2.1 Collision stopping power (Joy–Luo)

Joy and Luo modify Bethe's non-relativistic collision stopping power by replacing the mean ionization potential $J$ with an energy-dependent $J^{*} = J/(1 + kJ/E)$, which folds into the logarithm as

$$
\left(\frac{dE}{ds}\right)_{\rm coll}
= -78500\,\frac{\rho Z}{A E}\,
\ln\!\left[\frac{1.166\,(E + kJ)}{J}\right]
\quad [\mathrm{keV\,cm^{-1}}],
$$

with $E$, $J$ in keV, $\rho$ in g cm$^{-3}$, $A$ in g mol$^{-1}$, and the element-dependent $k = 0.731 + 0.0688\log_{10} Z$.

For a compound, Bragg additivity applies per atom. Writing the elemental factor $\rho Z/A$ in terms of the atomic number density $n$ [cm$^{-3}$], $\rho = nA/N_A$, so $\rho Z/A = nZ/N_A$, and with $n_i$ in $\AA^{-3}$ ($n = 10^{24} n_i$),

$$
\frac{\rho Z}{A} \;=\; \frac{10^{24}\,n_i Z}{6.02214076\times 10^{23}}
\;=\; \frac{n_i Z}{0.602214076}.
$$

Hence the compound form, in keV $\AA^{-1}$ (the $10^{-8}$ cm→Å conversion turns the 78500 prefactor into $7.85\times10^{-4}$),

$$
\left(\frac{dE}{ds}\right)_{\rm coll}
= -\frac{7.85\times10^{-4}}{E}
\sum_i \frac{n_i Z_i}{0.602214076}
\ln\!\left[\frac{1.166\,(E + k_i J_i)}{J_i}\right].
$$

Limiting case: as $E \to \infty$ the expression falls as $\ln E/E$; as $E \to J/k\cdot(1.166^{-1}-k)$ the logarithm passes through zero and the stopping power changes sign, so the model is only usable for $E \gg J$ — the enforced cutoff $E_{\rm cut}$ must stay well above $J$.

### 2.2 Elastic total cross sections

**Browning fit to Mott totals.** Browning et al. fit the tabulated Mott total elastic cross sections ($Z \le 92$, 0.1–30 keV) by

$$
\sigma_{\rm el}(E)
= \frac{3.0\times10^{-18}\,Z^{1.7}}
{E + 0.005\,Z^{1.7}\sqrt{E} + 0.0007\,Z^{2}/\sqrt{E}}
\quad [\mathrm{cm^2}],\qquad E\ \text{in keV}.
$$

**Screened Rutherford.** With the Bishop/Joy screening parameter

$$
\alpha(E) = 3.4\times10^{-3}\,\frac{Z^{0.67}}{E},
$$

the relativistically corrected screened-Rutherford total cross section is

$$
\sigma_{\rm SR}(E)
= 5.21\times10^{-21}\,\frac{Z^{2}}{E^{2}}\,
\frac{4\pi}{\alpha(1+\alpha)}
\left(\frac{E + 511}{E + 1024}\right)^{2}
\quad [\mathrm{cm^2}].
$$

The macroscopic cross section for a compound is $\Sigma = \sum_i n_i \sigma_i(E)$ [cm$^{-1}$], and the elastic mean free path in Å is $\lambda = 10^{8}/\Sigma$.

### 2.3 Free-path and element sampling

The flight length to the next elastic collision is exponential with mean $\lambda$, so inverse-CDF sampling gives $s = -\lambda \ln R$, $R \sim U(0,1)$. The scattering element is drawn with probability $n_i\sigma_i/\sum_j n_j\sigma_j$, which is the standard decomposition of a sum of independent Poisson collision processes.

### 2.4 Deflection sampling

The screened-Rutherford differential cross section is $d\sigma/d\Omega \propto \left[1-\cos\theta + 2\alpha\right]^{-2}$. Setting $u = 1-\cos\theta \in [0,2]$, the normalized density is

$$
p(u) = \frac{2\alpha(1+\alpha)}{(u+2\alpha)^{2}},
\qquad
F(u) = \frac{(1+\alpha)\,u}{u + 2\alpha}.
$$

Inverting $F(u)=R$,

$$
\boxed{\;\cos\theta = 1 - \frac{2\alpha R}{1 + \alpha - R}\;}
$$

and the first moment follows by direct integration:

$$
\langle 1-\cos\theta\rangle
= 2\alpha(1+\alpha)\int_0^2\!\frac{u\,du}{(u+2\alpha)^2}
= 2\alpha\Big[(1+\alpha)\ln\!\big(1+\tfrac1\alpha\big) - 1\Big].
$$

This identity is the map between a tabulated NIST SRD 64 transport (momentum-transfer) cross section and the effective screening parameter $\alpha(E)$ used in the sampler; it is monotonic in $\alpha$, so inversion is well posed. The azimuth is uniform on $[0,2\pi)$.

Given a deflection $(\theta,\phi)$ about the incoming unit direction $\hat{\mathbf{d}}$, the new direction is

$$
\hat{\mathbf{d}}' = \cos\theta\,\hat{\mathbf{d}}
+ \sin\theta\cos\phi\,\hat{\mathbf{u}}
+ \sin\theta\sin\phi\,(\hat{\mathbf{d}}\times\hat{\mathbf{u}}),
$$

for any unit $\hat{\mathbf{u}} \perp \hat{\mathbf{d}}$. The choice of $\hat{\mathbf{u}}$ is immaterial because $\phi$ is uniform.

## 3. Comparison with the implementation

| quantity | derived above | implementation | verdict |
| --- | --- | --- | --- |
| Joy–Luo elemental $dE/ds$ | $-7.85\times10^{-4}\rho Z/(AE)\ln[1.166(E+kJ)/J]$ | `_dEds_keV_per_ang` | identical |
| $k(Z)$ | $0.731 + 0.0688\log_{10}Z$ | `_dEds_keV_per_ang`, table build | identical |
| compound weight | $n_i Z_i/0.602214076$ | `coeff_i = (n_i / 0.602214076) * Z_i` | identical |
| Browning $\sigma_{\rm el}$ | see §2.2 | `_sigma_browning_cm2` | identical |
| $\alpha(E)$ | $3.4\times10^{-3}Z^{0.67}/E$ | `_alpha_sr_joy` | identical |
| $\sigma_{\rm SR}$ | see §2.2 | `_scatter_rates_sr_scalar` $\times$ hoisted $5.21\times10^{-21}Z^2 4\pi n$ | identical |
| $\lambda$ [Å] | $10^{8}/\Sigma$ | `lam_ang = 1e8 / total_rate` | identical |
| free path | $-\lambda\ln R$ | `step_j = -lam_ang * np.log(rng.random())` | identical |
| element draw | $\propto n_i\sigma_i$ | cumulative search over `rate_arr` | identical |
| $\cos\theta$ | $1 - 2\alpha R/(1+\alpha-R)$ | `_sample_cos_theta_from_alpha` | identical |
| $\langle 1-\cos\theta\rangle$ inverse | monotone inversion | `_alpha_from_first_moment` (bisection in $\log_{10}\alpha$) | identical |
| frame rotation | $\cos\theta\,\hat{\mathbf d} + \sin\theta(\cos\phi\,\hat{\mathbf u} + \sin\phi\,\hat{\mathbf d}\times\hat{\mathbf u})$ | `_rotate_direction_scalar` | identical |

Two implementation details worth recording, neither a divergence:

- `_rotate_direction_scalar` builds $\hat{\mathbf u}$ from $\hat{\mathbf d}\times\hat{\mathbf e}$ with $\hat{\mathbf e}=\hat x$ when $\lvert d_x\rvert < 0.9$ and $\hat y$ otherwise. The guard is sufficient: in the first branch $\lVert\hat{\mathbf d}\times\hat x\rVert = \sqrt{1-d_x^2} \ge 0.436$; in the second $\lvert d_y\rvert \le 0.436$ so $\sqrt{1-d_y^2} \ge 0.9$. The construction is never degenerate, and its overall sign is irrelevant under uniform $\phi$.
- The free path is sampled at the flight-start energy while the deflection is sampled at the flight-end energy. This is the usual condensed-history convention (the collision physically occurs at the end of the flight) and is consistent to first order in the per-flight energy loss, which the `energy-step-convergence` row measures at $\lesssim 2\%$ for ~99% of flights.

## 4. Filters

- **Units.** $\rho Z/(AE)$ is mol cm$^{-3}$ keV$^{-1}$ against the 78500 keV$^2$ cm$^2$ mol$^{-1}$ prefactor → keV cm$^{-1}$; the $10^{-8}$ factor gives keV $\AA^{-1}$. The compound rewrite is exact (§2.1). $n\sigma$ is cm$^{-1}$; $10^{8}/\Sigma$ is Å. Pass.
- **Limits.** $\alpha \to 0$ recovers unscreened Rutherford ($\langle 1-\cos\theta\rangle \to 0$, forward peaking); $\alpha \to \infty$ gives isotropic scattering ($\langle 1-\cos\theta\rangle \to 1$). The sampler maps the full unit interval onto the full angular range for every $\alpha$: $R \to 0$ gives $u = 0$ ($\cos\theta = 1$) and $R \to 1$ gives $u = 2\alpha/\alpha = 2$ ($\cos\theta = -1$). A single-element compound reduces to the elemental stopping formula. Pass.
- **Signs/conventions.** $dE/ds < 0$ for all $E > J/k$ in the supported range; $\cos\theta \in [-1,1]$; $\lVert\hat{\mathbf d}'\rVert = 1$ (explicitly renormalized). Pass.

## 5. Numeric evidence

All runs on CPU, `element="C"`/`"Si"`, `elastic_model` as noted.

**Kernel identity against independently coded closed forms** (script `/tmp` scratch, formulas as in §2):

| kernel | max relative difference |
| --- | --- |
| `_dEds_keV_per_ang` vs §2.1, C and Si, 1–300 keV | $1.9\times10^{-16}$ |
| `_sigma_browning_cm2` vs §2.2 | $0$ |
| `_alpha_sr_joy` vs §2.2 | $0$ |
| `_scatter_rates_sr_scalar` vs $n\sigma_{\rm SR}$ | $2.4\times10^{-16}$ |

**Sampling exactness.**

| test | result |
| --- | --- |
| $\langle 1-\cos\theta\rangle$, $\alpha \in \{10^{-4},10^{-3},10^{-2},0.1,1\}$, $4\times10^{6}$ draws | $\lvert z\rvert \le 1.7$ against the §2.4 identity |
| `_alpha_from_first_moment` round trip, $\alpha \in \{10^{-4},10^{-2},0.5,5\}$ | recovers $\alpha$ to 6 digits |
| first-flight length, C, 20 keV, $10^{5}$ electrons | mean $287.80 \pm 0.91$ Å vs $\lambda = 287.71$ Å ($z=+0.10$); KS against ${\rm Exp}(\lambda)$ $p = 0.71$ |

**Backscatter coefficient** (20 keV, $N_e = 2\times10^{4}$, bulk slab, binomial $\sigma \approx 0.003$), against the Hunger–Küchler (1979) empirical $\eta(E,Z)$:

| material | `mott` | `sr` | Hunger–Küchler |
| --- | --- | --- | --- |
| Si | 0.1598 | 0.1716 | 0.1636 |
| C | 0.0480 | 0.0524 | 0.0583 |

Silicon agrees within $-2.3\%$ (`mott`) and $+4.9\%$ (`sr`); carbon is low by $-17.7\%$ (`mott`) and $-10.1\%$ (`sr`). The result is insensitive to $E_{\rm cut}$ over 0.1–2 keV ($\Delta\eta < 0.001$), so the low-$Z$ deficit is model form, not cutoff truncation — the expected accuracy band for a single-scattering condensed-history model without inelastic angular deflection.

**Stopping-power validity ceiling.** Joy–Luo against an independently coded relativistic ICRU-37/Berger–Seltzer collision stopping power (no density-effect correction), ratio Joy–Luo / Bethe:

| $E$ [keV] | 1 | 5 | 10 | 25 | 50 | 100 | 200 | 300 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | 1.020 | 0.991 | 0.976 | 0.937 | 0.879 | 0.778 | 0.627 | 0.521 |
| Si | 1.067 | 0.996 | 0.978 | 0.939 | 0.880 | 0.780 | 0.629 | 0.522 |

Joy–Luo is a non-relativistic Bethe modification, so this divergence is expected and is a property of the cited model, not a coding error. It is nonetheless load-bearing: at 300 keV the stopping power is low by a factor $\approx 1.9$, which lengthens CSDA range and per-flight path length in the same proportion. The Browning fit is likewise used above its stated 0.1–30 keV validity window with no guard.

## 6. Verdict

- **Claim**: `electron-transport` — `montecarlo/transport.py::simulate_trajectories` — Joy–Luo slowing-down + Mott/screened-Rutherford elastic scattering → radiating segments.
- **Filters**: units **pass**; limits **pass**; signs/conventions **pass**.
- **Re-derivation**: **matches** — every kernel reproduces its cited closed form to machine precision, and the free-path, element, deflection, and azimuth samplers are statistically exact. No divergent factor, sign, exponent, unit, or convention found.
- **Verdict**: `rederived`, with a documented model-form validity ceiling (§5) rather than an implementation discrepancy.
- **Write-up**: `docs/validation/beam-transport/electron-transport.md`.
- **Suggested ledger change**: status `unverified → rederived`; record the Si/C backscatter agreement and the Joy–Luo/Browning validity ceiling in the notes. The ceiling is the quantitative case for the gated *Reference electron stopping data* and *Reference elastic scattering data* backlog items. A human applies `signed-off`.
