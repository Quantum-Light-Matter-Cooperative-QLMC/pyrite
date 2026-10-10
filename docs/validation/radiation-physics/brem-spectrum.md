# `brem-spectrum`

```{note}
**2026-09-25 (#174):** The Cartesian incident-panel interpolation derived below has been replaced by unit-base refinement with a $\ln T$ shape weight. See [Bremsstrahlung](../../physics/radiation-physics/bremsstrahlung.md). The ledger status is back to `filtered` until a fresh-context re-derivation covers the new interpolation. The derivation below remains valid for native panels, the MF=23 total and the isotropic estimator.
```

## Scope and independence

Validation: `brem-spectrum`. Fresh-context validation on 2026-09-13. The independent expression below was written before implementation bodies or the previous implementation-context report were read. Inputs were the target ledger row, function signatures/docstrings, and primary ENDF documentation. This checks the stated isotropic track-length model, not its accuracy against experiment.

## Sources

- [ENDF-6 Formats Manual, revision 215](https://nds.iaea.org/public/endf/endf-manual.pdf), sections 0.5.2.2 (Cartesian interpolation, equation 7), 6.2.3.1 (equation 6.3), 23.3 and 26.1–26.2.1. File 23 uses barns. File 26 gives unit product yield; its photon subsection is isotropic and precedes the electron energy-transfer subsection. The zeroth coefficient is integrated over angle, hence is a photon energy probability density, not a density per steradian.
- [LLNL EPICS2025 release](https://nuclear.llnl.gov/EPICS/index.html) and [2025 status report](https://nuclear.llnl.gov/EPICS/DOCUMENTS/2025-EPICS2025-Status.pdf). The release reports the August 2023 photon-distribution correction and unchanged numerical values in January 2025. The ledger cites NDS-IAEA-226; its full report was not retrieved during the initial derivation.
- [Koch and Motz, Rev. Mod. Phys. 31, 920 (1959)](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.31.920). The fallback expression is specified by the function docstring. Its source equation has not yet been independently recovered; matching that expression alone cannot certify the fallback's physical accuracy.

## Independent derivation (frozen before implementation inspection)

Let $T$ and $k$ denote electron kinetic and photon energy in eV. Let $\sigma_a(T)$ be the total bremsstrahlung cross section for element $a$, and $p_a(k\mid T)$ its normalized photon probability density. Then

$$
\frac{d\sigma_a}{dk}=\sigma_a(T)p_a(k\mid T),\qquad
\frac{d^2\sigma_a}{dk\,d\Omega}=\frac{\sigma_a(T)p_a(k\mid T)}{4\pi}.
$$

The angular factor follows also by integrating the zeroth Legendre term over azimuth. It is the isotropic File-26 approximation; it does not describe physical relativistic forward beaming.

Normalize each piecewise-linear native panel $q_i$ by its trapezoidal area $A_i=\int q_i(k)\,dk$, obtaining $p_i=q_i/A_i$. At an incident energy between panels, $w=(T-T_i)/(T_{i+1}-T_i)$, declared Cartesian lin-lin interpolation gives

$$
\widetilde p(k\mid T)=(1-w)p_i(k)+wp_{i+1}(k),\qquad
p(k\mid T)=\frac{\widetilde p(k\mid T)\mathbf{1}_{0<k\le T}}{C(T)},
\quad C(T)=\int_0^T\widetilde p(k\mid T)\,dk.
$$

Outside a panel's native photon support its density is zero. Truncation and renormalization are a declared PyRITE processing choice beyond simple Cartesian interpolation: they enforce energy support and preserve the tabulated total. For a linear interval beginning at $x$ with density $p_x$ and slope $m$, its partial contribution to $C$ is $p_xh+mh^2/2$. This normalization must be independent of the requested output grid. A finite native lower photon cutoff remains a library convention; this construction makes no prediction below it. The total cross section is interpolated independently on its own declared energy mesh.

For trajectory segment $s$ of length $L_s$, number density $n_a$, and a photon escape optical depth $\tau_s(k)=\sum_\ell\mu_\ell(k)d_{s\ell}$, the track integral is

$$
S(k,\hat{\mathbf n})=\frac{1}{4\pi N_e}\sum_s\sum_a n_a
\int_0^{L_s}\sigma_a(T_s(l))p_a(k\mid T_s(l))
\exp[-\tau_s(k,l)]\,dl.
$$

Freezing representative energy and escape depth within each segment reduces it to $n_aL_s\sigma_ap_a\exp(-\tau_s)/(4\pi N_e)$. This is segment quadrature, not exact integration when energy or escape depth varies along the segment. $N_e$ counts incident electrons, including ones contributing no surviving rows.

With $n_a$ in atoms per cubic angstrom and $L_s$ in angstrom, the conversions are $n_a^{\rm cm}=10^{24}n_a$, $L_s^{\rm cm}=10^{-8}L_s$, and $\sigma^{\rm cm^2}=10^{-24}\sigma^{\rm barn}$. Thus the result has units photons per incident electron per eV per steradian. Optical depth is nonnegative and dimensionless when $\mu$ and $d$ use reciprocal length units.

Filters: vanishing density/length gives zero; zero absorption gives the unattenuated track estimator; large positive optical depth suppresses emission; energies above $T$ contribute zero; normalized spectra integrate to the total cross section; all factors are nonnegative. In vacuum the only angular factor is $1/(4\pi)$. Compound species add incoherently by number density.

For the docstring's analytic fallback, define dimensionless momenta $p_j=\sqrt{(1+T_j/(m_ec^2))^2-1}$ and $\beta_j=p_j/(1+T_j/(m_ec^2))$, with $T_f=T_i-k$. Its stipulated expression is

$$
\frac{d\sigma}{dk}=\frac{16}{3}\alpha r_e^2 Z^2
\frac{1}{k p_i^2}\ln\frac{p_i+p_f}{p_i-p_f}
\frac{\beta_i}{\beta_f}
\frac{1-\exp(-2\pi Z\alpha/\beta_i)}{1-\exp(-2\pi Z\alpha/\beta_f)}.
$$

Here $k$ must be in eV for an output per eV; using keV requires a further $10^{-3}$. This is a nonrelativistic Born expression with Elwert correction and relativistic momenta substituted; it is not the full relativistic Bethe–Heitler cross section. The stipulated validity range is roughly $Z\lesssim30$ and $T\lesssim100$ keV. Missing-coverage fallback outside that range is an explicit accuracy limitation. As $k\to T^-$ the logarithm and final inverse velocity cancel to a finite limit, while an implementation may assign zero exactly at the endpoint without changing its integral. No finite photon-number total is implied as $k\to0$ for this unscreened analytic expression.

## Comparison with implementation

The EEDL path matches the frozen expression. `_extract_bremsstrahlung_table` requires the supported lin-lin laws, unit photon yield, no discrete or higher angular coefficients, ordered positive energies, nonnegative coefficients, and native area within 0.999–1.001 before normalization. The loader verifies packaged bytes and converts barns by $10^{-24}$. `_prepare_eedl_grid` evaluates each panel at fixed absolute photon energy with zero outside its support. `_piecewise_linear_cdf_at` uses the partial-interval quadratic integral derived above; `_prepare_eedl_segment_state` forms $C(T)$ and interpolates the total cross section separately. `_evaluate_prepared_eedl` multiplies the mixed PDF by $\sigma/C$ and enforces $0<k\le T$.

`mc_brem_spectrum` uses `E_repr_keV` when present, otherwise `E_keV`; it applies escape depth at `r_mid`, sums species by their number densities, converts length and density to centimetres, then divides by $4\pi N_e$. Its layered exponent is the sum of path length times each layer's attenuation coefficient. `_eedl_or_bh_weighted_scalar` and `_eedl_kernel_1e` use the same staged expression and exponent on CUDA. This is a static comparison, not device execution. The diagnostic `brem_endpoint_quadrature_error` delegates both sampled energies to the selected cross-section backend; its error-calibration claim belongs to `radiation-error-estimators`.

The analytic backend matches its docstring algebra away from its numerical endpoint guard. The exact support in `_brem_dsigma_dk_core` and the CUDA scalar is $T_i-k>10^{-6}$ keV, so the last $10^{-3}$ eV below the endpoint is also zeroed. The frozen derivation described endpoint exclusion only; this additional finite guard is a numerical approximation, not a derived physical cutoff. It does not establish an error in ordinary eV-binned spectra, but endpoint limit tests must state it. The logarithm's denominator is also floored at $10^{-30}$ in dimensionless momentum. Full source-to-code verification of this hybrid fallback remains incomplete.

## Numerical evidence

All new execution used `PYRITE_MC_BACKEND=cpu` and `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run`. No heavy transport or GPU work ran. A direct `endf-parserpy.EndfFile` read of the packaged tape, bypassing PyRITE's loader/interpolator, recovered the carbon MF=23 bracket:

| Incident energy (eV) | Cross section (barn) |
|---|---|
| 27404.5 | 45.5858 |
| 31552.1 | 41.2166 |

Linear interpolation gives $42.85162635741151$ barn at 30 keV, matching the existing regression value. This independently checks arithmetic/selection from the packaged primary data; it is not an independent evaluation of atomic theory or a separately downloaded tape checksum comparison.

For C, Si and W, a separate calculation integrated normalized native panels using inserted cutoff knots and trapezoids, without calling PyRITE's CDF or segment-state helpers. It compared five photon energies (100 eV, 1 keV, $\min(T/2,10\ \mathrm{keV})$, $T$, $T+1$ eV) at four incident energies. Maximum relative difference on positive bins was $4.17\times10^{-16}$; the forbidden bin was zero. This verifies evaluator arithmetic conditional on parsed panels.

The table quantifies probability discarded above $T$ by PyRITE's processing choice; percentages are before renormalization:

| Element | 25 keV | 30 keV | 100 keV | 300 keV |
|---|---|---|---|---|
| C | 0.35359% | 0.44955% | 0.60325% | 0.12048% |
| Si | 0.39172% | 0.50201% | 0.73278% | 0.14110% |
| W | 0.51421% | 0.64194% | 1.18674% | 0.12013% |

The largest corresponding increase of surviving density is 1.20099%, for W at 100 keV. These twelve points are not a global error bound, and renormalization agreement does not establish that the processed spectrum is experimentally more accurate than other physically supported interpolation choices.

A bounded attenuation probe used one 100-angstrom C segment at 30 keV, density 0.1 atoms per cubic angstrom, two incident electrons (only one emitting), a 50-angstrom escape path, and a 1-keV photon. Injected finite coefficients $\mu=0$, $0.02$, and $2$ inverse angstrom gave respectively $1.448373123340891\times10^{-11}$, $5.328266952223835\times10^{-12}$, and $5.388058060454713\times10^{-55}$ photons per eV per steradian per incident electron. The ratios are $1$, $\exp(-1)$ and $\exp(-100)$.

> **Superseded by issue #274 (2026-10-01).** `mc_brem_spectrum` now passes its coefficients through `materials/attenuation.py::_finite_mu_or_raise`: an undefined coefficient at a node `E ≥ 1 eV` raises, and only sub-eV nodes keep zero attenuation. EPDL2025 coefficients are finite at 1 eV. The paragraph below records the policy as validated at the time.

Nonfinite coefficients follow a separate policy: `nan_to_num` in `mc_brem_spectrum` maps NaN and either infinity to zero attenuation. Injecting positive infinity or NaN therefore returns the vacuum value, not the opaque limit. No ordinary finite-opacity failure was demonstrated. A real coefficient lookup at positive photon energy 1 eV returns NaN for C and W, demonstrating an out-of-coverage case. This policy must be excluded from a claim of validated self-absorption at all energies; unknown absorption is not physical transparency. Direct 30, 50 and 100 keV lookups were finite for both elements, so the old inline comment claiming missing attenuation above 30 keV is not a current coverage bound.

The focused command

```bash
PYRITE_MC_BACKEND=cpu pyrite-dev test tests/montecarlo/test_bremsstrahlung_eedl.py tests/montecarlo/test_chunk_invariance.py tests/montecarlo/test_radiation_error_estimators.py
```

passed **37 tests**, with five existing warnings from line-radiation square-root and error-estimator threshold paths. This covers source total/native and interpolated normalization, default/explicit backend selection, warning and failure policy, isotropic assembly, chunking and diagnostic selection. Equality against the legacy fallback is a behavior anchor, not independent physics validation of that fallback.

## Analytic Bethe--Heitler/Elwert fallback: independent verification, 2026-09-13

Closes the gap left open above: the Koch--Motz source expression for the analytic fallback backend is recovered here from primary literature, then compared term by term against `montecarlo/spectrum/brem.py::_brem_dsigma_dk`, its fused core `::_brem_dsigma_dk_core`, and the CUDA scalar twin `montecarlo/spectrum/brem_jit_kernel.py::_dsigma_weighted_scalar` with its hoisted `::_brem_incident_prefactor_core`. The EEDL path is out of scope and is not revisited. Source recovery preceded reading any implementation body; only the ledger row and the two function docstrings were read first.

### Source recovery and what could not be retrieved

- Koch and Motz, *Rev. Mod. Phys.* **31**, 920 (1959) could **not** be retrieved in full: the APS record is paywalled and no open full text was reachable. The formula labels, their bracketed relativistic form, and the stated validity ranges quoted below are therefore reconstructed from independent primary sources rather than read off the review itself. This is stated so the reconstruction is not mistaken for a direct quotation.
- Elwert, *Ann. Phys.* **34**, 178 (1939) could not be retrieved either. Its correction factor was recovered from two independent modern sources that quote it directly.
- [Itoh, Kawana, Nozawa and Kohyama, arXiv:astro-ph/9804204](https://arxiv.org/abs/astro-ph/9804204), equations (1), (2) and (34). Equation (1) gives the relativistic Bethe--Heitler (1934) cross section multiplied by the Elwert (1939) factor; equation (2) defines the Sommerfeld parameters; equation (34) gives the non-relativistic Elwert-corrected Gaunt factor.
- [RHESSI thick-target bremsstrahlung code documentation](https://hesperia.gsfc.nasa.gov/ssw/packages/xray/doc/brm_thick_doc.pdf), which states that the relativistic Bethe--Heitler cross section *is* Koch and Motz equation 3BN, and that the Elwert (1939) factor is applied to it multiplicatively. This fixes the label correspondence used below.

From these, the implemented expression is Koch and Motz formula **3BN(a)** -- the non-relativistic, unscreened, angle-integrated Born limit of 3BN -- carrying the multiplicative **Elwert** factor. 3BN itself is angle-integrated, unscreened, Born, and valid at all energies; 3BN(a) is its $T_i\ll m_ec^2$ specialization. Neither is 2BN, which is doubly differential in photon energy and photon angle.

### Independent expression, written before reading the implementation body

Work in $m_ec^2=511$ keV units. For incident and final kinetic energies $T_i$ and $T_f=T_i-k$, the Lorentz factors, dimensionless momenta and velocities are

$$
\gamma_j=1+\frac{T_j}{m_ec^2},\qquad
p_j=\sqrt{\gamma_j^2-1},\qquad
\beta_j=\frac{p_j}{\gamma_j},\qquad j\in\{i,f\}.
$$

Equation (1) of Itoh et al. is the Born cross section

$$
\frac{d\sigma}{dk}=\alpha Z^2r_e^2\frac{p_f}{p_i}\frac{1}{k}
\,f_{\rm E}\,\bigl\{\,\cdots\bigr\},
$$

whose brace is the Bethe--Heitler bracket. Taking $T_i\ll m_ec^2$, the bracket together with $p_f/p_i$ collapses to the familiar non-relativistic soft form, so that

$$
\frac{d\sigma}{dk}\Big|_{\rm 3BN(a)}
=\frac{16}{3}\,\alpha\,r_e^2\,Z^2\,\frac{1}{k\,p_i^2}
\ln\frac{p_i+p_f}{p_i-p_f}.
$$

This is checked against the Gaunt-factor convention independently: with $d\sigma/dk=\tfrac{16\pi}{3\sqrt3}\alpha r_e^2Z^2/(k\beta_i^2)\,g_{\rm ff}$ and equation (34)'s $g=\tfrac{\sqrt3}{\pi}\ln\lvert (p_f+p_i)/(p_f-p_i)\rvert\,f_{\rm E}$, the $\pi$ and $\sqrt3$ cancel and the prefactor $16/3$ with a single inverse squared incident momentum is recovered. In the non-relativistic regime where 3BN(a) is derived, $p_i$ and $\beta_i$ coincide, so $1/p_i^2$ and $1/\beta_i^2$ are the same expression; the choice of $p_i^2$ is the one Koch and Motz carry.

The Elwert factor, from equation (2) with $a_j=\alpha ZE_j/(p_jc)=\alpha Z/\beta_j$, is

$$
f_{\rm E}=\frac{a_f}{a_i}\,
\frac{1-\exp(-2\pi a_i)}{1-\exp(-2\pi a_f)}
=\frac{\beta_i}{\beta_f}\,
\frac{1-\exp(-2\pi Z\alpha/\beta_i)}{1-\exp(-2\pi Z\alpha/\beta_f)}.
$$

The ordering is fixed and not symmetric: the **initial** velocity supplies the numerator exponential and the numerator of the velocity ratio; the **final** velocity supplies the denominator of both. Writing it the other way inverts the correction.

Units: $r_e^2$ in cm$^2$, $\alpha$, $Z^2$, $p_i^2$, the logarithm and $f_{\rm E}$ all dimensionless, so $d\sigma/dk$ carries cm$^2$ divided by whatever energy unit $k$ is expressed in. A result in cm$^2$/eV therefore requires $k$ in **eV**; a $k$ in keV would leave cm$^2$/keV, a factor $10^3$ high.

Limiting cases fixed in advance:

- $\beta_i,\beta_f\to1$ (or $Z\alpha\ll\beta$): $1-e^{-2\pi a}\to2\pi a$, so $f_{\rm E}\to (a_f/a_i)(a_i/a_f)=1$ and the plain Born 3BN(a) is recovered.
- $k\to T_i^-$: $p_f\to0$ makes the logarithm vanish as $2p_f/p_i$ while $\beta_i/\beta_f$ diverges as $1/\beta_f$ and the denominator $1-\exp(-2\pi Z\alpha/\beta_f)\to1$. The product is **finite and non-zero** at the endpoint -- the Elwert factor precisely cancels the Born zero. A correct implementation must not produce a spike or a zero there.
- $k\to0$: $p_i-p_f\propto k$, so $k\,d\sigma/dk$ grows only as $\ln(1/k)$. The unscreened expression has an integrable-energy but log-divergent-number soft tail; no finite photon count is implied.
- Positivity: $p_i>p_f>0$ makes the logarithm positive, and both exponential factors lie in $(0,1)$, so the whole expression is strictly positive.

Validity, as reconstructed: 3BN(a) requires $T_i\ll m_ec^2$, and the Elwert correction requires the Sommerfeld parameter to stay small, $\alpha Z/\beta\ll1$, which for the endpoint value of $\beta_f$ restricts it to low $Z$. Independent sources place the Elwert-corrected Born cross section as accurate for $Z\lesssim26$ in the non-relativistic regime. The docstring's "$Z\lesssim30$, $T\lesssim100$ keV" is consistent with that and slightly optimistic in $Z$.

### Term-by-term comparison with the implementation

`_brem_dsigma_dk` divides `k_eV` by `1e3` to keV, calls the fused core, and tags `Z` as `REAL` first -- the comment records that an untyped scalar makes `cupy.fuse` infer float16 for the $\sim3\times10^{-27}$ prefactor and flush it to zero. The core is:

```text
T_f = T_i - k
ok  = (T_f > 1e-6) & (k > 0.0)
p_i = sqrt(T_i * (T_i + 2*mc2)) / mc2
p_f = sqrt(T_f * (T_f + 2*mc2)) / mc2
beta_i = p_i / (1 + T_i/mc2)
beta_f = p_f / (1 + T_f/mc2)
born_log = log((p_i + p_f) / maximum(p_i - p_f, 1e-30))
elwert   = beta_i/beta_f * (1 - exp(-2*pi*Z*ALPHA_FS/beta_i))
                         / (1 - exp(-2*pi*Z*ALPHA_FS/beta_f))
dsig = 16/3 * ALPHA_FS * R_E_CM2 * Z**2
       / maximum(k*1e3, 1e-30) / p_i**2 * born_log * elwert
```

| Term | Independent expression | Code | Agreement |
| --- | --- | --- | --- |
| Prefactor | $\tfrac{16}{3}\alpha r_e^2Z^2$ | `16/3 * ALPHA_FS * R_E_CM2 * Z**2` | matches |
| $Z$ dependence | $Z^2$ in prefactor, $Z^1$ in $f_{\rm E}$ exponents | same | matches |
| Momentum | $p_j=\sqrt{\gamma_j^2-1}$ | `sqrt(T*(T+2 mc2))/mc2` | algebraically identical |
| Velocity | $\beta_j=p_j/\gamma_j$ | `p/(1+T/mc2)` | matches |
| Logarithm | $\ln\frac{p_i+p_f}{p_i-p_f}$ | same, ordered $i$ then $f$ | matches |
| Elwert ratio | $\beta_i/\beta_f$ | `beta_i/beta_f` | matches |
| Elwert numerator | initial $\beta_i$ | `1 - exp(-2 pi Z alpha/beta_i)` | matches |
| Elwert denominator | final $\beta_f$ | `1 - exp(-2 pi Z alpha/beta_f)` | matches |
| $1/k$ | $k$ in eV | `k*1e3` restores eV after the keV conversion | matches |
| $1/p_i^2$ | incident momentum squared | `p_i**2` | matches |

`R_E_CM2 = 7.9407877e-26` is $r_e^2$ for $r_e=2.8179403262\times10^{-13}$ cm, high by $2.3\times10^{-9}$ relative. `ALPHA_FS = 1/137.035999` differs from the CODATA $\alpha$ in the tenth digit. Both are rounding, not divergence.

The one substantive difference from the source is one the docstring already declares: 3BN(a) is derived for $T_i\ll m_ec^2$ and the code substitutes the **relativistic** $p_j$ and $\beta_j$ into it. That is a deliberate extrapolation, not the relativistic 3BN cross section; outside the non-relativistic regime it is neither 3BN(a) nor 3BN.

Endpoint and floor handling, versus the analytic limits:

- The `T_f > 1e-6` keV guard removes the last 1 meV below the endpoint, where the true limit is finite and non-zero. The masked-out band is $10^{-9}$ of a 30 keV endpoint, so the missing area is negligible on any realistic eV grid, but it means the code does **not** exhibit the finite endpoint value exactly at $k=T_i$; it returns zero. The `T_f = where(ok, T_f, 1e-6)` clamp only feeds the masked lane and cannot leak, because the final `where(ok, dsig, 0)` discards it.
- The `maximum(p_i - p_f, 1e-30)` floor cannot bind on the CPU path, since `T_f > 1e-6` keV already forces $p_i-p_f>0$ in float64. It is a float32 defence for the CUDA twin.
- The `maximum(k*1e3, 1e-30)` floor is likewise unreachable behind `k > 0.0`.
- `k > 0` excludes the soft-photon log divergence at exactly zero. Correct: the expression has no value there.

CUDA twin. `_brem_incident_prefactor_core` hoists $n\,L\,10^{-8}\cdot\tfrac{16}{3}\alpha r_e^2Z^2\,\beta_i\,(1-e^{-2\pi Z\alpha/\beta_i})/p_i^2$ and `_dsigma_weighted_scalar` returns `incident_prefactor * born_log / (k_eV * beta_f * den_f)`. Multiplying out gives exactly the CPU expression times $n L_{\rm cm}$: the $\beta_i$ and the initial exponential sit in the hoisted factor, $1/\beta_f$ and the final exponential in the per-energy factor, so the Elwert ordering survives the split. Its guards (`T_f <= 1e-6` or `k <= 0` returns zero; `dp` and `k_eV` floored at $10^{-30}$) are the same boundaries written as branches; `F32_TWO_PI_ALPHA` folds $2\pi\alpha$ with the CODATA $\alpha$. Only the float32 working precision differs. This is static and algebraic: no CUDA device was available in this environment.

Units at the call site. `mc_brem_spectrum` forms `(n_i * 1e24 * path_cm) @ (dsig * T_abs)`, which converts atoms per cubic angstrom to cm$^{-3}$ and multiplies by centimetres. No barn conversion is applied to the analytic branch, and none should be: the analytic result is already cm$^2$/eV, whereas the EEDL branch converts barns by $10^{-24}$ inside its own loader. The two backends therefore arrive at the reduction in the same units, which is what makes the `where(available, eedl, bethe_heitler)` mix in `_bremsstrahlung_dsigma_dk` dimensionally legitimate.

### Numerical evidence

An independent NumPy transcription of the frozen expression above -- written from the derivation, using CODATA $\alpha$ and $r_e$ rather than the module constants -- was compared against `_brem_dsigma_dk` for $Z=6$ over $T_i\in\{10,30,100\}$ keV and $k\in\{1,500,5000,9000,29000,99000\}$ eV under `PYRITE_MC_BACKEND=cpu`. Maximum relative difference $3.3\times10^{-9}$, which is the $2.3\times10^{-9}$ $r_e^2$ rounding plus the $\alpha$ difference; no term-level disagreement survives. Representative value: $Z=6$, $T_i=30$ keV, $k=500$ eV gives $1.00514\times10^{-26}$ cm$^2$/eV from both.

A float64 transcription of the CUDA prefactor split reproduces the CPU core to the same $3.3\times10^{-9}$, confirming the hoist is algebra-preserving.

Limits, measured:

- Elwert to unity with rising $\beta_i$ at $k=T_i/2$, $Z=6$: $f_{\rm E}=1.3987$, $1.2314$, $1.0773$, $1.0127$ at $T_i=1$, $10$, $100$, $1000$ keV ($\beta_i=0.0625$, $0.195$, $0.548$, $0.941$). Monotone toward the Born limit.
- Endpoint: at $Z=6$, $T_i=30$ keV, evaluating at $k=T_i-\delta$ for $\delta=10^3,10^2,10,1,10^{-1},10^{-2},10^{-3}$ eV gives $3.4888$, $3.3037$, $3.2900$, $3.28866$, $3.28852$, $3.28851$, $3.288507\times10^{-29}$ cm$^2$/eV -- convergent to a finite non-zero limit, as derived, with no spike and no premature zero until the $10^{-6}$ keV guard.
- Soft tail: $k\,d\sigma/dk$ at $T_i=30$ keV falls from $1.07\times10^{-23}$ at $k=1$ eV to $9.9\times10^{-25}$ near the endpoint, varying by one order over five decades in $k$ -- the logarithmic behaviour derived, not a power law.
- Positivity holds on the whole tested grid; all $k\ge T_i$ cells are exactly zero.

No published tabulated cross section was reproduced. Seltzer--Berger or EEDL comparison would test the *accuracy* of 3BN(a)+Elwert rather than its faithful transcription, and the ledger already treats the EEDL backend as the default and the analytic form as an admitted-accuracy fallback.

### Verdict for this section

`rederived`. The implemented analytic backend is Koch and Motz 3BN(a) with the Elwert (1939) correction, transcribed correctly in prefactor, $Z$ dependence, logarithm argument, Elwert numerator/denominator ordering, and eV unit handling, on both the CPU core and the CUDA scalar twin. Three qualifications, none an error: the Koch and Motz original could not be retrieved and the formula identity rests on independent sources that quote it; relativistic momenta are substituted into a non-relativistic formula, which the docstring declares; and the $T_i-k>10^{-6}$ keV guard returns zero in a band where the analytic limit is finite and non-zero. CUDA agreement is algebraic only, still unexecuted on hardware.

## Retained implementation-context history

The previous 2026-09-03 report established implementation-context filters for packaged checksum/structure, native normalization, fixed-energy interpolation, cutoff, isotropic assembly and fallback. It also recorded one-time panel staging, CUDA fused reduction, conservative portable chunk admission, delayed GPU OOM classification/retry, and dataset identity/provenance separation. Relevant existing anchors remain `test_bremsstrahlung_eedl.py`, `test_chunk_invariance.py`, `test_spectrum_cuda_cheap_hoists.py`, `test_gpu_oom_retry.py`, `test_backend_selection.py`, `test_groove.py`, `test_radiation_error_estimators.py`, `test_profiles.py` and `test_public_api.py` in their ledger-listed directories. Only the three focused files above were rerun by this verifier. The earlier CUDA equivalence test was hardware-gated and unexecuted; that gap remains.

## Verdict and suggested ledger edit

- **Claim**: `brem-spectrum` — `src/pyrite/montecarlo/spectrum/brem.py::mc_brem_spectrum` and the ledgered EEDL/CUDA helpers — ENDF MF=23/MT=527, MF=26/MT=527; Formats Manual sections 0.5.2.2, 6.2.3.1 and 26.
- **Filters**: units `pass`; limits `pass` for finite supported attenuation and the stated numerical endpoint guard; signs/conventions `pass`.
- **Re-derivation**: `matches` for the EEDL estimator and declared processing; full primary-source verification of the analytic fallback is incomplete.
- **Verdict**: `filtered` for the complete ledger row. The EEDL portion has independent re-derivation evidence; the composite claim cannot yet advance.
- **Write-up**: `docs/validation/radiation-physics/brem-spectrum.md`.
- **Suggested ledger change**: keep `filtered`; replace the blanket pending independent-derivation sentence with: “Independent EEDL estimator derivation completed 2026-09-13; C/Si/W interpolation probes match to $4.17\times10^{-16}$ relative at twelve sampled incident energies. Cutoff processing removes 0.1201–1.1867% probability in these probes. Primary-source verification of the hybrid analytic fallback, CUDA hardware equivalence and human sign-off remain pending. Self-absorption validation is limited to finite supported attenuation; nonfinite coefficients are currently treated as transparent.” Human applies it.

That EEDL-only verdict is superseded by the closure below: the analytic fallback was independently re-derived in the section above, and the coordinating context supplied the remaining checks.

## Coordinated closure, 2026-09-13

Supplied by the coordinating context after both independent verifications above. No code was changed by either verifier or by this closure.

- **Documentation checks.** `pyrite-dev docs` builds this page with no new Sphinx warning, and `pyrite-dev test tests/dev/test_docs.py tests/dev/test_validation_ledger.py tests/dev/test_doc_blocks.py` passed **68 tests**. The rendered page carries 46 parsed math nodes and no unparsed math left as literal text.
- **CUDA equivalence anchor executed on hardware.** The gap recorded in the 2026-09-03 report and repeated above is closed: `PYRITE_TEST_BACKEND=cuda UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run --extra nvidia pyrite-dev test tests/montecarlo/test_spectrum_cuda_cheap_hoists.py` passed **5 tests** on an NVIDIA GeForce RTX 3060 Ti, including `test_eedl_brem_raw_kernel_matches_staged_numpy_reference` and `test_brem_raw_kernel_incident_hoist_matches_old_formula` for one and two layers. The fused float32 CUDA reduction therefore matches the independent staged NumPy reference on a real device, not only algebraically.
- **Unrelated failure observed in the same GPU run.** `tests/montecarlo/test_gpu_oom_retry.py::test_brem_oom_halves_only_brem_and_preserves_original_case` fails when a CUDA device is actually present: it asserts `_attempted_brem_chunk == 100000`, but device-memory chunk admission clamps the attempt to 67104. That is a test/environment coupling in the OOM-retry anchor, not a physics discrepancy in this claim, and it is left for a separate task.
- **Composite verdict.** `rederived`. Both halves of the row -- the EEDL track-length estimator and the analytic 3BN(a)+Elwert fallback -- now have independent fresh-context re-derivations that match, with the qualifications each section records: cutoff renormalization is a declared processing choice, self-absorption validation covers finite supported attenuation only (nonfinite coefficients read as transparent), the Koch and Motz original could not be retrieved, relativistic momenta are substituted into a non-relativistic formula, and the $T_i-k>10^{-6}$ keV guard zeroes a band where the limit is finite. Human sign-off remains pending and was not assigned.

## Fresh-context verification of #174 unit-base interpolation, 2026-09-25

### Independent derivation before implementation inspection

This section was derived from the ledger row, the `_unit_base_panels`, `_prepare_eedl_grid`, `_eedl_brem_dsigma_dk` and `mc_brem_spectrum` docstrings, and the [ENDF-6 Formats Manual](https://nds.iaea.org/public/endf/endf-manual.pdf), section 0.5.2.2, equations (7)–(14), before reading the implementation bodies. ENDF's unit-base transform supplies the normalized coordinate and its Jacobian. The logarithmic incident-energy shape weight, geometric sub-panel spacing, endpoint holding, cutoff and final renormalization are PyRITE processing choices, not the EEDL file's declared `INT=2` law.

Let native incident panels be $T_i<T_{i+1}$, with photon supports $[a_j,b_j]$, widths $h_j=b_j-a_j>0$, and piecewise-linear unit-area densities $p_j(k)$ in eV$^{-1}$. Define

$$
x=\frac{k-a_j}{h_j},\qquad q_j(x)=h_jp_j(a_j+h_jx),\qquad \int_0^1q_j(x)\,dx=1.
$$

At a sub-panel energy $T\in[T_i,T_{i+1}]$, the stated refinement requires distinct weights:

$$
u=\frac{T-T_i}{T_{i+1}-T_i},\qquad
w=\frac{\ln(T/T_i)}{\ln(T_{i+1}/T_i)},\qquad
a(T)=(1-u)a_i+ua_{i+1},\quad b(T)=(1-u)b_i+ub_{i+1}.
$$

The independent unit-base expression is

$$
q_T(x)=(1-w)q_i(x)+wq_{i+1}(x),\qquad
p_T(k)=\frac{q_T((k-a(T))/(b(T)-a(T)))}{b(T)-a(T)}.
$$

The Jacobian is essential: $q_T$ is dimensionless and $p_T$ has units eV$^{-1}$. Positivity and unit area follow from $0\le w\le1$. At $T=T_i$ or $T_{i+1}$, both weights recover the corresponding native panel. If $b_i=T_i$ and $b_{i+1}=T_{i+1}$, linear endpoint interpolation gives $b(T)=T$ exactly. The union of native $x$ knots represents both piecewise-linear curves without approximation, so its linear mixture also integrates to one. Geometric spacing with at most 32 intervals per incident-energy decade controls refinement density; it changes neither this expression nor the native limits. The ENDF manual's lin-lin unit-base scheme would use $u$ for the shape as well; $w$ is the declared empirical departure for the sparse EEDL panels.

Runtime interpolation between adjacent refined energies $R_m\le T\le R_{m+1}$ is specified as Cartesian at fixed $k$, with $v=(T-R_m)/(R_{m+1}-R_m)$ and density $\widetilde p(k\mid T)=(1-v)\bar p_m(k)+v\bar p_{m+1}(k)$, where $\bar p_m$ holds the endpoint value of panel $m$ through the next panel's endpoint. The physical density is $\mathbf 1_{0<k\le T}\widetilde p/C(T)$ with $C(T)=\int_0^T\widetilde p(k\mid T)\,dk$. For each linear segment with left density $y$ and slope $s$, the partial integral of width $d$ is $yd+sd^2/2$; this gives exact normalization for the represented piecewise-linear curve, independent of output energy bins. The extension can add nonzero probability above $R_m$, so the $k\le T$ cut and normalization remain necessary. Nonnegative native panels and convex weights preserve signs; the cutoff cannot create negative density. The total MF=23 cross section still multiplies the normalized MF=26 density, giving cm$^2$/eV after barn conversion, and the established $1/(4\pi)$ track-length factor supplies per steradian.

### Code comparison and anchors

`_unit_base_panels` constructs the union $x$ mesh, multiplies each native density by its own width, and combines adjacent rows with the geometric sub-panel fraction. Because $T=T_i(T_{i+1}/T_i)^{j/n}$, that fraction is exactly the independently derived $w=j/n$. It separately computes the linear-in-$T$ range fraction $u$ for both photon endpoints. The final native row is appended explicitly. The cumulative table trapezoids the piecewise-linear $q_T$ and therefore has unit area up to source floating-point normalization. These are term-by-term matches. The docstring cites section 0.5.2.3; the consulted ENDF manual puts the two-dimensional unit-base transform in section **0.5.2.2**. This is a citation correction, not a formula difference.

`_prepare_eedl_grid` divides by each refined width, stages the resulting per-eV rows once on the output grid, and extends the lower row at its tip density through the next endpoint. `_prepare_eedl_segment_state` forms the runtime Cartesian fraction $v$, evaluates each refined panel's exact piecewise-linear CDF at $T$, adds the lower panel's constant-density extension up to $T$, mixes those surviving areas, and divides the separately interpolated MF=23 total by that area. `_evaluate_prepared_eedl` multiplies the staged-row mixture by this scale only at $0<k\le T$. Thus the output-grid interpolation and normalization agree with the derivation. The extension is zero for the final row and the last incident interval uses its preceding row at $v=1$.

The CUDA path flattens the same staged $(\text{panel},k)$ array in row-major order, passes the same lower row, fraction, availability and normalized cross-section scale, and evaluates the same two-row mixture with the $0<k\le T$ guard. Its scalar path has no separate interpolation formula or CDF. This is a static source-to-code comparison; this verifier did not run CUDA hardware. The existing CUDA test uses hand-staged rows and checks the fused mixture, guard, weighting and attenuation, but does not independently construct the new unit-base panels.

On CPU, `tests/montecarlo/test_bremsstrahlung_eedl.py` passed **14 tests** with `NUMBA_DISABLE_JIT=1`. Its direct anchors verify native-row reconstruction, unit-area sub-panels, $b(T)=T$, spacing, cutoff-normalized totals, and between-panel C/W agreement with Seltzer–Berger at 0.1 and 1 MeV. Those source comparisons support the empirical $\ln T$ choice at sampled points; they do not prove a global error bound or make it the ENDF-declared interpolation law. Initial collection without `NUMBA_DISABLE_JIT=1` failed because this read-only worktree prevented Numba from locating a writable cache; no physics assertion failed.

The required `tests/dev/test_docs.py` passed **6 tests**. `pyrite-dev docs` could not delete its existing `docs/_autosummary` directory in this read-only worktree. An isolated Sphinx build of this unchanged page in `/tmp`, using the repository's MyST `dollarmath` and `amsmath` extensions, succeeded. Inspection of the rendered #174 section found every physics expression in `class="math notranslate"` elements and no unrendered math delimiters. The isolated build emitted one expected unresolved relative-link warning because the rest of the documentation tree was not copied.

### Verdict for #174

- **Claim**: `brem-spectrum` — `src/pyrite/montecarlo/spectrum/brem_unit_base.py::_unit_base_panels` and the ledgered EEDL/CUDA helpers — ENDF-6 Formats Manual section 0.5.2.2, equations (7)–(14), with PyRITE's declared $\ln T$ shape weight and endpoint processing.
- **Filters**: units `pass`; limits `pass` for native panels, unit area, $b(T)=T$ and $k>T$ cutoff; signs/conventions `pass` for nonnegative convex mixtures and isotropic $1/(4\pi)$ factor.
- **Re-derivation**: `matches` — unit-base Jacobian, distinct $\ln T$ shape and linear-$T$ endpoint weights, staged Cartesian mixture and exact surviving-area normalization agree with the code. The source-section number in the `_unit_base_panels` docstring differs from the consulted manual (0.5.2.3 versus 0.5.2.2).
- **Verdict**: `rederived` for the #174 interpolation claim. CUDA comparison here is static; the existing hardware anchor in the prior record remains separate evidence. Human sign-off remains pending.
- **Write-up**: `docs/validation/radiation-physics/brem-spectrum.md`.
- **Suggested ledger change**: change `brem-spectrum` status from `filtered` to `rederived`; replace the sentence that says #174 needs fresh-context re-derivation with: “#174 unit-base refinement independently re-derived 2026-09-25: Jacobian, $\ln T$ shape and linear-$T$ endpoints, native limits, staged Cartesian interpolation, cutoff normalization and CUDA argument flow match. CPU EEDL anchors pass; CUDA path was reviewed statically in this verification. The $\ln T$ weight and endpoint processing are PyRITE choices beyond EEDL's declared `INT=2` law; human sign-off remains pending.” Correct the docstring's ENDF section citation to 0.5.2.2 in a separate implementation edit. Human applies ledger changes.
