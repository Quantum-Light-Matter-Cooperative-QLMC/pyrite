# `frank-tamm-homogeneous-dielectric`

## Scope and source

Validation: `frank-tamm-homogeneous-dielectric` (issue #587). This is a fresh-context verification at revision `3cb78c29`. Before reading any implementation body it used only the ledger row, the derivation docstrings of the listed owners, the [bulk Cherenkov physics page](../../physics/radiation-physics/cherenkov-radiation.md), and the sources: I. M. Frank and I. E. Tamm, Dokl. Akad. Nauk SSSR **14**, 109 (1937); J. D. Jackson, *Classical Electrodynamics*, 3rd ed., §13.5; and the PDG review *Passage of Particles Through Matter* (Cherenkov radiation section). The independent external reference is Geant4 11.4.2 `G4Cerenkov` (`checks/cherenkov_geant4`).

The claim covers the owners `frank_tamm_yield`, `cherenkov_threshold_beta`, `BulkEmissionYield` and `ChargedPassage` in `src/pyrite/montecarlo/spectrum/cherenkov.py`. They give the internal bulk Cherenkov yield of one charge $ze$ at constant $\beta$ in an unbounded homogeneous isotropic nonmagnetic dielectric with an explicit, weakly absorbing permittivity $\varepsilon(E)$:

$$
\frac{d^2N}{dx\,dE}=\frac{\alpha z^2}{\hbar c}\left(1-\frac{1}{\beta^2n^2(E)}\right)\Theta\bigl(\beta n(E)-1\bigr),
\qquad
\cos\theta_C=\frac{1}{\beta n(E)},
\qquad
n=\operatorname{Re}\sqrt{\varepsilon}.
$$

The threshold is local in photon energy. Active media, $\operatorname{Re}\varepsilon\le0$ and loss tangents above the gate are refused. Interface (transition) radiation, escape, absorption along the photon path, strong absorption and resonant X-ray Cherenkov emission are out of scope.

## Cheap filters

| Check | Result |
|---|---|
| Units | $\alpha$ is dimensionless and $\hbar c$ is in eV m, so $\alpha z^2/\hbar c$ is in eV$^{-1}$ m$^{-1}$: photons per metre per eV. $\sin^2\theta_C$ is dimensionless. `photons_per_eV` multiplies by a length in m, giving eV$^{-1}$. Pass. |
| Prefactor | $\alpha/\hbar c=(1/137.035999084)/(197.3269804\ {\rm eV\,nm})=369.810\ {\rm eV^{-1}\,cm^{-1}}$ (computed here: 369.81018). Pass. |
| Below and at threshold | $\beta n\le1$ gives no real cone, so the yield is zero. At $\beta n=1$ the factor $1-1/(\beta n)^2$ vanishes, so the step function and the formula agree there. Pass. |
| Onset | With $\beta n=1+\eta$, $\sin^2\theta_C=1-(1+\eta)^{-2}=2\eta+O(\eta^2)$: linear onset. Pass. |
| $n\le1$ | $\beta<1$ and $n\le1$ give $\beta n<1$: no emission for vacuum or X-ray $\varepsilon=1-\delta$. Pass. |
| $\beta\to1$ | $\cos\theta_C\to1/n$, the largest cone; yield $\to(\alpha z^2/\hbar c)(1-1/n^2)$. Pass. |
| Nondispersive | Constant $n$ gives a flat spectrum, $369.81\,z^2\sin^2\theta_C$ eV$^{-1}$ cm$^{-1}$. Pass. |
| Charge | The field radiated is linear in $ze$ and the yield is quadratic: $z^2$, independent of sign. Pass. |
| Sign convention | Passive $e^{-i\omega t}$ means $\operatorname{Im}\varepsilon\ge0$. The principal root of $\varepsilon$ with $\operatorname{Re}\varepsilon>0$ has $\operatorname{Re}\sqrt\varepsilon>0$ and $\operatorname{Im}\sqrt\varepsilon\ge0$, so $k=n\omega/c$ has a decaying imaginary part. This is the physical branch. Pass. |

## Independent derivation

### Frank–Tamm yield and cone

Take a charge $ze$ moving along $\hat{\mathbf z}$ at $v=\beta c$ in a medium with $\mu=1$ and permittivity $\varepsilon(\omega)$. Fourier-transforming Maxwell's equations in $t$ and in the transverse coordinates gives each frequency component of the field with longitudinal wavenumber $k_z=\omega/v$. Its transverse wavenumber obeys

$$
k_\perp^2=\frac{\omega^2}{c^2}\varepsilon(\omega)-\frac{\omega^2}{v^2}
=\frac{\omega^2}{v^2}\bigl(\beta^2\varepsilon-1\bigr).
$$

For real $\varepsilon=n^2$ and $\beta n>1$, $k_\perp$ is real and the field is an outgoing conical wave. Its direction obeys $\cos\theta_C=k_z/k=1/(\beta n)$. The energy flux through a cylinder of radius $\rho$ about the track gives Jackson's result (§13.5, from Fermi's energy-loss expression in the transparent limit)

$$
\frac{d^2W}{dx\,d\omega}=\frac{z^2e^2}{c^2}\,\omega\left(1-\frac{1}{\beta^2\varepsilon(\omega)}\right)
\quad\text{(Gaussian units)},
$$

restricted to frequencies with $\beta^2\varepsilon(\omega)>1$. Dividing by the photon energy $\hbar\omega$ and writing $E=\hbar\omega$, so that $d\omega=dE/\hbar$, gives

$$
\frac{d^2N}{dx\,dE}=\frac{1}{\hbar\omega}\frac{1}{\hbar}\frac{z^2e^2}{c^2}\omega\left(1-\frac{1}{\beta^2n^2}\right)
=\frac{z^2e^2}{\hbar^2c^2}\left(1-\frac{1}{\beta^2n^2}\right)
=\frac{\alpha z^2}{\hbar c}\sin^2\theta_C,
$$

using $\alpha=e^2/\hbar c$. This is the PDG form, $d^2N/dx\,dE\approx369.81\,z^2\sin^2\theta_C$ eV$^{-1}$ cm$^{-1}$. All the energy at frequency $\omega$ flows on the single cone $\theta_C(\omega)$, so the angular density is $(d^2N/dx\,dE)\,\delta(\theta-\theta_C)$.

### Per-energy threshold

The emission condition $\beta^2\varepsilon(\omega)>1$ is evaluated separately at each frequency. The threshold speed is therefore a function of energy, $\beta_t(E)=1/n(E)$. For a fixed $\beta$ the emitted band is $\{E: n(E)>1/\beta\}$. When $n\le1$ there is no threshold below $\beta=1$, which the code represents as $\beta_t=\infty$.

### Weak loss

Write $\varepsilon=\varepsilon'(1+it)$ with loss tangent $t=\varepsilon''/\varepsilon'\ge0$. Then

$$
\operatorname{Re}\sqrt{\varepsilon}=\sqrt{\varepsilon'}\operatorname{Re}(1+it)^{1/2}
=\sqrt{\varepsilon'}\left(1+\frac{t^2}{8}+O(t^4)\right).
$$

So $n=\operatorname{Re}\sqrt\varepsilon$ and $\sqrt{\operatorname{Re}\varepsilon}$ differ only at second order (numerically, $t=10^{-3}$ gives $1.24999996\times10^{-7}$ against $t^2/8=1.25\times10^{-7}$). With absorption, $k_\perp$ acquires an imaginary part and the "radiated" energy depends on the observation radius. The Frank–Tamm form is then the leading term of the transparent limit, and the absorption corrections enter at relative order $t$ away from threshold. The near-threshold region $\lvert\beta n-1\rvert\lesssim t$ is smeared rather than sharp. The stated $O(\tan\delta)$ bound and the threshold-smearing caveat are consistent with this order count; this record does not derive the size of the first-order coefficient. The 1e-3 gate therefore limits the neglected correction to about 0.1% away from threshold.

### Dispersive closed form used by the tests

For $n^2=a+bE^2$ with $\beta^2a>1$ across the band,

$$
\int\left(1-\frac{1}{\beta^2(a+bE^2)}\right)dE
=E-\frac{1}{\beta^2\sqrt{ab}}\arctan\!\left(E\sqrt{b/a}\right),
$$

which matches the test primitive. For $a=1.7$ and $\beta=0.9$, $\beta^2a=1.377>1$, so the whole 1–6 eV band radiates and the integral needs no threshold piece.

## Source-to-code comparison

The implementation was read only after the derivation above.

| Item | Derivation | Code (`src/pyrite/montecarlo/spectrum/cherenkov.py`) | Result |
|---|---|---|---|
| Prefactor | $\alpha/\hbar c$ in eV$^{-1}$ m$^{-1}$ | `FRANK_TAMM_PER_EV_M = ALPHA_FS / (HBARC_EV_ANG * 1e-10)` (line 52). Its constants are $1/137.035999$ and 1973.269804 eV Å, giving 369.810 eV$^{-1}$ cm$^{-1}$ | match |
| Index | $\operatorname{Re}\sqrt\varepsilon$, principal root | `n = np.sqrt(eps).real` (line 218) | match |
| Threshold | strict $\beta n>1$ | `above = beta_n > 1.0` (line 220) | match; zero at equality |
| Yield | $z^2\sin^2\theta_C$ times prefactor | `FRANK_TAMM_PER_EV_M * passage.charge**2 * sin2`, where `sin2 = 1 - 1/beta_n**2` above threshold and 0 otherwise (lines 221, 227) | match |
| Cone | $\cos\theta_C=1/(\beta n)$, undefined below threshold | `cos_theta = 1/beta_n` above, NaN below (line 228); `theta_rad = arccos` (line 112) | match |
| Passivity | $\operatorname{Im}\varepsilon\ge0$ | refuses `eps.imag < 0` (line 209) | match |
| Strong loss | refuse $\operatorname{Re}\varepsilon\le0$ and $\varepsilon''>t_{\max}\varepsilon'$ | lines 211 and 213 | match |
| Threshold helper | $\beta_t=1/n$, or $\infty$ for $n\le1$ | `cherenkov_threshold_beta` (lines 170–177) | match (see finding 1) |
| Path scaling | $dN/dE=L\,d^2N/dx\,dE$ | `photons_per_eV` (lines 114–118) | match |
| Angular delta | each energy's whole yield goes in the bin holding $\theta_C$ | `angular_histogram` (lines 120–141): half-open bins; below-threshold and out-of-range values are dropped | match |
| Passage | $0\le\beta<1$, finite $z$ | `ChargedPassage.__post_init__` (lines 81–89) | match; the $\beta\to1$ limit is approached, never equalled |

Numeric spot checks against values computed here, not by implementation helpers, were as follows. For $\beta=0.99$ and $n=1.33$, $\sin^2\theta_C=0.42320$, which gives 688.6 photons cm$^{-1}$ over 1.5–5.9 eV. For silica (Malitson), $n(E)=1.47$ at $E=3.0913$ eV, so the $\beta=1/1.47$ band starts there, inside the 1.5–5.9 eV window ($n$ = 1.4529 to 1.5382).

## Geant4 reference fairness

`G4Cerenkov` (11.x) computes the mean photon number of a step as `GetAverageNumberOfPhotons(charge, beta, ...) * step_length`. Here `beta` is the step mean $(\beta_{\rm pre}+\beta_{\rm post})/2$. The photon count is drawn from a Poisson distribution. The energy is sampled uniformly on the `RINDEX` table range and accepted with weight $\sin^2\theta/\sin^2\theta_{\max}$ (energies below threshold have $\sin^2<0$ and are always rejected). The direction is placed on the cone $\cos\theta=1/(\beta n(E))$ about the step chord (`aStep.GetDeltaPosition().unit()`), with `RINDEX` interpolated linearly in energy. The comparison reproduces each of these.

- **Step-mean beta.** `cherenkov.cc` (`Recorder::UserSteppingAction`, line 140) writes $0.5(\beta_{\rm pre}+\beta_{\rm post})$ and the step length. `compare.py` groups the path length in $10^{-6}$ bins of $\beta$. Near the silica threshold the grouping moves a step's band yield by at most about $5\times10^{-5}$ relative, below the 0.14% Poisson precision of that case.
- **Chord axis.** The polar angle is measured about the chord $(\mathbf r_{\rm post}-\mathbf r_{\rm pre})/\lvert\cdot\rvert$, the axis `G4Cerenkov` rotates onto. Multiple scattering between steps therefore does not smear the reference cone.
- **Linear `RINDEX`.** `_predict` interpolates $n$ linearly in energy and then squares it. Geant4 integrates $1/n^2$ by the trapezoid rule on the nodes. With 441 silica nodes over 4.4 eV, the difference is second order in node spacing and far below the statistical precision. Constant-index tables are exact.
- **Primary only.** Secondaries are killed at stacking. That happens after `GetSecondaryInCurrentStep`, so the primary's photons are still counted, and delta rays add no Cherenkov light. Optical processes other than Cherenkov are switched off.
- **Statistics.** A per-step Poisson count followed by multinomial binning gives independent Poisson counts per bin. The tests use a 5σ total-count bound plus an absolute 0.5% ratio bound, which is the looser of the two for these photon counts. They also require energy reduced chi-square below 1.5 and angle reduced chi-square below 2.0, over bins with at least 10 expected photons. They check that no photon appears where none is expected, and that the mean angle agrees within 5 standard errors. These are reasonable two-sided gates. For the constant-index cases the angle test checks the cone position only, because nearly all photons fall in one 0.1° bin; the README says so.

An independent recomputation was done from the committed reference JSON with exact constants and exact Malitson silica (not the interpolated table). It used trapezoid integration on 200,001 points, with no PyRITE code. Results:

| Case | Geant4 photons | Independent expectation | Ratio | z |
|---|---:|---:|---:|---:|
| `water_mu_b099` | 13,773,785 | 13,771,275 | 1.00018 | 0.68 |
| `silica_mu_threshold` | 541,847 | 541,054 | 1.00147 | 1.08 |
| `n150_alpha_b095` | 16,511,681 | 16,516,461 | 0.99971 | −1.18 |

These match the README table. `water_p_below` has $\beta n\le0.7399\times1.33=0.984<1$ on every step and records zero photons. As the ledger notes, Geant4 implements the same lossless formula, so it anchors normalization, the dispersive threshold and the cone, but not the weak-loss treatment.

## Tests run

`UV_CACHE_DIR=~/.cache/pyrite-uv-cache uv run pyrite-dev test tests/montecarlo/spectrum/test_cherenkov.py tests/montecarlo/spectrum/test_cherenkov_geant4_reference.py` gave 41 passed and 2 skipped. The two skips are the energy and angle checks of the zero-yield proton case, which are skipped by design. Its zero-yield total check passes.

## Findings

1. `cherenkov_threshold_beta` (lines 170–177) applies none of the gates that `frank_tamm_yield` applies (passivity, $\operatorname{Re}\varepsilon>0$, loss tangent). For an active or strongly lossy $\varepsilon$ it still returns $1/\operatorname{Re}\sqrt\varepsilon$. This does not affect the yield, and the docstring does not promise a gate. However, the ledger sentence "active media, $\operatorname{Re}\varepsilon\le0$ and loss tangents above the gate are refused" applies only to `frank_tamm_yield`. This is a wording note, not a discrepancy.
2. The $O(\tan\delta)$ bound on the neglected absorptive correction (physics page lines 63–66; docstring lines 194–195) is an order statement. It is consistent with the derivation above, but neither the page nor this record derives or tests its coefficient. The weak-loss test checks only the second-order $\operatorname{Re}\sqrt\varepsilon$ shift. This is acceptable for a source term whose ledger scope says absorptive corrections are not modelled.
3. `ALPHA_FS = 1/137.035999` (in `materials/crystal.py`) is truncated relative to CODATA, a relative difference below $10^{-9}$. This is immaterial.

No factor, sign, exponent, unit or convention differs between the source and the code.

## Verdict

`rederived`. The independent derivation matches the implementation symbolically, dimensionally and numerically. The Geant4 comparison is set up fairly and reproduces independently. The suggested ledger change is to set the status to `rederived`, or to `anchored` given the green analytic and Geant4 reference tests already named in `Anchor`, and to link this write-up. Optionally, scope the "refused" clause to `frank_tamm_yield`. Human sign-off belongs to #277.
