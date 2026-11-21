# Optical instrument observable verification

Validation: `optical-instrument-observable`.

**Code:** `src/pyrite/detectors/optical_energy.py::CalibratedPulseEnergyDetector`,
`src/pyrite/detectors/optical_interferometer.py::polarized_interferogram`
(with `_port_spectra`), `src/pyrite/_optical_coherency.py::OpticalSpectralCoherency`,
`src/pyrite/detectors/optical_ctr_inference.py::GaussianCTRTemplates` and
`::fit_gaussian_ctr`.

**Sources:** linear energy-integrating detector response; two-arm field
interference and Fourier-transform spectroscopy; M. Geitz, *Bunch length
measurements*, DIPAC 1999, IT07 (Martin–Puplett layout); standard ideal-element
Jones analysis of the Martin–Puplett interferometer (MPI); Schmidt et al.,
arXiv:1803.00608, §§3.3–3.5 (phase-retrieval ambiguity); Wilks' theorem for the
one-parameter profile likelihood.

## Part I — Independent derivation

Written from the ledger row, the research note section
[propagation, detectors and observables](../../research/physics/optical-ir-emission.md#propagation-detectors-and-observables),
the module, class and function docstrings, and the public signatures only,
before any implementation body was read.

### I.1 Quantities and units

| symbol | meaning | unit |
| --- | --- | --- |
| $E$ | photon energy | eV |
| $\Phi(E)$ | collected spectral photon density | photons $\mathrm{eV}^{-1}$ pulse$^{-1}$ |
| $\eta(E)$ | energy-absorption fraction | 1, $\in[0,1]$ |
| $U$ | absorbed energy | J pulse$^{-1}$ |
| $G$ | isolated-pulse gain | V $\mathrm{J}^{-1}$ |
| $V$ | peak readout | V |
| $f_{\rm rep}$ | repetition rate | Hz |
| $\bar P$ | average optical power | W |
| $\mathsf J(E)$ | $2\times2$ coherency $\langle E_aE_b^*\rangle$, $\operatorname{tr}\mathsf J=\Phi$ | photons $\mathrm{eV}^{-1}$ pulse$^{-1}$ |
| $d$ | differential mirror displacement | m |
| $L$ | optical path difference (OPD) | m |
| $\sigma_L$ | RMS OPD jitter | m |
| $\sigma_t$ | RMS bunch duration | s |

### I.2 Calibrated linear pulse detector

A photon of energy $E$ (in eV) deposits $eE$ joules, absorbed with
probability-weighted energy fraction $\eta(E)$. The absorbed energy per pulse is

$$
U = e\int_{E_{\min}}^{E_{\max}} E\,\eta(E)\,\Phi(E)\,\mathrm dE ,
$$

with $e=1.602176634\times10^{-19}\ \mathrm{J\,eV^{-1}}$. Units:
$\mathrm{J\,eV^{-1}}\cdot\mathrm{eV}\cdot\mathrm{eV^{-1}}\cdot\mathrm{eV}=\mathrm J$
per pulse. On a strictly increasing nonuniform node set $E_0<\dots<E_{n-1}$ the
trapezoid rule is

$$
U \simeq e\sum_{i=0}^{n-2}\frac{E_{i+1}-E_i}{2}\,
\bigl[g(E_i)+g(E_{i+1})\bigr],\qquad g=E\,\eta\,\Phi ,
$$

which is exact for piecewise-linear $g$. $\eta$ must be known on the whole
integration interval (no extrapolation). The linear readout is
$V_{\rm lin}=G\,U$ (V); a hard saturation level $V_{\rm sat}$ gives
$V=\min(V_{\rm lin},V_{\rm sat})$ with a saturation flag. For isolated
identical pulses the average optical power is

$$
\bar P = f_{\rm rep}\,U_{\rm inc},\qquad
U_{\rm inc}=e\int E\,\Phi\,\mathrm dE \ \text{or}\ U ,
$$

in $\mathrm{s^{-1}\,J}=\mathrm W$. The gain $G$ is not a DC V/W responsivity,
so $\bar P$ must not be converted to volts with $G$. Limits:
$\eta\to0\Rightarrow U=V=0$; $\eta=1$ and $\Phi=\delta(E-E_0)N_\gamma$ gives
$U=eE_0N_\gamma$.

### I.3 Ideal Martin–Puplett Jones transfer

Work in the orthonormal basis $(\hat{\mathbf x},\hat{\mathbf y})$ of the two
differential-delay arm axes (after the 45° splitter grid the two orthogonal
linear components travel in different arms; the roof mirrors' 90° rotation only
routes them back to recombine and can be absorbed into the basis). The
independent derivation uses:

1. Input wire grid at angle $\theta_i$: projector onto
   $\hat{\mathbf u}=(\cos\theta_i,\sin\theta_i)$, so the transmitted amplitude
   is $s=\hat{\mathbf u}\cdot\mathbf E$ and its spectral photon density is

   $$
   S_{\rm in}(E)=\langle\lvert s\rvert^2\rangle
   =\hat{\mathbf u}^{\mathsf T}\mathsf J(E)\,\hat{\mathbf u}
   =c_i^2J_{xx}+s_i^2J_{yy}+2c_is_i\operatorname{Re}J_{xy},
   $$

   with $c_i=\cos\theta_i$, $s_i=\sin\theta_i$. Only $\operatorname{Re}J_{xy}$
   survives; a circular (imaginary) coherence does not.
2. Arms with power transmissions $T_1,T_2$ and relative retardance
   $\delta=kL$, $k=E/(\hbar c)$:

   $$
   \mathbf E_{\rm out}=s\,\bigl(\sqrt{T_1}\,c_i,\ \sqrt{T_2}\,s_i\,e^{i\delta}\bigr).
   $$
3. Output analyzer: port A projects on
   $\hat{\mathbf v}=(\cos\theta_o,\sin\theta_o)$, port B on
   $\hat{\mathbf v}_\perp=(-\sin\theta_o,\cos\theta_o)$, each followed by a
   power efficiency $\epsilon_{A,B}$.

The port spectral photon densities are therefore

$$
\boxed{
\begin{aligned}
\Phi_A&=\epsilon_A S_{\rm in}\bigl[T_1c_i^2c_o^2+T_2s_i^2s_o^2
+2\sqrt{T_1T_2}\,c_is_ic_os_o\,\cos\delta\bigr],\\
\Phi_B&=\epsilon_B S_{\rm in}\bigl[T_1c_i^2s_o^2+T_2s_i^2c_o^2
-2\sqrt{T_1T_2}\,c_is_ic_os_o\,\cos\delta\bigr].
\end{aligned}}
$$

The sign of the cosine is $+$ for the port parallel to $\hat{\mathbf v}$
and $-$ for its orthogonal complement; the sign of $\delta$ ($e^{\pm i\delta}$)
does not enter. Checks:

- **Energy conservation** ($\epsilon=1$): $\Phi_A+\Phi_B=S_{\rm in}(T_1c_i^2+T_2s_i^2)$,
  equal to $S_{\rm in}$ for lossless arms, independent of $\delta$.
- **Standard MPI** ($\theta_i=\theta_o=45^\circ$, $T=\epsilon=1$):
  $\Phi_{A,B}=\tfrac12S_{\rm in}(1\pm\cos\delta)$.
- **Zero delay** $\delta=0$: all light in A, none in B.
- **Large delay**: for a smooth finite band
  $\int g(E)\cos(EL/\hbar c)\,\mathrm dE\to0$ (Riemann–Lebesgue), so both
  ports tend to $\tfrac12\int g$.
- **Difference** $\Phi_A-\Phi_B=S_{\rm in}\cos\delta$, so
  $D(L)=V_A-V_B=Ge\int E\eta S_{\rm in}\cos(EL/\hbar c)\,\mathrm dE$, the
  response-weighted cosine transform of the research note, even in $L$.
- **Grid at $0^\circ$ or $90^\circ$** removes the fringe (single arm).

### I.4 OPD, double pass and jitter

Moving a roof (or plane) mirror by $d$ lengthens the round trip by $2d$, so
$L=L_0+2d$ and the optical delay is $\tau=L/c$. The retardance is
$\delta=EL/(\hbar c)$ with $\hbar c=197.3269804\ \mathrm{eV\,nm}$.

For shot-to-shot Gaussian OPD noise $\xi\sim\mathcal N(0,\sigma_L^2)$,

$$
\langle\cos k(L+\xi)\rangle=\operatorname{Re}\bigl(e^{ikL}\langle e^{ik\xi}\rangle\bigr)
=\cos(kL)\,e^{-k^2\sigma_L^2/2},
$$

which multiplies only the fringe term. A common beam arrival-time jitter adds
the same phase to both arms and cancels.

### I.5 Gaussian CTR template and charge jitter

For a normalized longitudinal profile
$\rho(t)=(2\pi\sigma_t^2)^{-1/2}e^{-t^2/2\sigma_t^2}$, the longitudinal form
factor is $F(\omega)=\int\rho(t)e^{i\omega t}\mathrm dt=e^{-\omega^2\sigma_t^2/2}$
and the intensity form factor is

$$
\lvert F\rvert^2=\exp\!\left[-\left(\frac{E\sigma_t}{\hbar}\right)^2\right],
\qquad \hbar=6.582119569\times10^{-16}\ \mathrm{eV\,s}.
$$

The collected spectrum for one bunch of $N$ electrons with frozen transverse
factors is

$$
\Phi(E;\sigma_t)=\Phi_{\rm inc}(E)+\Phi_{\rm coh}(E)\,\lvert F(E;\sigma_t)\rvert^2,
$$

with $\Phi_{\rm inc}\propto N$ and $\Phi_{\rm coh}\propto N(N-1)$. Its $1/e$
energy is $E_{1/e}=\hbar/\sigma_t$.

For a fluctuating population with mean $\bar N$ and RMS $r\bar N$, independent
of profile, linear shot averaging gives $\langle N\rangle=\bar N$ and

$$
\langle N(N-1)\rangle=\langle N^2\rangle-\bar N=r^2\bar N^2+\bar N^2-\bar N,
\qquad
\frac{\langle N(N-1)\rangle}{\bar N(\bar N-1)}=1+\frac{\bar N r^2}{\bar N-1}.
$$

### I.6 Fit statistic

With independent Gaussian voltage errors $\sigma_{jp}$,
$\chi^2(\sigma_t,\boldsymbol\nu)=\sum_{j,p}(V_{jp}-M_{jp})^2/\sigma_{jp}^2$
(plus $((a-1)/\sigma_a)^2$ for a declared fractional prior on scale $a$). The
profile $\chi^2_{\rm p}(\sigma_t)=\min_{\boldsymbol\nu}\chi^2$ has one
interest parameter; by Wilks, the 95% interval is
$\chi^2_{\rm p}-\min\chi^2_{\rm p}\le F^{-1}_{\chi^2_1}(0.95)=3.841459$.
An endpoint whose crossing is not observed inside the scan bounds is open.
Because only $\lvert F\rvert^2$ over a finite band is constrained, any
non-Gaussian profile with the same band-limited $\lvert F\rvert$ yields the same
data (Schmidt et al.); the result is conditional on the Gaussian family.

## Part II — Comparison with the implementation

Implementation bodies were read only after Part I was written (revision
`3a98f542`, branch `issues-616-617`).

### II.1 Term-by-term comparison

| item | independent (Part I) | implementation | result |
| --- | --- | --- | --- |
| absorbed energy | $U=e\int E\eta\Phi\,\mathrm dE$, trapezoid | `np.trapezoid(photons * energy * absorption, energy) * _ELECTRON_VOLT_J` | match |
| $\eta(E)$ | known on whole input band, no extrapolation | linear `np.interp` of knots; refuses input outside knot interval | match |
| readout | $V_{\rm lin}=GU$, $V=\min(V_{\rm lin},V_{\rm sat})$ | `expected = absorbed * pulse_gain_v_per_j`; `min(expected, saturation_v)`, flag `expected > saturation_v` | match |
| average power | $\bar P=f_{\rm rep}U$ (W), not converted with $G$ | `incident * repetition_hz`, `absorbed * repetition_hz`; no V/W path | match |
| input grid | $S_{\rm in}=\hat{\mathbf u}^{\mathsf T}\mathsf J\hat{\mathbf u}$ | `np.einsum("a,eab,b->e", u, j, u).real` | match |
| port A amplitudes | $(\sqrt{T_1}c_ic_o,\ \sqrt{T_2}s_is_o)$ | row 0: `cos(b)cos(a)`, `sin(b)sin(a)` times `sqrt(T)` | match |
| port B amplitudes | $(-\sqrt{T_1}c_is_o,\ \sqrt{T_2}s_ic_o)$ | row 1: `-sin(b)cos(a)`, `cos(b)sin(a)` times `sqrt(T)` | match |
| DC and fringe | $\sum a^2$, $2a_1a_2\cos\delta$ (sign $+$ for A, $-$ for B) | `dc = sum(amplitudes**2)`, `fringes = 2 * prod(amplitudes)` | match |
| wavenumber | $k=E/(\hbar c)$ | `2 * pi * e / _HC_EV_M` with $hc=1.23984198433\times10^{-6}$ eV m | match |
| OPD | $L=L_0+2d$, $\tau=L/c$ | `zero_opd_m + 2.0 * mirror_shift`; `opd / _C_M_S * 1e15` fs | match |
| OPD jitter | fringe $\times e^{-k^2\sigma_L^2/2}$ | `exp(-0.5 * (k * opd_jitter_rms_m) ** 2)` on the cosine only | match |
| port efficiency | multiplies port power | `spectra *= port_efficiencies` | match |
| temporal form factor | $\exp[-(E\sigma_t/\hbar)^2]$, $\sigma_t$ RMS | `exp(-square(energy_eV * rms_duration_fs / _HBAR_EV_FS))`, $\hbar=0.6582119569509$ eV fs | match |
| $1/e$ energy | $\hbar/\sigma_t$ | `_HBAR_EV_FS / rms_duration_fs` | match |
| charge jitter | pair $\times\bigl(1+\bar Nr^2/(\bar N-1)\bigr)$, incoherent unchanged | `1.0 + n_mean * r * r / (n_mean - 1.0)` on the coherent envelope only | match |
| fit statistic | weighted $\chi^2$ + $((a-1)/\sigma_a)^2$ prior, profiled nuisances | weighted `lstsq` with prior row $1/(\text{norm}\,\sigma_a)$, target $1/\sigma_a$; same terms re-added to `chi2` | match |
| interval | $\Delta\chi^2=F^{-1}_{\chi^2_1}(0.95)$ | `_DELTA_CHI2_95 = 3.841458820694124`; `brentq` crossing; `None` when the admissible set reaches a scan bound | match |
| fit kernel | linear in $\Phi$; same transfer and trapezoid | `_response_kernel` reuses `_port_spectra` on a unit linear source and `_trapz_weights` equal to trapezoid weights | match |

Dimensions: $\Phi$ [photons eV$^{-1}$ pulse$^{-1}$] $\times E$ [eV] $\times\mathrm dE$
[eV] $\times e$ [J eV$^{-1}$] = J pulse$^{-1}$; $\times G$ [V J$^{-1}$] = V;
J pulse$^{-1}\times$ Hz = W. $kL$, $k\sigma_L$ and $E\sigma_t/\hbar$ are
dimensionless with the eV m and eV fs constants used.

### II.2 Numeric comparison

Independent scripts (scratch only; not committed) using
`UV_CACHE_DIR=$HOME/.cache/pyrite-uv-cache uv run python` gave:

| check | independent reference | relative difference |
| --- | --- | --- |
| $U$ absorbed and incident, linear $\eta$, quadratic $\Phi$, 4001 nodes | `scipy.integrate.quad` of the analytic integrand | $7.7\times10^{-8}$, $5.0\times10^{-8}$ (trapezoid error) |
| $V=GU$, saturation | $V_{\rm lin}=3.2564$ mV, $V_{\rm sat}=1$ mV | reported 1 mV, flag true; $\bar P/(f_{\rm rep}U)=1$ |
| port spectra, random Hermitian PSD $\mathsf J$ (60 energies, complex $J_{xy}$), $\theta_i=0.37$, $\theta_o=1.1$, $T=(0.8,0.6)$, $\epsilon=(0.9,0.7)$, $L_0=1.3\,\mu$m, $\sigma_L=0.2\,\mu$m, 7 displacements | explicit Jones matrices $M=\sqrt{\epsilon}\,\hat{\mathbf v}^{\mathsf T}\operatorname{diag}(\sqrt{T_1},\sqrt{T_2}e^{ik(L+\xi)})P_u$, $\operatorname{tr}(M\mathsf JM^\dagger)$, 80-node Gauss–Hermite average over $\xi$ | $8.5\times10^{-10}$ (constant rounding) |
| lossless port sum | $S_{\rm in}$ | $3\times10^{-16}$ |
| standard 45°/45° MPI | zero delay $(S_{\rm in},0)$; $\tfrac12S_{\rm in}(1+\cos\delta)$ | $1\times10^{-15}$, exact 0, $2.7\times10^{-10}$ |
| $\lvert F\rvert^2$, $\sigma_t=40$ fs, $E=5,10,20$ meV | numerical Fourier transform of a normalized Gaussian | ratio $1.000000$ |
| pair factor | exact $\langle N(N-1)\rangle$ of a discretized normal population, $\bar N\approx50$, $r\approx0.3$ | ratio $1-2\times10^{-16}$ |
| $\Delta\chi^2$ | `scipy.stats.chi2.ppf(0.95, 1)` $=3.841458820694124$ | identical |
| Gaussian fit, synthetic noisy scan at $\sigma_t=120$ fs, 61 positions, fitted scale | own forward model (Part I formula + own trapezoid), closed-form profiled scale, 4001-point log grid, `brentq` crossings | $\hat\sigma_t$: 120.50 vs 120.59 (independent grid step 0.15%); interval (119.248, 121.776) vs (119.245, 121.780) fs, offset explained by the unrefined independent minimum ($\Delta\chi^2_{\min}=0.019$) |

### II.3 Findings (no code change made)

1. **Extra phase parameter.** `IdealMartinPuplett.phase_offset_rad` adds a
   constant to $\delta$ at all energies. It is not in the ledger claim or the
   `polarized_interferogram` docstring. A nonzero value breaks the ideal even
   interferogram. Unlike $L_0$, it is not dispersive. The default 0 reproduces
   Part I. The ledger should name it as a non-ideal parameter.
2. **Misleading comment.** The `_port_spectra` comment says the input grid
   "retains *intensity* only, not fictitious mutual coherence". The projection
   $\hat{\mathbf u}^{\mathsf T}\mathsf J\hat{\mathbf u}$ does, correctly, retain
   $2c_is_i\operatorname{Re}J_{xy}$. The numeric check with complex $J_{xy}$
   confirms that the code is right; only the comment is imprecise.
3. **Unmarked owners.** `IdealMartinPuplett` (angles, arm transmissions, jitter)
   and `OpticalSpectralCoherency` (basis, Hermitian/PSD and trace normalization)
   carry load-bearing conventions. Neither has a `Validation:` marker, and the
   row's `Code` field does not list them. This verification covered them
   through `_port_spectra`.
4. **Evenness.** The "even interferogram" limit holds in $L$. It holds in
   displacement $d$ only for $L_0=0$ and `phase_offset_rad` $=0$.
5. **Boundary statistics.** When the unconstrained scale is negative, the fit
   clips it to 0. The interval is then a boundary case, and Wilks'
   $\chi^2_1$ calibration of the $\Delta\chi^2=3.841$ threshold is not exact
   there. The predict flag uses `>` saturation, while the fit refuses at
   `>=`. Both are minor.
6. **Scope.** This verification establishes the ideal-element model only. It
   does not reproduce DIPAC 1999 IT07, and it does not use a measured detector
   calibration or measured data. Phase-retrieval non-uniqueness is respected:
   the fit is labelled conditional on the Gaussian family, and an open
   endpoint is reported as `None`.

## Verdict

- **Claim**: `optical-instrument-observable`. Owners:
  `optical_energy.py::CalibratedPulseEnergyDetector`,
  `optical_interferometer.py::polarized_interferogram`/`_port_spectra`,
  `optical_ctr_inference.py::{GaussianCTRTemplates, fit_gaussian_ctr}`.
  Basis: linear detector integration $U=e\int E\eta\Phi\,\mathrm dE$, ideal MPI
  Jones transfer, Gaussian $\lvert F\rvert^2=e^{-(E\sigma_t/\hbar)^2}$, and
  Wilks $\Delta\chi^2_{1,0.95}$.
- **Filters**: units pass; limits pass (zero/large delay, lossless
  conservation, $\eta\to0$, single-arm grid); signs/conventions pass (A $+$, B
  $-$ relative to the output axis; double pass $2d$; jitter
  $e^{-k^2\sigma_L^2/2}$).
- **Re-derivation**: `matches`.
- **Verdict**: `rederived`.

## Device-reduction re-derivation (2026-10-23, issues #616–617)

Fresh verifier; signatures/docstrings and ledger read before implementation.
Scope: already-collected photon densities, calibrated batch reduction and ideal
two-port readout. Source is linear detector integration and two-arm field
interference, as specified in the linked research equations; the DIPAC IT07
PDF could not be fetched in this review. No published numerical reproduction
or hardware calibration is claimed.

For each independent leading-axis index $b$, let $\Phi_b(E)$ have units
photons/(eV pulse), with $E$ measured in eV and $e$ the J/eV conversion.
Absorptance $\eta(E)$ is dimensionless. Conservation of photon energy gives

$$
U_b=e\int E\Phi_b(E)\,dE,\qquad
U_{a,b}=e\int E\eta(E)\Phi_b(E)\,dE,\qquad
V_b=gU_{a,b},\qquad P_b=fU_b,\quad P_{a,b}=fU_{a,b}.
$$

Here $g$ is isolated-pulse V/J and $f$ is repetition Hz. Trapezoidal
quadrature on common increasing nonuniform nodes applies only to the last
axis; batching changes neither weights nor normalization. Zero absorptance
annuls absorbed energy and voltage; unit absorptance gives identical incident
and absorbed energies. Neither bunch population nor repetition rate belongs
in per-pulse energy again.

Independently, an ideal input grid along
$u=(\cos\alpha,\sin\alpha)^T$ gives scalar photon density
$S=u^\dagger Ju$. Arms with power transmissions $t_x,t_y$ give diagonal
intensities $S t_x\cos^2\alpha$ and $S t_y\sin^2\alpha$.
Analyzing at $v=(\cos\beta,\sin\beta)^T$ and its orthogonal complement gives

$$
\begin{aligned}
\Phi_A&=p_A S[t_x\cos^2\alpha\cos^2\beta+
 t_y\sin^2\alpha\sin^2\beta+
 2\sqrt{t_xt_y}\cos\alpha\sin\alpha\cos\beta\sin\beta\,q\cos\phi],\\
\Phi_B&=p_B S[t_x\cos^2\alpha\sin^2\beta+
 t_y\sin^2\alpha\cos^2\beta-
 2\sqrt{t_xt_y}\cos\alpha\sin\alpha\cos\beta\sin\beta\,q\cos\phi],\\
\phi&=\frac{E(L_0+2d)}{\hbar c}+\phi_0,\qquad
q=\exp[-\tfrac12(E\sigma_L/(\hbar c))^2].
\end{aligned}
$$

The Gaussian factor follows from averaging the phase over RMS OPD jitter.
For unit port efficiencies their sum is the post-grid transmitted density;
fringes cancel exactly. Balanced 45-degree grids give complementary cosine
ports. The same linear detector integral applies to each delay independently.
These expressions and unit/limit/sign filters were recorded before reading
the implementation bodies.

### Scoped comparison and verdict

After recording the derivation, inspection of `predict_batch`, `predict`,
`_port_spectra` and `polarized_interferogram` found the same factors and port
signs. The scalar wrapper delegates the identical energy integral; two batched
port calls preserve the per-delay normalization and attach repetition rate
only to average power. An independent NumPy calculation used four nonuniform
nodes, two spectra, nonconstant absorptance and 13 Hz: energy, isolated-pulse
voltage and W agreed within relative tolerance $2\times10^{-15}$. A separate
explicit complex field/Jones oracle used non-45-degree grids, unequal lossy
arms/ports, phase offset, OPD offset and Gaussian jitter at three displacements;
all port spectra agreed within relative tolerance $10^{-13}$ (maximum absolute
photon-density difference $4.66\times10^{-10}$).

- **Filters**: units pass; limits pass; signs/conventions pass.
- **Re-derivation**: `matches`; no divergent factor, sign or normalization found.
- **Verdict**: `rederived` for this scoped device-reduction refactor.
- **Suggested ledger change**: retain `rederived`; append this scoped comparison
  to Checks. Actual CUDA execution, published anchors and measured calibration
  are not established by this review.

## Charge-jitter likelihood, published Martin–Puplett observable and scan convergence (fresh-context review, 2026-11-09)

Scope: commit `405d97a44` — `optical_ctr_jitter.py` (`ChargeJitterModel`, `_evaluate`, `jitter_profile_at_sigma`), its use in `fit_gaussian_ctr(charge_jitter=)`, `optical_interferogram_convergence.py`, and `tests/detectors/test_optical_interferometer_leissner.py`. Each derivation below was written before the corresponding implementation body was opened. Scripts: `scratch/v2_jitter_moments.py`, `scratch/v2_jitter_coverage.py`, `scratch/v2_jitter_coverage_gauss.py`, `scratch/v2_leissner_features.py`, `scratch/v2_convergence_checks.py` (not committed; run with `python -P`, because an existing `scratch/coverage.py` shadows the `coverage` package otherwise).

### A. Shot-averaged charge jitter

**Shot model.** With $N=q\bar N$, $\langle q\rangle=1$, $\operatorname{Var}q=r^2$, the pair factor of one shot is

$$
g(q)=\frac{N(N-1)}{\bar N(\bar N-1)}=\frac{\bar N q^2-q}{\bar N-1},
\qquad
V_p=a_p q+c_p g(q)=\alpha_p q+\beta_p q^2,
$$

$$
\alpha_p=a_p-\frac{c_p}{\bar N-1},\qquad \beta_p=\frac{\bar N c_p}{\bar N-1}.
$$

No large-$\bar N$ step is used, so everything below is exact at finite $\bar N$.

**Raw average.** Write $m_k=\langle q^k\rangle$. The scan value is the mean of $n$ independent shots, so

$$
\langle\bar V_p\rangle=a_p+c_p\frac{\bar N m_2-1}{\bar N-1}=a_p+c_p\Bigl(1+\frac{\bar N r^2}{\bar N-1}\Bigr),
$$

which needs only $m_1,m_2$ and holds for any charge law. The port covariance is

$$
n\,\operatorname{Cov}(\bar V_p,\bar V_{p'})=\alpha_p\alpha_{p'}r^2+(\alpha_p\beta_{p'}+\beta_p\alpha_{p'})(m_3-m_2)+\beta_p\beta_{p'}(m_4-m_2^2),
$$

which needs $m_3$ and $m_4$. In central moments $\mu_k$: $m_3-m_2=2r^2+\mu_3$ and $m_4-m_2^2=4r^2+4\mu_3+\mu_4-r^4$. The Gaussian assumption enters only here: $\mu_3=0$, $\mu_4=3r^4$ gives $2r^2$ and $4r^2+2r^4$.

**Per-shot normalization.** $V_p/q=\alpha_p+\beta_p q$ exactly, so $\langle\cdot\rangle=a_p+c_p$ and $n\operatorname{Cov}=\beta_p\beta_{p'}r^2$ for any law with $q>0$. This assumes $q$ is measured without error and that baselines are subtracted before the division. Readout noise added after the division is a separate per-port term.

**Likelihood.** For independent delays with $2\times2$ covariance $C_i(\theta)=R_i+J_i(\theta)$,

$$
-2\ln L=\sum_i\Bigl[\mathbf d_i^{\mathsf T}C_i^{-1}\mathbf d_i+\ln\det C_i\Bigr]+\text{const}.
$$

The log-determinant cannot be dropped because $J_i$ depends on $\sigma$ and on the scale. Constant port offsets $\mathbf o$ enter only the mean. If they are additive baselines (not proportional to $q$), they leave $C_i$ unchanged and are profiled exactly by GLS: $\mathbf o=(\sum_iW_i)^{-1}\sum_iW_i\mathbf d_i$. If a common scale $s$ multiplies the mean-bunch voltages, it multiplies $a$ and $c$, so $J\propto s^2$. The profile over $(s,\mathbf o)$ with $\Delta(-2\ln L)=3.841$ is the Wilks interval. It is asymptotic (in delay count and in signal-to-noise), not exact. It is justified for the full Gaussian likelihood, including the log-determinant term, and is not justified for the quadratic form alone.

**Comparison.**

| Term | Independent | Code | Result |
|---|---|---|---|
| Pair mean factor, raw | $1+\bar Nr^2/(\bar N-1)$ | `pair_mean_factor` | match |
| Pair mean factor, normalized | $1$ | `pair_mean_factor` | match |
| $\alpha,\beta$ | as above | `port_covariance` lines 98–99 | match |
| Raw covariance | $r^2,\ 2r^2,\ 4r^2+2r^4$ (Gaussian $\mu_3,\mu_4$) | lines 107–111 | match; Gaussian moments stated |
| Normalized covariance | $\beta\beta' r^2/n$ | line 105 | match; exact for any law |
| Scale | multiplies $a,c$ in mean and covariance | `_evaluate` lines 132–133 | match |
| Offsets | GLS, outside $C$ | line 141 | match |
| $\ln\det$ | $\sum\ln\det C_i$ up to a constant | line 144 (minus readout-only constant) | match |
| Scale prior | adds $((s-1)/\sigma_s)^2$ | line 174 | match |
| Threshold | $\Delta=3.841$ on the profiled objective | `fit_gaussian_ctr` threshold | match |

**Moments by direct shot simulation.** These use independent code: $2\times10^5$ trials of 8 shots, $\bar N\in\{3,40,6\times10^6\}$, $r\in\{0.08,0.3\}$, and Gaussian and gamma charge laws. My closed forms agree with the code to $2\times10^{-16}$ for the mean and to rounding for the covariance whenever the moments are Gaussian. The simulated means are within $2.1$ standard errors in every case. Simulated covariance agrees with the code within 1 % (Gaussian law, both modes) and within 0.5 % (gamma law, normalized mode). Under a gamma law in raw mode, the code's Gaussian-moment covariance is off by 3.4 % at $r=0.08$ and 43 % at $r=0.3$. My moment form with the gamma $m_3,m_4$ agrees within 0.4 %. The stated restriction ($r\ll1$, Gaussian moments) is therefore necessary, and the error is linear in skewness times $r$.

**Interval coverage.** Each case uses 41 delays, 120 energy nodes, $\sigma=200$ fs, $\bar N=6\times10^6$, $r=0.1$ and 8 shots. Jitter RMS is 21 (normalized) to 380 (raw) times the readout RMS. The true scale is 1.3 when fitted; with fitted offsets the true offsets are $(2,-1)\times10^{-5}$ V. Scans come from independent shot draws.

| Mode | fit_scale | offsets | trials | coverage | median CI width [fs] |
|---|---|---|---|---|---|
| raw | False | False | 400 | 0.935 ± 0.012 | 1.10 |
| raw | False | True | 400 | 0.938 ± 0.012 | 1.13 |
| raw | True | False | 400 | 0.948 ± 0.011 | 0.94 |
| raw | True | True | 400 | 0.927 ± 0.013 | 0.94 |
| normalized | False | False | 400 | 0.935 ± 0.012 | 1.06 |
| normalized | False | True | 400 | 0.927 ± 0.013 | 1.11 |
| normalized | True | False | 400 | 0.930 ± 0.013 | 0.89 |
| normalized | True | True | 400 | 0.940 ± 0.012 | 0.89 |
| raw, shots | False | False | 1000 | 0.949 ± 0.007 | — |
| raw, exact Gaussian data | False | False | 1000 | 0.947 ± 0.007 | — |
| normalized, shots | False | False | 1000 | 0.952 ± 0.007 | — |
| normalized, exact Gaussian data | False | False | 1000 | 0.935 ± 0.008 | — |

The pooled coverage is $0.941\pm0.003$ over 7200 fits. Every bias is below 0.02 fs against an SD of 0.24–0.30 fs, and no interval is open. In each of the 4000 fits of the last four rows, the code's interval contained the truth exactly when my independent $-2\ln L$ (own mean and covariance formulas) gave $\Delta<3.841$, and the mean $\Delta$ was 0.995–1.074. The roughly 1 % undercoverage also occurs for exactly Gaussian data, so it is finite-sample Wilks behaviour, not a model error.

**Findings (Part A).**

1. Low, stale metadata. `optical_ctr_inference.py:623` (`conditional_statistics`: "Independent normal voltage errors, known variances; delta chi2") and `:645` (`model_assumptions`: "conditional independent Gaussian voltage errors") are still written into a jitter fit. There, errors are port-correlated, and their covariance depends on the parameters.
2. Informational. In raw mode the covariance error is linear in the skewness of the charge law (gamma law: 3.4 % at $r=0.08$, 43 % at $r=0.3$). The docstring states the Gaussian-moment assumption correctly; a skewed measured charge law needs $\mu_3,\mu_4$.
3. Informational. The scale search is bounded to $e^{\pm3}$ around the chi-square scale. A clamped chi-square scale of 0 re-centres it at 1. No case failed here.
4. Informational. `with_charge_jitter` accepts $r\ge1$, but `ChargeJitterModel` refuses it (`optical_ctr_inference.py:148`).

Part A verdict: **rederived**.

### B. Published forward observable: Leissner et al., PAC 1999, p. 2172

**Eq. (5) from Eq. (4).** Expanding,

$$
\tilde S(\omega)=e^{-\sigma^2\omega^2}-2e^{-(\sigma^2+\xi^2)\omega^2}+e^{-(\sigma^2+2\xi^2)\omega^2},
$$

and with $\int_0^\infty e^{-a\omega^2}\cos\omega t\,d\omega=\tfrac12\sqrt{\pi/a}\,e^{-t^2/4a}$, multiplying by $2\sigma/\sqrt\pi$:

$$
S(t)=e^{-t^2/4\sigma^2}-\frac{2\sigma}{\sqrt{\sigma^2+\xi^2}}e^{-t^2/4(\sigma^2+\xi^2)}+\frac{\sigma}{\sqrt{\sigma^2+2\xi^2}}e^{-t^2/4(\sigma^2+2\xi^2)} .
$$

This is Eq. (5) as printed, confirmed numerically against quadrature to $10^{-15}$. $\int S\,dt\propto\tilde S(0)=0$.

**Ideal Martin–Puplett with energy-integrating detectors.** With spectral energy density $u(\omega)$ and $\tau=\mathrm{OPD}/c=2x/c$, $U_{1,2}=\tfrac12\int u(\omega)[1\pm\cos\omega\tau]\,d\omega$, so

$$
\frac{U_1-U_2}{U_1+U_2}=\frac{\int u(\omega)\cos\omega\tau\,d\omega}{\int u(\omega)\,d\omega}=\frac{S(\tau)}{S(0)}
$$

when $u\propto\tilde S$. A photon-density input must be $u/E\propto\tilde S/E$ for a flat absorptance and gain, because the detector weights photons by $E$. The test's `power / energy` mapping is correct.

**Spectral peak.** $d\ln\tilde S/d(\omega^2)=0$ gives $e^{-\xi^2\omega_p^2}=\sigma^2/(\sigma^2+2\xi^2)$, matching the test's `_xi_from_spectral_peak`, with $\omega=2\pi f$.

**My figure readings** (400 dpi render, axis ticks located in pixels):

| Feature | Reading | Uncertainty |
|---|---|---|
| Fig. 4 solid-curve maximum | 0.1056 THz | ±0.005 THz (polyline vertex) |
| Fig. 4 data maximum | ≈0.115 THz | — |
| Fig. 3 solid-curve central maximum | 0.895 | ±0.01 (absolute; not 1) |
| Fig. 3 first zero | ±2.24 ps | ±0.10 ps |
| Fig. 3 side minima, position | ±4.1 ps | ±0.2 ps |
| Fig. 3 side minima, value | −0.547 absolute | ±0.01 |
| Fig. 3 side minima / central maximum | −0.611 | ±0.015 |
| Fig. 3 secondary maxima | ≈ +0.09 at ±8.5 ps | — |
| $\sigma$ | 2 ps (Sec. 3.1, 20–30 % systematic) | — |

**Consistency.** $\sigma=2$ ps with $f_p=0.1056$ THz gives $\xi=0.753$ ps. The predicted first zero is 2.23 ps, the minimum is at 4.08 ps with depth/peak −0.616, and the side maximum is at 8.62 ps (0.103 on the figure's scale). All match my readings within uncertainty. With $f_p=0.100$ THz the test's $\xi=1.074$ ps gives 2.35 ps, 4.29 ps and −0.612. The figures are therefore consistent with $\omega=2\pi f$ and with Fig. 3 delay $=\mathrm{OPD}/c$. A mirror-travel delay would double every time feature. The depth ratio hardly constrains $\xi$: it changes by 0.006 over 0.10–0.11 THz.

**Test review.**

- `test_two_port_difference_equals_published_closed_form`: the forward model (Jones transfer at 45°, OPD $=2x$, $\int\eta E\,n\,dE$) reproduces Eq. (5)/$S(0)$ to $10^{-10}$ for four $\xi$. Correct.
- `test_published_fit_features_of_figures_3_and_4`: these pinned readings are biased. The test uses 0.10 THz, a 2.5 ps zero, 4.5 ps minima and "−0.55 of the central maximum". Mine are $0.1056\pm0.005$ THz, $2.24\pm0.1$ ps, $4.1\pm0.2$ ps and $-0.611\pm0.015$ of the central maximum. −0.55 is the absolute plotted value; the curve peaks at 0.895, not 1. The tolerances (0.5 ps, 0.1) are 2.5–7 times the reading uncertainty. The model passes either way, so this is not a physics error. Still, the comment misstates what is pinned, and the depth check at ±0.1 is weak.
- Not exercised: absolute modulation depth (Fig. 3 peaks at about 0.9, so the real instrument is not ideal), measured data points, calibration, non-45° or lossy optics, inference (`fit_gaussian_ctr`, jitter), and Fig. 5.

**Anchor scope.** Yes, `anchored` is supported for "published closed-form forward observable of an ideal Martin–Puplett with an energy-integrating detector; fit features of Figs. 3–4; no measured calibration, no digitized data". This holds provided the figure readings and the depth wording are corrected, for example to 0.105 ± 0.01 THz, 2.24 ± 0.2 ps, 4.1 ± 0.3 ps and −0.61 ± 0.03 relative to the central maximum (these tolerances are about twice the reading uncertainty). Two more scope items should be added: "normalized to the central maximum", and that inference and charge jitter are not covered. The anchor exercises the Jones transfer (ideal 45° only), the double-pass OPD and the energy-weighted detector integral. It does not exercise calibration, inference, absolute response or non-ideal optics.

Part B verdict: **rederived** for the forward observable. Eq. (5) is confirmed as printed; the pinned figure readings are imprecise (low severity).

### C. `interferogram_convergence`

**Constant removal.** Any two-beam port spectrum has the form $V_p(\phi)=A_p+\operatorname{Re}[B_pe^{i\phi}]$ after a linear detector. With $\phi\to\phi+\pi$, $B_p\to-B_p$, so $[V_p(\phi)-V_p(\phi+\pi)]/2=\operatorname{Re}[B_pe^{i\phi}]$ exactly. `_port_spectra` is $(\mathrm{dc}+\mathrm{fringe}\cdot q\cos\phi)\cdot\rho\cdot\eta_p$, linear in $\cos\phi$ (the $\max(\cdot,0)$ clamp acts only on roundoff), and the detector integral is linear below saturation, which is refused. Numerically, for lossy arms (0.8, 0.6), grids at 0.7 and 0.9 rad, ports (0.9, 0.7), OPD jitter and sloped, unequal detectors, $[V(0)+V(\pi)]/2$ is constant over 801 positions to $6\times10^{-16}$ relative.

**Inversion.** For $D(l)=\int_0^\infty W(E)\cos(2\pi El/hc)\,dE$ on $l\in(-\infty,\infty)$,

$$
\int_{-\infty}^{\infty}D(l)\cos(2\pi E'l/hc)\,dl=\frac{hc}{2}W(E') \quad\Rightarrow\quad W=\frac{2}{hc}\int_{-\infty}^{\infty}D\cos\,dl .
$$

The constant is $2/hc$ for a two-sided integral in OPD, which the code uses (it would be $4/hc$ for a one-sided OPD integral, and also $4/hc$ for a two-sided integral over mirror travel $x$, since $dl=2\,dx$). `np.gradient` on a uniform grid with halved ends gives trapezoid weights. For ideal 45° grids, the recovered $W$ matches $\eta g e\,u(E)$ to $5\times10^{-6}$ of the maximum.

**Nyquist and resolution.** Sampling $\cos(2\pi l/\lambda)$ needs $\Delta l<\lambda_{\min}/2$, with $\Delta l=2\Delta x$ and $\lambda_{\min}=hc/E_{\max}$, so the ratio is $2\Delta x/(hc/2E_{\max})$, as in line 137. The symmetric span $2l_{\max}$ gives $\Delta E=hc/(2l_{\max})$ with $l_{\max}=$ `opd[-1]`. Both agree with my values to $10^{-8}$ relative. A ratio of 1.2 produces a delay-step change of 0.064.

Part C verdict: **rederived**. One informational note: the Nyquist test uses the band edge, not the signal support, which is conservative.

### Verifier output

- **Claim**: `optical-instrument-observable` — `optical_ctr_jitter.py::{ChargeJitterModel, jitter_profile_at_sigma}`, `optical_ctr_inference.py::fit_gaussian_ctr`, `optical_interferogram_convergence.py::interferogram_convergence` — shot moments, Gaussian $-2\ln L$; Leissner et al. PAC 1999 Eqs. (4)–(5), Figs. 3–4.
- **Filters**: units pass; limits pass ($r=0$ reduces to chi-square; $\bar N$ finite is exact); signs/conventions pass.
- **Re-derivation**: `matches`. Divergences are wording/readings only: stale metadata `optical_ctr_inference.py:623,645`; Leissner test readings (0.10 THz, 2.5 ps, 4.5 ps, −0.55 "relative") against 0.1056 THz, 2.24 ps, 4.1 ps, −0.611.
- **Verdict**: `rederived` (A, B, C).
- **Suggested ledger change**: `anchored` with the proposed scope once the Leissner readings and depth wording are corrected. Add "normalized to the central maximum; ideal 45° optics; not inference, jitter, calibration or absolute modulation depth". Record jitter coverage 0.941 ± 0.003 over 7200 fits. Never `signed-off`.

## Transverse-source and detector-response systematic modes (fresh-context review, 2026-11-14)

**Scope.** Commit `88994e94a`:
`src/pyrite/detectors/optical_ctr_systematics.py::{CTRSystematicMode, systematic_voltage_modes}`,
and their use in
`src/pyrite/detectors/optical_ctr_inference.py::{_nuisance_design, _solve_with_priors, _profile_at_sigma, fit_gaussian_ctr(systematics=)}`;
tests `tests/detectors/test_optical_ctr_systematics.py`; the physics section
"Declared transverse-source and detector-response shapes" in
`docs/physics/radiation-physics/optical-wavefront-and-detectors.md`. The
response kernel $R$, the Gaussian template, and the plain/jitter fits are
covered by the earlier sections and are used here as given.

### S.1 Independent derivation

This section was written from the module and function docstrings, the physics section and the signatures, before the implementation bodies were read.

**Linear model.** The calibrated port voltage is $V_p(d)=\sum_E R_p(d,E)\,\Phi(E;\sigma)$, with
$R_p=T_p(d,E)\,\eta_p(E)\,E\,w_E\,e\,G_p$ ($T_p$ the Martin–Puplett port transfer, $w_E$ the trapezoid width) and
$\Phi=\Phi_{\rm self}+\Phi_{\rm pair,0}\,F(E;\sigma)$, $F=\exp[-(E\sigma/\hbar)^2]$.
A transverse 1σ fractional change $u_k$ of the pair envelope gives
$\Phi_{\rm pair,0}\to\Phi_{\rm pair,0}(1+a_ku_k)$. This is a source change, so it acts on both ports:

$$
D_k=R\,(u_k\,\Phi_{\rm pair,0}\,F).
$$

A response change $\eta_p\to\eta_p(1+a_ku_k)$ enters $R_p$ linearly:

$$
D_k[:,p]=R_p\,(u_k\,\Phi),\qquad p\in\{0,1\}\ \text{or both}.
$$

Both are exactly linear in $a_k$. With a source scale $s$ multiplying $\Phi$, the **exact** expectation is

$$
V=s\Bigl(K+\sum_k a_kD_k\Bigr)+\mathbf o .
$$

The implemented model $sK+\sum_ka_kD_k+\mathbf o$ is the reparametrization $a_k'=s\,a_k$. It is exact in the mean, but the prior $a_k\sim\mathcal N(0,1)$ becomes $a_k'\sim\mathcal N(0,s^2)$. Keeping a unit prior on $a_k'$ therefore declares a fractional 1σ of $u_k/s$ instead of $u_k$. The error is a factor $s$ on the systematic width, so it is first order in $s-1$ only while $s\approx1$. For a free scale with $s\to0$ the declared mode becomes effectively unconstrained.

**Marginal identity.** For residual $\mathbf r=\mathbf y-sK-\mathbf o$ and diagonal readout covariance $C$, completing the square in $\mathbf a$ (Woodbury) gives

$$
\min_{\mathbf a}\Bigl[(\mathbf r-D\mathbf a)^{\mathsf T}C^{-1}(\mathbf r-D\mathbf a)+\mathbf a^{\mathsf T}\mathbf a\Bigr]
=\mathbf r^{\mathsf T}(C+DD^{\mathsf T})^{-1}\mathbf r,
\qquad
\hat{\mathbf a}=D^{\mathsf T}(C+DD^{\mathsf T})^{-1}\mathbf r .
$$

Joint minimization over $(s,\mathbf o,\mathbf a)$ is therefore GLS in $(s,\mathbf o)$ with covariance $C+DD^{\mathsf T}$. A scale prior adds $((s-1)/f)^2$. The constraint $s\ge0$ makes this a convex quadratic program with one bound. If the unconstrained optimum has $s<0$, the constrained optimum lies at $s=0$ with the other parameters re-solved, and the prior term then contributes $1/f^2$. Because $D_k(\sigma)$ depends on $\sigma$, this profile is the constrained-nuisance (auxiliary-measurement) likelihood. It is not the Gaussian marginal likelihood, which would add $\ln\det(C+DD^{\mathsf T})$. The profile is a valid likelihood with $\mathbf a$ as parameters and $\mathbf a_{\rm obs}=\mathbf 0$ as unit-variance pseudo-data, so $\Delta\chi^2=3.841$ is its one-parameter Wilks interval. Each mode adds one parameter and one pseudo-datum, so the degrees of freedom are unchanged. Its frequentist guarantee is for the ensemble in which the pseudo-measurement fluctuates, i.e. $a\sim\mathcal N(0,1)$. For a *fixed* true pull the shrinkage of $\hat a$ towards 0 biases $\hat\sigma$ whenever $D_k$ is not orthogonal to $\partial K/\partial\sigma$.

**Prior pseudo-rows with normalized columns.** For a column $x_j=c_j/n_j$ the coefficient is $\beta_j=\theta_jn_j$. The scale row $\beta_s/(fn_s)-1/f=(s-1)/f$ and the pull row $\beta_k/n_k=a_k$ are correct.

**Degeneracy.** For $u=-A(E/E_0)^2$,
$D=-\tfrac{A}{E_0^2}R(E^2\Phi_{\rm pair,0}F)$ and
$\partial K/\partial\sigma=-\tfrac{2\sigma}{\hbar^2}R(E^2\Phi_{\rm pair,0}F)$. So

$$
D=\frac{A\hbar^2}{2\sigma E_0^2}\,\frac{\partial K}{\partial\sigma}
$$

holds *exactly*, and the shift per unit pull is $\delta=A\hbar^2/(2\sigma E_0^2)$. Units: eV²s²/(s·eV²) = s. Only the linearization of $K$ in $\sigma$ is first order. To that order the profile half-width becomes $1.96\sqrt{\sigma_{\rm stat}^2+\delta^2}$.

**`from_shifted_density`.** $u=(\Phi_{\rm shift}-\Phi_{\rm nom})/\Phi_{\rm nom}$ where $\Phi_{\rm nom}>0$, else 0. Expected refusals: a port on a transverse mode, an all-zero mode, a grid mismatch, duplicate names, and combination with `charge_jitter` (the jitter $-2\ln L$ has a parameter-dependent covariance and is a separate likelihood).

### S.2 Comparison with the implementation

| Item | Independent | Code | Result |
|---|---|---|---|
| Transverse $D_k$ | $R(u\Phi_{\rm pair,0}F)$, both ports | `optical_ctr_systematics.py:147` | match |
| Response $D_k$ | $R_p(u\Phi)$ on the declared port(s) | `:149–151` | match |
| $\hbar$ | $0.6582119569\ \mathrm{eV\,fs}$ | `:48` | match |
| Scale/pull prior rows | $(s-1)/f$; $a_k$ | `optical_ctr_inference.py:403–414` | match |
| Negative-scale clamp | $s=0$, re-solve, prior adds $1/f^2$ | `:467–471`, `:483–484` | match |
| Reported $\chi^2$ | data + $\sum a_k^2$ + scale prior | `:482–484` | match |
| d.o.f. | unchanged | `:619`, `:723` | match |
| Mode/scale coupling | $s\sum a_kD_k$ (exact) | $\sum a_kD_k$ (`:481`) | **diverges** for $s$ far from 1 (S.4, finding 1) |
| `from_shifted_density`, refusals | as above | `optical_ctr_systematics.py:70–130`; `optical_ctr_inference.py:624–626` | match |

### S.3 Numerical checks (independent scripts, `scratch/syst_*.py`)

The setup is 150 (coverage: 120) energy nodes over 0.5–25 meV, 61 (41) delays, unequal port efficiencies and gains, and $\sigma=200$–210 fs. $R$ is taken from `_response_kernel` as a fixed linear operator; it reproduces `gaussian_ctr_forward_voltage` to $1.1\times10^{-10}$, which is the $\hbar$ rounding of my script.

- **$D_k$.** A central finite difference of my exact forward model (pair envelope or port efficiency scaled by $1\pm u_k$) agrees with `systematic_voltage_modes` to $1.1\times10^{-10}$ for transverse, port-0, port-1 and both-port response modes. Single-port modes leave the other port exactly zero.
- **Marginal identity with nuisances.** Six cases: scale only; scale and offsets; scale, offsets and prior $f=0.1$; fixed scale with offsets; and the negative-scale clamp without and with a prior. Comparing `_profile_at_sigma` to my GLS with $C+DD^{\mathsf T}$, $\chi^2$ agrees to $\le9\times10^{-10}$, $s$ to $10^{-8}$, offsets to $3\times10^{-14}$ V, and pulls (against $D^{\mathsf T}(C+DD^{\mathsf T})^{-1}\mathbf r$) to $2\times10^{-8}$.
- **Degeneracy.** The least-squares ratio $D/(\partial K/\partial\sigma)$ equals $A\hbar^2/(2\sigma E_0^2)$ to $2\times10^{-8}$ at 100, 200 and 400 fs, with an orthogonal residual of $9\times10^{-9}$, so the relation is exact. For noiseless data at 200 fs, $A=0.02$ and $E_0=3.3$ meV, the widening is $\sqrt{w_1^2-w_0^2}/1.96=1.965$ fs against $\delta=1.989$ fs: 1.2 % linearization error.
- **Coverage** (1000 seeded scans per row, $\sigma_{\rm true}=200$ fs, bounds 20–1000 fs, 61 profile points, `rip` $=0.2\cos(2\pi E/6\,\mathrm{meV})$ transverse, `r1` $=0.05\sin(3E/E_{\max})$ port-1 response, `g` $=-0.1(E/3.3\,\mathrm{meV})^2$ transverse):

| Data / fit | noise [V] | coverage | median width [fs] |
|---|---|---|---|
| no modes declared, true `rip` pull 1.5 | $10^{-3}$ / $3\times10^{-4}$ | 0.168 / 0.000 | 12.8 / 3.8 |
| `rip`+`r1`, pulls $\sim\mathcal N(0,1)$, fixed scale | $3\times10^{-4}$ | 0.937 ± 0.008 | 9.7 |
| `g` (degenerate), pull $\sim\mathcal N(0,1)$, fixed scale | $10^{-3}$ / $3\times10^{-4}$ | 0.952 / 0.953 (± 0.007) | 46.1 / 44.1 |
| `rip`+`r1`, pulls fixed (1.5, −1), fixed scale | $10^{-3}$ / $3\times10^{-4}$ | 0.877 / 0.928 | 24.1 / 9.9 |
| `g`, pull fixed 1.5, fixed scale | $10^{-3}$ / $3\times10^{-4}$ | 0.970 / 1.000 | 42.2 / 39.7 |
| `rip`+`r1`, pulls $\sim\mathcal N(0,1)$, **scale fitted**, $s_{\rm true}=1.0$ | $3\times10^{-4}$ | **0.851 ± 0.011** | 32.8 |
| same, $s_{\rm true}=1.2$ / 2.0 | $3\times10^{-4}$ | **0.855 / 0.875** | 25.7 / 14.8 |
| no systematics, no modes, scale fitted (300 scans) | $3\times10^{-4}$ | 0.960 ± 0.011 | — |
| `rip`+`r1` $\sim\mathcal N(0,1)$, scale fitted with prior $f=0.1$ (300) | $3\times10^{-4}$ | 0.990 ± 0.006 | — |

The fixed-pull rows are predicted by my linearized GLS bias/variance calculation: 0.865, 0.970, 0.917 and 1.000 against the observed 0.877, 0.970, 0.928 and 1.000. That undercoverage is the documented shrinkage property of constrained nuisances, not an implementation error. The fitted-scale rows are not explained by it (finding 1).

### S.4 Findings (no code change made)

1. **High: the mode change is not scaled by the fitted source scale.** The code computes `predicted = scale*model + offsets + tensordot(pulls, drift)` at `optical_ctr_inference.py:479–481`, with $D_k$ at unit scale (`optical_ctr_systematics.py:133–152`). The exact model is $s(K+\sum a_kD_k)$. Profiling $a'=sa$ under a unit prior declares fractional 1σ $u_k/s$. The docstring (`optical_ctr_systematics.py:25–27`) and the physics page (`optical-wavefront-and-detectors.md:855`) call this "first order in $(s-1)u_k$". That holds only for $s\approx1$, and `fit_scale=True` (the default, `:573`) lets $s$ go to 0 at small $\sigma$. In seeded trial 26 (true pulls (−0.58, 1.77), $\sigma_{\rm true}=200$ fs), the code reports $\hat\sigma=49.2$ fs with CI $(41.2, 53.5)$ fs, excluding the truth. At 49.2 fs the code profile is $\chi^2=95.06$ with $s=0.079$ and $a=(0.69,-0.18)$, i.e. fractional pulls of about 9σ. My exact bilinear profile there is 132.96, against 100.01 at 200 fs, so it rejects 49 fs at $\Delta\chi^2\approx33$. Coverage with a fitted scale is 0.851–0.875 against 0.937–0.953 with a fixed scale and 0.990 with a scale prior. The mean is still exact, so with the scale fixed or tightly constrained the implementation is correct. *Divergent term:* $\sum_ka_kD_k$ must be $s\sum_ka_kD_k$ (or the pull prior $a_k'^2/s^2$). Either profile the bilinear model exactly, linear in $(\mathbf a,\mathbf o)$ for each $s$ with a one-dimensional search or Gauss–Newton in $s$, or refuse `systematics` with an unconstrained `fit_scale`. The reported pull equals $s\,a$, not $a$ (1.514 against a true 1.009 at $s=1.5$).
2. **Medium: the pseudo-row conditioning failure falsely closes intervals.** `_nuisance_design` normalizes each column by its maximum and floors the norm at $10^{-300}$ (`optical_ctr_inference.py:410`). The pull pseudo-row entry is then $1/n_k$. `np.linalg.lstsq(..., rcond=None)` (`:428`) truncates every other singular value once $\max\lvert D_k\rvert/\delta y\lesssim10^{-13}$. In that case the scale, offsets and other pulls are zeroed: $\chi^2=2.1\times10^7$ instead of 140.19, and the drift already starts at $1.35\times10^{-13}$ (140.65). An exactly zero $D_k$ (a mode supported where $\Phi_{\rm pair,0}=0$) always triggers it. It occurs naturally in a $\sigma$ scan, because $F$ underflows at high $E$. With $u=-0.2(E/E_{\max})^8$ the code profile is 264772 against an exact 150334 at 2000 fs, and 2.1e7 against 150308 at 3000 fs. In a low-SNR scan ($\sigma_{\rm true}=1300$ fs, noise $3\times10^{-4}$ V) the no-mode fit is open on both sides. Adding this mode makes the code report $\hat\sigma=43.3$ fs with upper end 54.7 fs. The exact profile is admissible to the 3000 fs bound ($\Delta=3.53$), so the correct answer is open/multimodal. Fix: parametrize pulls directly, with column $D_k$ weighted by $1/\delta y$ and prior row 1 (no column normalization), or drop modes with negligible $\lvert D_k\rvert$.
3. **Low: fixed-pull undercoverage should be stated.** The interval covers at the nominal rate when the declared 1σ is a true prior spread. A fixed systematic offset that is not orthogonal to $\partial K/\partial\sigma$ is shrunk, giving coverage 0.877 at noise $10^{-3}$ for pulls (1.5, −1). The physics page says only that "the interval includes the declared systematics".
4. **Informational.** The Gaussian-likelihood marginal alternative would add $\ln\det(C+D(\sigma)D(\sigma)^{\mathsf T})$. The code's constrained-nuisance profile is a legitimate, different construction; its wording "equals the chi-square with covariance $C+\sum DD^{\mathsf T}$" is correct for the quadratic form at fixed $(s,\mathbf o)$. `from_shifted_density` silently sets $u=0$ where the nominal density is 0 and the shifted one is not (new support is dropped). Response modes may drive $\eta(1+au)$ outside $[0,1]$ for large pulls (linear model only). `fit_gaussian_ctr` recomputes `_profile_at_sigma` at the optimum only to recover pulls (`:666–679`); this is redundant but consistent.
5. Tests: `test_optical_ctr_systematics.py` and `test_optical_ctr_inference.py` pass (17). The identity test uses `fit_scale=False` only. The degeneracy and nondegenerate-recovery tests use fixed scale, and the unperturbed test with `fit_scale=True` has $s\approx1$, so none exercises finding 1 or 2.

### S.5 Verdict

- **Claim**: `optical-instrument-observable` (systematic-mode extension) — `optical_ctr_systematics.py::{CTRSystematicMode, systematic_voltage_modes}`, `optical_ctr_inference.py::{_nuisance_design, _solve_with_priors, _profile_at_sigma, fit_gaussian_ctr}` — linear unit-Gaussian nuisances, Woodbury marginal $C+DD^{\mathsf T}$, Wilks $\Delta\chi^2=3.841$.
- **Filters**: units pass; limits pass (zero pulls reduce to the plain fit; the degeneracy shift is exact); signs/conventions pass.
- **Re-derivation**: `differs`. $D_k$, prior rows, clamp, reported $\chi^2$, d.o.f., degeneracy coefficient, `from_shifted_density` and refusals match. The mode/scale coupling diverges: the code uses $\sum a_kD_k$ where the model is $s\sum a_kD_k$, which is not first order when the scale is fitted. There is also a numerical conditioning failure for negligible $D_k$.
- **Verdict**: `discrepancy` for `systematics` with `fit_scale=True` and no tight scale prior (coverage 0.85–0.88; false precise intervals) and for modes with vanishing voltage effect. `rederived` holds only for the fixed-scale path and the closed-form pieces (items 1, 2, 4, 5 of the scope).
- **Suggested ledger change**: do not extend `rederived` to the systematic modes yet. Add to Notes: "2026-11-14 fresh-context review: systematic modes match at fixed scale (identity to 1e-9, degeneracy exact, coverage 0.94–0.95 with N(0,1) pulls); discrepancy with fitted scale (mode not multiplied by s; coverage 0.85) and lstsq pseudo-row conditioning for negligible modes." Keep the row `anchored` for the existing scope. Never `signed-off`.

### S.6 Resolution re-check (2026-11-14, uncommitted working tree on `88994e94a`)

This re-check covers the fixes for findings 1, 2 and 4. `_nuisance_design` now leaves the pull columns unnormalized, with prior row 1. A new `_bilinear_profile` profiles $s(K+\sum_ka_kD_k)+\mathbf o$: at fixed $s$ it is linear in $\mathbf a$ and $\mathbf o$, and $s$ is scanned on $[0,s_{\max}]$ with 33 points and refined by bounded Brent in the best cell. `from_shifted_density` now refuses new support, and the docstring and physics page were updated. The tests (systematics, inference and jitter) pass: 28 passed.

**Finding 2 (conditioning): resolved.** Rerunning `syst_check_edge.py` gives $\chi^2=140.187$ for mode amplitudes from $10^{-2}$ down to $10^{-20}$, with pulls $\le6\times10^{-7}$. An exactly zero $D_k$ reproduces the mode-free fit ($s=1.389792$, $\chi^2=87029.9$). The low-SNR open-end scenario now matches the mode-free fit: at $\sigma_{\rm true}=1300$ fs the interval is (None, None) for no mode, the high-$E$ mode and the linear mode, and the 900 fs case agrees with the no-mode result.

**Finding 1 (scale coupling): resolved in the model; the $s$ search is not global.** Re-fitting trial 26 now gives $\hat\sigma=202.3$ fs with CI (186.9, 218.5) fs, which contains 200 fs; the pulls are $(-0.76, 1.06)$ against true $(-0.58, 1.77)$. With the fitted scale near the true one, the code equals my brute-force exact bilinear profile. I also checked the negative-scale, $s_{\rm true}=0$, $10^{-3}$ and 0.02 cases, with and without offsets and with and without a prior (fitted $s$ from 0 to 0.72); all agree to the printed precision.

Brute force (`syst_fix_ssearch.py`) used my own augmented solver for $\chi^2(s)$, a 3001-point $s$ grid up to $\ge50\max(\hat s,1)$, and Brent refinement. Mode sets were (`rip`, `r1`), plus a uniform both-port response mode `big` with $u=0.05$, 0.2 or 0.6, `hiE`, and (`rip`, port-0 slope). These were crossed with offsets on/off and priors none/0.1/0.5 at $\sigma\in\{25,80,200,600,2000\}$ fs. $\chi^2(s)$ has more than one grid local minimum in 1.5–4.6 % of profiles.

- **Well-specified data** ($s_{\rm true}\in\{0.3,1,1.5,3\}$, no unmodelled offsets, modes including `big` $u=0.05$): 19 of 900 profiles miss the global minimum. All of them are at $\sigma=25$ or 2000 fs, far from the truth, and all are in sets containing the scale-like mode. Two mechanisms appear:
  - The cap $s_{\max}=3\max(\hat s_{\rm lin})$ (`optical_ctr_inference.py:501`) excludes an optimum at large $s$. Example: code 252177 at $s=48.8$ against a global 251707 at $s=259$.
  - With a scale prior, $s_{\max}\ge1+5f$ (`:503`) makes the 33-point linear grid (`:506`) too coarse near $s=0$, so a narrow minimum is skipped. Example: prior 0.1, code 711.2 at $s=0.98$ against a global 421.8 at $s=0.0042$.
- **End-to-end effect** (replayed trial 46 of the $u_{\rm big}=0.2$ run: no unmodelled offsets, $s_{\rm true}=0.05$, $\sigma_{\rm true}=150$ fs, prior 0.5, modes `rip`, `r1`, `big`): `fit_gaussian_ctr` reports CI (64.9, None) fs. The brute-force profile is admissible from 33 fs. At 33 and 50 fs the code profile is 109.9 and 108.9, while the global values are 92.95 and 92.55 at $s\approx0.003$–0.005, a minimum the grid on $[0,3.5]$ never samples. The lower endpoint is falsely precise.
- **Misspecified data** (unfitted offsets present; trial 31): the code closes the upper end at 133 fs where the exact profile is open. That global minimum, however, sits at $s\approx16$–24 with $1+0.2a_{\rm big}\approx0$, i.e. an unphysical zero efficiency, because response pulls are not clipped. With negative-scale data and a response mode, the exact profile likewise flips sign through large pulls ($s=0.71$ for $s_{\rm true}=-0.4$; code and brute force agree).

*Divergent term:* the $s$ search returns a local, cap-limited minimum rather than $\min_{s\ge0}$. Suggested fix: seed the Brent refinement from every grid local minimum, the linear guesses $\hat s_{\rm lin}$ and a log-spaced grid near 0; extend $s_{\max}$ while the argmin sits at the upper edge; and document or bound the implied efficiency $1+a_ku_k\ge0$ for scale-like response modes.

**Finding 4 and docs wording.** `from_shifted_density` refusal: resolved. Shrinkage, constrained-nuisance and unclipped-$\eta$ wording: resolved. One residual (low): `optical_ctr_systematics.py:23` and `optical-wavefront-and-detectors.md:852` still say the covariance at fixed scale is $C+\sum_kD_kD_k^{\mathsf T}$. Under the now-exact bilinear model it is $C+s^2\sum_kD_kD_k^{\mathsf T}$; the two coincide only at $s=1$ or with a fixed scale.

**Verdict:** still differs, but narrowed. The model, prior bookkeeping, conditioning, refusals and wording match, as does the fixed-scale path. The fitted-scale path matches whenever the bilinear minimum lies in $[0,s_{\max}]$ and is sampled by the grid. That holds for every well-specified case near the true duration I tested, but not globally: a falsely precise endpoint was shown with a 0.5 scale prior and $s_{\rm true}=0.05$. **Recommended status:** `rederived` may be extended to the fixed-scale `systematics` path and to the fitted-scale path with $s$ near its prior. Keep the fitted-scale profile flagged in Notes ("$s$ search local/capped; fix pending") until the search is made global and re-verified. Never `signed-off`.

### S.7 Final re-check (2026-11-14, commit `ba9474505`)

This re-check covers the physical pull box with the bounded least squares, the global $s$ search, and the refusal of a uniform two-port response mode under a free, prior-less scale. The systematics, inference and jitter tests pass (30).

**Box derivation: matches.** The constraint $1+a_ku_k(E)\ge0$ for all $E$ is linear in $a_k$. Where $u>0$ it binds at $a_k\ge-1/u_{\max}$, and where $u<0$ at $a_k\le-1/u_{\min}$, which is `_pull_bounds`. The bounds are transformed correctly to normalized coefficients (bound × norm, norm 1 for pulls), and the scale bound is $\ge0$. One informational point: the box is per mode. Several modes acting on the same quantity really require $1+\sum_ka_ku_k\ge0$; the per-mode box is neither necessary nor sufficient for that, though the difference only matters outside the Gaussian regime.

**Bounded solve: exact.** `syst_final_qp.py` uses my own solver, which enumerates every free/lower/upper assignment of the bounded variables and keeps the best feasible equality-constrained solution. That is exact for this convex QP. Over 720 solves (bilinear branch at fixed $s$ including $s=0$; linear branch with $s\ge0$ fitted; fixed scale; offsets on/off; shapes up to $\lvert u\rvert=1.5$ so the box binds), the largest relative $\chi^2$ difference is $5.6\times10^{-14}$ and the largest projected-gradient KKT violation of the code's solution is $6\times10^{-11}$. A pull bound was active in 136 of 240 fixed-$s$ cases.

**Replays.**
- Trial 26: 202.3 fs, CI (186.9, 218.5) fs.
- Trial 46 (prior 0.5): CI (None, None), which equals my bounded brute force (admissible over the full 20–3000 fs).
- Trial 31 (prior 0.5): CI (None, 116.9) fs, against brute-force admissibility on [20, 115.5] fs over 41 grid points, i.e. consistent.
- Both prior-less `big` cases are now refused.

**Global $s$ search: matches for non-uniform modes; misses remain for nearly uniform two-port response modes.** `syst_final_global.py` profiles $\chi^2(s)$ with exact box QPs at each $s$ (explicit residual), on $\{0\}\cup$ 521 log points over $10^{-7}$–$10^{6}\times s_{\rm ref}$, with Brent at every local minimum. It covered 3 seeds × 2 scans × 4 durations × 3 offset/prior settings per mode set. Every miss was confirmed by evaluating the code's own `_linear_profile` at my $s$, which reproduced my value, so the evaluator is right and only the search misses.
- `rip`+`r1`, `rip`+two-port slope ($u=0.1+0.15E/E_{\max}$), and uniform *transverse* modes with $c=0.05$, 0.3 and 0.6: 0 misses in 72 profiles each (the incoherent term breaks the transverse degeneracy).
- `rip`+`r1`+ a two-port response $u=c[1+\delta\cos(7E/E_{\max})]$, which passes the $10^{-9}$ ptp test for every $\delta\ge3\times10^{-9}$, with a free prior-less scale (misses per 72 profiles):

| $c$ \ $\delta$ | $3\times10^{-9}$ | $10^{-6}$ | $10^{-3}$ | $3\times10^{-2}$ |
|---|---|---|---|---|
| 0.05 | 12 | 14 | 2 | 2 |
| 0.3 | 12 | 15 | 3 | 1 |
| 0.6 | 16 | 20 | 3 | 1 |

The mechanism is the one the refusal targets, but without exact uniformity. At $a\to-1/u_{\max}$ (the box edge) the efficiency becomes $\eta(1-u/u_{\max})$, a nearly zero, $\delta$-shaped remnant. A large $s$ rescales it at a fixed penalty $1/u_{\max}^2$, while the other pulls become cheap ($a_j\propto1/s$). The code grows $s_{\max}$ only while the argmin sits at the upper edge, so a deeper minimum at $s\sim10$–$10^6$ beyond an interior local minimum is never sampled.

At $c=0.05$ the 400 penalty confines the misses to durations far from the truth, where $\chi^2\gtrsim600$. At $c\ge0.3$ they reach the admissible region. End to end, for $c=0.6$, $\delta=10^{-3}$, $\sigma_{\rm true}=1300$ fs, noise $3\times10^{-4}$ V, offsets present and fitted, and a free scale (`syst_final_interval.py`): `fit_gaussian_ctr` reports $\hat\sigma=58.5$ fs with CI (50.9, 63.6) fs, `is_identified_95=True` and not multimodal. At 1415 fs the code's profile is 75.49 ($s=0.95$), whereas the exact bounded profile is 65.86 at large $s$. The exact profile is admissible on 54–62 fs *and* 756–2335 fs, which contains the truth. The result is a falsely identified, falsely precise interval that excludes the true duration.

**Refusal threshold.** Exact uniformity ($\mathrm{ptp}\le10^{-9}\max\lvert u\rvert$) is not the right criterion. The trade exists for any `port=None` response mode with a free, prior-less scale. Its cost is the finite penalty $1/u_{\max}^2$, and its reach grows as $\mathrm{ptp}(u)/u_{\max}$ shrinks. Misses persist at $\delta=3\times10^{-2}$ for every $c$ and are absent for the slope mode with $\mathrm{ptp}/u_{\max}=0.6$. Recommended approach, in order of robustness:
1. Require `scale_prior_rms_fraction` (or a fixed scale) whenever a `port=None` response mode is declared.
2. Otherwise refuse when $\mathrm{ptp}(u)/\max\lvert u\rvert$ is below an identifiability tolerance of order 0.1–0.3, or when $u_{\max}\gtrsim0.1$.
3. Whatever is chosen, make `_bilinear_profile` scan $s$ up to the box-edge asymptote rather than stopping on interior argmins, and flag the fit as unidentified when the best $s$ sits at the search's upper edge.

**Docs wording (low).** The page and module say "$\chi^2$ keeps falling toward $s\to\infty$ at $a_k\to-1/u_k$". For one uniform mode alone the asymptote carries the extra penalty $1/u^2$; the fall comes from the *other* pulls becoming cheap. "A uniform one … is refused" should state that nearly uniform modes behave the same. The covariance $C+s^2\sum_kD_kD_k^{\mathsf T}$ is now correct.

**Verdict:** matches for the box derivation, the bounded solve (exact, KKT), the bilinear model, the conditioning, the refusals as implemented, the fixed-scale path, and the fitted-scale path with transverse modes, single-port response modes, clearly non-uniform two-port response modes, or a scale prior. It still differs for a fitted, prior-less scale combined with a nearly uniform two-port response mode: the global $s$ search misses the box-edge branch, producing falsely precise, wrongly identified intervals at $u_{\max}\ge0.3$.

**Recommended status** for the `optical-instrument-observable` systematic-mode extension: `rederived`, scoped in Notes as "fixed scale, scale prior, or free scale without `port=None` response modes; nearly uniform two-port response modes with a free prior-less scale not covered (s search misses the box-edge branch; refusal threshold too narrow)". The existing `anchored` row stays for its earlier scope. Never `signed-off`.

## Temporal response, pulse trains and ADC (2026-11-15)

Fresh-context review of commit `811152027` (#617). Scope: `src/pyrite/detectors/optical_temporal.py` (`DetectorTemporalResponse`, `predict_pulse_train`, `Digitizer`) and `incident_photons_per_pulse` in `src/pyrite/detectors/optical_energy.py`. The derivation below was written from the ledger row, the module docstring and the signatures before the bodies were read. Scratch scripts are in the worktree's ignored `scratch/` directory: `temporal_ode_oracle.py`, `temporal_expm_oracle.py`, `temporal_edges.py` and `temporal_extreme_grid.py`.

### T.0 Independent derivation

**Model.** The channel is linear and time-invariant. For absorbed energies $U_k$ arriving at times $t_k$, the output is $V(t)=G\sum_kU_k\,h(t-t_k)$. Here $G$ is the calibrated isolated-pulse gain in V/J, and the normalized impulse response $h$ is dimensionless, causal, and satisfies $\max_{t\ge0}h=1$. That normalization makes the isolated-pulse peak exactly $GU$, which is the calibration convention of `CalibratedPulseEnergyDetector`. The response is a sum of exponentials, $h(t)=\Theta(t)\sum_ja_je^{-t/\tau_j}$, so

$$
H(\omega)=\int_0^\infty h(t)e^{-i\omega t}\,dt=\sum_j\frac{a_j\tau_j}{1+i\omega\tau_j}\ [\mathrm{s}],\qquad H(0)=\sum_ja_j\tau_j .
$$

**First-order low-pass.** The kernel $k_p(t)=\Theta(t)e^{-t/\tau_p}/\tau_p$ has unit area. For $\tau_j\ne\tau_p$,

$$
\left(e^{-t/\tau_j}*k_p\right)(t)=\frac{1}{\tau_p}e^{-t/\tau_j}\int_0^te^{-u(1/\tau_p-1/\tau_j)}du=\frac{\tau_j}{\tau_j-\tau_p}\left(e^{-t/\tau_j}-e^{-t/\tau_p}\right).
$$

The amplitude on $\tau_j$ is therefore $a_j\tau_j/(\tau_j-\tau_p)$, and the new pole $\tau_p$ carries $\sum_ja_j\tau_j/(\tau_p-\tau_j)$. In the frequency domain this is $H\to H/(1+i\omega\tau_p)$. At equal poles the result becomes $te^{-t/\tau}/\tau^2$, which is not a sum of exponentials, so a degenerate pole must be refused.

**First-order high-pass (AC coupling).** The kernel is $\delta-k_p$, i.e. $H\to H\,i\omega\tau_p/(1+i\omega\tau_p)$. The new amplitudes are $a_j-a_j\tau_j/(\tau_j-\tau_p)=a_j\tau_p/(\tau_p-\tau_j)$ on $\tau_j$, and $\sum_ja_j\tau_j/(\tau_j-\tau_p)$ on $\tau_p$. Then $H(0)=0$ identically, and $h(0^+)$ is unchanged because the $\delta$ passes the jump.

**Voltage-mode pyroelectric (Porter 1981).** I start from the circuit rather than the closed form. The thermal node obeys $\dot T=-T/\tau_T+P/C_{\rm th}$, so an absorbed impulse gives $T\propto e^{-t/\tau_T}$. The pyroelectric current is $i=pA\,\dot T$, which includes a $\delta$ at $t=0$, and it drives a parallel $RC$ with $\tau_E=RC$. Then

$$
V\propto e^{-t/\tau_E}-\frac{1}{\tau_T}\int_0^te^{-u/\tau_T}e^{-(t-u)/\tau_E}du
=\frac{\tau_Te^{-t/\tau_E}-\tau_Ee^{-t/\tau_T}}{\tau_T-\tau_E},
$$

$$
H(\omega)=\frac{\tau_T}{1+i\omega\tau_T}\cdot\frac{i\omega\tau_E}{1+i\omega\tau_E}\ \propto\ \frac{i\omega}{(1+i\omega\tau_T)(1+i\omega\tau_E)} .
$$

This is symmetric in $\tau_T\leftrightarrow\tau_E$. It equals the thermal pole AC-coupled by $\tau_E$, which is the docstring's construction. Further properties: $h(0^+)=1$ and $\dot h(0^+)=-(\tau_T+\tau_E)/(\tau_T\tau_E)<0$, so the peak sits at $t=0$ with value 1. The undershoot has zero net area. As $\tau_E\to\infty$, $h\to e^{-t/\tau_T}-(\tau_T/\tau_E)e^{-t/\tau_E}$ and $\lvert H\rvert\to\tau_T/\lvert1+i\omega\tau_T\rvert$ for $\omega\tau_E\gg1$.

**Trains.** Let the period be $T=1/f$. The steady state has pulses at $-kT$, $k\ge0$, observed at $s\in[0,T)$. The geometric sum gives

$$
v_\infty(s)=\sum_ja_je^{-s/\tau_j}\sum_{k\ge0}e^{-kT/\tau_j}=\sum_j\frac{a_je^{-s/\tau_j}}{1-e^{-T/\tau_j}} .
$$

With pulses at $kT$, $k=0,\dots,n-1$, and $m(t)=\min(\lfloor t/T\rfloor+1,n)$ pulses so far for $t\ge0$ (and $v=0$ for $t<0$), $s=t-(m-1)T$ and

$$
v_n(t)=\sum_ja_je^{-s/\tau_j}\frac{1-e^{-mT/\tau_j}}{1-e^{-T/\tau_j}} .
$$

After the last pulse, $m=n$ and $s>T$, which is a plain decay. The period mean is $\frac1T\int_0^Tv_\infty\,ds=f\sum_ja_j\tau_j=fH(0)$. The impulse comb $\sum_k\delta(t-kT)=f\sum_ne^{i2\pi nft}$ gives $v_\infty(t)=f\sum_nH(2\pi nf)e^{i2\pi nft}$, so the fundamental has peak amplitude $2f\lvert H(2\pi f)\rvert$ and RMS $\sqrt2\,f\lvert H(2\pi f)\rvert$. A lock-in reads the RMS.

**Units.** The average absorbed power is $P=Uf$ [W], and the mean voltage is $GUfH(0)=(G\,H(0))P$. The DC responsivity is therefore $G\int h$ [V/J·s = V/W]. It vanishes for an AC-coupled channel and is not the isolated-pulse gain. The limits are as follows.
- $f\to0$: $v_\infty\to h$, the peak tends to 1 and the mean to 0.
- $f\tau\gg1$ for a single pole: $v_\infty\to f\tau$ with peak-to-peak exactly $h(0^+)=1$.
- AC coupling: zero mean for every $f$.

**ADC.** $N=N_0+gV$ in counts, with $g$ in counts/V (either sign), clipped to $[0,2^b-1]$. When the analog noise dithers the quantizer, a round-to-nearest quantizer has error $\sim U(-\tfrac12,\tfrac12)$, independent of the signal, with variance $1/12$ count². That gives $\sigma_N^2=g^2\sigma_V^2+1/12$, and the unrounded $N$ is the dithered mean. The model holds for $\sigma_{\rm counts}\gtrsim0.5$ (Widrow); a truncating ADC shifts the mean by $-1/2$.

**Photon count.** $N_\gamma=\int(dN/dE)\,dE$ over the collected density, using the same trapezoid rule as the energy integral.

### T.1 Comparison with the implementation

| Quantity | Derivation | Code | Result |
|---|---|---|---|
| Low-pass amplitudes | $a_j\tau_j/(\tau_j-\tau_p)$; $\sum a_j\tau_j/(\tau_p-\tau_j)$ | `_with_pole` `lp_a` | matches |
| High-pass | $\mathrm{append}(a,0)-\text{low-pass}$ | `high_pass` branch | matches |
| Degenerate pole | must refuse | relative $10^{-6}$ refusal | matches |
| Normalization | $\max_{t\ge0}h=1$ | grid $\{0\}\cup$ 4097 log points over $[10^{-4}\tau_{\min},60\tau_{\max}]$ + bounded Brent | matches |
| Pyroelectric | $\tau_T$ pole AC-coupled by $\tau_E$ | `single_pole(thermal).with_ac_coupling(electrical)` | matches |
| $H(\omega)$, $H(0)$ | $\sum a_j\tau_j/(1+i\omega\tau_j)$, $\sum a_j\tau_j$ | `transfer`, `dc_integral_s` | matches |
| Steady train | $a_je^{-s/\tau_j}/(1-e^{-T/\tau_j})$, $s=t\bmod T$ | `train(n_pulses=None)` with `expm1` | matches |
| $n$-pulse train | factor $(1-e^{-mT/\tau_j})$, $m=\mathrm{clip}(\lfloor t/T\rfloor+1,0,n)$; $0$ for $t<0$ | same | matches |
| Mean, fundamental | $fH(0)$, $2f\lvert H(2\pi f)\rvert$ | `steady_state` | matches |
| Extrema | closed $[0,T]$; minimum at $T^-$ for a decaying train | `linspace`+`geomspace`+peak time, refined | matches |
| V, V/W, power | $GU$, $G\,H(0)$, $Uf$; lock-in RMS $=$ peak$/\sqrt2$ | `predict_pulse_train` | matches |
| ADC | $N_0+gV$ clipped; $\sqrt{g^2\sigma_V^2+1/12}$ with $\lvert g\rvert$ | `Digitizer` | matches (validity: T.6) |
| Photons | trapezoid of $dN/dE$ | `photon_count` | matches |

### T.2 Numerical checks (independent oracles)

- **ODE oracle** (`temporal_ode_oracle.py`). The channel is simulated as a circuit with thermal node, pyroelectric current into RC, amplifier low-pass and AC-coupling capacitor, integrated with DOP853 at rtol $10^{-12}$. Pulses are injected as state jumps, with no exponential-sum algebra. Five channels were run: pyro (fast and slow $\tau_E$), pyro+LP, bolometer+LP+AC and bolometer+LP. Results:
  - normalized impulse response: worst $\lvert\Delta h\rvert=2.0\times10^{-9}$ (ODE tolerance);
  - $\int h$ against the sampled trapezoid: within $3\times10^{-15}\sum\tau$ for AC channels; the bolometer+LP value $4.934819\times10^{-3}$ s agrees to $2\times10^{-9}$ relative;
  - $n=4$ trains from $-T/2$ to $6.5T$, i.e. before $t=0$, across all pulses and after the last: agree to $\le2\times10^{-9}$ at every rate. The only exceptions are samples lying exactly on a pulse instant, where the oracle's half-open interval and the code's `floor` assign the floating-point boundary differently. That is not a physics difference.
- **Exact state-transition oracle** (`temporal_expm_oracle.py`). The periodic orbit is solved as $x=(I-\Phi)^{-1}\Phi b$ with $\Phi=e^{AT}$. Mean and fundamental come from Van Loan block exponentials $\int_0^Te^{(A-i\omega I)s}ds$. Six channels were swept over 13 rates spanning $f\tau_{\max}=10^{-3}$ to $f\tau_{\min}=10^{3}$. Results:
  - steady waveform, mean and fundamental agree to $\le10^{-11}$ of the isolated peak for $f\tau_{\max}\le10^4$, with the fundamental generally at $10^{-15}$;
  - at $f\tau\sim10^5$ the residual grows to $1.2\times10^{-7}$. That is the oracle's ill-conditioned $(I-\Phi)^{-1}$: its mean deviates from the exact AC value 0 by the same amount, while the code's mean is $f\sum a_j\tau_j=O(10^{-11})$;
  - code extrema are never inside the oracle's sampled range (signed check). Where they differ, the code's refined extremum is the more extreme value.
- **Single-pole closed form**, $f=10^{-3}$–$10^{8}$ Hz with $\tau=2$ ms: peak $1/(1-r)$ and minimum $r/(1-r)$ with $r=e^{-T/\tau}$, relative error $\le2\times10^{-12}$; mean $f\tau$ exact; fundamental $2f\tau/\sqrt{1+(2\pi f\tau)^2}$ to $2\times10^{-16}$; peak-to-peak $=1$ to 12 digits at every rate. This covers the $f\to0$ and $f\tau\gg1$ limits.
- **$\tau_E\to\infty$**: $\lvert H_{\rm pyro}\rvert/\lvert H_{\rm single}\rvert-1=-5\times10^{-6},-5\times10^{-10},-5\times10^{-14}$ for $\tau_E/\tau_T=10^2,10^4,10^6$ at 50 Hz, i.e. $\propto(2\pi f\tau_E)^{-2}$ as expected. $\max\lvert h-h_{\rm single}\rvert\approx\tau_T/\tau_E$ is as derived. $\int h=0$ throughout.
- **AC zero mean**: every AC channel has $\lvert fH(0)\rvert\le4\times10^{-11}$ of the isolated peak up to 0.5 MHz. `predict_pulse_train` reports `mean_v` $=$ `dc_responsivity_v_per_w` × `average_absorbed_power_w` to $2\times10^{-37}$ V. Bolometer: V/W $=G\tau$ exactly.
- **$n$-pulse edges** (`temporal_edges.py` §3) against brute superposition of shifted `impulse`:
  - $\le1.6\times10^{-13}$ for $n=1,3,7$;
  - exactly 0 for $t<0$;
  - $n=400$ late equals the steady state to 0.0;
  - the steady state is periodic for $t<0$.
- **Closed interval**: the minimum equals $v(T^-)$ to $7\times10^{-16}$. For a low-passed train peaking mid-period, the code's peak and minimum match a $2\times10^5$-point dense scan to $10^{-12}$.
- **Near-degenerate poles**: with $\tau_p=\tau(1+\epsilon)$ and $\epsilon=1.5\times10^{-6}$ the amplitudes reach $1.8\times10^6$. $h$ still matches a two-pole `expm` oracle to $3\times10^{-10}$, and after AC coupling $\lvert\int h\rvert/\sum\lvert a\rvert\tau=2\times10^{-17}$, far inside the $10^{-9}$ `is_ac_coupled` tolerance.
- **Extreme separation**: for $\tau_T=1$ s, $\tau_p$ down to $10^{-12}$ s and $f$ down to 0.01 Hz (`temporal_extreme_grid.py`), the peak and minimum match an 800k-point log scan to $\le4\times10^{-14}$.
- **Calibrated gain**: `isolated_peak_v` $=GU$ exactly. As $f\to0$, `steady_peak_v` $\to GU$; at $10^{-3}$ Hz it equals $GU$ to print precision.
- **ADC**: Monte Carlo of a round-to-nearest quantizer gives the following (offset-averaged rms vs. model).

| $\sigma$ (counts) | 0.05 | 0.2 | 0.3 | 0.5 | 1 | 3 |
|---|---|---|---|---|---|---|
| model rms | 0.293 | 0.351 | 0.416 | 0.577 | 1.041 | 3.014 |
| MC rms | 0.167 | 0.337 | 0.414 | 0.577 | 1.038 | 3.011 |
| MC fixed-point mean bias | 0.20 | 0.13 | 0.05 | 0.002 | 0.000 | 0.003 |

  Negative `counts_per_v` maps and clips correctly, the noise uses $\lvert g\rvert$, and the clip mask is right.
- **Photons**: a Gaussian density on a 3002-point random nonuniform grid gives $-2.9\times10^{-5}$ relative error, matching the trapezoid truncation of the energy integral. A flat 3/eV density over $[1,5]$ eV gives exactly 12. The batch is linear.
- The repository's `tests/detectors/test_optical_temporal.py` and `test_optical_energy.py` pass (25).

### T.3 Findings (no code change made)

- **T.1 (low): `steady_state` and `predict_pulse_train` do not validate `repetition_hz`** (`optical_temporal.py:254`, `:318`). `train` rejects non-positive or non-finite rates, but these do not:
  - `steady_state(-50)` silently returns a negative mean ($-0.05$) and a *negative* fundamental "amplitude" ($-0.095$);
  - `nan` propagates silently;
  - `0` raises `ZeroDivisionError`, and `inf` raises a NumPy `geomspace` error.

  `predict_pulse_train` passes these through, so a negative rate produces a reading with negative average power. Suggested fix: apply the same check as `train`.
- **T.2 (low): non-integer `n_pulses` is truncated silently** (`:227`, `:239`). `n_pulses=2.7` behaves as 2, while 0.5 is rejected. Require an integral value.
- **T.3 (low, convention): renormalization after shaping.** `with_low_pass`/`with_ac_coupling` renormalize to unit peak (`:150`), so $G$ must have been calibrated *with* that shaping in place. Adding a shaping stage to a channel calibrated without it overstates every voltage by the lost peak factor, for example $1/0.850=+18\%$ for pyro($50$ ms, $2$ ms) + LP($100\,\mu$s). The pre-normalization peak is not exposed. The docstrings say "renormalized", but the module docstring should state the calibration consequence explicitly.
- **T.4 (info): saturation is checked on the positive peak only** (`:332`). AC-coupled trains swing to $\approx-\tfrac12$ of the isolated peak at high rate, and negative excursions are not checked. Negative-polarity channels are refused ("no positive peak"), which is consistent, but inverting front ends must be modelled with a sign outside.
- **T.5 (info): `peak_time_s` is indeterminate on a flat maximum.** At $\tau/\tau_p=10^{13}$ it is off by $9\times10^{-4}$ relative, although the peak *value* is exact to $10^{-15}$. Only the value is used downstream.
- **T.6 (info): ADC model validity** (`:364`–`:375`).
  - The $1/12$ term and the unrounded expected count assume a dithered round-to-nearest quantizer. At $\sigma\ge0.5$ count they hold to $<0.3\%$. At $0.3$ they are 0.5 % on average but biased by up to 0.13 count at fixed sub-LSB position. Below $0.2$ count they fail: rms is overstated and the mean is biased by up to $\pm0.5$. The module's "at least about one count" condition is adequate but not enforced or flagged.
  - The rounding convention (nearest vs. truncation, a $-1/2$ offset) is not stated.
  - Near the rails, `noise_rms_counts` does not shrink and $\mathrm{clip}(\langle N\rangle)\neq\langle\mathrm{clip}(N)\rangle$. Only the mask signals this.
- **T.7 (info): `is_ac_coupled` tolerance.** The relative $10^{-9}$ threshold sits $10^{8}$ above the observed roundoff ($\le8\times10^{-17}$), so true AC channels are always flagged. A DC-coupled response whose net area is $\le5\times10^{-10}$ of its gross area is also called AC-coupled, which is physically harmless since V/W $\approx0$ either way. The property only labels the response; V/W uses the actual $H(0)$.

### T.4 Verdict

- **Claim**: `optical-instrument-observable` (temporal extension) — `src/pyrite/detectors/optical_temporal.py::{DetectorTemporalResponse, predict_pulse_train, Digitizer}`, `optical_energy.py` `incident_photons_per_pulse` — LTI sum-of-exponentials response; first-order low-pass/high-pass composition; Porter (Ferroelectrics 33, 193 (1981)) voltage-mode pyroelectric shape; geometric-sum trains; Fourier-series mean $fH(0)$ and fundamental $2f\lvert H(2\pi f)\rvert$; linear clipped ADC with $1/12$ quantization.
- **Filters**: units pass (s, V/J, V/W, W, counts); limits pass ($f\to0$, $f\tau\gg1$, $\tau_E\to\infty$, AC zero mean, closed-interval extrema, $n\to\infty$); signs/conventions pass ($e^{-i\omega t}$ transfer, $\lvert g\rvert$ in noise, causal $t<0$).
- **Re-derivation**: `matches`. Every amplitude, geometric factor, mean, fundamental and unit agrees symbolically. Numerically, the two independent oracles agree to $\le2\times10^{-9}$ (ODE) and to roundoff (state-transition) over $10^{-3}\le f\tau\le10^{5}$.
- **Verdict**: `rederived` for this extension scope. T.1–T.2 are input-validation gaps outside the physics; T.3–T.7 are conventions and validity notes.
- **Write-up**: `docs/validation/detectors/optical-instrument-observable.md` (this section).
- **Suggested ledger change**: in the Notes field, add "Temporal/train/ADC extension (2026-11-15, `811152027`): `rederived` (fresh-context, ODE + exact state-transition oracles; T.1–T.7). Rate validation in `steady_state`/`predict_pulse_train` and integral `n_pulses` pending (T.1–T.2); $G$ must be calibrated with any added shaping (T.3); ADC model assumes dithered round-to-nearest, $\sigma\gtrsim0.5$ count." The existing `anchored` status stays for its earlier scope; the extension has no external anchor (synthetic calibration only). Never `signed-off`.

- **Applied (task owner, 2026-11-15):** T.1 rate validation in `steady_state`/`predict_pulse_train`; T.2 non-integral `n_pulses` refused; T.4 saturation now checks the larger of the positive peak and AC negative swing; T.3 and T.6 conventions stated in the docstrings and user doc. Tests in `test_optical_temporal.py`.

## Detector-class calibration schema (fresh-context review, 2026-11-15, `6b1bb88b3`)

Scope: `src/pyrite/detectors/optical_detector_schema.py` (`pulse_gain_from_power_responsivity`,
`OpticalDetectorCalibration`, NEP noise). The channel model
$V(t)=G\sum_k U_k\,h(t-t_k)$ with peak-normalized $h$ (`optical_temporal.py`) is taken as given.
Derivation written before reading the implementation bodies.

### S2-D1. Response to sinusoidally modulated absorbed power

For continuous absorbed power $P(t)$ the channel is $V=G\,(h*P)$, with $G$ in V/J, $h$
dimensionless and $H(\omega)=\int_0^\infty h(t)\,e^{-i\omega t}\,dt$ in s. For
$P(t)=P_0(1+\cos\omega t)$,

$$
V(t)=G P_0\left[H(0)+\operatorname{Re}\!\left(H(\omega)\,e^{i\omega t}\right)\right],
$$

so the voltage fundamental has amplitude $G P_0\lvert H(\omega)\rvert$ and the power
fundamental has amplitude $P_0$. The sinusoidal responsivity is

$$
R(f_m)=G\,\lvert H(2\pi f_m)\rvert
\quad\Rightarrow\quad
\boxed{G=R(f_m)/\lvert H(2\pi f_m)\rvert},
\qquad R(0)=G\,H(0)=G\!\int h\,dt .
$$

The ratio is the same for peak/peak and RMS/RMS of the two fundamentals. When
$H(0)=0$ (AC coupling) a DC responsivity does not exist. Units: (V/W)/s = V/J.

### S2-D2. Per-class closed forms

- **Bolometer, Golay cell (single thermal pole):** $h=e^{-t/\tau}$, $H=\tau/(1+i\omega\tau)$,
  so $G=R\sqrt{1+\omega_m^2\tau^2}/\tau$. Physical check: an energy $U$ absorbed in heat capacity
  $C$ gives $\Delta T=U/C$ and $V_{\rm peak}=S_T U/C=R(0)\,G_{\rm th}U/C=R(0)U/\tau$. This matches.
- **Pyroelectric (voltage mode):** $H_{\rm un}\propto i\omega\tau_T\tau_E/[(1+i\omega\tau_T)(1+i\omega\tau_E)]$
  (Porter shape), with $h_{\rm un}=(\tau_T e^{-t/\tau_E}-\tau_E e^{-t/\tau_T})/(\tau_T-\tau_E)$ and
  $\int h_{\rm un}\,dt=0$. Vendor energy responsivity (V/J, the peak voltage per pulse energy) maps
  directly to $G$.
- **Shaping low-pass:** $K_s=1/(1+i\omega\tau_s)$. With $h$ renormalized after shaping, $G$ is the
  peak at the *shaped* output. If $R$ refers to the detector head (before shaping), the code's
  $G$ is high by $1/\lvert K_s(\omega_m)\rvert=\sqrt{1+\omega_m^2\tau_s^2}$.

### S2-D3. Chopped (square-wave) calibrations

For a 50 % chop from $0$ to $P_{\rm on}$, the power fundamental has peak amplitude
$2P_{\rm on}/\pi$. A datasheet "responsivity" equals the sinusoidal $R$ multiplied by:

| Datasheet convention | Factor relative to $R$ |
|---|---|
| fundamental peak / $P_{\rm on}$ | $2/\pi=0.637$ |
| lock-in RMS / $P_{\rm on}$ | $\sqrt2/\pi=0.450$ |
| fundamental peak / ($P_{\rm on}/2$) | $4/\pi=1.273$ |
| output peak-to-peak / $P_{\rm on}$, $f_m\tau\ll1$ | $\to R(0)/R(f_m)$ |

The code and docs state the sinusoidal-fundamental convention explicitly. A
chopped datasheet value entered unconverted biases $G$ by the inverse factor.

### S2-D4. NEP and output noise

$\mathrm{NEP}(f_m)=v_n(f_m)/R(f_m)$ in W/$\sqrt{\rm Hz}$. The output RMS noise
$\sigma_V=R(f_m)\,\mathrm{NEP}\,\sqrt{\Delta f}$ requires two things:

- the output-referred noise density is white at $v_n(f_m)$ over the declared band $\Delta f$;
- NEP is quoted at the same $f_m$ as $R$.

Thermal-fluctuation (phonon) noise is filtered by the thermal pole, giving
$\sigma_V^2=\mathrm{NEP}^2G^2\int_0^{\Delta f}\lvert H\rvert^2df$. For a single pole with a
brick-wall band this gives an effective band of $\arctan(2\pi\Delta f\tau)/(2\pi\tau)$.

### Comparison with the implementation and independent numerics

Script: `scratch/schema_check.py` (closed forms plus an ODE time-domain chain, not implementation helpers).

| Check | Result |
|---|---|
| bolometer $G$ at $f_m=0,10,40,400$ Hz vs $R\sqrt{1+\omega^2\tau^2}/\tau$ | rel $\le1.1\times10^{-16}$ |
| NEP noise $R\,\mathrm{NEP}\sqrt{\Delta f}$ | exact |
| Golay + shaping $G$ vs closed-form two-pole peak $t^*=\frac{\tau\tau_s}{\tau-\tau_s}\ln\frac{\tau}{\tau_s}$ | rel $2\times10^{-16}$ |
| ODE chain driven by $P_0(1+\cos\omega t)$, lock-in fundamental / $P_0$ | $R$ to $2.5\times10^{-11}$ |
| ODE chain, square chop: three conventions | $2/\pi$, $\sqrt2/\pi$, $4/\pi$ to 6 digits |
| pyroelectric $G$ at 5, 20, 160 Hz vs Porter closed form | rel $\le4.4\times10^{-16}$ |
| `tests/detectors/test_optical_detector_schema.py` | 5 passed |

Symbolic and dimensional agreement: `pulse_gain_from_power_responsivity` returns $R/\lvert H(2\pi f_m)\rvert$
(L127). DC is refused when $\lvert H(0)\rvert\le10^{-9}\sum\lvert a_j\rvert\tau_j$, and any $f_m$
where $\lvert H\rvert\le10^{-12}\sum\lvert a_j\rvert\tau_j$ is also refused (L121–126). Noise is
$R\,\mathrm{NEP}\sqrt{\Delta f}$ (L230–231). The class temporal models match S2-D2 (L210–221).

### Findings

- **S2.1 (medium, convention; L10–12, physics doc L179–180).** $R$ and `pulse_gain_v_per_j`
  are per **absorbed** power or energy, and `build()` multiplies by the absorptance table. Golay and
  pyroelectric datasheet responsivities are normally per **incident** power or energy at a stated
  wavelength or source, window included. Entering such a value with a non-unit absorptance table
  double-counts absorption by $A(\lambda_{\rm cal})$. The record has no calibration
  wavelength/source field and gives no warning. Bolometer *electrical* (load-curve) responsivity is
  per absorbed power and is consistent. Suggest documenting $R_{\rm abs}=R_{\rm inc}/A(\lambda_{\rm cal})$,
  or recording the calibration source.
- **S2.2 (medium, documentation; L10–15, `pulse_gain_from_power_responsivity` docstring L108–109).**
  The sinusoidal convention is stated, but the schema does not warn that chopped datasheet values
  (e.g. Golay cells specified at a 20 Hz chop) need conversion. Unconverted, $G$ is silently biased
  by $\pi/2$, $\pi/\sqrt2$ or $\pi/4$ (S2-D3). Suggest a conversion note. One example: a 50 % chop
  read as lock-in RMS gives $R=\pi V_{\rm rms}/(\sqrt2 P_{\rm on})$.
- **S2.3 (low, assumption; L17–18, L230–231).** $\sigma_V=R(f_m)\,\mathrm{NEP}\sqrt{\Delta f}$
  assumes white output noise over the declared $\Delta f$, with NEP quoted at the responsivity's
  $f_m$. There is no separate NEP-frequency field. For phonon-noise-limited thermal detectors the
  result is conservative: 1.45× for $\tau=4$ ms with a 100 Hz brick-wall band (effective band
  47.4 Hz; the pole's ENBW is $1/(4\tau)=62.5$ Hz). $\Delta f$ is not tied to the shaping filter's
  ENBW $1/(4\tau_s)$. Suggest stating both assumptions.
- **S2.4 (low, reference plane; L219–220, L229).** Shaping is included in $H$ before
  $G=R/\lvert H\rvert$, and pulse-gain classes keep $G$ as the shaped-output peak. Both are correct
  only if every calibration refers to the final shaped output. A detector-head calibration combined
  with `shaping_time_constant_s` biases $G$ by $\sqrt{1+\omega_m^2\tau_s^2}$: 1.00003 in the test,
  1.012 at $\tau_s=2$ ms, 1.074 at $\tau_s=5$ ms (with $f_m=12.5$ Hz). For pulse-gain classes the
  bias is the shaped/unshaped peak ratio. `CalibratedPulseEnergyDetector` says "under its stated
  electronics/shaping conditions", but the schema docs do not say this. Suggest a one-line
  reference-plane statement.
- **S2.5 (info; L187–192).** The energy/absorptance table is validated only in `build()`. A record
  with a decreasing energy grid or absorptance $>1$ is constructed and hashed (probe: constructed,
  then refused at build). Suggest validating at construction.
- **S2.6 (info; L66).** `operating_temperature_k` is required but used only as hashed provenance.
  This is acceptable; say so in the docs.
- **S2.7 (info).** A `shaping_time_constant_s` equal to the detector $\tau$ is accepted by the
  record and refused only at `build()` (degenerate double pole from `optical_temporal`). Harmless.
- **S2.8 (info).** Single-pole Golay and bolometer models are approximations: Golay pneumatic and
  readout dynamics are simplified, and a bolometer $\tau$ must be the electrothermal-feedback
  effective value. The docs label these "single pole", which is adequate.

### Verifier output

- **Claim**: `optical-instrument-observable` (schema extension) —
  `src/pyrite/detectors/optical_detector_schema.py::{pulse_gain_from_power_responsivity, OpticalDetectorCalibration}`
  — LTI response to $P_0(1+\cos\omega t)$; NEP definition; Porter pyroelectric shape.
- **Filters**: units pass; limits pass ($f_m\to0$ gives $G H(0)$; AC-coupled DC refused;
  $\omega\tau\gg1$ gives $G\to R\omega$); signs/conventions pass with documentation findings
  S2.1–S2.4.
- **Re-derivation**: `matches`. $G=R(f_m)/\lvert H(2\pi f_m)\rvert$,
  $\sigma_V=R\,\mathrm{NEP}\sqrt{\Delta f}$ and the class impulse responses agree to machine
  precision. A time-domain ODE lock-in reproduces $R$ to $2.5\times10^{-11}$.
- **Verdict**: `rederived` for this scope. No math discrepancy. S2.1 and S2.2 are convention or
  documentation risks that can silently bias $G$ by $A(\lambda_{\rm cal})$ or 0.45–1.27× when
  datasheet values are entered unconverted.
- **Suggested ledger change**: append to Notes: "Detector-class schema `rederived` (fresh context
  2026-11-15 at `6b1bb88b3`): sinusoidal-fundamental $G=R/\lvert H\rvert$, NEP noise and class
  models match an ODE lock-in to 2.5e-11. Responsivities are per absorbed power at the shaped
  output; chopped or incident-power datasheet values need conversion (S2.1–S2.4)." Never `signed-off`.

- **Applied (task owner, 2026-11-15):** S2.1, S2.2 and S2.4 reference-plane conventions (absorbed power, sinusoidal fundamental with chopped conversion, shaped output) and the S2.3 noise assumption stated in the module docstring and user doc; S2.5/S2.7 records are validated eagerly at construction (absorptance table and shaping time constant).
