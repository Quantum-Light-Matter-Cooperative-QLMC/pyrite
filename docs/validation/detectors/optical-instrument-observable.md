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
