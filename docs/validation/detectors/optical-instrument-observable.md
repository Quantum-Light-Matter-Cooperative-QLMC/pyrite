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
