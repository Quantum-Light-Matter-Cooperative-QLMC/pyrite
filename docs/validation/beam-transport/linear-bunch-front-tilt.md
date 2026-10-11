# Linear bunch-front tilt

**Validation:** `linear-bunch-front-tilt`

**Independent verification:** fresh context, 2026-10-11. The derivation below
was recorded before inspection of implementation bodies. Source-to-code
comparison followed the owner's readiness signal; this is not human sign-off.

## Claim, sources, and assumptions

The claim is a linear arrival-time shear of an initially Gaussian electron
bunch, with unchanged mean velocity. The public descriptor is
`BunchFrontTilt(tilt_deg, azimuth_deg=0)`. The proposed delay helper is
`bunch_front_delay_ang(record, face_positions)`, returning time as a length
in angstroms with $c=1$.

The primary radiation source is Artru and Ispirian,
[Bunch coherence in parametric X-ray radiation](https://arxiv.org/pdf/physics/0208018),
Eqs. (11), (14), and (15): entry-event phase, finite-bunch intensity, and
normalized amplitude form factor. The imposed shear itself follows from
straight-ray plane kinematics and the Gaussian characteristic function;
it is not a fit to a published spectrum.

Tadenuma et al.,
[Generation of coherent THz Cherenkov radiation by electron bunch tilt control](https://doi.org/10.1103/PhysRevAccelBeams.25.110102),
provides experimental motivation for controlling the bunch-front orientation.
Its Eq. (1) distinguishes the single-electron and pair contributions. Its
rotated spatial-bunch parameterization and tilt angle are not substituted
for the arrival-time shear convention defined here.

Let the mean beam direction be $\hat{\mathbf e}_z$, its speed be
$v_0=\beta_0c$, and the incoming beam-normal reference plane pass through
the target origin. The independent incoming transverse offset is
$\mathbf w=(x,y)^{\mathsf T}$, distributed with zero mean and covariance
$\Sigma$. Let $\varepsilon$ be an independent zero-mean Gaussian residual
arrival time expressed as a length, with RMS
$\sigma_\varepsilon=c\sigma_t$. The analytic form factor assumes parallel
rays at the common mean velocity, a planar entrance face, and fixed
single-electron amplitude apart from the bunch phase. It does not describe
correlations between energy, divergence, or residual time and position.

## Independent derivation

Write the degree inputs as $\theta=\pi\,\mathrm{tilt\_deg}/180$ and
$\phi=\pi\,\mathrm{azimuth\_deg}/180$, with
$\mathbf u=(\cos\phi,\sin\phi)^{\mathsf T}$. Define

$$
\boxed{\mathbf a=\frac{\tan\theta}{\beta_0}\mathbf u,
\qquad \tau_{\rm imposed}=\mathbf a^{\mathsf T}\mathbf w.}
$$

Here $\tau=ct$. At a fixed laboratory time $\tau$, a zero-residual-time
particle has longitudinal coordinate
$z=\beta_0(\tau-\mathbf a^{\mathsf T}\mathbf w)$. The equal-arrival front
therefore obeys

$$
z+\tan\theta\,\mathbf u^{\mathsf T}\mathbf w=\beta_0\tau.
$$

Its normal is proportional to
$\hat{\mathbf e}_z+\tan\theta\,\mathbf u$. This proves both the tangent
and the division by $\beta_0$. Positive $\theta$ produces later arrival
at positive $\mathbf u^{\mathsf T}\mathbf w$; those particles lag in $z$
at a common time. The shear changes the arrival time, not the ray direction.

Let $\mathbf m=(m_x,m_y,m_z)$ be any normal to the entrance face through
the origin. For $m_z\ne0$, define

$$
\mathbf d=-\frac{(m_x,m_y)^{\mathsf T}}{m_z},\qquad
A=\begin{pmatrix}1&0\\0&1\\d_x&d_y\end{pmatrix},\qquad
\mathbf b=\frac{\mathbf d}{\beta_0}.
$$

The ray intersects the face at $\mathbf r_f=A\mathbf w$, with signed
flight-time increment $\tau_{\rm face}=\mathbf b^{\mathsf T}\mathbf w$.
Thus the complete face-arrival time is

$$
\boxed{\tau_f=\varepsilon+(\mathbf a+\mathbf b)^{\mathsf T}\mathbf w.}
$$

The flight increment is signed because the incoming reference plane and
face intersect at the origin. It is added exactly once. Defining tilt
directly from the face's longitudinal coordinate would mix the imposed
shear with this geometrical term.

### Front parallel to the entrance face

The target's signed polar tilt $\alpha$ and azimuth $\psi$ define its
laboratory normal as
$\mathbf m=(\sin\alpha\cos\psi,\sin\alpha\sin\psi,\cos\alpha)$.
For matching front and target angles, $\theta=\alpha$ and $\phi=\psi$,

$$
\mathbf d=-\tan\alpha\,(\cos\psi,\sin\psi)^{\mathsf T},\qquad
\mathbf a+\mathbf b=0,\qquad \boxed{\tau_f=\varepsilon.}
$$

Thus the bunch front is parallel to the entrance face and all particles
with the same residual time reach that face simultaneously. The front
normal is $\mathbf m/\cos\alpha$, proving parallelism for every azimuth
and signed $-90^\circ<\alpha<90^\circ$. This is exact even at grazing
incidence; it is not a small-angle statement. The mean beam direction
remains fixed. This geometrical alignment does not automatically cancel
the photon/crystal phase: its remaining transverse coefficient is
$-A^{\mathsf T}\mathbf q$.
For a public scene this contract uses the resolved physical target angles:
the existing scene-to-case conversion rounds target tilt and azimuth to
the nearest half degree. The bunch-front descriptor retains full precision.

For the specified phase convention, define
$\mathbf q=\omega\hat{\mathbf n}+\mathbf g$, where
$\hat{\mathbf n}$ is the emitted-photon direction, $\mathbf g$ is the
reciprocal vector in the same laboratory frame, and $\omega$ has units
of inverse angstroms. Then

$$
\Phi=\omega\tau_f-\mathbf q^{\mathsf T}\mathbf r_f
=\omega\varepsilon+\boldsymbol\kappa^{\mathsf T}\mathbf w,
\qquad
\boxed{\boldsymbol\kappa=\omega(\mathbf a+\mathbf b)-A^{\mathsf T}\mathbf q.}
$$

In particular, the transverse photon/crystal term and imposed delay
enter with opposite signs. Replacing the radiation convention by its
complex conjugate negates the entire phase and leaves intensity unchanged;
negating only the imposed delay changes the physical tilt direction.
Artru's Eq. (11) uses $\omega t+\mathbf q_g^{\mathsf T}\mathbf r$,
with $\mathbf q_g=-\mathbf k+\mathbf g_{\rm source}$. The prescribed
convention here therefore labels the reciprocal vector as
$\mathbf g=-\mathbf g_{\rm source}$ when $\mathbf k=\omega\hat{\mathbf n}$.
This is a convention mapping, not a change to the source's phase equation.
There is a separate printed sign inconsistency in the source's v1
Eq. (13): its preceding definitions
$\mathbf r=\mathbf v t+\mathbf s$ and
$t_l=-\mathbf m^{\mathsf T}\mathbf s/(\mathbf m^{\mathsf T}\mathbf v)$
give $\mathbf K=\mathbf q_g-h_g\mathbf m/(\mathbf m^{\mathsf T}\mathbf v)$
from Eq. (11), whereas Eq. (13) prints a plus. The present derivation uses
the entry-event phase and ray-plane solution directly; it does not import
that printed surface correction.

The characteristic function of independent centered Gaussians gives

$$
\boxed{F_{\rm amp}=\left\langle e^{i\Phi}\right\rangle
=\exp\left[-\frac12\left(\omega^2\sigma_\varepsilon^2
+\boldsymbol\kappa^{\mathsf T}\Sigma\boldsymbol\kappa\right)\right],
\qquad
F_{\rm intensity}=\lvert F_{\rm amp}\rvert^2
=\exp\left[-\omega^2\sigma_\varepsilon^2
-\boldsymbol\kappa^{\mathsf T}\Sigma\boldsymbol\kappa\right].}
$$

The covariance includes the cross term
$2\kappa_x\kappa_y\Sigma_{xy}$. Rotated elliptical spots must rotate
their covariance into the same incoming laboratory transverse basis.
The factor $1/2$ belongs to the amplitude exponent. A Gaussian intensity
form factor must not be multiplied into the amplitude and squared again.

## Phase matching and bounds

For positive-definite $\Sigma$, complete transverse phase matching is
$\boldsymbol\kappa=0$, equivalently

$$
\mathbf a_{\rm match}=\frac{A^{\mathsf T}\mathbf q}{\omega}-\mathbf b.
$$

If $\omega>0$ and this vector is finite, one representation is
$\phi=\operatorname{atan2}(a_{{\rm match},y},a_{{\rm match},x})$ and
$\theta=\arctan(\beta_0\|\mathbf a_{\rm match}\|)$.
For a singular covariance only the populated covariance directions need
matching: $\Sigma^{1/2}\boldsymbol\kappa=0$.

$$
0\le F_{\rm intensity}\le
\exp(-\omega^2\sigma_\varepsilon^2)\le1.
$$

Tilt can cancel the transverse phase gradient but cannot cancel the
independent residual-time damping. This model supplies no universal finite
coherence window in angle: its width depends on $\omega$, $\beta_0$,
$\Sigma$, face geometry, and the selected reflection/observation direction.

The descriptor's nonsingular geometric domain is finite
$-90^\circ<\mathrm{tilt\_deg}<90^\circ$ and finite azimuth, which is
periodic modulo $360^\circ$. The speed must satisfy $0<\beta_0\le1$.
Face projection additionally requires a nonzero normal-direction dot
product. No stronger numerical bound follows from the derivation.

## Temporal self term

For independent residual times of distinct electrons, the amplitude
damping is $f_t=\exp(-\omega^2\sigma_\varepsilon^2/2)$ and the pair
damping is $f_t^2$. For fixed complex individual contributions $C_j$
already containing their deterministic imposed and flight phases,

$$
\boxed{\left\langle\left|\sum_j C_j e^{i\omega\varepsilon_j}\right|^2\right\rangle
=f_t^2\left|\sum_j C_j\right|^2
+(1-f_t^2)\sum_j|C_j|^2.}
$$

The same-particle term is undamped; independently averaging the residual
time cannot remove a particle's own intensity. For $N$ identical iid
electrons, this yields the standard
$I_N=I_1[N+N(N-1)F_{\rm intensity}]$. A Monte Carlo estimator of the
continuum pair contribution may instead subtract its sampling diagonal;
that bookkeeping term must be distinguished from the physical temporal
self term above.

## Conditional and marginal duration

At a fixed incoming transverse offset the time RMS remains $\sigma_t$.
The incoming-plane marginal RMS and face-arrival marginal RMS are

$$
\sigma_{t,{\rm incoming}}^2
=\sigma_t^2+\frac{\mathbf a^{\mathsf T}\Sigma\mathbf a}{c^2},
\qquad
\sigma_{t,{\rm face}}^2
=\sigma_t^2+
\frac{(\mathbf a+\mathbf b)^{\mathsf T}\Sigma(\mathbf a+\mathbf b)}{c^2}.
$$

Using the marginal RMS as an independent temporal factor while retaining
the spatial shear phase counts tilt-induced variance twice and loses the
time-position covariance needed for phase matching.

## Filters and limiting cases

- **Units:** $\mathbf a$ and $\mathbf b$ are dimensionless; delay,
  $\mathbf r_f$, and $\sigma_\varepsilon$ are lengths; $\boldsymbol\kappa$
  is inverse length. Every phase and exponent is dimensionless.
- **Zero tilt:** $\mathbf a=0$ and
  $\boldsymbol\kappa=\omega\mathbf b-A^{\mathsf T}\mathbf q$, retaining
  the entrance-face flight delay.
- **Beam-normal face:** $\mathbf d=\mathbf b=0$, giving
  $\boldsymbol\kappa=\omega\mathbf a-\mathbf q_T$.
- **Point spot:** $\Sigma=0$ removes every tilt-dependent factor.
- **Zero residual duration:** $\sigma_t=0$ removes independent temporal
  damping but retains transverse mismatch.
- **Azimuth/sign symmetry:** $(\theta,\phi+\pi)$ and $(-\theta,\phi)$
  produce identical $\mathbf a$. Negative angle alone reverses the shear.
- **Nonrelativistic limit:** at fixed tilt and offset, delay scales as
  $1/\beta_0$, consistent with traversing a fixed longitudinal separation.

## Implementation comparison

An independent numeric reference uses $\beta_0=0.8$,
$\theta=\arctan(0.24)=13.495733280795813^\circ$, $\phi=0$,
$\mathbf d=(0.2,-0.1)^{\mathsf T}$, $\omega=0.02\,\mathrm{\AA}^{-1}$,
$\mathbf q=(0.01,-0.007,0.004)^{\mathsf T}\,\mathrm{\AA}^{-1}$,
$\sigma_\varepsilon=3\,\mathrm{\AA}$, and
$\Sigma=\left(\begin{smallmatrix}100&30\\30&400\end{smallmatrix}\right)
\mathrm{\AA}^2$. For $\mathbf w=(10,-5)^{\mathsf T}\,\mathrm{\AA}$,
the imposed delay is $3\,\mathrm{\AA}$, face position is
$(10,-5,2.5)^{\mathsf T}\,\mathrm{\AA}$, and signed face flight delay is
$3.125\,\mathrm{\AA}$. The phase gradient is
$\boldsymbol\kappa=(0.0002,0.0049)^{\mathsf T}\,\mathrm{\AA}^{-1}$.
Its total phase variance is $0.0132668$, giving amplitude
$0.9933885524312075$ and intensity $0.98682081610137$.

### Coordinate mapping and exact comparison

The implementation stores positions in the sample frame, where the
entrance face is the plane $p_z=0$. Let $\hat{\mathbf b}$ be the mean
beam direction in that frame and $J$ map the incoming laboratory
transverse offset to the face coordinates: $\mathbf p_{xy}=J\mathbf w$.
Since the incoming offset is orthogonal to $\hat{\mathbf b}$, its signed
flight distance is $s=\hat{\mathbf b}_{xy}^{\mathsf T}\mathbf p_{xy}$.
Consequently the independent equations become

$$
\tau_{\rm imposed}=(J^{-\mathsf T}\mathbf a)^{\mathsf T}\mathbf p_{xy},
\quad
\Sigma_f=J\Sigma J^{\mathsf T},\quad
\mathbf K_f=\omega\left(\hat{\mathbf n}_{xy}
-\frac{\hat{\mathbf b}_{xy}}{\beta_0}-J^{-\mathsf T}\mathbf a\right)
+\mathbf g_{xy}.
$$

The implementation's $\mathbf K_f$ is the negative of the face-coordinate
phase gradient; its quadratic exponent is therefore identical. No
reciprocal-vector sign is changed in isolation.

| Owner | Comparison |
| --- | --- |
| `montecarlo/bunch_front.py::BunchFrontTilt.gradient` | Degree conversion, tangent, sign, nominal speed, and azimuth match $\mathbf a$. |
| `transport/beam_entry.py::bunch_front_delay_ang` | `solve(J.T, gradient)` gives the exact incoming-offset delay for supported parallel rays. |
| `transport/beam_entry.py::resolve_bunch_times` | Adds imposed delay and signed face flight separately and exactly once; leaves elapsed transport clock untouched. |
| `spectrum/coherent_transverse.py::transverse_spot` | Records $\Sigma_f$ and adds $J^{-\mathsf T}\mathbf a$ to the existing face flight gradient. |
| `spectrum/coherent_transverse.py::transverse_form_factor` | Correct covariance cross term; exponent factor is $1$ for intensity and $1/2$ for amplitude. |
| `spectrum/cxr/_setup.py::_prepare_spectrum` | Nonzero fronts select analytic averaging on infinite slabs as well as supported finite footprints; removes full arrival/position offsets from the pair fields. |
| `spectrum/cxr/_per_hkl.py::_row_decoherence_factor` | Uses residual-time intensity factor times the corrected transverse intensity factor. |
| `spectrum/cxr/_temporal.py::coherent_offset_chi` | Uses residual-time and corrected transverse amplitude factors, each with the $1/2$ exponent. |

`validate_front_source` excludes divergent/Twiss, energy-spread, GDF,
non-Gaussian, explicit longitudinal/offset, and grooved inputs. This makes
the inversion exact for the supported front model. A point spot's zero
position produces zero imposed delay. Zero tilt is canonicalized to no
descriptor and preserves existing entry metadata and random draws.

### Windows, omission bounds, and temporal self terms

`transverse_envelope`, `_interval_exponent`,
`transverse_form_factor_upper`, and `transverse_slope_sign` all use the
same corrected face gradient from `TransverseSpot`. The quadratic
curvature remains nonnegative. Its minimizer and affine derivative are
therefore shifted consistently with the actual phase-matching peak.
The upper envelope bounds the future-frequency supremum; the interval
upper bound uses a lower enclosure of the same quadratic exponent.
`_row_transverse_certified` also uses that corrected gradient when splitting
the monotone sides. `_flat_energy_keep` applies the analytic certificates
to the newly supported infinite-slab front route. The independent
residual-time upper bound remains conservative because the transverse
intensity factor never exceeds one.

For temporal intensity, `add_coherent_row` forms `self_term` from each
particle's realized offset phase, forms the cross amplitude using the
analytic `chi`, and subtracts `cross_self` formed with that same `chi`.
Thus the physical self term is preserved while only inter-electron
coherence is attenuated. Upon frequency integration, Parseval's identity
recovers the self/cross decomposition derived above. The stored pulse RMS
remains the conditional residual duration; no marginal tilted-envelope
RMS is substituted into either temporal or spectral damping.

### Numerical evidence

An independent Rodrigues rotation and direct ray-plane intersections
checked matching target/front tilts $0^\circ$, $30^\circ$, $84.5^\circ$,
and $-30^\circ$ at azimuths $0^\circ$, $75^\circ$, and $145^\circ$.
All twelve cases retained only the prescribed residual face-arrival time,
within $2\times10^{-12}\,\mathrm{\AA}$.

A second comparison at $30\,\mathrm{keV}$, nonnormal face polar angle
$0.4$ rad and azimuth $0.7$ rad, and the off-diagonal covariance above
gave independently calculated transverse phase variances
$0.02463609730998937$, $0.0672071625628638$, and
$0.15259014807888258$ at $\omega=0.01$, $0.02$, and
$0.03\,\mathrm{\AA}^{-1}$. With residual RMS $3\,\mathrm{\AA}$,
the temporal amplitude factors were $0.9873131170683594$,
$0.9652157946136053$, and $0.9227978578543363$. The delay, intensity,
amplitude, and temporal-characteristic-function comparisons passed at
relative tolerance $10^{-14}$; interval and future-frequency upper bounds
enclosed the directly computed factors.

`tests/montecarlo/test_bunch_front_tilt.py`: 19 passed in 0.93 s, including
the matching-face contract, independent ray-phase/Gaussian comparison,
brute-force averaging, signed slopes, conditional/marginal variance, and
unchanged zero-tilt/random-stream behavior.

The owner corrected the reported input-domain guard: `gradient` now
explicitly rejects nonfinite or nonpositive kinetic energy before
computing speed. Source inspection confirms the correction.

**Verdict:** `rederived`. Units, limiting cases, signs, covariance,
amplitude/intensity factors, matching-face convention, window gradients,
and temporal self terms pass for the supported model. Suggested ledger
change: advance `linear-bunch-front-tilt` from `filtered` to `rederived`;
the owner may advance to `anchored` after recording the passing regression
anchor. Only the task owner edits the ledger; human sign-off remains
separate.

### Owner regression and backend evidence

The owner advanced the claim to `anchored` after the regressions above passed.
For the public HOPG c-axis example, the face-frame reciprocal vector has zero
transverse component. Direct evaluation of
`a = J.T @ (n_xy - b_xy / beta0)` at 30 keV, target tilt 84.5 degrees,
azimuth zero, and observation polar angle 145 degrees gives
`a = (40.707319860255346, 0)`: 135.7850031712784 fs/µm, or a front tilt of
85.7217094848694 degrees. This checks the stated geometry and form factor;
it does not establish an experimental source or integrated detector gain.

A bounded CPU/CUDA comparison on an NVIDIA RTX PRO 6000 Blackwell Server
Edition evaluated the changed joint transverse operator at 256 frequencies,
with both intensity and amplitude forms, for nine combinations of normal,
oblique, and grazing target geometry with zero, matched, and reversed front
gradients. CPU and CUDA agreed at `rtol=1e-12`, `atol=1e-14`; maximum absolute
intensity difference was zero. This checks the changed operator, not full
GPU transport or detector-integrated spectral parity.

## Independent verification of the scene inverse

The inverse was derived from the verified phase law before reading
`Scene.phase_matched_bunch_front_tilt` or
`campaign/front_matching.py::phase_matched_front_for_scene`. Setting
$\mathbf K_f=0$ gives

$$
\boxed{\mathbf a_{\rm match}=J^{\mathsf T}
\left(\hat{\mathbf n}_{xy}-\frac{\hat{\mathbf b}_{xy}}{\beta_0}
+\frac{\mathbf g_{xy}}{\omega}\right),\quad
\theta=\arctan\left(\beta_0\|\mathbf a_{\rm match}\|\right),\quad
\phi=\operatorname{atan2}(a_{{\rm match},y},a_{{\rm match},x}).}
$$

The zero-gradient case permits $\theta=\phi=0$. The transpose of $J$,
positive reciprocal-vector term, and laboratory azimuth are fixed by the
phase law; replacing the transpose by an inverse or reversing only the
reciprocal vector would fail the cancellation.

For the default vacuum line, the zero-scattering longitudinal mismatch is
$\omega-\beta_0\hat{\mathbf b}^{\mathsf T}
(\omega\hat{\mathbf n}+\mathbf g)=0$, hence

$$
\boxed{\omega_{\rm line}=
\frac{\beta_0\hat{\mathbf b}^{\mathsf T}\mathbf g}
{1-\beta_0\hat{\mathbf b}^{\mathsf T}\hat{\mathbf n}},\qquad
E_{\rm line}=\hbar c\,\omega_{\rm line}.}
$$

This default is a vacuum line estimate, not a refractively shifted peak.
An explicit photon energy instead fixes $\omega=E/(\hbar c)$ and solves
the transverse matching problem at that frequency. It need not satisfy
the nominal longitudinal line condition. In both cases the independent
residual-duration damping remains unchanged.

Source inspection confirms proper sample/laboratory rotation of beam and
observation directions, mean catalog reciprocal orientation, normalization
of an explicit laboratory ray, and the scalar detector's central-ray
default. The supported domain is a scalar, flat, monoenergetic slab with
the previously verified point/Gaussian source. A planar detector requires
an explicit ray. Integer nonzero Miller indices, finite positive beam and
photon energies, and a finite nonzero observation direction are enforced;
a nonpositive default line is rejected. Matching addresses one reflection,
frequency, and direction in the mean orientation.

For an independent off-axis check, HOPG $(102)$ has
$\mathbf g_{xy}=(2.5531025222184422,1.4740344284715303)^{\mathsf T}
\,\mathrm{\AA}^{-1}$ from its hexagonal lattice constants. At beam energy
$30\,\mathrm{keV}$, target tilt $30^\circ$, azimuth $75^\circ$, and
laboratory ray proportional to $(0.6,-0.4,0.7)$, direct Rodrigues rotation
and ray-plane intersections give default energy
$486.718502128692\,\mathrm{eV}$. The returned front angles are
$(77.90231131695276^\circ,35.0076403128878^\circ)$; direct ray-phase
residuals are at most $1.12\times10^{-15}$. At explicit energy
$1800\,\mathrm{eV}$ the angles are
$(57.56561284098481^\circ,36.778804207642885^\circ)$ and residuals at
most $1.78\times10^{-15}$.

A separate orientation check uses silicon's catalog $(440)$ surface,
the $(404)$ reflection, target tilt $35^\circ$, azimuth $-45^\circ$, and
laboratory ray proportional to $(0.2,0.6,0.7)$. An independently constructed
minimal rotation of the cubic reciprocal basis gives
$\mathbf g=(-0.9584343549456422,-5.586164791653802,3.2722995732997235)
\,\mathrm{\AA}^{-1}$ in the sample frame. At $1800\,\mathrm{eV}$ the
returned angles are $(67.18681170626635^\circ,-83.16404834735738^\circ)$
and direct phase residuals are at most $2.23\times10^{-15}$.

Scalar/explicit central-ray equivalence and invalid input/source guards
also passed. `tests/scan/test_bunch_front_matching.py`: 23 checks passed;
the combined helper and documentation checks passed 29 tests in 1.34 s.
These include signed/grazing off-axis ray phases, the planar ray contract,
and actual scene-to-case checks for off-grid target angles.

The owner corrected an initial raw-versus-resolved target-angle mismatch
by reusing `target._angles()`, exactly as scene lowering does. Both the
default line and inverse gradient now use the same resolved geometry as
simulation. Independent direct ray checks used raw angle pairs
$(30.24^\circ,75.24^\circ)$ and $(30.25^\circ,-75.25^\circ)$, which lower
to $(30^\circ,75^\circ)$ and $(30.5^\circ,-75.5^\circ)$ respectively.
For HOPG $(102)$ at $1800\,\mathrm{eV}$ and the off-axis ray above,
both corrected fronts leave phase-gradient residuals below
$9\times10^{-16}\,\mathrm{\AA}^{-1}$ in the actual lowered geometry.
The second case also verifies half-degree ties away from zero for a
negative azimuth. **Final inverse verdict:** `rederived`; signs, frame
rotations, transpose, line formula, and source-to-case geometry agree.
