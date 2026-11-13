# Line/TR phase reconciliation

## Claim, source and independence

Claim: `line-tr-phase-reconciliation`. Fresh-context verification, 2026-11-13,
against revision `ec70788c8ebc0507931d619f1899f7a91b01aefd`.
Only this claim is assessed; adjacent boundary, transport and quantum claims
are outside scope. This page records independent evidence, not human sign-off.

Source: Feranchuk, Ulyanenkov, Harada and Spence, *Physical Review E* **62**,
4225–4234 (2000), [doi:10.1103/PhysRevE.62.4225](https://doi.org/10.1103/PhysRevE.62.4225),
pp. 4226–4228, Eqs. (1), (2), (4), (5), (8), (13), (14).
A local PDF was read with text extraction; printed p. 4228 was also rendered
and visually inspected to disambiguate Eq. (14). The publisher PDF request
returned HTTP 401. An author-uploaded [full-text copy](https://www.researchgate.net/publication/12238791_Parametric_x-ray_radiation_and_coherent_bremsstrahlung_from_nonrelativistic_electrons_in_crystals)
was accessible independently. Local PDF SHA-256:
`4d06d3ff5a66d6b5737d9e2b51a25a81620580f35227c5ba67b3789ab008d3dc`.
No external-code or hardware evidence is asserted.

Important source qualification: printed Eq. (14) is **nonrelativistic**. It
contains neither a Lorentz factor nor the longitudinal relativistic projector.
The relativistic extension below follows from the relativistic equation of
motion; it is not a literal transcription of printed Eq. (14). The ledger's
phrase “relativistic Feranchuk Eq. (14)” must be read with that qualification.

Before reading implementation bodies, signatures and docstrings were extracted
with Python AST, without importing modules. Contracts:

- `_drive(omega, kxy, eps, current, polarization, magnetization)`:
  Maxwell source after eliminating longitudinal fields, per unit time;
  Gaussian constitutive relations and negative time-frequency convention.
- `_cbs_dipole(velocity, g, potential)`:
  charge-free first-order displacement, inverse eV; electron potential
  energy is minus the supplied potential coefficient; nonzero velocity–harmonic
  contraction and locally straight perturbative motion.
- `_segment_rhs(rhs, modes, media, heights, current, omega, kxy, g, potential, origin)`:
  accumulate the current and CBS contributions in harmonic rows.
- `_row_parameters(case, energy)`: resolve transfer harmonics in the Maxwell
  phase convention. Its docstring does not specify the harmonic conjugation;
  this must be checked after derivation.

## Independent derivation recorded before body inspection

Use Gaussian units, $\hbar=c=1$, signed electron charge $q<0$, electron mass
$m$, velocity $\mathbf v$, and $\gamma=(1-v^2)^{-1/2}$. Fields have time
factor $\exp(-i\omega t)$; the outgoing test wave has spatial wavevector
$\mathbf k$. Select the emission harmonic $\exp(-i\mathbf g\cdot\mathbf r)$,
with electron potential energy $-U\exp(-i\mathbf g\cdot\mathbf r)$.
The paper uses the opposite spatial harmonic in the conjugate radiation
integral; reversing that harmonic and conjugating the integral must be done
together. Its cell Fourier coefficient maps to $U=eU_{\mathbf g}/\Omega$
(energy), not to a second factor of charge times an already normalized energy.
Let $s=\mathbf v\cdot\mathbf g\ne0$.

The relativistic acceleration equation gives

$$
\frac{d\delta\mathbf v}{dt}
=\frac{1}{\gamma m}(\mathbf 1-\mathbf v\mathbf v^{\mathsf T})\mathbf F,
\qquad
\mathbf F=-i\mathbf gU\exp(-i\mathbf g\cdot\mathbf r).
$$

Consequently the coefficients of the periodic velocity and displacement are

$$
\mathbf d=\frac{U}{\gamma m s}(\mathbf g-\mathbf v s),
\qquad
\mathbf a=\frac{i\mathbf d}{s},
\qquad
\delta\mathbf r(t)=\mathbf a\exp(-i\mathbf g\cdot\mathbf r(t)).
$$

There is a single $1/\gamma$; longitudinal acceleration already has its
additional suppression in the projector. The nonrelativistic limit of the
resulting amplitude is the printed Eq. (14).

Vary the point current, including the displacement of its delta function:

$$
\delta\rho=-q\,\delta\mathbf r\cdot\nabla\delta(\mathbf x-\mathbf r),
\qquad
\delta\mathbf J=q\left[\delta\mathbf v\,\delta(\mathbf x-\mathbf r)
-\mathbf v\,\delta\mathbf r\cdot\nabla\delta(\mathbf x-\mathbf r)\right].
$$

This is exactly the bound current of

$$
\mathbf P=q\delta\mathbf r\,\delta(\mathbf x-\mathbf r),
\qquad \mathbf M=\mathbf P\times\mathbf v,
\qquad \delta\mathbf J=\partial_t\mathbf P+\nabla\times\mathbf M.
$$

In particular, retaining the full polarization is necessary. In spatial Fourier
variables its current coefficient is

$$
\delta\mathbf J=-i\left[(\omega-\mathbf k\cdot\mathbf v)\mathbf P
+\mathbf v(\mathbf k\cdot\mathbf P)\right].
$$

On the vacuum resonance $\omega=\mathbf v\cdot(\mathbf k+\mathbf g)$,
projection on a transverse polarization $\hat{\mathbf e}$ gives

$$
\hat{\mathbf e}\cdot\delta\mathbf J
=q\left[\hat{\mathbf e}\cdot\mathbf d
+\frac{(\hat{\mathbf e}\cdot\mathbf v)(\mathbf k\cdot\mathbf d)}{s}\right]
=-q A_{\rm CBS},
$$

$$
A_{\rm CBS}=-\frac{U}{\gamma m s}
\left[\{\mathbf g;\hat{\mathbf e}\}
+\frac{\{\mathbf g;\mathbf k\}(\mathbf v\cdot\hat{\mathbf e})}{s}\right],
\qquad
\{\mathbf a;\mathbf b\}=\mathbf a\cdot\mathbf b
-(\mathbf a\cdot\mathbf v)(\mathbf b\cdot\mathbf v).
$$

For PXR, independently solve Maxwell's equation for the straight current at
$\mathbf K=\mathbf k+\mathbf g$. With $\Delta=K^2-\omega^2$, its field
coefficient is $4\pi i q(\omega\mathbf v-\mathbf K)/\Delta$.
The first-order susceptibility current is $-i\omega\chi\mathbf E/(4\pi)$,
so its transverse coefficient is also $-q A_{\rm PXR}$, where

$$
A_{\rm PXR}=\frac{\chi}{\Delta}
\left[(\mathbf v\cdot\mathbf K)(\mathbf g\cdot\hat{\mathbf e})
-\omega^2(\mathbf v\cdot\hat{\mathbf e})\right].
$$

This is the sign and normalization of printed Eq. (13). Thus the two
amplitudes add with one common outgoing-field conversion, derived without
fitting intensities. A vacuum outgoing Green function for
$(\partial_z^2+k_z^2)E=-4\pi i\omega J_\perp$ gives

$$
E_{\rm out}=\frac{2\pi q\omega}{k_z}
(A_{\rm PXR}+A_{\rm CBS}) I,
\qquad
I=\int_{t_0}^{t_0+T}
\exp\left[i\omega t-i(\mathbf k+\mathbf g)\cdot\mathbf r(t)\right]dt.
$$

This coefficient refers to the plane-wave field coefficient at global $z=0$;
propagating to a different output plane adds its explicit propagation phase.
For $\mathbf r(t)=\mathbf r_0+\mathbf v(t-t_0)$ and
$\delta=\omega-(\mathbf k+\mathbf g)\cdot\mathbf v$,

$$
I=T\operatorname{sinc}(\delta T/2)
\exp\left[i\omega t_0-i(\mathbf k+\mathbf g)\cdot\mathbf r_0
+i\delta T/2\right],
\qquad \operatorname{sinc}(x)=\frac{\sin x}{x}.
$$

Keeping this midpoint phase, the harmonic's phase at the actual origin and
signed charge is essential for an absolute finite-passage comparison.
The source paper's intensity factor alone does not fix an output-plane phase;
Maxwell's outgoing Green function supplies that convention here.

For the tangential state $(E_x,E_y,H_x,H_y)$, eliminate longitudinal fields
using the stated constitutive laws. With lateral wavevector $(k_x,k_y)$,
the source part per unit longitudinal distance must be

$$
\mathbf S=4\pi\begin{pmatrix}
 k_xJ_z/(\omega\epsilon)-ik_xP_z/\epsilon+i\omega M_y\\
 k_yJ_z/(\omega\epsilon)-ik_yP_z/\epsilon-i\omega M_x\\
 J_y-i\omega P_y-ik_xM_z\\
 -J_x+i\omega P_x-ik_yM_z
\end{pmatrix}.
$$

A time-integrated moving point source uses the same coefficients per unit time:
there is no division by $v_z$ and no singularity for a horizontal passage.

## Cheap filters before implementation comparison

| Filter | Result | Reason |
| --- | --- | --- |
| Units | pass | $U/m$ and $\mathbf d$ are dimensionless; $s$ and $\omega$ have energy units; $\mathbf a$ and $I$ are inverse energy; both amplitudes and $\omega/k_z$ are dimensionless. |
| Limits | pass | $U\to0$ removes CBS; $\chi\to0$ removes PXR; $T\to0$ gives a field linear in $T$; on resonance $I=T$ times the origin phase; $v\ll1$ recovers printed Eq. (14). |
| Signs/conventions | pass | Both transverse currents are $-qA$; magnetization and longitudinal polarization follow directly from variation of the conserved point current. |

These are algebraic filters within the stated classical, first-order,
locally straight, vacuum/on-resonance comparison. The singular $s\to0$ limit
is excluded, not a physical prediction. No thick-crystal, absorption,
recoil or magnetic-moment equivalence to the current production line scorer
is inferred.

## Implementation comparison and adjudication

The independent equations above were saved before inspecting implementation
bodies or the author derivation. Serena was attempted for the four symbols;
its active project was the parent checkout and excluded this worktree as
ignored. Targeted AST source extraction was therefore used for body comparison.

| Owner | Exact comparison | Result |
| --- | --- | --- |
| `_cbs_dipole` | `potential` is $U/m$, not $U$; `delta_v` is $\mathbf d$ and the returned dipole is $i\mathbf d/s$. Its near-zero contraction guard excludes the singular limit. | matches |
| `_drive` | All four entries equal $\mathbf S$; matrix `inverse @ (...)` generalizes the scalar longitudinal inverse to the two harmonic rows. Neither $P_z$ nor any component of $\mathbf P\times\mathbf v$ is dropped. | matches |
| `_segment_rhs` | Charge is $\sqrt{\alpha}$ times `current.charge`, with charge units $+e$; this equals the signed $q$ used above for a negative input. Both dipole and charge drive receive the same charge factor. Transfer and reverse use $\pm\mathbf g$ and $\exp(\pm i\mathbf g\cdot\mathbf r_{\rm origin})$. The CBS source uses the other charge harmonic's phase. | matches |
| `_row_parameters` | Divides the already energy-normalized `U_g` by `M_E_EV` exactly once. Transfer uses the conjugated negative-index material coefficient; reverse uses the conjugated positive-index coefficient for both optical and static couplings. | matches |

The material definitions were read only after derivation to resolve boundary
units and Fourier notation: `U_g` already contains the paper's charge and cell
volume; `chi_g` is dimensionless. Direct atomic coefficients use positive
basis-position phases. Reversing the Miller index and conjugating gives the
positive transfer basis phase with conjugated atomic optical response, as
required by the emission/Maxwell frequency convention. For a real static
potential, conjugate Friedel pairing returns the original positive-index
coefficient. This check establishes the mapping, not new validation of atomic
optical tables or their accuracy near edges.

For absolute phase, targeted inspection of `_segment_faces`, `_time_integral`
and the `two_beam_current_fields` boundary conversions confirms that positions,
clocks, durations and face heights are divided by `HBARC_EV_ANG`, while
reciprocal vectors are multiplied by it. The phase integral includes
$\omega t_0$, the source position, the harmonic gauge, and the output face.
The output-face field therefore equals the derived expression multiplied by

$$
\exp\left[i\mathbf g\cdot\mathbf r_{\rm origin}+ik_z z_{\rm face}\right].
$$

The phase and integration interval remain well defined for $v_z=0$; no
conversion to a path integral introduces a spurious $1/v_z$. The author
[CBS phase derivation](../../physics/radiation-physics/diffracted-transition-radiation.md#cbs-phase-in-the-same-maxwell-basis)
was read after the independent derivation and agrees term by term. It is a
consistency comparison, not the independent source used to derive this page.

## Independent numerical spot check

A separate scratch calculation evaluated the outgoing vacuum Green integral
from $\mathbf P=q\mathbf a$ and $\mathbf M=\mathbf P\times\mathbf v$,
without calling implementation helpers for the reference. It used
96-point Gauss–Legendre quadrature of the complex exponential, not the
implementation's endpoint integral. It then compared all three complex
Cartesian components with the difference of coupled and bare fields returned
by `two_beam_current_fields`. No intensity or fitted phase entered.

Inputs: $\mathbf v=(0.16,-0.08,0.73)$,
$\hat{\mathbf n}=(0.28,0.19,\sqrt{1-0.28^2-0.19^2})$,
$\mathbf g=(0.014,-0.011,0.043)$ Å$^{-1}$,
$U/m=1.7\times10^{-5}+0.6\times10^{-5}i$,
$\mathbf r_0=(4,-1,7)$ Å, $\mathbf r_{\rm origin}=(2,-3,5)$ Å,
$t_0=11$ Å, $T=61$ Å, output face $z=80$ Å, $q=-\sqrt{\alpha}$.
The segment remains inside the vacuum layer; susceptibility is zero and
there are no exterior continuations. At each frequency the independently
constructed transverse current is

$$
\mathbf J_\perp=(\mathbf 1-\hat{\mathbf n}\hat{\mathbf n}^{\mathsf T})
[-i\omega\mathbf P+i\mathbf k\times\mathbf M].
$$

The reference is $-2\pi\mathbf J_\perp/n_z$ times the quadrature integral,
including both origin and output-face phases. Off resonance this is the
finite-window polarization/magnetization source, including its switching
current; it does not assert equivalence to a displacement abruptly clipped
without endpoint currents.

| Frequency / resonance | Photon energy (eV) | Relative complex-vector norm error |
| --- | --- | --- |
| 0.87 | 209.004130697347 | $2.41845\times10^{-12}$ |
| 1 | 240.234632985456 | $1.81041\times10^{-12}$ |
| 1.13 | 271.465135273565 | $2.68748\times10^{-12}$ |

The acceptance threshold was $10^{-9}$, selected for a double-precision
algebraic identity with subtraction of the bare field, not a source-data
uncertainty or empirical physical-accuracy claim. The complex potential,
negative charge, nonzero clock and origin test phase conventions beyond the
existing real-potential, zero-clock/origin examples. Reference code and logs
are scratch evidence; the equations, complete inputs and results here are the
durable record.

## Remaining filter checklist

- Notation/normalization: pass, including the paper's cell volume and charge
  folded into the supplied energy potential, one Lorentz factor, common
  field conversion and harmonic conjugation.
- Units at boundaries: pass for the material potential, dimensionless
  susceptibility, Å-to-inverse-eV conversion, and inverse-eV output field.
  No source table or measured reference data is transcribed for this claim.
- Reference transcription: printed Eqs. (13) and (14), p. 4228, were visually
  checked; the relativistic projector is explicitly derived here rather than
  attributed to the printed nonrelativistic formula.
- Tolerances: the pure-CBS existing test uses relative $10^{-10}$ and absolute
  $10^{-13}$ for a numerical identity; the joint weak-coupling test uses
  relative $10^{-5}$ at susceptibility $10^{-7}$, allowing higher-order
  scattering absent from the Born reference. These are consistency
  tolerances, not experimental uncertainties. The new scratch check uses
  $10^{-9}$ as described above.
- External-code and CPU/GPU parity evidence: not applicable to this bounded
  derivation; no such result is borrowed from an adjacent ledger claim.

## Verdict and ledger integration

**Re-derivation: matches. Verdict: `rederived`.** No divergent factor, sign,
exponent, unit or absolute phase was found for this bounded classical
vacuum/on-resonance claim. The independent derivation extends the printed
nonrelativistic source using relativistic point-current dynamics. This does
not certify the separate quantum production amplitudes, absorbing-medium
legacy-line agreement, or adjacent boundary/transport claims.

The owner reviewed this report and promoted the ledger row to `rederived`,
linked this page, and qualified the source as Feranchuk Eqs. (13)–(14) with
its relativistic extension derived from the equation of motion and Maxwell
current variation. The locally straight, first-order and nonzero-contraction
restrictions remain. Human sign-off remains separate in #277.

## Checks

- Focused existing CBS phase test, joint PXR/CBS phase test, and
  `tests/dev/test_docs.py`: **11 passed in 4.40 s**.
- Independent three-frequency complex-field spot check: **passed**;
  maximum relative norm error $2.68748\times10^{-12}$.
- Initial strict docs build identified only the missing toctree ownership of
  this new page. The verifier temporarily declared it an orphan within
  document-only authority; owner integration adds it to the radiation evidence
  toctree. The verifier's final canonical strict docs build **passed**,
  with demo export enabled and no Sphinx warnings. The demo exporter emitted
  its separate advisory that Plotly is not bundled in the Pyodide lockfile.
- Rendered article: **93** `math notranslate` elements (**13** display,
  **80** inline), matching every source expression; **zero** literal dollar
  delimiters in the article body.
- Final document-only docs tests: **9 passed in 0.08 s**. No implementation,
  test, ledger, issue or tracked generated file was changed.
