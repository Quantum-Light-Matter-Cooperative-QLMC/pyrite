# Experimental kinematic diffracted transition radiation

`Validation: dtr-kinematic-boundary`

`montecarlo.spectrum.dtr` computes a first-Born **field diagnostic**, separately
from production PXR/CBS. It diffracts the homogeneous stack's free TR field
once, through one crystal harmonic. `pxr_straight_amplitude` diffracts the
charge's own field through the same harmonic on a straight passage. Their
vectors share a Maxwell normalization and reference plane and can be summed
before squaring. This is the first slice of #581. An experimental
[opt-in transported-current production path](#opt-in-transported-current-production-scoring)
now combines a two-beam solve with a CBS current. A separate experimental
[two-beam straight-passage diagnostic](#two-beam-straight-passage-diagnostic)
is described below.

The own-field/free-field separation follows
[Artru and Rullhusen, NIM B **145**, 1–7 (1998)](https://doi.org/10.1016/S0168-583X(98)00241-9).
The publisher's metadata and abstract were checked; its numerical figures have
not been reproduced. Below is an implementation-author derivation from
Maxwell's wave equation, not an independent validation write-up or a
transcription of that paper's numbered equations.

## Harmonic and crossing source

Use Gaussian units, $\hbar=c=1$, positive charge $e=\sqrt{\alpha}$, and time
dependence $e^{-i\omega t}$. Solver coordinates increase from the substrate
interface at $z=0$ toward the emitting vacuum, with $n_z>0$. Backward emission
is represented by a charge entering from the top face with $v_z<0$. API lengths
in Å are divided by $\hbar c$ internally. One crystal layer has permittivity

$$
\varepsilon(\mathbf r)=\varepsilon_0(z)
+\chi_{\rm tr}e^{-i\mathbf g\cdot(\mathbf r-\mathbf r_0)}.
\tag{1}
$$

`chi_transfer` means the coefficient in (1), not an unlabelled line-table
value. The incident free or own field has transverse momentum
$\boldsymbol\kappa_0=\omega\hat{\mathbf n}_\parallel+\mathbf g_\parallel$.
The homogeneous stack's tangential own-field jumps are phased with

$$
\exp[i(\omega t_c-\boldsymbol\kappa_0\cdot\mathbf r_{\parallel,c})].
\tag{2}
$$

The existing continuity matrix, solved at $\boldsymbol\kappa_0$, yields both
free-wave directions inside the crystal. The input may be evanescent in an
outer medium; it is not assumed to be a vacuum photon. The first-order source
is $4\pi\mathbf P=\chi_{\rm tr}\mathbf E_{\rm free}
e^{-i\mathbf g\cdot(\mathbf r-\mathbf r_0)}$. Replacing the free field by the
crystal own field gives the diagnostic PXR term.

## Driven layer and finite integral

Let $\mathbf C(z)=4\pi\mathbf P(z)$ after transverse Fourier transformation,
with outgoing $\boldsymbol\kappa=\omega\hat{\mathbf n}_\parallel$.
Maxwell's driven equation
$[\nabla\times\nabla\times-\varepsilon_0\omega^2]\mathbf E=\omega^2\mathbf C$
gives the outgoing field of a polarization sheet in the unbounded crystal:

$$
d\mathbf E_\pm=\frac{i\omega^2}{2\lambda}
\left[\mathbf I-\frac{\mathbf K_\pm\mathbf K_\pm^{\mathsf T}}
{\varepsilon_0\omega^2}\right]\mathbf C(z)\,dz,
\quad \mathbf K_\pm=(\boldsymbol\kappa,\pm\lambda),\quad
\lambda=\sqrt{\varepsilon_0\omega^2-\kappa^2},\quad
\operatorname{Im}\lambda\ge0.
\tag{3}
$$

The sign follows from $H_y^+-H_y^-=i\omega C_x$ at normal incidence.
The outgoing wave is transverse; the longitudinal contact field stays at the
sheet. Its magnetic field is $\mathbf H_\pm=\mathbf K_\pm\times\mathbf E_\pm/\omega$.
Each incident wave, lattice factor and propagation leg reduces to

$$
I(a,b,d)=\int_0^d e^{iax+ib(d-x)}\,dx
=\frac{e^{iad}-e^{ibd}}{i(a-b)},\qquad I(a,a,d)=d e^{iad}.
\tag{4}
$$

Both endpoint exponentials are bounded in passive media. Near equality the
implementation uses $d\exp[i(a+b)d/2]\{1-[(a-b)d]^2/24+[(a-b)d]^4/1920\}$.
Exact diffraction matching is finite. At the crystal upper face the upward
source field enters the homogeneous outgoing continuity solve with positive
sign; at its lower face the downward source enters with negative sign. These
signs follow by moving the particular layer field to the right-hand side of
the upper-minus-lower tangential continuity equations. The outgoing stack
solve retains homogeneous Fresnel reflection and absorption in every layer.

## Phase, normalization and limits

The returned Cartesian vacuum amplitude has units $\mathrm{eV}^{-1}$ and the
same top-interface reference plane as `interface_transition_radiation_amplitude`:

$$
\frac{d^2N}{dE\,d\Omega}=\frac{E n_z^2}{16\pi^4}
\left|\mathbf a_{\rm PXR}+\sum_c\mathbf a_{{\rm DTR},c}\right|^2.
\tag{5}
$$

Both functions use positive charge; multiply both fields by $-1$ for an
electron. Changing lattice origin multiplies the diffracted row by
$e^{i\mathbf g\cdot\Delta\mathbf r_0}$. Translating a history in the surface
plane multiplies both PXR and DTR by
$e^{-i(\boldsymbol\kappa+\mathbf g_\parallel)\cdot\boldsymbol\Delta}$.
Their cross term survives a spot average. Ordinary TR gains
$e^{-i\boldsymbol\kappa\cdot\boldsymbol\Delta}$, giving the relative factor of
[`tr-line-cross-term`](../../validation/radiation-physics/tr-line-cross-term.md).

Zero harmonic gives zero diffracted field. Globally matched dielectric media
give zero free field and zero DTR, while a finite nonzero harmonic can still
scatter the own field. The vanishing-thickness limit is zero. Passive opaque
layers use bounded propagators. Vacuum PXR agrees with a closed dipole-volume
integral. A separate test differentiates the existing full Maxwell solve for
$\varepsilon(z)=\varepsilon_0+h e^{-ig_z z}$, represented by a fine staircase:
the complex derivative equals $(\mathbf a_{\rm PXR}+\mathbf a_{\rm DTR})/
\chi_{\rm tr}$. This is internal solver consistency, not a published-result
reproduction. Tests live in `tests/montecarlo/test_transition_radiation_dtr.py`.

## Alternating reference target and profile

The shipped diagnostic profile is
`src/pyrite/validation/reference_data/dtr_stack.toml`: vacuum / **5 nm amorphous
SiO₂ / 10 nm MoS₂ / 5 nm amorphous SiO₂** / vacuum, entrance-to-exit. Only MoS₂
(002) diffracts. Its lattice origin is at its substrate-side face, $z=50$ Å.
There is no assumption of registry across separate crystals.

The deterministic passage is normal to the faces, at 10 MeV kinetic energy;
observation is 0.05 rad from exact backward emission. A 120 eV window is
sampled at 0.1 eV spacing. It uses no Monte Carlo sample count or seed and
makes **no claim of validated 10 MeV transport**. This is a diagnostic TOML
profile, not a `pyrite run` campaign profile.

| Calculated scale at the profile's window center | Value |
| --- | ---: |
| Vacuum phase-matching energy | 1008.53 eV |
| $\gamma^{-2}$ | 0.00236 |
| $\lvert\chi_0\rvert$ | 0.00173 |
| Formation scale $\hbar c/\lvert q+\lambda\rvert$ | 59.5 nm |
| Born coupling scale $2\hbar c\lvert\lambda\rvert/(E^2\lvert\chi_{\rm tr}\rvert)$ | 456 nm |
| Absorption depth $\hbar c/(2\operatorname{Im}\lambda)$ | 542 nm |
| Maximum thickness / Born coupling scale across window | 0.0233 |

MoS₂'s longer (002) spacing gives a softer line than HOPG and comparable
dielectric/inverse-gamma-squared scales here. The caps exercise extra
interfaces while the crystal stays thin relative to these scales. This is a
diagnostic target, not an experimentally optimized radiator. The maximum
DTR-on/off difference is about 0.783 times the PXR-only peak in this model;
inspect the signed interference spectrum rather than interpreting this as
a flux enhancement.

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache MPLCONFIGDIR=/tmp/pyrite-mpl \
  uv run python -m pyrite.validation.dtr_reference \
  --output-dir /tmp/pyrite-dtr-reference
```

Outputs: `spectrum.csv`, `reference.json`, `spectrum.png`. `pxr` is DTR off on
the identical passage; `line` is PXR+DTR, with `dtr` and `pxr_dtr_cross` recorded
separately. `total` additionally includes ordinary TR and `tr_line_cross`:
this surface-normal reflection has a fixed termination and keeps that cross
term. Densities are photons/eV/sr per electron. JSON records profile and
resolved-material digests, model marker, scales and runtime. The initial
1201-node calculation took 0.17 s after imports on the development CPU;
plotting and process startup are additional.

## Validity and remaining production work

Only one scattering by the crystal harmonic is included. The Born condition is
$\eta=E^2\lvert\chi_{\rm tr}\rvert d/(2\sqrt{\lvert\lambda_{\rm in}
\lambda_{\rm out}\rvert})\ll1$, with $d$ in inverse eV. The low-level function
evaluates the formal first-order field beyond that condition for diagnostics;
it does not certify those results. It cannot reproduce thick-crystal primary
extinction. The reference reports its maximum $\eta$.

The PXR diagnostic assumes a globally straight passage; DTR retains the
existing locally straight crossing assumption. There is no CBS, mosaic
average, cross-reflection coherence, channeling, grooves, detector-acceptance
integration or CUDA validation. Production still needs the phase/sign/basis
reconciliation independently checked, broader catalog/campaign support and
the two-beam limits/published reproduction. The transported-current path below
implements an experimental Maxwell/CBS reconciliation and layered escape;
the reference diagnostic above remains first Born without CBS.
Independent validation is pending by user choice. The claims
`dtr-two-beam-boundary` and `line-tr-phase-reconciliation` remain open in #581.


## Two-beam straight-passage diagnostic

`Validation: dtr-two-beam-boundary`

`dtr_two_beam.two_beam_straight_field` is a separate experimental CPU
boundary diagnostic. It retains both polarizations, both propagation senses
and repeated diffraction between two transverse harmonics in one crystal
layer. Its ledger status is **unverified**. The checks below establish internal
consistency; they do not replace independent rederivation, a published
spectral-angular reproduction, or an extinction-law anchor. This section is
an implementation-author derivation from Maxwell's equations. Caticha's
[two-wave boundary formulation](https://doi.org/10.1103/PhysRevA.40.4322)
is source context, not a numbered equation transcribed here.

With the same frequency convention as above, define transverse harmonics
$\boldsymbol\kappa_j=\omega\hat{\mathbf n}_\parallel+j\mathbf g_\parallel$
for $j=0,1$. Write physical harmonic-1 fields as
$\mathbf E_1(z)=e^{ig_z z}\widetilde{\mathbf E}_1(z)$ and likewise for
$\mathbf H_1$. The permittivity in this gauge is the constant matrix

$$
\mathcal E=\begin{pmatrix}
\varepsilon_0 & \chi_{\mathbf g}e^{i\mathbf g\cdot\mathbf r_0}\\
\chi_{-\mathbf g}e^{-i\mathbf g\cdot\mathbf r_0}&\varepsilon_0
\end{pmatrix}.
$$

In an absorbing crystal the reverse harmonic is not generally the conjugate
of the transfer harmonic. The anti-Hermitian part
$(\mathcal E-\mathcal E^\dagger)/(2i)$ must be positive semidefinite;
active truncated permittivity is refused.

For two-component harmonic vectors, let $K_x,K_y$ be the diagonal transverse
momenta and $Y=(E_x,E_y,H_x,H_y)^{\mathsf T}$. Eliminating the longitudinal
components gives

$$
E_z=\mathcal E^{-1}\left[\frac{K_yH_x-K_xH_y}{\omega}
-\frac{4\pi iJ_z}{\omega}\right],\qquad
H_z=\frac{K_xE_y-K_yE_x}{\omega}.
$$

The tangential equations are $Y'=iMY+f$, with the ungauged blocks

$$
M=\begin{pmatrix}
0&0&K_x\mathcal E^{-1}K_y/\omega&\omega I-K_x\mathcal E^{-1}K_x/\omega\\
0&0&K_y\mathcal E^{-1}K_y/\omega-\omega I&-K_y\mathcal E^{-1}K_x/\omega\\
-K_xK_y/\omega&K_x^2/\omega-\omega\mathcal E&0&0\\
\omega\mathcal E-K_y^2/\omega&K_yK_x/\omega&0&0
\end{pmatrix},
$$

and

$$
f=\begin{pmatrix}
4\pi K_x\mathcal E^{-1}J_z/\omega\\
4\pi K_y\mathcal E^{-1}J_z/\omega\\
4\pi J_y\\-4\pi J_x
\end{pmatrix}.
$$

The gauge subtracts $g_z$ from harmonic 1's four diagonal entries. All lengths
in this system are inverse eV; API Å values are divided by $\hbar c$.

The uncoupled homogeneous-stack own and free fields give $Y_0$ exactly for a
globally straight charge. Solve only for the diffraction correction:

$$
\delta Y'=iM\delta Y+i(M-M_0)Y_0+(f-f_0).
$$

The last term is essential: the own field contains an external longitudinal
current, so it does not obey the source-free longitudinal relation. Omitting
this contact term produced a roughly 7% complex-field error in the initial
weak-coupling test. The retained correction uses the difference of the coupled
and uncoupled inverse permittivities multiplying that current. Current and
field phases use the same specified time and point on the straight trajectory.

Diagonalize $M=V\Lambda V^{-1}$. Classify outgoing modes by passive decay,
or by the sign of their normal Poynting flux when real. Reference an upward
mode at the lower face and a downward mode at the upper face, so propagation
through the layer never grows. The finite-layer source integrals use (4),
including its equal-exponent limit, and only the outward subset of modes is
evaluated at either face. Tangential continuity of both harmonics is imposed
at every interface. The two exterior half-spaces carry outgoing modes only.

The result separates the response to the observed charge harmonic
(`transition`, including diffraction extinction) from the response to its
second harmonic (`line`, including PXR and DTR). Their complex sum is `total`.
Use (5) for the photon density. Source charge is positive; multiply both fields
by the electron charge sign before combining with any other amplitude.
The surface termination rotates the second source row while leaving the
observed source row invariant.

Focused tests cover zero harmonics in both passage directions, the weak-coupling
limit against the existing Born PXR+DTR vector, the termination phase, active
permittivity rejection, and a passive $10^8$ Å opaque layer without growing
exponentials. With the reverse harmonic set to zero, the finite-strength
triangular system agrees with the Born vector through capped and uncapped
stacks, for oblique reciprocal vectors and both passage directions. Turning
the reverse harmonic back on gives the leading second-order correction to
the transition field and third-order correction to the line field. Separate
checks cover the two source rows' translation phases, invariance under choosing
another point on the same straight trajectory, transversality, and agreement
of spectral/angular array evaluation with individual solves. These are
diagnostic checks only. Independent verification of the transported-current
Maxwell/CBS reconciliation below, mosaicity, broader catalog/campaign support,
recoil/energy-loss justification, and a
published dynamical anchor remain open in #103/#581. This new diagnostic has
not been run on CUDA; the hardware evidence below concerns the existing
homogeneous TR/CTR implementation.

### Compare the two-beam and Born reference fields

The same alternating reference stack can be scored with both diagnostics:

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache MPLCONFIGDIR=/tmp/pyrite-mpl \
  uv run python -m pyrite.validation.dtr_reference --two-beam \
  --output-dir /tmp/pyrite-dtr-two-beam
```

The default invocation and its Born columns are unchanged. `--two-beam`
appends these photon-density columns to `spectrum.csv`:

| Column | Meaning |
| --- | --- |
| `two_beam_transition` | Observed charge-harmonic source row, including its diffraction extinction |
| `two_beam_line` | Second charge-harmonic source row, including PXR and DTR together |
| `two_beam_cross` | Signed interference of those two source rows |
| `two_beam_total` | Intensity of the summed complex fields; sum of the preceding three columns |
| `two_beam_minus_born_total` | Signed difference from the Born `total` on the identical passage |

The two-beam line row has no separate PXR-only or DTR-off interpretation.
Its transition row also differs from ordinary homogeneous-stack TR once
repeated diffraction matters. The figure overlays the Born and two-beam
totals and shows their signed difference in a second panel. JSON records a
distinct comparison model marker, hashes both transfer
and reverse susceptibilities, and reports the maximum absolute total-density
difference divided by the Born total's peak. The reverse susceptibility is
evaluated from the opposite Miller indices in the same frequency convention;
it is not obtained by conjugating the transfer susceptibility in an absorbing
crystal.

These remain **unverified CPU straight-passage diagnostics**. Comparing them
tests the Born approximation against this implemented two-harmonic model; it
does not establish a published extinction law, production PXR/CBS phase
agreement, or an energy-loss bound. The weak reference and an intentionally
thick variant have regression checks for finite output and their differing
departure from Born. Thickness changes can introduce finer fringes and require
a fresh spectral-grid convergence check.

## Opt-in transported-current production scoring

`Validation: dtr-transported-current`

`Physics(transition_radiation="two-beam")` replaces the production PXR/CBS
line scorer with the experimental CPU transported-current solve. The default
remains `"off"`: no two-beam field is computed and existing default cache
identities are preserved. `"history"` and `"bunch"` retain their earlier
homogeneous-stack TR behavior. The new claims remain **unverified**;
passing the consistency checks below does not establish independent physics
validation or the published DTR/PXR anchor.

The first production scope is one reflection in one perfect crystal layer,
with homogeneous outer layers, planar faces, equal line/continuum trial counts
and an explicit node-sampled energy grid. Multiple crystal layers/reflections,
mosaicity, automatic/bin-mean line grids, secondary cascades, adaptive precision,
temporal output and electron-block transport are refused. A track leaving a
side face is refused during scoring. CUDA transport can supply rows, but this
new scorer is CPU-only; earlier homogeneous TR/CTR hardware parity does not
validate it. The supplied grid must resolve the requested spectrum.

```python
import numpy as np
import pyrite as pr

physics = pr.Physics(
    transition_radiation="two-beam",
    transition_lattice_origin_ang=(0.0, 0.0, 0.0),  # Å, entrance-frame lattice origin
    radiative_model="uncoupled",
    bremsstrahlung_model="eedl",
)
numerics = pr.Numerics(
    cxr_trials=3,
    brem_trials=3,
    cxr_reflections=pr.config.CxrReflections(max_reflections=1),
    energy_bins=pr.detectors.EnergyBins(
        cxr=np.linspace(500.0, 1500.0, 9),
        brem=np.array([500.0, 5000.0]),
    ),
)
result = pr.simulate(
    pr.Beam(energy_keV=30.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
    pr.geometry.Slab("hopg", thickness_ang=100.0, tilt_deg=30.0, footprint=None),
    pr.Detector(),
    physics=physics,
    numerics=numerics,
)
```

This is a small execution smoke, not a converged spectral prediction.
The origin defaults to zero only in the opt-in model; changing it changes the
case/content/dataset identities. The model marker is
`two-beam-transported-current-v1`. Live scoring, observation directions,
line recomputation and trajectory replay share the same scorer. History-mode
streaming adds whole-electron blocks with the original incident normalization;
coherent replay needs all histories at once.

### Finite transported charge current

For a row starting at $(\mathbf r_j,t_j)$ with velocity $\mathbf v_j$ and
duration $T_j$, Fourier-transform the point current
$\mathbf J=q\mathbf v_j\delta(\mathbf r-\mathbf r_j(t))$ and integrate the
outgoing modal Green function in time:

$$
\int_0^{T_j} dt\,\exp\{i[\omega(t_j+t)
-\boldsymbol\kappa_h\cdot(\mathbf r_{\perp j}+\mathbf v_{\perp j}t)
-h g_z(z_j+v_{zj}t)+\lambda_a(z_f-z_j-v_{zj}t)]\}.
$$

Here $h=0,1$ is the gauged source harmonic and $z_f$ the face toward which
mode $a$ decays or propagates. All lengths/clocks in this integral are inverse
eV. `dtr_history` divides the transported Å values by $\hbar c$; velocities
remain in units of $c$. Evaluating the difference of bounded endpoint
exponentials, with an `expm1`/equal-exponent limit, avoids growing propagators
and supports horizontal rows without dividing by $v_z$. Each physical exterior
continuation supplies its vacuum own field at the corresponding outer face.
Internal transport subdivisions never create a boundary source. Histories
are grouped before squaring; a numerical substep is not a new emitter.

### CBS phase in the same Maxwell basis

`Validation: line-tr-phase-reconciliation`

This is an author derivation from the variation of a point-charge current and
the relativistic lattice force, using the potential convention of Feranchuk
Eq. (14), as implemented in `materials.crystal.U_g`. For electron potential
energy $-U_{\mathbf g}e^{-i\mathbf g\cdot(\mathbf r-\mathbf r_0)}$, the
locally straight first-order displacement is

$$
\delta\mathbf v=\frac{U_{\mathbf g}/m}{\gamma\,\mathbf v\cdot\mathbf g}
[\mathbf g-\mathbf v(\mathbf v\cdot\mathbf g)],\qquad
\delta\mathbf r=\frac{i\delta\mathbf v}{\mathbf v\cdot\mathbf g}.
$$

The row's oscillating factor $e^{-i\mathbf g\cdot(\mathbf r-\mathbf r_0)}$
multiplies both. Varying $q\mathbf v\delta(\mathbf r-\mathbf r(t))$ gives
$\delta\mathbf J=\partial_t\mathbf P+\nabla\times\mathbf M$, where
$\mathbf P=q\delta\mathbf r\delta(\mathbf r-\mathbf r(t))$ and
$\mathbf M=\mathbf P\times\mathbf v$. Use
$\mathbf D=\mathcal E\mathbf E+4\pi\mathbf P$ and
$\mathbf B=\mathbf H+4\pi\mathbf M$ in the same Maxwell system above.
In addition to the charge-current drive, its tangential source is

$$
4\pi\begin{pmatrix}
-iK_x\mathcal E^{-1}P_z+i\omega M_y\\
-iK_y\mathcal E^{-1}P_z-i\omega M_x\\
-iK_xM_z-i\omega P_y\\
-iK_yM_z+i\omega P_x
\end{pmatrix}.
$$

The longitudinal polarization contact term is retained. The reverse harmonic
uses $-\mathbf g$ and its own potential and susceptibility, with the opposite
termination phase. The lattice-force perturbation is local to each straight
transport piece; weak deflection by the periodic potential is assumed.
$\mathbf v\cdot\mathbf g=0$ is refused rather than treated as channeling.

In vacuum at $\omega(1-\hat{\mathbf n}\cdot\mathbf v)=\mathbf v\cdot\mathbf g$,
the effective transverse CBS current is $-q\mathbf A_{\rm CBS}$, while the
first-Born polarization current is $-q\mathbf A_{\rm PXR}$. Both therefore
produce the same outgoing-field conversion,

$$
\mathbf E_{\rm out}=\frac{2\pi q}{n_z}
(\mathbf A_{\rm PXR}+\mathbf A_{\rm CBS})
\int dt\,e^{i[\omega t-(\mathbf k+\mathbf g)\cdot\mathbf r+\mathbf g\cdot\mathbf r_0
+k_z z_{\rm face}]}.
$$

This fixes the common phase and their relative sign without fitting an
intensity. Away from that vacuum/on-resonance limit, the finite-current
Maxwell solve supplies dispersion, absorption, diffraction and polarization;
the old approximate line kernel is not added again. Tests compare the complex
vector against the relativistic Feranchuk sum in its weak-coupling limit.

### Signed production component accounting

Let $C$ be the coupled finite-current field (charge and CBS), $B$ the uncoupled
charge field of the same rows, and $T$ the existing physical-crossing TR field.
The scored joint field is $S=T+C-B$. Subtracting $B$ removes the ordinary
finite-current acceleration radiation before combining with the separately
scored continuum and the existing TR baseline. This is the transported-current
diffraction correction to that baseline; ordinary TR still has its earlier
locally straight crossing assumption. Diffraction of the transported
acceleration field remains in $C-B$.

With $a=E n_z^2/(16\pi^4)$, store `spec_transition` as
$a\langle|T|^2\rangle$ and `spec` as
$a[\langle|S|^2\rangle-\langle|T|^2\rangle]$. Thus the line slot can be
negative: it contains signed interference and extinction, not an independent
positive photon source. Adding the two components gives the joint density.
`include_transition=False` removes the baseline only, not all boundary terms;
a physical disabled comparison requires rerunning with the Physics option off.
Checkpoint round trips preserve the signed arrays and selected model.

The coherent companion applies the existing distinct-pair population estimator
to the joint per-history Cartesian fields, including their sampled space/time
offsets. It subtracts the same history-TR baseline, so adding the one stored
`spec_transition` component gives the coherent joint density. An analytic
offset factor is not applied again. Undersampled negative pair estimates warn
and remain unclipped. Neither this path nor its consistency tests provide a
TR energy-loss/recoil bound, a published dynamical extinction anchor, or
independent sign-off. Those requirements remain open in #103/#581.
