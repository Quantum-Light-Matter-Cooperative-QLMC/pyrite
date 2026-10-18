# Experimental kinematic diffracted transition radiation

`Validation: dtr-kinematic-boundary`

`montecarlo.spectrum.dtr` computes a first-Born **field diagnostic**, separately
from production PXR/CBS. It diffracts the homogeneous stack's free TR field
once, through one crystal harmonic. `pxr_straight_amplitude` diffracts the
charge's own field through the same harmonic on a straight passage. Their
vectors share a Maxwell normalization and reference plane and can be summed
before squaring. This is the first slice of #581. There is no production DTR
selector or reconciled production CBS field yet. A separate experimental
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
reconciliation against the Feranchuk line, layered coherent escape,
catalog/campaign integration and the two-beam limits/published reproduction.
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
exponentials. These are diagnostic checks only. Production Feranchuk/CBS
phase reconciliation, transported per-history integration, mosaicity,
normal catalog/campaign support, recoil/energy-loss justification, and a
published dynamical anchor remain open in #103/#581. This new diagnostic has
not been run on CUDA; the hardware evidence below concerns the existing
homogeneous TR/CTR implementation.
