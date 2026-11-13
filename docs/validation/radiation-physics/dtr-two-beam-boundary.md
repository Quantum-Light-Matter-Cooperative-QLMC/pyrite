# Two-harmonic Maxwell boundary field

Claim: `dtr-two-beam-boundary`. Independent verifier, 2026-11-12;
implementation base `ef31da0b4`. Target: `rederived`, never human sign-off.
This page inspects independent evidence, not a production-use prescription.

## Independence and source

The independent derivation below was written before reading implementation
bodies, tests, or the implementation-context DTR explanation. Inputs were only
the ledger record and AST-extracted signatures/docstrings. The signature is
`two_beam_straight_field(photon_energy_eV, direction, layers, substrate_eps,
velocity, time_ang, position_ang, *, crystal_layer, reciprocal_vector_ang,
chi_transfer, chi_reverse, lattice_origin_ang=(0, 0, 0))`.
It promises outgoing Cartesian complex fields in inverse eV for a globally
straight positive charge, with two transverse harmonics in one crystal layer.

Source equations are Maxwell's curl laws with local scalar relative
permittivity and unit permeability. Michael Fowler's
[Plane Electromagnetic Waves, Maxwell and planar-interface sections](https://galileoandeinstein.phys.virginia.edu/Elec_Mag/2022_Lectures/EM_46_Plane_Electromagnetic_Waves.html)
explicitly supplies the negative temporal-frequency convention, curl signs,
Poynting direction and continuity of tangential electric/magnetic fields.
The current term is the Maxwell–Ampère law; Gaussian units give the factor
$4\pi$ below. These equations, rather than a numbered Caticha formula, are the
starting point. [Caticha, Physical Review A **40**, 4322–4329 (1989)](https://doi.org/10.1103/PhysRevA.40.4322)
is formulation context only: its publisher abstract describes diffraction of
the electron field and arbitrary surface/crystal orientation. No paper equation,
figure, measured intensity or extinction law is being transcribed or certified.

## Independent derivation, recorded before implementation inspection

Use Gaussian natural units with $c=\hbar=1$, charge $q=+\sqrt{\alpha}$,
and time convention $\exp(-i\omega t)$. Define transverse Fourier amplitudes
with inverse transform measure $d\omega\,d^2\mathbf{k}_{\perp}/(2\pi)^3$.
The frequency-domain equations are

$$
\nabla\times\mathbf{E}=i\omega\mathbf{H},\qquad
\nabla\times\mathbf{H}=-i\omega\epsilon\mathbf{E}+4\pi\mathbf{J}.
$$

Let the observed harmonic have transverse momentum
$\mathbf{k}_{0\perp}=\omega\mathbf{n}_{\perp}$, and the second have
$\mathbf{k}_{1\perp}=\mathbf{k}_{0\perp}+\mathbf{g}_{\perp}$.
For a dielectric transfer harmonic proportional to
$\chi_t\exp[-i\mathbf{g}\cdot(\mathbf{r}-\mathbf{o})]$, the coefficient
mapping harmonic 1 to 0 is $\chi_t\exp(i\mathbf{g}\cdot\mathbf{o})$.
The reverse coefficient is $\chi_r\exp(-i\mathbf{g}\cdot\mathbf{o})$.
They need not be conjugates in an absorbing medium. Define

$$
\mathcal{E}=\begin{pmatrix}
\epsilon_b&\chi_t e^{i\mathbf{g}\cdot\mathbf{o}}\\
\chi_r e^{-i\mathbf{g}\cdot\mathbf{o}}&\epsilon_b
\end{pmatrix},\qquad
K_x=\operatorname{diag}(k_{0x},k_{1x}),\quad
K_y=\operatorname{diag}(k_{0y},k_{1y}).
$$

For each Cartesian component write the physical second harmonic as
$E_1(z)=e^{ig_z z}Y_1(z)$ (and likewise for $H$).
Thus the two-component tangential vectors form eight unknowns
$Y=(E_x,E_y,H_x,H_y)^T$. Before this longitudinal gauge shift, normal-component
elimination gives

$$
E_z=\mathcal{E}^{-1}(K_yH_x-K_xH_y)/\omega
-\frac{4\pi i}{\omega}\mathcal{E}^{-1}J_z,
\qquad H_z=(K_xE_y-K_yE_x)/\omega.
$$

The first term alone is insufficient for a driven charge: the second is the
longitudinal contact term. Substituting the two normal fields into the four
remaining curl equations gives

$$
\begin{aligned}
\partial_z E_x&=iK_xE_z+i\omega H_y,\\
\partial_z E_y&=iK_yE_z-i\omega H_x,\\
\partial_z H_x&=iK_xH_z-i\omega\mathcal{E}E_y+4\pi J_y,\\
\partial_z H_y&=iK_yH_z+i\omega\mathcal{E}E_x-4\pi J_x.
\end{aligned}
$$

Hence $\partial_zY=iMY+S$, where the ungauged matrix and drive are

$$
M_0=\begin{pmatrix}
0&0&K_x\mathcal{E}^{-1}K_y/\omega&\omega I-K_x\mathcal{E}^{-1}K_x/\omega\\
0&0&K_y\mathcal{E}^{-1}K_y/\omega-\omega I&-K_y\mathcal{E}^{-1}K_x/\omega\\
-K_xK_y/\omega&K_x^2/\omega-\omega\mathcal{E}&0&0\\
\omega\mathcal{E}-K_y^2/\omega&K_yK_x/\omega&0&0
\end{pmatrix},
\quad
S=4\pi\begin{pmatrix}
K_x\mathcal{E}^{-1}J_z/\omega\\
K_y\mathcal{E}^{-1}J_z/\omega\\J_y\\-J_x
\end{pmatrix}.
$$

The constant gauged generator is $M=M_0-g_zP_1$, with $P_1$ the projector
onto all four second-harmonic entries. Each physical second-harmonic source
also receives $e^{-ig_z z}$. Matrix order matters because $K_x,K_y$ do not
commute with $\mathcal{E}^{-1}$.

A straight trajectory through $(t_0,\mathbf{r}_0)$ has
$\mathbf{r}(t)=\mathbf{r}_0+\mathbf{v}(t-t_0)$ and $v_z\ne0$. Its transverse
Fourier current is

$$
\mathbf{J}_h(z)=\frac{q\mathbf{v}}{|v_z|}
\exp\left[i\omega t_0-i\mathbf{k}_{h\perp}\cdot\mathbf{r}_{0\perp}
+i p_h(z-z_0)\right],\qquad
p_h=\frac{\omega-\mathbf{k}_{h\perp}\cdot\mathbf{v}_{\perp}}{v_z}.
$$

The absolute value is the delta-function Jacobian; the signed denominator in
$p_h$ retains the passage sense. In a homogeneous medium put
$\mathbf{Q}_h=(\mathbf{k}_{h\perp},p_h)$,
$\rho_h=q e^{i\omega t_0-i\mathbf{k}_{h\perp}\cdot\mathbf{r}_{0\perp}-ip_hz_0}/|v_z|$.
Since $\mathbf{Q}_h\cdot\mathbf{v}=\omega$, the particular charge field is

$$
\mathbf{E}_{h,p}(z)=
\frac{4\pi i\rho_h[\omega\mathbf{v}-\mathbf{Q}_h/\epsilon_b]}
{\mathbf{Q}_h^2-\epsilon_b\omega^2}e^{ip_hz},\qquad
\mathbf{H}_{h,p}=\mathbf{Q}_h\times\mathbf{E}_{h,p}/\omega.
$$

In each ordinary layer, add upward and downward free fields to this particular
solution and match tangential fields at every face. Restricted to the crystal,
that background has precisely three exponentials: own charge, upward wave,
downward wave. If $Y_b$ solves the uncoupled driven problem with generator
$M_b$, its coupled correction satisfies

$$
\partial_z\delta Y=iM\delta Y+i(M-M_b)Y_b+(S-S_b).
$$

The final term includes the longitudinal contact term from the changed inverse
permittivity. The two charge harmonics are independent linear source rows;
solve both with the same operator and add their complex observed fields.

For a mode $Mv_a=\lambda_av_a$, variation of constants for a drive
$S(z)=s\exp[ip(z-z_s)]$ gives

$$
f_a(z)=\int_{z_a}^{z}e^{i\lambda_a(z-u)}
(V^{-1}s)_a e^{ip(u-z_s)}\,du.
$$

Choose $z_a$ at the lower face for upward modes and upper face for downward
modes. The latter integral reverses orientation and has a minus sign. The
coincident-exponent limit is the interval length times the common exponential;
no physical singularity occurs at $p=\lambda_a$. Outgoing modes have
$\operatorname{Im}\lambda_a>0$ upward and $<0$ downward; real modes use the
sign of $\operatorname{Re}(E_xH_y^*-E_yH_x^*)$. Anchoring modes at their
originating face keeps passive propagation factors bounded. Match the total
particular plus modal tangential fields, exclude incoming exterior modes,
and reconstruct the observed vacuum normal field as
$E_z=-\mathbf{n}_{\perp}\cdot\mathbf{E}_{\perp}/n_z$.

This boundary solve keeps every power of the retained two-harmonic coupling.
Expanding its resolvent once gives the first-Born scattered field; turning
both couplings off gives the homogeneous transition-radiation solution.
For separate observed and diffracted charge rows, $a=a_0+a_1$, so

$$
|a|^2=|a_0|^2+|a_1|^2+2\operatorname{Re}(a_0\cdot a_1^*).
$$

There is no freedom to fit their relative phase. Translating the trajectory
reference point along the same worldline leaves its current phase unchanged.
Changing the termination origin changes the coupling phase, not the charge
normalization.

## Filters before implementation inspection

- Units: pass symbolically. With energy/inverse length as one unit, $M$,
  $\lambda$, $p$ and $g_z$ have inverse-length units; permittivity and velocity
  are dimensionless. The transverse-time Fourier electric and magnetic fields
  have inverse-energy units, current amplitude is dimensionless and
  $\partial_zY$ and $S$ are dimensionless. Angstrom inputs require division
  by $\hbar c$ before multiplying energies. Spectral density
  $E n_z^2|a|^2/(16\pi^4)$ has inverse-energy units; its absolute angular
  normalization is a caller convention, not a new Caticha intensity anchor.
- Limits: pass analytically for zero coupling, first derivative in transfer,
  coincident source/mode exponents and reference-point translation. Passive
  mode factors decay in their direction of travel. These are model limits,
  not a published thick-crystal extinction-law validation.
- Signs/conventions: pass for negative time frequency, transfer from harmonic
  1 to 0, gauge subtraction, signed propagation and positive charge. A scalar
  real-space permittivity may be complex; passivity of its retained harmonic
  matrix must be checked, not inferred from diagonal absorption alone.

## Implementation comparison and numerical evidence

The independent derivation above was durably recorded before opening
implementation bodies. The following comparison was appended afterward.

### Symbolic diff

After the preceding derivation was saved, implementation bodies were inspected.
Serena rejected the ignored worktree path; targeted AST extraction/direct reads
were used instead. The pre-inspection section above was not rewritten to match
the implementation.

| Boundary | Independent expression versus implementation | Result |
|---|---|---|
| `_maxwell` | All eight nonzero blocks have the same multiplication order, sign and frequency factor; odd diagonal entries subtract $g_z$ | pass |
| `_one_energy` | Off-diagonal permittivities have opposite origin phases; current correction uses $\mathcal{E}^{-1}-I/\epsilon_b$ | pass |
| Current contact | $J_z=q\operatorname{sgn}(v_z)$ times the current phase, giving the implemented $4\pi q\operatorname{sgn}(v_z)/\omega$ | pass |
| `_background` | Own particular field plus two homogeneous waves; harmonic gauge shifts each exponent and face phase | pass |
| `_source_faces` | Modal variation of constants with negative lower-face contribution; `_integral` is the exact divided exponential with its removable limit | pass |
| `_boundary_response` | Upper-minus-lower tangential continuity; exterior modes outgoing only; source jumps are $-f_{\rm low}$ and $+f_{\rm high}$ | pass |
| `TwoBeamField` | Sum complex source rows before squaring; output vacuum normal component enforces transversality | pass |

The own-field helper `_particular_field` agrees with the independently derived
Gaussian charge field, including $4\pi i q/|v_z|$ and the longitudinal
$\mathbf{Q}/\epsilon_b$ term. `_stack` converts reciprocal Angstroms to eV
by multiplying by `HBARC_EV_ANG`, and layer thicknesses to inverse eV;
`_background` converts time/position phases using the same constant. No extra
cell-volume, atom-count or polarization average appears. The two-harmonic
matrix is a finite Fourier truncation of a local scalar dielectric; omitted
harmonics are outside this claim. Invertibility and a separable modal basis
are implicit numerical requirements; exact grazing/degenerate eigenmodes and
singular permittivities are not certified by this comparison.

### Independent numerical comparison

A separate scratch oracle builds the driven generator by applying the four curl
component equations to eight basis vectors, including the normal-current term.
It does not call `_maxwell`, `_background`, `_modes`, `_source_faces`, `_integral`,
`_free_system` or `_particular_field` to construct its reference field.
For a homogeneous crystal of thickness $d$, it uses the direct propagator
$P=\exp(iMd)$ and a particular solution
$b=[i(pI-M)]^{-1}S$. Analytic isotropic exterior plane waves give matrices
$V_-$ and $V_+$. For each charge harmonic the outgoing coefficients solve

$$
[-PV_-,V_+]\begin{pmatrix}c_-\\c_+\end{pmatrix}
=P Y_{p,\rm vac}(0)-Y_{p,\rm vac}(d)
+b e^{ipd}-Pb.
$$

This tests the total forced boundary problem directly, independently of the
implementation's uncoupled-background correction and bounded modal integrals.
A thin slab is deliberate: direct transfer matrices are ill-conditioned in
opaque stacks, where the implementation's bounded formulation is preferable.

Parameters: photon energy $1000$ eV; direction
$(0.6,0.2,\sqrt{0.6})$; crystal permittivity $0.995+0.0001i$;
thickness $80$ Angstrom; vacuum substrate and emitting region; velocity
$(0.25,0.1,\pm0.8)$; reciprocal vector $(0.015,-0.01,0.04)$ inverse Angstrom;
trajectory point $(7,-4,11)$ Angstrom at time $19$ Angstrom; origin
$(2,1,-3)$ Angstrom. Both source rows and all three Cartesian components are
compared. Coupling cases are zero, one-way
$\chi_t=10^{-4}+2\times10^{-5}i$, and the same transfer with
$\chi_r=8\times10^{-5}-10^{-5}i$.

| Passage sign | Coupling | Relative field norm error | Maximum absolute component error, inverse eV |
|---|---|---|---|
| negative | zero | $1.20\times10^{-13}$ | $3.54\times10^{-19}$ |
| negative | one-way | $1.20\times10^{-13}$ | $3.54\times10^{-19}$ |
| negative | two-way | $9.15\times10^{-14}$ | $2.69\times10^{-19}$ |
| positive | zero | $5.36\times10^{-14}$ | $2.83\times10^{-19}$ |
| positive | one-way | $5.36\times10^{-14}$ | $2.83\times10^{-19}$ |
| positive | two-way | $3.20\times10^{-14}$ | $2.16\times10^{-19}$ |

The independently assembled generator differs from the implemented one by at
most $2.28\times10^{-13}$ eV in any entry. Probe assertions used relative
$10^{-8}$ plus absolute $10^{-13}$ inverse eV: an intentionally loose FP64
algebra-comparison gate chosen before execution, not source uncertainty or an
accuracy claim. Actual agreement is much tighter. A later retained regression
could tighten this after conditioning studies; this scratch probe alone is not
an `anchored` regression.

### Evidence and required filters

- Notation/normalization: pass. Ledger, its generated `domain-inventories.md` row, public docstring and `TwoBeamField`
  agree on the observed/diffracted source rows, complex interference, positive
  charge and inverse-eV Cartesian fields. `_maxwell` documents the homogeneous
  elimination; the longitudinal driven correction is explicit in `_one_energy`.
- Units at every boundary: pass. Source equations use Gaussian natural units;
  API energy is eV, coordinates/time are Angstrom with unit light speed,
  reciprocal vector inverse Angstrom, kernel wave numbers eV and kernel
  distances inverse eV. Output is inverse eV. There is no source-table loader
  for this claim.
- Reference data: not applicable. No tabulated value, measured datum or paper
  figure was transcribed. Source access confirms the Maxwell convention and
  planar matching; Caticha attribution is context only. The author-linked
  full paper could not be fetched, so no paper-equation agreement is asserted.
- Tolerances: pass for their stated purpose. Numerical tolerances cover FP64
  linear algebra; existing weak/reverse-coupling tests allow finite-order
  truncation, not experimental uncertainty. The 1% reverse-coupling ratio test
  explicitly permits the next powers of coupling; it is an internal limit.
- Extant evidence: pass within scope. Base commit
  `ef31da0b4181228cf0ea840843e1b42501c60196` and the named existing two-beam
  regression module are present. Scratch command and JSON are
  `scratch/issue-581/direct_maxwell.py` and
  `scratch/issue-581/direct-maxwell.json` in the verification worktree;
  durable equations, parameters and results are recorded above. No external
  code, published anchor, CUDA/FP32 parity or hardware artifact is cited as
  evidence for this claim.
- Markers: all six ledger-owned class/functions carry
  `Validation: dtr-two-beam-boundary`; the unmarked private boundary helper is
  not a missing ledger-owned marker.

Existing focused checks: `pyrite-dev test
 tests/montecarlo/spectrum/test_transition_radiation_two_beam.py
 tests/dev/test_docs.py` passed **41 tests** on 2026-11-12. These include zero
coupling in both senses, finite one-way Born agreement with caps/oblique
reciprocal vectors, repeated-scattering orders, source translation, passive
opacity, termination phase, transversality and broadcasting. They are internal
consistency checks and do not establish a published absolute intensity anchor.

Verdict: **rederived** for the stated straight-passage two-harmonic Maxwell
boundary claim. No divergent factor, sign, unit or convention was found.
The task owner may change this claim's status from `unverified` to `rederived`
and link this write-up. Published-data anchoring, an independently verified
extinction law, transported-current validity, CBS phase, recoil, mosaics,
production promotion and device parity remain outside this verdict.

Documentation verification: final `tests/dev/test_docs.py` rerun passed **9 tests**. Full `pyrite-dev docs` and an incremental Sphinx HTML build emitted
this page but returned nonzero for the single expected `toc.not_included`
warning: the verifier may not edit the domain index. Owner integration must add
`dtr-two-beam-boundary` to `docs/validation/radiation-physics/index.md` and link
this page from the ledger, then rerun the build. Inspection of the emitted final
HTML found **95 math expressions**, all in `math notranslate` elements, with
no residual dollar delimiters in the article. This navigation gap does not
alter the physics verdict.
