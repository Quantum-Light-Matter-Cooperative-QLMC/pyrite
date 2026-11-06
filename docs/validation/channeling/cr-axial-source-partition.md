# `cr-axial-source-partition`: finite-grid transition ownership and on-axis mapping

**Status:** see the [ledger row](../ledger-channeling.md#cr-axial-source-partition). Task-owner derivation (2026-11-05), followed by the fresh-context verification below.
Code: `src/pyrite/montecarlo/channeling/axial_source_partition.py` and the attenuation
branch of `axial_qrt_source.on_axis_photon_spectrum`.

This claim covers a supplied finite global eigensystem and on-axis observation.
It does not establish a material source, a physical axial dissipator, off-axis
recoil, Hilbert-space convergence or standard-run admission; those stay under
`cr-axial-persisted-channel`.

## Transition ownership

Let $H=\sum_n E_n|n\rangle\langle n|$ and let the positive-frequency current be
$A_\lambda=\sum_{E_i-E_f>\tau}a^\lambda_{fi}|f\rangle\langle i|$. For a declared
boundary $E_b$, let $Q_b$ project on states with $E_n<E_b$ and
$Q_u=1-Q_b$. Define

$$
A^\mathrm{ax}_\lambda=Q_bA_\lambda,\qquad A^\mathrm{ff}_\lambda=Q_uA_\lambda .
$$

These sum to $A_\lambda$. ADR-0018 assigns bound→bound and free→bound transitions
to channeling radiation and free→free transitions to coherent bremsstrahlung. A
lowering current has $E_f<E_i$, so a free final state implies a free initial state
and $Q_uA_\lambda=Q_uA_\lambda Q_u$: no bound→free block exists. The owner depends
only on the final state. A boundary inside a level with degeneracy tolerance $\tau$
has no basis-independent meaning and is refused.

## Interference

The observed correlation expands into both blocks and a cross term,

$$
G=G^\mathrm{ax,ax}+G^\mathrm{ff,ff}+G^\mathrm{ax,ff}+G^\mathrm{ff,ax}.
$$

For $z\ge z'$,
$G^{ab}_\lambda(z,z')=\mathrm{Tr}[A^a_\lambda e^{\mathcal L(z-z')}(\rho(z')A^{b\dagger}_\lambda)]$.
The operator $X=\rho A^{\mathrm{ff}\dagger}=\rho A^\dagger Q_u$ has its right index in
$Q_u$. Hamiltonian evolution in the eigenbasis only multiplies matrix elements
by phases, so it keeps that index in $Q_u$. Then

$$
\mathrm{Tr}[Q_bA\,e^{-iH\Delta/\hbar v}X\,e^{iH\Delta/\hbar v}]
=\mathrm{Tr}[A\,(\cdots)\,Q_uQ_b]=0
$$

by cyclicity. The term $G^{\mathrm{ff,ax}}$ vanishes by the same right-index argument with the blocks swapped, and the
two are related by $G^{ab}(z,z')^*=G^{ba}(z',z)$. So the cross term vanishes under unitary
evolution, for any incident density and at every photon energy.

Jumps can break this through two routes. For $Q,Q'$ the two block projectors,
$G^{\mathrm{ax,ff}}$ can be nonzero only if some gain $Q_bL_rQ_u$ or the no-jump coupling
$Q_u(\sum_rL_r^\dagger L_r)Q_b$ is nonzero, and swapping $b$ and $u$ gives the condition for
$G^{\mathrm{ff,ax}}$. The condition is necessary, not sufficient: a rank-one $|a\rangle\langle f|$
satisfies it and still gives zero, as does a fan-out jump that only creates coherence.
The implementation therefore computes the total from the unsplit current and reports
the defect instead of assuming zero. Capture and bound→bound share final states, so
their mutual cross term survives Hamiltonian evolution and this result does not cover
any later split of the axial block.

## Line shape

The two limits below follow directly from the GKSL generator and the source form
$d^2N/dE\,d\Omega=\alpha E\,(4\pi^2h_v^2)^{-1}\sum_\lambda
\iint dz\,dz'\,e^{iED(z-z')/h_v}G_\lambda(z,z')$, with $h_v=\hbar v$ and
$D=1-\beta$. Let $\delta=(ED-\Delta)/h_v$, with $\Delta$ the Bohr gap.

For excited-state decay at rate $r$ with incident weight in the upper level,
$G=|a|^2e^{-i\Delta(z-z')/h_v}e^{-r(z+z')/2}$. The double integral factorizes:

$$
\left|\frac{1-e^{(i\delta-r/2)\ell}}{r/2-i\delta}\right|^2 .
$$

For pure dephasing of the upper level at rate $r$, $L=\sqrt r|e\rangle\langle e|$,
the population stays constant and
$G=|a|^2e^{-i\Delta(z-z')/h_v}e^{-r|z-z'|/2}$. The result depends on $|z-z'|$ and
is the finite-path Lorentzian of half-width $r/2$ in $\delta$:

$$
\int_0^\ell\!\!\int_0^\ell dz\,dz'\cos[\delta(z-z')]e^{-r|z-z'|/2}.
$$

## Attenuation and on-axis geometry

Beer–Lambert intensity attenuation $e^{-\mu L}$ acts on the photon amplitude as
$e^{-\mu L/2}$, at both depths of the two-time form. On resonance without damping,

$$
\left|\int_0^\ell e^{-\mu(\ell-z)/2}\,dz\right|^2
=\left(\frac{1-e^{-\mu\ell/2}}{\mu/2}\right)^2,
$$

which differs from the intensity-level form $\ell(1-e^{-\mu\ell})/\mu$. Zero
absorption reproduces the unattenuated result exactly.

With inward unit normal $\hat n$, axis $\hat a$ and $\cos\alpha=\hat a\cdot\hat n>0$,
axial depth $z$ lies at normal depth $z\cos\alpha$. A forward photon along
$\hat a$ reaches the exit plane of a slab of thickness $T$ after
$L=(T-z\cos\alpha)/\cos\alpha=T/\cos\alpha-z$. Off-axis rays are refused because
the photon recoil phase changes the Bloch sector. The slab is laterally infinite.

Summing both transverse polarizations is $\sum_\lambda A_\lambda^\dagger A_\lambda$,
which is invariant under a real rotation of the transverse basis. The
unpolarized on-axis yield therefore needs no azimuth mapping.

## Omitted yield of a truncated subspace

At long path, a stationary incident density gives the secular directional rate
per depth

$$
\frac{dN}{dz\,d\Omega}=\frac{\alpha}{2\pi h_vD^2}
\sum_\lambda\sum_{f,i,i'}^{E_i=E_{i'}}
\rho_{ii'}\,a^\lambda_{fi}\,a^{\lambda*}_{fi'}\,(E_i-E_f).
$$

This is the golden-rule form $\alpha E_0|a|^2/(2\pi h_vD)$ with $E_0=(E_i-E_f)/D$.
Dropping coherence between nondegenerate levels needs
$\theta=|E_i-E_{i'}|\ell/h_v\gg1$, plus a splitting large against any scattering linewidth.
For equal weights and full coherence the true rate is $1\pm\sin\theta/\theta$ times the
secular one, so unresolved coherent pairs are refused. The linewidth condition is not
checked. The omitted rate for a retained set $S$ is the full-grid rate minus the rate with
$i,i',f\in S$. $S$ must hold whole multiplets, or the result depends on the basis
chosen inside a multiplet. The result bounds omission only within the represented
grid. Grid, supercell and Bloch-sector refinement stay outside this claim.

## What native scoring owns

Free→free stays owned by CBS under ADR-0018 and is never scored by the axial source.
A straight mean zone row exactly along the axis has $\mathbf v\cdot\mathbf g=0$ for
zero-order $\mathbf g$, so its native resonance energy vanishes and it emits no such
line. That is a property of the straight row, which discards transverse momentum. It
is not a property of the block: each represented free→free transition has $E_i-E_f>0$.
In the plane-wave limit that gap equals $\hbar\bar{\mathbf v}\cdot\mathbf g$ and the element matches the
native Born amplitude; dressed states depart from both.

Native output therefore depends on the row direction. It vanishes for zero-order
$\mathbf g$ on the exact axis and reappears at tilted handoff directions, above a
numerical resonance-root threshold. A production adapter must reconcile the two
before combining them: treating the block as absent from native scoring can double count after handoff,
and treating it as native-scored can omit it on the zone row. Nonzero axial-order
$\mathbf g$ and incoherent bremsstrahlung stay native, following the ADR-0018 convention
of passing straight mean segments to native scorers.

## Fresh-context verification (2026-11-05)

Commit `9f8b9452c`. Order followed: ledger row and ADR-0018, then the regression
cross-term derivation and a dense oracle that imports nothing from PyRITE, with
the closed forms below derived on paper. Only then the task-owner text above,
the implementation bodies and the tests. The numerical scripts are scratch files
and are not committed.

**Verdict: `discrepancy`, confined to the native-ownership statement.** The
equations of the claim (current split, Hamiltonian cross term, line shapes,
attenuation, path, secular prefactor) are rederived and match the code. First
divergent term: the Assumptions field and "What native scoring owns" evaluate
$\mathbf v\cdot\mathbf g$ at the axis direction, while each represented free→free
transition carries $\hbar\bar{\mathbf v}\cdot\mathbf g=E_i-E_f>0$ with the
transition's own transverse velocity. No code path consumes that statement, so no
number changes; the partition and the ledger contract do not agree until it is
corrected.

### Independent derivation

**Split.** With $Q_b$, $Q_u$ spectral projectors of $H$ and $A=\sum_{E_i-E_f>\tau}a_{fi}|f\rangle\langle i|$,
$Q_bA+Q_uA=A$. For $f\in Q_u$ and $E_i>E_f+\tau\ge E_f$ the initial state is in
$Q_u$, so $Q_uAQ_b=0$. A cut inside a level leaves the cross term zero (any
projector on eigenvectors commutes with $H$) but rotating the level changes
$\lVert Q_bA\rVert$: oracle values 3.972 and 3.952 at a 0.7 rad rotation. The
refusal reason is therefore basis dependence, as written.

**Cross term.** For $z\ge z'$ and $\mathcal U(X)=UXU^\dagger$ with $[U,Q_{b,u}]=0$,

$$
G^{\mathrm{ax,ff}}(z,z')
=\mathrm{Tr}[Q_bA\,U\rho A^\dagger Q_uU^\dagger]
=\mathrm{Tr}[A\,U\rho A^\dagger\,Q_uQ_b\,U^\dagger]=0 ,
$$

and the same with $b\leftrightarrow u$, for any $\rho$. The two orderings at fixed
$z\ge z'$ are separate functions. Conjugation gives
$G^{ab}(z,z')^*=G^{ba}(z',z)$, not $G^{ba}(z,z')$.

With jumps, let $X$ satisfy $XQ_b=0$. The Hamiltonian term keeps this. The
remaining terms give $(L X L^\dagger)Q_b=LXQ_u(Q_bLQ_u)^\dagger$ and
$XL^\dagger LQ_b=X\,Q_uL^\dagger LQ_b$. Hence
$G^{\mathrm{ax,ff}}\equiv0$ at all orders whenever

$$
Q_bL_rQ_u=0\ \ \forall r
\quad\text{and}\quad
Q_u\Big(\sum_rL_r^\dagger L_r\Big)Q_b=0 ,
$$

and $G^{\mathrm{ff,ax}}\equiv0$ under the same condition with $b\leftrightarrow u$.
This is necessary for a nonzero term, not sufficient: a rank-one $L=|a\rangle\langle f|$
passes it and still gives zero, because its image of $X$ is diagonal and $A$ has
no diagonal.

**Line shapes.** For $L=\sqrt r|g\rangle\langle e|$, $\rho_{ee}(z')=e^{-rz'}$ and
$|e\rangle\langle g|$ decays at $r/2$, so $G=|a|^2e^{-i\Delta(z-z')/h_v}e^{-r(z+z')/2}$.
For $L=\sqrt r|e\rangle\langle e|$ the sandwich on $|e\rangle\langle g|$ vanishes and
the anticommutator gives decay $r/2$. Closed form of the dephasing double integral,
with $q=r/2-i\delta$:

$$
\int_0^\ell\!\!\int_0^\ell dz\,dz'\,e^{i\delta(z-z')}e^{-r|z-z'|/2}
=2\,\mathrm{Re}\left[\frac{\ell}{q}-\frac{1-e^{-q\ell}}{q^2}\right].
$$

**Attenuation and path.** Field transmission $e^{-\mu L(z)/2}$ gives
$A(z)A(z')$ in the two-time form. Ray tracing to the plane $\hat n\cdot\mathbf r=T$
gives $L=(T-z\cos\alpha)/\cos\alpha$.

**Secular rate.** Starting from
$d^2N/dE\,d\Omega=\alpha E\,(4\pi^2h_v^2)^{-1}\lvert\int dz\,a\,e^{i(ED-\Delta)z/h_v}\rvert^2$,
use $\lvert\int_0^\ell\rvert^2\to2\pi\ell\,\delta[(ED-\Delta)/h_v]=2\pi\ell\,(h_v/D)\,\delta(E-\Delta/D)$.
Then

$$
\frac{dN}{dz\,d\Omega}=\frac{\alpha}{4\pi^2h_v^2}\,\frac{\Delta}{D}\,2\pi\frac{h_v}{D}\lvert a\rvert^2
=\frac{\alpha\,\Delta\,\lvert a\rvert^2}{2\pi h_vD^2}.
$$

A cross term between two lines needs the same final state because the trace closes
on $f$, so it needs $E_i=E_{i'}$. For two lines split by $\epsilon$, with equal
weights, full coherence and fixed path, the frequency-integrated ratio to the secular rate is

$$
1\pm\frac{\sin\theta}{\theta},\qquad \theta=\frac{\epsilon\,\ell}{h_v}.
$$

### Comparison with the implementation

| Item | Check | Result |
| --- | --- | --- |
| Split | mask axis, complement, $Q_uAQ_b$, refusal `gap <= bohr_tolerance_eV` | matches |
| Cross term, Hamiltonian | oracle and `partition_on_axis_spectra`, random coherent $\rho$, 6 levels | $\lvert$defect$\rvert$/total $=1.5\times10^{-16}$ |
| GKSL and ordering | column-major `_generator`; `values[later, earlier]`; $v_iG_{ij}v_j^*$ | matches the stated convention |
| Decay | closed form, $r=4\times10^{-3}$ and $10^{-7}$ per Å | $4.2\times10^{-6}$, $1.1\times10^{-10}$ |
| Dephasing | closed form above, $r=6\times10^{-3}$ and $10^{-7}$ per Å | $6.3\times10^{-7}$, $2.4\times10^{-8}$ |
| $r\to0$ | finite-path $\mathrm{sinc}^2$ | $1.1\times10^{-10}$ |
| Rate discrimination | decay amplitude at $r$ instead of $r/2$, on resonance | ratio 0.42 to the code, so a wrong factor fails |
| Dephasing convention | $L=\sqrt r\,\sigma_z$ gives width $2r$ | $6.3\times10^{-5}$ |
| Attenuation | off resonance, tilted $\alpha=30°$, zone shorter than exit path | $1.2\times10^{-7}$ |
| Attenuation and decay | $\lvert(e^{x\ell}-1)/x\rvert^2$, $x=i\delta+\mu/2-r/2$ | $1.2\times10^{-7}$ |
| Exit path | independent ray trace | $5.7\times10^{-14}$ Å |
| Secular rate | hand sum; rest dipole with $\gamma^2(1+\beta)^2$ Jacobian reduces algebraically to the code | $2\times10^{-16}$ |
| Full retained set | `omitted_rate` | exactly 0 |

The test oracles for decay, dephasing and attenuation are valid. The test's
`dblquad` result is confirmed by the closed form above, and its rates are correct,
including the $r/2$. They share a retyped prefactor
$\alpha E/(4\pi^2h_v^2)$ with the code, so they are independent in line shape only;
the prefactor is pinned separately by the rest-dipole oracle, which is independent.

### Jump classes

Implementation, 6 levels, $E_b=-2$ eV, random coherent $\rho$, seven photon energies.

| Generator | Largest $\lvert$defect$\rvert$/total | Comment |
| --- | --- | --- |
| None | $1.5\times10^{-16}$ | |
| Diagonal dephasing | $1.7\times10^{-16}$ | commutes with $Q$ |
| All rank-one population transfers | $4.1\times10^{-16}$ | includes free→bound |
| Rank-one $\lvert a\rangle\langle f\rvert$ | $8.3\times10^{-17}$ | necessary condition met, still zero |
| Fan-out $\lvert a\rangle\langle c\rvert+\lvert f\rangle\langle c\rvert$ | $4.4\times10^{-16}$ | creates $\rho_{af}$, still zero |
| $\lvert a\rangle\langle f\rvert+\lvert c\rangle\langle d\rvert$ (the anchor's class) | $7.6\times10^{-4}$ | right index $Q_u\to Q_b$ |
| $\lvert f\rangle\langle a\rvert+\lvert c\rangle\langle d\rvert$ | $1.8\times10^{-3}$, positive | $Q_b\to Q_u$; $G^{\mathrm{ax,ff}}=0$, $G^{\mathrm{ff,ax}}\ne0$ |
| $\lvert c\rangle\langle a\rvert+\lvert c\rangle\langle f\rvert$ | $3.5\times10^{-3}$, negative | only the no-jump term reaches $G^{\mathrm{ax,ff}}$ |

### Native CBS against the free→free block

Plane-wave limit, $k_f=k_i-g$. First-order dressing gives

$$
E_i-E_f=\frac{\hbar^2(k_i^2-k_f^2)}{2\gamma m}=\hbar\,\bar{\mathbf v}\cdot\mathbf g,
\qquad
\lvert a\rvert=\frac{\lvert U_{\mathbf g}\rvert}{mc^2}\,
\frac{g}{\gamma\,\bar{\mathbf v}\cdot\mathbf g},
\qquad
\bar{\mathbf v}=\frac{\hbar(\mathbf k_i+\mathbf k_f)}{2\gamma m}.
$$

Two consequences follow.

1. $\bar{\mathbf v}\cdot\mathbf g=0$ forces zero gap, which the lowering current
   excludes. So in this limit no element of the block has vanishing
   $\bar{\mathbf v}\cdot\mathbf g$, even with an exactly aligned cell and no tilt.
2. The native peak
   $\alpha\omega/(4\pi^2\hbar c)\,\lvert A\rvert^2t_L^2$ with $t_L=\ell/\beta$
   equals $\alpha E\lvert A\rvert^2\ell^2/(4\pi^2h_v^2)$, the axial prefactor, and
   $\lvert A_{\rm CBS}\rvert=(U/m)\,g/(\gamma\mathbf v\cdot\mathbf g)$ at leading order. The block is
   native CBS, as ADR-0018 says.

Si, 5 MeV, $(2\,2\,0)$, $U_{\mathbf g}=4.32$ eV, `mc_cxr_spectrum` on one straight
segment, $\hat n=\hat z$, $k_i=3g$, $k_f=2g$, $\psi=\bar k/p=2.94$ mrad:

| Quantity | Value |
| --- | --- |
| Dressed element / Born, $U/\Delta=2\times10^{-4}$, 0.0046, 0.046, 0.23 | 1.00000, 0.99999, 0.99858, 0.96366 |
| Native CBS component / Born prefactor formula, $\ell=10^2$, $10^3$, $10^4$ Å | 0.99565, 0.99193, 0.95565 |
| Line position, axial $\Delta/D$ / native in-medium | 4390.2 / 4359.65 eV |

The $\ell$ trend is the native escape factor. The $-0.43\%$ residual matches the
bracket correction $\omega\psi/g\approx0.2\%$ in amplitude.

Native output against direction, same reflection:

| $\psi$ | Native output | Controlling term |
| --- | --- | --- |
| $0$ | exactly zero, 11 eV to 60 keV | `omega = v_dot_g / resonance_denom` is zero; gate `resonance_eV <= F32_TEN` |
| 0.05 to 0.7 mrad | zero | three-pass root fails $\lvert\Delta D\rvert/D\le10^{-3}$: replayed values $0.249$, $0.0287$, $3.69\times10^{-3}$, $1.63\times10^{-3}$ at 0.3, 0.5, 0.7, 0.8 mrad (0.8 mrad replay only) |
| 0.9 mrad and above | line, 1264 eV at 0.9 mrad | replayed $7.95\times10^{-4}$, passes; replayed energies match the native peaks |

At $\psi=0$ the mixed-order $(1\,1\,1)$ gives a line at 0.528 MeV, which supports
the closing sentence of "What native scoring owns". $(0\,0\,4)$ gives zero because
$\mathbf g\parallel\mathbf v\parallel\hat n$ has no transverse amplitude. $(2\,2\,4)$
at 2.11 MeV also gave zero; that case was not diagnosed.

### Findings

**Blocking: native-ownership statement.** Ledger Assumptions, "What native scoring
owns".

- *What holds.* A zone row exactly along the axis produces no zero-order native
  line.
- *What does not.* "the free→free block has no native resonance ($\mathbf v\cdot\mathbf g=0$)"
  describes the row, not the block. The row lacks the line because a straight mean
  segment discards the transverse momentum that the free state carries.
- *Population error.* Handoff leaves along `on_shell_direction` with the sampled
  $k_\perp$, and any row with sufficient tilt has $\mathbf v\cdot\mathbf g\ne0$. Both
  get this line natively. A free state at $E_\perp=20$, 100, 300 eV sits at
  $\psi=2.7$, 6.0, 10.5 mrad, above the observed native threshold, for a
  $g$ aligned with its transverse motion. "Not a native contribution" therefore
  fails outside one direction, and a production adapter that relies on it can double
  count.
- *Mechanism.* "carry the denominator $\gamma(\mathbf v\cdot\mathbf g)$, so they have
  no resonance" is wrong. That denominator raises the amplitude as
  $\mathbf v\cdot\mathbf g\to0$. The absence comes from the resonance energy going to
  zero and the 10 eV gate, then from the root-convergence rejection up to about
  0.8 mrad here. The second is a numerical gate, not kinematics, and the text does
  not mention it.
- *ADR-0018.* The row cites ADR-0018 as source while contradicting its sentence
  that the PXR/CBS kernel already produces free→free, with no outcome note on the
  ADR. Owner assignment in code is compliant: free→free is never scored here. The
  repository also disagrees with itself: `AxialPartitionSpectra` speaks of "a native
  free→free score", and `global_axial_source` refuses a nonoverlapping-split
  assumption.

**Important: cross-term text.**

- "The other cross term is the conjugate of this one" is false at fixed $z\ge z'$.
  The Hamiltonian result survives by the repeated argument above. The bound→free
  jump row shows the two differ in the oracle: $\lvert G^{\mathrm{ax,ff}}\rvert=0$,
  $\lvert G^{\mathrm{ff,ax}}\rvert=0.086$.
- "A jump term can carry the right index from $Q_u$ to $Q_b$" omits the reverse
  direction and the no-jump term. "Final-state coherence between them" is also the
  wrong criterion: a fan-out jump creates that density coherence with zero defect.
  Replace it with the necessary condition above.
- The code measures the defect by subtraction, so its value is unaffected.

**Important: secular validity.** The prefactor and the multiplet handling are
correct, and dropping nondegenerate coherence is correct only for
$\lvert E_i-E_{i'}\rvert\,\ell/h_v\gg1$ and a splitting large against the jump
linewidth. `same_level` instead uses the absolute `bohr_tolerance_eV` of
$10^{-10}$ eV, and the closure takes no path. Example: a pair split by
$10^{-4}$ eV at $\ell=1$ µm has $\theta=5\times10^{-4}$. The true rate is then twice
the closure value or zero, depending on relative phase, while the closure reports
$5.73\times10^{-7}$ either way, and the exact-degenerate closure gives
$1.147\times10^{-6}$. Tunnel-split string levels fall in this range. The residual
also falls only as $1/\theta$ and oscillates. `cr-capture-yield` clusters unresolved
levels rather than exact ones. State this condition in the row, or add a path or
resolution argument. "Gives the omitted yield" should read as a rate per depth at
one density.

**Minor.**

1. Capture is correctly owned by the axial block, but ADR-0018 reports it as a
   separate component and this block lumps it with bound→bound. These share final
   states, so their cross term survives Hamiltonian evolution: 5.7% of the axial peak
   in the oracle. A later split needs its own defect, and this zero-cross argument
   does not cover it.
2. Boundary convention: `energies <= boundary_eV` here against
   `basis.energies < solver.barrier_eV` in `emission.py`. Planar emission uses the
   barrier; barrier plus margin is the tracked and handoff subspace. A caller passing
   barrier plus margin would give free→margin transitions to channeling radiation.
   Name the emission barrier in the contract. The axial block also admits every
   represented free initial state, broader than the planar tracked window.
3. Refraction is declared neglected but unbounded. $\delta/D=0.6\%$ at 4.4 keV and
   about 12% at 0.94 keV for 5 MeV Si, and it enters the line position.
4. `emitting_subspace_closure` validates neither Hermiticity nor positivity, and
   `max(full - retained_rate, 0.0)` would hide the resulting negative omission.
   `partition_axial_currents` trusts the lowering contract of the supplied dataclass.
5. Ledger limiting cases without an anchor: decay and dephasing reducing to sinc,
   and a full retained set omitting nothing. Both pass here. No negative test covers
   rank-one or dephasing jumps, although the test name says "only". The comment in
   that test has a typo, `|0|`.

### Suggested replacement text

For the Assumptions field and "What native scoring owns":

> Free→free stays owned by CBS under ADR-0018 and is never scored here. A zone row
> exactly along the axis has $\mathbf v\cdot\mathbf g=0$ for zero-order $\mathbf g$, so
> its native resonance energy is zero and it emits no line, because the straight
> mean segment discards transverse momentum. Each represented free→free transition
> carries $\hbar\bar{\mathbf v}\cdot\mathbf g>0$ and in the Born limit equals the
> native CBS line. Handed-off electrons and tilted rows receive that line natively,
> subject to the native energy gate and in-medium root convergence. A production
> adapter must state, per population, whether native scoring supplies, omits or
> overlaps the block; exact alignment grants no exemption.

Status remains the owner's decision and is unchanged by this section. Re-verification
after the text correction can be limited to this statement and the cross-term text.

## Re-verification

Commit `6878645fb`, 2026-11-05, same verifier as the section above. Scope: native-ownership
text, cross-term text and its tests, closure guards, strict barrier and lowering
check. Refraction, the capture split and per-population native reconciliation moved
to #992 and were not re-checked.

**Verdict: `discrepancy`, in text and test evidence only.** Every code path in scope
behaves as derived and no number changes. First divergent term: line 16 above still
defines $Q_b$ by $E_n\le E_b$, while the code and the ledger row use the strict
`energies < boundary_eV`.

**Confirmed.**

- *Cross term.* $G^{ab}(z,z')^*=G^{ba}(z',z)$ and the gain and no-jump conditions match
  the derivation above. A separate dense GKSL and quadrature agrees with
  `partition_on_axis_spectra` on the anchor's fixture, with defect/total $+0.1556$,
  $+0.1440$, $-0.3600$ for the three `test_block_coupling_jumps_*` cases. In the third
  case the sandwich term vanishes on the incident support, so the id "no-jump-coupling"
  is accurate.
- *Native text.* The row and zone-row distinction, the ownership and the numerical
  threshold are correct.
- *Closure.* Unresolved coherent pairs are refused, and refusal was exercised. At the
  default threshold each pair's frequency-integrated residual is at most $1/\theta=1\%$,
  for equal weights and by $2\lvert\rho_{ii'}\rvert\le\rho_{ii}+\rho_{i'i'}$ otherwise.
- *Partition.* The strict barrier matches `emission.py`, and a level exactly at the
  barrier is free. The lowering check indexes $E_i-E_f$ correctly and passes the
  production builder, with exact zeros in $Q_uAQ_b$.

**Required for `rederived`.**

1. Write-up line 16: replace $E_n\le E_b$ with $E_n<E_b$.
2. Ledger Assumptions and "What native scoring owns": $E_i-E_f=\hbar\bar{\mathbf v}\cdot\mathbf g$
   holds only in the plane-wave limit, and the qualifier must cover it. Only
   $E_i-E_f>0$ is general. Dressed gap over $\hbar\bar{\mathbf v}\cdot\mathbf g$ at Si
   $(2\,2\,0)$, $U=4.32$ eV: 0.987, 1.18, 1.88 for $n=3\to2$, $2\to1$, $1\to0$. My
   earlier suggested text also lacked the qualifier.
3. `test_jumps_that_do_not_couple_*`, id `dephasing`: elements
   `((0, 0), (1, 1), (2, 2), (3, 3))` give $L=\sqrt{0.05}\,\mathbb 1$, whose generator
   is zero, so the case repeats `hamiltonian` and the total is unchanged. The ledger
   check "zero under dephasing" is therefore not exercised. The physics holds:
   `((1, 1), (3, 3))` changes the total by $-26\%$ with defect $0$ to roundoff, in both
   codes.

**Should fix.**

4. `min_coherence_phase` is unvalidated: NaN, $-1$ and $0$ each accept the unresolved
   pair and return the secular rate, which is wrong by up to a factor of 2. Refuse
   non-finite or nonpositive values.
5. The coherence test has an absolute floor, `max(..., 1.0)`. Scaling the same pair by
   $10^{-13}$ passes the guard, so the relative `omitted_fraction` is unguarded for a
   very small density. The trace is also unbounded, unlike `regression_correlation`,
   which refuses a trace above one.
6. The closure has no lowering check. A raising element inside $S$ returns a negative
   `full_rate` of $-5.1\times10^{-7}$ with no error. The negative-omission refusal fires
   only when the element lies outside $S$, is reachable only through that broken
   contract, and has no test.

**Minor wording and fixtures.**

- "with the left index" should say the same right-index argument with the blocks
  swapped.
- "treating the block as absent" should say absent from native scoring.
- `AxialPartitionSpectra`: "The block is CBS-owned" should name the free→free block.
- The `rank-one` case never populates $|2\rangle$, so its jump acts only through
  no-jump decay. Adding $|2\rangle$ to the fixture density would exercise the
  diagonal-gain argument.
- The ledger limiting case for sinc is anchored for decay only, not dephasing.

**Follow-up at commit `0a539020b`: `rederived`.** Items 1–6 and the listed minors are
applied correctly, with one fixture exception below. This was a check of those fixes,
with no new derivation. The anchor module passes (31 tests).

- `dephasing` is no longer vacuous: $L=\sqrt{0.05}(|1\rangle\langle1|+|3\rangle\langle3|)$
  changes the total by $-26\%$ with defect at roundoff.
- `rank-one` is unchanged by the fix, and my suggestion was wrong. Its generator is
  nontrivial (total $-17\%$, defect at roundoff), but the population added on $|2\rangle$ only
  rescales the case: the total scales by exactly $2/3$ and the relative numbers do not
  move. The gain needs $X_{22}=\rho_{23}a^*\ne0$, and $\rho_{23}=0$ here. With
  $(|1\rangle+|2\rangle+|3\rangle)/\sqrt3$, $\lvert X_{22}\rvert=1.3\times10^{-3}$, the
  defect stays at roundoff, and the block-coupling cases still give $+0.156$, $+0.144$,
  $-0.311$. Not blocking, because the claim holds there and in the random-density oracle.
- The negative-omission refusal is now unreachable for valid input, since lowering,
  positivity and whole multiplets force a nonnegative omission. It stays as an untested
  roundoff guard, which is not blocking.
