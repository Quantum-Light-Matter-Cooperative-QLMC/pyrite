# `longitudinal-energy-diagonal-emission`

## Claim and source

Research claim (#629). One electron sits on an energy comb
$E_n=E_0+n\hbar\Omega$ with longitudinal density matrix $\rho_{nm}$ and emits
one photon in a static medium. The claim has three parts:

1. The reduced photon state is
   $\rho_{\rm ph}(\omega,\omega')=\sum_{nm}\rho_{nm}A_n(\omega)A_m^*(\omega')\,\delta(E_n-E_m-\hbar(\omega-\omega'))$.
   Its diagonal, the photon count $\sum_n\rho_{nn}\lvert A_n\rvert^2$, and
   photon–electron coincidence counts do not depend on $\rho_{n\ne m}$. This
   holds for any time-independent (position-dependent) interaction with exact
   recoil.
2. For a laterally infinite static slab at normal incidence, the fixed-energy
   longitudinal mismatch $\Delta$ vanishes at the `line-recoil-resonance` root.
3. The slab formation amplitude is $\int_0^L e^{i\Delta z}\,dz$.

The ledger cites three sources: energy conservation in a static medium; Zhang
et al., arXiv:2401.05978, for the photon-state dependence; and Shi et al.,
Nat. Commun. 15, 7803 (2024), Eq. (6), for the recoil convention. The code under
review is
`src/pyrite/quantum/_longitudinal.py::{photon_density_matrix,slab_mismatch_inv_ang,slab_formation_amplitude}`.
The comparison text is `docs/research/physics/longitudinal-coherence-recoil.md`
§2–3.1.

I wrote the derivation below from the ledger row, the docstrings and the
signatures. I read the implementation bodies and the research note only after
it was fixed.

## Derivation (a): reduced photon state

**Setup.** Let $H=H_0+V$ with $H_0=H_e+H_{\rm med}+\sum_\mu\hbar\omega_\mu
a_\mu^\dagger a_\mu$. Here $\mu=(\omega,\hat{\mathbf n},\lambda)$ labels a
photon mode. $V$ holds the electron–medium potential and the electron–photon
coupling. "Static medium" means $H$ is time independent and the medium's
initial state is diagonal in energy; an infinite-mass crystal is the special
case that never changes state. The initial field is vacuum. The S-matrix then
conserves free energy: $[S,H_0]=0$.

Write the electron's free eigenstates as $\lvert E,\alpha\rangle$, where
$\alpha$ collects the degenerate labels (direction, transverse momentum, spin).
The input state is

$$
\rho_{\rm in}=\sum_{nm}\rho_{nm}\lvert E_n\rangle\langle E_m\rvert\otimes\lvert0\rangle\langle0\rvert ,
$$

so each comb tooth is one incident state $\lvert E_n\rangle$. The one-photon
part of $S\lvert E_n\rangle\lvert0\rangle$ is

$$
\sum_\mu\sum_\beta A_{n,\beta}(\mu)\,\lvert E_n-\hbar\omega_\mu,\beta\rangle\lvert1_\mu\rangle ,
$$

and $[S,H_0]=0$ forces the final electron energy to be exactly
$E_n-\hbar\omega$. The recoil is exact: no expansion in $\hbar\omega/E$ has
been made. The ledger's $\hat M\lvert E_n\rangle=A_n(\omega)\lvert
E_n-\hbar\omega\rangle$ is the special case of a single final state $\beta$ per
energy.

**Partial trace.** Take energy-normalized final states,
$\langle E,\beta\vert E',\beta'\rangle=\delta(E-E')\delta_{\beta\beta'}$. Tracing
over them gives

$$
\rho_{\rm ph}(\mu,\mu')=\sum_{nm}\rho_{nm}\sum_\beta A_{n,\beta}(\mu)A^*_{m,\beta}(\mu')\,
\delta\bigl(E_n-E_m-\hbar(\omega-\omega')\bigr).
$$

With a single final state this is the ledger's expression. Every factor and
sign matches, and the photon labels $\hat{\mathbf n},\lambda$ ride along.

**Diagonal.** At $\omega=\omega'$ the delta requires $E_n=E_m$, so $n=m$ for any
$\Omega\neq0$. Then

$$
\frac{dN}{d\mu}\propto\sum_n\rho_{nn}\sum_\beta\lvert A_{n,\beta}(\mu)\rvert^2 ,
$$

which is independent of $\rho_{n\ne m}$. The $\delta(0)$ at $n=m$ is the usual
continuum normalization. It becomes the count per unit frequency once each tooth
has a finite width; the code replaces it with a Kronecker delta on a commensurate
grid.

The result is stronger than the ideal-tooth statement in assumption (A3).
Suppose the incident state is a general $\rho(E,E')$ with finite or overlapping
teeth. The same trace gives
$\int dE\,\rho(E,E)\sum_\beta\lvert A_\beta(E,\mu)\rvert^2$. The spectrum
therefore depends only on the energy distribution, with no condition that the
teeth be separated.

**General form.** Let $O$ be any observable with $[O,H_0]=0$. Then
$\operatorname{Tr}(O\,S\rho_{\rm in}S^\dagger)=\operatorname{Tr}(S^\dagger OS\,\rho_{\rm in})$,
and $S^\dagger OS$ also commutes with $H_0$. Only the energy-diagonal blocks of
$\rho_{\rm in}$ contribute. This holds to every order in $V$, not only at first
order, and it holds for any position dependence of $V$. The claim "any
time-independent (position-dependent) interaction and exact recoil" is
therefore **justified**, for observables that commute with $H_0$.

**Which counts qualify.** The following commute with $H_0$:

- the frequency-, direction- and polarization-resolved photon number;
- the frequency-integrated photon number;
- a coincidence of that photon number with an electron projector diagonal in
  final electron energy, either energy-resolved EELS or integrated over energy.
  This includes energy- and channel-resolved projectors
  $\lvert E,\beta\rangle\langle E,\beta\rvert$.

**Which do not.** Two classes of observable do not commute with $H_0$, and they
do depend on $\rho_{n\ne m}$:

- An electron measurement that projects onto a superposition of final energies.
  Examples are arrival-time detection, a second phase-locked PINEM stage, or
  electron interferometry. This is the standard post-selection setting in
  free-electron quantum optics.
- Time-resolved or phase-referenced photon detection. The photon off-diagonals
  at $\omega-\omega'=(n-m)\Omega$ appear there as intensity modulation at
  harmonics of $\Omega$.

The ledger's "every photon–electron coincidence count" is therefore
**overstated**. The research note §3.1 correctly says the coincidence counts
are "diagonal in the final electron energy".

**Off-diagonal support.** $\rho_{\rm ph}(\omega,\omega')\neq0$ requires
$\omega-\omega'=(n-m)\Omega$ with $\rho_{nm}\neq0$. With degenerate final
states it also requires a nonzero overlap
$\sum_\beta A_{n,\beta}A^*_{m,\beta}$. A mixture that is diagonal in $n$ gives
zero.

## Derivation (b): slab mismatch

The electron moves along $+z$, the slab normal: $\mathbf p_i=p_i\hat{\mathbf
z}$ with $p_i=p(T)$ and $pc=\sqrt{T(T+2mc^2)}$. Lateral translation invariance,
modulo the reciprocal lattice, conserves transverse momentum. With the
`line-recoil-resonance` convention
$\mathbf p_f=\mathbf p_i-\hbar\mathbf g-\hbar\mathbf k$,

$$
\mathbf p_{f\perp}=-\hbar(\mathbf g_\perp+\mathbf k_\perp),\qquad
\lvert\mathbf p_f\rvert=p(T-\hbar\omega),\qquad
p_{f,z}=+\sqrt{p(T-\hbar\omega)^2-\hbar^2\lvert\mathbf g_\perp+\mathbf k_\perp\rvert^2}.
$$

The last equality uses energy conservation; the $+$ sign selects the forward
branch. The $z$ factor of the transition matrix element is

$$
\int_0^L e^{-ip_{f,z}z/\hbar}\,e^{-ik_zz}\,e^{-ig_zz}\,e^{ip_iz/\hbar}\,dz
=\int_0^Le^{i\Delta z}dz,\qquad
\Delta=\frac{p_i-\hbar g_z-\hbar k_z-p_{f,z}}{\hbar}.
$$

Here $e^{-i\mathbf g\cdot\mathbf r}$ is the potential component consistent with
$+\hbar\mathbf g$ transferred to the crystal, matching the convention. $\Delta$
is measured in Å$^{-1}$.

**Vacuum root.** The `line-recoil-resonance` root solves
$\lvert\mathbf p_i-\hbar\mathbf g-\hbar\mathbf k\rvert=p(T-\hbar\omega)$ with
$\mathbf k=(\omega/c)\hat{\mathbf n}$. Squaring and using
$(E-\hbar\omega)^2-m^2c^4$, I recover its docstring root
$\hbar\omega=(\beta\,\hat{\mathbf u}\cdot\mathbf Q-Q^2/2E)/(1-\beta\,\hat{\mathbf u}\cdot\hat{\mathbf n}+\mathbf Q\cdot\hat{\mathbf n}/E)$,
with $\mathbf Q=\hbar c\mathbf g$. At that root the transverse parts agree by
construction, so

$$
p_{f,z}^2=\bigl(p_i-\hbar g_z-\hbar k_z\bigr)^2 .
$$

Hence $\Delta=0$ **if and only if $p_i-\hbar g_z-\hbar k_z>0$**, meaning the
vector-equation final electron moves forward. On the backward branch the
positive square root gives $\Delta=2(p_i-\hbar g_z-\hbar k_z)/\hbar\neq0$.

For keV–MeV electrons, $p_i/\hbar\sim10^2$–$10^3$ Å$^{-1}$ while
$g\sim1$ Å$^{-1}$, so the forward condition always holds in the PXR/CBS regime.
It still belongs in the claim, because the statement "vanishes at the root" is
otherwise unconditional.

**Slope.** Differentiating at the root, using $dp/dE=1/v$,

$$
\hbar c\,\frac{d\Delta}{d(\hbar\omega)}=-n_z+\frac{k_f/\beta_f+(\mathbf g_\perp+\mathbf k_\perp)\cdot\hat{\mathbf n}_\perp}{k_{f,z}} .
$$

This is positive whenever the transverse term is small. It is not guaranteed
for every geometry, so "positive slope" is a property of the tested geometries
and not part of the claim.

**Approximations.** The photon dispersion is the vacuum one (A5): there is no
refractive index and no absorption. The slab is laterally infinite and the
incidence is normal (A4).

## Derivation (c): formation amplitude

$$
F(\Delta)=\int_0^Le^{i\Delta z}dz=\frac{e^{i\Delta L}-1}{i\Delta}
=L\,e^{i\Delta L/2}\,\frac{\sin(\Delta L/2)}{\Delta L/2},\qquad
\lvert F\rvert^2=\frac{4\sin^2(\Delta L/2)}{\Delta^2}.
$$

- Units: $F$ is in Å.
- $\Delta\to0$: $F\to L$, the coherent length.
- $\Delta\to-\Delta$: $F\to F^*$.
- $\lvert F\rvert$ is even in $\Delta$, so the sign convention of $\Delta$ does
  not reach the counts.

## Cheap filters

| check | result |
| --- | --- |
| units | $\rho_{\rm ph}$ carries the units of $\lvert A\rvert^2$ (per unit energy through $\delta$); $\Delta$ in Å$^{-1}$ ($pc$ in eV divided by $\hbar c$ in eV Å); $F$ in Å; pass |
| limits | $\Omega$-separated teeth leave only $n=m$ on the diagonal; a mixture gives zero photon coherence; $\Delta\to0$ gives $F=L$; at the root $\Delta=0$ on the forward branch; pass |
| signs/conventions | $\mathbf p_f=\mathbf p_i-\hbar\mathbf g-\hbar\mathbf k$ matches `recoil_resonance_eV` ("crystal takes $+\hbar\mathbf g$"); $p(E-\hbar\omega)$ is evaluated from kinetic energy, equivalent to total energy; pass, with the forward-branch condition |
| notation across ledger, docstring, note and inventory | identical $\Delta$, $p_{f,z}$ and $\rho_{\rm ph}$; the inventory row copies the ledger claim; pass |

## Comparison with the implementation

Read after the derivation above was fixed.

`photon_density_matrix` sets the photon grid to
$\omega_a=\omega_0+a\,\Omega/s$. The final electron energy is then
$E_0-\hbar\omega_0+(ns-a)\hbar\Omega/s$, so the index `f = n*subdivisions - a`
is the correct final-energy label. The function builds the joint amplitude
`joint[f, n, a]` and contracts
`einsum("nm,fna,fmb->ab", rho, joint, conj(joint))`. That is the explicit trace
with a Kronecker delta in place of $\delta(E_n-E_m-\hbar(\omega-\omega'))$, and
it matches derivation (a) for one final state per energy.

`slab_mismatch_inv_ang` computes the same quantities as derivation (b):

```text
k_photon = photon * n / HBARC_EV_ANG
transverse = g[:2] + k_photon[:2]
k_final_z = sqrt((p(T - photon)/HBARC)^2 - |transverse|^2)
return p(T)/HBARC - g[2] - k_photon[2] - k_final_z
```

It takes the positive square root and returns NaN for an evanescent or
negative-$T_f$ final state. That matches derivation (b), including its
forward-branch behaviour.

`slab_formation_amplitude` returns
`L * exp(1j*half) * np.sinc(half/np.pi)` with `half = Delta*L/2`. NumPy's
$\operatorname{sinc}(x)=\sin(\pi x)/(\pi x)$, so this is exactly
$L e^{i\Delta L/2}\sin(\Delta L/2)/(\Delta L/2)$, and it is finite at
$\Delta=0$. It matches derivation (c).

There is no symbolic difference in any of the three functions.

## Numerical checks

The anchors in `tests/montecarlo/quantum/test_longitudinal_coherence_reference.py`
pass (34 passed). Independent checks used a scratch script (not committed).
None of its reference values come from PyRITE helpers.

| check | result |
| --- | --- |
| general static S-matrix: 6 orders, 4 degenerate final channels per energy, random energy-conserving $A_{n,\beta}$; pure, re-phased and dephased states | photon counts, energy-resolved coincidences and energy- and channel-resolved coincidences agree to $\le3.6\times10^{-15}$ (scale 12) |
| counterexample: electron projected onto $(\lvert f\rangle+\lvert f+\hbar\Omega\rangle)/\sqrt2$ | pure 0.1513, re-phased 0.1615, mixture 0.1696; such coincidences **do** depend on $\rho_{n\ne m}$ |
| photon coherence | exactly 0 off comb multiples; $O(1)$ at multiples |
| `photon_density_matrix` against an independent trace | $4\times10^{-16}$ |
| $\Delta$ at an independently coded vacuum root; five geometries, 30 keV–5 MeV, including $g_\perp\neq0$ and backward photons | own $\Delta$ and PyRITE's both $\le5\times10^{-13}$ Å$^{-1}$, against $k\sim10^2$–$10^3$ Å$^{-1}$; slope positive in all five |
| backward-branch edge: $T=200$ eV, $g_z=14$ Å$^{-1}$, $\hat{\mathbf n}=-\hat{\mathbf z}$ | vector-equation $p_{f,z}/\hbar=-6.74$ Å$^{-1}$; PyRITE gives $\Delta=-13.48=2p_{f,z}/\hbar$, consistent with derivation (b), which requires the forward condition |
| $F(\Delta)$ against adaptive quadrature, $L=333$ Å, $\Delta\in\{0,10^{-6},0.013,0.4,-0.9\}$ | $\le1.5\times10^{-13}$ |

## Comparison with the research note

| item | note / ledger | this derivation | agreement |
| --- | --- | --- | --- |
| `eq-lcr-photon` | $\sum\rho_{nm}A_nA_m^*\delta(\cdot)$ | same; with degenerate final states the product becomes $\sum_\beta A_{n,\beta}A_{m,\beta}^*$ | matches; the single-final-state form is a model simplification, and the diagonal statement survives degeneracy |
| "any time-independent Hamiltonian, exact recoil" | stated | $[S,H_0]=0$ makes every $H_0$-commuting observable blind to $\rho_{n\ne m}$, at all orders | justified |
| coincidences, note §3.1 | "EELS-coincidence counts, which are diagonal in the final electron energy" | same | matches |
| coincidences, ledger Claim and note summary bullet (line 7) | "every photon–electron coincidence count" | false for electron projectors that are not energy-diagonal (counterexample above) | **wording overstated** |
| frequency-integrated counts | stated | integral of the diagonal | matches |
| (A2) "no phonon created" | assumption | needed only for the single-state form of $\hat M$. Diagonal independence also holds with medium excitation if the medium starts energy-diagonal (thermal); a driven or time-modulated medium breaks it | assumption stricter than needed; no change required |
| (A3) "finite tooth width only broadens each line" | stated | the spectrum depends only on $\rho(E,E)$ for any tooth shape | matches; the stronger statement is available |
| `eq-lcr-slab`, "$\Delta_n=0$ exactly at the #249 root" | unconditional | requires $p_i-\hbar g_z-\hbar k_z>0$ | **missing condition**; holds throughout the PXR/CBS regime |
| formation amplitude | $\int_0^Le^{i\Delta z}dz$ | same | matches |

## Adjudication

All three functions re-derive with no differing factor, sign, exponent, unit or
convention:

- the reduced photon density matrix, its comb-multiple support and the
  independence of its diagonal from $\rho_{n\ne m}$;
- the fixed-energy slab mismatch and its zero at the vacuum recoil root;
- the slab formation amplitude.

The position-dependent, exact-recoil scope is justified, and it extends to all
orders for observables that commute with $H_0$.

Two wording fixes are needed:

1. **Coincidence scope.** Replace "every photon–electron coincidence count" with
   "photon–electron coincidence counts that are energy-resolved (or
   energy-integrated) for the electron". Electron detection that is not
   energy-diagonal, and time-resolved photon detection, do see $\rho_{n\ne m}$.
   This applies to the ledger Claim, the domain-inventory row, and the research
   note summary bullet (line 7).
2. **Root branch.** Add "for a forward final electron, $p_i-\hbar g_z-\hbar k_z>0$"
   to "vanishes at the `line-recoil-resonance` root". This applies to the ledger
   Claim, the `slab_mismatch_inv_ang` docstring, and the sentence after
   `eq-lcr-slab` in the note.

Verifier verdict: `rederived`, provided those wording fixes are applied. Human
sign-off is still required (#277).

## Owner source check (2026-11-08, after the verdict)

Two further sources, checked by the claim owner, agree with the theorem:

- Karnieli et al., Sci. Adv. 7, eabf8096 (2021): the diagonal
  $\rho_{\rm ph}(\omega,\omega)$, which is the photodetection probability,
  does not depend on the wavefunction. The off-diagonal spectral coherence
  does.
- Kfir et al., Sci. Adv. 7, eabf6380 (2021): the cathodoluminescence intensity
  and spectral profile do not depend on the electron state. Phase correlations
  with the modulating laser, which extend to its harmonics, do; this matches
  support only at $\omega-\omega'=(n-m)\Omega$.
