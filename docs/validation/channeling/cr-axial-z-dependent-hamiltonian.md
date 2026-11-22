# `cr-axial-z-dependent-hamiltonian`: depth-piecewise Hamiltonian and rates of the fibred axial model

**Status:** see the [ledger row](../ledger-channeling.md#cr-axial-z-dependent-hamiltonian). Task-owner derivation (2026-11-22, #1343). The fresh-context verification is recorded below it.

## Problem

A Si/SiGe superlattice grown along [001] and observed along ⟨100⟩ modulates
the axial string potential along the beam. The [001] strings do not kink.
Commensurate growth keeps the in-plane lattice, so the transverse cell is
common to every layer. What changes from layer to layer is the spacing
along the string (5.390 vs 5.590 Å), the composition and so the depth of the
string potential ([research note](../../research/physics/strain-acoustic-modulation-1058.md), §7).
A longitudinal hypersound wave is the frozen, periodic version of the same
modulation (Grigoryan et al. 2001, Eq. (2.4)). Layers of 10–50 nm are
comparable to the 2p–1s beat length (45.5 nm) and the dephasing length
(≈ 35 nm). The per-zone secular Lorentzian line model therefore does not
apply, and the two-time correlation has to span layers.

## Model

The slab is cut into zones $k=0,\dots,N-1$ that own $[z_{k-1},z_k)$, with
$z_{-1}=0$ and the last zone running to the exit. An interface at or beyond
the exit is ignored, so the exit record stays in the zone before it. A record
exactly at an interface sees the post-interface state and the new zone's
currents. A zone with an empty retained subspace is refused (`ValueError`). Each zone carries its own
fibred absorbing model of `cr-axial-fiber-unraveling`: the Bloch Hamiltonian
$H_k(\kappa)$ on the shared order window, its retained cut, and its rate
operators (lattice or continuum TDS and electronic kicks). All zones share the
reciprocal cell, the order window $R$ and the outgoing box $B$. In zone $k$
the retained subspace is $P_k(\kappa)=V_{k,\kappa}V_{k,\kappa}^\dagger$, with
$V_{k,\kappa}$ the orthonormal eigenvectors of $H_k(\kappa)$ below the cut.

### Continuity at an interface

The transverse state obeys the paraxial depth equation
$i\hbar v\,\partial_z\psi=H(z)\psi$, which is first order in $z$. For bounded
$H$, integrating over $[z_I-\epsilon,z_I+\epsilon]$ gives
$\psi(z_I^+)-\psi(z_I^-)=O(\epsilon)$. The transverse wavefunction is therefore
continuous, with no derivative condition and no reflected wave (backscatter is
outside the paraxial model). On the shared order basis, continuity is the
identity. The carrier $\kappa$ is conserved because $H_k$ and $H_{k+1}$
commute with the same lattice translations. Restricting to the retained
subspace gives the sudden re-projection

$$
c_{k+1}=V_{k+1,\kappa}^\dagger V_{k,\kappa}\,c_k ,\qquad
f=(1-P_{k+1})V_{k,\kappa}c_k ,
$$

with $\lVert V_kc_k\rVert^2=\lVert c_{k+1}\rVert^2+\lVert f\rVert^2$ exactly.
The complement $f$ lies above the new zone's cut. In the absorbing model it is
an outgoing state: it leaves the retained model like the outgoing part of a
jump. On the density matrix the interface is the trace-decreasing map
$\rho\mapsto T\rho T^\dagger$, $T=V_{k+1}^\dagger V_k$, and the removed part
$f f^\dagger$ is rank one per pure state.

### Unraveling

Castin, Dalibard & Mølmer (arXiv:0805.4002, Eqs. (4), (8)) draw a waiting
threshold $r\sim U(0,1)$ and stop when the unnormalized norm $\lVert c(z)\rVert^2$
first falls below $r$. Between jumps, the no-jump evolution is now the
product of per-zone propagators $e^{-i(z-z_I)H_{\mathrm{eff},k}/\hbar v}$
interleaved with the maps $T$. The norm law still holds if the **same**
threshold spans every interface the segment crosses. A drop at an interface
below $r$ is an event at $z_I$. Conditional on surviving to $z_I$, its
probability is
$(\lVert c_k\rVert^2-\lVert c_{k+1}\rVert^2)/\lVert c_k\rVert^2$, the removed
weight, and its outcome is the pure state $f$ (Weyl-measured on the box). A
jump inside zone $k$ uses zone $k$'s rates and operators, unchanged. Hence
the ensemble reproduces the per-zone GKSL maps composed with
$\rho\mapsto T\rho T^\dagger$.

### Two-time correlation across zones

The companion estimator of `cr-axial-fiber-regression` needs every record
operator applied to the primary to be applied to the companion as well. $T$
is linear and the primary is not renormalized at an interface, so the
companion receives $T$ with no divisor. In the eigen-coordinates of
`_Companions`, $y_{k+1}=R_{k+1}^{-1}V_{k+1}^\dagger V_kR_ky_k$. The estimator
stays linear in the record operator, so it is unbiased for the composed map
with no extra draws, and
$G_{ab}(z_i,z_j)=\mathrm{Tr}[J_a^{(k_i)}\Lambda_{z_i\leftarrow z_j}(\rho(z_j)J_b^{(k_j)\dagger})]$
with $\Lambda$ the composed map. This keeps all coherence across layers. No
per-layer line model is assumed.

### Currents

$J_a^{(k)}$ are the velocity/c lowering elements ($E_i-E_f>0$) in zone $k$'s
retained eigenbasis, as in `cr-axial-fiber-regression`. Velocity itself is
zone independent; only the positive-frequency selection is local. For a
modulation with drive quantum $\epsilon_s=\hbar k_sv$, the first-order
amplitude of the sound-assisted line $f\leftarrow i$ is
$\sum_n[J_{fn}\varepsilon^{(i)}_n+\varepsilon^{(f)*}_nJ_{ni}]$, with
$\varepsilon^{(\cdot)}$ the dressing admixtures. The local lowering rule keeps
initial-state dressing, which includes the near-resonant pathways of
Grigoryan's Figs. 1–2 (2p dressed by 3p, then 3p→1s; 1s dressed by 3s, then
3s→2p). Final-state dressing is kept as well, because the companion
$J\hat u$ propagates under the modulated dynamics. What is dropped is the
insertion of raising elements $J_{ni}$ ($n$ above $i$), i.e. the
final-state-dressing terms whose intermediate state lies above the initial
one. The filter acts in each zone's own eigenbasis, so it also drops the
non-adiabatic part of $J_{fn}\varepsilon^{(i)}_n$ with $E_n<E_f$ (empty when
$f$ is the ground state). Every dropped term is detuned by at least the
photon energy, so all resonant pathways are kept. Off-resonant first-order
sideband intensities are qualitative only. In the verifier's 4-level toy,
compared with exact propagation and the full current, resonant lines agreed
to 0.9% and the inverse line to 4.8%. The off-resonant ν = −1 and ν = +1
lines differed by +10% and −37%. In a prototype (below) a variant that also inserts raising elements
moved the inverse 1s→2p line by 1.5% at 4 µm. That variant is not
implemented: it adds negative-frequency leakage through the sharp slab
window, of the same order as the line at 2 µm.

### Periodic limit

`periodic_fiber_zones` cycles caller-supplied phases. For a smooth
$U(\rho,z)$ the caller builds them at slice midpoints, $N$ slices per period. The staircase has fundamental amplitude
$\mathrm{sinc}(\pi/N)$ of the cosine, so the sideband intensity is low by
$\approx(\pi/N)^2/3$: 1.3% at $N=16$. It also has harmonics at
$(N\pm1)k_s$, with relative amplitude $1/(N\pm1)$, far above the photon grid
for $N\ge8$. A superlattice is two phases with slice $L_\mathrm{layer}$,
which is the exact piecewise-constant model.

## Assumptions and scope

- Every zone is a fibred absorbing model of `cr-axial-fiber-unraveling` on one
  shared transverse cell, order window and outgoing box. Strings do not kink
  (no transverse re-registration or tilt at an interface).
- Sudden, atomically sharp interfaces on the paraxial depth equation. No
  backscatter, no interfacial strain profile or interdiffusion.
- Interface complements above the new cut are outgoing and terminal, the
  no-return assumption of the absorbing model. Their later re-binding in a
  deeper zone is not represented.
- Frozen crystal: the modulation is static in depth. A hypersound wave enters
  only through $z=vt$ with $v_s\ll v$, so the lattice is quasi-static.
- On-axis observation; local lowering currents as above.
- Out of scope: transverse or helical drives, time-dependent crystals,
  production cases and defaults, CUDA, material models for SiGe (virtual
  crystal, alloy-disorder scattering), convergence and measured agreement.

## Limiting cases

- A constant sequence (the same object in every zone) merges to the one-zone
  model and runs the unzoned code path. Draws, events, correlation and spectrum
  are bitwise those of the current engine.
- Distinct but identical zones take the interface path. $T=1$ up to
  roundoff, so events and $G$ agree with the one-zone model to 1e-8.
- No jumps and no drive: a ground-state primary emits nothing. With the drive
  it emits only through the modulation (inverse transitions).
- Off-resonant periodic drive: the ν = ±1 lines at $\Delta\mp\epsilon_s$ scale as
  μ², ν = ±2 are higher order and the ν = 0 line changes only at O(μ²)
  (Grigoryan et al. 2001, Eqs. (2.20), (3.7)).

## Grigoryan et al. (2001) periodic-limit check

*Source.* Radiat. Eff. Defects Solids 153, 221 (2001), Zotero `3436K9M2`
(scanned PDF, figures read). Their model is a single string,
$U=-(1+\mu\cos k_sz)a/\rho$, with 2D-hydrogen levels
$\varepsilon=-\varepsilon_*/2n^2$, $n=\tfrac12,\tfrac32,\tfrac52$ for 1s, 2p and
3s/3p ($\varepsilon_3-\varepsilon_2=\tfrac{32}{225}\varepsilon_*$,
$\varepsilon_3-\varepsilon_1=1.92\varepsilon_*$), at 10 MeV with μ = 0.01. Their
scalar term $+\Delta\cos k_sz$ is left out of the analog: it drops out of the
ν = ±1 amplitude, whose Eq. (3.12) contains only μ. They
show the following steady-state rates in first-order perturbation theory:

- Fig. 1a: 2p→1s via ν = −1 grows for ξ = 0.1427 and 0.1425, close to 3p–2p
  resonance (0.1422).
- Fig. 1b: the inverse 1s→2p via ν = −1 for ξ = 1.9202, close to 3s–1s.
- Fig. 2: 3p→1s via ν = +1.

*Analog.* The anchor uses a square lattice of softened Coulomb strings (cell
6 Å, $a$ = 6 eV Å, core 0.15 Å, 81 orders, 10 MeV) with levels 1s, 2p
(+8.17 eV) and a split n = 3 shell. The lattice screens the hydrogen
degeneracy. The modulation is $(1+\mu\cos k_sz)U$ with 16 slices per period.
Spectra use a Blackman–Harris depth window on $G$ only to separate lines
from the sharp-slab sinc tails. That window is analysis, not physics.

- Off resonance ($\epsilon_s$ = 2.5 eV, 2p start), the ν = ±1 lines appear at
  $(\Delta\mp\epsilon_s)\gamma^2(1+\beta)$. Their area ratio for μ = 0.05 vs
  0.025 is 3.95 and 3.98. ν = ±2 is ≤ 0.9% of ν = ±1, and the main line drops
  by 1.3% at μ = 0.05.
- With $\epsilon_s$ just above $\varepsilon_{3s}-\varepsilon_{1s}$ and a 1s start,
  emission is identically zero without the drive. With the drive, ≈ 90% of it
  lies in the inverse line at $\epsilon_s-\Delta_{2p1s}$ (Fig. 1b analog).

*First-order breakdown.* A detuning sweep at μ = 0.01 toward 3p–2p resonance
stays finite at δ = 0 (ν = −1 area 7.1e-7 at 3 µm), where the first-order
rate diverges as $1/\delta^2$. The engine is non-perturbative in μ. The sweep
is not a clean $1/\delta^2$ law in this toy: a second p-like doublet 0.54 eV
above 3p interferes. Their steady-state rates also need $L\gg2\pi\hbar v/\delta$
for the dressing to settle. At their ξ = 0.1427, $\delta\approx4.8\times10^{-4}\varepsilon_*$,
so $L_\delta/\lambda_{2p1s}=\Delta_{2p1s}/\delta\approx3.7\times10^3$ beat lengths
of the 2p–1s line. That is ≈ 170 µm on the Si ⟨100⟩ 3.5 MeV scale
(λ = 45.5 nm), far beyond the ≈ 35 nm dephasing length. Their near-resonant
enhancement is a steady-state, coherent-limit result. Together with the 1%
strain μ implies (research note §8), it is a model check, not a feasibility
claim.

## Checks (anchor module)

`tests/montecarlo/quantum/test_channeling_axial_fiber_zones.py`:

- an empty retained zone is refused;
- the constant sequence is bitwise the unzoned engine, for `fiber_on_axis_source`
  and `unravel_fiber_outgoing`;
- identical copies cross interfaces with no effect (1e-8);
- an independent dense oracle (three zones with different potential and rate
  scales, interfaces off the record grid, per-zone block GKSL maps on two
  carriers, $T\rho T^\dagger$ at each interface):
  - jump-free: interface outgoing events occur only at interface depths, and
    the retained trace drops below 0.95;
  - with carrier-changing kicks, and with kicks plus a lattice Kraus column:
    correlation $z$ RMS < 1.3, max < 5; spectrum $|z|<4.5$; trace within 5
    binomial SE;
- dropping the interfaces (first-zone model throughout) is rejected at > 10 SE;
- truncated-thickness spectra of zoned histories equal the full slab;
- refusals (interface count/order, mismatched window/box/cell, empty);
- Grigoryan analog (slow): μ² sidebands, ν = ±2 suppression, the inverse
  transition only under drive.

A mutation that drops the companion interface map fails two oracle tests.

## Fresh-context verification (2026-11-22)

Verifier: fresh context (Claude Opus 5.5) that did not write the
implementation. Commit under test: `8127e51a1` (one commit on `origin/main`).
The derivation below was written from the ledger row, this page's claim
statements, the cited sources and the already verified
`cr-axial-fiber-unraveling` and `cr-axial-fiber-regression` write-ups. It was
written before the implementation bodies were read. Scratch oracles live in
the worktree's ignored `scratch/verify1343/` and are not part of the anchor.

### Independent derivation

*Interface map.* The paraxial depth equation $i\hbar v\,\partial_z\psi=H(z)\psi$
is first order in $z$. With $H$ bounded across a step,
$\psi(z_I^+)=\psi(z_I^-)$. No derivative matching and no reflected wave
appear. Both zones are periodic on one lattice, so they share the carrier
$\kappa$ and the order basis. In zone $k$ the retained window vector is
$\chi=V_kc$, with $V_k^\dagger V_k=1$. Split it on zone $k+1$:

$$
\chi=V_{k+1}c'+f,\qquad c'=V_{k+1}^\dagger V_kc,\qquad f=(1-P_{k+1})V_kc,
\qquad \lVert c\rVert^2=\lVert c'\rVert^2+\lVert f\rVert^2 .
$$

$\chi$ lies on the window, so $f$ has no out-of-window part. $f$ lies in the
span of the eigenvectors of $H_{k+1}(\kappa)$ above the new cut, so it is an
outgoing state of the absorbing model. On densities the retained map is
$\rho\mapsto T\rho T^\dagger$ with $T=V_{k+1}^\dagger V_k$, a contraction.

*Record law.* Between jumps, the unnormalized no-jump state is
$\tilde c(z)=U_{k}(z-z_{k-1})T_{k-1}\cdots T_0U_0(z_0)c_0$. Each factor is a
contraction, so $\lVert\tilde c(z)\rVert^2$ is non-increasing and right
continuous, with downward steps at interfaces. Draw $r\sim U(0,1)$ once and
let the event depth be the first $z$ with $\lVert\tilde c(z)\rVert^2\le r$.
Then $P(\text{no event before }z)=\lVert\tilde c(z)\rVert^2$, exactly the
no-jump survival of the composed map. An interior crossing in zone $k$ has
density $-\partial_z\lVert\tilde c\rVert^2=\langle\tilde c|\Gamma_k|\tilde c\rangle$.
Selecting a channel of zone $k$ by its rate reproduces zone $k$'s gains. A
step at $z_I$ catches $r\in(\lVert T\tilde c\rVert^2,\lVert\tilde c\rVert^2]$.
Its probability, conditional on survival to $z_I^-$, is
$(\lVert\tilde c\rVert^2-\lVert T\tilde c\rVert^2)/\lVert\tilde c\rVert^2$, the
removed weight, and its output is the pure state $f$. A fresh threshold after
each jump makes the records Markov. Summing over records therefore gives the
Dyson series of the per-zone GKSL maps composed with $T\rho T^\dagger$ at each
interface. The tallies of outgoing branches at $z_I$ equal
$\mathrm{Tr}[(1-P_{k+1})V_k\rho(z_I^-)V_k^\dagger]$. Without a fresh draw,
the same $r$ has to carry across an interface: a redraw at $z_I$ would bias
the survival law.

*Ownership convention.* Zone $k$ owns $[z_{k-1},z_k)$, so a record exactly
at $z_I$ sees the post-interface state, zone $k+1$'s basis and its currents.
A history that ends at $z_I$ is not alive there. A jump that lands exactly on
$z_I$ has measure zero; either order is admissible.

*Companion.* For the estimator of `cr-axial-fiber-regression`,
$\hat w=K_Rw/\lVert K_Ru\rVert$. The interface factor $T$ enters $K_R$
linearly, and the primary carries it unnormalized until the next divisor. The
companion must therefore receive $T$ with no divisor. Every later divisor is
the primary's: the pre-jump norm $\lVert T\cdots\rVert$ and $\lVert PA\chi\rVert$.
In eigen-coordinates $c=R\,a$ with $R$ the right eigenvectors of the current
$H_{\rm eff}$, the companion therefore maps as
$y\mapsto R_{k+1}^{-1}V_{k+1}^\dagger V_kR_ky$, after $y$ has been advanced
to $z_I$ in zone $k$.

*Currents.* Let the drive be $H=H_0+\mu W\cos k_sz$. To first order in $\mu$,
the dressed amplitude of the sideband $f\leftarrow i$ is
$\sum_n[J_{fn}\varepsilon^{(i)}_n+\varepsilon^{(f)*}_nJ_{ni}]$, which matches
the expression above. Take $J^+$ in zone $k$'s *local* eigenbasis. It keeps
the adiabatic (static) part of every admixture, because that part rotates
with the basis. Of the non-adiabatic part it drops the terms that need a
negative-frequency local element:

- (a) $\varepsilon^{(f)*}_nJ_{ni}$ with $E_n>E_i$;
- (b) $J_{fn}\varepsilon^{(i)}_n$ with $E_n<E_f$.

Suppose a resonant intermediate state satisfies
$E_n\simeq E_i+\nu\epsilon_s$ (initial dressing) or $E_n\simeq E_f-\nu\epsilon_s$
(final dressing). The emitted photon then has energy $E_n-E_f$ or $E_i-E_n$,
up to the detuning. A positive photon energy makes the needed element
lowering, so it is kept. Every dropped term is detuned by at least the photon
energy, so resonant pathways survive. Off resonance, the dropped terms are of
the same order as the kept ones. Case (b) is empty when $f$ is the ground
state.

*Periodic slicing.* Midpoint sampling of $\cos k_sz$ with $N$ slices is the
sampled sequence convolved with a box of width $\lambda_s/N$. The box
transform at $mk_s$ is $\mathrm{sinc}(\pi m/N)$, and the sampled sequence
puts its weight on $m\equiv\pm1\pmod N$. The fundamental is therefore
$\mathrm{sinc}(\pi/N)$, the intensity deficit is
$1-\mathrm{sinc}^2(\pi/N)\approx(\pi/N)^2/3$, and the harmonics at
$(N\pm1)k_s$ have relative amplitude
$\lvert\mathrm{sinc}(\pi(N\pm1)/N)\rvert/\mathrm{sinc}(\pi/N)=1/(N\pm1)$.
No other harmonics appear and the DC term is zero.

*Grigoryan et al. 2001.* Eq. (2.6b) (p. 223) gives
$\varepsilon_{NE}=-\varepsilon_*/[2(N-\tfrac12)^2]+b$ with $N=n+\lvert m\rvert$.
So $N-\tfrac12=\tfrac12,\tfrac32,\tfrac52$, giving $-2$, $-\tfrac29$ and
$-\tfrac2{25}$ in units of $\varepsilon_*$. That makes
$\varepsilon_3-\varepsilon_2=\tfrac{32}{225}=0.14222$ and
$\varepsilon_3-\varepsilon_1=1.92$. Eq. (4.3) defines $\xi=\hbar k_s/\varepsilon_*$.
Section 4 and the figure captions (pp. 233–235) give μ = 0.01, 10 MeV,
ξ = 0.1427 / 0.1425 for Fig. 1a (2p→1s, ν = −1), 1.9202 / 1.9201 for
Fig. 1b (inverse 1s→2p, ν = −1) and 0.145 / 0.144 for Fig. 2 (3p→1s, ν = +1).
At ξ = 0.1427, $\delta=4.78\times10^{-4}\varepsilon_*$ and
$\Delta_{2p1s}=\tfrac{16}{9}\varepsilon_*$. Then
$L_\delta/\lambda_{2p1s}=\Delta/\delta=3.72\times10^3$, and
$3.72\times10^3\times45.5$ nm = 169 µm. The second value, ξ = 0.1425, gives
$6.4\times10^3$, which only strengthens the steady-state caveat. Eq. (2.4)
also carries a scalar term $+\Delta\cos k_sz$. It is common to every
transverse state, and the $\nu=\pm1$ amplitude of Eq. (3.12) contains only
μ, so the analog may omit it.

### Comparison with the implementation

| Item | Derivation | Code | Result |
|---|---|---|---|
| Interface map | $c'=V'^\dagger Vc$, $f=(1-P')Vc$ on the window | `_interface` | match |
| Threshold | one $r$ per segment, kept across interfaces | inner `while` of `_trajectory`; `rng.random()` once per outer loop | match |
| Interface event | event if $\lVert c'\rVert^2\le r$, output $f$ embedded in the box and Weyl-measured, depth $z_I$ | `outgoing(out, depth)` with `out[window_in_box] = lost` | match |
| Jumps after an interface | zone-$k$ rates, operators and fibres; pre-jump state normalized | `_jump(active, …)`, `state / norm(state)` | match |
| Record at $z_I$ | post-interface state, zone $k+1$ currents | segments `[start, stop)`; next zone starts at $z_I$ | match |
| `observe` | normalized state, alive weight | `_observe_segment` normalizes | match |
| Companion | $y\mapsto R'^{-1}V'^\dagger VRy$, no divisor; later divisors are the primary's | `_Companions.interface`, `jump` with `np.linalg.norm(state)` | match |
| Constant sequence | same object ⇒ one zone | `merged` (`is`), `_make_engine` returns `_Engine`; `_Engine.zones = ((self, inf),)` gives `end = max_depth` | match |
| Periodic slices | caller-supplied midpoint phases, cycled | `periodic_fiber_zones` | match |

The single-zone path does exactly what `origin/main` does; this was checked by
reading the diff. It draws `rng.random()` once per outer iteration, calls
`_waiting(fiber, c, r, depth, max_depth, …)` and reports `observe` and `carry`
over `[start, stop)` with `final = hit is None`. It then calls `_jump` with the
same engine. No extra draw or operation enters, and `_primaries` and
`unravel_fiber_outgoing` only swap `_Engine` for `_make_engine`.

### Numerical evidence

The independent oracle is `scratch/verify1343/oracle.py`. Its model:

- an order window $n_x\in[-2,2]$, $n_y=0$, on one carrier;
- a random lattice Kraus column ($D=4$) plus an extra unresolved loss;
- four zones with different random $H$, cuts and rates, retaining 3, 2, 3 and
  3 states;
- interfaces at 900, 1500 and 2400 Å, all *on* the 300 Å record grid.

The oracle builds dense per-zone GKSL generators with `expm`, applies
$T\rho T^\dagger$ for interfaces in $(z_0,z_1]$, and computes its own
zone-local lowering currents. It also tallies the removed weight per
interface.

- With jumps, $N=10^4$ and seeds 3, 5 and 9, the correlation (120 real and
  imaginary entries) gives $z$ RMS 1.75, 1.16 and 1.07, with maxima 3.8, 2.5
  and 2.7. The retained trace agrees to within $|z|\le2.9$, and outgoing
  fractions at each interface to within $|z|\le1.9$. Seed 3 carries one
  common 1.9σ fluctuation of the 900 Å death fraction into every correlated
  trace entry; seeds 5 and 9 sit at $|z|\le1.5$.
- Without jumps, at $N=4\times10^4$: the removed weights 0.6987, 0.0330 and
  0.0886 compare with engine values 0.6983, 0.0339 and 0.0881 (all
  $|z|<1$).
- Mutation: a spurious divisor (`y / 0.8`) in `_Companions.interface` gives
  correlation $z$ RMS 6.3 and a maximum of 13.5, so the oracle resolves the
  companion map.
- The `observe` contract (`scratch/verify1343/observe_check.py`,
  $N=6000$): the mean $W\,\mathbb 1\,cc^\dagger$ at checkpoints 600, 900,
  1200, 1500 and 2000 Å matches the oracle density elementwise to
  $\le5\times10^{-3}$, within the binomial SE of about $9\times10^{-3}$. At
  900 Å it has the post-interface dimension 2.
- Periodic slicing, by FFT of the staircase: the fundamental equals
  $\mathrm{sinc}(\pi/N)$ to $10^{-9}$. The intensity deficit is 0.05036,
  0.01279 and 0.00321 at $N$ = 8, 16 and 32, against $(\pi/N)^2/3$ = 0.0514,
  0.0129 and 0.0032. The harmonics at $N\pm1$ equal $1/(N\pm1)$, and all
  others are below $2\times10^{-16}$.
- Local lowering filter (`scratch/verify1343/sideband.py`): a 4-level s/p
  toy with exact $z$ propagation and a Blackman–Harris window, compared with
  the full current. Near the 3p–2p resonance the $\Delta+\epsilon_s$ line is
  within 0.9%, and the inverse 1s→2p line is +4.8% at μ = 0.02 with 0.04
  detuning. Off resonance ($\epsilon_s$ = 0.3 against gaps of 0.6–0.7) the
  ν = −1 and ν = +1 lines differ by +10% and −37%. A filter in the fixed
  unperturbed basis is far worse (ν = +1 low by 99%).
- Anchor module, including the slow tests: `pyrite-dev test --slow
  tests/montecarlo/quantum/test_channeling_axial_fiber_zones.py`, 11 passed
  in 39.7 s. Neighbour modules `test_channeling_axial_fiber_regression.py` and
  `test_channeling_axial_fiber_unraveling.py`: 44 passed, 1 slow skipped.

### Findings

1. **Minor (convention/documentation).** An interface at or beyond the slab
   exit is silently ignored. In `_trajectory`,
   `end = min(zones[zone][1], max_depth)` and the `final` return precede the
   interface. The record at $z=$ `max_depth` therefore uses zone $k$'s state
   and currents. That contradicts the stated ownership "zone $k$ owns
   `[interfaces[k-1], interfaces[k])`" (`FiberZones` docstring, Model
   section). In the oracle, an interface placed at 3000 Å = exit removed
   0.20 of the weight at the last record, where the engine removed none.
   Physically the engine's choice is the sensible one, since a zero-thickness
   exit zone should not act. Document it: "interfaces at or beyond the exit
   are ignored; the exit record belongs to the last zone of positive
   thickness."
2. **Minor (robustness).** A zone whose retained subspace is empty at the
   carrier crashes in `_Engine.fiber`: `np.linalg.cond` raises a
   `LinAlgError` on the empty eigenbasis. The law has a clean answer (the
   whole weight is outgoing at $z_I$), but the engine refuses through an
   unhelpful numpy error rather than a `ValueError`. No silent wrong result.
3. **Minor (characterization).** The currents paragraph names only (a), the
   final-state raising insertions. In the local basis the filter drops the
   *non-adiabatic* parts of both (a) and (b), and keeps every resonant
   pathway (see the derivation). "Non-resonant when tuned to a Δm = 0 pair of
   the initial state" is true. The general statement is that every dropped
   term is detuned by at least the photon energy. Off-resonant sideband
   *intensities* are only qualitative: −37% / +10% in the toy above.
   The anchor checks only μ² scaling and ν = ±2 suppression, so it is
   unaffected. The ledger should say so explicitly.
4. **Nit.** `periodic_fiber_zones` cycles caller-supplied phases and does
   not sample $U(\rho,z)$ itself. The Periodic-limit paragraph says it
   "samples … at slice midpoints". The anchor builds midpoint phases, which
   is correct.
5. **Nit.** The Grigoryan model also has the scalar $\Delta\cos k_sz$ term of
   Eq. (2.4). It cancels from the ν = ±1 amplitude (Eq. (3.12) contains μ
   only), so omitting it in the analog is correct but unstated.

### Verdict

- **Claim**: `cr-axial-z-dependent-hamiltonian` —
  `axial_fiber_unraveling.py::{FiberZones,periodic_fiber_zones,_ZonedEngine,_make_engine,_interface,_trajectory,unravel_fiber_outgoing}`,
  `axial_fiber_regression.py::{_Companions.interface,fiber_on_axis_source}`.
  Sources: paraxial continuity; the record law of Castin, Dalibard & Mølmer,
  Eqs. (4) and (8); Grigoryan et al. 2001, Eqs. (2.4), (2.6b), (2.20), (3.7)
  and (4.3).
- **Filters**: units pass; limits pass (constant sequence bitwise, identical
  copies, jump-free, ground state under drive); signs and conventions pass.
- **Re-derivation**: `matches`. The interface map, one threshold, the
  terminal branch, the companion with no divisor, the eigen-frame transport,
  zone-local currents and the periodic slicing error all agree. The
  Grigoryan numbers reproduce.
- **Verdict**: `rederived`.
