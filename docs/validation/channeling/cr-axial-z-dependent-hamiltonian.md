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
$z_{-1}=0$ and the last zone running to the exit. Each zone carries its own
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
one. Those terms are non-resonant when the drive is tuned to a $\Delta m=0$
pair of the initial state. In a prototype (below) a variant that also inserts raising elements
moved the inverse 1s→2p line by 1.5% at 4 µm. That variant is not
implemented: it adds negative-frequency leakage through the sharp slab
window, of the same order as the line at 2 µm.

### Periodic limit

`periodic_fiber_zones` samples a smooth $U(\rho,z)$ at slice midpoints, $N$
slices per period. The staircase has fundamental amplitude
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
$\varepsilon_3-\varepsilon_1=1.92\varepsilon_*$), at 10 MeV with μ = 0.01. They
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
