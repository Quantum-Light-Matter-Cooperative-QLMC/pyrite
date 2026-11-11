# `cr-planar-fiber-regression`: on-axis two-time source of the planar fibre model

**Status:** task-owner derivation (2026-11-11, #1056); fresh-context verdict
`rederived` (2026-11-11, below); ledger `anchored`.
Code: `src/pyrite/montecarlo/channeling/planar_fiber.py::planar_on_axis_source`.
The estimator is the anchored companion regression of
`cr-axial-fiber-regression`
(`axial_fiber_regression.fiber_on_axis_source`,
`fiber_thickness_spectra`, `fiber_lowering_currents`), used unchanged on the
planar channels of `cr-planar-fiber-unraveling`.

## Why

Production planar emission (`cr-line-shape`, `cr-yield-assembly`,
`cr-capture-yield`) is a secular model. It integrates line yields from the
density matrix and attaches a mode profile from the secular coherence
equation of each transition, with capture clustered by initial energy. It
drops the coherence between different transitions (non-secular cross
terms) and replaces the finite-depth two-time correlation by a stationary
mode profile. The planar fibre model is the same Lindblad dynamics on a
continuous carrier, and the companion estimator gives its two-time
correlation, and hence the spectrum, without either approximation. That
makes it a reference for how much the secular model drops.

## Derivation

**Source law.** For the fibred absorbing model with map $\Lambda$ (no-jump
propagation, drawn carrier-moving jumps, retained projection),
`cr-axial-fiber-regression` gives, for recorded depths $z_j\le z_i$,
$G_{ab}(z_i,z_j)=\mathrm{Tr}[J_a\Lambda_{z_i-z_j}(\rho(z_j)J_b^\dagger)]$,
estimated without bias by each primary history and its companions, and the
on-axis spectrum
$S(E)=\frac{\alpha E}{4\pi^2(\hbar v)^2}\sum_a\sum_{ij}v_{ia}G_{ij,aa}v_{ja}^*$,
$v_{ia}=w_iA_i(E)e^{iE(1-\beta)z_i/\hbar v}$ (trapezoid weights $w_i$,
amplitude attenuation $A$). Nothing in that derivation depends on the cell
or on the dimension of the order window. It needs (i) the channels'
generator, (ii) the currents $J_a$ in each fibre's retained eigenbasis,
and (iii) a Hamiltonian kinetic term $\hbar^2k^2/2\gamma m$, so that $J$ are
the velocities. `cr-planar-fiber-unraveling` supplies (i). The planar
`_hamiltonian` has kinetic term `HBAR2_2M / gamma * (k + n g)^2`, which
gives (iii).

**Planar currents.** The planar window orders are $(n,0)$, $n=-M..M$, and the
carrier is $(\kappa_x,\kappa_y)$ with $\kappa_y$ the accumulated
out-of-plane momentum (`cr-planar-fiber-unraveling`). The absolute
transverse wavevector of order $n$ is $(\kappa_x+ng,\kappa_y)$, so
`fiber_lowering_currents` builds

* $J_x=\frac{\hbar c}{\gamma mc^2}V^\dagger\mathrm{diag}(\kappa_x+ng)V$,
* $J_y=\frac{\hbar c}{\gamma mc^2}\kappa_yV^\dagger V=\frac{\hbar\kappa_y}{\gamma mc}\mathbb 1$.

Then it keeps the positive-frequency elements $[f,i]$ with
$E_i-E_f>0$. $J_y$ is diagonal, so its positive-frequency part vanishes
identically: only the in-plane current radiates, as in the planar model
(whose dipoles are $x_{fi}$ only). With $[H,x]=-i\hbar p_x/\gamma m$
(`cr-dipole-matrix-element`),
$\langle f|J_x|i\rangle=\frac{\hbar}{\gamma mc}\langle f|k_x|i\rangle=
\frac{i(E_f-E_i)}{\hbar c}x_{fi}$. The fibre's velocity currents are
therefore the production dipoles in the velocity gauge, and
$|J_{fi}|^2=(\Delta E/\hbar c)^2|x_{fi}|^2$.

**Dropped constant.** $H(\kappa)=H_x(\kappa_x)$ omits
$\hbar^2\kappa_y^2/2\gamma m$, a fibre-global phase. Every record pairs a
primary and a companion in the same fibre, $\langle w|J_x|u\rangle$, so the
phase $e^{-i\epsilon_y(z_i-z_j)/\hbar v}$ that both acquire since insertion
cancels. Jumps change $\kappa_y$ for both alike.

**Partition.** `boundary_eV` = planar barrier keeps final states strictly
below the barrier: bound→bound lines plus capture from retained
above-barrier states (barrier ≤ $E_i$ < barrier + margin). This is the
partition of `channeling_spectrum` with `include_capture=True` and
`final_states="bound"`. The in-cut free→free emission is excluded on both
sides. Initial states are the fibre's retained states (below the cut), as
the solver's tracked subspace.

**Kinematics.** Photons along the beam axis ($\theta=0$). The Doppler factor
is $1-\beta$ for every record: the electron's own out-of-plane angle
$\theta_y=\kappa_y/k$ does not enter, as in `channeling_spectrum` at
$\theta=0$ with zero divergence and aperture. Fixed energy (no stopping),
as the solver at zero stopping.

## Assumptions

`cr-planar-fiber-unraveling` (first Born, isotropic Einstein displacements,
scalar local-density electronic kicks, fixed energy, absorbing cut at
barrier + margin) and `cr-axial-fiber-regression` (absorbing no-return
model: emission after an electron leaves the retained subspace, and its
interference with earlier emission, are not represented; on-axis
observation; velocity/c currents in each fibre's retained eigenbasis).
Dipole order (no quadrupole or magnetic lines).

## Limiting cases

* No scattering (`mechanisms=()`): the estimator is deterministic and equals
  the dense quantum regression with $H=\mathrm{diag}(E)$ and no jumps
  (`cr-quantum-regression-correlation`).
* $J_y\equiv0$ after the positive-frequency mask, for any $\kappa_y$.
* A single isolated two-level line without scattering gives
  $G=e^{-i\Delta(z-z')/\hbar v}$ and the finite-thickness sinc² profile at
  $E=\Delta/(1-\beta)$. The secular model instead has a zero-width mode
  there, so the two models differ at small depth by construction.

## Checks

`tests/montecarlo/quantum/test_channeling_planar_fiber_regression.py`:

* in-plane currents vs `transverse_states(...).dipole_ang` through
  $|J_{fi}|=|\Delta E||x_{fi}|/\hbar c$ at three carriers (30 transitions
  each, relative 1e-9), $J_y$ zero to rounding;
* no-scattering source vs `regression_correlation` (correlation to 1e-9 of
  its scale, spectrum to 1e-9 relative), with the bound-final partition
  checked.

The estimator's unbiasedness with carrier-moving jumps is anchored by
`cr-axial-fiber-regression` (dense two-carrier GKSL oracle) for arbitrary
fibre channels. The planar channels' generator is anchored by
`cr-planar-fiber-unraveling`.

Research comparison with the production secular model:
`checks/planar_fiber_spectrum.py`, results in
[the dechanneling boundary note](../../research/physics/dechanneling-boundary.md).

## Fresh-context verification (2026-11-11)

Verifier: fresh context, independent of the implementation. Order followed:
ledger row and docstrings first, then an independent derivation (below),
then the implementation (`planar_on_axis_source`, `fiber_lowering_currents`,
`_Companions`, `signed_on_axis_spectrum`, `planar_fiber_inputs`,
`bloch._hamiltonian`, `TransverseStates.dipole_ang`,
`emission.emission_lines` masks, `DensityMatrixSolver._basis`). Code
reviewed at `8179040b0`. The probes ran at `8a54d2851`/`5ea0d32bc`, which leave
every reviewed file unchanged (`git diff --stat 8179040b0 HEAD` over
`src/pyrite/montecarlo/channeling/`, `src/pyrite/materials/channeling/`, the
anchor test, the check script, this page and the ledger part is empty).

**Verdict: `rederived`.** Nothing is blocking. Minor findings are listed at the end.

### Independent derivation

**Estimator scope.** The companion estimator of `cr-axial-fiber-regression`
needs only three things. (a) A carrier-block-diagonal generator: each fibre
$\kappa$ has its own no-jump generator $H_{\rm eff}(\kappa)$, and Kraus jumps
map fibre $\kappa$ to fibre $\kappa+q$. (b) Currents that act within one fibre.
(c) The MCWF record law. None of these refers to the cell shape, the
dimension of the order lattice, or a $y$ harmonic. The planar model supplies
(a) with window orders $(n,0)$, $n=-M..M$. The $y$ reciprocal vector
$g_y=10^9$ Å⁻¹ multiplies only $n_y=0$, and $\kappa_y$ is a label, as
`cr-planar-fiber-unraveling` (rederived) established. The estimator therefore
applies unchanged. In the code, `_Companions.orders` is the `ij` meshgrid of
$(-M..M)\times\{0\}$, i.e. $(n,0)$ in the same order as the rows of
`spectrum(kappa)`, which come from `eigh(_hamiltonian(..., orders=-M..M))`.

**Velocity currents.** The planar `_hamiltonian` is
$H_x=\hbar^2(\kappa_x+ng)^2/2\gamma m+U_{n-n'}$. Its velocity is
$v_x=\partial H/\partial(\hbar k_x)=\hbar(\kappa_x+ng)/\gamma m$, so
$J_x=v_x/c=\hbar c(\kappa_x+ng)/\gamma mc^2$. This is diagonal on the plane
waves, and in the eigenbasis it is $V^\dagger{\rm diag}(\cdot)V$. Since
$[H_x,x]=-i\hbar\,v_x$, for $f\ne i$

$$
\langle f|J_x|i\rangle=\frac{1}{c}\langle f|v_x|i\rangle
=\frac{i}{\hbar c}\langle f|[H_x,x]|i\rangle
=\frac{i(E_f-E_i)}{\hbar c}\,x_{fi}.
$$

That is the claimed sign and factor. Out of the plane there is no $y$
potential, so the kinetic term in $y$ is $\hbar^2\kappa_y^2/2\gamma m$ times
the identity on the fibre, and $J_y=\hbar\kappa_y/(\gamma mc)\,\mathbb 1$.
Its positive-frequency part, $\sum_{E_f<E_i}|f\rangle\langle f|J_y|i\rangle\langle i|$,
vanishes because $\langle f|i\rangle=0$. More generally, any constant shift of
the momenta adds a multiple of the identity, which is invisible to
off-diagonal elements. Records live in a single fibre, so no inter-fibre
$J_y$ element arises.

**Dropped constant.** Write $\epsilon_y(\kappa)=\hbar^2\kappa_y^2/2\gamma m$.
In each segment the no-jump propagator in fibre $\kappa$ is
$e^{-i\epsilon_y\Delta z/\hbar v}\,e^{-iH_{\rm eff,x}\Delta z/\hbar v}$. The
regressed operator $X=\Lambda(\rho J^\dagger)$ is unravelled as
$|u\rangle\langle w|$. Both factors are propagated by the same operator in the
same fibre, and a drawn jump $K$ acts as $K|u\rangle\langle w|K^\dagger$, so
both move to the same new $\kappa_y$. The phases
$e^{-i\epsilon_y\Delta z}$ and $e^{+i\epsilon_y\Delta z}$ therefore cancel in
$\langle w|J_x|u\rangle$, segment by segment, for any sequence of
$\kappa_y$-changing jumps. Loss rates and jump probabilities depend on norms
only, so the constant does not affect them either. The constant also cannot
shift a Bohr frequency, because $f$ and $i$ share $\kappa_y$. Its only
physical trace is the longitudinal kinematics: the electron direction
$\theta_y=\kappa_y/k$ changes the Doppler factor and the angular pattern. The
declared on-axis assumption drops that (see the minor findings).

**Partition.** Production builds `emission_lines` with
`final_states="bound"` and `include_capture=True`. Its masks are
`initial_ok = bound | (subspace & ~bound)` (= `subspace`), `final_ok = bound`,
with `bound = E < barrier` and `subspace = E < barrier + margin`, where
`barrier` = `profile.extrema()[1]` in `DensityMatrixSolver`. On the fibre
side, the retained states are those with $E<$ `cutoff` $=$ `barrier + margin`
(`planar_fiber_inputs`, same `extrema()` and the same 300 eV default margin).
`fiber_lowering_currents(boundary_eV=barrier)` keeps $[f,i]$ with $E_f<$
`barrier` and $E_i-E_f>10^{-10}$ eV. Hence lowering, bound final, and any
retained initial: bound→bound plus capture from
$\mathrm{barrier}\le E_i<\mathrm{barrier}+\mathrm{margin}$. In-cut
free→free is excluded, and so are upward and diagonal elements. The
subspaces match: production's tracked set at the solver's discrete
$\kappa_x$ is the fibre's retained set at continuous $\kappa_x$.

**Kinematics and units.** For a record without jumps, $G(z_i,z_j)=\sum_f
a_f(z_i)a_f^*(z_j)$ with $a_f(z)=\sum_i J_{fi}c_ie^{-i(E_i-E_f)z/\hbar v}$. On
axis ($\mathbf n=\hat z$), the classical radiation integral
$d^2I/d\omega d\Omega=(e^2\omega^2/4\pi^2c)|\int\boldsymbol\beta_\perp
e^{i\omega(t-z/c)}dt|^2$, divided by $\hbar^2\omega$, with $dt=dz/v$, gives

$$
\frac{d^2N}{dE\,d\Omega}=\frac{\alpha E}{4\pi^2(\hbar v)^2}
\sum_a\Big|\int dz\,J_a(z)\,e^{iE(1-\beta)z/\hbar v}\Big|^2,
$$

in photons/(eV sr electron). This has the code's prefactor and phase:
`signed_on_axis_spectrum` uses $E/(\gamma^2(1+\beta))=E(1-\beta)$. The
Doppler factor is the beam's $1-\beta$ for every record. That matches
`channeling_spectrum` at $\theta=0$ with zero divergence and aperture, whose
centres $\Delta/(1-\beta\cos\theta_e)$ also ignore the electron's own
scattering angle.

All five points agree with the task-owner derivation and with the code.

### Numerical checks (independent of the implementation helpers)

Probe scripts live in the session scratchpad and are not committed. All runs
are Si(110), 293.15 K, 17 MeV.

| Check | Result |
|---|---|
| Velocity currents vs **position-integral** dipoles $\int\psi_f^*x\psi_i$ over the cell centred on the well ($M=16$, $\kappa_x=0$; this is not `dipole_ang`, which uses the same $\langle f\lvert k\rvert i\rangle$ identity) | Complex values agree in sign and phase. Relative difference: $1.8\times10^{-5}$ (0→1), $1.5\times10^{-3}$ (0→3), $5.9\times10^{-3}$ (1→2), 11% (2→3). The residual grows as state 3 (just below the 6.62 eV barrier) leaks out of one cell, where the position operator is ill-defined. Parity-forbidden 0→2 and 1→3 are $<10^{-17}$. |
| $J_y$ at $\kappa_y=0.03$ Å⁻¹ | $\max\lvert J_y\rvert=1.5\times10^{-21}$, against $\lvert J_x\rvert\sim5\times10^{-4}$ |
| No scattering: `planar_on_axis_source` vs my own closed form (own $H$ from `potential_eV`, own CODATA constants, own currents, bound mask and photon law $\sum_f\lvert\sum_zw_ze^{iE(1-\beta)z/\hbar v}a_f(z)\rvert^2$) | max abs diff / peak: $1.2\times10^{-13}$ ($\theta=20$ µrad, $M=12$, 0.5 µm), $3.3\times10^{-13}$ (60 µrad, $M=16$, 2 µm), $5.2\times10^{-13}$ (1 mrad, $M=16$, 2 µm, capture lines populated). Totals 9.0144e-3, 3.6113e-2, 2.8383e-2 photons/sr on both sides. |
| Trapezoid vs exact $\int_0^L$ (finite-thickness sinc² limit) at $dz=50$ Å | max relative $1.7\times10^{-4}$, $2.8\times10^{-5}$, $7.1\times10^{-5}$ |
| $\kappa_y$ label invariance with scattering (TDS + electronic, $M=16$, 2 µm, 8 primaries, seed 3; entry $\kappa_y=0$ vs 0.05 Å⁻¹; histories with 0–4 TDS and 3–9 kick jumps, final $\lvert\kappa_y\rvert$ up to 46 Å⁻¹) | events identical; spectrum max diff $3.8\times10^{-16}$ of peak, correlation $6.5\times10^{-16}$ |
| Anchor test node run | `test_channeling_planar_fiber_regression.py`: 4 passed |

**Fault injection** (wrapper kwargs monkeypatched, then the anchor test
functions called directly):

| Fault | Detected by |
|---|---|
| `boundary_eV=None` (free→free included) | no-scattering test (AssertionError) |
| `boundary_eV` = cutoff | no-scattering test |
| `boundary_eV` = barrier − 2 eV | no-scattering test |
| carrier passed as `(0, 0)` | no-scattering test |
| `hbar_v` × 1.001 | `_admit` consistency (ValueError) |
| kinetic energy × 1.01 (Doppler) | `_admit` consistency (ValueError) |
| diagonal $J_y$ kept (mask bypassed) | currents test, $\kappa_y=4\times10^{-3}$ case only |
| $\gamma$ dropped from the current | currents test, all carriers |
| raising instead of lowering elements | currents test, all carriers |
| carrier dropped from the current momenta | **not detected, and correctly so**: it adds $\kappa_x\mathbb 1$, which has no off-diagonal part (the same identity that removes $J_y$) |

**Ensemble comparison with production.** 1600 primaries (ids 0–1599, seed 0)
of `checks/planar_fiber_spectrum.py run demo_17` ($M=32$, $dz=50$ Å, 300 eV
margin, absorption off), against `analyze` (`channeling_spectrum`, 16 cells,
TDS + electronic, $\theta=0$). Fates: 1541 retained, 49 outgoing, 10
unresolved. Fibre/production yield ratio per line window, with batch SEM:

| Thickness | 4→3 (2.34 keV, 82–88% capture) | 3→2 (7.01 keV) | 2→1 (10.2 keV) | 1→0 (18.5 keV) | Total |
|---|---:|---:|---:|---:|---:|
| 0.5 µm | 0.716(10) | 1.753(16) | 0.877(10) | 1.301(24) | 1.024 |
| 1 µm | 0.894(16) | 1.083(16) | 0.991(14) | 1.081(19) | 1.016 |
| 2 µm | 0.895(21) | 0.983(16) | 1.042(17) | 1.055(14) | 1.028 |

At 2 µm the strong bound lines agree within 6%, and the band total within 3%.
The capture-dominated 4→3 window is 10.5% low. At 0.5 µm the departures are
large, as expected from the non-stationary finite-depth correlation, which
the secular model replaces by a stationary mode profile. RMS widths are
1.1–1.6 times production's and centroids shift by +0.1 to +0.75 keV, both
shrinking with depth. These are model-difference findings for the research
note; they are not defects of this claim.

### Minor findings (non-blocking)

1. **On-axis assumption, quantified.** In the 2 µm sample, final
   $\lvert\kappa_y\rvert$ ranges over 0.4–46 Å⁻¹ ($k=8885$ Å⁻¹, so
   $\theta_y\le5.2$ mrad). For individual records,
   $\gamma^2\theta_y^2$ reaches 0.03, i.e. a ~3% Doppler shift and ~6%
   pattern factor; typical values are $\lesssim10^{-3}$. This is the declared
   assumption, and production at zero divergence makes the same one, so the
   comparison is like-for-like. It is still an absolute-spectrum caveat for
   thick targets.
2. **The `dipole_ang` oracle is not fully independent.** It derives $x_{fi}$
   from the same $\langle f|k|i\rangle$ relation, so the currents test
   checks eigenvector, energy and unit consistency with production, not the
   velocity-gauge identity itself. The position-integral probe above covers
   that identity. Consider pinning one deep pair (0→1, agreement
   $2\times10^{-5}$) by position integral.
3. **The no-scattering anchor builds its oracle currents with
   `fiber_lowering_currents`.** A fault inside that function is caught only
   by the currents test. The diagonal-$J_y$ fault is caught only at the one
   carrier with $\kappa_y\ne0$. This is adequate, but the coverage is thin.
4. **The `diagonal` and free→free current parts are excluded on both sides.**
   This includes the zero-frequency steps of $J_y$ at $\kappa_y$-changing
   jumps, which are the incoherent-bremsstrahlung-like part of the out-of-plane
   current. The claim's wording "no positive-frequency part" is exact for the
   lowering-current source. That scattering radiation is owned elsewhere and
   is not part of this source.
5. **The research harness bypasses the wrapper.** It calls
   `fiber_thickness_spectra` with `boundary_eV=inputs.barrier_eV` directly,
   not `planar_on_axis_source`. The arguments are identical, but a future
   wrapper edit would not propagate to the study.

### Suggested ledger change

`cr-planar-fiber-regression`: `unverified` → `rederived`. Since the anchor is
green, the task owner may then advance it to `anchored`. Add to Notes:
"2026-11-11 fresh-context verdict `rederived` (write-up §Fresh-context
verification): velocity-gauge identity confirmed against position-integral
dipoles (2e-5 deep pair, sign included); no-scattering source equals an
independent closed form to 5e-13; κ_y label invariance with κ_y-changing
jumps to 6e-16; boundary/carrier/γ/raising/J_y faults detected; 1600-primary
demo_17 ensemble vs `channeling_spectrum` at 2 µm: strong lines 0.98–1.06,
capture-dominated 4→3 0.90, total 1.03." Never `signed-off`; human sign-off
remains in #277.
