# `cr-planar-fiber-regression`: on-axis two-time source of the planar fibre model

**Status:** task-owner derivation (2026-11-11, #1056), awaiting fresh-context
verification.
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
