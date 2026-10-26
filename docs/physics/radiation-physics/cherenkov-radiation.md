# Bulk Cherenkov radiation (Frank–Tamm)

Standalone internal source term for conventional Cherenkov emission of one
charged particle in a homogeneous, isotropic, weakly absorbing dielectric
(#587). The dielectric response is an explicit input. There is no catalog
lookup, transport coupling, escape, or detector response. Entry point:
`pyrite.montecarlo.spectrum.cherenkov.frank_tamm_yield`.

## Source equation

`Validation: frank-tamm-homogeneous-dielectric`

A particle of charge $ze$ moves at constant speed $\beta c$ along a straight
line through an unbounded nonmagnetic medium with relative permittivity
$\varepsilon(E)$. With the passive $e^{-i\omega t}$ convention
($\operatorname{Im}\varepsilon\ge0$) the refractive index is
$n(E)=\operatorname{Re}\sqrt{\varepsilon(E)}$ (principal root). The photon
yield per unit path and photon energy is the Frank–Tamm result
(Frank and Tamm 1937; Jackson, *Classical Electrodynamics*, 3rd ed., §13.5):

$$
\frac{d^2N}{dx\,dE}=\frac{\alpha z^2}{\hbar c}
\left(1-\frac{1}{\beta^2n^2(E)}\right)\Theta\bigl(\beta n(E)-1\bigr),
\qquad
\cos\theta_C(E)=\frac{1}{\beta n(E)} .
$$

$\alpha/\hbar c=369.81\ \mathrm{eV^{-1}\,cm^{-1}}$. All photons of energy $E$
lie on the cone $\theta=\theta_C(E)$ about the velocity, so the angular
density is $d^3N/dx\,dE\,d\theta=(d^2N/dx\,dE)\,\delta(\theta-\theta_C)$.
The equivalent form $1-1/(\beta^2n^2)=\sin^2\theta_C$ is the usual one.

`frank_tamm_yield(passage, energy_eV, epsilon)` returns a
`BulkEmissionYield`: `yield_per_m_per_eV` (photons m⁻¹ eV⁻¹), the cone
`cos_theta`/`theta_rad` (NaN below threshold), `above_threshold`,
`photons_per_eV()` for the `ChargedPassage.path_length_m`, and
`angular_histogram(theta_edges_rad)`, which deposits each energy's yield in
the polar bin holding $\theta_C$. `epsilon` is an array broadcastable to the
energies or any object with an `epsilon(energy_eV)` method, such as
`materials.optical.OpticalResponse`.

## Threshold and dispersion

The threshold is local in energy: photons of energy $E$ are emitted only
where $\beta n(E)>1$, i.e. $\beta>\beta_t(E)=1/n(E)$
(`cherenkov_threshold_beta`). In a dispersive medium the emitted band is
therefore the set of energies where $n(E)>1/\beta$, not a single cut on
$\beta$. Exactly at $\beta n=1$ the yield is zero, and just above it rises
linearly, $\sin^2\theta_C\approx2(\beta n-1)$. Media with $n\le1$ (vacuum,
or X-ray energies where $\varepsilon=1-\delta$) never radiate.

## Assumptions and validity

- Unbounded homogeneous isotropic nonmagnetic medium ($\mu=1$), scalar
  local permittivity.
- Constant $\beta$ along a straight path, no recoil
  ($E\ll\beta\gamma mc^2$). Slowing down is handled by the caller as
  segments of nearly constant $\beta$ (Geant4 does the same per step).
- Weak loss: the loss tangent $\operatorname{Im}\varepsilon/\operatorname{Re}\varepsilon$
  must not exceed `max_loss_tangent` (default $10^{-3}$). Absorption enters
  only through $n=\operatorname{Re}\sqrt\varepsilon$, which differs from
  $\sqrt{\operatorname{Re}\varepsilon}$ only at second order in the loss
  tangent. The absorptive corrections to the lossless Frank–Tamm form are
  not modelled; the gate bounds them to $O(\tan\delta)$ away from threshold.
  The sharp lossless threshold is kept, although loss smears it over
  $|\beta n-1|\lesssim\tan\delta$. Larger loss tangents, active media
  ($\operatorname{Im}\varepsilon<0$) and $\operatorname{Re}\varepsilon\le0$
  are refused.
- The result is an *internal source yield*. Photon absorption along the
  emission path, refraction or total internal reflection at the exit face,
  and detector response are not included.

## Limiting cases

- Nondispersive medium: the yield is flat in energy,
  $369.81\,z^2\sin^2\theta_C$ photons eV⁻¹ cm⁻¹.
- $\beta\to1$: $\cos\theta_C\to1/n$, the maximum cone angle.
- $\beta\le1/n$ or $n\le1$: zero yield, no cone.
- Charge enters as $z^2$; its sign does not.

## Scope boundary

Out of scope, and owned elsewhere:

- **Transition radiation and interfaces.** A finite slab changes the field
  at its faces; the interface terms are
  [transition radiation](transition-radiation.md) and
  [optical planar TR](optical-transition-radiation.md). The planar-stack TR
  solve already contains the Cherenkov pole of a Cherenkov-active layer, so
  this bulk source must not be added to a TR result for the same layer
  without an explicit double-counting decision.
- **Finite-slab escape and absorption.** No escaped or detected yield is
  implied; [#692's proposed production contract](../../research/physics/production-cherenkov.md)
  defines the geometry, TR refusal and validation gates for that extension.
- **Strong absorption and resonant X-ray Cherenkov emission** near
  absorption edges, which need the complex susceptibility (see the in-medium
  resonance discussion in [photon escape and dispersion](photon-escape-and-dispersion.md)).

## Validation

Analytic checks (`tests/montecarlo/test_cherenkov.py`): prefactor,
nondispersive closed form and cone, $z^2$ scaling, threshold and linear
onset, vacuum and $n<1$, $\beta\to1$ cone, a dispersive closed-form band
integral, per-energy threshold, weak-loss continuity, and refusals.
Independent Monte Carlo reference: Geant4 11.4.2 `G4Cerenkov`
(`checks/cherenkov_geant4`, compared in
`tests/montecarlo/test_cherenkov_geant4_reference.py`). Ledger:
[`frank-tamm-homogeneous-dielectric`](../../validation/ledger-core-coherent-physics.md#frank-tamm-homogeneous-dielectric).
