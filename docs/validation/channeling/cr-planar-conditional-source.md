# Conditional planar Gaussian source and handoff (#680)

**Status: experimental production envelope, not an independently validated physical prediction.**
This source adds a new *versioned* `channeling_source` case key only when a
quantum beam is requested. Cases without this key are unchanged. It does not
admit any axial production configuration (#623/#624).

## Admission contract

Only `TransverseStateSpec.recipe("gaussian", dimension=1, waist_ang=w)` with
zero transverse center and uniform lattice-impact averaging is supported.
The incoming beam must be monoenergetic and cannot use Twiss transverse slopes
or GDF phase space. A classical macroscopic spot may still set its entrance
footprint. Coherent fixed-impact initial states, Hermite/Laguerre–Gaussian
modes, arbitrary mixtures, 2D vortex beams, and sampled arrays continue to
refuse. The Bloch source must span at most 32 occupied momentum bins (bounded
from a 6-sigma Gaussian window); a too-broad source refuses, not degrades.

The Gaussian amplitude `exp(-x²/w²)` has momentum probability
`|psi_tilde(k_x)|²/(2π)` with `sigma_k = 1/w`. We use an independent
counter-addressed normal deviate per macro-electron; its signed normal
momentum is `k_x = p * psi_effective + delta_k`, where `p` is the relativistic
electron momentum in inverse angstroms. The momentum determines a Bloch cell
`l = floor(k_x / (g/n_cells) + 1/2)`. Outside the plane-wave band the primary
starts native transport at entry; grid overflow is **not** interpreted as a
physical dechanneling event. In-band cell `l` initializes the corresponding
unit trace plane-wave density at entrance, collapses it onto the retained
subspace, and propagates a conditional Lindblad zone. Its dechanneling fate,
depth and transverse energy are sampled using the existing per-electron keyed
handoff stream, conditional on the sampled cell. All electrons with the same
cell share one deterministic conditional zone solve. This avoids the incorrect
alternative of independently drawing an unconditional handoff fate after
sampling source momentum.

Transport and CR share the same Gaussian source descriptor and effective
refracted incidence. The CR estimator uses an unconditional projected density
(the mixture over Bloch bins), as appropriate for an expected-value estimator.
Live scoring and replay regenerate deterministic electron entrance angles and
geometric censoring identically; line rescore never depends on the sampled
handoff fate. Recomputed CR does not resample primary Monte Carlo scattering.

## Approximations and validation needed

1. The Gaussian is averaged uniformly across crystal lattice impact; the
   macroscopic beam spot is independent of that microscopic phase. This is
   inappropriate for positioned coherent probes and fixed-impact electron
   wavefunctions.
2. Each continuous momentum sample is represented by its Bloch bin center for
   the conditional zone, matching the existing Bloch-grid approximation.
3. The outgoing handoff angle and sign still follow #506's effective
   transverse-energy model (depth-energy separability, symmetric sign,
   band-edge stand-in for overflow). Retaining the momentum-bin condition
   does not make these approximations exact.
4. The conditional solver cost scales with the number of distinct occupied
   bins; the physical model needs convergence checks in basis size, number of
   Bloch cells and propagation step. The 32-bin ceiling limits unexpected
   simulation cost but does **not** prove convergence.
5. Native transport conserves the original per-electron scattering RNG
   streams by injecting before launch without drawing from those streams.
   GPU and remote operation, replay, reline, radiation yields, and physics
   normalization still need targeted independent evidence on real hardware.

The fast tests pin source admission, distinct case identity, and deterministic
angle sampling; slow tests cover conditional handoff and live versus
reconstructed channeling radiation. They do not constitute scientific
validation of the physical approximations.
