# Mixed elastic transport

Detailed transport ([Elastic scattering](elastic-scattering.md)) samples every
elastic collision. On a long high-$Z$ track most of those collisions turn the
electron by a tiny angle: W at 1 MeV on 100 µm takes about 15,000 rows per
history. `Physics(transport_mode="mixed")` follows PENELOPE's class II
scheme{cite:p}`salvat2024penelope` instead. Collisions that deflect the
electron strongly stay explicit, and the many weak ones between them are
replaced by one artificial deflection per step, applied at a random point of
the step, the hinge. The default is `transport_mode="single-scattering"`, and
detailed transport remains the reference.

```python
import pyrite as pr
from pyrite.transport import simulate_trajectories

physics = pr.Physics(transport_mode="mixed")  # C1 = C2 = 0.05
physics = pr.Physics(transport_mode="mixed", mixed_c1=0.1, mixed_c2=0.05, mixed_max_step_um=5.0)
tracks = simulate_trajectories(beam, target, physics=physics)
```

Mixed transport serves trajectory-level tallies: terminal fractions, exit
energy and angle, and deposition. `pyrite.simulate`, sweeps and the CXR line
scorer refuse mixed cases (#990), so spectrum runs stay on detailed transport.
Profiles and the CLI accept the same settings as `--transport-mode`,
`--mixed-c1`, `--mixed-c2` and `--mixed-maximum-step-um`, and the spectrum
commands then refuse the case.

## Model

The tabulated ELSEPA angular law of each element is split at a cutoff. With
$\mu=(1-\cos\theta)/2$, $\lambda_{\rm el}$ the elastic mean free path and
$\lambda_1$ the first transport mean free path, hard collisions occur with mean
free path

```{math}
:label: eq-mixed-hard-mfp

\lambda_h=\max(\lambda_{\rm el},\,C_1\lambda_1),
```

so the mean soft deflection between hard collisions,
$\langle\mu\rangle_{\rm soft}$, is at most $C_1$. Where
$\lambda_{\rm el}\ge C_1\lambda_1$ every collision is hard and that energy
range is simulated in detail. The hard collisions are the top quantiles of the
detailed sampler, so their angles come from the same table.

Soft collisions enter through their first two moments, which fix the mean and
variance of the hinge's polar deviation over a step $s$. The deviation is drawn
from PENELOPE's two-box law, which matches both moments with one uniform draw.
A step ends at the next hard elastic, hard inelastic or hard radiative event,
or at the first of three limits:

- four hard transport mean free paths;
- a mean fractional energy loss of $C_2$;
- a hundredth of the layer thickness, or `mixed_max_step_um` when that is
  shorter.

Hard events are proposed at an upper bound of each rate over the step's energy
range and accepted at the end-point rate (thinning), so the hard counts match
detailed transport as the electron slows down. A step stopped by a boundary
drops its pending hinge, and the next step starts in the new layer.

The scheme changes the trajectory structure below the cutoff angle, not the
physics model. With $C_1=0$ nothing is soft, and transport is bit for bit
detailed.

## Choosing the parameters

$C_1$ and $C_2$ accept values in $[0, 0.2]$, PENELOPE's range. Larger values
save more rows and approximate the detailed law more coarsely. The default
$0.05$ is the setting the equivalence study tests.

The step limit decides the cost on thin layers. A layer crossing takes at
least about a hundred steps, roughly two hundred rows. So the scheme saves rows
only where detailed transport would take far more: thick targets, high $Z$, or
MeV electrons. On Si at 100 keV and 1 µm, mixed transport costs more rows than
detailed transport.

PENELOPE recommends a tenth of the expected track length. With continuous
stopping, that limit shifted the 1–3 MeV transmitted-cone and backscatter
energy tallies by up to 10%. A hundredth removed the shift. An explicit
`mixed_max_step_um` can only shorten the layer limit.

## Requirements and limits

- `elastic_model="elsepa"` and `energy_model="midpoint"`, the defaults.
- Runs on the per-electron CPU core and the exact CUDA core, in both transport
  precisions. `transport_core="auto"` picks the per-electron core where it would
  otherwise pick lockstep. Any other core is refused.
- Continuous stopping and `inelastic_model="shell-soft-hard"` are supported,
  with or without the shell soft straggling sampler. Urban straggling is
  refused: its unbounded losses leave the thinning window, which lowers the
  hard counts measurably.
- Coherent emission (PXR, CBS, coherent bremsstrahlung), transition radiation,
  temporal profiles, channeling, photon transport, grooved geometry and
  detector-cone splitting are refused. The emission models read the small-angle
  structure of the flight; the others have no mixed specialization.
- Incoherent CXR lines are refused. Each row radiates a finite-time line of
  width about $\hbar/T_{\rm row}$, so the step limit, not the physics, sets the
  line width. Standard bremsstrahlung is not reachable, because spectrum runs
  always score CXR (#990).
- `transport_mode` and its parameters are part of a case's identity, recorded
  under the model marker `penelope-random-hinge.v2`. A default case carries no
  mixed keys.

## Validation

`Validation: mixed-elastic-split`, `Validation: mixed-elastic-soft-moments` and
`Validation: mixed-elastic-hinge`: see the
[derivation](../../validation/beam-transport/mixed-elastic-transport.md), its
[thinning re-verification](../../validation/beam-transport/mixed-elastic-reverification.md),
the [CUDA verification](../../validation/beam-transport/mixed-elastic-cuda-verification.md)
and the [equivalence and cost study](../../validation/beam-transport/mixed-elastic-equivalence.md)
against detailed transport and Geant4.
