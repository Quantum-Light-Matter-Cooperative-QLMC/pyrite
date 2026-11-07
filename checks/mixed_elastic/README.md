# Mixed elastic transport against detailed single scattering (#894)

Paired equivalence, cost and Geant4 comparison of `transport_mode="mixed"` on the
six #821 anchors. Results and verdicts:
[`docs/validation/beam-transport/mixed-elastic-transport.md`](../../docs/validation/beam-transport/mixed-elastic-transport.md).

| File | Purpose |
|---|---|
| `pilot.sbatch` | Planning pilot on its own seeds: sizes populations, sets margin scales. Never inferential. |
| `build_protocol.py` | Freezes `protocol.json` from a pilot: arms, seeds, populations, bounds, margins. |
| `run_arms.py`, `heldout.sbatch` | Runs the frozen design. Resumable: a finished sample is never rerun. |
| `analyze.py` | Applies `checks/equal_precision/paired_equivalence.py` to the frozen family. |
| `g4-build.sbatch`, `g4-mixed.sbatch` | Geant4 single scattering, WentzelVI plus single Coulomb, and Goudsmit-Saunderson arms. |

Scores come from `checks/geant4_performance/pyrite_bench.py`.

## Protocol

A comparison passes only when its whole simultaneous interval lies inside its
margin, fails when the interval lies wholly outside, and is otherwise unresolved.
The margin is 2% of the reference mean from the pilot, frozen before any held-out
sample exists. The interval is the #838 intersection of Maurer-Pontil empirical
Bernstein and an exact mismatch bound, at 95% joint confidence over the whole
family, with known score supports: 1 for indicators, the beam energy for energies,
and the energy-conservation count for photon bands.

Observables follow the #821 headline rule: fractions and exit bins below 2%, and
photon bands below 1e-3 per primary, are reported but not in the family. Both a
matched-bounds family (continuous stopping, as #821) and a default-physics family
(`shell-soft-hard`) are tested, each against its own detailed reference.
