# Mixed elastic equivalence and cost

Equivalence of `transport_mode="mixed"` with detailed ELSEPA transport, its
cost, and Geant4's mixed schemes on the same scenes (#894). The derivation is
in [Mixed elastic transport with a random hinge](mixed-elastic-transport.md).
Claims: `mixed-elastic-split`, `mixed-elastic-soft-moments`,
`mixed-elastic-hinge`.

## Scenes and protocol

The six #821 anchors: Si 300 keV on 100 µm, W 300 keV on 10 µm, Si 1 MeV on
1 mm, W 1 MeV on 100 µm, Si 3 MeV on 1 mm and W 3 MeV on 100 µm, at normal
incidence on the exact CUDA kernel (RTX 5080). Scores come from
`checks/geant4_performance/pyrite_bench.py`. Two families are tested, each
against its own detailed reference:

- **matched**: continuous stopping and the #821 bounds;
- **default**: `inelastic_model="shell-soft-hard"` with a 50 eV cutoff.

The protocol, `checks/mixed_elastic/protocol.json` (v2), was frozen from
planning pilot job 4076 before any held-out sample existed. It fixes the arms
($C_1=C_2=0.05$ and $C_1=C_2=0.2$), held-out seeds 97001–97004, populations
from a 6 GPU-hour budget, and 446 headline comparisons. A comparison passes
when its whole simultaneous interval lies inside a margin of 2% of the pilot
reference mean, and fails when the interval lies wholly outside it. Otherwise
it is unresolved. The interval is the #838 intersection of the Maurer-Pontil
empirical Bernstein bound and an exact mismatch bound, at 95% joint confidence
over the family. The headline rule follows #821: fractions and exit bins below
2%, and photon bands below $10^{-3}$ per primary, are reported but not tested.

The v2 protocol tests the default step limit of a hundredth of the layer
thickness. The superseded v1 protocol tested a tenth (jobs 4040–4049: 200
passed, 246 unresolved, 0 failed). Its diagnostics found the shifts that moved
the limit; see [the derivation's energy control](mixed-elastic-transport.md#decisions).

## Held-out results

Jobs 4079–4087 run the v2 family and job 4088 analyzes it.

## Geant4 mixed schemes

Geant4 11.4.2 `g4bench` on the same six scenes, with matched bounds (`eIoni` with
fluctuations off, `eBrem`, a 1 keV photon threshold, no delta rays, step
function (0.2, 1 mm)). Three elastic arms:

- **SS**: single scattering with the `G4eDPWACoulombScatteringModel`, the
  reference;
- **WVI**: WentzelVI multiple scattering below a 0.15 rad limit angle, single
  Coulomb scattering (`G4eCoulombScatteringModel`) above it;
- **GS**: Goudsmit–Saunderson condensed history as in
  `G4EmStandardPhysicsGS` (`fUseSafetyPlus`, skin 3, range factor 0.08), with
  no explicit hard events.

Job 4033, three seeds of 200,000 histories per arm and scene, 32 threads on
an AMD Ryzen 9 9950X3D. The table gives the warm time per history of SS, each
mixed arm's speedup over SS, its steps per history, and its largest shift from
SS among the backscattered, transmitted and stopped fractions and the
transmitted energy, among those with $|z|>3$. $z$ uses independent standard
errors.

| Scene | SS µs | SS steps | WVI speedup | WVI steps | WVI largest shift | GS speedup | GS steps | GS largest shift |
|---|---|---|---|---|---|---|---|---|
| Si 300 keV | 29.6 | 1,744 | 5.7× | 34 | none | 8.5× | 21 | stopped −4.2% (z −5.0) |
| W 300 keV | 37.0 | 2,204 | 1.4× | 227 | stopped +20.5% (z +18.9) | 10.4× | 24 | stopped −9.5% (z −9.4) |
| Si 1 MeV | 203.7 | 12,608 | 22.2× | 150 | backscattered −3.8% (z −7.1) | 54.9× | 33 | backscattered −2.3% (z −4.2) |
| W 1 MeV | 262.2 | 16,477 | 4.4× | 1,105 | transmitted energy −11.5% (z −33.3) | 64.3× | 41 | transmitted energy +5.3% (z +14.6) |
| Si 3 MeV | 94.6 | 5,781 | 31.9× | 6 | backscattered −20.8% (z −12.0) | 25.8× | 19 | backscattered −12.2% (z −6.9) |
| W 3 MeV | 139.2 | 8,590 | 22.5× | 44 | backscattered +2.0% (z +4.1) | 40.4× | 23 | backscattered −7.9% (z −16.9) |

Both Geant4 schemes shift terminal tallies well beyond the 2% margin of this
study on five of six scenes. These shifts combine the condensed scheme with a
different angular model: WVI's single-scattering part uses the Wentzel
cross section, not DPWA, and GS uses its own tables. Geant4's mixed schemes are
therefore not equivalent to its single scattering at this level. PyRITE's
scheme keeps the detailed ELSEPA sampler for hard events and closes the soft
moments on the same law, which is what the held-out family tests.

Per scene and observable, including photon bands and the cone:
[geant4-4033-summary.json](../check-records/mixed-elastic/geant4-4033-summary.json).
