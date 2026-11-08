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

Jobs 4079–4087 ran the v2 family on the mixed code of `ea0066df4` (run as
`6b0a0a37b` before a squash). Job 4100 analyzed it. Of the 446 comparisons,
**195 passed, 251 are unresolved and none failed.**

| Family | Arm | Passed | Unresolved | Failed |
|---|---|---|---|---|
| matched | $C_1=C_2=0.05$ | 54 | 56 | 0 |
| matched | $C_1=C_2=0.2$ | 54 | 56 | 0 |
| default | $C_1=C_2=0.05$ | 45 | 68 | 0 |
| default | $C_1=C_2=0.2$ | 42 | 71 | 0 |

| Observable kind | Passed | Unresolved |
|---|---|---|
| Terminal fractions | 47 | 13 |
| Transmitted energy | 20 | 4 |
| Exit angle and energy bins | 128 | 124 |
| Photon energy, photon bands and the cone | 0 | 110 |

Every unresolved comparison is unresolved because its interval is wider than
the margin, not because it lies outside it. The largest $|z|$ in the family is
3.4: a pass, Si 3 MeV transmitted angular bin 1 at $C=0.2$, +0.40%. The
largest unresolved $|z|$ is 3.2. Both are below the Bonferroni level, about
3.9 for 446 comparisons. The unresolved terminal fractions and energies sit
within ±1.1% of the reference, with interval half-widths of 0.8–1.4 margins.

The photon comparisons cannot pass at this design. Their known score bound is
the beam energy for the emitted photon energy, and for a count the number of
photons that energy conservation allows per primary, up to $E_0/10$ keV. The
empirical Bernstein range term scales with that bound. Their interval
half-widths are 2–33 margins, and 59–1,200 for the cone.
No photon comparison has $|z|>3$.

The tallies that moved the step limit are back within noise. Under continuous
stopping at $C=0.05$, the Si 1 MeV backscatter fraction above 0.9 $E_0$ is
+5.0% ($z$ 1.7), against +9.85% at a tenth of the layer. It is below the 2%
headline floor, so it is reported but not tested. The Si 3 MeV transmitted
angular bin 1, +0.51% ($z$ 5.4) at a tenth, passes at +0.22% ($z$ 1.9).

**Deviation.** One detailed reference sample, `shell-soft-hard` W 3 MeV seed
97004, hit the scene's `max_steps=100000` on 1 of 7,348 histories in every
chain job. It was rerun alone with `--max-steps 400000` (job 4099), the
transport default. The cap only refuses a history, and each history has its
own random stream, so every other history of the sample is unchanged.

Every comparison with its interval, margin and status:
[heldout-v2-summary.json](../check-records/mixed-elastic/heldout-v2-summary.json).

## Cost

Median warm request time per history over the four seeds, on the exact CUDA
kernel, and rows per history. Requests include per-primary score capture, so
absolute times are not comparable with Geant4's; the speedups are.

| Scene | Matched rows, detailed → $C=0.05$ | Matched speedup, $C=0.05$ / $0.2$ | Default rows, detailed → $C=0.05$ | Default speedup, $C=0.05$ / $0.2$ |
|---|---|---|---|---|
| Si 300 keV | 1,034 → 373 | 2.2× / 2.4× | 1,132 → 588 | 1.9× / 1.9× |
| W 300 keV | 1,976 → 655 | 2.7× / 3.3× | 2,188 → 997 | 2.4× / 2.6× |
| Si 1 MeV | 7,859 → 630 | 10.6× / 14.6× | 8,588 → 3,134 | 3.1× / 3.0× |
| W 1 MeV | 15,303 → 1,833 | 6.9× / 12.0× | 16,743 → 5,442 | 3.5× / 4.0× |
| Si 3 MeV | 5,481 → 230 | 17.6× / 18.6× | 5,846 → 1,505 | 4.5× / 4.5× |
| W 3 MeV | 8,225 → 379 | 12.3× / 13.8× | 9,285 → 2,434 | 4.1× / 4.3× |

Under `shell-soft-hard` the hard inelastic collisions, which stay explicit,
set the floor on rows, so $C=0.2$ gains little over $C=0.05$. On the thin
300 keV scenes the step limit sets it: a hundredth of the layer forces about
two hundred rows per crossing.

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
moments on the same law. In the held-out family above, no comparison failed
and no shift reached $|z|=3.9$. The trade is speed: Goudsmit–Saunderson saves
8.5–64× over Geant4 single scattering, and PyRITE's mixed scheme saves 2.2–18×
over detailed transport.

Per scene and observable, including photon bands and the cone:
[geant4-4033-summary.json](../check-records/mixed-elastic/geant4-4033-summary.json).
