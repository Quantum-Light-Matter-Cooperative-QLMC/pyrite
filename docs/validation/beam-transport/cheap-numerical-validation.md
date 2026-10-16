# Small numerical and backend validation pilot

This issue #510 follow-up tests inexpensive prerequisites for a larger numerical
study. It uses production FP64 transport, with no change to source defaults or
model markers. The [paired precision study](transport-precision-equivalence.md)
and its held-out populations remain separate.

## Frozen workload

`checks/equal_precision/cheap-validation.json` fixes six existing scenes: Si/W
300 and 800 keV anchors, thin Si at 100 keV, and thick W at 100 keV/75° incidence.
The absolute slab dimensions, material densities and 10 keV electron tracking
threshold remain those of `manifest.json`.

- GPU sensitivity: 500 primaries per seed, seeds 65001/65002; baseline
  `max_dE_frac=0.05`, hard-photon cutoff 1000 eV; separately halve the energy
  cap to 0.025 or the hard-photon cutoff to 500 eV. This gives 36 samples.
- Backend spot-check: 64 primaries per seed, seeds 66001/66002, baseline
  controls on production CPU and GPU cores. This gives 24 samples.
- Comparator configuration: six 64-primary Geant4 `empenelope` samples, seed
  67001, using the previously pinned 11.4.2/TestEm5 executable. These refresh
  material/production-cut dumps; they are not a cross-engine accuracy test.

Every PyRITE process performs a full matching warmup with seed +1000000,
then a measured request. Every block is checked for terminal completeness,
energy conservation, photon support and direction normalization. PXR/CBS call
profiling is enabled throughout. Coverage/control/audit overhead is included in
the retained diagnostic times, so no speed ratio is inferred from these runs.

Jobs 1410 (CPU) and 1411 (GPU) completed all 66 samples, without failures or
budget deferrals, in 53 seconds and 3 minutes 24 seconds respectively. They
requested one CPU each, 8/24 GiB RAM, and one RTX 5080 for the GPU job. Their
allocation limits were five and ten minutes. CPU job 1412 verified both evidence
manifests, analyzed scores, passed 40 focused tests and linted the harness.

## Checks and sensitivity

All recorded trajectories passed the frozen bookkeeping tolerances. Maximum
observed errors across warmup and measurement were 1.99e-12 keV for the primary
energy ledger, 5.69e-14 keV for a hard-photon debit and 2.23e-16 for direction
normalization. Raw score ledgers and terminal/angular partitions also passed.
Resolved controls matched the frozen study; PXR/CBS execution audits recorded
no calls. Source hashes, ordered stream fingerprints and table identities were
checked before comparing pairs.

Both materials' staged tables cover the full 10–800 keV electron range used by
these scenes:

| Table | Minimum energy (keV) | Maximum energy (keV) | Nodes |
| --- | ---: | ---: | ---: |
| ELSEPA | 0.1 | 100000 | 61 |
| SBETHE stopping | 1 | 1000000 | 391 |
| BremsLib incident grid | 0.01 | 30000 | 261 |

This checks interpolation-domain coverage, not table-density convergence or
physical-model validity throughout that domain.

In the CPU/GPU spot-check, terminal fractions, angular bins and photon counts
matched primary by primary. The largest energy-score difference was
2.24e-10 keV per primary; the largest absolute mean energy difference was
1.81e-12 keV per incident primary. These are observed spot-check errors, not
population bounds or distributional identity.

Halving the step-energy cap left all discrete scores unchanged. The largest
energy-score difference was 2.28e-13 keV per primary. Halving the hard-photon
cutoff changed some events and terminal categories; for example, W 300 keV
transmission changed by -0.003 and its ≥10 keV photon yield by -0.001 per
primary in the 1000-primary paired population. Changing the explicit/soft
partition can alter individual histories even with the same primary streams.
These small samples do not establish a systematic cutoff bias.

The fixed inference family contains 288 comparisons: 16 means × six scenes ×
three tests. It uses the previous study's finite-sample paired-interval method,
with one joint 95% budget. Diagnostic margins are 0.005 for fractions/bins,
0.005 of incident energy for energy scores, and the frozen prior Si absolute
photon margins specified in `cheap-validation.json` for every scene. These are
not the one-third target-RSE budgets. **All 288 intervals are unresolved**;
none qualifies numerical convergence or equivalence. Equal observed counts
still retain uncertainty. No selected subset is analyzed if the fixed grid is
incomplete. Geant4 samples are excluded from this coupled inference family.

## Does the step cap actually bind?

A separate CPU probe, job 1413, examines `EVENT_SUBSTEP` rows with 64 primaries
per scene, fresh seed 68001 and caps 0.05, 0.025 and 0.001. Its submitted script
fixes all eighteen requests before execution. It completed in 67 seconds under
a two-minute limit. The full warmup and trajectory checks still apply; its
scores are not pooled into the sensitivity family.

Measured energy-limited rows, after discarding warmup:

| Scene | Cap 0.05 | Cap 0.025 | Cap 0.001 |
| --- | ---: | ---: | ---: |
| Si 300 keV / 100 µm | 0 | 0 | 5589 |
| W 300 keV / 10 µm | 0 | 0 | 2063 |
| Si 800 keV / 100 µm | 0 | 0 | 4 |
| W 800 keV / 10 µm | 0 | 0 | 0 |
| Si 100 keV / 1 µm | 0 | 0 | 220 |
| W 100 keV / 100 µm / 75° | 0 | 1 | 19712 |

This explains why the original two caps gave almost identical scores. The
[energy control](energy-controlled-propagation.md) only refines a flight when
its predicted mean energy loss reaches the cap. A later convergence design
needs demonstrated active controls and several refinement levels; W 800 keV
still needs a tighter activity probe. Agreement between two inactive settings
is not evidence of convergence.

## Comparator configuration audit

The reused executable's SHA-256 is
`fde4584d62bcc7839dda5a15a6dc52c1fdb925f0d2beaa892f15d3ff91c4b544`.
The retained source snapshot's three patched files match the previous build's
`benchmark-patch.json` after hashes. This stage did not rebuild Geant4.
Fresh raw dumps report:

| Material | Gamma production threshold | Electron production threshold |
| --- | ---: | ---: |
| Silicon | 0.99 keV | 4.04011 MeV |
| Tungsten | 0.99 keV | 29.5285 MeV |

The macro requests a 1 nm gamma range cut and 10 mm electron range cut; the
material-dependent energy conversions above are the observed production cuts.
The separate lowest-electron-energy setting is 10 keV. Secondary tracking is
also suppressed by the retained benchmark stacking patch. These production
thresholds must not be presented as 10 keV electron tracking cuts or assumed to
be exactly the same controls as PyRITE's 1 keV hard-photon partition. Matching
accuracy still requires demonstrated cutoff/step/model checks in each engine.

## Evidence and reproduction

- [Complete small-pilot results and provenance](../check-records/equal-precision/cheap-validation-summary.json)
- [Separate step-activity counts](../check-records/equal-precision/step-activity-summary.json)

Raw scores, macros, logs, table/source fingerprints, submitted scripts, hardware
records and Geant4 source snapshot remain in the ignored
`scratch/cheap-validation-results/evidence.tar.gz` archive (SHA-256
`77c8527d73a1f50939145570ce2d6c73e77e1dd4c17f7dc90219239163004f1e`).
The separate activity archive is `activity-evidence.tar.gz` (SHA-256
`605b62ac1613cd0c53909034834cfc9e5b3ddc541d2aa578fe2eaa06b04e3bc4`).
Both downloads were verified before extraction; the staged lab checkout retains
both archives. Reproduce through maintained `pyrite remote sync`, then bounded
SLURM jobs from that checkout:

```bash
cpu=$(sbatch --parsable --partition=cpu --mem=8G --time=00:05:00 \
  checks/equal_precision/cheap-validation.sbatch cpu)
sbatch --dependency=afterok:$cpu --partition=gpu --nodelist=DESKTOP-QNIHO3D \
  --gres=gpu:1 --mem=24G --time=00:10:00 \
  checks/equal_precision/cheap-validation.sbatch gpu

uv run python checks/equal_precision/analyze_cheap_validation.py \
  scratch/cheap-validation/1410 scratch/cheap-validation/1411 \
  --protocol scratch/cheap-validation/1410/source/cheap-validation.json \
  --output /tmp/cheap-validation-summary.json

sbatch checks/equal_precision/step-activity.sbatch
```

Adjust pinned lab checkout/executable paths for a different machine. Use the
captured protocol bytes for historical analysis. The standalone activity probe
is also reproducible one scene/cap at a time with `probe_step_activity.py`.

Remaining cheap work is to find active step controls for W 800 keV and freeze
an informative refinement ladder. Electron-tracking cutoff, table density,
quadrature/detector and comparator-model convergence remain separate gaps.
Production stays unchanged; the large precision studies remain deferred.
