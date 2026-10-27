# Thick-target photon transport against Geant4

Claim: `photon-full-track-thick-brem`. Status: **discrepancy**.

The historical eight-case matrix retains that status. The matched-cascade
W 300 keV rerun below clears its two prior failures; the other seven cases
have not yet been rerun with complete Geant4 vacancy cascades.

This external comparison measures the complete primary-electron source plus
photon-only transport chain. It does not independently rederive its constituent
samplers or validate charged feedback, alloy fluorescence, or 511 keV escape.

## Reference and scope

The pinned check in `checks/photon_full_track/` compares
Geant4 11.4.2 TestEm5, `empenelope`, with the experimental PyRITE CPU weighted
BremsLib source and ordinary aperture hits. Eight stopping targets cover Si
and W at 0.3, 1, 3 and 10 MeV. Two ideal square apertures collect transmitted
and reflected photons. Both engines suppress charged secondary transport,
delta rays and straggling. Photon interactions and fluorescence remain active.
The check README owns geometry, transport settings, original gates, their
post-inspection revision, and known source/atomic-data differences.

Geant4 job 2022 supplies the independent reference; PyRITE mixture-proposal
job 2324 supplies 20,000 primaries per case. The tested pre-rebase commit is
`c2cb361a`. The check's `provenance.json` records source hashes verified
against the remote checkout and SHA-256 hashes of all original reductions
and committed compact projections. These projections retain independent
history blocks, including covariance between spectral windows.

## Result collected 2026-10-24

**98 of 100 revised gates pass.** Both failures occur in reflected W at
300 keV, in the 1–10 keV band:

| Observable | PyRITE | Geant4 | Combined standard error | Allowed absolute difference |
|---|---|---|---|---|
| Redistributed photons per primary | 0.00012457 | 0.00010420 | 0.00000369 | 0.00001106 |
| Redistributed fraction | 0.34205 | 0.30737 | 0.00905 | 0.02715 |

The yield is 19.6% high; the fraction differs by 0.03468. Neither satisfies
the existing tolerance or three-standard-error allowance. Their cause is
unresolved. Passing W K-line gates do not remove this low-energy discrepancy.
Totals and direct counts remain report-only under the explicitly documented
post-inspection gate revision; this is not evidence that all spectra agree.

The defensive source proposal resolves the earlier uniform sampler's poor
precision: in gated redistributed-yield bands, the PyRITE-to-Geant4 standard
error ratio is 0.36–1.42. The earlier all-pass uniform result was dominated
by large errors and supports no promotion. Neither run anchors this claim.

## Reduction audit and regression

Fraction uncertainties use paired block fluctuations of numerator minus
fraction times denominator. Line uncertainties use the peak-minus-sidebands
count within each block before estimating its standard error. This retains
correlations from common electron histories. The earlier reducer ignored
denominator fluctuations and line-window covariance; repairing these errors
leaves the same two failures. Tolerances have not been changed.

`tests/montecarlo/test_photon_full_track.py` tests exact cancellation for
correlated synthetic blocks, verifies compact projections, and pins all
eight cases with exactly the two unresolved failures. This is a regression
of the discrepancy, not a successful physics anchor. Running the standalone
comparison returns a failing exit code until the discrepancy is resolved.

The report-only creation-angle diagnostic (job 2293) also finds more forward
PyRITE photons in several large photon-energy-fraction bins. It uses selected
photons rather than clustered histories for its printed errors and does not
establish the cause of the W redistribution excess. A matched-source,
process-resolved comparison is needed before attributing that excess to
source angular shape, interaction sampling, or atomic relaxation data.

## Matched incident photons: vacancy-cascade settings (2026-10-27)

Job **2544** completed all twelve CPU tasks with exit code zero, using clean
checkpoint `64ab0b5f`. Each arm transported 100,000 normally incident photons
at 12, 20 or 30 keV through the same 0.1 mm W slab. This removes electron
transport and bremsstrahlung creation from the comparison. Geometry, source
coordinates, settings and reproduction commands are in the check README's
matched incident-photon section.

The original Geant4 reference disables Auger generation. In
[`G4UAtomicDeexcitation::GenerateAuger`](https://github.com/Geant4/geant4/blob/v11.4.2/source/processes/electromagnetic/lowenergy/src/G4UAtomicDeexcitation.cc),
that setting returns before daughter vacancies are enqueued. Enabling Auger
alone retains a second suppression: the original 1 km electron production
cut rejects Auger electrons before their daughter vacancies are enqueued.
The `cascade` arm enables Auger and ignores deexcitation production cuts;
the stacking action still kills every charged secondary before transport.
PyRITE already follows these nonradiative vacancy branches while discarding
their emitted electrons in photon-only mode. Thus the original comparison
does not match the two engines' atomic-cascade settings.

Reflected 1–10 keV photons per incident photon, multiplied by 1,000;
uncertainties are standard errors from 100 equal independent-history blocks:

| Incident energy | Geant4 Auger off | Geant4 Auger on, original cuts | Geant4 complete cascade | PyRITE |
|---|---|---|---|---|
| 12 keV | 7.72 ± 0.26 | 7.67 ± 0.26 | 8.07 ± 0.25 | 7.88 ± 0.28 |
| 20 keV | 3.71 ± 0.19 | 3.38 ± 0.17 | 4.58 ± 0.22 | 4.23 ± 0.20 |
| 30 keV | 1.35 ± 0.11 | 1.32 ± 0.12 | 1.93 ± 0.14 | 1.90 ± 0.13 |

All Geant4 reflected photons in this band have creator `phot`. Complete
cascades raise this yield by 23% at 20 keV and 43% at 30 keV relative to the
Auger-off runs. PyRITE agrees with the complete-cascade arm within 1.2
combined standard errors at all three incident energies. These observations
support the cascade-settings mismatch as a contributor to the earlier
fluorescence excess; they do not establish its contribution after integration
over the thick-target bremsstrahlung source.

`checks/photon_full_track/monoenergetic-results/` retains all twelve compressed
spectral block reductions and their recomputable summaries. `provenance.json`
pins source, executable, data, log and reduction hashes, seed and allocation.
Raw Geant4 logs remain in the isolated staging checkout. A post-run
completed-history/energy check accepts all nine Geant4 logs and reproduces
their block counts exactly. Regression tests preserve those counts' integrity,
their total/direct/redistributed partition and the reported uncertainties.

The full-track claim remains **discrepancy**, with the original two failures
and tolerances preserved. The next discriminating full-track comparison uses
complete Geant4 vacancy cascades, still killing charged secondaries. Alloy
fluorescence, 511 keV escape and charged-feedback anchoring remain outside
this diagnostic's scope.

## Matched-cascade W full-track rerun (2026-10-27)

Geant4 job **2557** transported ten independent shards of one million
300 keV primary electrons each through the original 0.1 mm W slab.
The pinned macro generator reproduces all 59 historical default macro
contents exactly; this explicitly selected arm changes only Auger generation,
deexcitation cut handling and the output name. Geometry, source, seeds,
electron transport and charged-secondary suppression are preserved. The
PyRITE result is the unchanged 20,000-primary job 2324 record.

All ten reference tasks completed with exit code zero. Reduction job **2567**
verified completed primary counts/energies and both cascade flags in every
log, then applied the original comparison gates. **All 11 W-case gates
pass, with no missing gates.** Both previous reflected 1–10 keV failures clear:

| Observable | PyRITE | Complete-cascade Geant4 | Combined standard error | Allowed absolute difference |
|---|---|---|---|---|
| Redistributed photons per primary | 0.0001245723 | 0.0001256000 | 0.0000042376 | 0.0000127129 |
| Redistributed fraction | 0.3420517 | 0.3488889 | 0.0087088 | 0.0261263 |

The redistributed yield ratio is 0.9918, compared with 1.1955 against the
historical Auger-off reference. No tolerance, PyRITE model or output changed.
The controlled settings change removes the observed W discrepancy, supporting
the atomic-cascade mismatch identified by the monoenergetic diagnostic.
This does not exclude smaller residual differences between the engines.

`checks/photon_full_track/cascade-results/` retains the compact reference
history blocks, the targeted report with both old and new comparisons, and
provenance. The clean tested checkpoint is `981c1141`; the isolated remote
checkout is `~/.local/opt/pyrite-341-cascade-981c1141`. Source, binary, macro,
raw-log, raw-reduction and committed-artifact hashes are retained. Original
references and the historical 98/100 result remain intact. Regression tests
recompute the targeted report, require all 11 gates and preserve the historical
two-failure regression separately.

This is a successful W-case comparison, not a uniformly corrected eight-case
matrix or an independent rederivation. The full ledger status stays
**discrepancy** pending the remaining seven matched-cascade references and
adjudication. Alloy fluorescence, 511 keV escape and charged feedback still
require their separate comparisons.

## Complete-cascade eight-case matrix (2026-10-27)

CPU array **2568** completed the remaining 49 million-primary shards:
Si at 0.3/1/3/10 MeV and W at 1/3/10 MeV. Dependent reduction job
**2577** completed with exit zero, verifying completed primary histories
and cascade settings in every log. The seven reports pass **89/89 gates**,
with zero lost historical gates. Together with the prior W 300 keV rerun,
the uniformly corrected matrix passes **100/100 gates**.

The source checkpoint, Geant4 executable, original seeds, electron settings,
unchanged PyRITE job 2324 results and comparison tolerances are preserved.
Only the previously documented vacancy-cascade settings differ from the
historical reference. The historical 98/100 result remains reproducible.

`checks/photon_full_track/cascade-matrix-results/` retains compact
independent-history blocks, historical and matched reports, source/macro/log/
reduction hashes, scheduler evidence and all 49 shard resource records.
Raw logs and fine-bin reductions remain in the isolated staging checkout
under `~/.local/opt/pyrite-341-cascade-981c1141/matrix`.
The regression recomputes all seven reports, requires all 100 gates including
the prior W case, and checks unchanged PyRITE input hashes.

The claim remains **discrepancy** pending fresh-context independent
adjudication of this corrected matrix. This evidence covers the bounded
photon-only thick-target comparison; alloy fluorescence, 511 keV escape
and charged-feedback validation remain separate.
