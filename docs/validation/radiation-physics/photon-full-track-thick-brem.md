# Thick-target photon transport against Geant4

Claim: `photon-full-track-thick-brem`. Status: **anchored** (2026-10-29).

The historical Auger-off reference passes 98/100 revised gates. With complete
Geant4 vacancy cascades, the corrected eight-case matrix passes **100/100**
under unchanged gates and PyRITE outputs. Fresh-context adjudication below
rederived the estimators and decisions; the corrected-matrix regression pins
all 100 gates. The historical two-failure regression is retained separately.

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

The fresh-context adjudication below rederives this corrected matrix; the
claim is now **anchored** by its regression. This evidence covers the bounded
photon-only thick-target comparison; alloy fluorescence, 511 keV escape
and charged-feedback validation remain separate.

## Independent estimator derivation (2026-10-27)

This fresh-context derivation was recorded before reading the scorer or
reducer bodies. The quantity is the expected number of photons crossing a
physical aperture in an energy window per incident primary electron, including
zero-hit histories. It is neither a density per keV nor per steradian.
For equal independent blocks of $n$ primaries, let $X_b$ be the block's
weighted count divided by $n$. With $B$ blocks,

$$
\widehat\mu_X=\frac{1}{B}\sum_b X_b,\qquad
\operatorname{Cov}(\widehat\mu_X,\widehat\mu_Y)
=\frac{\sum_b(X_b-\overline X)(Y_b-\overline Y)}{B(B-1)}.
$$

The yield standard error is the square root of this diagonal covariance.
Direct and redistributed counts must partition total counts within every
block. For redistributed fraction $f=\overline R/\overline T$, the delta-method
standard error retains the paired denominator fluctuation:

$$
\operatorname{SE}(f)^2
=\frac{\sum_b[(R_b-\overline R)-f(T_b-\overline T)]^2}
{B(B-1)\overline T^2}.
$$

It vanishes when every block has the same fraction; treating the denominator
as exact fails this limit. For a net line with equal-width peak and adjacent
side windows, form $L_b=P_b-(S_{b,-}+S_{b,+})/2$ first, then estimate its
mean and error from the same blocks. Linear continuum cancels; the covariance
between all three windows remains. Counts per primary are dimensionless;
fractions and all their errors are dimensionless. Energy-window boundaries
use keV consistently; aperture geometry carries no extra solid-angle factor.

Independent engines give combined error
$\sigma=\sqrt{\operatorname{SE}_P^2+\operatorname{SE}_G^2}$.
Redistributed-yield and net-line gates require
$|P-G|\le\max(0.10G,3\sigma)$; fraction gates require
$|f_P-f_G|\le\max(0.02,3\sigma)$, with the documented 1,000-reference-count
eligibility. Total/direct yields remain report-only under the disclosed
post-inspection revision. These gates define bounded agreement, not a
universal confidence statement or a comparison of all spectral shapes.

### Implementation comparison and independent adjudication

The source and reducer were inspected after the expressions above were
recorded. The actual line-side windows each have **half** the peak width,
rather than the equal widths used in the preliminary example above. Their
correct integrated continuum is therefore $S_{b,-}+S_{b,+}$ and the actual
net count is $L_b=P_b-S_{b,-}-S_{b,+}$. Integrating any linear continuum over
these three symmetric windows verifies this coefficient exactly. The
implementation uses this correct half-width convention; there is no
factor-of-two discrepancy. Net-line eligibility uses 1,000 **net** reference
counts, rather than 1,000 unsubtracted peak counts.

`_mean_se`, `_fraction` and `_line` agree with the independent block
expressions for the evidence's equal blocks. Each Geant4 record has 100
equal blocks, including zero-hit blocks; each PyRITE record has 400 blocks
of 50 original electron histories. Multiplying PyRITE's stored spectral
density by bin width and block population restores weighted counts before
reduction. Scoring divides by original primary population, not photon
proposals or accepted hits; no aperture-area or solid-angle division is
appropriate. Total, direct and redistributed arrays partition within every
block to floating-point precision. Direct means an uncollided source photon;
Geant4 fluorescence remains redistributed even without a subsequent collision.
Source attenuation is applied by tracking once. Energy-band integration uses
whole enclosed histogram bins, consistently on both engines; compact records
retain those projections rather than reconstructing missing fine-bin shape.

A separate NumPy calculation, importing no comparison/scoring helpers,
recomputed all reported means, block standard errors, correlated fractions,
net lines, ratios, combined errors, gate limits and decisions from the compact
blocks. Every numeric report field agrees within $3\times10^{-13}$ relative
precision (absolute floor $10^{-18}$). All historical gate keys remain present:

| Case | Complete-cascade primaries | Passing revised gates |
|---|---:|---:|
| Si 0.3 MeV | 30,000,000 | 7/7 |
| Si 1 MeV | 10,000,000 | 10/10 |
| Si 3 MeV | 3,000,000 | 14/14 |
| Si 10 MeV | 1,000,000 | 14/14 |
| W 0.3 MeV | 10,000,000 | 11/11 |
| W 1 MeV | 3,000,000 | 13/13 |
| W 3 MeV | 1,000,000 | 13/13 |
| W 10 MeV | 1,000,000 | 18/18 |

The independent calculation also reproduces the historical 98/100 result,
with exactly the two reflected W 300 keV failures. Original PyRITE inputs,
all 16 historical compact-reference/result digests, both targeted-W artifact
digests, all 83 corrected-matrix manifest entries and all 12 monoenergetic
artifact digests match. All seven matrix exit records are zero. Retained
history counts and recorded job completion evidence support all eight cases;
no low-count disappearance has removed an old gate.

Read-only inspection of the retained lab checkout independently checked all
59 compressed raw-log digests against the compact records and all completed
million-electron run summaries, including the expected incident energies and
both cascade flags. All 59 cascade macros restore the historical macro bytes
after reversing only Auger generation, deexcitation-ignore-cut and output-name
changes. Thus geometry, electron transport, primary seeds and the charged
secondary kill setting remain fixed. The retained TestEm5 stacking body keeps
primaries and photons and kills every charged secondary at creation; its
integer switch preserves mode 3. The executable digest matches
`fe3fc52d65477c780e9b77c4eb1385c752b66c403b045b988855a5d48541cc4f`.

The retained Geant4 atomic-deexcitation source matches
`90625567d9421cbc3dcdb3aaba0e3af2ad1200a94db4b3a4bdda0e5cec185d53`.
Both it and the pinned
[upstream 11.4.2 source](https://github.com/Geant4/geant4/blob/v11.4.2/source/processes/electromagnetic/lowenergy/src/G4UAtomicDeexcitation.cc)
return from `GenerateAuger` before enqueuing the two daughter vacancies when
Auger generation is disabled or its electron fails the production cut.
Generating the electron with deexcitation cuts ignored and then killing its
track preserves both vacancies. This is a physical settings correction,
not a tolerance adjustment. The monoenergetic arms and controlled full-track
rerun support this mechanism as the cause of the resolved low-energy W
failure, without proving exact equality of the engines' atomic data.

The tested corrected-reference checkpoint `981c1141` and its four source
digests remain available locally. The original pre-rebase `c2cb361a` object
is unavailable in the current Git database; its archived lab source files
still exist and match all three original sampler/scorer/benchmark digests.
Raw logs, fine-bin reductions, and executable remain available in the staging
locations recorded above. Their retention matters: compact blocks certify
these projected observables, not discarded spectral detail.

**Filters:** units pass; zero-history/partition/constant-fraction/linear-
continuum limits pass; sign and normalization conventions pass; settings,
reference integrity, evidence retention and gate coverage pass. The tolerance
basis remains the disclosed model allowance and post-inspection revision;
this is not a preregistered validation of total/direct spectra. Known
BremsLib/PenBrem angular and Compton-vacancy differences remain.

**Independent verdict: rederived for this bounded CPU measurement.** The
corrected matrix clears the historical discrepancy under the unchanged revised
gates. Existing regressions pin all corrected reports, historical failures,
covariance cancellations, input hashes, complete-run checks and missing-gate
refusal; the task owner advanced the claim to **anchored** on 2026-10-29 on this
adjudication. No constituent sampler, CUDA backend, alloy, annihilation line,
charged feedback or general shower validation is implied. Human sign-off
remains separate.

Verification: `UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cpu uv run
pyrite-dev test tests/montecarlo/test_photon_full_track.py tests/dev/test_docs.py`
passed **32 tests**. The documentation build with writable uv tool directories
passed, including the marimo demo export; rendered mathematics was inspected
in the generated validation page.
