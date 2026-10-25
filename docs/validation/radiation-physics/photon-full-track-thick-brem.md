# Thick-target photon transport against Geant4

Claim: `photon-full-track-thick-brem`. Status: **discrepancy**.

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
