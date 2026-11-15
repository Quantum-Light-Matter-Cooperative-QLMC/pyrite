# Binary-target secondary fluorescence against Geant4

Claim: `photon-alloy-secondary-fluorescence`. Status: **anchored**
(2026-11-15), against the Geant4 Livermore reference after fresh-context
adjudication.

PyRITE's photon-only transport passes all 8 gated windows against Geant4
`emlivermore`. That includes Ga Kα at 15 and 30 keV, where about a third
of Ga K emission is As→Ga secondary fluorescence. The pre-registered Geant4
`empenelope` reference failed: it passes only 2/7, because its Ga K yield in
GaAs is about half of an independent analytic value. Penelope agrees with
Livermore, PyRITE and the analytic value for pure Ga, so the failure is
specific to Ga in the compound. The adjudication below accepted Livermore as
the reference on that basis, with the post-hoc switch disclosed.

## Reference and scope

`checks/photon_full_track/alloy_fluorescence.py` sends 11.5, 15 and 30 keV
photons normally into 0.2 mm GaAs (Geant4 NIST `G4_GALLIUM_ARSENIDE`,
5.31 g/cm³, Ga:As = 1:1, confirmed in the Geant4 log). It uses the
thick-target check's 5 cm footprint and its two 10 cm square apertures, 10 cm
in front of the slab (R) and behind it (T). As Kα (10.51–10.54 keV) and Kβ
(11.73 keV) lie above the Ga K edge (10.367 keV), so As fluorescence
re-excites Ga. At 11.5 keV, below the As K edge (11.867 keV), Ga K emission is
primary only.

Both engines kill photon-born charged particles at creation. Geant4 uses
complete vacancy cascades (Auger on, deexcitation cuts ignored), the settings
that resolved the W benchmark mismatch in
[the thick-target comparison](photon-full-track-thick-brem.md). Each engine
ran 10⁶ histories per energy in ten independent shards of 100 equal history
blocks. The check README owns the settings, commands and known model
differences.

## Pre-registered gates

The gates were fixed in code before any PyRITE spectrum was inspected. Raw
counts per incident photon in four single 0.5 keV windows (Ga Kα, Ga Kβ,
As Kα, As Kβ) are gated wherever the reference holds at least 1000 counts.
A window passes when `|P − G| ≤ max(0.05 G, 3σ)`, where σ combines both
engines' history-block standard errors. As windows are not gated at 11.5 keV.

## Results

PyRITE / Geant4 for every window with at least 1000 reference counts in
either arm (R unless marked T):

| Energy | Window | vs Penelope (gated reference) | vs Livermore (diagnostic) |
|---|---|---|---|
| 11.5 keV | Ga Kα | **1.928** (46σ, fail) | 0.992 (pass) |
| 11.5 keV | Ga Kβ | **1.846** (16σ, fail) | 0.963 (pass) |
| 15 keV | Ga Kα | **1.157** (9.8σ, fail) | 0.995 (pass) |
| 15 keV | Ga Kβ + As Kα2 | 1.030 (pass) | 0.978 (pass) |
| 15 keV | As Kα1 | 1.013 (pass) | 1.009 (pass) |
| 30 keV | Ga Kα | **1.187** (6.9σ, fail) | 0.995 (pass) |
| 30 keV | Ga Kβ + As Kα2 | 1.005 (ungated, 981 counts) | 0.924 (pass, −1.8σ) |
| 30 keV | Ga Kα T | **1.261** (6.0σ, fail) | 1.067 (pass) |

At 15 and 30 keV, PyRITE attributes 0.32 and 0.36 of its Ga photoabsorptions
above the K edge to later flights, mostly As K fluorescence; at 11.5 keV the
share is 0.011. A model without As→Ga secondary fluorescence would therefore
fall about a third short on the 15 and 30 keV Ga Kα gates, far outside the
5% band.

## Diagnosis of the Penelope disagreement

- **Analytic check.** A thick-target primary estimate at 11.5 keV for R uses
  Ω/4π = 0.0641 for the aperture, a Ga K-absorption share of about 0.71 from
  EPDL Ga/As photoabsorption, ω_K(Ga) = 0.507, a Kα fraction of 0.88 and
  escape ≈ 0.68. It gives about 0.0137 Ga Kα per photon. PyRITE gives 0.01381,
  Livermore 0.01392 and Penelope 0.00717.
- **Cross sections are not the cause.** Geant4's own Penelope Ga table
  (`pdgph31.p08`) gives a K-shell share of 0.87 and τ ≈ 174 cm²/g at
  11.36 keV, consistent with EPDL. The two Geant4 arms agree on As K within
  1.7% and on scattered 11.5 keV photons within 6%.
- **The deficit is compound-specific.** The Penelope/Livermore Ga K ratio in
  GaAs is about 0.51 at 11.5 keV and 0.84–0.86 at 15 and 30 keV. A
  follow-up scan (`alloy_edge_scan.py`, job 12873) sends the same beams into
  0.2 mm of pure Ga, Ge and As at 18 energies across each K edge. There
  Penelope and Livermore K-alpha yields agree within 0.981–1.009 (SE ≈ 3%),
  including Ga at 10.5–12.5 keV. In pure Ga at 11.5 keV, PyRITE/Livermore
  is 0.996. All three engines therefore agree for the element alone. The
  Penelope deficit appears only once Ga and As share one material, which
  points to compound handling in Geant4's Penelope photoelectric or
  deexcitation path. That code path was not traced. Both Geant4 arms used
  the same seeds, so in pure elements they replay nearly identical histories
  and the ratios scatter less than their nominal SE.

## Reproducibility and adjudication

`checks/photon_full_track/alloy-results/` holds merged history blocks for all
nine engine/energy records, both reports and `provenance.json` (jobs 12771
and 12843, tested commits, staging roots, source/data/executable hashes and
resources). Regression tests recompute both reports from the committed blocks
and require exactly the five Penelope failures and zero Livermore failures.

Before any promotion, a fresh-context adjudication must decide whether the
Livermore arm is an acceptable reference for this claim. It must also
rederive the block estimators and all gate decisions. The pure-element scan
is committed as `alloy-results/edge-scan.json` with its PyRITE arm. A third
engine (for example standalone PENELOPE or PenRed, #648) on the GaAs case
would strengthen that decision further.

## Fresh-context adjudication (2026-11-15)

Verdict: `rederived`. The adjudicator wrote an independent reducer, which
reproduced every mean, SE, gate limit, gated flag and pass flag of both
reports to 4e-16 relative. All nine record hashes match `provenance.json`.

| Ga Kα per photon on R, 11.5 keV | Value |
|---|---|
| Analytic, xraylib 4.3.0 (Scofield/Kissel), ω_K = 0.5166 | 0.01409 |
| Analytic, Krause 1979, ω_K = 0.507 | 0.01383 |
| PyRITE | 0.01381 |
| Geant4 Livermore | 0.01392 |
| Geant4 Penelope | 0.00717 |

The analytic estimate integrates the thick-target primary formula over the
real aperture (solid-angle fraction 0.0642). It is independent of the
EPDL/EADL data PyRITE and Livermore share. Scattered-photon excitation adds
about 1%. For pure Ga the analytic yield is about 0.0199, and all three
engines give 0.0194–0.0195.

The primary-only yield accounts for 0.72 and 0.67 of the observed Ga Kα on R
at 15 and 30 keV, and about 0.65 on T at 30 keV. Without As→Ga secondary
fluorescence, all three Ga Kα gates would fail by many sigma.

Harness review found no defect that invalidates either arm. Seeds and
history blocks are independent, and the frame mapping and shard merge are
correct. Window labels were corrected. PyRITE's As Kα2 (10463.3 eV) lies in
the window keyed `Ga_Kb`, and As Kα1 (10500.7 eV) sits 0.7 eV above the
10.5 keV edge. Both Geant4 arms place these lines the same way, so every
decision is like-for-like. Penelope's 15 keV "Ga Kβ" pass, however, says
little about Ga Kβ itself. A regression now pins these placements against
the packaged EADL energies.

Remaining limitations:

- The anchor replays committed blocks; no test reruns PyRITE on GaAs.
- The two reports reuse the same PyRITE records.
- The Livermore choice was made after inspection; a third engine (PenRed,
  #648) would remove that element.

