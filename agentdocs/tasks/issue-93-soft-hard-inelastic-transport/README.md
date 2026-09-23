# issue-93-soft-hard-inelastic-transport

Issue: #93. Branch: `issue-93-soft-hard-inelastic-transport`.

Decision from issue owner: match the corrected `stp.dat` stopping mean and
validate microscopic rates separately. The host-side candidate follows this;
production transport remains unchanged.

## Source audit, 2026-09-22

The issue describes `asymptotic.dat` (`CS0A`, `CS1A`, `CS2A`) as sharing the
corrected stopping model with `stp.dat`. This is false in the vendored SBETHE
source: the block writing `asymptotic.dat` is explicitly labelled **"Bethe
asymptotic formulas ... of neutral DHFS atoms (uncorrected)"** in
`src/pyrite/data/xsgen/sbethe/sbethe.f` near line 277. It calls `ASACSS`,
which reads free-atom asymptotic parameters from `atparams.tab`. The production
`stp.dat` is separately shell- and density-corrected.

For shipped tables, `CS1A / stopping_cs_eV_cm2` at 1, 10, 100 keV is:

| material | 1 keV | 10 keV | 100 keV |
| --- | ---: | ---: | ---: |
| silicon | 1.175 | 1.080 | 1.043 |
| MoS2 | 1.056 | 1.111 | 1.036 |
| SiO2 | 1.191 | 1.061 | 1.021 |

Measured with the project test runner against `resolve_catalog_table(...)`.
Thus a direct soft/hard split of `asymptotic.dat` cannot preserve the production
mean stopping. The asymptotic moments remain useful independent comparisons;
they are not an exact differential distribution and do not determine one.

The [Geant4 Penelope ionisation model reference](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/electron_incident/ionisation/penelope_ionisation.html)
gives the oscillator distant/close cross sections, the Møller energy-transfer
term, and the close-collision recoil angle (equations 127–133). It is a
candidate source for a transfer-spectrum implementation using SBETHE's OOS
distribution. Its shell-resolved binding energies are absent from `OOS.dat`;
the optical density alone cannot label an inner-shell vacancy or distinguish
ionization from excitation. The first model must state that limit and must not
create a vacancy from the optical table.

An initial OOS-bin GOS implementation on this branch gives raw total inelastic
cross sections of 0.39 (Si), 0.42 (MoS2), and 0.63 (SiO2) times SBETHE's
free-atom asymptotic `CS0A` at 100 keV. Its raw second moments are 0.95, 0.92,
and 0.96 times `CS2A`, respectively. At 1 keV the second-moment ratios are
only 0.23–0.34. The first moment can be normalized to corrected stopping, but
that does **not** validate the inelastic mean free path or transfer spectrum.
Independent IMFP/spectral benchmarks and the limits of the OOS-bin
approximation remain a gate before transport activation.

NIST SRD 71 TPP-2M supplies an independent 50–2000 eV predictive IMFP
comparison. At 1/2 keV, the calibrated model is within 5% for Si and about
22–23% low for SiO2 (25% bounded test; NIST quotes 20.5% absolute standard
uncertainty). This does not cover higher energies or the transfer spectrum.

## Implementation order

1. Build and independently check a GOS/Møller transfer spectrum from the OOS
   distribution. Document quadrature, density effect, transfer bounds, units,
   and how its first moment relates to production `stp.dat`. Reject energies
   or materials where a positive, finite spectrum cannot be built.
2. Partition that **one** spectrum at configurable `Wc`. Use its soft first
   moment for continuous loss and its hard zeroth moment for discrete event
   rate; do not also apply unrestricted continuous stopping. Expose hard
   transfer sampling and Møller recoil, with a secondary-state output above a
   separate production threshold.
3. Integrate the event into the CPU transport flight scheduler and event
   contract, then CUDA paths and run identity. Keep legacy mode unchanged until
   the new mode passes seeded CPU/GPU checks.
4. Validate stopping moments, IMFP, spectra, recoil, threshold convergence,
   and coherent-radiation segment behavior. Add derivation/ledger records and
   obtain fresh-context verification. Only a human can sign off.

No production inelastic mode is enabled by this audit.

## Follow-up, 2026-09-23

The host partition now clips a linear close bin at `Wc` after constructing the
raw spectrum on a fixed grid. Previously inserting `Wc` into the grid changed
the interpolated spectrum, raw moments, calibration, and total rate as the
threshold moved. Focused tests now hold those quantities fixed across cutoff
values and check samples from a clipped bin. Transport activation remains
gated by independent transfer-spectrum validation and event integration.

## Follow-up, 2026-09-23: loss-shape benchmark

The host partition now exposes an exact CDF for its hard-event transfer
spectrum. Seeded event samples match that CDF. Si's 1 eV-binned modal loss is
15 eV at 1, 5, and 50 keV, within the broad 17 ± 10 eV plasmon region in the
independent KESS Penn/Bethe–Fano comparison. Werner's REELS paper supplies a
measured Si bulk-loss distribution in a figure. Its 1 and 3 keV peak reads
roughly 0.10 eV⁻¹; a conservative lower bound is 0.08 eV⁻¹. The model peaks
at 0.046 and 0.045 eV⁻¹ in 1 eV bins, remaining below 0.047 eV⁻¹ with 0.5
or 2 eV bins. Renormalizing over the plotted 0–50 eV window gives only 0.055
and 0.056 eV⁻¹. A strict expected-failure test tracks this shape deficit.
This fails the differential-spectrum gate. Do not activate the model in CPU or CUDA
transport until a physically justified material-response correction and
seeded CPU/GPU checks exist.

## Follow-up, 2026-09-23: plasmon-area gate

Figure 3(b)'s retrieved bulk curve places about 0.46 collision probability in
14–20 eV by sparse figure reading; a conservative 0.35 bound accounts for
plot and retrieval uncertainty. The candidate GOS spectrum puts 0.241 and
0.238 there at 1 and 3 keV, or 0.292 and 0.294 when normalized to the
plotted 0–50 eV range. A strict expected-failure test now tracks this area
deficit as well as the peak-height deficit. The broad atomic OOS resonance
shape controls the window; the omitted transverse term is negligible at these
energies. SBETHE's `DENSIT` computes a stopping correction and does not export
a finite-momentum material response. Next physics implementation needs a
sourced Si response with independent shape validation before CPU/CUDA event
integration. The host candidate remains inactive.

## Follow-up, 2026-09-23: dielectric bulk candidate

An independent [Si oscillator dataset](https://zenodo.org/records/6024064)
from Yubero et al. (1993) supplies a fitted valence ELF with momentum
dispersion. A host-only implementation of the published bulk dielectric
integral now passes the conservative Si REELS peak and 14–20 eV probability
bounds at 1 and 3 keV. This identifies a plausible material-response route.
The oscillator fit itself comes from REELS, so the Werner comparison does not
fully validate it independently. The supplied three oscillators cover valence
losses only; a source-supported core tail, corrected-stopping closure, total
rate and angular-transport checks remain before this can replace the GOS
spectrum or enter CPU/CUDA transport. See
`docs/validation/beam-transport/dielectric-bulk-loss.md`.

## Follow-up, 2026-09-23: conditional dielectric recoil

The Si valence candidate now samples recoil momentum from the same finite-q
ELF used by its differential loss rate and returns the primary polar angle.
Direct log-k quadrature checks the sampled recoil quantiles; a fixed-seed
check covers the stochastic path. It remains host-only. The core tail,
corrected-stopping closure, secondary state, azimuth, and independent
absolute-rate validation still gate transport integration.

## Follow-up, 2026-09-23: native OOS core edge

The host GOS builder can now isolate raw core oscillators at a positive
duplicated OOS shell edge, without applying whole-material stopping
calibration. In shipped Si data the first edge is 102.2154 eV and divides
3.769 valence from 10.237 core electron strengths. Fixed-grid core moments
are cutoff-independent, and sampled core transfers remain above the edge.
Vos and Grande (2019) caution that atomic GOS may miss the shallow-core shape;
this is an input diagnostic rather than a validated combined spectrum. See
`docs/validation/beam-transport/gos-core-edge.md`.

## Follow-up, 2026-09-23: fixed-grid valence partition

The host Si dielectric candidate now partitions one caller-supplied valence
loss grid at `Wc` and samples hard losses from its exact piecewise-linear CDF.
Its total valence rate and first moment stay fixed across cutoffs, including
an interior-bin cutoff, and the hard rate vanishes at the grid endpoint.
This is a partial valence response with a caller-selected high-loss limit; it
does not yet close the SBETHE corrected stopping mean or total IMFP. Core-tail
composition, independent absolute-rate evidence, and grid convergence still
gate CPU/CUDA transport activation. See
`docs/validation/beam-transport/dielectric-bulk-loss.md`.

## Follow-up, 2026-09-23: valence loss-grid convergence

Adaptive integration checks the 0–100 eV Si valence zeroth and first moments
at 1 and 3 keV against 0.5, 0.25, and 0.125 eV linear grids. Both errors fall
monotonically; the largest finest-grid relative error is 4.7e-6. This closes
the numerical loss-grid check for that fixed partial interval. High-loss
endpoint selection, core-tail composition, corrected-stopping closure, and
independent absolute-rate validation still gate transport activation.

## Follow-up, 2026-09-23: high-loss endpoint and raw core diagnostic

At 1 and 3 keV, extending the fixed-grid Si dielectric valence interval from
the first native OOS core edge (102.2154 eV) to 200 eV changes its first
moment by 11.2% and 11.7%, while its rate changes by 1.7%. Adding the raw
OOS core first moment to the edge-truncated valence moment gives 1.790031 and
0.851456 eV/Å, versus corrected SBETHE targets of 1.692945 and 0.891196
eV/Å. The raw sums miss in opposite directions. This is a diagnostic only:
the core edge is not a proven valence endpoint, and scaling to force stopping
would not validate the spectrum or rate. See
`docs/validation/beam-transport/dielectric-bulk-loss.md`.

## Follow-up, 2026-09-23: independent optical core-edge check

Yang et al. (2019) report an independently retrieved Si optical ELF of
3.453856004 at 16 eV and 0.047536284 at 110 eV. The Yubero three-oscillator
valence fit gives 3.211891 and 0.000777612 at those points. Its valence
plasmon agrees within 7%, but its analytic tail is about 61 times low at the
L edge. A passing valence-point test and strict expected-failure core-point
test now record this measured gate. The comparison does not provide a
finite-momentum core model or a composition rule; the host candidate remains
inactive. See `docs/validation/beam-transport/dielectric-bulk-loss.md`.

## Follow-up, 2026-09-23: shell-rate source and scope

PENELOPE-2024 section 3.2.6 uses precomputed Bote–Salvat K/L/M/N
electron-impact ionization cross sections. Its `material.f` packages those
tables; it is not the cross-section solver. SBETHE supplies stopping and OOS
data but no shell-resolved impact-ionization rates. EADL supplies atomic
relaxation data, while the existing packaged EEDL MF=23/MT=534–572 tape
supplies shell-resolved electron-impact rates and binding energies. Use EEDL
as issue #93's initial shell-rate baseline; evaluate Bote–Salvat as the
separate optional/reference backend in #92, with the rate comparison in #86.

The packaged EEDL parser now exposes shell rates to host code without an
xraydb relaxation join. Its per-shell energy grids and binding energies are
retained. These totals do not specify the differential energy-transfer law or
secondary kinematics; those still need a consistent construction and
validation before CPU/CUDA transport integration. The Si dielectric branch
is a material-specific optional study, not a gate for the material-general
EEDL baseline.
