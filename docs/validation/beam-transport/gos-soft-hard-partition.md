# GOS soft/hard transfer partition

Validation: `gos-soft-hard-partition`. Independent verdict: rederived;
human sign-off remains pending.

## Source and intended quantity

The [Geant4 PENELOPE ionisation reference](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/electron_incident/ionisation/penelope_ionisation.html)
defines soft and hard partial cross sections as integrals of one transfer
distribution across a production cutoff. SBETHE's production `stp.dat`
provides corrected mean collision stopping. Its `asymptotic.dat` is explicitly
**uncorrected free-atom asymptotic output** in the vendored source, and cannot
be substituted for the production mean.

The raw optical GOS spectrum $g_E(W)=d\sigma_{\mathrm{GOS}}/dW$ has moment
$M_1=\int_0^E Wg_E(W)\,dW$. At each incident energy, one positive scale
$a=S_{\mathrm{SBETHE}}/M_1$ gives

$$
S_{\mathrm{soft}}=a\int_0^{W_c}Wg_E(W)\,dW,\qquad
S_{\mathrm{hard}}=a\int_{W_c}^{E}Wg_E(W)\,dW,\qquad
\sigma_{\mathrm{hard}}=a\int_{W_c}^{E}g_E(W)\,dW.
$$

Here $S$ has units eV cm$^2$ per molecule and $\sigma$ has cm$^2$ per
molecule. The implementation builds a piecewise-linear close spectrum on a
$W_c$-independent grid, clips the intersected bin at $W_c$ by linear
interpolation, and classifies each distant resonance on one side. Soft and
hard first moments therefore sum to $S_{\mathrm{SBETHE}}$ under the same
quadrature.
For $W_c\to0$, the soft first moment vanishes; for $W_c\geq E$, the hard
rate vanishes. A transport core must apply only $S_{\mathrm{soft}}$ along
flights and sample $\sigma_{\mathrm{hard}}$ at event points.

The source spectrum is a sum of delta resonances at $W_i$ and a continuous
close term on $[W_i,E/2]$. For any nonnegative raw spectrum with positive
first moment, $a=S_{\mathrm{SBETHE}}/M_1>0$, and additivity over the two
cutoff intervals gives

$$
S_{\mathrm{soft}}+S_{\mathrm{hard}}
=a\left(\int_0^{W_c}+\int_{W_c}^{E}\right)Wg_E(W)\,dW
=S_{\mathrm{SBETHE}}.
$$

The corrected table gives a collision stopping *cross section* in
eV cm$^2$ per molecule; multiplication by molecular number density gives
eV per unit path length. The dimensionless factor $a$ changes both rate and
first moment. Closure holds only if the same cutoff convention assigns each
delta resonance to exactly one side and the same $a$ multiplies every term.

## Source-to-code comparison

`_linear_moments` integrates the linear close-bin density through its zeroth,
first and second moments. `build_gos_partition` fixes the close-bin grid before
reading $W_c$, clips the hard part of the intersected bin, partitions distant
delta resonances by $W_i>W_c$, and applies the same $a$ to every soft and
hard term. The raw moments, calibration, and total rate therefore stay fixed
as $W_c$ varies; the first moments close to the corrected table value.
An oscillator exactly at the cutoff belongs to the soft side. The
source-to-code comparison agrees for that partition convention.

The stated limiting case $W_c\geq E$ now holds even if a resonance sits
exactly at $W_i=E$: all accessible resonances have $W_i\leq E\leq W_c$,
and all close bins end at or below $E/2$. The synthetic endpoint anchor
`test_equal_to_incident_energy_is_soft_even_for_a_delta_resonance` puts one
resonance at $W_i=W_c=E$ and finds zero hard rate and hard first moment,
with the full corrected stopping cross section on the soft side.

The common scale guarantees mean-loss bookkeeping, not microscopic accuracy.
At 100 keV, the raw model's total cross section is 0.39, 0.42 and 0.63 of
SBETHE's asymptotic `CS0A` for Si, MoS2 and SiO2. The asymptotic source is
not itself an IMFP oracle. External IMFP and transfer-spectrum validation is
required before this model becomes a transport option.

As an independent low-energy total-rate anchor, the [NIST SRD 71 User Guide](https://www.nist.gov/system/files/documents/srd/SRD71UsersGuideV1-2.pdf)
Appendix A gives the TPP-2M predictive IMFP formula for 50–2000 eV. At
1 and 2 keV, the calibrated model gives 25.0 and 42.1 Å for Si versus
TPP-2M's 23.9 and 41.5 Å. For SiO2 it gives 23.6 and 40.0 Å versus 30.2 and
52.3 Å. NIST estimates 20.5% absolute standard uncertainty for TPP-2M;
the SiO2 discrepancy is about 22–23%, so this is a bounded comparison, not
precision validation. This source covers neither 100 keV nor the differential
transfer spectrum.

## Differential loss-shape checks

`hard_transfer_cdf` integrates the stored delta resonances and linear close
bins independently of the random sampler. With a fixed seed, 10,000 sampled
3 keV Si hard events reproduce six CDF points within 0.015 absolute
probability. This checks sampling of the proposed spectrum, not its physical
shape.

The [KESS silicon transport study](https://publikationen.bibliothek.kit.edu/1000024959/1919154)
compares independent Penn dielectric and Bethe–Fano loss distributions in
Fig. 4.5(a). It places their most probable Si loss in the 17 ± 10 eV plasmon
region across its plotted energies. For the present GOS model, 1 eV bins peak
at 15 eV at 1, 5, and 50 keV; 61–63% of the hard-event probability lies from
7 to 27 eV when $W_c=1$ eV. The comparison is a broad mode check. KESS uses
different material response and provides no machine-readable curve here.

[Werner's Si REELS study](https://arxiv.org/abs/cond-mat/0503470) retrieves a
normalized bulk differential inverse mean free path from measured 1 and 3 keV
reflection spectra in Fig. 3(b). Reading its open-circle peak from the plot
gives roughly 0.10 eV$^{-1}$ near 16–17 eV; even a conservative lower bound
is 0.08 eV$^{-1}$. The current model peaks at 0.046 and 0.045 eV$^{-1}$ in
1 eV bins at 1 and 3 keV. Using 0.5 or 2 eV bins leaves each model peak
below 0.047 eV$^{-1}$. Even renormalizing the model over the plotted 0–50 eV
window raises its peaks only to 0.055 and 0.056 eV$^{-1}$. The
`test_silicon_loss_peak_height_against_reels`
anchor is marked expected-failure to preserve this discrepancy.

The plot has no numerical table; the values above are figure readings, not
digitized point data. Werner identifies a roughly 12 eV shoulder and a
spurious second-plasmon feature near 32 eV in the retrieval, and describes
limitations of the surface-loss subtraction. Those caveats do not establish
agreement with the current model's much lower peak. A material-response
model and further quantitative loss-spectrum comparisons are needed before
production transport activation. The independent transfer-spectrum gate
remains open.

The discrepancy also covers the plasmon's *area*. Reading Fig. 3(b)'s open
circles against its 0.04 eV$^{-1}$ ordinate ticks gives roughly 0.06, 0.10,
0.08 and 0.04 eV$^{-1}$ at 14, 16, 18 and 20 eV. A trapezoid through those
points places about 0.46 of the normalized collision probability in 14–20 eV.
The deliberately lower bound 0.35 leaves room for figure-reading error,
sampling scatter and the retrieval artifacts Werner describes. The paper
retrieves one bulk curve from the 1 and 3 keV measurements; it does not give
separate measured curves for those energies.

| Model energy | $P(14\leq W<20\,\mathrm{eV})$ | Divided by $P(W<50\,\mathrm{eV})$ |
| --- | ---: | ---: |
| 1 keV | 0.241 | 0.292 |
| 3 keV | 0.238 | 0.294 |

`test_silicon_plasmon_window_probability_against_reels` records the
conservative 0.35 window bound as a strict expected failure. Window
renormalization favors the present model, yet leaves a gap of at least 0.056
probability. This integrated comparison does not depend on the width of a
single histogram bin. It is a second gate on the same experimental curve, not
an independent experiment.

At 1 and 3 keV, distant longitudinal resonances account for 85.8% and 86.9%
of the model's total hard-event rate; close collisions account for 14.2% and
13.1%. Distant transverse terms contribute below $10^{-5}$ of the rate in
both cases. Within 14–20 eV, the longitudinal terms supply about 0.225 and
0.224 of the 0.241 and 0.238 total probabilities. The broad atomic OOS
resonances therefore dominate the failing shape. SBETHE's dielectric-related
`DENSIT` routine corrects stopping using Fano's integral; its `OOS.dat` output
contains no tabulated finite-momentum Si loss response. Adding the omitted
transverse density correction alone cannot explain the observed plasmon-area
gap. A finite-momentum, material-specific response must be sourced and checked
before changing the transport spectrum.
