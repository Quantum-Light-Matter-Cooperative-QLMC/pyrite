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

## Follow-up, 2026-09-24: material shell-rate preparation

The host can now combine packaged EEDL shell cross sections with a crystal or
medium's catalog element number densities. It retains each element and shell
designator, returns rates in inverse angstroms, and rejects extrapolation for
an accessible shell outside its EEDL projectile grid. The adapter lives beside
the existing EEDL parser in `montecarlo/spectrum/` to preserve the package
dependency graph. Its source, mixture rule, units, and limits are recorded at
`docs/validation/beam-transport/eedl-material-shell-rates.md`. Independent
shell-rate comparisons and fresh-context physics validation remain open.
Transport still has no EEDL hard-event mode; before integration, the shared
data access needs an acyclic owner usable by the flight scheduler.

## Follow-up, 2026-09-24: shared EEDL data owner

The checksum-pinned EEDL subshell parser now lives in
`montecarlo/eedl_ionization.py`. Characteristic X-ray scoring retains its
existing imports as compatibility aliases, while material shell-rate
preparation reads the shared owner directly. The data dependency no longer
requires transport to import spectrum. This refactor changes no transfer
physics and does not enable hard-event scheduling.

## Follow-up, 2026-09-24: PENELOPE-2024 source audit

The local PENELOPE-2024 manual, §§3.2.1–3.2.2, defines a shell oscillator by
population $f_k$, binding energy $U_k$, resonance energy $W_k$, and a
close-collision threshold $Q_k$. For a bound shell $Q_k=U_k$, **not** $W_k$
(Eq. 3.56). The populations sum to $Z$ (Eq. 3.60); their log-weighted
resonances reproduce material mean excitation energy $I$ (Eq. 3.61). Bound
$W_k$ additionally depends on the material plasma energy and a material
parameter (Eqs. 3.63–3.65). A conduction band has $U=0$ and a distinct
plasmon prescription (Eq. 3.62). Inner-shell distant losses have a broadened
distribution above $U_k$ (Eqs. 3.76–3.80). The density correction and close
Møller kinematics follow §§3.2.2–3.2.3; the bound-shell close limit uses
$(E+U_k)/2$ (Eq. 3.88). Hard-event energy accounting gives an inner-shell
secondary $E_s=W-U_k$ (Eq. 3.124 and following text).

PENELOPE replaces inner-shell GOS rates with Bote–Salvat tables, retains the
conditional GOS loss/recoil distributions, and rescales outer-shell rates to
keep stopping (Eqs. 3.141–3.142). Its reference shell configurations and
ionization energies come from `pdatconf.p14` (§7.1.1). That file is not in
this checkout. Packaged EEDL supplies $U_k$ and total rates, but not $f_k$,
$W_k$, or the conduction-band partition. SBETHE OOS supplies optical strength
without reliable vacancy labels. The current OOS-bin candidate uses its
resonance as the close threshold, so it is a simplified Geant4-style
approximation, not the PENELOPE-2024 bound-shell construction. Do not use it
to attach EEDL vacancies or enable transport. Next shell-model input needs a
documented, legally distributable source for shell populations, mean
excitation energy, and conduction-band/plasma parameters, plus an explicit
element/shell mapping to EEDL designators.

The already supported SBETHE v2 reference-data fetch includes `pdatconf.p14` in
the user data directory. The local fetched copy has SHA-256
`cd239554bb6e823692ea4611d443df8684b4cace06006fc271a4168cb78c62d2`.
The public SBETHE deposit lists CC BY-NC 3.0; it is not the separately
licensed NEA PENELOPE distribution, and byte identity with that distribution
has not been checked.
Its nine-column records contain $Z$, shell designator/label, occupation,
ionization energy, Compton profile, and widths. All 99 elements have occupation
sums equal to $Z$ in a read-only format audit. This supplies an accessible
atomic population source for host-side study without bundling its data; any
broader redistribution or use must respect the source terms. It does not supply the material
conduction-band partition or establish that its binding energies equal the
EEDL values. A typed loader and explicit EEDL shell mapping can use the
fetched source after recording those differences and a material input rule.
The existing `plasma_energy_eV` computes the all-electron material plasma
energy, and SBETHE's material input records the chosen mean excitation energy.
Neither determines which outer electrons belong to a conduction band.

## Follow-up, 2026-09-24: atomic shell input loader

The host now reads the fetched, checksum-pinned `pdatconf.p14` into typed
free-atom shells and checks that each element's occupations sum to $Z$. An
exact-designator EEDL diagnostic retains both binding energies and rejects
unmatched vacancy channels. The real Si records expose a gap: EEDL has
designator 7, absent from the SBETHE Si configuration. No one-to-one
vacancy mapping or material conduction-band rule is established, so the shell
GOS event construction remains gated. See
`docs/validation/beam-transport/sbethe-atomic-shell-inputs.md`.

## Follow-up, 2026-09-24: EEDL label join

The Si "designator 7" gap came from numbering, not physics. EEDL designators
are ENDF-6 MF=23 codes (`MT - 533`, with O8/O9 slots). `pdatconf.p14` uses
PENELOPE codes that diverge after O7, so the integer join would mislabel Cs
and heavier elements. The join now uses x-ray labels, sharing the ENDF label
map with characteristic scoring through `eedl_ionization.EEDL_SUBSHELL_LABELS`.
EEDL spin-orbit partners that SBETHE leaves empty (Si M3, Mo N5, …) join the
filled SBETHE $n,l$ shell. All 99 elements now join, with 41 merges, and
partner binding differences stay within 3.4 eV. Remaining shell-GOS input gate: a material
conduction-band / outer-shell rule, including resonance energies (PENELOPE
Eqs. 3.61–3.65) from $I$ and plasma energy.

## Follow-up, 2026-09-24: conduction-band rule audit

PENELOPE-2024 §3.2.1 (p. 117–118): $f_{cb}$ and $W_{cb}$ "should be
identified with" the effective plasmon electron count and plasmon energy,
estimated from EELS or optical data. When unavailable, it uses a fallback
default: $f_{cb}$ counts electrons with ionisation energy below "say, 15
eV", and $W_{cb}=\sqrt{f_{cb}/Z}\,\Omega_p$ (Eq. 3.62). Bound shells use
$W_k=\sqrt{(aU_k)^2+2f_k\Omega_p^2/3Z}$ (Eq. 3.63), with one $a$ solved from
$Z\ln I=f_{cb}\ln W_{cb}+\sum_k f_k\ln W_k$ (Eq. 3.64). Compounds follow
Bragg additivity (Eq. 3.65). The manual's own examples override the default
with measured plasmons (Al $W_{cb}=15$ eV) and show that $W_{cb}$ strongly
changes IMFP, while stopping stays insensitive. §7.1.1 says `material`
prompts the user for plasmon energy and strength. `material.f` is not
available locally, so its coded default is unchecked.

Vendored SBETHE instead replaces all atomic OOS below `WTH = 50` eV with
one damped oscillator (`sbethe.f` ~1575–1720). Its strength is the atomic
OOS strength below 50 eV, its initial resonance is the Eq. 3.62 form, and a
bisected width fits $I$ (lowering $W_R$ if needed). An optional band gap
zeroes it below $W_g$. This width is fit to $I$, not to loss data, which
likely explains the broad Si OOS plasmon measured earlier.

Our SBETHE catalog decks pass no band gap for any material checked
(silicon, mos2, sio2, ws2, hbn, diamond), so all are run as conductors.
Stopping is insensitive, but this matters for the OOS spectral shape.
Out of #93 scope; flag separately.

Eq. 3.62–3.64 with fetched `pdatconf` shells, catalog $I$ and all-electron
$\Omega_p$ (per formula unit):

| material | $U<15$ eV: $f_{cb}$, $W_{cb}$, $a$ | $U<50$ eV: $f_{cb}$, $W_{cb}$, $a$ |
| --- | --- | --- |
| silicon | 4, 16.60, 2.208 | 4, 16.60, 2.208 |
| mos2 | 14, 19.06, 1.802 | 24, 24.95, 1.978 |
| sio2 | 12, 19.10, 2.726 | 16, 22.06, 3.460 |
| ws2 | 14, 19.08, 2.014 | 36, 30.59, 2.363 |
| hbn | 6, 21.39, 2.521 | 8, 24.70, 2.955 |
| diamond | 2, 22.05, 1.988 | 4, 31.18, 1.693 |

Si is threshold-independent and matches the ~16.7 eV bulk plasmon. The 15 eV
rule drops 2s/3s valence electrons (C 2s 16.6 eV, O 2s 28.5 eV, S 3s
20.2 eV). It therefore gives low diamond/oxide/sulfide $W_{cb}$ against
commonly quoted bulk plasmons (diamond ~33, SiO₂ ~22, MoS₂ ~23, hBN
~26 eV; unsourced recollection, not yet checked). A rule choice is needed
before the shell-set slice.

## Follow-up, 2026-09-24: shell oscillators with measured conduction bands

Owner decision: use measured plasmon data per material, falling back to the
manual's 15 eV default. `montecarlo/transport/shell_oscillators.py` builds
PENELOPE Eqs. 3.62–3.64 oscillators from `pdatconf` shells, catalog $I$,
all-electron $\Omega_p$, and `data/conduction_band.toml`. Sourced entries:
Si 4 e / 16.7 eV (Yang 2019 ELF max), SiO₂ 16 e / 23.6 eV (Da 2013 ELF max;
Saito 2025 22 eV corroborates), MoS₂ 18 e / 23.0 eV (Moynihan 2020). Each
measured band consumes whole outermost shells with a clear gap. Each
measured value lies within 7% of Eq. 3.62, and $a$ = 2.20 / 3.20 / 1.78.
Ledger row `penelope-shell-oscillators` is unverified. Next: shell GOS DCS
(distant longitudinal/transverse + close Møller, $Q_k=U_k$), raw stopping vs
`stp.dat`, then EEDL inner-shell rate substitution and outer-shell rescaling.

Owner decision, 2026-09-24: the SiO₂ $W_{cb}$ is Saito et al.'s amorphous
22 eV, matching the catalog's amorphous density, instead of Da et al.'s 23.6 eV
(2.65 g/cm³). $a$ becomes 3.4701. `penelope-shell-oscillators` is rederived.
