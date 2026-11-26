# Non-basal CXR from fiber-textured HOPG and h-BN

**Status:** Phase I research note for [#1480](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1480), raised by the question in [#1476](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1476). It changes no production default. Calculations: `checks/fiber_nonbasal_cxr_1480.py`. Claims: `cxr-fiber-azimuthal-kinematics`, `graphite-nonbasal-structure-factor` (ledger: [core coherent physics](../../validation/ledger-core-coherent-physics.md)).

**Question.** The catalog marks HOPG and pyrolytic h-BN `texture = "fiber"`, and since #1301 (`reflection-yield-tolerance-selector`, v7) the reflection selector keeps only `(00l)` for them. The stated reason was that random in-plane crystallite azimuths remove the non-basal lines. Is that restriction an adequate CXR approximation, and where?

**Short answer.** No, not in general. Random azimuths average non-basal intensity; they do not remove it. The intensity of one reflection is $\lvert A_g\rvert^2\ge0$ at every azimuth, so its average is positive. Phase averaging only removes interference between distinct reflections, and that interference is already zero for unshaped electrons (final-state orthogonality, see [Coherence](#coherence-what-averages-and-what-does-not)). Under the production per-reflection model, basal-only captures 2–75 % of the fiber-averaged yield in the scenes measured below at $\lvert g\rvert\le6$ Å⁻¹ (median about 30 %). The full sum converges only near 24 Å⁻¹, which halves those shares: converged, basal-only carries 5–19 % of the CXR line yield over 20°–135° (0.4 % at the basal back-reflection null), and the omitted part is mostly CBS. The matched continuum subtraction absorbs most of the extra line yield, so total CXR + continuum changes by only 3–7 %. The (00l) peak positions do not move. Basal-only is adequate only for narrow (00l) windows at backward, near-axial, high-energy geometry. Answers to each acceptance question: [Acceptance questions](#acceptance-questions).

## Literature

- **Shi et al. 2024** (Nat. Commun. 15, 7803; Zotero `Q4I3JISU`, SI `6YMV48GX`). Eq. (6) is the exact recoil root used here. SI §S5 (WSe₂, graphite): "the dominant peaks are associated with $g_\perp=0$ … nonzero $g_\perp$ … leads to extra peaks, those additional peaks may not be distinguishable from the incoherent background in experiments." That is a statement about one fixed single-crystal orientation and one observation angle, with no numbers. It agrees with our finding that $(002)$ is the largest single family at backward angles. It does not bound the six-member, azimuth-summed families, and it does not address forward angles or oblique incidence.
- **#1476 correspondent (PXR/CBS researcher).** Non-basal contributions are "small for HOPG or hBN when looking at emissions from (002) or (004)", but "sometimes there's no dominant plane so we have to consider all of them". They recommend the $p_i=p_f+\hbar g+\hbar k$ convention, which production already uses. Our numbers are consistent with "small inside a narrow (002) window at backward angles", where recoil shifts the strong non-basal bands off the $(002)$ peak. They are not consistent with "small" for integrated or wide-window yields.
- **Textured polycrystals.** Eliseyev, Kubankin et al., "Parametric X-ray radiation from relativistic electrons interacting with a textured polycrystals", RREPS-15 (2015), measured nonzero, orientation-dependent PXR from a textured W foil (7 MeV, (200), 90°–180°). Their abstract calls PXR from polycrystals ("polarization bremsstrahlung", Nasonov and co-workers) well studied. Both support averaging of intensity, not its cancellation. The full paper and its fiber-texture model were not retrieved (bot check).
- **Mosaic pyrolytic graphite.** Adishev, Kaplin et al. (2003, 5.7 MeV, 0.3 mm mosaic PG) observed the $(002)$ Bragg-geometry PXR peak. Fiorito et al. (PAC 1993, NRL, CAPG) observed up to eight basal orders. Vnukov et al. (2001) show that diffraction of real photons in mosaic crystals contributes to measured spectra. These are basal-geometry measurements and do not test non-basal families.
- **Thermal factors.** Graphite: Trucano & Chen, Nature 258, 136 (1975), $U_{11}=0.0031$, $U_{33}=0.016/0.017$ Å² (packaged by #535). h-BN: Pease (1952) gives $00l$ only (Zotero `RV4Z2ZWJ`). Neither Zotero nor a web search found a sourced in-plane h-BN ADP. Per the repository Zotero rule, that is not proof that none exists.
- **Not found** (Zotero and web, this pass): a calculation or measurement of non-basal PXR/CBS from fiber-textured HOPG or h-BN; orientation-averaged CBS for layered materials beyond the isotropic average already used by `brem-lattice-coherent-fraction`.

## Code audit: where the basal restriction and scalar B act

- **Catalog → domain.** `texture = "fiber"` is parsed in `materials/_parse.py::_crystal_texture` and stored as `CRYSTALS[key]["texture"]` with axis `surface_hkl`. `materials/crystal.py::_reflection_families` keeps only multiples of the reduced axis. `dominant_reflections` and the selector (`montecarlo/runner/reflection_cutoff.py`) both enumerate through it, so shells, the tail estimate, the selected set and its identity (`selection_identity`: `texture`, `texture_axis_hkl`) are all basal.
- **Domain → CXR rows.** `mc_cxr_spectrum` itself knows nothing of texture; it scores whatever `hkl_list` it receives at one orientation (`azimuth_rad`). A non-basal list with any roll is scored correctly, which is what the check arms use.
- **Domain → continuum.** `runner/lattice_share.py` subtracts the share of the scored set (`SCORED_CUTOFFS_CASE_KEY`), via `spectrum/brem_lattice_share.py::reflection_shell_weights`. It too is basal under the current selector.
- **Scalar B applied to g.** `materials/crystal.py::debye_waller` through `structure_factor`, `chi_g`, `U_g`, `reflection_coupling_tables` / `emission_coupling_tables` (both line routes), `_reflection_families` (ranking metric), and `brem_lattice_share.reflection_shell_weights` (shell `|U_g|^2`). Every site takes one scalar per call; none projects a tensor.
- **Thermal tensor available.** `materials/channeling/thermal.py::UniaxialThermalSite` (#535) holds the HOPG carbon tensor and already exposes `b_factor_along(cos2_c, T)` and `covariance`. The opt-in model reuses it through a per-reflection callable B (`crystal.per_reflection_b`), so no third material representation is introduced. h-BN has no row.

## Model

PyRITE sums CXR over reflections as intensities: $I=\sum_g I_g$. For a fiber texture with uniform azimuth $\phi$ about the fiber axis $\hat c$, the target is

$$
I_\text{fiber}=\frac1{2\pi}\int_0^{2\pi}\sum_g I_g(\phi)\,d\phi .
$$

The basal-only model computes $\sum_l I_{00l}$, which does not depend on $\phi$. The omitted part is the azimuth average of every $g\not\parallel\hat c$.

A non-basal family is the set of reflections with the same $l$, the same $\lvert g_\perp\rvert$ and the same $\lvert S\rvert$. In P6₃/mmc it has six members related by 60° rolls about $\hat c$. Summing all six members at each $\phi$ gives a function with period 60°, so a uniform average over one 60° sector equals the full-circle average. The check evaluates arms both ways: as deterministic midpoint nodes on the sector (`Q<n>`), and as stratified uniform samples on the full circle with one azimuth per random electron stratum (`S<n>`). The `S<n>` arms do not assume the sixfold closure, and they model one independent crystallite per electron.

## Structure factors

Production basis (`data/cifs/hopg.cif`): layer A at $(0,0,0),(\tfrac13,\tfrac23,0)$, layer B at $(0,0,\tfrac12),(\tfrac23,\tfrac13,\tfrac12)$. With $\omega=e^{2\pi i/3}$ and $m=h-k$,

$$
S_\text{geo}=1+\omega^{h+2k}+(-1)^l\bigl(1+\omega^{2h+k}\bigr)
=1+\omega^{m}+(-1)^l\bigl(1+\omega^{-m}\bigr).
$$

So $\lvert S_\text{geo}\rvert^2=16$ for $m\equiv0$ and even $l$, $0$ for $m\equiv0$ and odd $l$, $1$ for $m\not\equiv0$ and even $l$, and $3$ for $m\not\equiv0$ and odd $l$. Odd $l$ is extinct only on the $m\equiv0$ rows, which include $(00l)$. The $\{10l\}$ rows with odd $l$ are allowed. Their lines fall at half-integer multiples of the $(002)$ energy, where the basal model has no line at all.

`structure hopg` (|g| ≤ 6 Å⁻¹; independent-atom Cromer–Mann $f_0$; $e^{-2W}$ with the #535 uniaxial tensor and, for comparison, with the catalog $B_{33}$ applied isotropically):

| family | m | $g_z$ (Å⁻¹) | $g_\perp$ | $\lvert S\rvert^2_\text{geo}$ | $\lvert S\rvert^2 f_0$ | $e^{-2W}$ tensor | $e^{-2W}$ $B_{33}$ iso |
|---|---:|---:|---:|---:|---:|---:|---:|
| (100) | 6 | 0 | 2.948 | 1 | 9.73 | 0.973 | 0.867 |
| (110) | 6 | 0 | 5.106 | 16 | 59.2 | 0.922 | 0.651 |
| (101) | 6 | 0.936 | 2.948 | 3 | 26.8 | 0.959 | 0.854 |
| (002) | 1 | 1.873 | 0 | 16 | 299.5 | 0.944 | 0.944 |
| (102) | 6 | 1.873 | 2.948 | 1 | 7.18 | 0.919 | 0.818 |
| (112) | 6 | 1.873 | 5.106 | 16 | 54.1 | 0.871 | 0.615 |
| (103) | 6 | 2.809 | 2.948 | 3 | 16.3 | 0.855 | 0.761 |
| (004) | 1 | 3.745 | 0 | 16 | 101.0 | 0.793 | 0.794 |
| (104) | 6 | 3.745 | 2.948 | 1 | 4.12 | 0.772 | 0.688 |

The multiplicity-weighted $\lvert S\rvert^2$ of $(112)$ (6 × 54.1) exceeds that of $(002)$, and $(101)$ is about half of it. $\lvert S\rvert^2$ is not a yield: PXR and CBS weight it with $g$-dependent couplings and polarization factors (see [Single-flight yields](#single-flight-yields)).

h-BN (`structure hbn`): $S=f_B[1+\omega^m(-1)^l]+f_N[\omega^m+(-1)^l]$. Odd-$l$ rows survive only through $f_B-f_N$ (for example $(101)$: $\lvert S\rvert^2=7.7$), so h-BN's half-integer lines are much weaker than graphite's.

**Thermal factors.** The catalog $B_{33}=1.30$ Å² is $8\pi^2U_\parallel$, which is valid for $g\parallel c$ only. For general $g$, $W=\tfrac12 g^\mathsf TUg=\tfrac12(U_\perp g_\perp^2+U_\parallel g_z^2)$. Graphite has $U_\perp=0.0031$, $U_\parallel=0.0165$ Å² (Trucano & Chen 1975, with $U_{11}$ fixed from Ludsteck 1972 rather than refined; packaged for channeling by #535 in `channeling_thermal.toml`). Applying $B_{33}$ isotropically underestimates in-plane-heavy families: by 29 % for $(112)$ and 33 % for $(110)$ in intensity. The check passes the tensor to the scalar-$B$ kernel as $B_\text{eff}=8\pi^2(U_\perp g_\perp^2+U_\parallel g_z^2)/g^2$, which is exact for one uniaxial tensor shared by all sites. Graphite's two sites differ slightly in $U_{33}$ (0.016 and 0.017); the check uses their mean, as the catalog does. h-BN has no packaged in-plane ADP. Its numbers below use the catalog $B_{33}=1.73$ Å² isotropically, which underweights its non-basal families.

## Coherence: what averages and what does not

Phase averaging over crystallites, interference between reflections, and averaging of intensities are distinct effects:

- **Cross-reflection interference.** For an unshaped plane-wave electron, distinct transverse $g$ leave the electron in orthogonal final states. The trace over the final electron removes $A_gA_{g'}^*$ exactly ([shaped-electron note](shaped-electron-cxr-1102.md)), so intensities add whether or not the lines overlap in energy. A coherent field sum across distinct $g_\perp$ would be wrong here. This holds independently of the classical argument in `cross-reflection-coherence`, which relies on spectral separation and on segment-position decorrelation, and which was measured only on basal sets.
- **Orientation averaging.** $\langle A_g\rangle_\phi=0$ does not imply $\langle\lvert A_g\rvert^2\rangle_\phi=0$. Random azimuth removes the coherent sum of fields from different crystallites, not the intensity of each.
- **Basal reflections** have the same $g$ in every crystallite. Their coherence between crystallites depends on stacking registry and domain size; see [Domain coherence](#domain-coherence-open).

## Kinematics

Production convention: the electron gives up $\hbar g$ and emits $\hbar k$, $p_i=p_f+\hbar g+\hbar k$, with the line where $v\cdot g>0$ (`materials/recoil.py`; this is the convention the #1476 correspondent recommends). The exact vacuum root is

$$
\hbar\omega=\frac{\beta\,\hat u\cdot Q-Q^2/2E}{1-\beta\,\hat u\cdot\hat n+Q\cdot\hat n/E},\qquad Q=\hbar c\,g .
$$

- **Axial degeneracy.** With $\hat u=\hat c$, $\hat u\cdot g=g_z$ for every member and every $\phi$, so all reflections with the same $l$ share the classical root. Checked to $4\times10^{-16}$.
- **Recoil splitting.** $Q^2=\hbar^2c^2(g_z^2+g_\perp^2)$ lowers a non-basal line by $\hbar c\,g_\perp^2/(2E\beta g_z)$ relative to its basal partner, independent of $\phi$. $Q\cdot\hat n$ adds $\hbar c\,g_\perp n_\perp\cos(\phi-\phi_n)/(ED)$, which spreads the line over $\phi$ with an arcsine profile. HOPG at 119°:

  | $T$ | (002) exact (eV) | (102) range (eV) | (112) range (eV) |
  |---|---:|---:|---:|
  | 30 keV | 1038.8 | 1003.6–1020.0 | 944.3–971.3 |
  | 100 keV | 1595.2 | 1563.0–1583.7 | 1512.1–1547.1 |
  | 300 keV | 2081.7 | 2057.1–2076.0 | 2020.2–2052.4 |
  | 1 MeV | 2386.7 | 2373.5–2384.5 | 2354.1–2373.1 |

  At 30 keV the $(112)$ band sits 7–9 % below $(002)$. A narrow window on the $(002)$ peak therefore misses much of it, and a wide window or an integrated yield catches it.
- **Oblique incidence.** At tilt $\alpha$ the classical numerator is $g_z\cos\alpha+g_\perp\sin\alpha\cos\phi$. Its half-range over mean is $(g_\perp/g_z)\tan\alpha$ exactly: 2.75 % at 1°, 13.8 % at 5° for $\{10l\}$, $l=2$ ($g_\perp/g_z=1.574$). Broadening redistributes yield; it does not remove it. The $hk0$ families ($g_z=0$) radiate only when $\hat u\cdot g\neq0$, that is, at oblique incidence or after multiple scattering.
- **Vanishing recoil.** The CXR route's $\hbar\to0$ hook reproduces the classical root to $2\times10^{-16}$. The relative shift falls as $1/E$ in the ultra-relativistic limit.
- **Production parity.** The CXR route's `_line_recoil_terms` root agrees with `recoil_resonance_eV` to $1.6\times10^{-15}$ over eleven families, three energies and thirteen azimuths.

## Single-flight yields

One straight flight of 500 nm in a semi-infinite slab, through the production incoherent kernel (`mc_cxr_spectrum`, bin-mean quadrature, Henke form factors, self-absorption). Each family's members are summed and averaged over eight sector azimuths; graphite uses the tensor $B_\text{eff}$. The table gives non-basal yield over basal yield ($\lvert g\rvert\le6$ Å⁻¹; all energies):

| crystal | $T$ | tilt | 20° | 45° | 119° | 160° |
|---|---|---|---:|---:|---:|---:|
| HOPG | 30 keV | 0° | 22.5 | 4.05 | 0.75 | 3.6 |
| HOPG | 30 keV | 5° | 35.0 | 5.34 | 0.92 | 4.9 |
| HOPG | 100 keV | 0° | 28.1 | 7.01 | 0.84 | 2.1 |
| HOPG | 100 keV | 5° | 69.6 | 11.9 | 1.02 | 2.7 |
| HOPG | 1 MeV | 0° | 49.8 | 90.4 | 0.96 | 0.12 |
| HOPG | 1 MeV | 5° | 187 | 364 | 1.33 | 0.16 |
| h-BN | 30 keV | 0° | 10.6 | 2.19 | 0.53 | 2.4 |
| h-BN | 100 keV | 0° | 12.4 | 3.40 | 0.48 | 1.2 |
| h-BN | 1 MeV | 0° | 22.2 | 37.2 | 0.57 | 0.07 |
| h-BN | 1 MeV | 5° | 263 | 328 | 0.93 | 0.11 |

Observation angles are polar angles from the beam (+z = fiber axis). The 45° column (rerun 2026-11-26, `single-flight-45/`) replaces a first pass at 90°, which grazes the infinite slab and is fully absorbed. At 45° the basal share of the single-flight yield is 8–20 % for HOPG at 30–100 keV (h-BN 11–31 %); at 1 MeV the (002) PXR is so polarization-suppressed at 45° that (101) alone is 41× the whole basal yield (axial) and the 5° tilt adds a (110) line at 195×. At forward angles the basal PXR is polarization-suppressed, so non-basal families dominate. At 119° the $(002)$ line remains the largest single family, consistent with Shi et al. SI §5 ("the dominant peaks are associated with $g_\perp=0$"), yet the six-member non-basal families together match it. The leading omitted families are $(112)$, $(101)$ (a new line at half the $(002)$ energy), $(103)$ and, with tilt, $(110)$.

## Frozen-transport arms

The #1301 artifacts named in the plan (`cleo_sub`, `bells_45deg`, job 13676 shells) are no longer on qlmc or locally, so every arm uses a fresh seeded transport instead. Arms that share a transport are checked by segment fingerprint. One local transport per scene (300 electrons, production `_simulate_trajectories`, catalog composition, tilt azimuth 180° as in the lab profiles). The lab detector direction is mapped into the sample frame with `detector_directions`. Every arm scores the same segments through `mc_cxr_spectrum` (incoherent, bin-mean, Henke), cutoff $\lvert g\rvert\le6$ Å⁻¹. Arms:

- **A** basal only, catalog $B_{33}$ (the current model);
- **B** full allowed domain at one fixed roll $\phi=0$;
- **S8, S32, S128** full domain; each electron belongs to one of $N$ random strata, and each stratum takes one stratified-uniform azimuth on the full circle (one crystallite per electron);
- **Q32** full domain on 32 midpoint rolls of one 60° sector (reference);
- **B33iso** Q32 with $B_{33}$ applied isotropically (thermal-factor sensitivity).

Columns: A/Q is the basal-only share of the full-fiber yield. "Window" is full over basal inside ±2 % of the basal peak. The two right columns give where the full-fiber peak sits and the largest families as shares of the total.

| scene (lab detector 45° unless noted) | A/Q | window | full peak (basal peak) | leading families |
|---|---:|---:|---|---|
| HOPG 30 keV, 500 nm, axial | 0.18 | 1.25 | 1548 eV (same) | (112) 0.26, (101) 0.21, (002) 0.14, (110) 0.13 |
| HOPG 30 keV, 500 nm, tilt 2° | 0.20 | 1.23 | 1532 (same) | (112) 0.26, (101) 0.20, (002) 0.15 |
| HOPG 30 keV, 500 nm, tilt 15° | 0.32 | 1.15 | 1426 (same) | (002) 0.25, (112) 0.24, (101) 0.16 |
| HOPG 30 keV, 500 nm, tilt 15°, 135° | 0.32 | 1.09 | 938 (same) | (002) 0.30, (112) 0.24, (101) 0.17 |
| HOPG 30 keV, 30 nm, tilt 15° | 0.26 | 1.09 | 684 (1430) | (101) 0.28, (112) 0.24, (002) 0.22 |
| HOPG 30 keV, 10 µm, tilt 15° | 0.26 | 2.20 | 1928 (2812) | (112) 0.22, (004) 0.16, (110) 0.12 |
| HOPG 300 keV, 500 nm, tilt 15° | 0.037 | 8.5 | 2318 (4658) | (101) 0.36, (112) 0.28, (110) 0.11 |
| HOPG 300 keV, 500 nm, tilt 60° | 0.37 | 1.33 | 2384 (same) | (002) 0.35, (101) 0.27, (112) 0.21 |
| HOPG 1 MeV, 500 nm, tilt 15° | 0.020 | 16.3 | 3274 (6544) | (101) 0.45, (112) 0.28, (201) 0.09 |
| HOPG 1 MeV, 500 nm, axial, 135° | 0.75 | 1.10 | 2088 (same) | (002) 0.72, (101) 0.13, (112) 0.06 |
| h-BN 30 keV, 500 nm, tilt 15° | 0.38 | 1.26 | 1438 (same) | (002) 0.32, (112) 0.19, (102) 0.17 |
| h-BN 300 keV, 500 nm, tilt 60° | 0.51 | 1.52 | 2402 (same) | (002) 0.48, (112) 0.19, (102) 0.19 |

- **Azimuth sampling converges fast in yield, slower in shape.** S8, S32 and S128 agree with Q32 within ±0.5 % in integrated yield in every scene (the spread is stratum-population noise at 300 electrons). Fixed $\phi=0$ (B) is within 0.1–2 %. In spectral shape (sector quadrature, 30 keV, tilt 15°), 8 rolls differ from 32 by L1 4.1 %, and 128 by 1.5 %.
- **The reciprocal cutoff converges only near 24 Å⁻¹.** At 6 Å⁻¹ the full-fiber line sum is about half its converged value, so every A/Q in the table overstates the basal share by roughly 2× (see [Reciprocal-cutoff convergence](#reciprocal-cutoff-convergence-and-arm-f)). The cutoff, not the roll count, drives cost.
- **Thermal tensor matters.** B33iso/Q32 = 0.854 (30 keV, tilt 15°) and 0.877 (300 keV, tilt 60°). Lifting the basal restriction while keeping the isotropic $B_{33}$ would err by 12–15 % in total yield.
- **PXR/CBS split.** The non-basal yield is relatively CBS-rich. The PXR share falls from 0.31 (basal) to 0.21 (full) at 30 keV, tilt 15°, and from 0.83 to 0.52 at 300 keV, tilt 60°. At 1 MeV, tilt 15°, the basal arm has strong PXR–CBS destructive interference (diagonal PXR alone exceeds the total).
- **Below 300 eV.** 0–14 % of the full yield, mostly $hk0$ lines with small $v\cdot g$. These sit near the carbon K edge, where the line model and the absorption tables carry the most uncertainty.
- **Thick target.** The first 1 mm attempt (tilt 45°, lab 45°) returned no lines because that detector direction lies in the exit face. It was rerun at two non-grazing geometries, both viewed through the entrance face; see [Thick target](#thick-target).

Raw records (JSON and NPZ per scene, single-flight table, share script): `~/dev/pyrite-benches/1480-fiber-nonbasal/` with `SHA256SUMS.all`, outside the repository.

### Thick target

HOPG, 30 keV, 1 mm (electrons stop within about 10 µm), 300 electrons, frozen check with nested cutoffs 6/8/10 Å⁻¹. Yields are above 300 eV; Q32 is the 32-node sector quadrature. Raw: `thick/`.

| scene | A/Q at 6 / 8 / 10 Å⁻¹ | full yield 6 / 8 / 10 (rel. to 10) | B, S8 vs Q32 | PXR, CBS share (A → Q32) | leading families (Q32) |
|---|---|---|---|---|---|
| tilt 45°, lab 90° (45° from normal) | 0.33 / 0.24 / 0.19 | 0.57 / 0.80 / 1 | +0.02 %, −0.01 % | 0.62, 0.11 → 0.25, 0.41 | (002) 0.13, (112) 0.11, (114) 0.07, (110) 0.07 |
| tilt 15°, lab 135° (30° from normal) | 0.21 / 0.15 / 0.12 | 0.55 / 0.78 / 1 | +0.01 %, 0.00 % | 0.65, 0.11 → 0.23, 0.44 | (112) 0.12, (002) 0.08, (110) 0.08, (114) 0.07 |

The thick target keeps the (002) peak position at 45° tilt (976 eV) and adds no new dominant line. The basal share is the lowest of all measured scenes and is still falling at 10 Å⁻¹, as in the thin-target ladder below. Below 300 eV is 5–7 % of the full yield; the 15° scene shows a narrow 10 eV maximum from $hk0$ lines with $v\cdot g\to0$ that carries 2.6 % of the yield and lies outside any physically meaningful band.

### Reciprocal-cutoff convergence and arm F

`checks/fiber_paired_1480.py --ladder` scores the opt-in model (`cxr-fiber-intensity-average`) on one frozen transport per scene: HOPG 500 nm, 600 electrons, ladder-case mosaic quadrature, 32 sector rolls, 2 eV bins. Each |g| band is scored once, and each cutoff gets the matched continuum subtraction over its own full prefix domain. The smallest cutoff reproduces `score_fiber_case` to $3\times10^{-16}$ (CXR) and exactly (continuum). The runs used the RTX 5080 with device FP64 after a parity check against FP64 CPU on the same transport (32 electrons, cutoffs 6/10): GPU FP64 yields within $2\times10^{-6}$ and spectral L1 $\le4\times10^{-5}$; GPU FP32 within $3\times10^{-6}$ and L1 $\le1.3\times10^{-4}$. Raw: `ladder/`, `gpu-parity/`.

| cutoff (Å⁻¹) | reflections | 30 keV, tilt 15°, lab 45°: CXR / basal share / total | 300 keV, tilt 60°, lab 45°: CXR / basal share / total |
|---:|---:|---|---|
| 6 | 108 | 1.098e-6 / 0.307 / 9.618e-6 | 5.457e-7 / 0.341 / 5.886e-6 |
| 8 | 242 | +25.3 % / 0.248 / +1.1 % | +21.4 % / 0.281 / +0.7 % |
| 10 | 478 | +16.8 % / 0.213 / +0.9 % | +15.0 % / 0.245 / +0.7 % |
| 12 | 852 | +13.0 % / 0.189 / +0.8 % | +12.1 % / 0.218 / +0.6 % |
| 16 | 2008 | +11.4 % / 0.169 / +0.7 % | +10.6 % / 0.197 / +0.6 % |
| 20 | 3974 | +4.7 % / 0.162 / +0.3 % | +4.5 % / 0.189 / +0.3 % |
| 24 | 6828 | +2.0 % / 0.159 / +0.1 % | +2.0 % / 0.185 / +0.1 % |

Percentages are steps from the previous cutoff. Yields are per electron per steradian, integrated over 20–25000 eV (30 keV) and 20–60000 eV (300 keV).

- **Lines converge slowly; the total barely moves.** From 6 to 24 Å⁻¹ the CXR line yield grows 1.97× (30 keV) and 1.85× (300 keV), with the last step at 2 %. The matched subtraction grows from 5.2 % to 13.0 % (30 keV) and from 3.5 % to 8.6 % (300 keV) of the amorphous continuum, with no clamped nodes. It cancels 65 % and 61 % of the added line yield. The total changes by 3.8 % and 3.0 %. The isotropic shell kernel and the fiber-oriented lines balance only in this integrated sense; the spectral shape still changes.
- **Converged basal share.** At 24 Å⁻¹ basal-only carries 0.16 (30 keV) and 0.19 (300 keV) of the CXR line yield, about 0.52–0.54 of the 6 Å⁻¹ value. Applying the same factor to the HOPG rows of the frozen-arm table (0.02–0.37 at 6 Å⁻¹, excluding the 1 MeV backward scene) gives roughly 0.01–0.20. The 1 MeV backward scene (0.75 at 6 Å⁻¹) and h-BN (no tensor, model refuses it) were not laddered.
- **20–6000 eV band.** At 30 keV the in-band line yield grows 1.94× to 24 Å⁻¹ (last step +1.9 %). This reproduces the earlier 150-electron qualification (+22 % / +43 % at 8 / 10 Å⁻¹; here +25 % / +46 %).
- **Arm F (converged reference).** Same transport as the 30 keV ladder (fingerprints equal), 128 sector rolls at 6/10/16 Å⁻¹. Against 32 rolls, integrated CXR differs by +0.06 %, +0.04 % and +0.03 %, but the spectral L1 is 3.3 %, 2.9 % and 2.4 %: integrated yield is roll-converged at 32, line shapes are not. Combining both axes, the converged reference for this scene is the 24 Å⁻¹, 32-roll ladder rung (CXR 2.161e-6, total 9.986e-6 per electron per sr), with an estimated residual of about +1–2 % in line yield from the cutoff tail (steps of 4.7 % then 2.0 %), under 0.1 % from rolls, and about 2–3 % spectral L1 from rolls. Its basal share, 0.16, is the converged arm-A/arm-F ratio. Wall 19.3 min on the GPU (832 s for the 16 Å⁻¹ band alone at 128 rolls), 1.4 GB host RSS. Raw: `armF/`.
- **Cost.** Line scoring of the 24 Å⁻¹ band (2854 new reflections × 32 rolls × 600 electrons) took 373 s on the RTX 5080. The whole 30 keV ladder took 15.6 min wall and 1.4 GB host RSS; the 300 keV ladder 11.5 min. Cost grows with reflections × rolls (about cutoff³). On 8 CPU threads the same work is 4–5× slower; the thick 45° frozen scene took 81 min on 16 CPU threads against 19 min for the 15° scene on the GPU.

### Converged observables: detector angle and PXR/CBS split

`fiber_paired_1480.py --ladder 6 8 10 12 16 20 24 --components`, the same transports as the ladder (fingerprints equal; totals reproduce the ladder exactly), RTX 5080 device FP64, 32 rolls. HOPG 500 nm. The 30 keV, tilt 15° scene is scored at four detector polar angles; the angle to the fiber axis is in parentheses. Split columns give PXR / CBS / interference ($2\,\mathrm{Re}\,A_\text{PXR}A_\text{CBS}^*$) as fractions of each sum's CXR. "Window" is full over basal within ±2 % of the basal peak. Raw: `components/`.

| scene | basal share at 6 → 24 Å⁻¹ | CXR at 24 (per e per sr) | basal split | non-basal split at 24 | window at 6 → 24 | CXR 6→24, total 6→24 |
|---|---|---:|---|---|---|---|
| 30 keV, lab 20° (35°) | 0.100 → 0.048 | 3.24e-6 | 0.21 / 0.34 / +0.45 | 0.08 / 0.69 / +0.23 | 1.31 → 1.79 | 2.13×, +4.6 % |
| 30 keV, lab 45° (60°) | 0.307 → 0.159 | 2.16e-6 | 0.29 / 0.25 / +0.46 | 0.10 / 0.64 / +0.26 | 1.10 → 1.24 | 1.97×, +3.8 % |
| 30 keV, lab 135° (150°) | 0.285 → 0.157 | 1.38e-6 | 0.79 / 0.02 / +0.19 | 0.18 / 0.55 / +0.27 | 1.08 → 1.16 | 1.82×, +5.8 % |
| 30 keV, lab 160° (175°) | 0.008 → 0.004 | 1.20e-6 | 0.91 / 0.13 / −0.04 | 0.22 / 0.52 / +0.26 | 5.1 → 9.4 | 2.11×, +6.7 % |
| 300 keV, tilt 60°, lab 45° | 0.341 → 0.185 | 1.01e-6 | 0.78 / 0.02 / +0.20 | 0.07 / 0.69 / +0.24 | 1.03 → 1.06 | 1.85×, +3.0 % |

- **The omitted yield is mostly CBS.** At 24 Å⁻¹ the non-basal sum is 52–69 % CBS and 7–22 % PXR, with constructive interference of +0.23 to +0.27. Its PXR fraction falls with cutoff (high-|g| rows radiate mainly through CBS). Basal PXR dominates only backward (0.79–0.91) and at 300 keV.
- **Angular dependence.** The converged basal share is 0.16 at 45° and 135° but 0.05 forward (20°), where basal PXR is polarization-suppressed. At 160° the detector is 5° from the basal back-reflection direction ($-\hat c$). There the basal yield is 50× lower than at 135°, and non-basal and $hk0$ lines carry the spectrum; the full-fiber maximum is a 169 eV $hk0$ feature, and 17 % of the yield lies below 300 eV. That scene is a geometric null of the basal model, not a typical backward value. Above 300 eV, the basal shares at 24 Å⁻¹ are 0.056, 0.175, 0.18 and 0.004.
- **Narrow features.** The full-fiber peak stays on the basal (00l) peak in every scene except 160°. Within the ±2 % (002) window, non-basal adds 24 % (30 keV, 45°) and 16 % (135°) converged, against 10 % and 8 % at 6 Å⁻¹; at 300 keV, tilt 60°, it adds 6 %. The peak FWHM grows 21.6 → 23.2 eV (+7 %) at 30 keV and is unchanged (3.5 eV) at 300 keV. The peak line-to-continuum ratio rises 16 % and 7 %. The CXR centroid moves up by 420 eV and 1745 eV from 6 to 24 Å⁻¹: the added yield is spread over many weak high-energy lines rather than one new feature.
- **Signal to background.** The integrated CXR-to-continuum ratio at 45° rises from 0.13 to 0.28 (30 keV) and 0.10 to 0.20 (300 keV) between 6 and 24 Å⁻¹, partly because the matched subtraction removes 13 % and 9 % of the continuum.
- **Polarization** is not available: production CXR returns no polarization-resolved spectrum.
- **Cost.** Components add a direct three-buffer call per batch: 33–46 min per ladder on the RTX 5080, against 12–16 min without.

## CBS continuum consistency

`brem-lattice-coherent-fraction` removes from the BremsLib continuum, per scored reflection, the full-sphere orientation-averaged CBS share $\lvert U_g\rvert^2H(\lvert g\rvert)$. The continuum therefore keeps the average share of every **unscored** reflection. Under basal-only scoring:

- non-basal CBS is not lost; it stays in the continuum, smeared as an amorphous-like background;
- non-basal PXR is not in BremsLib at all, so it is lost.

Energy-weighted share of the scored set ($\lvert g\rvert\le6$ Å⁻¹; script archived at
`~/dev/pyrite-benches/1480-fiber-nonbasal/share1480.py`): non-basal over basal is 4.70, 4.66 and 4.60 for HOPG at 30 keV, 300 keV and 1 MeV, and 3.43, 3.40 and 3.36 for h-BN. The subtraction grows about 5.7× when the full domain is scored. The mechanism already takes an explicit reflection list, so the extension needs no new kernel. It needs the per-reflection tensor $B$ in $U_g$ (with $B_{33}$ isotropic the HOPG non-basal share is 24 % low) and the shells summed before the clamp. The opt-in model does both (`cxr-fiber-intensity-average`). The isotropic kernel is not texture-conditioned: for a fiber, the per-orientation line yield differs from the 4π average, so subtraction and lines balance only in total. #1250 is the precedent.

## Domain coherence (open)

Phase II measurements of all four points below, with a proposed per-crystal domain model: [Crystallite structure of fiber CXR](fiber-domain-structure-1485.md) (#1485).

- **Crystallite size and formation length.** CXR formation lengths are nanometres for keV photons. HOPG crystallites are micrometres in plane and of order 0.1 µm along $c$. One roll per coherent region is therefore well justified for HOPG, and the incoherent mean does not depend on whether one track crosses several crystallites: the mean is linear in the orientation distribution. Only line shapes and inter-segment phase in the coherent mode depend on domain size.
- **Several domains along one track.** In incoherent mode the result equals the per-segment orientation average. In the coherent mode (`coherent-emission`), phases across a domain boundary are not modelled; the opt-in model refuses coherent emission.
- **Basal coherence between domains.** Basal $g$ is common to all rolls, but stacking-registry shifts between crystallites randomize the basal phase too. This matters only for the coherent mode.
- **Turbostratic stacking.** The model assumes Bernal crystallites. Turbostratic layers (random rolls layer by layer) turn $(hkl)$ with $g_\perp\neq0$ into $(hk)$ rods with continuous $g_z$. Their lines then spread in energy rather than sitting at the Bernal $l$ values. Pyrolytic h-BN is turbostratic by the 2026-11-18 decision, so its non-basal numbers above are a Bernal idealisation. HOPG has a few-percent rhombohedral and turbostratic fraction.
- **Mosaic spread** (0.4–3.5° FWHM) tilts $c$ and adds the oblique broadening above. It has not been combined with the roll average in these runs.

## Applicability envelope

All A/Q values in the frozen-arm table are at $\lvert g\rvert\le6$ Å⁻¹. The converged basal share is about half of each (0.52–0.54 in the two laddered scenes). Basal-only is adequate (about 10 % inside the (00l) window, peak position unchanged) only for **backward detection at high energy with near-axial incidence** (1 MeV, axial, 135°: window +10 %) **when the observable is a narrow window on the (00l) peak**. Even there, it omits 25 % of the integrated yield (A/Q = 0.75). It fails for:

- integrated or wide-window yields: A/Q = 0.02–0.51 across all other measured scenes;
- forward and side detection, where basal PXR is polarization-suppressed (single flight: non-basal 10–50× basal at 20°);
- oblique incidence at 300 keV–1 MeV (15° tilt, 45°): the strongest feature moves to the odd-$l$ (101) line at half the (002) energy;
- thick targets and multiple scattering, which add $hk0$ lines;
- any lifting of the restriction while $B_{33}$ stays isotropic (12–15 % error).

Even in the (002) window, non-basal adds 9–33 % at 6 Å⁻¹ in typical 30–300 keV scenes, because recoil shifts the (102) and (112) bands only 3–9 % below (002). Converged (24 Å⁻¹) it adds 6 % (300 keV, tilt 60°), 16 % (30 keV, 135°), 24 % (45°) and 79 % (20°).

## Results to reassess

The basal restriction affects any HOPG/h-BN number read as a total, forward or side yield. It does not affect the line positions of the (00l) set.

- **Tutorials using HOPG.** `03-pxr-line-energies` predicts ideal (00l) energies, which are unaffected. `04-beams-crystals-target-geometry` (135°), `08-parameter-sweeps` (yield against thickness) and `11-optimizing-xray-source` (source optimisation) report basal-only yields. At their 30 keV, backward geometries the measured basal share is 0.18–0.32 of the full fiber yield. Their text should say the yields are basal-only (deferred to #1483).
- **#1301 selector evidence** (cleo_sub, bells_45deg): valid as convergence within the basal set; it is not completeness evidence. The selector claim now records this ("Fiber-domain applicability (#1480)").
- **#1250 continuum subtraction** for HOPG/h-BN: non-basal CBS currently stays in the continuum as an isotropic average, while non-basal PXR is absent. Total HOPG yields in published PyRITE comparisons are low by the missing PXR and mis-shaped by the averaged CBS.
- **#1476 correspondence and Shi et al. 2024 SI §S5**: both concern narrow (002)/(004) windows. The +9–33 % in-window non-basal addition at 30–300 keV is within neither statement's scope, so it should be checked against their geometry (#1486).
- **h-BN results** inherit the same restriction, plus a Bernal idealisation of a turbostratic material (#1485) and no in-plane ADP (#1482).
- **Experimental comparisons** of HOPG integrated or forward-angle PXR yields that were judged against PyRITE basal-only output need re-reading. Anchored ledger claims that use HOPG (for example `line-recoil-in-crystal`, `cbs-amplitude`, `line-energy-dispersion`) concern line positions or per-reflection amplitudes, which the restriction leaves intact. Any whose anchor is an integrated HOPG yield should be re-checked (not audited row by row here).

## Acceptance questions

1. **Elimination or redistribution?** Redistribution. Each reflection's intensity is non-negative at every roll, so its average is positive. Phase averaging removes only cross terms, which already vanish for unshaped electrons. Measured: 8, 32 and 128 rolls agree in yield within 0.5 % (0.06 % between 32 and 128), and fixed $\phi=0$ is within 2 %.
2. **Size for HOPG.** Converged at 24 Å⁻¹, basal-only is 5–19 % of the CXR line yield across 20°–135° (30 keV and 300 keV), so the omitted part is 4–20× the basal yield. Total CXR + continuum changes only 3–7 %, because the matched subtraction cancels 61–65 % of the added lines. **For h-BN**, measured only at 6 Å⁻¹ with isotropic $B_{33}$ (no in-plane ADP, #1482): basal share 0.38 and 0.51, and single-flight non-basal/basal ratios about half of HOPG's. Converged h-BN numbers wait for #1482, and its turbostratic stacking makes Bernal numbers an idealisation (#1485).
3. **Dominant families.** At 6 Å⁻¹: (112), the odd-$l$ (101) at half the (002) energy, (110) with tilt or multiple scattering, (103), and (114) in thick targets. Beyond 6 Å⁻¹ no family dominates: the CXR doubles over thousands of weak, mostly CBS rows.
4. **Scaling.** *Energy:* at oblique forward incidence the basal share falls from 0.32 (30 keV) to 0.037 (300 keV) and 0.020 (1 MeV, 6 Å⁻¹, tilt 15°), because (101) takes over; at backward axial 1 MeV it is 0.75. *Detector:* 0.05 forward, 0.16 at 45° and 135°, 0.004 at the basal back-reflection null. *Thickness:* 30 nm and 10 µm 0.26, 1 mm 0.12–0.19 at 10 Å⁻¹ (all lower converged). *Multiple scattering and tilt:* add $hk0$ lines and broaden by $(g_\perp/g_z)\tan\alpha$.
5. **Recoil and azimuthal broadening.** At 119° and 30 keV, (112) sits 7–9 % below (002), spread over 27 eV; at 1 MeV, 0.6–1.4 %. Oblique broadening is 2.75 % at 1° and 13.8 % at 5° for $\{10l\}$, $l=2$. In the cutoff ladder the (002) peak FWHM grows 7 % at 30 keV (21.6 → 23.2 eV) and not at all at 300 keV. The (00l) peak positions do not move.
6. **Anisotropic thermal factors.** Essential. Isotropic $B_{33}$ lowers in-plane-heavy families by 29–33 % in intensity and total fiber yield by 12–15 %.
7. **CBS continuum extension.** Subtract the isotropic lattice-coherent share over the same full scored domain, with the per-reflection tensor $B$ in $U_g$, summing shells before the clamp (done in the opt-in route). It grows from 5.2 % to 13.0 % of the continuum at 30 keV (3.5 % to 8.6 % at 300 keV), with no clamped nodes. Lines and continuum then balance in total only; a texture-conditioned diffuse kernel is open (#1485).
8. **When basal-only is adequate.** Only for narrow (00l)-window observables at backward, near-axial, high-energy geometry (1 MeV, axial, 135°: window +10 %, at 6 Å⁻¹). It fails for integrated, wide-window, forward or side, oblique and thick-target observables, and near the basal back-reflection direction. See [Applicability envelope](#applicability-envelope).
9. **Cost of a fuller model.** A converged fiber ladder (24 Å⁻¹, 6828 reflections, 32 rolls, 600 electrons, 500 nm) takes 12–16 min on one RTX 5080 (device FP64, parity-checked against CPU FP64 to $2\times10^{-6}$), with 1.4 GB host memory. CPU is 4–5× slower. Cost scales with reflections × rolls, about cutoff³. Integrated yield needs 8 rolls; line shapes need about 128. A production default needs the selector over the fiber domain (#1484) to trim the cutoff tail.
10. **Open domain and coherence assumptions.** One roll per coherent region (well justified for HOPG); no inter-domain phase in coherent mode (the model refuses coherent emission); basal phase between stacking-shifted crystallites; turbostratic rods (pyrolytic h-BN, a few percent in HOPG); mosaic combined with roll; texture-conditioned continuum (#1485).
11. **Results to reassess.** Tutorials 04, 08 and 11 (basal-only yields at 30 keV backward); #1301 selector evidence (convergence within the basal set only); #1250 HOPG/h-BN continuum subtraction; the #1476 and Shi et al. SI §S5 window statements (#1486); all h-BN totals; HOPG integrated or forward yields compared with experiment. Line-position and per-reflection anchors are unaffected. Details: [Results to reassess](#results-to-reassess).

## Phase II

1. **Thermal tensor in CXR.** Per-reflection $B(g)$ from the #535 uniaxial data (done for HOPG in the opt-in model). Source an in-plane h-BN ADP before any h-BN full-fiber scoring.
2. **Texture as a reflection-domain choice.** Separate catalog texture (single crystal, fiber, powder) from the reflection domain (basal-only as an explicit approximation, full fiber).
3. **Fiber quadrature.** Sector midpoints with verified roll closure; 8 rolls suffice for integrated yield (0.5 %); 32 rolls reach 0.06 % in yield but 2–3 % spectral L1 against 128 (arm F), so narrow line shapes need about 128.
   **Cost.** Converged cutoff is about 24 Å⁻¹ (6828 reflections). A full 32-roll ladder to that cutoff costs 12–16 min on one RTX 5080 for 600 electrons through 500 nm (device FP64, parity-checked); CPU is 4–5× slower. Roll count is not the cost driver.
4. **Selector over the full fiber domain.** The tolerance stop measured on fiber-averaged rows.
5. **Continuum consistency.** Matched subtraction over the scored full domain (done in the opt-in paired route).
6. **Domain and stacking models.** Turbostratic rods, mosaic with roll, coherent-mode domain phases.
7. **Validation against measurement.** Every number here is the production per-reflection model agreeing with itself across arms. An independent calculation or a measured non-basal HOPG spectrum is required before any default changes.
