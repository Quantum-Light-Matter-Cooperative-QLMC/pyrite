# Candidate materials for nonrelativistic PXR and CBS

## Purpose and scope

This note surveys crystalline materials that are missing from cxr-mc's current
material registry and may be promising for parametric X-ray radiation (PXR) or
coherent bremsstrahlung (CBS) driven by electron beams in the tens to low
hundreds of keV range. The selection criteria are:

1. a strong dominant coherent line;
2. a large coherent-to-incoherent emission ratio;
3. high absolute coherent flux;
4. narrow experimental linewidth; and
5. practical availability as a sufficiently large, well-oriented crystal.

No detector threshold is imposed. Sub-keV and soft-X-ray lines are therefore
fully in scope.

The current registry already contains diamond, silicon, LiF, sapphire, HOPG,
h-BN, MoSe2, WSe2, MoTe2, PtSe2, HfS2, HfSe2, ZrSe2, WS2, MoS2, NbS2, and
NbSe2. The principal omissions identified here are V2O5, TiS2, selected
intercalated niobium dichalcogenides, MoO3, GaS, ZrS2, SnS2, and TiSe2.

This is a research survey, not a validation-ledger sign-off. Numerical results
from the literature must be reproduced in the cxr-mc geometry before they are
treated as predictions for this project.

## Physical selection criteria

Zhai et al. derive the following scaling for the ratio of tunable coherent
emission to bremsstrahlung in a fixed photon-energy window:

\[
\frac{\Phi_{\mathrm{tunx}}}{\Phi_{\mathrm{brem}}}
\propto
\frac{E_p |\chi_{\mathbf g}|^2}
{(Z_{\mathrm{eff}}/E_p) T_{\mathrm{accum}}}.
\]

Here, \(\chi_{\mathbf g}\) measures the diffraction strength of a reciprocal
lattice vector, \(Z_{\mathrm{eff}}\) controls the bremsstrahlung scale, and
\(T_{\mathrm{accum}}\) incorporates the scattered-electron distribution and
self-absorption. A long electron mean free path also increases the coherent
interaction length. These considerations explain why layered crystals made
from light or intermediate-mass elements can have a much larger
coherent-to-bremsstrahlung ratio than conventional dense crystals.

This ratio does **not** determine absolute coherent flux by itself. Heavy
chalcogenides can have greater absolute flux even when a lighter compound has
a better coherent-to-background ratio. The source paper demonstrates this
tradeoff explicitly for h-BN versus MoSe2 and WSe2.

Two further distinctions matter:

- The published screening metric is most directly a PXR metric because it is
  built around \(|\chi_{\mathbf g}|^2\). CBS depends on the Fourier component
  \(U_{\mathbf g}\) of the electron-crystal potential and may reorder the
  candidates.
- The reported denominator is continuous bremsstrahlung in a 10 eV window,
  not total incoherent emission. Characteristic X-rays, detector response, and
  spectral overlap must be included before interpreting the result as a total
  signal-to-background ratio.

## Quantitative reconstruction of the published material screen

The strongest quantitative source found is the 2025 Nature Communications
study by Zhai et al. Its Supplementary Table 5 lists 21 materials and four
dominant planes per material. The deposited Figure 2a workbook contains the
mean-free-path coordinate, the governing material factor, and the simulated
coherent-to-bremsstrahlung ratio, but not material names on each row.

The values below were reconstructed by separating the workbook into its 84
ordered plane series and aligning those series with Supplementary Table 5. The
17 electron-energy points correspond to the paper's ordered 10--300 keV grid;
the entries at 30, 100, and 150 keV are shown. The final column is the ratio of
the best plane's score to the second-best of the four selected planes at
100 keV. It is a useful single-line selectivity proxy, but it is not an
absolute line-flux ratio because the bremsstrahlung denominator varies with
photon energy.

Validation: `zhai-material-screen-reconstruction`

| Material | Best plane | Ratio at 30 keV | Ratio at 100 keV | Ratio at 150 keV | Best/second at 100 keV | Registry status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| V2O5 | (010) | 9.665 | 19.289 | 24.919 | 8.59 | Missing |
| TiS2 | (003) | 7.010 | 13.428 | 17.376 | 47.27 | Missing |
| MoSe2 | (002) | 6.451 | 12.039 | 13.010 | 37.35 | Present |
| FeNb3S6 | (002) | 3.906 | 8.299 | 10.534 | 6.30 | Missing |
| WSe2 | (002) | 4.189 | 7.803 | 9.593 | 21.50 | Present |
| NiNb3Se6 | (002) | 2.744 | 5.896 | 7.932 | 5.63 | Missing |
| graphite | (002) | 1.600 | 3.833 | 5.216 | 160.78 | Present as HOPG |
| h-BN | (002) | 0.916 | 3.815 | 5.238 | 38.79 | Present |
| MnPS3 | (010) | 0.522 | 1.219 | 1.477 | 6.14 | Missing |
| ZrS3 | (10-1)/(10-2) | 0.749 | 1.264 | 1.682 | 1.00 | Missing |
| MnNb3S6 | (100)/(001) | 0.525 | 1.039 | 1.339 | 1.05 | Missing |

The supplementary table lists CrPS4 twice. The two corresponding raw-data
blocks differ materially, so they cannot be assigned unambiguously without
clarification from the authors. They are excluded from the primary ranking.
The paper also uses formula orderings such as `Nb3FeS6`; the conventional
intercalate notation `FeNb3S6` or `Fe1/3NbS2` is used below where useful. Exact
phase identity and CIFs must be checked before implementation.

The source calculations used non-tilted crystals, an observation angle of
119 degrees, a solid angle of 0.066 sr, and integration over 10 eV around each
coherent peak. Consequently, the ratios are screening evidence rather than
predictions for cxr-mc's 90-degree geometries.

## Highest-priority candidates

### 1. V2O5: strongest intrinsic coherent-to-bremsstrahlung candidate

Orthorhombic V2O5 is the strongest missing material in the published screen.
Its (010) ratio is approximately 9.7 at 30 keV, 19.3 at 100 keV, and 30.6 at
300 keV. The leading score remains about nine times larger than the runner-up
near 100 keV.

Using \(d_{010}\simeq 4.369\) angstrom and the magnitude of the simple
beam-parallel-to-\(g\), 90-degree relation
\(E_\gamma \simeq \hbar c\,\beta |g|\), the line is roughly:

| Electron energy | Nominal (010) photon energy |
| ---: | ---: |
| 30 keV | 0.932 keV |
| 100 keV | 1.556 keV |
| 150 keV | 1.800 keV |

The units follow from \(\hbar c\) in eV angstrom and \(|g|=2\pi/d\) in
inverse angstrom; the line energy tends to zero as \(\beta\) tends to zero.
These estimates reuse Validation: `line-energy-dispersion`. That ledger claim
currently has `discrepancy` status because the reciprocal-harmonic sign
convention is unresolved. The beam-aligned energy magnitudes used here are
still useful screening values, but are not a new validation of the production
dispersion convention.

Advantages include moderate effective atomic number, a layered structure, a
strong low-order plane, and demonstrated growth of millimetre- to
centimetre-scale crystals. The main concerns are:

- high-quality single crystals are not routine HQ Graphene stock;
- V2O5 can reduce toward lower oxides under vacuum annealing;
- an insulating or poorly conducting specimen may charge under the beam;
- published reciprocal-space maps show a small but nonzero angular
  distribution and possible substructure; and
- its absolute coherent photon flux was not reported in the material screen.

V2O5 is therefore the highest-priority custom-growth or collaborator target,
but not yet the best-supported off-the-shelf purchase.

### 2. 1T-TiS2: best balanced and immediately procurable candidate

TiS2 is the clearest omission from the existing dichalcogenide set. Its (003)
ratio exceeds those of MoSe2 and WSe2 at the sampled energies, and its
best-to-second-plane score is approximately 47 at 100 keV. This makes it the
strongest direct candidate for a spectrum dominated by one coherent line.

With \(c=5.70\) angstrom and \(d_{003}=1.90\) angstrom, the nominal (003) line
is:

| Electron energy | Nominal (003) photon energy |
| ---: | ---: |
| 30 keV | 2.143 keV |
| 100 keV | 3.577 keV |
| 150 keV | 4.139 keV |

HQ Graphene advertises synthetic 1T-TiS2 crystals with 6--8 mm lateral size,
greater than 99.995% purity, and single-crystal XRD showing the (001)--(004)
family. This is sufficient to make TiS2 the recommended first procurement.
The vendor data do not provide a high-resolution rocking-curve width.

Sulfur K emission near 2.31 keV may overlap the coherent line at some lower
beam energies. At higher beam energy the (003) line moves away from this fixed
characteristic line, which should improve the coherent-to-total-background
ratio even if the coherent-to-bremsstrahlung ratio alone is already favorable.

### 3. Fe1/3NbS2 and Ni1/3NbSe2: promising ordered intercalates

The paper-labeled FeNb3S6 and NiNb3Se6 compounds have substantially better
ratios than their Mn analogue. Their strongest (002) scores at 100 keV are
approximately 8.3 and 5.9, with leading-to-runner-up separations of roughly
6.3 and 5.6.

These compounds are ordered intercalates of layered niobium dichalcogenides.
Their approximately 12 angstrom repeat along \(c\) places the (002) line in
the soft-X-ray range: about 0.7 keV at 30 keV and 1.1 keV at 100 keV for a
representative \(d_{002}\simeq 6\) angstrom.

They are research targets rather than immediate purchases because:

- large, compositionally uniform crystals usually require custom chemical
  vapor transport growth;
- intercalant ordering, vacancies, and polytype selection can add satellite
  lines or broaden the nominal reflection;
- the exact formula and structure used in the published simulation need to be
  recovered from the authors or an unambiguous CIF; and
- characteristic emission and self-absorption have not been evaluated for
  the proposed line window.

MnNb3S6 is not comparably attractive: several planes have nearly equal scores,
which conflicts with the desired single-line spectrum.

## High-priority materials not included in the published screen

The following candidates are extrapolations from crystal structure,
composition, and current availability. They should be modeled before purchase
unless a particularly well-characterized sample is available.

### Alpha-MoO3

Layered orthorhombic MoO3 is the closest readily purchasable analogue to V2O5.
HQ Graphene lists synthetic 2--5 mm crystals with greater than 99.995% purity.
Its layered oxide structure and strong basal reflections are attractive. Mo
should increase diffraction strength and possibly absolute flux relative to
V2O5, but also increases bremsstrahlung, characteristic emission, and
self-absorption. MoO3 should be the first uncalculated oxide added to the
screen.

### Beta-GaS

Beta-GaS is a particularly interesting low-effective-\(Z\) candidate. HQ
Graphene lists 8 mm, greater than 99.995% synthetic crystals with a hexagonal
layered cell, \(a=3.60\) angstrom and \(c=15.53\) angstrom. Its likely long
electron mean free path favors the coherent-to-bremsstrahlung ratio, while
the large repeat distance puts useful basal harmonics in the soft-X-ray
range. The main uncertainty is whether one allowed basal harmonic dominates
after the full basis and absorption are included.

### ZrS2 and SnS2

ZrS2 and SnS2 are simple structural analogues of 1T-TiS2 and should be screened
using the same (00l) family.

- ZrS2 is the ratio-oriented choice: replacing Se with S and avoiding Hf
  should increase electron mean free path and reduce self-absorption. Its
  practical drawbacks are availability and possible oxygen contamination.
- SnS2 is the availability- and flux-oriented choice. HQ Graphene lists 8 mm,
  greater than 99.995% crystals with \(c=5.89\) angstrom. The nominal (003)
  line is about 2.07 keV at 30 keV and 3.46 keV at 100 keV. Sn increases
  coherent scattering strength but also the incoherent background.

### TiSe2

TiSe2 is the natural flux-forward partner to TiS2. HQ Graphene lists 8 mm,
greater than 99.995% 1T crystals with the (001)--(004) family observed in
single-crystal XRD. Selenium should raise absolute coherent strength but also
bremsstrahlung, characteristic emission, and self-absorption. The material
also undergoes a charge-density-wave transition below roughly 200 K, so room-
temperature and cryogenic structures should not be mixed.

### Lower-priority exploratory candidates

GaSe and InSe remain plausible because large high-purity crystals and strong
basal texture are commercially available. Their longer, polytype-dependent
unit cells introduce more allowed harmonics and a greater risk of stacking
disorder than TiS2 or GaS. In2Se3 should be deferred until the vendor supplies
an unambiguous phase-specific CIF; product metadata for 2H and 3R material are
not interchangeable.

Black phosphorus has favorable low atomic number but is air sensitive and has
strong phosphorus characteristic emission in the same broad soft-X-ray region.
Heavy Re-, Ta-, Pt-, Bi-, and Te-rich layered compounds may produce high
absolute flux, but their short electron mean free paths, greater
bremsstrahlung, stronger self-absorption, and dense characteristic spectra
make them poor first choices for coherent purity.

## Materials deprioritized by the direct screen

- **MnPS3:** the leading line is reasonably isolated, but the ratio is only
  about 1.2 at 100 keV.
- **ZrS3:** its two strongest planes have effectively equal scores, so it
  fails the single-line criterion even before linewidth effects.
- **MnNb3S6:** several planes are comparable and the ratio is only about 1.0
  at 100 keV.
- **CrPS4:** the supplementary table and deposited ordering are ambiguous
  because CrPS4 appears twice. The two possible blocks reach only about 0.9
  or 1.4 at 100 keV, so neither assignment challenges the leading candidates.
- **PtS2 and other very heavy dichalcogenides:** potentially useful for
  absolute flux, but unfavorable for coherent purity and interaction length.

## Linewidth and crystal-quality requirements

A van der Waals structure does not guarantee a narrow measured line. HOPG is
the counterexample: it is layered and easily available, yet its mosaic spread
can dominate the line width. The experimental width can include contributions
from:

- rocking-curve or mosaic spread;
- stacking faults, twins, and intercalant disorder;
- beam divergence and energy spread;
- finite detector solid angle;
- depth-dependent electron scattering and energy loss; and
- detector energy response.

Vendor purity, Raman spectra, EDX, and an ordinary theta--2-theta scan do not
establish sufficiently low mosaicity. Before procurement, request:

1. a high-resolution rocking curve for the actual target reflection;
2. the instrumental broadening used in quoting its FWHM;
3. a Laue, EBSD, or equivalent orientation map over at least the beam
   footprint;
4. evidence for absence of stacking twins or intergrowths;
5. phase-specific CIF and refined lattice parameters;
6. thickness and lateral-size maps; and
7. vacuum and electron-beam stability information for oxides and air-sensitive
   compounds.

## Recommended action sequence

### Procurement

1. **Purchase or request a qualified 1T-TiS2 sample first.** It has the best
   combination of direct quantitative evidence, line isolation, and current
   commercial availability.
2. **Pursue high-quality V2O5(010) through a grower or collaborator.** It is
   the strongest coherent-to-bremsstrahlung candidate but requires more sample
   qualification.
3. **Treat Fe1/3NbS2 and Ni1/3NbSe2 as custom-growth research targets.** Do not
   model or order them until the exact simulated structures are identified.

### cxr-mc screening

Add and evaluate, in order:

1. TiS2 and V2O5;
2. MoO3, GaS, ZrS2, and SnS2;
3. TiSe2;
4. the ordered Fe and Ni niobium intercalates after structural clarification.

For every material, calculate PXR and CBS rankings separately over at least
10--150 keV, scan all experimentally accessible orientations rather than only
the basal family, and retain both absolute coherent flux and the following
background-aware score:

\[
R_{\mathrm{total}} =
\frac{N_{\mathrm{PXR\ or\ CBS}}}
{N_{\mathrm{brems}} + N_{\mathrm{characteristic}}}
\quad\text{in the actual line window}.
\]

The final multi-objective ranking should report:

- absolute PXR and CBS line photons per electron or per beam current;
- coherent-to-bremsstrahlung and coherent-to-total-background ratios;
- leading-line to runner-up-line flux ratio;
- intrinsic and mosaic-broadened FWHM;
- absorption and escape fraction;
- saturation thickness and electron energy dependence; and
- uncertainty from crystal phase, lattice parameters, and mosaicity.

## Sources

1. Q. Zhai et al., “Enhanced tunable X-rays from bulk crystals driven by
   table-top free electron energies,” *Nature Communications* **16**, 11218
   (2025), [doi:10.1038/s41467-025-66063-6](https://www.nature.com/articles/s41467-025-66063-6).
2. Q. Zhai et al., deposited figure data for the same work,
   [doi:10.21979/N9/WZAMZ0](https://doi.org/10.21979/N9/WZAMZ0).
3. HQ Graphene, [catalog of available 2D crystals](https://www.hqgraphene.com/All-2Dcrystals.php)
   and product data for [TiS2](https://www.hqgraphene.com/TiS2.php),
   [MoO3](https://www.hqgraphene.com/MoO3.php),
   [GaS](https://www.hqgraphene.com/GaS.php),
   [SnS2](https://www.hqgraphene.com/SnS2.php), and
   [TiSe2](https://www.hqgraphene.com/TiSe2.php).
4. K. Anzenhofer et al., “The crystal structure and magnetic susceptibilities
   of MnNb3S6, FeNb3S6, CoNb3S6 and NiNb3S6,” *Journal of Physics and
   Chemistry of Solids* **31**, 1057--1067 (1970),
   [article record](https://www.sciencedirect.com/science/article/pii/002236977090315X).
5. J. Contreras et al., “An attempt to synthesize MnNb2S4 and the structural
   characterization of MnNb3S6,” *Revista Latinoamericana de Metalurgia y
   Materiales* **29**, 32--37 (2009),
   [open article](https://www.rlmm.org/ojs/archivos/29%281%29/RLMM%20Art-09V29N1-p32.pdf).
6. “Optical anisotropy of pristine and reduced V2O5(010),” *Scientific
   Reports* (2025),
   [article](https://www.nature.com/articles/s41598-025-07519-z).
7. Bulk V2O5 single-crystal growth and characterization,
   [open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9653758/).
