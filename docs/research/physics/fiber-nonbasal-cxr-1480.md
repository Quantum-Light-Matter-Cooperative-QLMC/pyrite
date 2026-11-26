# Non-basal CXR from fiber-textured HOPG and h-BN

**Status:** Phase I research note for [#1480](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1480), raised by the question in [#1476](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1476). It changes no production default. Calculations: `checks/fiber_nonbasal_cxr_1480.py`. Claims: `cxr-fiber-azimuthal-kinematics`, `graphite-nonbasal-structure-factor` (ledger: [core coherent physics](../../validation/ledger-core-coherent-physics.md)).

**Question.** The catalog marks HOPG and pyrolytic h-BN `texture = "fiber"`, and since #1301 (`reflection-yield-tolerance-selector`, v7) the reflection selector keeps only `(00l)` for them. The stated reason was that random in-plane crystallite azimuths remove the non-basal lines. Is that restriction an adequate CXR approximation, and where?

**Short answer (so far).** No, not in general. Random azimuths average non-basal intensity; they do not remove it. The intensity of one reflection is $\lvert A_g\rvert^2\ge0$ at every azimuth, so its average is positive. Phase averaging only removes interference between distinct reflections, and that interference is already zero for unshaped electrons (final-state orthogonality, see [Coherence](#coherence-what-averages-and-what-does-not)). Under the production per-reflection model, the non-basal families carry yield comparable to the basal set in most of the geometries measured below.

## Literature

- **Shi et al. 2024** (Nat. Commun. 15, 7803; Zotero `Q4I3JISU`, SI `6YMV48GX`). Eq. (6) is the exact recoil root used here. SI §S5 (WSe₂, graphite): "the dominant peaks are associated with $g_\perp=0$ … nonzero $g_\perp$ … leads to extra peaks, those additional peaks may not be distinguishable from the incoherent background in experiments." That is a statement about one fixed single-crystal orientation and one observation angle, with no numbers. It agrees with our finding that $(002)$ is the largest single family at backward angles. It does not bound the six-member, azimuth-summed families, and it does not address forward angles or oblique incidence.
- **#1476 correspondent (PXR/CBS researcher).** Non-basal contributions are "small for HOPG or hBN when looking at emissions from (002) or (004)", but "sometimes there's no dominant plane so we have to consider all of them". They recommend the $p_i=p_f+\hbar g+\hbar k$ convention, which production already uses. Our numbers are consistent with "small inside a narrow (002) window at backward angles", where recoil shifts the strong non-basal bands off the $(002)$ peak. They are not consistent with "small" for integrated or wide-window yields.
- **Textured polycrystals.** Eliseyev, Kubankin et al., "Parametric X-ray radiation from relativistic electrons interacting with a textured polycrystals", RREPS-15 (2015), measured nonzero, orientation-dependent PXR from a textured W foil (7 MeV, (200), 90°–180°). Their abstract calls PXR from polycrystals ("polarization bremsstrahlung", Nasonov and co-workers) well studied. Both support averaging of intensity, not its cancellation. The full paper and its fiber-texture model were not retrieved (bot check).
- **Mosaic pyrolytic graphite.** Adishev, Kaplin et al. (2003, 5.7 MeV, 0.3 mm mosaic PG) observed the $(002)$ Bragg-geometry PXR peak. Fiorito et al. (PAC 1993, NRL, CAPG) observed up to eight basal orders. Vnukov et al. (2001) show that diffraction of real photons in mosaic crystals contributes to measured spectra. These are basal-geometry measurements and do not test non-basal families.
- **Thermal factors.** Graphite: Trucano & Chen, Nature 258, 136 (1975), $U_{11}=0.0031$, $U_{33}=0.016/0.017$ Å² (packaged by #535). h-BN: Pease (1952) gives $00l$ only (Zotero `RV4Z2ZWJ`). Neither Zotero nor a web search found a sourced in-plane h-BN ADP. Per the repository Zotero rule, that is not proof that none exists.
- **Not found** (Zotero and web, this pass): a calculation or measurement of non-basal PXR/CBS from fiber-textured HOPG or h-BN; orientation-averaged CBS for layered materials beyond the isotropic average already used by `brem-lattice-coherent-fraction`.

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

**Thermal factors.** The catalog $B_{33}=1.30$ Å² is $8\pi^2U_\parallel$, which is valid for $g\parallel c$ only. For general $g$, $W=\tfrac12 g^\mathsf TUg=\tfrac12(U_\perp g_\perp^2+U_\parallel g_z^2)$. Graphite has $U_\perp=0.0031$, $U_\parallel=0.0165$ Å² (Trucano & Chen 1975, packaged for channeling by #535 in `channeling_thermal.toml`). Applying $B_{33}$ isotropically underestimates in-plane-heavy families: by 29 % for $(112)$ and 33 % for $(110)$ in intensity. The check passes the tensor to the scalar-$B$ kernel as $B_\text{eff}=8\pi^2(U_\perp g_\perp^2+U_\parallel g_z^2)/g^2$, which is exact for one uniaxial tensor shared by all sites. Graphite's two sites differ slightly in $U_{33}$ (0.016 and 0.017); the check uses their mean, as the catalog does. h-BN has no packaged in-plane ADP. Its numbers below use the catalog $B_{33}=1.73$ Å² isotropically, which underweights its non-basal families.

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
| HOPG | 30 keV | 0° | 22.5 | – | 0.75 | 3.6 |
| HOPG | 30 keV | 5° | 35.0 | – | 0.92 | 4.9 |
| HOPG | 100 keV | 0° | 28.1 | – | 0.84 | 2.1 |
| HOPG | 1 MeV | 0° | 49.8 | – | 0.96 | 0.12 |
| h-BN | 30 keV | 0° | 10.6 | – | 0.53 | 2.4 |
| h-BN | 1 MeV | 0° | 22.2 | – | 0.57 | 0.07 |

Observation angles are polar angles from the beam (+z = fiber axis). The 45° column is pending a rerun (the first run used 90°, which grazes the infinite slab and is fully absorbed). At forward angles the basal PXR is polarization-suppressed, so non-basal families dominate. At 119° the $(002)$ line remains the largest single family, consistent with Shi et al. SI §5 ("the dominant peaks are associated with $g_\perp=0$"), yet the six-member non-basal families together match it. The leading omitted families are $(112)$, $(101)$ (a new line at half the $(002)$ energy), $(103)$ and, with tilt, $(110)$.

## Frozen-transport arms

*Pending: matrix in progress.* Arms: A basal only (current model, catalog $B_{33}$); B full allowed domain at fixed $\phi=0$; S8/S32/S128 full domain with stratified uniform azimuths on the full circle, one per electron stratum; Q32 full domain on 32 sector nodes (reference). Smoke result (HOPG 30 keV, 500 nm, tilt 15°, lab 45°, 100 electrons): A/Q = 0.31; S8/Q = 1.00006; B/Q = 1.004.

## Domain coherence (open)

*Pending.*

## Applicability envelope and Phase II

*Pending.*
