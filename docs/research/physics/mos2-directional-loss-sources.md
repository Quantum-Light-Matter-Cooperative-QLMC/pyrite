# 2H-MoS2 directional electronic-loss: input sourcing

Sourcing note for [#536](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/536) (parent survey #525). It states what a directional electronic-loss kernel would consume for 2H-MoS2, what the literature reachable today supplies, and which papers the maintainer must still obtain. It admits no value into production data and proposes no oscillator strengths. Quoted numbers are what the cited source reports, not PyRITE inputs. Status date: 2026-10-16.

## Bottom line

Updated after the maintainer added Zeppenfeld 1970, Yue 2017, Beal-Hughes 1979, Nerl 2017, Kumar-Ahluwalia 2014 and the full Moynihan 2020 PDF to Zotero.

* **A directional pair $\epsilon_\perp(\omega)$, $\epsilon_\parallel(\omega)$ for bulk 2H-MoS2 exists in the sources in hand, but only as published figures over 0-30 eV.** Zeppenfeld 1970, Fig. 2, gives both complex components from 60 keV EELS by Kramers-Kronig (KK) analysis. No numbers are tabulated; digitizing is required.
* **It does not cover 0-50 eV for $\epsilon_\parallel$.** Zeppenfeld stops at 30 eV. Bell-Liang cover $\epsilon_\perp$ to 50 eV, but no source in hand gives $\epsilon_\parallel$ over 30-50 eV, where the Mo 4p edges (40-50 eV) sit.
* **The normalisation is partly assumed.** $\epsilon_{1\perp}(0)=14.9$ is taken from Landolt-Bornstein. $\epsilon_{1\parallel}(0)=9$ is **assumed** by Zeppenfeld as giving "optimal consistency", and the author marks the $\epsilon_\parallel$ curves below about 3 eV as uncertain (dashed). No $\epsilon_\parallel$ sum-rule integral is reported; the f-sum is evaluated from $\epsilon_{2\perp}$ only.
* **Cross-checks.** Yue 2017 (IXS plus TD-LDA) independently supports the qualitative anisotropy: the 8.6 eV mode is absent for $Q\parallel c$ and the 23 eV mode is isotropic. Its data are uncalibrated intensities, not an ELF with a sum rule. Bell-Liang and Beal-Hughes cross-check $\epsilon_\perp$ only.
* **Verdict.** MoS2 can be built into a *labelled, approximate* uniaxial model over 0-30 eV from sources in hand after digitizing, with an explicit, unvalidated extension over 30-50 eV and an assumed $\epsilon_\parallel(0)$. It cannot be built as a measured 0-50 eV directional dataset. See {ref}`sec-mos2-maintainer-list` for what remains.
* hBN (Tarrio and Schnatterly 1989) remains the only material with tabulated directional parameters; it is the prototype for the model, not a substitute for MoS2 input.

## What the kernel consumes

Read from `src/pyrite/data/conduction_band.toml`, `montecarlo/transport/shell_oscillators.py`, and `montecarlo/channeling/operators.py` at `main` 5072fc6a. ADR-0018 fixes only the Lindblad-jump architecture and says nothing about the valence response. No dedicated channeling-materials page exists; the closest is [Channeling radiation physics](channeling-radiation-physics.md).

* **Catalog row.** `[mos2]` holds `electrons_per_formula = 18`, one `resonance_eV = 23.0`, and a Moynihan 2020 basis string. The 8.6 eV pi plasmon is stated as not represented. `ConductionBand` carries exactly these scalars.
* **Oscillator set.** `build_shell_oscillators` makes one conduction-band oscillator ($U=0$, $f_{cb}=18$, $W_{cb}=23$ eV) plus whole bound shells (PENELOPE-2024 Eqs. 3.62-3.64). The measured band must consume whole outermost shells; a split shell raises. For MoS2, 18 is the chemical valence count (Mo 4d5 5s1 plus 2 S 3s2 3p4).
* **Kernel.** `electronic_channels` turns each oscillator $(f_k, U_k, W_k)$ into a distant channel (strength, $q_{min}$, $q_{max}$) and a close Moller channel. All are functions of the scalar $|q_\perp|$ only. `electronic_kernel` integrates $w(q)$ over the transverse plane with $q=\sqrt{p^2+q_y^2}$ and a local-density factor $\sqrt{\rho_q}$. There is no direction dependence anywhere: the response is radial, scalar, and shared by all planes of a crystal.
* **Consequence.** A directional kernel needs $w(\mathbf q_\perp)$, not $w(|q_\perp|)$, so the oscillator set must become direction-resolved and the angle between $\mathbf q_\perp$ and $\hat c$ must enter the integrand.

### Frame for MoS2 planes

The beam lies in the channeling plane, so $q_z=\omega/v$ is negligible against $q_\perp$ at the loss energies of interest. This gives the angle $\alpha$ between $\mathbf q_\perp$ and $\hat c$:

```{list-table} Angle between the transverse momentum transfer and the c axis, for beam in the channeling plane.
:name: tbl-mos2-frame
:header-rows: 1

* - Plane
  - Beam
  - $c$ in transverse plane
  - $\cos\alpha$
* - basal (001)
  - perpendicular to $c$
  - along plane normal $x$
  - $q_x/\lvert q_\perp\rvert$
* - prismatic, beam perpendicular to $c$
  - perpendicular to $c$
  - along in-plane transverse $y$
  - $q_y/\lvert q_\perp\rvert$
* - prismatic, beam along $c$ (axial)
  - parallel to $c$
  - along beam, so $\alpha=90^\circ$
  - 0 (pure in-plane response)
```

The basal case is the hard one: the plane-normal coordinate carries the E-parallel-c response, so the dominant transfers that set channeling decoherence ($q_x$ near multiples of $g$) are the ones that probe $\epsilon_\parallel$.

### What a uniaxial model needs

The candidate form is the Tarrio-Schnatterly Sec. II relation

```{math}
:label: eq-mos2-uniaxial

\epsilon(\hat q,\omega)=\epsilon_\perp(\omega)\sin^2\alpha+\epsilon_\parallel(\omega)\cos^2\alpha ,
\qquad
\mathrm{ELF}(\hat q,\omega)=-\operatorname{Im}\frac{1}{\epsilon(\hat q,\omega)} .
```

The loss function of the mixed $\epsilon$ is not a linear mix of the two losses, so the required inputs are complex $\epsilon_\perp(\omega)$ and $\epsilon_\parallel(\omega)$ individually, not two loss peaks:

1. $\epsilon_{\perp}$ and $\epsilon_{\parallel}$ (or two ELFs plus $\epsilon_1(0)$ for each, so KK recovery is possible) from the valence range up to the first core edge. For MoS2 the Mo 4p edges sit at 40-50 eV (Habenicht 2015, Fig. 2), so the relevant range is about 0-40 eV.
2. A representation as delta oscillators per direction: resonance energies $W_i$, strengths $f_i$, widths if used, with $\sum_i f_i^{\perp}=\sum_i f_i^{\parallel}=18$ (the total f-sum is direction independent) and the logarithmic (Bethe) sum $\sum f_i\ln W_i$ consistent with the catalog mean excitation energy. Whole-shell partition rules from `build_shell_oscillators` still apply.
3. Normalisation provenance per direction: whether the sum rule is measured, imposed, or used to fix the scale ($\epsilon_1(0)$).
4. Geometry, $q$ range, temperature, and polytype, since the kernel is evaluated at transfers up to the first reciprocal-lattice multiples.

## Sources examined

All full-text reads are listed with the exact section, table, or figure used. "Mixed" means the source does not isolate a polarization.

```{list-table} 2H-MoS2 and related sources for directional loss data.
:name: tbl-mos2-sources
:header-rows: 1

* - Source
  - Access and read
  - Geometry and range
  - Normalisation and tabulated parameters
  - Use
* - Bell and Liang, Adv. Phys. 25, 53 (1976), doi:10.1080/00018737600101362 (Zotero `D2EF5UG7`)
  - Full text (PDF `YGCBVDRV`): abstract, Secs. 3.1-3.2, 4.1, 5.3, Tables 1 and 4, Fig. 6 caption, reference list
  - 50 keV along $c$, scattering angle 1.0 +/- 0.1 mrad, so $q\approx0.12$ Å$^{-1}$ (derived from $k_0\theta$), $\hat q$ within 75-90 degrees of the basal plane; 0-50 eV; natural molybdenite, 50-100 nm flakes, 2H. Temperature not stated in the passages read.
  - $\epsilon_1(0)=15.0$ (Table 1, from interference measurements and Zeppenfeld 1970) imposed as the KK anchor; high-energy tail fitted above 50 eV; quoted loss-function error up to 30% near the double-loss peak. Table 4, MoS2: $\omega_L=8.7$ eV, $\omega_p=14.4$ eV, $n=8$, $\omega_T=4.7$ eV; and $\omega_L=23.1$ eV, $\omega_p=21.6$ eV, $n=18$, sub-bands 4.7 eV (8 e) plus 13.0 eV (10 e), $\epsilon_0(\omega_L)=1.1$. $\omega_T$ values are +/-10%.
  - E-perp-c only. The only quantitative tabulated oscillator description for MoS2. Not a final strength set: a 2-term centre-of-gravity model, estimated from the JDOS.
* - Same, 0.3 mrad zero-deflection spectra (Sec. 3.2)
  - same
  - Mixed polarization, finite acceptance; zero-loss peak intrudes
  - none
  - Described by the authors as qualitative information on E-parallel-c. Not usable.
* - Habenicht, Knupfer, Buechner, PRB 91, 245203 (2015), doi:10.1103/PhysRevB.91.245203 (arXiv:1505.06401)
  - Full text read (arXiv): Sec. II and Fig. 2
  - 172 keV, 20 K, $q=0.1$ Å$^{-1}$ along GM and GK (in-plane), 0-70 eV, 2H film; $\Delta E$ 82 meV
  - Loss functions after elastic and multiple-scattering removal; no KK normalisation in the sum-rule sense for the wide range. Peaks 8.8 and 23.3 eV, Mo 4p at 40-50 eV.
  - E-perp-c cross-check at low $q$ and 20 K for Bell-Liang. Figure only, so extraction requires digitizing.
* - Habenicht et al., J. Phys.: Condens. Matter 30, 205502 (2018), doi:10.1088/1361-648X/aabd02 (arXiv:1805.01826)
  - Abstract and header read
  - In-plane (GK, GM) indirect excitons
  - none relevant
  - Not directional; low-energy only.
* - Hong et al., PRB 93, 075440 (2016), doi:10.1103/PhysRevB.93.075440 (arXiv:1602.02682)
  - Abstract and geometry passages read
  - 60 kV, convergence 0.3 mrad, angle-resolved low loss (up to about 5 eV), mono- to few-layer. Separates $q\perp c$ and $q\parallel c$ components.
  - Spectra normalised only by peak ratios; no sum rule. Gives out-of-plane gap 2.4 +/- 0.2 eV in a monolayer.
  - Qualitative proof that out-of-plane response is observable. Not bulk, not above 5 eV.
* - Ermolaev et al., Nat. Commun. 12, 854 (2021), doi:10.1038/s41467-021-21139-x (arXiv:2006.00884)
  - Full text read (arXiv): ellipsometry range and methods
  - Bulk MoS2 flakes, ordinary and extraordinary $\epsilon$ by imaging spectroscopic ellipsometry, 360-1700 nm (about 0.73-3.4 eV)
  - Fitted Tauc-Lorentz and Lorentz models in its SI (not read)
  - Open-access $\epsilon_\parallel$ below 3.4 eV only. Covers the excitonic and band-edge region, not the 8 and 23 eV region that controls the stopping sum.
* - Moynihan et al., J. Microsc. 279, 256 (2020), doi:10.1111/jmi.12900 (Zotero `IDUU3SWI`, arXiv:2012.09924)
  - Read in the prior pass (issue comment); see also the row below
  - 60 kV STEM, 31/44 mrad, mono- to few-layer
  - Authors call their KK normalisation unreliable
  - Qualitative; the catalog row's source. Not an input source for directional data.
* - Yue et al., PRB 96, 125118 (2017), doi:10.1103/PhysRevB.96.125118 (Zotero `E93NDTV2`)
  - Full text (PDF `P5XGAU27`): abstract, Secs. II-IV, Figs. 2 and 4. Supplement (Table S1, Figs. S1-S6) not in hand.
  - IXS at 13.843 keV, resolution 2 eV, $\Delta Q\approx0.25$ Å$^{-1}$, room temperature, +2 to +50 eV. Eight spectra each along K, M and A ($A\parallel c$), $Q=0.184$-2.65 Å$^{-1}$. TD-LDA ELF calculations for K, M, A (Fig. 4a-c) and $\epsilon(Q,\omega)$ with and without local-field effects for K and A (Fig. 4d,e).
  - Spectra normalised to incident intensity with a constant background removed; no absolute scale, no sum rule. Voigt fits in Table S1 (supplement). Calculated plasmon fit $\hbar\omega_p(0)=23.0$ eV, $\alpha=0.29$ (free-electron gas: 21.6 eV, 0.49); the experimental $\alpha$ is stated to be significantly smaller.
  - The 23 eV mode is isotropic along K, M and A in experiment and calculation, with the calculated intensity "a little higher" along A below 0.5 Å$^{-1}$. The 8.6 eV mode is weakly dispersive below 0.9 Å$^{-1}$ in plane and is hard to observe along $c$ (absent in the calculation along A). Supports Zeppenfeld qualitatively; gives the only finite-$Q$ data. Digitizing Fig. 4 would give an ab initio $\epsilon_\parallel$-direction ELF to 50 eV, as a calculation, not a measurement.
* - Beal and Hughes, J. Phys. C 12, 881 (1979), doi:10.1088/0022-3719/12/5/017 (Zotero `HYBN58ZH`)
  - Full text (PDF `DGNVN8QC`): Secs. 1-2, discussion of Figs. 1-4; no table of optical constants found
  - E-perp-c normal-incidence reflectance of basal faces, room temperature, measured 1.0-1.3 eV to 14 eV; **14-30 eV is taken from the Bell-Liang EELS-derived reflectance**; above 30 eV $R\propto E^{-4}$; extrapolated to 0 eV from absorption coefficients.
  - $\epsilon_1(0)\approx15$-16 (extrapolated); $n_{eff}$ shows a weak saturation near 8 electrons at the 8.81 eV partial plasmon and approaches 18 above the 20 eV plasmon (Fig. 2(d), MoS2 panel).
  - Independent of Bell-Liang for $\epsilon_\perp$ only in 1-14 eV. Above 14 eV it is **not independent**. Figures only, so digitizing is required. E-perp-c only.
* - Schuster et al., PRB 94, 085201 (2016), doi:10.1103/PhysRevB.94.085201
  - **Blocked: HTTP 403.** Title via Crossref: "Nongeneric dispersion of excitons in the bulk of WSe2"
  - WSe2 excitons
  - n/a
  - **Not a MoS2 source.** The #536 comment lists it as a pristine WSe2/MoS2 candidate; it is WSe2 only and low energy.
* - Zeppenfeld, Opt. Commun. 1, 377 (1970) (Zotero `899KPSPC`; printed pages 377-378; the Bell-Liang reference list cites p. 337)
  - Full text (PDF `VUGI9WE4`, 2 pages, rendered): Sec. 2, Figs. 1 and 2, references
  - 60 keV electrons, scattering angle 0.8 mrad ($q\approx0.10$ Å$^{-1}$, derived from $k_0\theta$), $q$ kept near perpendicular to the beam; $\hat q$ varied against $c$ by selecting different azimuthal angles. Natural and synthetic samples agree. Energy axis 0-30 eV. Optical limit assumed, not tested for MoS2 (checked for graphite up to 2 mrad).
  - KK analysis after Bassani-Tosatti. Real part anchored by $\epsilon_1^\perp(0)=14.9$ (Landolt-Bornstein) and **assumed** $\epsilon_1^\parallel(0)=9$ ("optimal consistency"); low-energy parts below about 3 eV are drawn dashed as uncertain. Peaks at 8.75 eV (6 electrons) and about 23 eV (18 electrons) from the f-sum evaluated on $\epsilon_2^\perp$. For $q\parallel c$ the 23 eV peak is "nearly unaffected", the 8.75 eV plasmon vanishes, and a new maximum appears near 15 eV, assigned to S 3s/3p bands.
  - Fig. 1: $-\operatorname{Im}1/\epsilon$ for both directions. Fig. 2: $\epsilon_1,\epsilon_2$ (with $\times10$ high-energy insets) for $E\perp c$ and $E\parallel c$. No tables. This is the only measured $\epsilon_\parallel$ over the valence range in hand. Figures only.
* - Kumar and Ahluwalia, chapter 3 in *MoS2: Materials, Physics, and Devices* (Springer, 2014), doi:10.1007/978-3-319-02850-7_3 (Zotero `CN5QZLL7`)
  - Full text (PDF `F2S8VKPG`): Sec. 3.2.2 and Fig. 3.3
  - Ab initio (not measured) $-\operatorname{Im}1/\epsilon$ for $E\perp c$ and $E\parallel c$, bulk to monolayer, 0-30 eV, reproduced from Kumar and Ahluwalia, Physica B 407, 4627 (2012). The chapter states no method details for it.
  - No normalisation or sum rule stated; Fig. 3.3 only, no table.
  - Not tabulated, not measured. The original Physica B paper (not in hand) would be needed for the code, functional and broadening. Calculation-only cross-check.
* - Nerl et al., npj 2D Mater. Appl. 1, 2 (2017), doi:10.1038/s41699-017-0003-9 (Zotero `YIRV5PIU`, arXiv:1702.00359)
  - Full text (PDF `RHDUYLTP`): grep-level read of Sec. 3 and Fig. 5
  - STEM 34 mrad convergence, 44 mrad collection, few-layer terraces; TDDFT-RPA with in-plane and out-of-plane $q$ mixed by the beam geometry
  - none
  - Not usable: mixed $q$ directions, few-layer, no tensor components.
* - Moynihan et al. (full PDF `E8QJ2U38`, item `JSULQQ6R`)
  - Full text: Fig. 4 and the KK-thickness discussion
  - DFT-RPA bulk $\epsilon$ (Fig. 4a) with zero crossings of $\epsilon_1$ near 8 and 23 eV; polarization direction not stated in the passages read; monolayer KK result depends strongly on the thickness used for normalisation
  - Normalisation unreliable (authors)
  - Not a tensor source. Fig. 4a is an uncharacterised bulk DFT $\epsilon$.
```

The Ahmad 2017 (`XNZTLK4M`) and Taft-Philipp 1965 (`RMIJPTUW`) findings of the prior pass concern WSe2 and graphite and are not repeated here.

## Conclusions per quantity

```{list-table} MoS2 inputs by availability, after the second source pass.
:name: tbl-mos2-conclusions
:header-rows: 1

* - Quantity
  - Obtainable from sources in hand?
  - Where
  - Normalisation and digitizing
* - $\epsilon_\perp(\omega)$, 0-30 eV
  - Yes, after digitizing (a)
  - Zeppenfeld Fig. 2 (upper); Bell-Liang Figs. 2(a) and 3; Beal-Hughes Figs. 1-2 (1-14 eV direct)
  - Anchored by $\epsilon_1(0)=14.9$ (Zeppenfeld), 15.0 (Bell-Liang), 15-16 extrapolated (Beal-Hughes). Three consistent KK reconstructions, but not three independent datasets: Beal-Hughes takes 14-30 eV from Bell-Liang.
* - $\epsilon_\perp(\omega)$, 30-50 eV
  - Yes for the loss function and $\epsilon$, after digitizing (a)
  - Bell-Liang only (0-50 eV; fitted $\omega^{-3}$ tail above 50 eV)
  - Error up to 30% near the double-loss peak; Mo 4p edges at 40-50 eV inside the window.
* - $\epsilon_\parallel(\omega)$, 3-30 eV
  - Yes, after digitizing (a), with caveats
  - Zeppenfeld Fig. 2 (lower) and Fig. 1 (lower)
  - $\epsilon_1^\parallel(0)=9$ **assumed**, not measured. Pure $q\parallel c$ purity is not quantified (0.8 mrad, azimuthal selection, mixed polarization for finite acceptance). Optical limit assumed.
* - $\epsilon_\parallel(\omega)$, 0-3 eV
  - Partly (a)
  - Ermolaev 2021 (open access, 0.73-3.4 eV); Zeppenfeld dashed region
  - Ermolaev's SI (c-axis Cauchy fit) gives about 6.1 against the assumed 9; kept at 9 with a bounded systematic, see {ref}`sec-mos2-eps1par-decision`.
* - $\epsilon_\parallel(\omega)$, 30-50 eV
  - **No (b)**
  - None measured. Yue Fig. 4(c,e) gives a TD-LDA calculation along A (to 50 eV)
  - Calculation only, no experimental anchor; a high-energy tail would be an explicit assumption, not data.
* - Per-direction f-sum and Bethe-sum
  - $\perp$ partially; $\parallel$ **no**
  - $n_{eff}\to18$: Bell-Liang Fig. 6(a), Table 4; Beal-Hughes Fig. 2(d); Zeppenfeld (text)
  - No $\epsilon_\parallel$ integral is reported anywhere in hand. It must be computed from the digitized $\epsilon_{2\parallel}$ and will be incomplete above 30 eV.
* - $\pi$ electron count (6 or 8)
  - **Unresolved**
  - Zeppenfeld, Yue and Kumar say 6 electrons at 8.75 eV; Bell-Liang and Beal-Hughes show $n_{eff}\approx8$ at the partial plasmon
  - Relevant to any $\pi$/total-valence partition (issue constraint). Not resolved by the sources.
* - Angle dependence $\epsilon(\hat q)$ (Eq. {eq}`eq-mos2-uniaxial`)
  - Plausible, not tested
  - Zeppenfeld measured only $q\perp c$ and $q\parallel c$; Tarrio-Schnatterly tested it for hBN
  - No MoS2 tilt-series; intermediate $\alpha$ untested.
* - $q$-dependence at channeling transfers (up to about 1 Å$^{-1}$)
  - Qualitative only
  - Yue Figs. 2-4 ($Q=0.18$-2.65 Å$^{-1}$, three directions); Habenicht 2015 ($q=0.1$ Å$^{-1}$)
  - IXS is uncalibrated, 2 eV and 0.25 Å$^{-1}$ resolution. The 23 eV mode is isotropic to about 2 Å$^{-1}$ (experiment and TD-LDA); 8.6 eV is in-plane only.
* - Temperature
  - Room (Zeppenfeld not stated; Yue and Beal-Hughes room); 20 K (Habenicht)
  - n/a
  - Thermal tensors and the layered TDS oracle are separate prerequisites.
```

(sec-mos2-maintainer-list)=
## Remaining for the maintainer

With the new papers in hand, the minimal missing set is small but not empty.

1. **No measured $\epsilon_\parallel$ above 30 eV exists in hand.** Decision needed: (i) accept a flagged extension (for example the $q\parallel c$ TD-LDA loss function of Yue Fig. 4, or a free-electron tail), which makes MoS2 an approximate, computed-above-30-eV model, or (ii) find a measurement. Nothing in the Zeppenfeld or Bell-Liang reference lists reaches $E\parallel c$ above 30 eV.
2. **Yue 2017 Supplementary Material** (Table S1 Voigt fits, Figs. S1-S6 including RPA calculations). Not in Zotero. Needed only if the finite-$Q$ dispersion or the RPA comparison is to be used.
3. **Landolt-Bornstein, Vol. II/8, p. 2-159 (Springer, 1962)** or another measured $\epsilon_1^\perp(0)$ source, and an independent $\epsilon_1^\parallel(0)$. Zeppenfeld assumes $\epsilon_1^\parallel(0)=9$. Candidates: Ermolaev 2021 SI (open access); Liang, J. Phys. C 6, 551 (1973) and 8, 1763 (1975), and Klucker-Skibowski-Steinmann, Phys. Status Solidi B 65, 703 (1974), which Bell and Liang cite for optical $E\parallel c$ work. Titles and relevance not checked.
4. **Kumar and Ahluwalia, Physica B 407, 4627 (2012)** (the original of Fig. 3.3 in the chapter), only if a calculated $\epsilon_\perp,\epsilon_\parallel$ pair with documented method is wanted as a cross-check. Not a measurement.
5. **Method reference (optional):** Bassani and Tosatti, Phys. Lett. A 27, 446 (1968) and Tosatti, Nuovo Cimento B 63, 54 (1969), for the anisotropic KK relation Zeppenfeld and Bell-Liang used.

Not needed any longer: Beal-Hughes (read; independent only 1-14 eV) and Yue's main text (read). Not usable: Nerl 2017, Moynihan 2020 (qualitative or mixed-$q$).

## Corrections to the earlier comment

* Schuster 2016 (PRB 94, 085201) is a WSe2 exciton-dispersion paper; it does not supply pristine MoS2 or pristine bulk WSe2 loss functions.
* Habenicht 2015 (PRB 91, 245203) is a MoS2 source, in-plane at low $q$.
* Zeppenfeld 1970 gives the E-parallel-c data the first pass could not find, but over 0-30 eV with an assumed $\epsilon_1^\parallel(0)$, so the earlier statement that no quantitative $\epsilon_\parallel$ source was reachable is superseded.

## Next slice

Decide the 30-50 eV treatment (item 1). Then digitize Zeppenfeld Figs. 1-2 and Bell-Liang Figs. 2-3 and 6(a) with stated uncertainty; fit per-direction oscillator sets with each $\epsilon_1(0)$ treated as a free, flagged anchor; check both sum rules and the 6-versus-8 electron partition; and test the uniaxial mixing against hBN (Tarrio-Schnatterly Figs. 2, 8, 9) before any MoS2 row is added. Until then MoS2 stays at its scalar in-plane catalog row and must be reported as not an anisotropic channeling anchor.


(sec-mos2-digitization-protocol)=
## Digitization protocol (#536, #541)

The directional model reads four digitized curves of Zeppenfeld Fig. 2 and checks them against the two loss functions of Fig. 1. Nothing in the code invents a value: with no registered dataset `electronic_kernel` keeps the scalar response and `load_directional_loss("mos2")` raises.

**Files.** Under `src/pyrite/data/directional_loss/mos2/`, one CSV per curve with header `energy_eV,value` and `#` provenance lines (source figure, axis calibration points, digitizer, estimated tolerance in eV and in value):

| File | Curve | Notes |
|---|---|---|
| `eps1_perp.csv`, `eps2_perp.csv` | Fig. 2 upper panel, `E ⊥ c` | joined branches below 15 eV and the ×10 branch above, already divided by 10 |
| `eps1_par.csv`, `eps2_par.csv` | Fig. 2 lower panel, `E ∥ c` | same; the 0–3 eV parts are dashed (uncertain, per the author) |
| `elf_perp.csv`, `elf_par.csv` (optional) | Fig. 1 `-Im 1/ε`, `q ⊥ c` (left axis) and `q ∥ c` (right axis) | the ×3 branches below 14 eV (⊥) and 17 eV (∥) divided by 3 |

**Reading the figure** (the agent's reading, from zooming on the 300 dpi rendering of the Zotero PDF `VUGI9WE4`; confirm against the plot before digitizing). In both Fig. 2 panels the curve starting at `ε₁(0)` (14.9 for ⊥, 9 for ∥) is **ε₁**, and the steep curve rising from the baseline near 1.6 eV (⊥) or 2 eV (∥) is **ε₂**. They cross near 2.2 eV (⊥, value about 24). The curve labelled ε₂ above 3 eV (shoulder 12.5 at 4 eV, 2.5 at 8.5 eV) is the continuation of that rising ε₂; the one labelled ε₁ continues the descent of ε₁. Sanity checks: `ε₂/(ε₁²+ε₂²)` at 8.8 eV must give the Fig. 1 peak (about 0.46 after the ×3 division) and ε₂ must be non-negative everywhere. Axis ticks, read from the rendering: Fig. 2 energy ticks at 0, 5, 10, 15, 20, 25, 30 eV; left ε ticks every 5 units (⊥ spans −10…30, ∥ −10…20).

**Registering.** Uncomment the `[mos2]` block in `src/pyrite/data/directional_loss.toml`, keep `status = "provisional"`, regenerate `CHANNELING_MODEL` (new suffix) in `montecarlo/channeling/handoff.py`, then run `tests/montecarlo/test_channeling_directional.py` (the MoS₂ tests stop skipping). Items 2–6 of #541 (ε₁∥(0), the 6-versus-8 π electron count, per-direction f-sum, the 30–50 eV extension, the intermediate-angle test) stay open and keep the warning.

**Chosen 30 eV treatment.** Above the last tabulated energy the loss follows the free-electron tail `ELF(ω) = Im ε(ω_max)(ω_max/ω)³`, the `ε → 1` asymptote matched to `Im ε` (maintainer decision, 2026-10-22). Matching `ELF(ω_max)` instead overcounts: at 31.5 eV ε₁ ≈ 0.55, so the ELF is still about 3 Im ε on the 23 eV plasmon shoulder, and that tail gave closure 1.19 (⊥) / 1.34 (∥). The tail share of the f-sum is reported per direction (`tail_fraction`) and each direction is renormalized to 18 electrons only when its raw closure is within the entry's `closure_tolerance`.

## Registered dataset (2026-10-22)

The maintainer digitized Zeppenfeld Fig. 2 (all four curves, main and ×10 branches). `src/pyrite/data/directional_loss/mos2/eps{1,2}_{perp,par}.csv` join each main branch (to 14-15 eV) with its ×10 branch divided by 10 (to 31.5 eV), sorted by energy; each `ε₂` file adds one figure-baseline zero at the first `ε₁` energy (about 0.59 eV). Branch assignment in the 2-3 eV crossing was checked against a 600 dpi rendering. Tool and axis calibration points were not recorded; estimated tolerance ±0.1 eV and ±0.3 (main) / ±0.03 (×10 branch, after division). Fig. 1 is digitized too (`elf_perp.csv`, `elf_par.csv`: main branches plus the ×3 branches below 14 eV (⊥) and 17 eV (∥) divided by 3; ±0.1 eV, ±0.05 main, ±0.02 ×3). Fig. 1 ⊥ is on the left axis, ∥ on the right.

```{list-table} Diagnostics of the registered MoS2 dataset (catalog Ω for 18 electrons = 21.61 eV).
:name: tbl-mos2-registered
:header-rows: 1

* - Quantity
  - ⊥ (`E ⊥ c`)
  - ∥ (`E ∥ c`)
* - ELF peaks (Zeppenfeld: 8.75 and 23 eV ⊥; 23 eV ∥, no π plasmon)
  - 8.75, 23.27 eV
  - 23.01 eV; ELF(8.5 eV) at 0.38 of the ⊥ π peak
* - `n_eff` from `ε₂` at 9 / 20 / 31.5 eV
  - 7.1 / 14.6 / 16.3
  - 5.4 / 15.0 / 16.8
* - `n_eff` from the ELF at 31.5 eV
  - 13.4
  - 13.2
* - Raw closure with tail (tail share)
  - 0.886 (0.16)
  - 0.929 (0.21)
* - Static sum vs `1 − 1/ε₁(0)`
  - 0.866 vs 0.933
  - 0.816 vs 0.889
```

The `ε₂` route gives the per-direction f-sum asked for by #541: both directions reach 16-17 of 18 electrons by 31.5 eV, the ∥ direction slightly more. `n_eff⊥` at the 8.75 eV π plasmon is 6.8-7.1, between the 6 (Zeppenfeld) and 8 (Bell-Liang, Beal-Hughes) of the partition question; the digitization tolerance does not resolve it. The static-sum shortfall (7-8%) is the part of `1 − 1/ε₁(0)` the tail cannot supply; it depends on the assumed `ε₁∥(0) = 9` for ∥.

**Fig. 1 against Fig. 2.** The ELF computed from the digitized Fig. 2 $\epsilon$ reproduces the digitized Fig. 1 over 3-30 eV within 1.2σ (⊥) and 1.4σ (∥), where σ propagates the stated reading errors through $\epsilon_2/|\epsilon|^2$ and adds the Fig. 1 reading error; no point exceeds 2σ (the anchor test). The largest absolute gap is at the 23 eV plasmon (∥: 3.52 in Fig. 1 against 3.19 from Fig. 2), where $\epsilon_1\approx0$ makes the ELF $\approx1/\epsilon_2$ with $\epsilon_2\approx0.28$, so a 0.03 reading error is already 10%. Fig. 1 peaks: ⊥ 8.73 eV (0.46) and 23.2 eV (3.34); ∥ 23.0 eV (3.52) with the π region at 0.17. Fig. 1 to 31.5 eV holds 13.7 (⊥) and 13.9 (∥) electrons on the loss f-sum, against 13.4 and 13.2 from Fig. 2.

(sec-mos2-eps1par-decision)=
## Decision: $\epsilon_1^\parallel(0)$ (2026-10-22)

**Kept at 9, as the anchor of the dataset, with Ermolaev's measured value recorded as a conflict.** Reasons and the bound:

* $\epsilon_1^\parallel(0)=9$ is not a separate input. It is the normalization from which Zeppenfeld built the Fig. 1 ∥ scale and the Fig. 2 ∥ curves by KK. Replacing it means re-deriving $\epsilon_\parallel$ ourselves from the digitized Fig. 1.
* That re-derivation was done as a sensitivity test (anisotropic KK of `elf_par.csv`, zero below the 1.84 eV onset, the Im-ε-matched ω⁻³ tail above 31.8 eV, scale fixed by the static sum). With 9 it reproduces the digitized Fig. 2 ∥ to $|\Delta\epsilon_1|\le1.3$ and $|\Delta\epsilon_2|\le1.7$ over 3-30 eV, and gives $\epsilon_1^\parallel(0.6\,\text{eV})=9.2$ against 9.3 digitized. That reconstruction error is larger than the effect of the anchor itself.
* With Ermolaev's 6.1 the ELF∥ scale changes by 6% (0.974 against 1.035), which the per-direction f-sum renormalization removes. The binned strengths change by at most 1.7% of N (L1 over bins, pure ∥; 0 for pure ⊥) and at most 5.7% in any bin above 1% of N. The static target moves from 0.889 to 0.836.
* Ermolaev's value is a measurement, but it comes from extrapolating a transparent-range Cauchy fit (800-1700 nm) to $\lambda\to\infty$. Hieu's 8.92 is GGA-PBE, which typically overestimates $\epsilon_\infty$. The two published values bracket the anchor and do not settle it.
* The discrepancy is real at 0.6 eV: Zeppenfeld's dashed ε₁∥ reads 9.3, while Ermolaev's fit gives 6.2 at the same wavelength. It is confined to the region the author already marks as uncertain (below 3 eV), which carries almost no f-sum weight.

Consequence: the dataset stays `provisional`. The $\epsilon_1^\parallel(0)$ item of #541 is closed as a decision with a quantified ≤2% (L1) systematic, not as support for the value 9.
