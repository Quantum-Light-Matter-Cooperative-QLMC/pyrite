# Characteristic radiation

PyRITE models electron-impact characteristic X rays as direct atomic vacancies relaxed through a deterministic radiative and nonradiative cascade. Vacancy-production cross sections come from EEDL; the relaxation topology and branching ratios come from EADL; line energies, natural level widths, and optional Elam fluorescence yields come from xraydb. The result is an incoherent, self-absorbed track-length estimate evaluated on the fine line-energy grid.

## Atomic data and vacancy yield

For element {math}`a` and initially ionized subshell {math}`p`, PyRITE reads the EEDL {cite:p}`perkins1991eedl` ENDF-6 File 23 sections `MT=534` through `MT=572`. Their TAB1 records give the electron-impact ionization cross section {math}`\sigma_{ap}(T)` in barns as a function of incident-electron energy. Declared lin-lin interpolation is used inside each table's range and the cross section is zero outside it. The packaged tape is checksum-pinned before it is parsed.

### Relaxation cascade

A vacancy relaxes through a chain of radiative and nonradiative (Auger and Coster--Kronig) transitions, each of which creates new, less tightly bound vacancies. PyRITE reads the EPICS2025 EADL {cite:p}`perkins1991eadl` ENDF-6 File 28, `MT=533`. For each subshell {math}`i` it gives the binding energy {math}`B_i`, the occupancy, and every transition: radiative {math}`i\to j` with photon energy and probability, and nonradiative {math}`i\to(j,k)` that fills {math}`i` from {math}`j` and ejects an electron from {math}`k`. The probabilities of one subshell sum to one.

Let {math}`D_{ij}` be the expected number of vacancies created in {math}`j` by one decay of a vacancy in {math}`i`: a radiative {math}`i\to j` adds its probability to {math}`D_{ij}`; a nonradiative {math}`i\to(j,k)` adds its probability to {math}`D_{ij}` and to {math}`D_{ik}`. The expected number of vacancies that ever occupy {math}`i` per primary vacancy in {math}`p` is

```{math}
:label: eq-characteristic-cascade

V = (\mathbb{1}-D)^{-1} = \sum_{m\ge0} D^m .
```

Subshells are ordered by decreasing binding energy, spin-orbit partners of equal EADL binding energy by increasing ENDF designator. Every EADL transition fills a hole from a less tightly bound subshell, so {math}`D` is strictly upper-triangular and nilpotent in that order: the series terminates after at most as many terms as there are subshells and is evaluated exactly by forward substitution, with no iteration cutoff and no convergence tolerance. The loader checks that ordering for every transition, along with the per-subshell sum {math}`\sum{\rm FTR}=1` (to the {math}`10^{-5}` precision of the tabulated six-figure values), positive photon energies, nonnegative electron energies, and occupancies within {math}`2j+1` that sum to {math}`Z`.

The relaxation cutoff {math}`B_{\rm cut}` bounds this propagation by binding energy: a subshell with {math}`B_i\leq B_{\rm cut}` does not decay, its row of {math}`D` is zero, and its vacancies are terminal. The default and minimum is 50 eV. Subshells bound at or below it are also not created as primaries.

With {math}`R_{i\ell}` the probability that one decay of a vacancy in {math}`i` emits line {math}`\ell`, the photons of line {math}`\ell` per primary vacancy in {math}`p` are

```{math}
:label: eq-characteristic-cascade-yield

Y_{ap\ell} = \sum_i V_{pi}\,R_{i\ell}.
```

The cascade is exact expectation propagation, not sampling. It is linear in the primary vacancy population and matches the track-length estimator, so it adds no variance and no per-segment cost: {math}`Y_{ap\ell}` is a precomputed matrix, and scoring multiplies the segment's shell cross sections by it.

### Line energies and widths

Each EADL radiative transition {math}`i\to j` above 50 eV becomes a line. When xraydb {cite:p}`elam2002` tabulates the same {math}`(i,j)` level pair — including grouped levels such as `M4,5`, which claim each component pair — the line takes xraydb's energy and label. Otherwise it takes the EADL transition energy and the IUPAC label `i-j`, and `CharacteristicCrossSectionTable.line_source` records which source each line's energy came from. Both kinds take their natural width from xraydb, below. xraydb lines with no EADL radiative counterpart, such as the dipole-forbidden K-L1, carry no EADL probability and are omitted.

### Fluorescence-yield option

EADL's radiative branching ratio {math}`\omega_i=\sum_\ell R_{i\ell}` differs from the Elam values xraydb carries (Krause's compilation for K and L). For K subshells bound above 200 eV the EADL/Elam ratio stays within 0.87--1.20. For L subshells from about Ti upward it spans 0.47 (W L1) to 2.0 (Ti L2); lighter elements reach 7.2 (S L1). `fluorescence_yields="elam"` rescales each subshell's radiative branch to xraydb's {math}`\omega_i` and its nonradiative branch to {math}`1-\omega_i`, keeping the EADL shape of each branch. It is an approximation, not a correction: the two sources disagree in their nonradiative branching too, for example Cu L2-L3 Coster--Kronig is 0.009 in EADL and 0.47 in the Elam tables xraydb carries, and rescaling does not reconcile that. The option also rescales M, N and O subshells wherever xraydb tabulates a coarser yield for them. The default is `"eadl"`, and the choice is part of the characteristic-model marker.

### Track-length estimate

A material segment of length {math}`L_j` therefore produces the expected integrated line yield

```{math}
Y_{jap\ell}
=\frac{n_a L_j\,\sigma_{ap}(T_j)\,Y_{ap\ell}}{4\pi N_e}
  \exp[-\tau_j(E_{a\ell})].
```

Here {math}`n_a` is the elemental number density, {math}`N_e` is the number of incident electrons, and {math}`\tau_j` is the Beer--Lambert optical depth from the segment to the surface along the observation direction. The {math}`1/(4\pi)` factor is the isotropic-emission approximation.

Index {math}`p` is the subshell the *electron* ionized; the photon may come from any subshell the cascade reaches. With {math}`B_{\rm cut}` above every binding energy, {math}`V=\mathbb{1}` and the estimator reduces to the direct-vacancy product {math}`\omega_{ap}I_{ap\ell}` with EADL branching.

## Natural Lorentzian line shape

Each emitted transition is represented by a normalized Lorentzian,

```{math}
L_\ell(E)=\frac{1}{\pi}
\frac{\Gamma_\ell/2}
     {(E-E_\ell)^2+(\Gamma_\ell/2)^2},
\qquad
\int_{-\infty}^{\infty}L_\ell(E)\,dE=1.
```

The transition full width at half maximum is the sum of the initial- and final-level natural widths, {math}`\Gamma_\ell=\Gamma_{\rm initial}+\Gamma_{\rm final}`, as tabulated by xraydb from atomic-level width compilations {cite:p}`krauseoliver1979,keskirahkonenkrause1974`. For a combined final label such as `M4,5`, PyRITE uses the mean of the available component widths. A missing final-state width contributes zero; a missing initial width is an error because it would leave the line's principal broadening undefined.

The code integrates the profile over each energy bin analytically rather than sampling it at bin centres. For bin edges {math}`E_b^-` and {math}`E_b^+`,

```{math}
q_{\ell b}=\frac{1}{\pi}
\left[
\tan^{-1}\!\left(\frac{2(E_b^+-E_\ell)}{\Gamma_\ell}\right)
-\tan^{-1}\!\left(\frac{2(E_b^--E_\ell)}{\Gamma_\ell}\right)
\right].
```

The bin masses retain the normalization of the physical Lorentzian on the whole energy axis. They are not renormalized over the requested window:

```{math}
0 < P_{\ell,W}=\sum_{b\in W}q_{\ell b}\leq 1.
```

Thus a finite window records the fraction {math}`P_{\ell,W}` of the integrated line yield and does not redistribute omitted tails into retained bins. A line centre outside the window still contributes its physical in-window tail. Changing either window boundary leaves every bin with unchanged edges unchanged. The spectral density contributed to bin {math}`b` is {math}`Y_{jai\ell}q_{\ell b}/\Delta E_b`, in photons {math}`\mathrm{eV}^{-1}\,\mathrm{sr}^{-1}` per incident electron. `pyrite.montecarlo.spectrum.characteristic.characteristic_line_window_mass` reports the captured and omitted probability per line. The spectrum function warns when a line centre lies inside the requested grid but less than `CHARACTERISTIC_SEVERE_TRUNCATION_FRACTION` (50%) of its mass is captured. Off-grid line centres contribute their tails without that warning.

### Bin-edge convention

{math}`E_b^-,E_b^+` come from `_energy_bin_edges_and_widths`, reusing the interior-midpoint, reflected-half-width convention shared with `pyrite._grid_semantics.node_bin_edges_and_widths`: interior edges sit at {math}`\tfrac12(E_i+E_{i+1})`, and each outer edge mirrors the adjacent spacing outward. The low edge is clamped to 0 eV if the reflected edge would be negative. This changes the first edge without adding a bin: there are still `grid.size + 1` edges. The upper edge is not clamped.

Measured transition-metal emission features can require several Lorentzians to describe unresolved satellites and asymmetric structure {cite:p}`holzer1997`. PyRITE deliberately uses one natural-width Lorentzian per transition because the available atomic tables do not provide a portable multi-component fit for every element and line. Detector response is still applied downstream and will dominate whenever its resolution is broader than the natural width. Satellite structure, chemical shifts, and multiple-vacancy broadening are therefore outside this model.

## Relaxation and transport scope

The relaxation model treats independent atoms and relaxes every primary vacancy through {eq}`eq-characteristic-cascade` down to the 50 eV binding-energy cutoff, so K-fed L vacancies, L-fed M and N vacancies, and L- and M-shell Coster--Kronig transfer all radiate from where they end up.

Energy is accounted within a declared approximation. Per primary vacancy, its binding energy equals the expected photon energy, plus the expected Auger-electron energy, plus the binding energy still held by terminal vacancies, plus a transition-energy defect: EADL's transition energies are computed separately from its single-vacancy binding energies, so {math}`B_i-B_j-B_k` and the tabulated electron energy do not agree exactly. The identity closes to the {math}`10^{-5}` branching-sum precision (`relaxation_energy_budget`). Over {math}`3\leq Z\leq98`, the defect stays under 1.5% of the primary binding energy for every K primary. For L primaries bound above 100 eV it reaches 6.3% (Si L2 at 104 eV) and 6.0% (K L1 at 381 eV), and still about 3% for Z ≥ 30 (Ga and Zn L1).

Validity limits:

- **Untransported Auger electrons.** Their energy is booked in the balance above but they are not produced as transport particles, so the vacancies they would create in neighbouring atoms and their own bremsstrahlung are absent. Coupling them back needs a stochastic cascade and is outside this model.
- **Multiple-vacancy effects.** EADL rates are for one hole. The cascade applies them independently to every vacancy it creates, with no shift of energies or rates from spectator holes, and no satellite structure.
- **Missing experimental M and N line data.** xraydb has no line list for many M and N transitions, such as Au M1, M2 and N1--N7, so those lines use EADL transition energies. These are relativistic calculations that can differ from measured M/N energies by tens of eV, with no multiplet splitting. Their natural widths still come from xraydb level widths. PyRITE does not claim M-shell spectroscopy.
- **EADL L3 branch shape for 3d metals.** EADL makes Lℓ (L3-M1) much stronger relative to Lα1 (L3-M5) than the xraydb tables do: 2.9 against 0.12 for Fe, 1.5 against 0.09 for Ni, 1.1 against 0.08 for Cu, and 0.83 against 0.07 for Zn. The gap closes at high Z: 0.065 against 0.052 for Au. The `"elam"` yield option keeps EADL's branch shape and cannot correct this. L-line ratios of 3d metals are therefore not claimed.
- **Secondary fluorescence.** {math}`\exp(-\tau_j)` is a pure sink: a characteristic photon absorbed on its way out does not re-emit. The error is small for a line below its own element's absorption edge and is not bounded for alloys or multilayers.
- **Sub-keV lines from above-floor parents.** The cascade emits M and N lines well below 1 keV from vacancies created by electrons above the transport floor. This partly compensates for direct sub-keV production the floor omits, but it does not validate sub-keV transport.

Characteristic emission uses the bremsstrahlung electron population and its default 1 keV transport cutoff. This retains more low-energy ionization path than the default 5 keV PXR/CBS population. The present stopping and scattering model is not validated below 1 keV, so characteristic scoring enforces {math}`E_{\rm cut}\geq1\,\mathrm{keV}`: an omitted cutoff resolves to 1 keV and an explicitly lower cutoff is rejected. Low-binding-energy vacancies that could physically be produced below 1 keV are therefore omitted. PyRITE does not claim precision characteristic yields for incident energies near this floor. The 50 eV relaxation cutoff bounds atomic relaxation and does not lower the electron transport-validity floor.

Flat slabs, finite footprints, blazed grooves, and layered samples reuse the same photon-escape geometry as the other radiation kernels. In a multilayer, each layer emits from its own elements and the photon is attenuated by every layer on its escape path; passive absorber layers do not become emitters for a different layer.

## Composition, storage, and analysis

Atomic relaxation is incoherent, so one component adds to either the incoherent or the optional coherent PXR/CBS spectrum. It is kept as its own array, `spec_characteristic`; `spec` and `spec_coherent` exclude it, and consumers add it once through `pyrite._spectral_components.line_spectrum` (`Result.line_total()` for API results). Checkpoints store it in `characteristic.h5`, separate from the line `line.h5` and continuum `brem.h5` datasets. The analysis app shows characteristic radiation by default and provides a **show characteristic radiation** checkbox that drops the component from displayed records without changing stored results. A missing characteristic file is valid and contributes no characteristic component.

## Limits and validation

- Zero density, path length, cross section, fluorescence yield, or branching intensity gives zero characteristic yield.
- A relaxation cutoff above every binding energy gives {math}`V=\mathbb{1}` and the direct-vacancy yield.
- Forward substitution reproduces an independent dense inverse of {math}`\mathbb{1}-D` to {math}`10^{-12}`, and {math}`D` is nilpotent for every tested element.
- Each decay row of {math}`D` sums to the radiative probability plus twice the nonradiative probability, which is one plus the number of extra holes the decay creates.
- Raising the cutoff above a subshell's binding energy removes every line from that subshell but leaves lines from more tightly bound subshells unchanged.
- With zero attenuation, a finite window integrates to {math}`P_{\ell,W}nL\sigma Y/(4\pi N_e)`; the infinite-window limit recovers the complete line yield.
- Narrowing a window only removes Lorentzian probability. It does not rescale bins retained by both windows; off-grid centres retain nonzero tails.
- Splitting a constant-energy segment preserves the total when its attenuation weight is also held fixed (in particular, with zero attenuation), because the estimator is then linear in path length. Subdividing a real absorbing track changes the segment-midpoint escape quadrature and therefore the total: for a uniform path of optical depth 2, one midpoint gives 0.3679, two equal subsegments give 0.4148, and exact path integration gives 0.4323.
- Increasing optical depth suppresses the line monotonically.
- Natural Lorentzian broadening is source physics; detector broadening remains a separate downstream operation.

This window convention and the EADL cascade are encoded in the characteristic-model marker used by dataset identities and case-content keys. The marker includes the EEDL and EADL checksums, the endf-parserpy and xraydb versions, and the fluorescence-yield source, and ends `eadl-cascade-eadl-yields-lorentzian-segment-escape-v7` by default. Records written under earlier markers are not reused: `v6` records hold L-shell-only Coster--Kronig yields on a line-energy cutoff, which is not the same spectrum.

Implementation owner: `pyrite.montecarlo.spectrum.characteristic.mc_characteristic_spectrum`. The source-to-code mapping, dimensional analysis, assumptions, and regression anchors are recorded in [Characteristic-radiation validation](../../validation/radiation-physics/characteristic-radiation.md) under ledger row `characteristic-radiation`.
