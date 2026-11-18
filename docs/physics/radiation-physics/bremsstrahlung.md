# Bremsstrahlung

PyRITE's default `bremsstrahlung_model="bremslib"` uses the direction-resolved BremsLib double differential cross section and requires tables for every layer element. Missing tables raise before computation unless `--allow-physics-fallback` explicitly permits the isotropic Livermore Evaluated Electron Data Library (EEDL) {cite:p}`perkins1991eedl`; EEDL can also be selected explicitly, together with `radiative_model="uncoupled"`. Both score the incoherent background along transported electron segments. The continuum shares the transport segments, escape geometry, and per-electron normalization with the [line kernel](coherent-radiation.md), but can use a separate photon-energy grid. Combining line and continuum requires compatible energy coordinates and density conventions; see [spectral observables](spectral-observables.md).

## EEDL fallback: MF=23/MT=527 and MF=26/MT=527

ENDF-6 organizes evaluated data by **file number** (`MF`, the kind of data) and **reaction number** (`MT`, the physical interaction). The pair therefore names a section of the EEDL evaluation; it is not a fitted parameter:

- `MF=23` contains integrated photo-atomic and electro-atomic cross sections. Its `MT=527` section is the total electro-atomic bremsstrahlung cross section {math}`\sigma(T)` as a function of incident-electron energy, tabulated in barns.
- `MF=26` contains secondary-particle energy/angle distributions. Its matching `MT=527` section describes the products of the same bremsstrahlung interaction. PyRITE uses the first subsection, whose `ZAP=0`, `LAW=1`, `LANG=1`, `NA=0` records give an isotropic tabulated photon-energy probability density. The second subsection uses `LAW=8` to report the outgoing electron's average energy loss and is not needed by the photon spectrum estimator.

The two sections are used together because they supply complementary pieces: MF=23 fixes the total probability of a bremsstrahlung event, while MF=26 fixes the conditional photon-energy shape. Their product is the required differential cross section. The section definitions follow ENDF-6 {cite:p}`trkov2018endf6`.

## EEDL cross section and spectrum

For element {math}`Z`, incident electron kinetic energy {math}`T`, and photon energy {math}`k`, the EEDL energy-differential cross section is

```{math}
\frac{d\sigma_Z}{dk}(T,k) = \sigma_Z^{23,527}(T)\,P_Z^{26,527}(k\mid T),
\qquad \int_0^T P_Z(k\mid T)\,dk=1.

```

`endf-parserpy` reads both sections from the checksum-pinned packaged `EEDL.endf`. PyRITE accepts the tape only when the total and photon tables are finite, ordered, non-negative, and declare the supported ENDF lin-lin laws. The MF=23 values are converted from barns to cm². Each native MF=26 photon density is normalized after parsing to remove only source-record rounding error.

The tape declares lin-lin interpolation for the total cross section, the secondary photon-energy axis (`LEP=2`), and the incident-energy panels (`INT=2`). PyRITE follows the first two, but not the declared Cartesian law between incident-energy panels. EEDL gives photon spectra at only 8–10 decade-spaced incident energies per element (carbon: 14.1 keV, 251 keV, 1.19 MeV and 12.2 MeV between 10 keV and 100 GeV). Interpolating adjacent panels at a fixed absolute photon energy leaves every {math}`k` above the lower panel's endpoint with only the upper panel's share, so the spectrum collapses towards {math}`k\to T`. At 30 keV in carbon it fell to 0.09–0.15 of Seltzer–Berger for {math}`k/T\ge0.5` (#174, `brem-source-comparison`).

PyRITE instead uses ENDF unit-base interpolation {cite:p}`trkov2018endf6`. Panel {math}`i`, with photon range {math}`[a_i,b_i]`, maps onto the reduced variable {math}`x=(k-a_i)/(b_i-a_i)\in[0,1]`, where its density becomes the unit-area {math}`q_i(x)=(b_i-a_i)P_i(k)`. Sub-panels are inserted at 32 geometrically spaced incident energies per decade:

```{math}
q(x\mid T) = (1-w)\,q_i(x) + w\,q_{i+1}(x),\qquad
w=\frac{\ln(T/T_i)}{\ln(T_{i+1}/T_i)},\qquad
[a,b](T) = [a_i,b_i] + \frac{T-T_i}{T_{i+1}-T_i}\bigl([a_{i+1},b_{i+1}]-[a_i,b_i]\bigr).

```

The photon range is interpolated linearly in {math}`T`, so {math}`b=T` because every EEDL panel ends at its incident energy. The shape weight is linear in {math}`\ln T`, the spacing of the native panels. Against Seltzer–Berger this weight is several times more accurate between panels than one linear in {math}`T`. All panels share the union of the native {math}`x` nodes, which represents each native panel exactly and keeps every mixture piecewise linear with unit area. Native panels are reproduced unchanged.

Between adjacent sub-panels the runtime interpolation stays Cartesian at fixed photon energy. Each sub-panel is held at its endpoint density up to the next sub-panel's endpoint, so the mixture stays continuous for {math}`k` between them. The implementation then imposes {math}`0<k\le T` and renormalizes the surviving piecewise-linear density exactly, so integrating the differential cross section over the physical photon range recovers the interpolated MF=23 total.

Against Seltzer–Berger, for the 24 catalogue elements from 10 keV to 1 MeV, the result is:

- within 0.96–1.34 pointwise for {math}`0.05\le k/T\le0.95`;
- within 0.98–1.81 for {math}`k/T>0.95`;
- within 0.99–1.07 in the radiative first moment.

The remaining error is panel sparsity, not interpolation.

The ENDF-6 electro-atomic format describes the bremsstrahlung photon subsection as an isotropic, angle-independent tabulated spectrum (`LAW=1`, `LANG=1`, `NA=0`) {cite:p}`trkov2018endf6`. PyRITE uses this isotropic angular model.

## Optional Bethe--Heitler backend

An analytic alternative is available with `cross_section_model="bethe-heitler"` on `mc_brem_spectrum`. It is the nonrelativistic, unscreened Born Bethe--Heitler form with the {cite:t}`elwert1939` correction, using relativistic momenta as a weakly-relativistic extension {cite:p}`kochmotz1959`:

```{math}
\frac{d\sigma}{dk}=\frac{16}{3}\alpha r_e^2 Z^2
\frac{1}{k p_i^2}\ln\!\left(\frac{p_i+p_f}{p_i-p_f}\right)
\frac{\beta_i}{\beta_f}
\frac{1-e^{-2\pi\alpha Z/\beta_i}}{1-e^{-2\pi\alpha Z/\beta_f}}.

```

Momenta are carried in units of {math}`m_ec`, built from the exact relativistic relation {math}`p=\sqrt{T(T+2m_ec^2)}/(m_ec^2)` and {math}`\beta=p/(1+T/m_ec^2)`, with {math}`T_f=T_i-k`. This backend is also the fallback when EEDL lacks an element or incident-energy range. Every fallback is identified by `PhysicsFallbackWarning`; campaign workers refuse it unless explicitly opted in. Direct low-level calls retain warning-based behavior. Malformed or checksum-mismatched EEDL data are not treated as missing coverage and remain hard errors.

## Default BremsLib direction-resolved backend

`cross_section_model="bremslib"` replaces isotropic emission with the BremsLib v2.0 double differential cross section {cite:p}`poskus2025bremslib`, taking the photon-energy spectrum from the same evaluation. BremsLib tabulates, per element, the scaled single differential cross section (SDCS) {math}`\chi(T_1,x)=(k/Z^2)\,d\sigma/dk` in mb and, at every grid node, the scaled double differential cross section (DDCS) {math}`(k/Z^2)\,d^2\sigma/(dk\,d\Omega)` in mb/sr over the photon emission angle {math}`\theta` measured from the incident electron direction, with {math}`x=k/T_1`. The complete grids are {math}`T_1` from 10 eV to 30 MeV (ratios up to 1.33), 13 values of {math}`x` from 0 to just below 1 (18 once staged, below), and 181--441 angles refined towards the forward peak. The cross sections PyRITE evaluates are

```{math}
\frac{d^2\sigma_Z}{dk\,d\Omega}(T,k,\theta)=\frac{Z^2}{k}\,\chi(T,x)\,S(T,x,\theta),
\qquad
\frac{d\sigma_Z}{dk}=\frac{Z^2}{k}\,\chi(T,x),
\qquad
\int_{4\pi}S\,d\Omega=1,

```

where the shape function {math}`S` is the DDCS divided by its own solid-angle integral, as the library defines it. Taking the shape and the spectrum from one evaluation keeps {math}`S` normalized against its own parent SDCS; PyRITE therefore never combines the BremsLib shape with the EEDL spectrum.

The interpolation is PyRITE's own. It is not a port of the vendor's `Interpolate_DCS` (a GPL-3 weighted least-squares spline fit), and is built from the library's published grids:

1. **Nodes.** Each node's DDCS is linear in {math}`\theta` between tabulated angles. It is rescaled so that the exact solid-angle integral of that linear interpolant, {math}`2\pi\sum_i\int_{\theta_i}^{\theta_{i+1}}f(\theta)\sin\theta\,d\theta`, equals the node's SDCS. All node grids are subsets of the finest one, so they are resampled onto it without changing any interpolant.
2. **Incident-energy refinement.** Backward emission falls steeply and convexly with {math}`T_1`, so linear interpolation across a library interval overshoots it. At load, each interval is split into four in {math}`\ln T_1`. Each sub-node's shape is the weighted geometric mean of its neighbours' shapes, renormalized to unit integral, and its SDCS is interpolated log-log. Library nodes are kept bit for bit.
3. **Evaluation.** The scaled DDCS is interpolated linearly in {math}`\theta`, in {math}`x=k/T` and in {math}`\ln T` on the refined grid. Each step is a convex combination of nodes whose angular integrals equal their SDCS, so the {math}`4\pi` integral of the interpolated DDCS is exactly the identically interpolated SDCS. The energy spectrum is recovered by construction, not by a separate renormalization.

**30--100 MeV extension** (#439). Above 30 MeV the library is incomplete: at 50, 60, 80 and 100 MeV it tabulates the SDCS for {math}`x\le 0.95` and the DDCS only at {math}`k=0` (40 MeV has {math}`k=0` alone and is skipped). Locally generated tables keep these *extension rows* (`build_table`, default `t1_max_MeV=100`) with the two tip SDCS columns stored as missing, and staging fills them, anchored on the last complete row {math}`T_c=30` MeV:

```{math}
\chi(T,x_m)=\chi(T,0.95)\,\frac{\chi(T_c,x_m)}{\chi(T_c,0.95)},\qquad
S(T,x,\theta)\propto S(T,0,\theta)\,\frac{S_{\rm 2BS}(T,x,\theta)}{S_{\rm 2BS}(T,0,\theta)}\,C(x,\theta),
\qquad
C=\frac{S(T_c,x,\theta)\,S_{\rm 2BS}(T_c,0,\theta)}{S(T_c,0,\theta)\,S_{\rm 2BS}(T_c,x,\theta)},
```

with {math}`x_m\in\{0.975,x_{\rm top}\}` and {math}`S_{\rm 2BS}` the Schiff density (Koch--Motz 2BS). Every library value is kept; {math}`x\to0` gives the library {math}`k=0` shape and {math}`T\to T_c` the complete row. A Bethe--Heitler tip is not used: the Born result turns negative at {math}`x=0.9999` for heavy elements. The tip cells carry about 2 % of the radiative first moment; predicting 30 MeV from 15 MeV this way reproduces the library's half-flux and 90 % angles within 1--3 % (the large-angle tail within 10--55 %). Radiative stopping over ESTAR stays within 2.5 % of its 30 MeV value up to 100 MeV for C, Si and W. The released catalogue tables (`tables-bremslib-2`) carry these rows. Validation: `bremslib-high-energy-extension`.

Between the last node, {math}`x_{\rm top}` (0.99, {math}`1-50\,{\rm eV}/T_1`, or 0.9999 depending on {math}`T_1`, per the library manual), and the kinematic tip, the top node's value is held. Above {math}`k=T` the cross section is zero.

**Electron--electron bremsstrahlung** (#439). BremsLib is computed for one electron in the screened central potential of the atom, which is the electron--nucleus term. Bremsstrahlung in collisions with the atomic electrons adds {math}`Z\chi_e` to {math}`Z^2\chi_n`. Following Seltzer and Berger {cite:p}`seltzer1985bremsstrahlung` (Eqs. 18--19),

```{math}
\chi(T,x)=\chi_{\rm BL}(T,x)\left[1+\frac{\eta(Z,T,x)}{Z}\right],\qquad \eta=\frac{\chi_e}{\chi_n},
```

with {math}`\eta` from their Table 4 (Haug's electron--electron cross section, the Fedyushin Coulomb factor and Hartree--Fock screening), transcribed for {math}`Z=6,13,29,47,74,92` into `data/bremsstrahlung_electron_electron.toml`. {math}`\eta` is interpolated linearly in {math}`\ln Z` (held outside 6--92), in {math}`\ln T` and, for {math}`x\ge0.1`, in {math}`x`. Table 4 has no node between {math}`x=0` and 0.1, where {math}`\eta` falls steeply from its soft-photon value (2.6--2.9 at 30--100 MeV) to about 1.15. There

```{math}
\eta(x)=\eta(0.1)+[\eta(0)-\eta(0.1)]\,g(x),\qquad
g(x)=\frac{(1+x/\kappa_s)^{-1}-(1+0.1/\kappa_s)^{-1}}{1-(1+0.1/\kappa_s)^{-1}},
```

with {math}`\kappa_s(T)` fitted so that the {math}`\chi_n`-weighted mean of {math}`\eta` reproduces Table 4's {math}`\bar\eta` (Eq. 27), using {math}`\chi_n` from the Seltzer--Berger digital tables. The fit is independent of {math}`Z` to within 8 % above 1 MeV ({math}`\kappa_s\approx0.04` at 1 MeV, 0.015 at 30--100 MeV), and the stored scale is the mean over the six elements (`checks/sb85_eta_small_kappa_fit.py`). Below 0.5 MeV the rule is linear. keV photons at 30--100 MeV ({math}`x\sim10^{-5}`) take the tabulated {math}`\eta(0)` and do not depend on this shape: the Si continuum there rises about 20 %.

At load, after the {math}`T_1` refinement, five sub-nodes ({math}`x=0.004,0.01,0.02,0.035,0.06`) are inserted between the library's 0 and 0.1. They take the linear interpolant of the SDCS and DDCS, so the electron--nucleus values served are unchanged, and every node's SDCS and DDCS is then multiplied by {math}`1+\eta/Z`. All consumers read this one staged table: the scorers, the fused CUDA reducer and the coupled partition and sampler, including the dielectric factor. The electron--electron photon is given the electron--nucleus angular distribution at the same {math}`(T,x)`; Seltzer and Berger give no angular data, and both are forward-peaked within {math}`\sim1/\gamma` at high energy. Limits: {math}`\eta\to0` at {math}`x\to1`, since {math}`k_{\max}<T` for an electron target; the radiative stopping integral reproduces {math}`1+\bar\eta/Z` times the nuclear one within 0.005 in {math}`\bar\eta` (0.007 on the staged 18-node grid). `prepare_bremslib_table(..., electron_electron=False)` stages the electron--nucleus library alone. Against ESTAR radiative stopping the term removes the low-{math}`Z` shortfall (C 0.885 → 1.042 at 30 MeV); the 24 catalogue elements then sit at 0.974--1.057 of Seltzer--Berger in first moment from 10 keV to 30 MeV, a residual electron--nucleus difference within that paper's quoted 3--10 %. Validation: `bremslib-electron-electron`.

Tables come from `pyrite.xsgen`. The released catalogue tables install with `pyrite tables fetch bremslib`, and other elements are generated from a local BremsLib checkout. Because the physics core does not import `pyrite.xsgen`, a driver resolves them with `pyrite.xsgen.bremslib.tables.load_bremsstrahlung_tables(elements)` and passes them as `bremslib_tables`. Fallback coverage:

- With explicit `"bremslib"`, a released element whose table has not been fetched is an error that names the fetch command.
- An element with no table, or a segment energy outside the table's range, warns and uses isotropic EEDL for only that element or those segments.

Runs select the continuum through `Physics.bremsstrahlung_model` (also `Settings`, profile physics and `pyrite profile physics set --bremsstrahlung-model`). The default `"bremslib"` requires a table for every element of its crystal and absorber layers and otherwise fails before computation unless `--allow-physics-fallback` permits EEDL; `"eedl"` selects the packaged evaluation and needs an explicit `radiative_model="uncoupled"`. Legacy `"auto"` keeps its historical table-checked resolution, and covered cases resolve identically under either spelling. A case records only the resolved choice, so `dataset_identity`, `case_content_key` and the provenance marker (`BREMSSTRAHLUNG_BREMSLIB_MODEL` or `BREMSSTRAHLUNG_MODEL`) name what actually ran, and the tables' manifest digests join the xsgen identity markers. The default followed the source comparison and recommendation in `brem-source-comparison` (#86): BremsLib is the recommended source for the 24 catalogue elements from 10 keV to 30 MeV, with its electron–electron shortfall at low Z and MeV energies recorded there; the staged tables have since added that term (above). The low-level `mc_brem_spectrum` takes the same `cross_section_model="auto"` default: BremsLib when the segments carry directions `v_hat` and a table resolves for every composition element (the supplied `bremslib_tables`, else the installed release through the driver-side loader), otherwise a warned EEDL fallback. The runner pins the case's already-resolved choice, so `"auto"` never re-resolves it. Public archive hosting is deferred; archives are distributed internally until then.

## Coupled soft/hard radiative transport

The coupled BremsLib mode adds soft radiative stopping and samples hard photon events that debit electron energy. Its partition, event record, straggling treatment, and scoring boundary are documented in [Hard BremsLib photon events](hard-bremsstrahlung-events.md). The default couples when BremsLib tables resolve; campaign preflight requires explicit permission (`--allow-physics-fallback`) to substitute uncoupled scoring for missing-source or grooved configurations. Because `bremslib-soft-hard` is the default, that permission degrades it like legacy `"auto"`; without the flag every coupled request is strict. See the [fallback policy](../../guides/physics-fallbacks.md) for measured row metadata and cache admission. Uncoupled transport debits only collision stopping, so between 30 and 100 MeV `build_cases` refuses a case that resolves uncoupled (missing tables, explicit `"eedl"`, or a groove) unless the run sets `radiative_model="uncoupled"`; such a case carries no coupled keys, so its identity records the choice (`campaign.bremsstrahlung_cases.require_coupled_radiative`). Above 100 MeV, where the tables end, no coupled mode exists and this screen does not apply.

### Dielectric (Ter--Mikaelian) suppression

At relativistic energy the polarization of a homogeneous medium suppresses soft photons. The coupled mode uses the leading Ter--Mikaelian factor (Ter-Mikaelian, *High-Energy Electromagnetic Processes in Condensed Media*, 1972; PDG, *Passage of Particles Through Matter*)

```{math}
F(T,k)=\frac{k^2}{k^2+a^2},\qquad a=\gamma(T)\,\hbar\omega_p,\qquad \gamma=1+\frac{T}{m_ec^2},
```

with {math}`k`, {math}`a` and {math}`\hbar\omega_p` in eV and {math}`\gamma` from the kinetic energy. One {math}`\hbar\omega_p` per layer comes from the whole layer's electron density (`materials.attenuation.plasma_energy_eV`; Si 31.05 eV, so {math}`a\approx1.85` keV at 30 MeV and 6.11 keV at 100 MeV). The same suppressed DCS feeds every consumer: soft stopping {math}`\int_0^{k_c}\chi F\,dk`, the hard rate and photon-energy inverse CDF {math}`\int_{k_c}^{T}\chi F/k\,dk` (CPU and CUDA cores), the hard-event element choice, and the expected-value and soft continuum scorers, which multiply the DDCS by {math}`F` and refuse a material or EEDL fallback that does not match the transport record. Because {math}`F` is curved below {math}`k\sim a`, each linear BremsLib cell {math}`\chi=A+Bk` is integrated exactly (`transport.dielectric_radiative.dielectric_cell_moments`, with {math}`x=k/a`: {math}`\tfrac12\ln(1+x^2)`, {math}`x-\arctan x`, {math}`\tfrac12[x^2-\ln(1+x^2)]`, and a six-term series below {math}`x=0.1`), rather than scaling the table nodes. {math}`F` is angle independent, so the conditional photon direction is unchanged.

Limits: {math}`\hbar\omega_p=0` or {math}`k\gg a` gives {math}`F\to1` and the original partition exactly; {math}`k\ll a` gives {math}`F\approx k^2/a^2`, which makes the infrared rate finite. Suppression is on for every coupled run (`hard_radiative.radiative_core_args`), so the model has no step in beam energy. At low {math}`\gamma` it reaches only {math}`k\sim\hbar\omega_p`: with {math}`k_c=1` keV it lowers Si soft radiative stopping by about 5 % and total radiative stopping by 0.2--0.7 % at 10--40 keV, where the leading high-{math}`\gamma` form is itself only indicative. Transport records `dielectric_model = "ter-mikaelian-v1"` and the layer plasma energies, and `BREMSSTRAHLUNG_BREMSLIB_MODEL` carries `tm-coupled-v2`. The fused float32 CUDA continuum reducer applies the same factor per segment (`brem_jit_kernel._bremslib_or_fallback_weighted_scalar`). Omitted: LPM suppression, interfaces and transition radiation, optical dispersion, and the dielectric change of the angular law. For uncoupled or post-hoc estimates, `spectrum.dielectric_suppression.dielectric_omission_bound` bounds the fractional yield overestimate above {math}`k_{\min}` for fixed trajectories. Validation: `dielectric-suppression`.

## Lattice-coherent share in crystals

BremsLib tabulates isolated-atom bremsstrahlung. In a crystal, the Debye--Waller part {math}`e^{-2W}` of the same emission is lattice-coherent: the CXR route scores it as coherent-bremsstrahlung (CBS) lines at the reciprocal vectors it keeps. Orientation-averaging the screened-potential sum gives

```{math}
\Big\langle\Big|\sum_j u(\mathbf q)e^{i\mathbf q\cdot\mathbf R_j}\Big|^2\Big\rangle
= N|u|^2\big(1-e^{-2W}\big)+e^{-2W}|u|^2\Big|\sum_j e^{i\mathbf q\cdot\mathbf R_j}\Big|^2 ,
```

and the second term is the line sum. Adding the CBS lines to the full amorphous continuum therefore counts that share twice: about 8 % of the total 0.5--10 keV yield on the #642 Si (300 keV) and MoS{sub}`2`/Si (30 keV) scenes at {math}`|\mathbf g|\le30` Å{sup}`-1`.

Each crystalline layer's continuum is therefore scored with the share of exactly the reflections its line route scores removed. The share is the orientation-averaged Born--Debye--Waller CBS-only line yield {math}`S_j(T,\theta,k)` per unit path, from the production CBS amplitude with exact recoil in vacuum kinematics. It is subtracted from element {math}`j`'s scaled DDCS at every table node as {math}`k\,S_j/(n_jZ^2)`, clamped at zero, and the node's SDCS follows the new DDCS's solid-angle integral. Because the orientation average of one reflection depends only on {math}`|\mathbf g|`,

```{math}
S_j=\sum_{\mathbf g\ \rm scored}|U_{\mathbf g}|^2\,\frac{n_j|u_j(g)|^2}{\sum_i n_i|u_i(g)|^2}\,H(|\mathbf g|;T,\theta,x),\qquad
x=\frac{\hbar\omega(1-\beta\cos\theta)}{\hbar c\,|\mathbf g|\,\beta},
```

where {math}`x` is {math}`\hat{\mathbf g}\cdot\hat{\mathbf v}` at the classical resonance. The kernel {math}`H` depends on no crystal. It is one released table (`pyrite tables fetch brem-lattice-share`, built by `scripts/release_brem_lattice_share_table.py`) on these nodes:

- {math}`|\mathbf g|`: 69 log nodes over 0.25--75.4 Å{sup}`-1`;
- {math}`T`: 36 log nodes over 1 keV--1.06 MeV;
- {math}`\theta`: 5° steps;
- {math}`x`: 210 bins.

It is interpolated linearly in {math}`\ln|\mathbf g|` and {math}`\theta` at fixed {math}`x`. Between {math}`T` nodes the total is interpolated in {math}`\ln T` and the normalized shape linearly. Interpolation error:

- Si-scene probes at these spacings: {math}`|\mathbf g|` at most 1 % per bin, {math}`\theta` at most 0.7 % per bin (2.7 % at 10°), {math}`T` at most 0.4 % in total.
- Over the whole released grid, dropping one node (twice the spacing) misplaces 4.9 % ({math}`|\mathbf g|`), 1.2 % ({math}`\theta`) and 4.6 % ({math}`T`) of the mass, mass-weighted. The worst nodes are at the recoil edge in {math}`|\mathbf g|` and at 1--4 keV, where the share is small.
- End to end (Si 30 keV, 45°, {math}`G=30`), the share is 1.003 of the direct quadrature, 0.98--1.03 per bin.

Reflections outside the kernel's {math}`|\mathbf g|` range (0.25--75.4 Å{sup}`-1`) contribute no share. Only very large cells reach the lower edge.

Measured selections subtract the share of the set they select. The streamed route, which keeps the build-time families, subtracts that set's share. The measured selector itself stops on each shell's lines less this share; see [reflection selection](../materials/reflection-selection.md#default-measured-selection).

Table nodes with {math}`T` below the line route's electron cutoff `E_cut_lines_keV` (5 keV by default) keep the full DDCS: the line route scores no CBS from those electrons, so nothing would replace the removed share. On a 16-history MoS{sub}`2`/Si transport, the share below the cutoff was 2.8 % of the share and 0.23 % of the continuum, with 95 % of it at 0.5--2 keV. Segments between the two table nodes that bracket the cutoff receive an interpolated partial correction. This is a slight over-subtraction: about 3e-3 of the share at Si 30 keV, a few 1e-4 of the continuum.

Each corrected run records `brem_lattice_share` in its output and provenance: the correction model and, for each layer, whether it was corrected (with the line cutoff, scored reflection count and each element's removed fraction) or the reason it was not.

Only emission changes. Transport keeps the full tables, because the electron loses that energy to the CBS lines instead. Corrected tables keep their parent's key and digest, which the coupled partition check compares. The correction carries its own identity marker.

Validated on frozen transports of the two #642 scenes (#1250):

- **Si 300 keV:** the share matches the Monte Carlo CBS within 0.999--1.008 for {math}`G\ge12` Å{sup}`-1`.
- **Si 300 keV, temperature check:** the match holds within 0.2--0.7 % while CBS changes 2.6× with {math}`B` between 0.23 and 0.92 Å{sup}`2`.
- **MoS{sub}`2`/Si 30 keV:** before the line-cutoff floor, the share was 4.5 % above the Monte Carlo CBS (0.4 % of the continuum), concentrated at 0.5--2 keV. The beam stops in this scene, and the share removed from electrons below `E_cut_lines_keV` (2.8 % of the share, 95 % of it at 0.5--2 keV, measured on a 16-history transport) accounts for most of that excess. The remainder is partly explained by the share's floor at {math}`T=1`--2 keV.

Not corrected, and stated as such:

- the isotropic EEDL continuum, which warns with `LatticeShareUncorrectedWarning`;
- elements without a BremsLib table;
- channeling cases;
- incident energies above 1.06 MeV.

Grazing photon exit is also uncovered: refraction suppresses the production lines there, while the share uses vacuum kinematics. Validation: `brem-lattice-coherent-fraction`.

**Migration.** Default crystalline BremsLib spectra lose the double-counted share: 3--4 % of the total 0.5--10 keV yield on the #642 scenes at {math}`G=8` Å{sup}`-1` and about 8 % at {math}`G=30` Å{sup}`-1`, and the total spectrum's dominant peak falls by 9--12 % at {math}`G=30`. Line spectra are unchanged at a fixed reflection set. Crystalline BremsLib checkpoints written before the correction do not resume: their identity lacks the `brem_lattice_share_model` marker. Pass the private case key `_brem_lattice_share=False` only to reproduce the old double-counting sum in validation harnesses.

## Per-segment yield

For the EEDL and Bethe--Heitler backends emission is taken **isotropic**. For a segment of length {math}`L` traversed in an element of number density {math}`n_Z`, the contribution to the observed spectrum is

```{math}
\frac{d^2N}{dE\,d\Omega}=\frac{1}{4\pi}\,n_Z\,L\,\frac{d\sigma}{dk}\,T_{\rm abs},

```

with {math}`T_{\rm abs}` the Beer--Lambert escape transmission from the segment midpoint along the observation direction. Contributions are summed over segments and electrons and divided by the electron count, so the returned quantity is **photons per eV per steradian per incident electron** — the same units as the line spectrum.

Isotropy approximates a scattering-randomized electron population in the intended weakly relativistic regime; it does not resolve directional emission from an individual electron. The BremsLib backend does: each segment's {math}`\frac{1}{4\pi}\frac{d\sigma}{dk}` is replaced by {math}`\frac{d^2\sigma}{dk\,d\Omega}(T,k,\theta)` with {math}`\cos\theta=\hat v\cdot\hat n`, the straight-flight electron direction against the observation direction. In a crystal, its lattice-coherent fraction, which forms the CBS lines, is removed for the scored reflections ([above](#lattice-coherent-share-in-crystals)). The isotropic EEDL continuum keeps that fraction, so adding it to the line spectrum double counts it.

Compound targets sum each element's cross section at its own number density. The analytic backend carries explicit {math}`Z^2` scaling; EEDL supplies an evaluated cross section for each element. Self-absorption uses the summed attenuation of the whole composition. In a layered stack each layer's segments radiate with that layer's composition and every photon is attenuated across the whole stack.

## Runtime implementation

The EEDL tables are prepared once per element and requested photon-energy grid. Every native incident-energy panel is interpolated onto that grid once, while the incident-panel bracket, interpolation fraction, total cross section, and exact cutoff-normalization factor are stored as one-dimensional segment arrays. Those staged values are reused for every reduction chunk; the tabulated physics is not reparsed or recopied for each chunk.

On CUDA with float32 spectra, one fused kernel interpolates the two adjacent panels, applies the {math}`k\le T` support, selects Bethe--Heitler only for uncovered segments, applies the existing absorption model, and reduces directly into the output bin. This avoids a dense segment-by-energy cross-section scratch array. Other backends and float64 use the same equations through the portable chunked implementation. Its memory admission uses the larger EEDL working-set estimate, and delayed CUDA/ROCm allocation errors are recognized at the point where they surface so the runner can retry the bremsstrahlung phase with a smaller chunk. Non-memory runtime errors are not retried.

## Escape and geometry

The escape path reuses the line kernel's geometry: a flat slab uses the z-only distance to the exit face, a finite rectangular footprint takes the nearest of the prism's six faces along the fixed observation direction, a layered absorber sums {math}`\mu_i\,\Delta z_i` across the stack, and a blazed groove replaces the distance with the exact periodic working-facet path. Grooved escape is single-slab only and requires the exact working-facet normal; other combinations raise rather than silently using flat attenuation. Details and limits are in [Photon escape and in-medium dispersion](photon-escape-and-dispersion.md).

Transport supplies material segments only, so vacuum legs advance the clock but do not radiate.

## Energy range and cutoffs

- `k <= 0` and `k > T` are hard-zero bins for EEDL. The optional analytic backend also sets the endpoint {math}`k=T` to zero.
- The EEDL tables cover their declared incident and secondary-energy ranges. Requests outside the incident-energy range warn and use Bethe--Heitler for only the affected segments.
- The normalized EEDL spectrum extends to 0.1 eV in the packaged tape. A simulation grid with a higher lower bound intentionally records only the corresponding partial total cross section.
- Background calculations use a low transport cutoff (typically 1 keV): electrons below the line-radiation cutoff still radiate into the soft X-ray window.
- One transport can be shared by radiation populations with different cutoffs, so each kernel re-applies the stopping rule to a population-specific energy floor. Rows starting below the floor are dropped; a terminal flight that crosses it is shortened — start time and energy unchanged, length and midpoint truncated — with the truncation distance solving {math}`E_{\rm end}=E_{\rm cut}` under the midpoint rule, so the shortened flight's state is reconstructed rather than lost. A higher-cutoff consumer therefore cannot recover radiation from the lower-cutoff tail.
- Attenuation is available only within the optical tables' supported ranges. Where {math}`\mu` is unavailable, the background kernel substitutes zero attenuation. This is a numerical fallback, not evidence that the material is transparent. Production grids apply the material-dependent floor described in [energy-grid semantics](energy-grid-semantics.md); explicit grids still need their data support checked.

## Path-integral order

Each row contributes one evaluation of the integrand rather than a quadrature along the flight. Under `energy_model="midpoint"` that evaluation uses the representative energy `E_repr_keV`, making it a midpoint rule (second order in the flight length); frozen rows keep the left-endpoint one (first order). Refinement controls the residual path-quadrature error. `brem_endpoint_quadrature_error` measures the residual difference, opt-in and read-only.

The continuum needs no coherent flight grouping. Its contribution is linear in segment length when energy and attenuation are held fixed. Subdivision of an evolving, absorbing track changes the quadrature and is not exactly invariant.

## Per-primary band reduction

On CUDA with the FP32 array backend, the per-primary band scorer fuses the
existing FP32 cross-section and attenuation cells with FP32 photon quadrature
partials and FP64 accumulation into original primary IDs. A partial belongs to
one transport segment; this precision choice does not narrow stored trajectory
positions, times or phases, or change coherent/incoherent CXR arithmetic. Empty
primary histories retain zero scores. Angular-table staging is capped at 65,536
escaped rows and never allocates a full segment × photon matrix. CPU, global
`PYRITE_FP64=1`, and unsupported model/table combinations retain portable scoring.

The reduction order differs from the portable FP32 matrix reduction, so parity
is within rounding tolerance rather than bitwise. Such changes can move an
adaptive stopping decision at a threshold. Adaptive BremsLib policies that
monitor bremsstrahlung therefore carry `brem_band_reduction_model` in dataset
identity, case content keys and resolved provenance; fixed counts, CXR-only
policies and other continuum models omit it. The current generation is
`cuda-fused-fp32-row-fp64-primary-band-v2`. FP32 row partials cover only the
photon quadrature for one segment; FP64 primary sums avoid accumulating
roundoff over long histories. Benchmark controls also compare FP64 photon
partials and FP32 primary sums without changing the source-table model identity.

## External backgrounds

`load_external_brem` interpolates a two-column external background (energy [eV], intensity) onto the spectral grid, for comparison against or subtraction from an independently simulated continuum — e.g. a NIST DTSA-II simulation. The file is treated as an **as-detected** spectrum: window efficiency and detector resolution are not re-applied, and energies outside the file's range interpolate to zero. The intensity must already be in the detected units of the plot it joins.

## Limits and assumptions

- EEDL and Bethe--Heitler emission is isotropic, appropriate only to the intended weakly relativistic regime; no backend carries polarization;
- the BremsLib backend takes the emission angle from each segment's straight-flight direction and one fixed observation direction (a point detector); the DDCS is azimuthally symmetric (unpolarized beam, unoriented target). Interpolation error is under about 1.5 % in the DDCS and 0.4 % in the SDCS at the library's own spacing, estimated from drop-one-node tests over 20 keV--3 MeV for C, Si and W. Against the vendor's `Interpolate_DCS` output for Au it is at most 0.8 % on the {math}`T_1` grid, and a 0.4 % median (5.4 % at the last point before the tip) between {math}`T_1` nodes on the 0-degree forward peak;
- the BremsLib portable path costs about four times the EEDL path on CPU; a fused CUDA reducer matches the portable CUDA path on the hardware-gated synthetic-table comparison (including an out-of-range fallback segment);
- EEDL is elemental atomic data; molecular bonding, density-dependent emission effects, and interactions below the configured transport cutoff are outside this model;
- the optional/fallback Bethe--Heitler form is unscreened and approximate, especially for high {math}`Z`; its Elwert factor is not a substitute for an evaluated relativistic cross section;
- segment contributions and incident electrons are summed incoherently, with no phase and therefore no coherent-emission counterpart;
- in crystals, the lattice-coherent (CBS) share of the BremsLib continuum is removed for the scored reflections (above); the EEDL continuum keeps it;
- self-absorption uses the same layered escape model as line radiation, with no feedback on emission; see [Multilayer materials](../materials/multilayer-materials.md).

## Validation

Ledger rows: `bremslib-angular-model` for the default direction-resolved backend, `bremslib-radiative-partition` and `bremslib-radiative-event-spectrum` for the coupled mode, `dielectric-suppression` for its Ter--Mikaelian factor, `bremslib-electron-electron` for the electron--electron term, `brem-lattice-coherent-fraction` for the crystalline lattice-share subtraction, `brem-spectrum` for EEDL parsing, interpolation, normalization, fallback selection, and per-segment assembly, `external-brem-subtraction` for the weighted scale-only sideband fit that consumes a loaded external background, `substep-radiation-invariance` for the representative-energy evaluation, `radiation-error-estimators` for the opt-in quadrature estimator, and `finite-transverse-crystal` / `blazed-groove-geometry` for the escape variants. The full derivation, dimensional analysis, limits, and numeric comparison are in [Bremsstrahlung spectrum validation](../../validation/radiation-physics/brem-spectrum.md), with the literature comparison in [external bremsstrahlung comparison](../../validation/literature-ref/external-bremsstrahlung-comparison.md). Implementation owner: `pyrite.montecarlo.spectrum.brem.mc_brem_spectrum`.

## Channeling zone weights

Planar and axial channeling zone rows multiply the native per-length spectrum by a
depth-averaged transverse-flux weight. Its normalized screened-Born kernel
combines nuclear and electron localization with strengths $Z^2:Z$.
The kernel is evaluated at entrance energy and neglects photon-energy-dependent
localization and distinct nuclear/electron screening logarithms. Native
continuation rows retain unit weight; axial rows average over the 2D cell.

The weights are density expectations, not per-trajectory squared amplitudes.
They reproduce the amorphous rate for uniform transverse density. They do
not have a measured orientation-dependent accuracy bound; channeling admission
ε covers `spec_channeling` only. See the
[validation claim](../../validation/ledger-channeling.md#cr-zone-incoherent-flux-weight)
for assumptions and checks.

Zone rows are straight mean segments; the omitted angular spread of the
channeled electrons changes zone-row brem by about 2% forward and well under 1%
at large angles for `channeling_demo`
([measured bound](../../validation/ledger-channeling.md#cr-zone-row-angular-spread)).
