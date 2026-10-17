# Planar channeling material support

This survey records the bundled catalog at issue #525. Silicon has the
existing planar channeling anchors. Diamond now has its own effective valence
oscillator; its electronic kernel runs without the former graphite substitute.
Layered crystals remain refused: their thermal inputs and the present isotropic
scattering model do not yet support an end-to-end calculation.

This is a current-model no-go for layered production runs, not evidence that
these materials cannot channel. A lattice/basis, a catalog diffraction `B_ang2`,
or a scalar optical plasmon alone is insufficient.

## Reproduce the inventory

```bash
PYRITE_MC_BACKEND=cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python checks/channeling_material_survey.py > /tmp/channeling-materials.json
PYRITE_MC_BACKEND=cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python checks/quantum_decoherence_budget.py
```

The inventory covers all 49 crystal entries, including LiF, which has no
standalone material preset, and maps all 50 material presets to their crystals.
It calls `planar_profile` for `(100)`, `(010)`, `(001)`, `(110)`, `(111)` and
each catalog face at 293.15 K. It independently calls the electronic oscillator
constructor and the entry-angle mapping at 3 MeV. Each successful profile
reports its true period, well depth and bound-state count at Bloch momentum zero.
These probes do not run transport or certify yield/linewidth accuracy.

Results at #525: only Si and diamond pass the thermal gate; all other 47
crystals raise `ChannelingDataError` before the Fourier sum. Since #535, 2H-MoS₂
also builds static profiles from its room-temperature uniaxial rows. Fourteen crystals have
constructible electronic oscillators, including diamond. Consequently the
inventory cannot establish whether hexagonal extinctions or compound wells are
correct: those code paths remain behind missing thermal data. No artificial
displacements were inserted to bypass that gate.

## Survey conventions

- **Supported:** existing Si planar model, within ADR-0018's run restrictions.
- **Data-gap:** source-backed channeling thermal rows and/or electronic inputs
  are absent. Successful static diamond profiles do not certify its linewidths
  ([#523](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/523)).
- **Model-gap:** the material also needs anisotropic scattering, mosaic
  averaging, a shell partition change, or stack support. Data gaps still apply.
- **T:** missing thermal rows for the listed elements, specific to this crystal.
- **E:** no packaged valence oscillator; free-atom shell GOS infrastructure
  exists but does not supply a measured crystal valence response.
- **A:** all listed layered electronic rows are scalar optical approximations;
  directional loss response is unresolved for a channeling beam.
- **R:** the primary structure refinement linked by the row's COD record, or
  the named structure-ledger record when no COD id exists. R is a candidate for
  site ADPs at its stated temperature, not proof that it contains usable data.
  Obtain a temperature series or polarized phonon calculation if it does not.
- **L:** the entry has no measured mosaic width in the catalog; single-crystal
  quality must be measured on the specimen. No numerical mosaic class is
  inferred from a growth method. **H:** HOPG's declared mosaic is 0.8° FWHM.
- **O:** no material-specific measured planar CR/dechanneling anchor was found
  in the sources read here; a generalized abTEM comparison is still required.
  Existing Si oracle results cannot be transferred to another crystal.

For every row, lattice and basis are available in the CIF-backed catalog;
their provenance and any idealization are in the
[structure ledger](../../validation/ledger-crystal-structure-provenance.md).
The table's face is the crystal's default face, not a claim that its normal is
the channeling-plane normal. Plane reachability is discussed below.

<!-- BEGIN MATERIAL SURVEY -->
| Crystal | Status; missing inputs | Default face; near-normal candidate planes | Mosaic | Candidate sources; validation |
| --- | --- | --- | --- | --- |
| `2h_tas2` | model-gap; T: S, Ta; A | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 9007815](https://www.crystallography.net/cod/9007815.html); Bell–Liang (E); O |
| `2h_tase2` | model-gap; T: Se, Ta; A | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 2310532](https://www.crystallography.net/cod/2310532.html); Bell–Liang (E); O |
| `4h_sic` | model-gap; T: C, Si; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | R: `4h-sic-0004-structure`; E source not established; O |
| `6h_sic` | model-gap; T: C, Si; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 9010158](https://www.crystallography.net/cod/9010158.html); E source not established; O |
| `black_phosphorus` | model-gap; T: P; E | `(0,1,0)`; (0,0,1), (1,0,0) | L | [R: COD 9012486](https://www.crystallography.net/cod/9012486.html); E source not established; O |
| `diamond` | supported | `(4,0,0)`; (0,0,1), (0,1,0) | L | [R: COD 9008564](https://www.crystallography.net/cod/9008564.html); diamond line energies; widths #523 |
| `fes2` | data-gap; T: Fe, S; E | `(2,0,0)`; (0,0,1), (0,1,0) | L | R: `fes2-pyrite-structure`; E source not established; O |
| `fete` | model-gap; T: Fe, Te; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 4102703](https://www.crystallography.net/cod/4102703.html); E source not established; O |
| `gep` | model-gap; T: Ge, P; E | `(1,0,-1)`; (0,1,0) | L | [R: COD 1562070](https://www.crystallography.net/cod/1562070.html); E source not established; O |
| `ges` | model-gap; T: Ge, S; E | `(1,0,0)`; (0,0,1), (0,1,0) | L | [R: COD 8104282](https://www.crystallography.net/cod/8104282.html); E source not established; O |
| `gese` | model-gap; T: Ge, Se; E | `(1,0,0)`; (0,0,1), (0,1,0) | L | [R: COD 4003515](https://www.crystallography.net/cod/4003515.html); E source not established; O |
| `gese2` | model-gap; T: Ge, Se; E | `(0,0,1)`; (0,1,0) | L | R: `gese2-beta-p21c-structure`; E source not established; O |
| `hbn` | model-gap; T: B, N; A | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 9008997](https://www.crystallography.net/cod/9008997.html); Pease (T), Tarrio (E); O |
| `hfs2` | model-gap; T: Hf, S; E; whole-shell partition | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | R: `hfs2-structure`; Bell–Liang (E); shell-model change needed; O |
| `hfse2` | model-gap; T: Hf, Se; E; whole-shell partition | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | R: `hfse2-cif-migration`; Bell–Liang (E); shell-model change needed; O |
| `hfte2` | model-gap; T: Hf, Te; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | R: `hfte2-001-structure`; E source not established; O |
| `hopg` | model-gap; T: C; A; #524 | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | H | [R: COD 9011577](https://www.crystallography.net/cod/9011577.html); Trucano/Chen (T), Taft (E); O |
| `lif` | data-gap; T: F, Li; E | `(1,0,0)`; (0,0,1), (0,1,0) | L | [R: COD 1010990](https://www.crystallography.net/cod/1010990.html); E source not established; O |
| `mos2` | model-gap; 295 ± 10 K uniaxial rows (static profile and anisotropic TDS); A | `(0,0,2)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 9007660](https://www.crystallography.net/cod/9007660.html); Schönfeld/Feldman (T), Moynihan (E); O |
| `mose2` | model-gap; T: Mo, Se; A | `(0,0,2)`; (0,1,0), (1,0,0), (1,1,0) | L | R: `mose2-cif-migration`; Bell–Liang (E); O |
| `mote2` | model-gap; T: Mo, Te; A | `(0,0,2)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 2310465](https://www.crystallography.net/cod/2310465.html); Bell–Liang (E); O |
| `mote2_product` | model-gap; T: Mo, Te; E | `(0,0,2)`; (0,1,0), (1,0,0), (1,1,0) | L | R: `mote2-product-structure`; E source not established; O |
| `nbs2` | model-gap; T: Nb, S; A | `(0,0,2)`; (0,1,0), (1,0,0), (1,1,0) | L | R: `nbs2-2ha-structure`; Bell–Liang (E); O |
| `nbse2` | model-gap; T: Nb, Se; A | `(0,0,2)`; (0,1,0), (1,0,0), (1,1,0) | L | R: `nbse2-2ha-structure`; Bell–Liang (E); O |
| `nbte2` | model-gap; T: Nb, Te; E | `(0,0,1)`; (0,1,0) | L | [R: COD 2310357](https://www.crystallography.net/cod/2310357.html); E source not established; O |
| `pds2` | model-gap; T: Pd, S; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 2310589](https://www.crystallography.net/cod/2310589.html); E source not established; O |
| `pdse2` | model-gap; T: Pd, Se; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 4310736](https://www.crystallography.net/cod/4310736.html); E source not established; O |
| `pdte2` | model-gap; T: Pd, Te; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 2004955](https://www.crystallography.net/cod/2004955.html); E source not established; O |
| `ptbi2` | model-gap; T: Bi, Pt; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | R: `ptbi2-p31m-structure`; E source not established; O |
| `pts2` | model-gap; T: Pt, S; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 1537200](https://www.crystallography.net/cod/1537200.html); E source not established; O |
| `ptse2` | model-gap; T: Pt, Se; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 1539198](https://www.crystallography.net/cod/1539198.html); E source not established; O |
| `ptte2` | model-gap; T: Pt, Te; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 1537197](https://www.crystallography.net/cod/1537197.html); E source not established; O |
| `res2` | model-gap; T: Re, S; E | `(0,0,1)`; independent cut/azimuth | L | R: `res2-mp572758-structure`; E source not established; O |
| `rese2` | model-gap; T: Re, Se; E | `(0,0,1)`; independent cut/azimuth | L | [R: COD 1539529](https://www.crystallography.net/cod/1539529.html); E source not established; O |
| `sapphire` | model-gap; T: Al, O; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 9008081](https://www.crystallography.net/cod/9008081.html); E source not established; O |
| `silicon` | supported | `(4,4,0)`; (0,0,1) | L | [R: COD 9011998](https://www.crystallography.net/cod/9011998.html); Si anchors |
| `tate2` | model-gap; T: Ta, Te; E | `(0,0,1)`; (0,1,0) | L | [R: COD 2310358](https://www.crystallography.net/cod/2310358.html); E source not established; O |
| `tis2` | model-gap; T: S, Ti; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 1537865](https://www.crystallography.net/cod/1537865.html); E source not established; O |
| `tise2` | model-gap; T: Se, Ti; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 1537860](https://www.crystallography.net/cod/1537860.html); E source not established; O |
| `tite2` | model-gap; T: Te, Ti; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 1008063](https://www.crystallography.net/cod/1008063.html); E source not established; O |
| `v2o5` | model-gap; T: O, V; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 2020756](https://www.crystallography.net/cod/2020756.html); E source not established; O |
| `vse2` | model-gap; T: Se, V; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 2107137](https://www.crystallography.net/cod/2107137.html); E source not established; O |
| `vte2` | model-gap; T: Te, V; E | `(0,0,1)`; (0,1,0) | L | [R: COD 1535594](https://www.crystallography.net/cod/1535594.html); E source not established; O |
| `ws2` | model-gap; T: S, W; A | `(0,0,2)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 9012191](https://www.crystallography.net/cod/9012191.html); Bell–Liang (E); O |
| `wse2` | model-gap; T: Se, W; A | `(0,0,2)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 9012193](https://www.crystallography.net/cod/9012193.html); Schutte (T), Ahmad (E); O |
| `wte2` | model-gap; T: Te, W; E | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | [R: COD 2310355](https://www.crystallography.net/cod/2310355.html); E source not established; O |
| `zrse2` | model-gap; T: Se, Zr; A | `(0,0,1)`; (0,1,0), (1,0,0), (1,1,0) | L | R: `zrse2-cif-migration`; Bell–Liang (E); O |
| `zrte3` | model-gap; T: Te, Zr; E | `(0,0,1)`; (0,1,0) | L | [R: COD 1559502](https://www.crystallography.net/cod/1559502.html); E source not established; O |
| `zrte5` | model-gap; T: Te, Zr; E | `(0,1,0)`; (0,0,1), (1,0,0) | L | R: `zrte5-cmcm-structure`; E source not established; O |
<!-- END MATERIAL SURVEY -->

The `mos2-on-sapphire` and `mos2-on-sio2-si` presets reuse the MoS₂ crystal and
add substrates/layers; they are **model-gap**, since channeling stacks remain
refused. `mote2_product` also has a sapphire substrate and a separate structural
entry, so its thermal or electronic parameters must not silently inherit those
of bulk `mote2`.

## Layered candidates and sources

### Graphite and HOPG

Trucano and Chen's neutron refinement {cite:p}`trucano1975` refines only
U₃₃ (0.016–0.017 Å² at room temperature). It fixes the basal U₁₁ = 0.0031 Å²
from Ludsteck (1972), and its two carbon sites do not differ significantly.
Chen and Trucano's comparison of diffraction and phonon-spectrum thermal motion
{cite:p}`chen1978` has not been read.
The [existing basal diffraction record](../../validation/materials/hopg-debye-waller-00l.md)
already distinguishes two carbon sites with different normal displacements.
The channeling thermal loader currently groups by element, so a single C row
also erases that distinction. Neither the diffraction scalar nor diamond's
isotropic C row supplies graphite's temperature dependence.

The packaged electronic row uses Taft and Philipp's in-plane optical response
(DOI `10.1103/PhysRev.138.A197`); it represents the combined valence loss by one
effective oscillator and omits the separate low-energy π plasmon. The current
radial electronic kernel has no directional dielectric input. Basal channeling
places the plane-normal transfer along the interlayer direction, so the
in-plane source cannot be presented as a measured basal-channeling response.

HOPG additionally requires mosaic quadrature
([#524](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/524)).
Its declared 0.8° is about 14 mrad FWHM; compare a computed Lindhard angle for
the chosen energy and plane, using consistent FWHM/RMS conventions. No critical
angle is claimed for a profile that cannot yet be built. Kish/natural graphite
would need a distinct specimen declaration with measured spread; clearing
HOPG's mosaic field would misrepresent the sample.

### hBN

Pease's refinement is the starting structural/thermal source
candidate{cite:p}`pease1952`; the available Zotero item `MFWKLT5G` has no
attachment. A lattice thermal-expansion curve alone cannot replace atomic
displacement data. Site B and N displacements, their temperature range and
directional convention remain to be read and transcribed.

The existing Tarrio–Schnatterly electronic row (DOI
`10.1103/PhysRevB.40.7852`) explicitly distinguishes in-plane and out-of-plane
losses, but the implementation stores one in-plane effective resonance.
High-pressure-grown single crystals are candidates for a narrow rocking curve,
not an assumed mosaic measurement. An exfoliated flake makes normal incidence
easy but basal edge entry difficult; finite lateral dimensions matter.

### MoS₂, WSe₂ and other dichalcogenides

Schönfeld, Huang and Moss measured anisotropic MSDs in 2H- and 3R-MoS₂ at room
temperature {cite:p}`schoenfeld1983`. Their Table 3 2H values (⟨u²⟩ in Å²:
Mo 0.0048 basal and 0.0077 along c; S 0.0038 and 0.0049) are now packaged as
uniaxial rows valid at 295 ± 10 K (#535, `cr-uniaxial-thermal-projection`).
These rows give the static planar potential: `(001)` has a `c/2` period with
S–Mo–S sandwich wells, while `(100)` and `(110)` (period `a/2`) use the basal
displacement. Well depths at 295 K are 66.4 eV `(001)`, 28.8 eV `(100)`/`(110)`
and 7.8 eV `(101)`, converged to 0.01 eV by `s ≤ 4 Å⁻¹`. The basal well binds
33, 65 and 129 states at 40, 160 and 640 MeV (`√γ` scaling), so its basis must
exceed the bound count: the default 32 plane-wave orders undercount above
~40 MeV.
The TDS kernel rotates their full covariance into the plane-normal/beam frame
(#535, `cr-lindblad-jump-operators`); a beam direction is required off the
basal planes. Feldman and Boyer's lattice-dynamics MSDs {cite:p}`feldman1981`
give their temperature dependence only as a figure, and their room-temperature
ordering along c (Mo below S) contradicts the measurement. They are therefore
not used, and no temperature series exists. The WSe₂ refinement (COD 9012193;
DOI `10.1016/0022-4596(87)90057-0`) publishes no thermal parameters, so the
W/Se gap remains open.

MoS₂ and WSe₂ already have sourced scalar electronic rows (DOIs
`10.1111/jmi.12900`, `10.1088/1361-648X/aa63a7`). Bell and Liang supply the
existing MoSe₂, MoTe₂, WS₂, NbS₂, NbSe₂, TaS₂, TaSe₂ and ZrSe₂ rows and are
also a candidate for missing Hf/Zr-compound losses{cite:p}`bell1976`.
Their experiment used beam incidence along c and momentum transfer in the
basal plane; this directional scope is explicit in the paper, not a general
isotropic channeling measurement. WS₂'s stored row also retains its 3R-to-2H
polytype caveat. HfS₂/HfSe₂ have a separate whole-shell partition problem:
Hf 4f prevents the current model from consuming the chemical valence as whole
outermost shells; supplying a plasmon energy alone will not fix that.

CVT bulk crystals and exfoliated flakes are candidate specimens. Require a
measured rocking curve and azimuth, adequate edge thickness, and finite-size
checks. TMD X–M–X sandwiches need the complete basis and the true Fourier
period; a layer spacing is not automatically the channel width. After thermal
support exists, check basal and both prismatic directions, extinction order,
well locations, bound states per well and convergence with harmonic cutoff.

## Reachable planes and sample cuts

For a hexagonal `(0001)` face and a beam near its normal, the beam crosses the
basal layers; it does not run within a basal channel. The prismatic families
`{10-10}` and `{11-20}` contain c and are geometrically accessible. The code
uses three-index planes `(100)` and `(110)` for those representatives.
Choose an in-plane azimuth away from low-index axes; normal c-axis incidence
alone is an axial configuration, outside the planar claim.

Basal channeling needs an edge face whose normal lies in the layers, with an
appropriately chosen beam azimuth. Tilting a basal-face slab to exactly grazing
incidence is not an equivalent finite-sample model: the beam must still enter
through the declared face. Use `CrystalCut` and `plane_incidence` with the
reciprocal metric, particularly for monoclinic/triclinic entries. For each face
in the table, a candidate plane is accessible at near-normal entry only if its
normal is nearly perpendicular to the face normal. The inventory records the
actual geometric angle; missing thermal data prevents a critical-angle test.

Black phosphorus and ZrTe₅ illustrate why `(001)` is not universally the
layer normal: their catalog faces are `(010)`. Cubic Si/diamond samples can use
an independently declared cut to reach `(110)` or `(111)` channels; their
default face should not be mistaken for that channeling alignment.

## Thermal schema decision and current-model no-go

The decision recorded in [ADR-0018](../../adr/0018-quantum-channeling-zone.md)
is to retain existing isotropic rows for cubic Si/diamond and require a
site-resolved displacement covariance for layered support. A future row needs
a site identifier tied to the CIF basis, temperature samples, a declared
Cartesian frame and covariance units, plus source/uncertainty and interpolation
scope. CIF crystallographic tensor components must be converted using the cell
metric before rotation. A single-temperature refinement must not acquire an
invented Debye curve.

Projecting the covariance onto a selected plane suffices for the static
potential but not for TDS: `operators._tds_strength` and `_tds_diagonal`
integrate transfer in two transverse directions using one isotropic variance.
The plane-local covariance and beam azimuth must reach both scattering terms,
including the full-loss diagonal, while preserving positive jump kernels and
the isotropic limit. The radial electronic kernel independently needs a
declared treatment of anisotropic loss response. These changes need new
source-to-code verification under `cr-debye-b-of-t`,
`cr-planar-continuum-potential` and `cr-lindblad-jump-operators`.

Thus **HOPG, hBN, MoS₂ and WSe₂ are currently a documented no-go for sourced
end-to-end planar runs**. For graphite/MoS₂ the anisotropic displacement
literature is directly relevant; Bell–Liang's directional loss measurement
independently limits reuse of scalar electronic data. This conclusion does
not label the missing tensor model as implemented or advance any ledger status
for hexagonal/multi-element channeling. #535 later admitted 2H-MoS₂ uniaxial
rows (`cr-uniaxial-thermal-projection`, `anchored`) and carried their
plane-local covariance and beam azimuth into the TDS jump kernel, full-loss
diagonal and `Γ(Δ)`, with cross terms (`cr-lindblad-jump-operators`;
[fresh re-derivation](../../validation/channeling/cr-lindblad-jump-operators-anisotropic.md)
completed). The electronic kernel is still radial, and the only MoS₂ loss-function
source is qualitative, so end-to-end MoS₂ runs stay refused until #536 supplies
a directional kernel. Thermal inputs for hBN, WSe₂ and graphite are tracked in
[#540](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/540).

The abTEM harness now constructs a general catalog-basis supercell with the
profile's covariances and an explicit in-plane beam. The
[MoS₂ basal comparison](../../validation/channeling/cr-density-matrix-split-operator-mos2.md)
records a maximum grouped-population discrepancy of 0.00274 over 1 µm,
including beam, grid, slice and solver convergence checks. It records the
remaining discrepancy without asserting a new pass threshold or a measured
CR anchor. Thermal/potential review and directional electronic response
remain outstanding support gates.

## Diamond electronic input and refreshed budget

The undoped control in Bhattacharya et al. has a volume plasmon near 34 eV;
its boron-doped sub-eV intervalence excitation is a different signal
{cite:p}`bhattacharya2025`. `conduction_band.toml` now assigns the four carbon
valence electrons to that effective oscillator and retains two K-shell
electrons. This uses the existing PENELOPE shell model; it is not a measured
full GOS or a linewidth validation. See
[independent verification](../../validation/channeling/diamond-valence-oscillator.md).

Re-running `quantum_decoherence_budget.py` at 293.15 K without any oscillator
override gives:

| Diamond (110) kinetic energy | Γ electronic at 1 Å (Å⁻¹) | ℓ(d/2) (µm) | ℓ(1 Å) (µm) |
| --- | --- | --- | --- |
| 30 keV | 1.52e-3 | 0.0831 | 0.0573 |
| 300 keV | 2.75e-4 | 0.461 | 0.318 |
| 3 MeV | 1.69e-4 | 0.748 | 0.516 |
| 30 MeV | 1.66e-4 | 0.764 | 0.527 |

The zero-separation, TDS saturation and TDS inverse-velocity-squared checks
pass. These position-coherence lengths are not bound-state line coherence
lengths. Measured diamond linewidths and their residual budget remain #523.

A bounded public-API smoke run also produces a finite, nonzero channeling
spectrum for diamond: 9 MeV, `(110)` channels, `(001)` face, 3° tilt at 315°
azimuth, 500 Å thickness, four electron trials per component, eight reciprocal
orders, four Bloch cells and 100 Å depth steps. This checks the `simulate`
integration; those coarse numerics are not a converged experimental prediction.

## Source-reading scope

The Bell–Liang paper was read from Zotero attachment `YGCBVDRV` (parent
`D2EF5UG7`), including its experimental geometry and Table 4. Its local API
had no indexed text; the stored PDF was read directly. Pease's item was found
but had no attachment. Publisher full texts were authorized by the maintainer
for this issue; the diamond value comes from Bhattacharya's publisher Results
section. Other listed papers/refinements are candidate acquisition sources
unless an existing repository validation record already documents a reading.
No new anisotropic number is transcribed from an abstract or search snippet.

## Tracked extensions

The current no-go has two independently owned extensions:
[site thermal tensors, TDS and layered oracle (#535)](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/535)
and [directional electronic loss and remaining inputs (#536)](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/536).
The catalog diffraction audit remains #6; HOPG mosaic support remains #524.
No tensor thermal claim or hexagonal potential claim is promoted by this survey.
