# 2H Niobium Dichalcogenides Design

## Goal

Add bulk 2H-NbS2 and 2H-NbSe2 as complete, runnable cxr-mc materials. The
catalog keys will be `nbs2` and `nbse2`, and both will model the metallic
2H-a polytype in space group P6_3/mmc rather than reusing the 2H-c stacking of
the existing Mo/W dichalcogenides.

In2Se3 is deliberately outside this change. It is a promising later material,
but its product metadata does not unambiguously identify the 2H versus 3R
alpha polytype and therefore is not ready for a sourced production entry.

## Polytype Choice

Both 2H and 3R NbS2 are experimentally established, while bulk 2H-NbSe2 is
the common, well-established NbSe2 phase. The 2H polytype is selected for both
new materials.

For basal reflections, 2H `(002)` and 3R `(003)` have identical plane spacing
and structure-factor density, so the larger 3R cell does not itself increase
PXR or CBS coupling. A representative cxr-mc comparison using the four
automatically ranked reflection families, a 30 keV straight segment, the
500--2500 eV band, and a grid of polar/azimuthal orientations favored 2H:

- NbS2: approximately 5--8 percent greater integrated PXR/CBS yield.
- NbSe2: approximately 1.5--2 times greater integrated PXR/CBS yield, mainly
  because more strong symmetry-related non-basal reflections contribute.

This benchmark is a model-based selection aid, not an experimental validation
claim. The exact basal equivalence is the limiting-case check for the stacking
comparison.

## Crystal Structures

Each conventional hexagonal cell contains two formula units: two Nb atoms and
four chalcogen atoms. The basis will explicitly expand the P6_3/mmc Wyckoff
sites because the TOML loader does not apply space-group symmetry:

- Nb occupies 2b at `(0, 0, 1/4)` and `(0, 0, 3/4)`.
- S or Se occupies 4f, expanded from `(1/3, 2/3, z)`.

The aligned Nb columns distinguish metallic 2H-a stacking from the offset
metal columns in the existing 2H-c Mo/W entries.

The production values will be:

| key | lattice a (angstrom) | lattice c (angstrom) | 4f z |
| --- | ---: | ---: | ---: |
| `nbs2` | 3.320 | 11.970 | 0.113 |
| `nbse2` | 3.4459 | 12.5607 | 0.116 |

For NbS2, the experimental lattice parameters `(a, c) = (3.320 A, 11.970 A)`
come from the supplement to El Youbi et al.; that supplement separately
adopts `z = 0.113` from Heil, Schlipf, and Giustino. For NbSe2, the
room-temperature lattice parameters `(a, c) = (3.4459 A, 12.5607 A)` come
from Wang et al., while `z = 0.116` is the lower endpoint of the experimental
`0.116-0.118` range summarized by Johannes, Mazin, and Howells. The
P6_3/mmc 2b/4f basis is cross-checked against the AFLOW prototype.

Each TOML entry will carry a derivation comment, `Validation:` marker, source,
stoichiometry, positive-volume, finite-coupling, and 2H-a stacking limiting
case. Corresponding rows will be added to the physics validation ledger. The
status will remain `unverified`; only a human may sign off a claim.

## Material and Transport Configuration

`materials.registry.MATERIAL_CONFIGS` will add bulk scan rows for `nbs2` and
`nbse2`:

- labels `NbS2` and `NbSe2`;
- `B_ang2 = 0.6`, matching the current exploratory TMD convention;
- c-axis beam orientation `(0, 0, 2)`;
- automatic `dominant_reflections` selection rather than pinned planes;
- 10 micrometre default bulk thickness;
- the existing 25, 30, and 35 keV beam-energy grid and full polar/azimuthal
  scan used by bulk TMDs;
- line grids wide enough to retain the selected soft-X-ray families and the
  standard wide bremsstrahlung grid.

Niobium will be added once to `TRANSPORT_ELEMENTS` with sourced atomic number,
atomic weight, and mean ionization potential. It has no bundled NIST Mott
transport table, so the existing cached screened-Rutherford fallback will be
used. `Nb` will also be included in the edge-prone set so its soft-X-ray edge
structure is retained when evaluating PXR/CBS couplings.

No new public API or polytype abstraction is needed.

## Tests and Validation

Tests will be written first and will cover:

1. Both keys load with positive cell volume, six basis atoms, and exact 1:2
   Nb-to-chalcogen stoichiometry.
2. Lattice constants, expanded 2b/4f coordinates, aligned Nb columns, and cell
   volumes match the sourced 2H-a structures.
3. Representative structure factors, `chi_g`, and `U_g` are finite; an allowed
   reflection has nonzero coupling.
4. Both materials project into the scan and crystal registries, produce
   positive number-density compositions, select reflections automatically,
   and build runnable cases.
5. Niobium transport parameters are available and missing Mott data takes the
   existing screened-Rutherford path.
6. Focused crystallography and sweep tests pass, followed by Ruff and the full
   repository test suite. Physics-validation ledger consistency is checked as
   part of the documentation review.

The README material catalog will be updated to list the two new 2H-a TMDs.
Unrelated working-tree changes will not be modified.

## In2Se3 Evaluation

Alpha-In2Se3 remains a worthwhile follow-up candidate:

- its estimated electron density, about 1.49 electrons per cubic angstrom,
  lies between NbS2 and NbSe2;
- modeled absorption lengths are approximately 0.36 micrometres at 1 keV,
  0.77 micrometres at 2 keV, and 2.1 micrometres at 3 keV, so absorption is
  competitive with current TMDs;
- large oriented bulk crystals are commercially available.

Its longer quintuple-layer repeat shifts basal emission toward lower photon
energies, which is unfavorable for the Timepix threshold, although non-basal
families may remain useful. Before adding it, the exact purchased alpha phase
must be confirmed from a CIF or vendor XRD: primary crystallography gives
`c = 19.217` angstrom for 2H alpha-In2Se3 and `c = 28.750` angstrom for 3R,
whereas the product page mixes a hexagonal/`002` description with inconsistent
cell units. Once the sample phase is known, In2Se3 should receive its own
structure-factor and detected-yield comparison rather than being inferred from
composition alone.

## Sources

- Leroux et al., *Polytypism and superconductivity in the NbS2 system*,
  Dalton Transactions (2021), https://doi.org/10.1039/D0DT03636F
- El Youbi et al., experimental NbS2 lattice parameters in the supplement,
  Physical Review B 103, 155105 (2021),
  https://doi.org/10.1103/PhysRevB.103.155105
- Heil, Schlipf, and Giustino, adopted NbS2 chalcogen coordinate,
  Physical Review B 98, 075120 (2018),
  https://doi.org/10.1103/PhysRevB.98.075120
- AFLOW prototype `AB2_hP6_194_b_f-002` (NbS2/NbSe2 2b/4f basis),
  https://aflow.org/p/Z5TL/
- Wang et al., room-temperature bulk 2H-NbSe2 lattice refinement,
  https://doi.org/10.1063/5.0172460
- Johannes, Mazin, and Howells, experimental NbSe2 chalcogen-coordinate range,
  Physical Review B 73, 205102 (2006),
  https://doi.org/10.1103/PhysRevB.73.205102
- Popovic et al., *Controlled Crystal Growth of Indium Selenide, In2Se3, and
  the Crystal Structures of alpha-In2Se3*, Inorganic Chemistry (2018),
  https://doi.org/10.1021/acs.inorgchem.8b01950
