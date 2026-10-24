# Adding materials

`pyrite material create crystal|amorphous` registers a validated definition in your catalog. With
bundled selection, it writes only to the user layer (`catalog.user`,
`PYRITE_USER_CATALOG`, or the platform configuration directory). Names cannot
replace bundled or existing user material, crystal, or medium identifiers,
including names that differ only by case. An explicitly selected complete
catalog writes in place and stays independent of the user layer.

Creation does not generate transport tables, fetch datasets, change profile
membership, or assign sample geometry. Use `pyrite material list` and
`pyrite material show NAME` to inspect source (`bundled`, `user`, `catalog`),
elemental number densities in atoms/angstrom³, and bulk density in g/cm³.
`-o json` returns the usual versioned result envelope.

## Amorphous media

Supply exactly one composition representation and an explicit bulk density.
A formula does not determine density:

```bash
# Elemental solid.
pyrite material create amorphous amorphous-carbon --formula C --density-g-cm3 2.0

# Compound.
pyrite material create amorphous silica-glass --formula SiO2 --density-g-cm3 2.2

# Mixed medium; repeat an option for each constituent.
pyrite material create amorphous si-ge-mixture --mass-fraction Si=0.5 --mass-fraction Ge=0.5 \
  --density-g-cm3 3.5

# Atomic fractions or stoichiometric ratios are explicit alternatives.
pyrite material create amorphous si-ge-atoms --atomic-fraction Si=0.5 --atomic-fraction Ge=0.5 \
  --density-g-cm3 3.5
pyrite material create amorphous silica-ratios --atom-count Si=1 --atom-count O=2 \
  --density-g-cm3 2.2
```

All fractions must be finite and positive and sum to one (absolute tolerance
`1e-8`). Atom counts are positive ratios and need not sum to one. Simple ASCII
formulas allow positive integer or decimal counts, such as `Si0.5Ge0.5`.
Parentheses, charges, isotope labels and zero counts are unsupported. Repeated
element symbols in formulas are combined; repeated fraction/count CLI entries
are rejected to avoid accidental duplicates.

Conversion uses the existing transport atomic weights in
`materials/_transport_data.py` and Avogadro's constant. For relative atom counts
`c_i`, the native density is `n_i = rho * N_A * 1e-24 * c_i / sum(c_j * A_j)`;
mass fractions first convert to relative counts `w_i / A_i`. Number densities
are serialized at 15 significant digits so equivalent normalized inputs have
the same representation. This is the inverse of the existing catalog-to-mass
conversion ([validated mass-density convention](../validation/beam-transport/sbethe-material-inputs.md));
it introduces no new stopping-power or mixture model.

Only existing transport elements are accepted. For example, `--formula Fr`
fails with an unsupported-element error and the supported-element list;
`--density-g-cm3 0` fails before registration. Adding an element's interaction
constants or isotopic enrichment is outside this workflow.

Media are stored in `media/`; they have no fictitious crystal or CIF. They can
be used in `Layer`, substrates, filters, and electron transport. The current
primary `Slab` radiator remains crystalline: an amorphous medium does not gain
PXR/CBS physics by being registered. Crystal scan recipes live in `materials/`.

## Crystalline materials

Reuse a configured crystal without altering its cut or physics inputs:

```bash
pyrite material create crystal my-silicon --crystal silicon
```

For a new fully occupied CIF, explicitly supply its formula, phase, one
orientation and isotropic Debye–Waller factor:

```bash
pyrite material create crystal lab-si --cif ./silicon.cif --formula Si --phase diamond \
  --surface-hkl 1 1 1 --b-ang2 0.46
```

The example's physical inputs are illustrative: choose values justified for
your sample. Photon grids belong to numerics and resolve automatically unless
explicitly configured there. Legacy `--photon-grid` arguments are deprecated and ignored.
`--surface-hkl` specifies reciprocal Miller indices; `--beam-uvw` is the
mutually exclusive direct-lattice alternative. An existing-crystal reference
rejects configuration overrides; this workflow does not add an editor for cuts.

Gemmi and the existing catalog validator expand the CIF and derive cell volume,
composition and density. Formula stoichiometry must agree with its sites. CIF
formulas use the catalog's integer-count syntax. Partial occupancy and
unsupported transport elements fail. An optional `--density-g-cm3` checks the
lattice-derived density to relative tolerance `1e-6`; it cannot rescale the CIF.

Omitting `--mosaic-fwhm-deg` retains the existing perfect-crystal policy.
Reflection families use the existing automatic selection policy. There is no
inferred Debye–Waller factor or photon grid. CIF import is not independent
physics validation: imported structures are marked unverified, and no claim is
marked signed off.

The CIF's bytes are captured during build and copied to a SHA-256 filename in
managed `cifs/` storage. The definition survives moving/deleting the original
file. Path traversal and symlinks are rejected. Publication validates the full
candidate catalog first, atomically creates records without overwriting, and
rolls back newly created records/assets on a failed save.

## Python: build, inspect, then save

```python
from pyrite.materials import build_medium, build_crystal, save_material

silica = build_medium("python-silica", formula="SiO2", density_g_cm3=2.2)
print(silica.composition)  # immutable (element, atoms/angstrom³) pairs
print(silica.density_g_cm3)
print(silica.identity)  # physical-definition SHA-256, excludes name
inline_medium = silica.medium  # MediumSpec usable without registration

catalog = save_material(silica)  # explicit persistence, returns MaterialCatalog
assert catalog.media["python-silica"] == inline_medium

crystal = build_crystal("python-si", crystal="silicon")
catalog = save_material(crystal)
```

`build_medium` also accepts `mass_fractions`, `atomic_fractions` or `atom_counts`
dictionaries. `build_crystal` accepts a local `cif`, `formula`, `phase`,
`surface_hkl` or `beam_uvw`, `B_ang2`, optional
`mosaic_fwhm_deg`, and optional density consistency check. `catalog=Path(...)`
selects an independent complete catalog for existing-crystal lookup or save.

Build operations return an immutable `MaterialDefinition` and do not persist
anything. `save_material` returns a fresh immutable `MaterialCatalog` and
invalidates the parse cache. Saving to the currently selected catalog also
refreshes PyRITE's loaded default simulation bindings and compatibility crystal
registry, so a newly registered crystal is available in the same Python
process. Previously held immutable catalog snapshots remain unchanged. An
explicit save to another catalog does not change the selected catalog.

## Tables, remote runs and maintainers

Transport tables remain downstream of composition and structure resolution.
Missing table/data inputs retain the existing actionable preparation errors;
registration alone does not make unsupported interaction data available. See
[Getting started](getting-started.md) for the current table workflow.

`pyrite remote` ships user object TOML and managed CIFs through the same shared
catalog export used for profiles, beams and detectors. Local and remote
resolution use the same normalized inputs. Generated transport tables are not
catalog source. Resolved case composition/structure remains in the existing
checkpoint identity and provenance, rather than depending on a later lookup
of the original authoring file.

Maintainers in an editable source checkout can deliberately publish a bundled
definition with `pyrite-dev material create crystal ...` or
`pyrite-dev material create amorphous ...`. This route ignores the user
layer and external-catalog selection, validates the candidate, and refuses an
installed package without its source checkout. Ordinary CLI/API authoring never
uses this maintainer context.
