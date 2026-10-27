# Materials and crystallography

Work with material catalogs, authoring, atomic form factors, crystallography and attenuation. See [material authoring](../material-authoring.md).

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.fetch_tables
   pyrite.materials.MaterialCatalog
   pyrite.materials.MaterialConfigError
   pyrite.materials.MaterialIdentity
   pyrite.materials.CrystalInfo
   pyrite.materials.CrystalSpec
   pyrite.materials.MediumSpec
   pyrite.materials.MaterialSpec
   pyrite.materials.MaterialValidationSpec
   pyrite.materials.ScanSpec
   pyrite.materials.LayerSpec
   pyrite.materials.load_material_catalog
   pyrite.materials.atomic.cromer_mann_f0
   pyrite.materials.atomic.henke_dispersion
   pyrite.materials.atomic.atomic_form_factor
   pyrite.materials.crystal.reciprocal_g_vector
   pyrite.materials.crystal.structure_factor
   pyrite.materials.crystal.chi_g
   pyrite.materials.crystal.U_g
   pyrite.materials.crystal.absorption_length_ang
   pyrite.materials.crystal.dominant_reflections
   pyrite.materials.attenuation.linear_attenuation_inv_mm
   pyrite.materials.MaterialDefinition
   pyrite.materials.build_medium
   pyrite.materials.build_crystal
   pyrite.materials.save_material
```

## Material authoring

Use `pyrite.materials.MaterialDefinition`, `build_medium`, `build_crystal` and
`save_material` to author materials. Builders validate without registration;
`save_material` persists and returns a fresh immutable catalog. Density is in
g/cm³, composition in atoms/Å³ and photon grids in eV. See
[Material authoring](../../reference/material-authoring.md) for inputs and storage rules.

The provisional `pyrite.materials.optical.OpticalResponse.to_record()` and
`OpticalResponse.from_record(record)` preserve explicit optical response
arrays, disjoint supported bands and provenance through JSON round trips.
They validate the existing content digest without reloading source data.
These response records remain outside the supported API contract and do not
enable optical production transport; see the
[proposed Cherenkov integration boundary](../../research/physics/production-cherenkov.md#identity-and-portable-response-records).

`pyrite.materials.CATALOG` is the immutable bundled `MaterialCatalog`.
`CRYSTALS`, `MATERIALS` and `MATERIAL_LABELS` are compatibility projections.
A material's display label derives from its `MaterialIdentity`; its cut names
the slab normal, independently of the reflections selected for scoring.
