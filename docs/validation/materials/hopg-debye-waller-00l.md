# Validation: `hopg-debye-waller-00l`

## Claim and source

- Claim: the catalog's scalar `B_ang2 = 1.30` is a common basal projection for 2H graphite `(00l)`, not an isotropic displacement parameter.
- Code: `data/materials.toml::crystals.hopg.B_ang2`.
- Source: Trucano & Chen, "Structure of graphite by neutron diffraction", *Nature* **258**, 136--137 (1975), doi:10.1038/258136a0; COD 9011577.
- Intended quantity and units: one amplitude Debye--Waller coefficient in square angstroms for the reciprocal `c*` direction.

This replaces the legacy `0.8 Ang^2`, which carried no value-level provenance. COD 9011577 is the record `crystals.hopg.cod_id` already cites for the structure, so the entry now takes its displacement data from the same primary refinement as its lattice and basis instead of from an unsourced scalar.

## Independent derivation

The source CIF gives two inequivalent unit-occupancy carbon sites, C1 at `(0, 0, 1/4)` (Wyckoff `2b`) and C2 at `(1/3, 2/3, 1/4)` (`2c`), with

$$
U_{11}=U_{22}=0.00310\ {\rm \AA^2}\ \text{(both sites)},\qquad
U_{33,\mathrm{C1}}=0.01600,\quad U_{33,\mathrm{C2}}=0.01700\ {\rm \AA^2}.
$$

Both sites have multiplicity 2 in the `P6_3/mmc` cell, so the multiplicity-weighted basal projection is the plain mean

$$
\bar U_{33}=\frac{2U_{33,\mathrm{C1}}+2U_{33,\mathrm{C2}}}{4}
           =\frac{0.01600+0.01700}{2}=0.01650\ {\rm \AA^2}.
$$

Using the crystallographic convention `B = 8 pi^2 U`,

$$
B_{33}=8\pi^2\bar U_{33}=1.30279\ {\rm \AA^2},
$$

which rounds to `1.30 Ang^2`. The two site values differ by `0.0010 Ang^2`, 6.1% of the mean, so a common basal projection costs at most half that in either site's amplitude; the source quotes no uncertainties, so no propagated sigma is recorded.

The strong anisotropy is the physically expected one for a layered crystal: `U33 / U11 = 5.3`, the soft interlayer modes along `c`. That is exactly why this value is scoped to `(00l)` and must not be reused as `Biso`.

For a basal `(00l)` reflection with `g = 2 pi l / c`, the amplitude factor is

$$
\exp\left(-\frac{B_{33}g^2}{16\pi^2}\right)
=\exp\left(-\frac{2\pi^2U_{33}l^2}{c^2}\right),
$$

the same equivalence used for `pdte2-debye-waller-001` and `2h-tas2-debye-waller-002`. Since `g || c` for the pinned family, `U33` is the only tensor component the catalog's reflections probe, and the scalar reduction is exact in that direction up to the site-to-site spread above.

## Specimen state

The audit's earlier pass held this value at `0.8 Ang^2` on the grounds that Trucano & Chen refined a natural 2H graphite single crystal, not an HOPG specimen. That objection does not apply to the displacement parameter: a mean-square displacement is a lattice-dynamical (phonon) property of the graphite layer stack, whereas HOPG differs from a natural single crystal in mosaic spread and grain size. PyRITE models the mosaic separately and explicitly, through `crystals.hopg.mosaic_fwhm_deg = 0.8`, so adopting the refined `U33` does not double-count any specimen effect. Freund, Munkholm & Brennan's 1996 HOPG diffraction model, cited in the audit as the alternative, uses an *assumed* scalar rather than a refined HOPG ADP, so it is not a competing measurement.

## Cheap filters

- Units: `U33` and `B33` are both in square angstroms; `8 pi^2` is dimensionless and the exponent is dimensionless.
- Limits: `U33 -> 0` gives unit amplitude; increasing `U33` or `l` suppresses the reflection; `(00l)` probes only the `c*` projection.
- Signs/conventions: passive thermal motion requires a negative exponent. `B = 8 pi^2 U`, not its inverse. Direction-specific; not `Biso`.

## Pinned-reflection sensitivity

At 10 keV, against the retired `0.8 Ang^2`:

| reflection | `|g|` [1/Ang] | `|F|^2` change, `0.8 -> 1.30` |
|---|---:|---:|
| `(0,0,2)` | 1.8725 | -2.20% |
| `(0,0,4)` | 3.7450 | -8.50% |

Both pinned families move, `(004)` appreciably, so this is a production-visible change. It propagates to the Zhai anchor, whose crystal is `hopg`; see the `zhai-hbn-921-detected` ledger row.

The serialized golden moves accordingly: `hopg` `structure_factor` for `(002)` goes `16.99996 -> 16.81227` (amplitude), regenerated through `pyrite-dev regen-golden`. `B_ang2` is hashed into `case_content_key`, so the pinned digest in `tests/materials/test_profiles.py` moved once, orphaning CAS records minted under `0.8` rather than serving them for a case that now diffracts differently.

## Verdict

- **Filters:** units pass; limits pass; signs/conventions pass.
- **Re-derivation:** matches; no divergent factor, sign, unit, or convention.
- **Verdict:** `rederived`.
- **Suggested ledger change:** record under `debye-waller-catalog-provenance`; human applies any status change and `signed-off` requires human certification.
