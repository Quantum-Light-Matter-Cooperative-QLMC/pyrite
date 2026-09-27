# Validation: `2h-tas2-debye-waller-002`

## Claim and source

- Claim: the catalog's scalar `B_ang2 = 0.53` is a common basal projection for 2H-TaS2 `(002)`, not an isotropic displacement parameter.
- Code: `data/catalog/crystals/2h_tas2.toml::B_ang2`.
- Source: Meetsma *et al.*, *Acta Cryst. C* **46**, 1598--1599 (1990), doi:10.1107/S0108270190000014; COD 9007815.
- Intended quantity and units: one amplitude Debye--Waller coefficient in square angstroms for the reciprocal `c*` direction.

## Independent derivation

The source CIF gives unit-occupancy sites with

$$
U_{33,\mathrm{Ta}}=0.0065(2)\ {\rm \AA^2},\qquad
U_{33,\mathrm{S}}=0.0068(8)\ {\rm \AA^2}.
$$

For two Ta and four S atoms in the conventional cell, equivalently one Ta and two S atoms per formula unit, the declared common-site approximation is

$$
\bar U_{33}=\frac{U_{33,\mathrm{Ta}}+2U_{33,\mathrm{S}}}{3}
           =0.0067000\ {\rm \AA^2}.
$$

With `B = 8 pi^2 U`,

$$
B_{33}=8\pi^2\bar U_{33}=0.52900\ {\rm \AA^2},
$$

which rounds to `0.53 Ang^2`. Independent propagation of the quoted uncertainties gives `sigma(U33) = 0.000537 Ang^2` and `sigma(B33) = 0.0424 Ang^2`; the stored precision is appropriate. The site values differ by `0.0003(8) Ang^2`, well within one combined standard uncertainty.

For basal `(002)`, `g = 4 pi / c`, so the amplitude factor becomes

$$
\exp\left(-\frac{B_{33}g^2}{16\pi^2}\right)
=\exp\left(-\frac{B_{33}}{c^2}\right)
=\exp\left(-\frac{8\pi^2U_{33}}{c^2}\right).
$$

This is the `l = 2` specialization of the standard anisotropic basal projection. The odd `(001)` extinction comes from 2H stacking, not from the Debye--Waller factor.

## Cheap filters

- Units: `U33` and `B33` are in square angstroms; every exponent is dimensionless.
- Limits: `U33 -> 0` gives unit amplitude; increasing `U33` or reflection order suppresses intensity; `(002)` probes only `c*`.
- Signs/conventions: thermal motion attenuates, so the exponent is negative. `B = 8 pi^2 U`; the scalar is a basal projection, not `Biso`.

## Implementation comparison

Inspection after fixing the derivation found:

- the source CIF independently downloaded from COD 9007815 contains `Ta U33=0.00650` and `S U33=0.00680 Ang^2`;
- `crystals/2h_tas2.toml` stores `B_ang2 = 0.53` and pins basal `(002)`;
- `debye_waller` returns `exp[-B (g / 4 pi)^2]`, exactly the independently derived amplitude factor;
- `structure_factor` applies that factor once to each atom.

A separate numeric calculation used only the source cell/basis and `xraydb.f0` (no PyRITE crystallography helper). At 10 keV, changing the common scalar from `0.60` to `0.53 Ang^2` changes `|F002|^2` by `+0.0957151%`. Replacing the site-specific source tensors by the common `0.53 Ang^2` projection changes it by `-0.0235936%`. These reproduce the ledger values (`+0.096%` and `-0.024%`).

Non-physics documentation finding: `data/cifs/2h_tas2.cif` still says the catalog uses the old `0.6 Ang^2` placeholder. Production TOML and its golden snapshot use `0.53 Ang^2`.

Focused verification passed `test_packaged_catalog_matches_independent_serialized_golden`, but the broader Re/Ta surface-contract parameterization still hard-codes `0.6 Ang^2` and fails for 2H-TaS2 (`0.53` obtained). This stale assertion blocks advancing the claim to `anchored`; it does not change the re-derivation.

## Verdict

- **Filters:** units pass; limits pass; signs/conventions pass.
- **Re-derivation:** matches; no divergent factor, sign, unit, or convention.
- **Verdict:** `rederived`.
- **Suggested ledger change:** `unverified -> rederived`; human applies it. Do not mark `anchored` until stale test assertion is reconciled, and never mark `signed-off` without human certification.
