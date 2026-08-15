# Validation: `pdte2-debye-waller-001`

## Claim and source

- Claim: the catalog's scalar `B_ang2 = 0.61` is a common basal projection
  for 1T-PdTe2 `(001)`, not an isotropic displacement parameter.
- Code: `data/materials.toml::crystals.pdte2.B_ang2`.
- Source: Pell, Mironov & Ibers, *Acta Cryst. C* **52**, 1331--1332
  (1996), doi:10.1107/S0108270195016246; COD 2004955.
- Intended quantity and units: one amplitude Debye--Waller coefficient in
  square angstroms for the reciprocal `c*` direction.

## Independent derivation

The source CIF gives unit-occupancy sites with

$$
U_{33,\mathrm{Pd}}=0.0074(4)\ {\rm \AA^2},\qquad
U_{33,\mathrm{Te}}=0.0079(3)\ {\rm \AA^2}.
$$

For one Pd and two Te atoms per formula unit, the declared common-site
approximation is the multiplicity-weighted projection

$$
\bar U_{33}=\frac{U_{33,\mathrm{Pd}}+2U_{33,\mathrm{Te}}}{3}
           =0.0077333\ {\rm \AA^2}.
$$

Using the crystallographic convention `B = 8 pi^2 U`,

$$
B_{33}=8\pi^2\bar U_{33}=0.61062\ {\rm \AA^2},
$$

which rounds to `0.61 Ang^2`. Independent propagation of the quoted
uncertainties gives `sigma(U33) = 0.000240 Ang^2` and
`sigma(B33) = 0.0190 Ang^2`; the stored precision is appropriate. The site
values differ by `0.0005(5) Ang^2`, so a common basal projection is consistent
within one combined standard uncertainty.

For a basal `(00l)` reflection with `g = 2 pi l / c`, the amplitude factor is

$$
\exp\left(-\frac{B_{33}g^2}{16\pi^2}\right)
=\exp\left(-\frac{B_{33}l^2}{4c^2}\right)
=\exp\left(-\frac{2\pi^2U_{33}l^2}{c^2}\right).
$$

This equivalence fixes the direction and rules out interpreting `0.61` as
`U33` or applying a second Debye--Waller factor.

## Cheap filters

- Units: `U33` and `B33` are both in square angstroms; `8 pi^2` is
  dimensionless and the exponent is dimensionless.
- Limits: `U33 -> 0` gives unit amplitude; increasing `U33` or `l` suppresses
  the reflection; `(001)` probes only the `c*` projection.
- Signs/conventions: passive thermal motion requires a negative exponent.
  `B = 8 pi^2 U`, not its inverse. This scalar is direction-specific and must
  not be reused as `Biso` for non-basal reflections.

## Implementation comparison

Inspection after fixing the derivation found:

- the source CIF independently downloaded from COD 2004955 contains
  `Pd U33=0.0074(4)` and `Te U33=0.0079(3) Ang^2`;
- `materials.toml` stores `B_ang2 = 0.61` and pins only basal `(001)`;
- `debye_waller` returns `exp[-B (g / 4 pi)^2]`, exactly the independently
  derived amplitude factor;
- `structure_factor` applies that factor once to each atom.

A separate numeric calculation used only the source cell/basis and `xraydb.f0`
(no PyRITE crystallography helper). At 10 keV, changing the common scalar from
`0.60` to `0.61 Ang^2` changes `|F001|^2` by `-0.0191239%`. Replacing the
site-specific source tensors by the common `0.61 Ang^2` projection changes it
by `-0.0714434%`. These reproduce the ledger values (`-0.019%` and `-0.071%`).

Non-physics documentation finding: `data/cifs/pdte2.cif` still says the
catalog uses the old `0.6 Ang^2` placeholder. Production TOML and its golden
snapshot use `0.61 Ang^2`.

Focused verification passed `test_packaged_catalog_matches_independent_serialized_golden`,
but the broader Pd/Pt surface-contract parameterization still hard-codes
`0.6 Ang^2` and fails for PdTe2 (`0.61` obtained). This stale assertion blocks
advancing the claim to `anchored`; it does not change the re-derivation.

## Verdict

- **Filters:** units pass; limits pass; signs/conventions pass.
- **Re-derivation:** matches; no divergent factor, sign, unit, or convention.
- **Verdict:** `rederived`.
- **Suggested ledger change:** `unverified -> rederived`; human applies it.
  Do not mark `anchored` until stale test assertion is reconciled, and never
  mark `signed-off` without human certification.
