# `mos2-debye-waller-00l`

## Claim and source

- Claim: the catalog's scalar `B_ang2 = 0.47` is a common basal projection for 2H-MoS2 `(00l)`, transferred from the 3R deposit of the same refinement, not an isotropic displacement parameter.
- Code: `data/catalog/crystals/mos2.toml::B_ang2`.
- Source: Schoenfeld, Huang & Moss, "Anisotropic mean-square displacements (MSD) in single crystals of 2H- and 3R-MoS2", *Acta Cryst. B* **39**, 404--407 (1983), doi:10.1107/S0108768183002645; COD 9007660 (2H, the structure this entry cites) and COD 9007661 (3R, same paper).
- Intended quantity and units: one amplitude Debye--Waller coefficient in square angstroms for the reciprocal `c*` direction.

This replaces the shared `0.6 Ang^2` placeholder. The earlier audit pass recorded "no usable ADP" for MoS2 after checking COD 1010993 (Dickinson & Pauling 1923) and COD 1531960 (Petkov 2002, a PDF study); it did not check COD 9007660, which is the record `crystals.mos2.cod_id` already cites and which comes from a paper whose entire subject is the anisotropic MSDs of this material. That deposit omits the ADP loop, but the 3R deposit from the same paper carries it.

## Cross-polytype transfer, and why it is admitted here

The audit rejects substituting data across compounds, and rejected 1T-NbS2 ADPs for the 2H-NbS2 entry. That rejection turns on a coordination change: 1T is octahedral, 2H trigonal-prismatic, so the layers themselves differ and their phonons differ with them.

2H- and 3R-MoS2 are not related that way. Both are built from identical trigonal-prismatic S-Mo-S layers with the same intralayer bonding and the same layer thickness; they differ only in the stacking sequence of those layers (two-layer versus three-layer repeat). The mean-square displacement of an atom within its layer is therefore the same quantity in both, which is why the source paper measures and reports them together. The transfer is recorded here as an explicit scope caveat, not as a silent substitution.

If this is judged too permissive, the fallback is to revert to `0.6 Ang^2` and record "not reported"; the sensitivity table below shows the choice is worth under 1% at the pinned reflections either way.

## Independent derivation

The 3R deposit gives unit-occupancy sites with

$$
U_{33,\mathrm{Mo}}=0.00750\ {\rm \AA^2},\qquad
U_{33,\mathrm{S1}}=U_{33,\mathrm{S2}}=0.00510\ {\rm \AA^2}.
$$

The 2H cell has `Z = 2`, hence 2 Mo and 4 S per cell, so the multiplicity-weighted basal projection is

$$
\bar U_{33}=\frac{2U_{33,\mathrm{Mo}}+4U_{33,\mathrm{S}}}{6}
           =\frac{2(0.00750)+4(0.00510)}{6}=0.00590\ {\rm \AA^2}.
$$

Using `B = 8 pi^2 U`,

$$
B_{33}=8\pi^2\bar U_{33}=0.46585\ {\rm \AA^2},
$$

which rounds to `0.47 Ang^2`. The source quotes no uncertainties.

The Mo and S projections differ by `0.0024 Ang^2`, 41% of the weighted mean, which is a larger site-to-site spread than PdTe2 or 2H-TaS2 tolerated. It is admissible here only because the pinned reflections are weakly attenuated: at `|g| = 1.02 1/Ang` the amplitude exponent is `B g^2 / 16 pi^2 = 0.0031`, so even a 41% error in one site's `B` perturbs its amplitude by about 0.1%. A material with a shorter `c` axis would not get this pass.

In-layer anisotropy is `U33 / U11 = 1.56` for Mo and `1.34` for S -- much weaker than graphite's 5.3, as expected for the more strongly bound Mo-S layers. The scalar remains direction-specific: `(00l)` only.

For a basal `(00l)` reflection with `g = 2 pi l / c`, the amplitude factor is

$$
\exp\left(-\frac{B_{33}g^2}{16\pi^2}\right)
=\exp\left(-\frac{2\pi^2U_{33}l^2}{c^2}\right).
$$

## Cheap filters

- Units: `U33` and `B33` are both in square angstroms; the exponent is dimensionless.
- Limits: `U33 -> 0` gives unit amplitude; increasing `U33` or `l` suppresses the reflection; `(00l)` probes only the `c*` projection.
- Signs/conventions: negative exponent; `B = 8 pi^2 U`; not `Biso`.

## Pinned-reflection sensitivity

At 10 keV, against the retired `0.6 Ang^2` placeholder:

| reflection | `|g|` [1/Ang] | `|F|^2` change, `0.6 -> 0.47` |
|---|---:|---:|
| `(0,0,2)` | 1.0222 | +0.17% |
| `(0,0,4)` | 2.0443 | +0.69% |

The change is a provenance improvement, not a numerical one. MoS2's long `c = 12.295 Ang` puts the pinned reflections at small `|g|`, where the Debye--Waller factor is nearly flat: spanning the entire physically plausible range `B = 0` to `B = 3.45 Ang^2` moves `|F002|^2` by only 4.5% and `|F004|^2` by 16.7%.

## Verdict

- **Filters:** units pass; limits pass; signs/conventions pass.
- **Re-derivation:** matches; no divergent factor, sign, unit, or convention.
- **Verdict:** `rederived`, conditional on the cross-polytype transfer scoped above being accepted.
- **Suggested ledger change:** record under `debye-waller-catalog-provenance`; human applies any status change and `signed-off` requires human certification.
