# `mos2-debye-waller-00l`

## Claim and source

- Claim: the catalog's scalar `B_ang2 = 0.47` is the common basal (`c`-axis) projection for 2H-MoS2, from the 2H row of the source refinement. It is not an isotropic displacement parameter.
- Code: `data/catalog/crystals/mos2.toml::B_ang2`.
- Source: Schoenfeld, Huang & Moss, "Anisotropic mean-square displacements (MSD) in single crystals of 2H- and 3R-MoS2", *Acta Cryst. B* **39**, 404--407 (1983), doi:10.1107/S0108768183002645, Table 3 (2H-MoS2, synthetic crystal, X-ray, room temperature). COD 9007660 (2H, the structure this entry cites) omits the ADP loop, but the paper prints the 2H values. Zotero `MUKTID3R`.
- Intended quantity and units: one amplitude Debye--Waller coefficient in square angstroms for the reciprocal `c*` direction.

This replaces the shared `0.6 Ang^2` placeholder.

**Revised 2026-11-18 (#1302).** An earlier version derived the value from the 3R row (COD 9007661) on the belief that 2H data were unpublished. Table 3 of the same paper reports 2H directly, so no polytype transfer is needed. The stored value is unchanged.

**Superseded preliminary source.** J.-J. Huang's 1981 MS thesis (Univ. of Houston, hdl:10657/17304, Zotero `5PINURE8`) fitted nine `(00l)` reflections of a synthetic 2H crystal. It reported `B_Mo = 1.06`, `B_S = 0.64 Ang^2` along `c` (R = 3.7%) and an initial in-plane estimate of `0.5 Ang^2` for both atoms from limited transmission data on a natural sample. The 1983 paper, by the same group, re-measured `00.6`--`00.34` with absorption, Lorentz--polarization and TDS corrections (R = 1.9%) and cites the thesis only for odd-`l` contamination in natural samples. The thesis values are about 1.7x larger and are not used.

## Independent derivation

Table 3, 2H-MoS2:

| | `B_perp` (along `c`) [Ang^2] | `<u^2>_perp` [1e-3 Ang^2] | `B_par` (in-plane) [Ang^2] | `<u^2>_par` [1e-3 Ang^2] |
|---|---:|---:|---:|---:|
| Mo | 0.61 (TDS-corrected 0.62) | 7.7 | 0.38 (0.40) | 4.8 |
| S | 0.39 (0.40) | 4.9 | 0.30 (0.32) | 3.8 |

The 2H cell has `Z = 2`, hence 2 Mo and 4 S per cell. The multiplicity-weighted basal projection is

$$
\bar B_{33}=\frac{2B^{\perp}_{\mathrm{Mo}}+4B^{\perp}_{\mathrm{S}}}{6}
=\frac{2(0.62)+4(0.40)}{6}=0.473\ {\rm \AA^2}
$$

from the TDS-corrected values, rounded to `0.47 Ang^2`. The uncorrected values give `0.463`, and the earlier 3R route gave `0.466`. The source states that the TDS correction does not change `<u^2>` within error, and it quotes no uncertainties.

The Mo and S projections differ by 0.22 Ang^2, 47% of the weighted mean. This is admissible only because the `(00l)` reflections are weakly attenuated: at `|g| = 1.02 1/Ang` the amplitude exponent is `B g^2 / 16 pi^2 = 0.0031`.

In-layer anisotropy is `B_perp / B_par = 1.6` for Mo and 1.3 for S, as the source reports. The in-plane weighted mean is `(2(0.38)+4(0.30))/6 = 0.33 Ang^2`. The scalar is direction-specific: valid for `(00l)` only.

**Scope warning.** MoS2 is not pinned, so automatic selection applies this `c`-axis scalar to every reflection. For in-plane reflections it over-damps intensity relative to `B_par ~= 0.33` by about 3% at `(110)` (`|g| ~= 4 1/Ang`), 19% at `|g| = 10 1/Ang`, and 2x at `|g| = 20 1/Ang`, where MoS2 selection stops. The per-site uniaxial values above are already transcribed for channeling (`data/channeling_thermal.toml`, `cr-uniaxial-thermal-projection`). Catalog PXR/CBS scalars have no tensor path yet; that is tracked in #6.

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
- **Verdict:** `rederived` for `(00l)`; the scalar is not valid for in-plane or oblique `g` (scope warning above).
- **Suggested ledger change:** record under `debye-waller-catalog-provenance`; human applies any status change and `signed-off` requires human certification.
