# Debye-Waller provenance and anisotropy validation audit

This audit owns provenance and model-scope decisions for catalog `B_ang2` values. It does not treat a plausible numerical value, a database default, or an uncited composition average as validated thermal-displacement data.

## Baseline

Snapshot: 2026-07-25, 48 crystal entries in `src/pyrite/data/materials.toml`.

| state | count | crystals |
|---|---:|---|
| primary refinement, temperature, and conversion recorded | 6 | h-BN, GeP, GeS, GeSe, PdTe2, 2H-TaS2 |
| approximate literature range only | 1 | sapphire |
| distinct value without value-level provenance | 4 | diamond, silicon, LiF, HOPG |
| shared `B_ang2 = 0.6 Å²` requiring source recovery or replacement | 37 | all remaining entries |

The 37 repeated values span different elements, phases, bonding, and source temperatures. Several bundled CIF comments explicitly call `0.6 Å²` a modeling placeholder. Until each entry is checked, the repeated value is an unverified catalog default, not a material property.

Existing recoverable records:

- h-BN: Pease room-temperature basal-reflection fit, `B33 = 3.45 Å²`; direction-specific approximation for pinned `(002)/(004)`, not `Biso`.
- GeP: composition-average `B = 0.35 Å²` from a 90 K refinement.
- GeS: `B = 1.07 Å²` derived from reported Ge/S isotropic values.
- GeSe: `B = 1.10 Å²` derived from Ueq values in the selected 250.72 K refinement.
- PdTe2: `B33 = 0.61 Å²` from the multiplicity-weighted mean of Pd `U33 = 0.0074(4) Å²` and Te `U33 = 0.0079(3) Å²` at 113(2) K.
- 2H-TaS2: `B33 = 0.53 Å²` from the multiplicity-weighted mean of Ta `U33 = 0.0065(2) Å²` and S `U33 = 0.0068(8) Å²` at 295 K.

These remain `unverified` ledger evidence until independently reproduced.

## Standard-profile production subset

This pass is restricted to the 21 materials explicitly listed by `[profiles.standard]` in the packaged catalog. Materials in the separate `high_energy` profile and materials outside `standard` are excluded.

### Scalar replacements

| material | source record | reported ADP | catalog reduction | pinned-reflection sensitivity |
|---|---|---|---|---|
| PdTe2, 1T `P-3m1` | Pell, Mironov & Ibers, *Acta Cryst. C* **52**, 1331–1332 (1996), 113(2) K, [doi:10.1107/S0108270195016246](https://doi.org/10.1107/S0108270195016246), [COD 2004955](https://www.crystallography.net/cod/2004955.html) | Pd `U33=0.0074(4)`, Te `U33=0.0079(3) Å²`; full anisotropic tensors; unit occupancies | values agree within combined uncertainty; multiplicity-weighted `U33=(0.0074+2×0.0079)/3=0.00773 Å²`, hence `B33=8π²U33=0.6106 Å²`, stored as `0.61 Å²`; basal `(001)` only | at 10 keV, new common scalar changes `|F001|²` by −0.019% from `B=0.6`; it differs by −0.071% from the site-specific tensor result |
| 2H-TaS2, `P63/mmc` | Meetsma *et al.*, *Acta Cryst. C* **46**, 1598–1599 (1990), 295 K, [doi:10.1107/S0108270190000014](https://doi.org/10.1107/S0108270190000014), [COD 9007815](https://www.crystallography.net/cod/9007815.html) | Ta `U11=0.0034(1)`, `U33=0.0065(2)`; S `U11=0.0048(5)`, `U33=0.0068(8) Å²`; unit occupancies | multiplicity-weighted `U33=(0.0065+2×0.0068)/3=0.00670 Å²`, hence `B33=0.5290 Å²`, stored as `0.53 Å²`; basal `(002)` only | at 10 keV, new common scalar changes `|F002|²` by +0.096% from `B=0.6`; it differs by −0.024% from the site-specific tensor result |

Both reductions are direction-specific compatibility values, not `Biso`. They are valid only because each catalog entry pins a basal `00l` family and the refined site projections are close. Do not reuse them for non-basal reflections.

### Recovered ADPs requiring richer representation

| material | phase, temperature, source | recovered displacement data | decision |
|---|---|---|---|
| HOPG | natural 2H graphite, nominal room temperature; Trucano & Chen, *Nature* **258**, 136–137 (1975), [doi:10.1038/258136a0](https://doi.org/10.1038/258136a0), [COD 9011577](https://www.crystallography.net/cod/9011577.html) | inequivalent C sites have `U11=0.00310` and `U33=0.0160/0.0170 Å²`, implying mean `B33=1.30 Å²` | **Adopted 2026-09-13**, superseding the earlier "keep legacy `0.8 Å²`" decision. The specimen-state objection does not apply to an ADP: a mean-square displacement is a lattice-dynamical property, while HOPG differs from a natural single crystal in mosaic spread, which `mosaic_fwhm_deg` models separately. COD 9011577 is the record the entry's `cod_id` already cites, so the value now shares its source with the lattice and basis. Freund, Munkholm & Brennan (1996) uses an assumed scalar, not a refined HOPG ADP, so it is not a competing measurement. Basal `(00l)` only; see [hopg-debye-waller-00l](hopg-debye-waller-00l.md). |
| PdSe2 | ambient Pbca; Kim *et al.*, *Inorg. Chem.* **43**, 1943–1949 (2004), [doi:10.1021/ic0352396](https://doi.org/10.1021/ic0352396), [COD 4310736](https://www.crystallography.net/cod/4310736.html) | Pd `Ueq=0.01341(6)`, Se `Ueq=0.01205(6) Å²`; Pd tensor `(0.00789,0.00765,0.02470)`, Se `(0.00968,0.00888,0.01759) Å²` plus small cross terms; unit occupancies | strong out-of-plane anisotropy and distinct Pd/Se tensors. Keep placeholder pending per-site tensors. |
| PtBi2 | trigonal `P31m`, 295 K coordinate/ADP table; Feng *et al.*, *Nat. Commun.* **10**, 4765 (2019), [doi:10.1038/s41467-019-12805-2](https://doi.org/10.1038/s41467-019-12805-2), Supplementary Tables II–III | `Ueq`: Pt `0.0202(11)`, Bi1 `0.0234(18)`, Bi2 `0.0199(12)`, Bi3 `0.0202(11) Å²`; `U33`: `0.023(3)`, `0.022(5)`, `0.017(3)`, `0.019(2) Å²`; unit occupancies | four site tensors; source also mixes 273 K cell with 295 K coordinate/ADP table. Keep placeholder pending per-site tensors. |
| V2O5 | ambient alpha `Pmmn`, 292 K; Enjalbert & Galy, *Acta Cryst. C* **42**, 1467–1469 (1986), [doi:10.1107/S0108270186091825](https://doi.org/10.1107/S0108270186091825), [COD 2020756](https://www.crystallography.net/cod/2020756.html) | isotropic `U`: V `0.0068(1)`, O1 `0.0153(9)`, O2 `0.0100(8)`, O3 `0.0114(13) Å²`; unit occupancies | site values differ by more than 2×. Keep placeholder pending per-site isotropic support. |
| ZrTe3 | parent `P21/m`, 293 K; Furuseth & Fjellvåg, *Acta Chem. Scand.* **45**, 694–697 (1991), [doi:10.3891/acta.chem.scand.45-0694](https://doi.org/10.3891/acta.chem.scand.45-0694), [COD 1559502](https://www.crystallography.net/cod/1559502.html) | `Ueq`: Zr/Te1 `0.0110`, Te2 `0.0138`, Te3 `0.0145 Å²`; full monoclinic-frame tensors reported; unit occupancies | distinct site tensors and non-orthogonal cell require frame conversion. Keep placeholder pending per-site Cartesian tensors. |

### Source recovery without usable ADPs

Literal `Uiso=0` in legacy COD conversions below means the source record did not carry a refined displacement parameter; it is not evidence for zero atomic motion.

| materials | recovered primary/deposited record | result |
|---|---|---|
| MoS2 (superseded), MoSe2, MoTe2 | COD 1010993/1531960, 2310945, and 2310465 | **Superseded for MoS2 on 2026-09-13**; see the MoS2 row below. MoSe2 (COD 2310945, James & Lavik 1963) and MoTe2 (COD 2310465, Puotinen & Newnham 1961; COD 9009147, Wyckoff 1963 secondary compilation) still contain no usable ADP; converted zero fields rejected. Re-checked 2026-09-13 against the full COD formula listings for `Mo Se2` and `Mo Te2`: no other matching-phase deposit carries displacement data. |
| WS2, WSe2 | Schutte, de Boer & Jellinek (1987), [doi:10.1016/0022-4596(87)90057-0](https://doi.org/10.1016/0022-4596(87)90057-0), COD 9012191/9012193 | matching 2H deposits provide coordinates but no ADPs |
| NbS2, NbSe2 | matching 2H records COD 1538044 and 1539310; Brown & Beerntsen COD 2310533 is 3R, not catalog 2H | no usable matching-phase ADP; 1T-NbS2 COD 7204814 reports site-specific values but is the wrong polytype |
| NbTe2, PdS2, PtS2, PtSe2, PtTe2, 2H-TaSe2, TaTe2, WTe2 | COD 2310357, 2310589, 1537200, 1537202, 1537197, 2310532, 2310358, 2310355 | legacy conversions contain zero placeholders, not reported ADPs |
| HfS2, HfSe2, HfTe2, ReS2, ZrSe2, ZrTe5 | bundled structure is vendor/DFT/secondary-table based or lacks a primary refinement citation with ADPs | no phase-compatible primary ADP recovered in this pass |
| VSe2 | catalog structural sources and COD 1538289 provide no usable ADP | no replacement |

## 2026-09-13 pass: five production-critical entries

Scope requested: HOPG, h-BN, MoS2, MoSe2, MoTe2 only. The rest of the catalog is out of scope for this pass and its placeholders are unchanged.

| material | before | after | source status |
|---|---:|---:|---|
| HOPG | `0.8` (no provenance) | `1.30` | Trucano & Chen 1975 neutron refinement, COD 9011577 — the entry's own `cod_id`. Adopted; see [hopg-debye-waller-00l](hopg-debye-waller-00l.md) |
| h-BN | `3.45` | `3.45` (unchanged) | Pease room-temperature basal fit, already recorded; no action |
| MoS2 | `0.6` (placeholder) | `0.47` | Schoenfeld, Huang & Moss 1983, COD 9007661 (3R deposit of the same paper). Cross-polytype transfer, scoped in [mos2-debye-waller-00l](mos2-debye-waller-00l.md) |
| MoSe2 | `0.6` (placeholder) | `0.6` (unchanged) | **not reported.** Only matching-phase deposit is James & Lavik 1963 (COD 2310945), zeroed `U_iso`. Bronsema, De Boer & Jellinek 1986 (*Z. Anorg. Allg. Chem.* **541**, 15–17, [doi:10.1002/zaac.19865400904](https://doi.org/10.1002/zaac.19865400904)) is the refinement the entry's cell matches (`a=3.289`, `c=12.927`) and is the correct target, but is paywalled and its ADP table was not retrieved |
| MoTe2 | `0.6` (placeholder) | `0.6` (unchanged) | **not reported.** Puotinen & Newnham 1961 (COD 2310465) carries zeroed `U_iso`; Wyckoff 1963 (COD 9009147) is a secondary compilation with no ADPs |

### Why the two unresolved entries cost little

Pinned-reflection sensitivity at 10 keV, `|F|²` relative to the current catalog `B`, computed with `structure_factor` over the pinned basal families:

| material | `|g(002)|` [1/Å] | `(002)` span, `B = 0 … 3.45` | `(004)` span, `B = 0 … 3.45` | `(002)` over `B = 0.3 … 1.5` | `(004)` over `B = 0.3 … 1.5` |
|---|---:|---:|---:|---:|---:|
| HOPG | 1.8725 | 14.2% | 45.8% | 5.2% | 19.2% |
| h-BN | 1.8866 | 14.4% | 46.3% | 5.3% | 19.5% |
| MoS2 | 1.0222 | 4.5% | 16.7% | 1.6% | 6.2% |
| MoSe2 | 0.9721 | 4.0% | 15.2% | 1.4% | 5.6% |
| MoTe2 | 0.9002 | 3.5% | 13.2% | 1.2% | 4.8% |

The last two columns are the honest error bar on a placeholder: `B = 0.3–1.5 Å²` brackets any physically plausible room-temperature value for these materials.

The Mo dichalcogenides have long `c` axes (12.3–14.0 Å), so their pinned `(00l)` reflections sit at small `|g|` where the Debye–Waller factor is nearly flat. Within that realistic band the placeholder costs at most 1.6% on `(002)` and 6.2% on `(004)` — and those are full-band spans, so the error from `0.6 Å²` specifically is smaller still.

HOPG and h-BN are the opposite case: short `c` axes put `(004)` near `|g| = 3.8 1/Å`, where `|F|²` varies by 46% over the full range and ~19% over the realistic band. That is why HOPG was the high-value entry in this set, and why h-BN was already fixed.

Remaining work for MoSe2 and MoTe2 is source retrieval, not analysis: obtain the Bronsema 1986 ADP table for MoSe2, and a modern single-crystal refinement for 2H-MoTe2. Neither is on the critical path for present accuracy.

## Acceptance record

Every replacement must record:

1. material, phase or polytype, specimen state, and measurement temperature;
2. primary diffraction refinement, DOI, table or deposited-record locator;
3. reported convention (`Biso`, `Uiso`/`Ueq`, anisotropic `Uij`, or a reflection-direction fit);
4. atom/site association, occupancies, uncertainties, and coordinate frame;
5. exact conversion to catalog value and any averaging rule;
6. reflections for which a scalar approximation is valid;
7. intensity sensitivity at catalog-pinned reflections.

Use `B = 8π² Uiso`. For a Cartesian displacement tensor `U` and reciprocal vector `g` in Å⁻¹, amplitude attenuation is

```text
D_j(g) = exp[-0.5 gᵀ U_j g].
```

Current scalar implementation is the special case `U_j = B I / (8π²)`, producing `exp[-B |g|² / (16π²)]`.

## Model decision

Single crystal-wide scalar is defensible only when:

- source reports one isotropic parameter for all contributing sites; or
- source reports an effective displacement along the only modeled reflection direction, all catalog reflections are parallel to that direction, and the directional limitation is explicit.

Atom-specific isotropic parameters are required when refined sites or species have materially different `Uiso`/`Ueq`; a composition average changes relative site amplitudes and can change interference, not only overall attenuation.

Per-site tensors are required for non-basal reflections in anisotropic materials, multiple nonparallel reflection families, or sources whose directional ADPs cannot be reduced to one common projection. h-BN already demonstrates strong anisotropy, but its pinned basal-only use permits the documented `B33` approximation.

Do not extend schema yet. Evidence must first establish which catalog entries need richer data and whether primary sources provide compatible site labels and tensor frames. A future implementation should normalize source ADPs to per-expanded-site Cartesian `U` tensors in Å², evaluate `D_j(g)` inside each site sum in both `structure_factor` and `U_g`, and preserve scalar `B_ang2` as an explicit compatibility mode. This also affects sweep/checkpoint identity, golden catalog serialization, and validation-oracle comparisons.

## Work order

1. Reproduce four existing primary-source conversions and repair incomplete temperature/table locators.
2. Resolve high-impact anchors: HOPG, WSe2, MoSe2, MoTe2, TiS2, PtSe2, HfS2, HfSe2, and HfTe2. The first scoped pass found no acceptable replacement for these anchors; HOPG has a natural-graphite tensor candidate, while matching WSe2/MoSe2/MoTe2 records omit ADPs and Hf/PtSe2 sources remain incomplete.
3. Recover ADPs from primary refinements already cited for the 37 shared-value entries; record “not reported” rather than substituting another compound.
4. Audit diamond, silicon, LiF, and sapphire legacy values.
5. Run pinned-reflection sensitivity calculations before changing production values or schema.
