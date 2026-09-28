# `hfs2-structure`

**Claim.** Bulk 1T-HfS2 in space group P-3m1 (No. 164), represented by a three-atom primitive hexagonal cell with `a = 3.62 A`, `c = 5.80 A`, Hf on `1a`, and S on `2d` using the idealized octahedral coordinate `z = 0.25`.

**Code.** `src/pyrite/data/cifs/hfs2.cif` and `src/pyrite/data/catalog/crystals/hfs2.toml`

**Anchor.** `tests/materials/test_crystallography.py::test_hfs2_structure_sane`

**Sources.** 2D Semiconductors, *HfS2 Crystal* product data; Neal et al., *npj 2D Materials and Applications* **5**, 45 (2021), doi:10.1038/s41699-021-00226-z; Iwasaki, Kuroda, and Nishina, *J. Phys. Soc. Jpn.* **51**, 2233-2240 (1982), doi:10.1143/JPSJ.51.2233.

**Verifier context.** Independent session; did not author the implementation. The derivation below was completed from the cited claim and sources before inspection of the bundled CIF body.

## Source facts and assumptions

- The product page reports a hexagonal phase with `a = b = 0.362 nm`, `c = 0.580 nm`, `alpha = beta = 90 deg`, and `gamma = 120 deg`. Therefore the code-facing lengths must be `a = b = 3.62 A` and `c = 5.80 A`.
- Neal et al. identify room-temperature bulk HfS2 as the 1T polytype in `P-3m1` (No. 164), with each Hf at the center of an S octahedron. They also state that their calculations use a primitive three-atom cell, and cite Iwasaki et al. for the experimental structure.
- The exact sulfur internal coordinate is not supplied by the product page. This claim explicitly assumes `z = 1/4`; it is not a vendor-measured coordinate. It predicts a sulfur-plane separation `2 z c = 2.90 A`, consistent with Neal et al.'s reported `2.89 A` sheet thickness.

## Independent derivation

In the standard hexagonal setting of `P-3m1`, put Hf on `1a` and S on the symmetry-related `2d` pair:

```text
Hf: (0,   0,   0)
S:  (1/3, 2/3, z)
S:  (2/3, 1/3, -z)  mod 1
```

For the stated idealization `z = 1/4`, the sulfur heights are `1/4` and `3/4`. This is one Hf and two S atoms, hence one HfS2 formula unit in a three-atom primitive cell. The nearest-neighbor Hf-S distance is

```text
d(Hf-S) = sqrt(a^2/3 + (z c)^2)
         = sqrt(3.62^2/3 + (5.80/4)^2)
         = 2.544 A.
```

This agrees at the percent level with the approximately `2.53 A` Hf-S bond length quoted by Neal et al.; the small difference is consistent with the claim's explicitly idealized, rather than relaxed, sulfur coordinate.

Terminology caveat: `z = 0.25` gives the expected octahedral *coordination*, but it is not an exactly regular octahedron for the stated `a/c`. Exact 90-degree S-Hf-S angles would require `z = a/(sqrt(6)c) = 0.2548`. The implemented assumption remains source-consistent through both sheet thickness and bond length, but "idealized `z = 0.25` octahedral coordination" is more precise than "ideal octahedron."

The primitive hexagonal-cell volume is

```text
V = a b c sin(120 deg)
  = (sqrt(3)/2) a^2 c
  = (sqrt(3)/2)(3.62 A)^2(5.80 A)
  = 65.822 A^3.
```

Thus both the input-length units and the derived-volume units are consistent: nanometers convert to angstroms by a factor of 10, and the result is in cubic angstroms.

For a basal `00l` reflection, the three-site structure factor is

```text
F_00l = f_Hf + f_S exp(2 pi i l z) + f_S exp(-2 pi i l z)
       = f_Hf + 2 f_S cos(2 pi l z).
```

At `z = 1/4` and `l = 1`, the two sulfur terms cancel but the Hf term remains: `F_001 = f_Hf`, which is nonzero. This is also consistent with the primitive `P` lattice having no centering extinction for `001`. Therefore odd `(001)` is allowed; the claim does not assert that it is necessarily intense relative to every other reflection.

## Diff against the implementation

The bundled CIF and catalog row implement exactly the independently derived cell:

| quantity | independently derived / sourced | implemented | result |
|---|---:|---:|---|
| crystal system | hexagonal | `hexagonal` | pass |
| `a` | 3.62 A | 3.62 A | pass |
| `c` | 5.80 A | 5.80 A | pass |
| volume | 65.822 A^3 | 65.82 A^3 through the loader | pass |
| Hf `1a` | `(0, 0, 0)` | `(0, 0, 0)` | pass |
| S `2d` | `(1/3,2/3,1/4)`, `(2/3,1/3,3/4)` | exact pair | pass |
| stoichiometry | 1 Hf + 2 S | 1 Hf + 2 S | pass |
| `(001)` | `F_001 = f_Hf`, nonzero | anchor requires `abs(F_001) > 1` | pass |

There is no factor, sign, coordinate-setting, or unit discrepancy. The source chain supports the 1T phase and `P-3m1` assignment; the product page supports the lattice lengths after `nm -> A` conversion. The sulfur `z = 0.25` value is an explicit modeling assumption rather than a measured product-page value. The only finding is the terminology caveat above; it does not change the implemented coordinates or any checked result.

Canonical anchor command:

```text
uv run python scripts/dev.py test tests/materials/test_crystallography.py -k test_hfs2_structure_sane
```

Result: `1 passed, 18 deselected`.

## Adjudication

Units, limiting behavior, and crystallographic conventions pass. The independent derivation matches the implementation, and the focused anchor is green. Recommended ledger status: **`rederived`**. Suggested row update: change status `unverified -> rederived`, record that the volume, stoichiometry, ideal `2d` basis, and `(001)` allowance pass, and link this write-up. Prefer the note wording "idealized `z = 0.25` octahedral coordination" over "ideal octahedral `z = 0.25`." Only a human may apply the ledger edit or mark the claim `signed-off`.
