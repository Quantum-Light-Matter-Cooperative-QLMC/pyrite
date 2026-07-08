# Validation: `hbn-structure`

**Claim.** Bulk hexagonal boron nitride (h-BN), space group P6₃/mmc (No. 194),
represented as an explicit four-atom conventional-cell basis with
`a = 2.504 Å`, `c = 6.661 Å`.

**Code.** `src/cxr_mc/data/crystal_structures.toml::[hbn]`
**Anchor.** `tests/test_crystallography.py::test_hbn_structure_sane`
**Source.** Standard bulk h-BN; canonical structure determination
Pease, *Acta Cryst.* **5**, 356 (1952): `a = 2.5040 Å`, `c = 6.6612 Å`.
**Verifier context.** Independent session; did not author the implementation.
Derived from the cited canonical structure, not from the code.

## Independent derivation

Bulk h-BN in the P6₃/mmc setting has `Z = 2` formula units per conventional
cell (4 atoms: 2 B + 2 N). The canonical Wyckoff assignment (Pease 1952) is

- **B** on `2c`: `(1/3, 2/3, 1/4)`, `(2/3, 1/3, 3/4)`
- **N** on `2d`: `(1/3, 2/3, 3/4)`, `(2/3, 1/3, 1/4)`

The defining structural feature of *bulk* h-BN (AA′ / "eclipsed" stacking,
distinct from graphite's AB) is that **each B sits directly above/below an N
in the adjacent layer, and vice versa** — the interlayer registry is
B↔N, not B↔B.

Hexagonal cell volume: `V = (√3/2) a² c`.
With `a = 2.504 Å`, `c = 6.661 Å`:

```
V = (√3/2)(2.504)²(6.661) = 36.169 Å³
```

## Diff against the implementation

Implemented basis (`crystal_structures.toml`):

| element | position |
|---------|-----------------------|
| B | `(0, 0, 0)` |
| N | `(1/3, 2/3, 0)` |
| B | `(1/3, 2/3, 1/2)` |
| N | `(0, 0, 1/2)` |

This is **not** written in the Wyckoff `z = 1/4, 3/4` setting, but it is the
*same crystal*. Applying the rigid origin shift `t = (2/3, 1/3, 3/4)` to the
canonical Wyckoff positions reproduces the implemented basis atom-for-atom
(verified numerically, mod-1, for all four atoms). A pure origin translation
leaves `|F(g)|²` — the only structure-factor quantity cxr-mc consumes —
invariant, so the two descriptions are physically identical.

Checks (all pass):

| check | derived / expected | implemented | result |
|-------|--------------------|-------------|--------|
| lattice `a` | 2.504 Å (Pease) | 2.504 Å | ✓ |
| lattice `c` | 6.661 Å (Pease) | 6.661 Å | ✓ |
| cell volume | 36.169 Å³ | `V_cell` = 36.17 Å³ | ✓ |
| formula units `Z` | 2 (4 atoms) | 4-atom basis | ✓ |
| stoichiometry | 2 B, 2 N (equiatomic) | 2 B, 2 N | ✓ |
| basis ≡ Pease Wyckoff | under shift `(2/3,1/3,3/4)` | exact match | ✓ |
| interlayer registry | AA′ eclipsed, B↔N | B(0,0,0) below N(0,0,½); N(⅓,⅔,0) below B(⅓,⅔,½) | ✓ |

## Anchor coverage note

Before this validation, `test_hbn_structure_sane` pinned only `a`, `c`,
`V_cell`, and the B/N counts — a future edit that scrambled the basis
`z`-coordinates could have broken AA′ stacking while still passing. The test
was strengthened here to assert the eclipsed B↔N interlayer registry, so the
anchor now pins the physics that this write-up verified.

## Related: B/N transport constants

The same h-BN work added `B` and `N` rows to
`transport.py::TRANSPORT_ELEMENTS` (claim `electron-transport`). Filter-level
check only:

- `B`: `Z = 5`, `A = 10.81 g/mol` ✓; `N`: `Z = 7`, `A = 14.007 g/mol` ✓.
- Mean ionization `J`: B = 76 eV, N = 82 eV. These are the **ICRU-37 mean
  excitation energies** and are consistent with the light elements already in
  the table (C = 78 eV, O = 95 eV → the sequence B 76 < C 78 < N 82 < O 95 is
  monotonic and matches ICRU-37 exactly). Note the table header comment reads
  "(Berger-Seltzer values)", but the light-element entries are ICRU-37 I-values,
  not the `9.76·Z + 58.5·Z^−0.19` Berger-Seltzer *formula* (which gives
  ~92 eV for B). This labeling predates the h-BN work; the new B/N values
  follow the table's *actual* de-facto convention (ICRU-37), so they are
  internally consistent.

The `electron-transport` *algorithm* claim (Joy–Luo + Mott/Browning) is
unchanged and remains `unverified`; adding two element rows does not verify it.

## Adjudication

Filters pass; the independent derivation reproduces the implemented basis
exactly (including the AA′ interlayer registry), and the anchor test — now
extended to pin that registry — is green. Recommended status: **`rederived`**
(independent derivation matches). Final `signed-off` is a human decision.
