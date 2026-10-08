# Structural CIF adapter

Validation: `crystals-cif-adapter`.

## Independent source derivation

This derivation was recorded before inspecting the implementation body. The
claim concerns structural data conversion, with no scattering-factor model.
The signatures are `load_crystal_from_cif(path, mosaic_fwhm_deg=None)` and
`gemmi_structure_to_crystal_info(crystal, mosaic_fwhm_deg=None)`, returning an
internal crystal dictionary.

Sources: [Gemmi small-structure API](https://gemmi.readthedocs.io/en/stable/chemistry.html#small-structures)
describes `SmallStructure`, fractional sites and chemical occupancy;
[Gemmi symmetry operations](https://gemmi.readthedocs.io/en/stable/symmetry.html)
documents affine fractional-coordinate operations;
[IUCr CIF core dictionary](https://www.iucr.org/__data/iucr/cif/dictionaries/cif_core_2.4.2.dic.pdf)
defines cell parameters and affine fractional symmetry operations. The
implementation dependency under review is Gemmi 0.7.5.

Assumptions: a valid nondegenerate three-dimensional unit cell; an asymmetric
unit whose sites have chemical occupancy one; a complete, consistent space-group
description; neutral element identities for the internal basis. Disorder and
partial occupancy are outside the integer-atom basis representation. Mosaic
width is optional metadata in degrees and does not alter structural coordinates.

For lattice vectors collected as columns of $A$, fractional coordinates
$\mathbf f$ correspond to $\mathbf r=A\mathbf f$. The metric is

$$
A^{\mathsf T}A=
\begin{pmatrix}
a^2 & ab\cos\gamma & ac\cos\beta\\
ab\cos\gamma & b^2 & bc\cos\alpha\\
ac\cos\beta & bc\cos\alpha & c^2
\end{pmatrix}.
$$

Thus lengths remain in angstroms, angles remain in degrees in the dictionary,
and fractional coordinates remain dimensionless. In trigonometric evaluation
the degree angles must first be converted to radians. The cell volume is

$$
V=\lvert\det A\rvert
=abc\sqrt{1+2\cos\alpha\cos\beta\cos\gamma
-\cos^2\alpha-\cos^2\beta-\cos^2\gamma},
$$

in cubic angstroms. An orthogonal cell gives $V=abc$; a hexagonal cell with
$\alpha=\beta=90^\circ$, $\gamma=120^\circ$ gives
$V=abc\sqrt{3}/2$. Uniform scaling of lengths by $s$ scales volume by $s^3$.

Each symmetry operation acts on fractional coordinates as

$$
\mathbf f'=W\mathbf f+\mathbf w \pmod{\mathbb Z^3}.
$$

The basis for each input site is the set of distinct orbit positions, with
representatives in $[0,1)^3$. Two images of the same site that differ by an
integer translation are one atom. This matters at special positions: orbit
size equals group order divided by site-stabilizer order. Occupancy must remain
the chemical occupancy one; converting to crystallographic occupancy would
incorrectly divide special-position weights. For P1 the identity is the only
operation, so each input site contributes exactly one basis entry modulo a
lattice translation. For inversion, $(0.1,0.2,0.3)$ contributes itself and
$(0.9,0.8,0.7)$; the origin contributes once.

The dimensional, P1/orthogonal limiting-case, and affine-sign conventions pass
these source-level filters. Basis order is a serialization convention and must
be deterministic; it does not change this orbit-set claim.

## Implementation comparison

`gemmi_structure_to_crystal_info` copies `cell.parameters` into the six lattice
fields and copies `cell.volume` without a unit conversion. It rejects chemical
occupancy differing from one by more than $10^{-12}$, obtains explicit CIF
operations or the complete named space-group operation set, applies
`Op.apply_to_xyz`, reduces coordinates modulo one, and sorts by element and
coordinate. Each site's orbit is deduplicated using periodic fractional
componentwise distance at tolerance $10^{-12}$; different source sites remain
separate. This implements the affine orbit derivation to floating-point
precision, including centered space groups. Missing symmetry metadata is
rejected. `load_crystal_from_cif` uses `gemmi.read_small_structure` before
calling that adapter. The compatibility function
`crystals_crystal_to_crystal_info` copies an already expanded caller-provided
basis, with the same occupancy filter, wrapping, and sorting.

Independent analytic examples executed against installed Gemmi 0.7.5:

| Example | Expected | Observed |
| --- | --- | --- |
| P1 translated point | $(0.1,0.8,0.3)$ from $(1.1,-0.2,0.3)$ | matches within $10^{-12}$ |
| Inversion general point | $(0.1,0.2,0.3)$ and $(0.9,0.8,0.7)$ | matches within $10^{-12}$ |
| Inversion origin | one atom at $(0,0,0)$ | matches |
| Inversion near origin | $(0.01,0,0)$ and $(0.99,0,0)$ in cubic 4 angstrom cell | matches |
| F m -3 m origin | origin and three face centers | four positions, matches |
| Hexagonal metric, lengths 3, 4, 5 angstroms | $V=30\sqrt{3}$ cubic angstroms | 51.96152422706633, relative error below $10^{-14}$ |

The independent mosaic-width example preserves 0.2 degrees. A synthetic
already-expanded compatibility object with orthogonal lengths 3, 4, 5
angstroms, volume 60 cubic angstroms, and translated fractional site
$(1.1,-0.2,0.3)$ retains the expected volume, mosaic width and wrapped site.
Changing its occupancy to 0.5 raises `ValueError`, as required. The focused
project command

```text
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/materials/test_crystallography.py -k 'cif or crystals_crystal_to'
```

checks deterministic P1 mapping, non-P1 inversion expansion, partial-occupancy
rejection, all packaged P1 CIF/catalog matches, and parametrized exact/nearby
special positions with both explicit and named symmetry operations.
Final focused result: eight passed, 55 deselected. Coordinate anchors use
absolute tolerance $10^{-12}$; deterministic ordering is checked exactly.

### Resolved backend tolerance discrepancy

The [pinned Gemmi 0.7.5 implementation](https://github.com/project-gemmi/gemmi/blob/v0.7.5/include/gemmi/small.hpp#L213-L230)
uses a 0.4 angstrom special-position tolerance. For images of each input site,
it discards an image whenever its periodic Cartesian distance from an existing
image is less than that tolerance. This differs from exact equivalence modulo
integer lattice translations. It does not compare different input sites.

An independent adversarial point demonstrates the difference: for a cubic
4 angstrom P -1 cell, a fully occupied atom at $(0.01,0,0)$ has exact orbit
$\{(0.01,0,0),(0.99,0,0)\}$. Their periodic separation is 0.08 angstroms;
the adapter returns only $(0.01,0,0)$. Such close full-occupancy images are
unphysical for ordinary atomic structures, but the unrestricted mathematical
orbit claim would still include them.

The first divergent convention in the initially reviewed version was the equivalence criterion:
exact fractional equivalence is replaced by periodic Cartesian separation
less than 0.4 angstroms. The revised adapter no longer uses this Gemmi helper;
its explicit affine expansion retains both nearby positions and deduplicates
only within fractional numerical tolerance. The independent adversarial point
now passes. No separated-image assumption is needed beyond resolution of
distinct coordinates at the stated $10^{-12}$ numerical tolerance.

## Adjudication

Units, limiting cases and symmetry conventions pass; the revised
implementation matches the independent derivation. Verdict: `rederived`.
Suggested owner ledger edit: record `rederived` and this write-up; once the
focused regression anchors pass, the owner may record `anchored` and include
`test_cif_special_positions_and_nearby_distinct_sites` in the anchor list.
Human sign-off remains separate.

Documentation test/build and rendered-math inspection are delegated to the
primary owner after this page is added to the documentation index.
