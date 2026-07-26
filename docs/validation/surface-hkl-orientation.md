# Independent validation: `surface-hkl-orientation`

## Fresh re-review (2026-07-25)

A new pre-implementation derivation reproduced the reciprocal normal
`g_hkl = 2 pi A^-T h`, proper minimal rotation, and composition
`R = Rz(phi) R0` below without a divergent factor, sign, transpose, or
convention. Focused current-tree verification passed:
`tests/test_surface_orientation.py` plus the catalog golden, `11 passed`.
Verdict remains `rederived`; human ledger update still pending.

## Scope and pre-implementation record

This derivation was written before inspecting the body of
`montecarlo/geometry.py::_orientation_R` or its tests.  The ledger and function
interface state that the function constructs the active rotation applied to
every reciprocal-space column vector.  It takes either a direct-lattice
direction `beam_uvw` or the reciprocal normal `surface_hkl`, maps that selected
direction to sample `+z` by a proper minimal rotation, and then applies a
right-handed azimuth about sample `+z`.  The two selectors are mutually
exclusive.

The inputs are a lattice, integer indices, and angles in radians.  A direct
vector has the lattice's length unit and a reciprocal vector has its inverse;
normalization removes either unit, so the returned rotation is dimensionless.
The assumptions recorded by the interface are that Miller indices use the
reciprocal basis supplied by the lattice, sample `+z` is the outward slab
normal, zero azimuth selects the minimal-rotation in-plane convention, and
positive azimuth is right-handed about `+z`.

## Independent derivation

Let the direct primitive vectors be the columns of

\[
A = [\mathbf a_1\ \mathbf a_2\ \mathbf a_3].
\]

For direct indices \((u,v,w)\), the selected Cartesian direction is

\[
\mathbf d_{uvw} = A
\begin{bmatrix}u\\v\\w\end{bmatrix}
=u\mathbf a_1+v\mathbf a_2+w\mathbf a_3.
\]

Let the reciprocal basis be the columns of

\[
B=2\pi A^{-T}=[\mathbf b_1\ \mathbf b_2\ \mathbf b_3],
\qquad \mathbf a_i\mathbin{\cdot}\mathbf b_j=2\pi\delta_{ij}.
\]

(Dropping the conventional common factor \(2\pi\) does not change an
orientation.)  The Cartesian normal to the \((hkl)\) lattice planes is

\[
\mathbf g_{hkl}=B
\begin{bmatrix}h\\k\\l\end{bmatrix}
=h\mathbf b_1+k\mathbf b_2+l\mathbf b_3.
\]

This is perpendicular to both independent translations within the plane,
because their index-space dot product with \((h,k,l)\) vanishes.  Therefore it,
not the generally different direct vector
\(h\mathbf a_1+k\mathbf a_2+l\mathbf a_3\), is the exact surface normal of a
nonorthogonal cell.

Choose \(\mathbf q=\mathbf d_{uvw}\) for the direct path or
\(\mathbf q=\mathbf g_{hkl}\) for the surface path and normalize it:

\[
\hat{\mathbf q}=\frac{\mathbf q}{\lVert\mathbf q\rVert},\qquad
\hat{\mathbf z}=(0,0,1)^T.
\]

For nonzero indices and \(\hat{\mathbf q}\ne-\hat{\mathbf z}\), define

\[
\mathbf v=\hat{\mathbf q}\times\hat{\mathbf z},\qquad
c=\hat{\mathbf q}\mathbin{\cdot}\hat{\mathbf z},
\]

and let \([\mathbf v]_\times\mathbf x=\mathbf v\times\mathbf x\).  Rodrigues'
shortest active rotation from \(\hat{\mathbf q}\) to \(\hat{\mathbf z}\) is

\[
R_{q\to z}=I+[\mathbf v]_\times
 +\frac{[\mathbf v]_\times^2}{1+c}.
\]

Direct substitution gives
\(R_{q\to z}\hat{\mathbf q}=\hat{\mathbf z}\).  It is orthogonal with
determinant \(+1\), hence preserves handedness.  For
\(\hat{\mathbf q}=+\hat{\mathbf z}\), its continuous limit is \(I\).  For
\(\hat{\mathbf q}=-\hat{\mathbf z}\), the formula is singular because the
shortest rotation has angle \(\pi\) but no unique axis; any deterministic unit
axis \(\hat{\mathbf n}\perp\hat{\mathbf z}\) gives the proper rotation
\(R_{q\to z}=2\hat{\mathbf n}\hat{\mathbf n}^T-I\).

The conventional active, right-handed sample azimuth is

\[
R_z(\phi)=
\begin{bmatrix}
\cos\phi&-\sin\phi&0\\
\sin\phi& \cos\phi&0\\
0&0&1
\end{bmatrix},
\]

so the requested composition for column vectors is

\[
\boxed{R=R_z(\phi)R_{q\to z}}.
\]

The order matters: \(R_z\) leaves the already aligned normal at `+z`, while
rolling the crystal in-plane.  A positive \(\phi\) maps `+x` toward `+y` when
viewed from `+z`, which is the right-hand convention.

## Filters and limiting cases

- **Units:** \(A\mathbf u\) has length, \(B\mathbf h\) has inverse length, and
  normalization makes both paths dimensionless.  All rotation entries are
  dimensionless functions of angles in radians.
- **Orthogonal one-axis equivalence:** for diagonal
  \(A=\operatorname{diag}(a,b,c)\), `beam_uvw=(1,0,0)` gives a vector parallel
  to `surface_hkl=(1,0,0)`, and likewise for the other single axes.  Their
  rotation matrices must therefore agree exactly (up to irrelevant positive
  reciprocal scale).
- **Nonorthogonal distinction:** in general \(A\mathbf h\not\parallel
  A^{-T}\mathbf h\); only the reciprocal construction is guaranteed normal to
  the \((hkl)\) plane.
- **Already aligned:** \(\hat{\mathbf q}=+\hat z\) and \(\phi=0\) gives the
  identity.  With no selected orientation and zero azimuth, the documented
  construction-frame sentinel is `None`.
- **Antialigned:** \(\hat{\mathbf q}=-\hat z\) requires a proper \(\pi\)
  rotation with determinant `+1`, not a reflection.
- **Azimuth:** after alignment, all \(\phi\) retain
  \(R\hat{\mathbf q}=\hat z\); `+pi/2` sends the aligned crystal's sample `+x`
  direction toward sample `+y`.

## Implementation comparison

This section was completed after the independent derivation above was fixed.

The implementation constructs the direct vector as
`u*a1 + v*a2 + w*a3`, and constructs the reciprocal vector through
`reciprocal_g_vector(surface_hkl, lattice)`.  That helper uses

\[
\mathbf b_1=2\pi\frac{\mathbf a_2\times\mathbf a_3}{V},\quad
\mathbf b_2=2\pi\frac{\mathbf a_3\times\mathbf a_1}{V},\quad
\mathbf b_3=2\pi\frac{\mathbf a_1\times\mathbf a_2}{V},
\]

which is exactly \(B=2\pi A^{-T}\).  `_rotation_between` uses the normalized
axis \(\mathbf k=(\hat q\times\hat z)/s\) in the equivalent Rodrigues form

\[
I+s[\mathbf k]_\times+(1-c)[\mathbf k]_\times^2.
\]

Since \(s^2=1-c^2\), this is algebraically identical to
\(I+[\mathbf v]_\times+[\mathbf v]_\times^2/(1+c)\).  Its parallel branch is
the identity and its antiparallel branch is a deterministic proper
\(\pi\)-rotation.  The azimuth matrix has the independently derived signs and
is left-multiplied, `Rz @ R`, so it acts after normal alignment.  No factor,
sign, transpose, or composition-order discrepancy was found.

An independent NumPy calculation, constructing \(A\), \(2\pi A^{-T}\), and
the reference Rodrigues matrix without repository geometry helpers, gave:

- general nonorthogonal lattice, `surface_hkl=(2, 0, -1)`,
  `azimuth_rad=0.41`: maximum matrix difference `0.0`, alignment error
  `1.11e-16`, determinant `1.0`;
- the same lattice, direct `beam_uvw=(1, 2, 3)`, `azimuth_rad=0.37`: maximum
  matrix difference `2.22e-16`, alignment error `2.22e-16`, determinant
  `1.0000000000000002`;
- orthorhombic `surface_hkl=(0, 0, -1)`: the antiparallel branch maps the
  selected direction exactly to `+z`, with determinant `1.0` and zero measured
  orthogonality error.

Catalog parsing enforces exactly one of `beam_uvw` and `surface_hkl`, requires
three integer components, and rejects the zero index triple.  The private
orientation function also rejects simultaneous direct and reciprocal inputs.
The focused command

```text
uv run python scripts/dev.py test tests/test_surface_orientation.py tests/test_material_catalog.py -k 'surface or orientation'
```

passed: `11 passed, 27 deselected`.

## Verdict

- **Filters:** units pass; limits pass; signs/conventions pass.
- **Re-derivation:** matches; no divergent term or convention.
- **Verdict:** `rederived`.
- **Suggested ledger change:** change `surface-hkl-orientation` from
  `unverified` to `rederived`; a human applies this change.  Do not mark it
  `signed-off`.
