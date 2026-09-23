# Crystal structure representation

Every coherent-emission quantity in PyRITE is a sum over the atoms of one unit cell, evaluated at reciprocal-lattice vectors of that cell. This page defines how a crystal is represented — lattice, basis, cell volume, orientation reference — and what that representation does and does not carry.

## Lattice geometry

A lattice is a small record naming a crystal system and its parameters. Five systems are supported, each with an explicit choice of direct vectors:

```{list-table} Supported crystal systems and their direct lattice vectors.
:name: tbl-crystal-structure-systems
:header-rows: 1

* - System
  - Parameters
  - $\mathbf a_1, \mathbf a_2, \mathbf a_3$
* - cubic
  - $a$
  - $(a,0,0)$, $(0,a,0)$, $(0,0,a)$
* - tetragonal
  - $a, c$
  - $(a,0,0)$, $(0,a,0)$, $(0,0,c)$
* - orthorhombic
  - $a, b, c$
  - $(a,0,0)$, $(0,b,0)$, $(0,0,c)$
* - hexagonal
  - $a, c$
  - $(a,0,0)$, $(-a/2,\, a\sqrt3/2,\, 0)$, $(0,0,c)$
* - general
  - $a,b,c,\alpha,\beta,\gamma$
  - the standard triclinic construction below
```

The hexagonal choice puts $\gamma = 120°$ between $\mathbf a_1$ and $\mathbf a_2$, which is the crystallographic convention and the one the layered materials in the catalog are indexed in.

The general (triclinic) construction fixes the orientation gauge by placing $\mathbf a_1$ along $x$ and $\mathbf a_2$ in the $xy$ plane:

```{math}
:label: eq-crystal-structure-triclinic

\mathbf a_1 = (a, 0, 0), \quad
\mathbf a_2 = (b\cos\gamma,\; b\sin\gamma,\; 0), \quad
\mathbf a_3 = (c_x, c_y, c_z),
```

with

```{math}
:label: eq-crystal-structure-triclinic-c

c_x = c\cos\beta, \quad
c_y = c\,\frac{\cos\alpha - \cos\beta\cos\gamma}{\sin\gamma}, \quad
c_z = \sqrt{c^2 - c_x^2 - c_y^2}.
```

Because the CIF importer emits `general` for every structure it loads, this is in practice the path all catalog crystals take; the specialized systems remain available for hand-constructed lattices and reduce to the same vectors.

## Reciprocal lattice

The reciprocal basis follows the physics convention with the $2\pi$ **included**:

```{math}
:label: eq-crystal-structure-reciprocal

\mathbf b_1 = 2\pi\frac{\mathbf a_2\times\mathbf a_3}{V_{\rm cell}},
\qquad
V_{\rm cell} = \mathbf a_1\cdot(\mathbf a_2\times\mathbf a_3),
```

and cyclic permutations, so that

```{math}
:label: eq-crystal-structure-g-vector

\mathbf g_{hkl} = h\,\mathbf b_1 + k\,\mathbf b_2 + l\,\mathbf b_3,
\qquad
|\mathbf g_{hkl}| = \frac{2\pi}{d_{hkl}}.
```

{eq}`eq-crystal-structure-g-vector` is the single most load-bearing convention in the repository. The alternative crystallographic convention $|\mathbf g| = 1/d$ differs by $2\pi$ and would propagate a factor $(2\pi)^2$ into the Debye–Waller exponent and the CBS $1/g^2$ denominator. Every form-factor argument is written in terms of $s = g/(4\pi)$ precisely so that the conversion happens once, at the boundary. The reciprocal basis is cached per lattice, since it is rebuilt on every reflection lookup in the hot path.

## Basis

The basis is an ordered list of `(element, fractional position)` pairs covering the **full** unit cell, not the asymmetric unit. Symmetry expansion is performed by the external CIF parser before PyRITE sees the structure; the adapter then performs a deterministic normalization:

* fractional coordinates are wrapped into $[0,1)$;
* sites are sorted by `(element, position)`, so the basis order is reproducible and hash-stable across loads;
* **partial occupancy is rejected**, not averaged. A site with occupancy other than 1 raises rather than being silently treated as a whole atom.

That last rule is a deliberate scope boundary. Fractional occupancy in a structure factor is a mean-field statement about a disordered sublattice; the kinematic sum implemented here has no representation for the diffuse scattering that accompanies it, so a partially occupied CIF is refused rather than half-modelled.

The adapter is structural only: it converts lattice parameters, expanded basis, and cell volume, and nothing else. X-ray form factors, structure factors, reflection selection, attenuation, and transport all stay in PyRITE.

## Derived quantities

Two quantities are computed from the structure and used everywhere downstream.

**Cell volume** $V_{\rm cell}$ [$\AA^3$] normalizes both couplings ($\chi_{\mathbf g}$ and $U_{\mathbf g}$ each carry $1/V_{\rm cell}$) and is taken from the CIF rather than recomputed from the lattice parameters.

**Element number densities** are derived by counting basis sites:

```{math}
:label: eq-crystal-structure-number-density

n_i = \frac{N_i}{V_{\rm cell}},
\qquad n_i~\text{in Å}^{-3},
```

where $N_i$ is the number of sites of element $i$ in the cell. This is the composition every non-crystallographic consumer sees — attenuation, stopping power, bremsstrahlung — so the crystal's density is the CIF's cell, and there is no separately maintained mass density that could drift out of agreement with the lattice. See [Material composition](material-composition.md).

## Orientation reference

A catalog crystal must declare **exactly one** of two orientation references, and the parser rejects a row that gives both or neither:

* `beam_uvw` — a direct-lattice zone axis $[uvw]$ placed along $+z$;
* `surface_hkl` — a reciprocal-lattice plane normal $(hkl)$ placed along $+z$.

The two coincide for orthogonal one-axis cuts and differ for nonorthogonal cleavage planes, where the reciprocal form is the correct one. **Every packaged crystal declares `surface_hkl`**: a slab normal is the normal of the cut face, and $\mathbf g_{hkl}$ is that normal by construction in any lattice, whereas the direct axis coincides with it only under symmetry the catalog does not assert. `beam_uvw` remains valid config and is the spelling a `Sweep` or `LayerSpec` override uses. Both are followed by the configured right-handed azimuth about $+z$; the geometry is documented in [Coherent PXR and CBS radiation](../radiation-physics/coherent-radiation.md) and [Transport geometry](../geometry/transport-geometry.md).

Layered materials may also declare `layers_per_cell`, the number of formula layers stacked along $c$ in the cell. It exists so a thickness can be requested in **layers** rather than ångströms, converted as $c/\text{layers per cell}$ — the natural unit for a few-layer van der Waals film.

## Provenance

Each catalog crystal carries identifying and traceability fields alongside its CIF: a human-readable name, the polytype or phase (2H, 1T, $T_d$, …), an optional Crystallography Open Database id or Materials Project id, and a required validation id naming the ledger row that records where the structure came from and what was checked. The phase field is not decoration: for the transition-metal dichalcogenides the same formula has several polytypes with different stacking, different $c$, and different allowed reflections, so a structure is only identified by formula *and* phase.

## Assumptions and limits

* A perfect, infinite, periodic lattice within each crystallite. Defects, vacancies, dislocations, strain, and grain boundaries are not represented.
* No solid solutions or site disorder — full occupancy is required.
* One fixed cell per crystal: no thermal expansion, no temperature or pressure dependence of the lattice parameters. The CIF's refinement temperature is a property of the source, and is recorded in provenance rather than modelled.
* Thermal motion enters only through a single scalar Debye–Waller factor; see [Structure factor and Debye–Waller](structure-factor.md).
* Crystallite misorientation is modelled separately and does not perturb the cell; see [Crystal mosaicity](crystal-mosaicity.md).
* The lattice is used for radiation only. Electron transport treats the material as an amorphous average, so channeling and coherent elastic scattering are out of scope.

## Validation

`crystals-cif-adapter` (`anchored`) covers the CIF conversion — lattice, basis, and volume round trip, non-P1 symmetry expansion, and partial-occupancy rejection. `cod-lattice-catalog-geometry` (`anchored`) checks the six lattice parameters of every database-pinned catalog crystal against the pinned external record; it is a **geometry-only** claim and covers 35 of 48 catalog crystals, saying nothing about the fractional basis, occupancy, orientation, or Debye–Waller factors. `surface-hkl-orientation` (**`unverified`**) covers the reciprocal orientation mapping. Individual structures carry their own rows in the [crystal structure provenance ledger](../../validation/ledger-crystal-structure-provenance.md).

See the [validation ledger](../../validation/physics-validation-ledger.md). No row here is human `signed-off`.

Implementation owners: `pyrite.materials._cif`, `pyrite.materials.catalog` (`CrystalSpec`, `CrystalInfo`), and the lattice helpers in `pyrite.materials.crystal` — `_direct_lattice_vectors`, `_reciprocal_basis`, `reciprocal_g_vector`.
