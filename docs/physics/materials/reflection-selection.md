# Reflection selection

Line emission is a sum over reciprocal-lattice vectors, and that sum has to be truncated. Which reflections enter affects both yield and scoring cost. PyRITE uses measured line-yield selection for supported incoherent runs, a family count for other routes or explicit requests, and explicit catalog pins where declared. Crystal texture restricts the candidate domain before selection.

## Default measured selection

With neither `n_families`, `max_reflections` nor `line_yield_tolerance` set,
the spectrum phase scores reciprocal shells $|\mathbf g|\in(G-2,G]$ Å⁻¹ until
the estimated omitted yield is at most 1e-3 of the kept total on the
selection band. For automatic `kinematic-ceiling` and `resonance-population`
axes, this is the shared clamped ceiling band: coarse 500 eV bins integrate
each line profile exactly, independently of the production stop and spacing.
A shell's yield on that band is read off its line arrays in closed form
(`sincsq_band_mass`), with no bin reduction. When the prefix covers the
population and the production axis spans the band, each shell is scored once on
the production axis and its spectrum is kept; otherwise the selected reflections
are then scored on the production axis, whose tail audit excludes the reference
pass. Extending a refused measured stop keeps
the same selection band. Explicit fixed axes retain their own band and sum
the shell spectra directly. Grouped flights (`max_dE_frac`) and truncated
sinc profiles and explicit numeric node-support windows retain that direct
route because the reference band-mass route does not apply. Bin integration
inherits the reducer's existing approximation limits; it introduces no
center-sampling approximation to the represented line profiles.
The automatic axis bounds the candidate cap before selection, so it may be substantially
wider than the four-family axis. The estimator is qualified on the Si and
MoS₂/Si scenes in
[the selector evidence](../../validation/radiation-physics/reflection-yield-tolerance-selector.md);
it is not a rigorous error bound for arbitrary scenes.

What counts as omitted depends on the continuum. A crystalline BremsLib
continuum has the lattice-coherent share of the scored reflections removed
([bremsstrahlung](../radiation-physics/bremsstrahlung.md#lattice-coherent-share-in-crystals)),
so a shell's CBS lines largely replace continuum the correction took out.
For such a radiator, each shell's stop quantity is its **net yield**

$$
s_k=L_k-S_k,
$$

the shell's in-band line yield $L_k$ less the in-band continuum share $S_k$
the correction removes for those reflections. Above $|\mathbf g|\approx16$
Å⁻¹ about 80 % of the line increment is CBS, so $s_k$ is mostly PXR and
PXR×CBS interference. The geometric tail runs on $|s_k|$, as a fraction of
the consistent total: the kept net yield plus the radiator layer's
uncorrected in-band continuum. $S_k$ is the production continuum scorer run
with tables holding only that shell's share, on the case's continuum
electrons, at 96 log nodes per decade across the line band. The scored line
spectrum and the selection record are unchanged in kind. The record adds
`metric` (`"net"` or `"lines"`), `background_yield` and each shell's `share`.

Radiators whose continuum is not corrected (EEDL continuum, channeling,
correction disabled) keep the line-yield metric, a tail fraction of the kept
line yield.

The selection scores the first `SELECTION_HISTORIES` (4096) line and
continuum electrons of the counter-addressed population (#1281), so its cost
does not grow with the history count; the selected set is then scored on
every history. Smaller populations select on every history. Streamed
runs hold the leading blocks on the host until the prefix is complete and
select on them one block at a time, summing the blocks, so streamed and
retained runs score the same set (yields agree to summation order) while the
device holds one block.
`CXR_REFLECTION_SELECTION_MODEL` is `measured-shell-cutoff-v7` since #1301,
so older measured-selection checkpoints do not resume.

## Fiber texture

The catalog's HOPG and pyrolytic h-BN use an axial coherent-line model:
grains have random in-plane azimuths, and retained reciprocal vectors are
parallel to the declared fiber axis `surface_hkl`. For the basal axis this
means `(00l)`, with systematic extinctions still removed by the structure factor.
At the default 2 Å⁻¹ shell width the retired `(002)/(004)` pins are the first
two shells, rather than a hard limit. Fixed-count routes rank the same domain:
`n_families=4` gives `(002)`–`(008)` for HOPG/h-BN. The existing tolerance determines the last shell.

Let $A$ be the nonsingular reciprocal basis and $a$ the primitive integer
surface normal. Collinearity is preserved by an invertible basis:

$$
hA=\lambda aA\quad\Longleftrightarrow\quad h=\lambda a.
$$

Since $a$ is primitive, integer Miller vectors on that line are exactly
$h=na$, $n\in\mathbb Z\setminus\{0\}$. Enumeration generates these multiples
up to the same form-factor cap **before** applying the Debye–Waller factor.
The metric floor, shells, tail estimate and reconstructed continuum-share
set all use this domain. Both axis signs describe the same set; a general
nonorthogonal cell needs no Cartesian-angle tolerance.

This is the catalog's axial model, rather than a claim that textured specimens
have no diffuse or powder-ring radiation. For basal HOPG/h-BN, it keeps the
directional B33 thermal factor on its valid reciprocal axis. Ordered h-BN flakes
require in-plane displacement data and are outside this default. The mosaic
FWHM is unchanged.

Dataset/case identities include the texture and primitive axis. Each selection
record includes `texture`, `texture_axis_hkl` and `selected_set_sha256`; the
digest includes the sorted selected Miller vectors and their texture domain.
Frozen pre-change HOPG transport measurements and shell shares are recorded in
[#1301](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1301)
and anchored by `tests/data/reflection_selection_fiber_1301.json`. The paired
4096-history cleo_sub measurements omit 0.758% (30 keV) and 0.102% (1 MeV) of
the full basal line yield with the retired pins, inside the [10, 3700] eV
reference band. On the full production axes (job 13696, same transports),
automatic selection raises the 30 keV HOPG and h-BN line yield by 3.8 % and
3.3 %, mostly from (006) lines near 5–6 keV, and the 1 MeV yield by 0.1 %; the
lattice-share continuum falls by at most 0.15 %. These are scene-specific
measurements, not a universal omitted-yield guarantee.

## Explicit controls and qualification

An explicit `CxrReflections(n_families=4)` or `--n-families 4` restores the
four-family policy. Explicit counts take precedence over automatic selection.
An explicit `line_yield_tolerance` requests selection and retains its
incompatibility errors. Automatic selection falls back to four families for
coherent emission, line roulette, temporal profiles, channeling and two-beam
transition radiation. Streamed scoring selects on the prefix (#1281). Catalog-pinned lists remain pinned.
Diffracted bremsstrahlung and Kossel terms keep the build-time reflection set.

The accuracy gain has a measured cost. On qlmc-ace (ROCm), the 64-history,
0.5–10 keV gate at 0.09375 eV bin spacing took 26.3 s for Si and
176–208 s for layered MoS₂/Si with incremental selection. The four-family
arms took about 1–2 s. On two CPU threads (qlmc-queen, same gate, node
quadrature, 101,335 nodes), Si took 24 s at four families and 2,185 s
(about 36 min) for the measured selection, which keeps G30 / 13,688
reflections. Layered MoS₂/Si took 156 s at four families, and its selected arm
had not finished when the 90 min job limit cancelled it. Treat the CPU path as
impractical for these wide-band, high-energy scenes; use a GPU or narrow the
declared line axis. These measurements exclude transport. Selection, ledger
status and the GPU measurements are in
[#1173](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1173);
#1200 sizes each selection call by its coupling-table memory and admits it
before scoring. With the sets unchanged, GPU selection got faster on three
axes:

- 1.6–2.2× faster on the 0.5–10 keV band.
- 2.3× faster on the layered automatic axis.
- No faster on the Si automatic axis.

See the
[reflection-selection benchmarks](../../repo-design/compute/reflection-selection-benchmarks.md).

## Automatic ranking

The fixed-family path enumerates candidate reflections and keeps the strongest *families*.

**Enumeration.** All integer triples with $|\mathbf g| \le g_{\max}$ are generated, with $g_{\max} = 8$ $\AA^{-1}$ by default. The per-axis index bound is exact rather than a guess:

```{math}
:label: eq-reflection-selection-index-bound

|h_i| \le \frac{g_{\max}\,|\mathbf a_i|}{2\pi},
```

which follows from $\mathbf b_i \cdot \mathbf a_i = 2\pi$ and bounds the search box tightly for any crystal system, including strongly anisotropic layered cells where a cubic guess would be badly wrong on one axis.

**Ranking.** Candidates are ordered by

```{math}
:label: eq-reflection-selection-metric

M(\mathbf g) = \frac{|S(\mathbf g)|}{g^2},
```

with the Debye–Waller factor already inside $S$. The $1/g^2$ is not a heuristic weight — it makes {eq}`eq-reflection-selection-metric` proportional to $|\chi_{\mathbf g}|$ evaluated at *each reflection's own* line energy. The argument: $\chi_{\mathbf g} \propto \lambda^2 S \propto S/\omega^2$, and the resonance energy of a reflection scales as $\omega_{\rm res} \propto g$ at fixed geometry, so $\chi_{\mathbf g} \propto S/g^2$. Ranking by {eq}`eq-reflection-selection-metric` therefore ranks reflections by the coupling strength they will actually radiate with, not by their structure factor at some fixed reference energy.

**Family grouping.** Symmetry-equivalent reflections are grouped by identical $(|\mathbf g|, M)$ to rounding. This recovers symmetry mates without any space-group machinery: two reflections related by a symmetry operation of the crystal necessarily share both quantities exactly, up to floating-point noise. All members of the top $n$ families are returned, Friedel mates $\pm(hkl)$ included, since $+\mathbf g$ and $-\mathbf g$ radiate into different directions and both must be summed. A representatives-only mode returns one deterministic member per family, for visualizations that should show each family once.

**Cutoff.** Families whose metric falls below $10^{-9}$ of the top family are dropped as forbidden or negligible, which removes extinct reflections without a special case for them.

Four families follow the published practice this project benchmarks against.
That truncation does not bound omitted yield: the #1143 Si and layered
MoS₂/Si scenes lose about 61 % and 78–79 % of their 0.5–10 keV yield.

## Explicit pinning

A catalog crystal may instead pin its reflection families outright. When it does, a free-text **reason is required** — the schema rejects a pin without one — so that a hand override always carries its justification next to it. Pinned representatives are given as positive-representative triples and are expanded automatically to both reciprocal directions.

The motivating case is fiber texture. HOPG is a fiber-textured aggregate: its $c$ axes are aligned but its in-plane orientations are random across grains, so in-plane reflections are incoherent across the illuminated volume and must not be summed as if they were a single crystal. Only the basal $(00l)$ reflections survive, and HOPG pins $(002)$ and $(004)$ with exactly that reason recorded.

A pin is an **emission set**: it restricts which reflections radiate. It is not an orientation reference. A symmetric-cut single crystal radiates through in-plane and oblique reflections as well, so its cut does not justify a pin (#1302). The cut is `surface_hkl`, and the catalog primary reflection used for allowed-geometry estimates and targeted longitudinal timing is the first non-extinct harmonic of that cut, or an explicit `reference_hkl` where the conventional reference differs (6H-SiC (0006), the doubled-cell ReS₂ (002), 1T-TiS₂ (003)). Besides texture, the remaining reason for a pin is Debye–Waller validity: PdTe₂ and 2H-TaS₂ store the basal $B_{33}$ projection, valid only for $\mathbf g\parallel c$, so they keep their basal family until per-site displacement tensors are supplied.

This is the general caveat about the automatic ranking, stated as a rule: {eq}`eq-reflection-selection-metric` ranks the **crystal structure only**. Texture, preferred orientation, and the sample's actual grain statistics are not inputs to it. A material whose grains are not equivalent to a single crystal needs its families pinned by hand.

## What the ranking does not account for

* **Texture and preferred orientation** — as above.
* **Detector acceptance.** A reflection is ranked by coupling strength, not by whether its resonance lands in the observed energy window or its emission cone reaches the detector. A strongly coupled reflection outside the configured line grid still costs kernel time.
* **Geometry.** Ranking is done before the crystal is oriented, so the electron direction, tilt, and azimuth do not enter. Which pinned or ranked reflections actually radiate is decided later by the resonance condition.
* **Energy dependence.** The ranking is evaluated at a single reference photon energy (1 keV by default). This is exact for the non-resonant term, which is energy-independent; it matters only through the anomalous terms of edge-prone elements, and only enough to reorder near-degenerate families.
* **Absorption.** A reflection whose line sits below an absorption edge of the material may be strongly self-absorbed on the way out. That is applied downstream, in [photon escape](../radiation-physics/photon-escape-and-dispersion.md), not in the ranking.

## Assumptions and limits

* Kinematic single-crystal coupling for every enumerated reflection, inheriting the [structure-factor](structure-factor.md) assumptions.
* Family grouping infers symmetry from numerically identical metrics rather than from the space group; accidental degeneracy between genuinely inequivalent reflections would merge them into one family. Since both members would then be returned together and both would be summed, the consequence is a family-count accounting artifact, not a dropped reflection.
* The $g_{\max} = 8$ $\AA^{-1}$ cap corresponds to $s = 0.64$ $\AA^{-1}$, well inside the fitted range of the form-factor parameterization, but it is a hard truncation: reflections above it are never considered.
* Cost scales with the number of returned reflections, so raising the family count is a real runtime decision, not a free accuracy improvement.

## Validation

The ranking metric is a **selection policy**, not a physical claim, and carries no validation-ledger row of its own. What is pinned is its *output*: the catalog golden snapshot serializes, for **every** catalog crystal, the top two ranked families at a 1 keV reference energy together with $|\mathbf g|$ and the complex structure factor of the leading reflection (`tests/materials/test_material_catalog.py::test_catalog_matches_serialized_physics_for_every_crystal`). A change in selection therefore fails the golden comparison rather than passing silently. The snapshot is regenerated deliberately, never repaired in place — see the `regen-golden` workflow.

The physics the metric is built from is separately covered — `structure-factor` (`anchored`) for $S$ and its Debye–Waller factor, and the per-crystal rows in the [crystal structure provenance ledger](../../validation/ledger-crystal-structure-provenance.md) for pinned family choices such as the HOPG basal restriction.

See the [validation ledger](../../validation/physics-validation-ledger.md).

Implementation owners: `pyrite.materials.crystal.dominant_reflections`; pinned families in `pyrite.materials.catalog` (`CrystalSpec.hkl_families`, `CrystalSpec.hkl_list`); catalog primary reflection in `pyrite.materials.emission._primary_hkl` (`CrystalSpec.reference_hkl`); resolution at `pyrite.campaign.geometry.crystal_params`.
