# `cr-planar-zone-row-ownership`: planar zone rows drop own-row native lines

**Status:** see the [ledger row](../ledger-channeling.md#cr-planar-zone-row-ownership). Fresh-context independent verification (2026-11-10, #1055, HEAD `f6b33d1b3`).

The claim has three parts. First, native PXR/CBS line scoring for a planar
channeling case keeps every reflection off the plane's own reciprocal row on
every row, and keeps the own row only on rows outside the entrance zone.
Second, the persisted `spec_channeling`, in `final_states="tracked"` mode,
owns the own row's emission on zone rows: bound→bound, capture and the in-cut
free→free block. Third, characteristic, bremsstrahlung, axial and
channeling-off scoring are unchanged.

## Inputs used before reading the implementation

- The ledger rows `cr-planar-zone-row-ownership`, `cr-planar-free-free-block`
  (anchored, [write-up](cr-planar-free-free-block.md)), `cr-persisted-channel`
  and `cr-zone-handoff-accounting`.
- The axial precedent `cr-axial-zone-row-native-lines` and its
  [write-up](cr-axial-zone-row-native-lines.md).
- ADR-0018 "Emission", including the #1055 outcome.
- The [#1055 measurement note](../../research/physics/planar-zone-row-cbs-1055.md)
  and its record `docs/validation/check-records/planar-zone/zone-cbs-1055.json`
  (jobs 10912 and 10919, `channeling_model` v5).
- The native line kernel's stated model (`mc_cxr_spectrum`, Feranchuk
  Eqs. 13–14 as fused in `_cxr_amp_sq_core`). Each row is a free electron on a
  straight path with constant velocity. Its line sits at

$$
\omega_{\rm res}=\frac{\beta\,\hat{\mathbf v}\cdot\mathbf g\,c}{1-\beta\,\hat{\mathbf v}\cdot\hat{\mathbf n}},
$$

  with amplitude $A=A_{\rm PXR}+A_{\rm CBS}$, finite-time
  $\operatorname{sinc}^2$ profile, and normalization per incident line
  electron.

## Independent derivation

### The planar potential holds only the own row

Write the crystal potential as $V(\mathbf r)=\sum_{\mathbf G}V_{\mathbf G}
e^{i\mathbf G\cdot\mathbf r}$. The planar continuum potential of planes
$(hkl)$ is the average over the in-plane coordinates $\mathbf t$ with
$\mathbf t\perp\mathbf g_{hkl}$. The average of $e^{i\mathbf G\cdot\mathbf t}$
over a large in-plane area vanishes unless $\mathbf G\cdot\mathbf t=0$ for
every in-plane $\mathbf t$, that is unless $\mathbf G\parallel\mathbf g_{hkl}$.
So

$$
U(x)=\sum_{n\in\mathbb Z}V_{n\mathbf g_0}\,e^{i n g_0 x},\qquad
\mathbf g_0=\mathbf g_{\rm coprime}(hkl),\quad x=\hat{\mathbf g}_0\cdot\mathbf r .
$$

Every Fourier component of the transverse Hamiltonian, and so every
transverse momentum transfer that the density-matrix source can produce,
lies on the own row $\{n\mathbf g_0\}$. The scattering dissipator transfers
transverse momentum too, but it is incoherent scattering, not lattice
emission, and it is not a Fourier component of any reflection.

A native line at reflection $\mathbf g$ off this row has
$\hat{\mathbf v}\cdot\mathbf g$ fixed by the in-plane part of $\mathbf g$
along the beam. No transition between transverse eigenstates of $U(x)$
radiates there. Within the model, then, the zone source has no counterpart
for other reflections, and keeping them native cannot double count.
(Treating a channeled electron's transverse wave function as a plane wave at
the beam angle for those reflections remains an approximation. That is a
question of accuracy, not of overlap, and the claim does not cover it.)

### Parallel in index space is parallel in reciprocal space

$\mathbf g(hkl)=h\mathbf b_1+k\mathbf b_2+l\mathbf b_3=(h,k,l)\,B$. The
$\mathbf b_i$ are linearly independent in any lattice, so $B$ is nonsingular
and the map $\mathbb Z^3\to\mathbb R^3$ is linear and injective. Then
$(h,k,l)=\lambda(h',k',l')$ holds if and only if
$\mathbf g=\lambda\mathbf g'$. For integer index vectors, parallelism
(either sign) is exactly
$(h,k,l)\times(h',k',l')=\mathbf 0$. This needs both triples in the same
three-index basis with the same $B$, and a nonzero plane triple, since the
zero vector is parallel to everything.

### Additivity and normalization

For incoherent scoring, the native spectrum is

$$
\mathrm{spec}(E)=\frac{1}{N_e}\sum_{\mathbf g}\sum_{r}
\frac{dN_{r,\mathbf g}}{dE\,d\Omega},
$$

a double sum over reflections and rows with no cross terms. Split the index
set $\{(\mathbf g,r)\}$ into own/other reflections and zone/non-zone rows.
The sum is additive over that partition:

$$
S[\text{all};\text{all}]=S[\text{other};\text{all}]+S[\text{own};\notin\text{zone}]+S[\text{own};\in\text{zone}] .
$$

The new rule drops the last term. $N_e$ is the number of incident line
electrons. It must be passed unchanged to both parts, and every row-indexed
array must be filtered with the same mask.

In coherent scoring (`spec_coherent`), different reflections, orientations
and polarizations still add incoherently, as the temporal and coherent kernel
docstrings state. Within one reflection, however, amplitudes from different
rows interfere: rows of one electron, and rows of different electrons through
the bunch form factor. So $S_{\rm coh}[\text{own};\text{all}]$ is not
$S_{\rm coh}[\text{own};\notin\text{zone}]+S_{\rm coh}[\text{own};\in\text{zone}]$.
The difference is the interference between zone and non-zone own-row
amplitudes. Defining the kept part as "the kernel on the non-zone rows"
drops the zone self term together with that cross term. This is consistent
with ownership, because the zone's own-row emission now comes from an
incoherent expected-value source that has no amplitude to interfere with.
It is an omission nonetheless, and the claim does not state it.

### No double count, and what stays unscored

On a zone row, the electron occupies the solver's tracked states, with
$E_\perp<U_{\rm barrier}+{\rm margin}$. Over the retained basis, the zone's
own-row emission splits by final state into bound→bound, capture and
free→free. With `final_states="tracked"`, `cr-planar-free-free-block` puts all
three blocks in `spec_channeling` with no cross term, because the final
subspaces are orthogonal. The native own-row line on a zone row is a
first-Born, plane-wave estimate of the same current. It is not a separate
channel, so after the drop each zone path's own-row potential emission is
counted exactly once.

What remains unscored on zone rows:

1. **Free→free above the tracked cut.** These states form a sink in the
   density-matrix model. The claim declares this.
2. **The own-row PXR term and its interference with CBS.** The native
   amplitude is $A_{\rm PXR}+A_{\rm CBS}$. The density-matrix source is
   transverse-potential dipole emission and has no $\chi_{\mathbf g}$
   (diffracted-polarization) term. Dropping the own row on zone rows
   therefore also drops $\lvert A_{\rm PXR}\rvert^2$ and
   $2\operatorname{Re}A_{\rm PXR}A_{\rm CBS}^{*}$ for the own row. Neither
   owner supplies them. The claim does not declare this (measured below).
3. **In `spec_coherent`, the own-row coherent terms involving zone rows**, as
   derived above. The claim does not declare this either.

Handed-off electrons keep their native own-row CBS on their post-handoff
rows, which are not tagged as zone rows. Electrons with entry fate are
outside the retained ensemble, so `spec_channeling` does not count them. Any
non-zone rows they have are scored natively.

### Limiting cases (derived)

- **Beam in the planes, $\psi=0$.** On a zone row,
  $\hat{\mathbf v}\cdot\mathbf g_{\rm own}=0$, so $\omega_{\rm res}=0$ and
  no own-row line falls on a positive grid. `spec` is unchanged.
  `spec_channeling` does change, because it gains the free→free block
  (34% of CR for Si in the #1055 note).
- **No zone rows.** The own-row part runs on all rows, so `spec` is the full
  native sum, up to floating-point reassociation of the two partial sums.
- **No own-row reflection in `hkl_list`.** Nothing to split, so `spec` is
  unchanged.
- **Channeling off, or axial.** The rule does not apply. Axial has its own
  zone-row rule (`cr-axial-zone-row-native-lines`).
- **Untagged planar transport, or a multilayer stack.** The rows cannot be
  attributed, so a correct implementation must refuse rather than guess.

## Implementation comparison

- `is_own_row` (`src/pyrite/montecarlo/runner/planar_zone_lines.py:53-62`)
  tests `np.cross(hkl, plane_hkl) == 0` on integers, which is the
  index-space criterion derived above. Both triples use the same basis.
  `planar_profile` (`src/pyrite/materials/channeling/potential.py:189-200`)
  builds $U$ from $n\cdot$`hkl_c` through `reciprocal_g_vector`, and the line
  kernel maps `hkl_list` through the same `hkl @ _reciprocal_basis(lattice)`
  (`src/pyrite/materials/crystal.py:156`). The profile loop confirms that the
  potential contains only the own row. The case record refuses a zero
  `plane_hkl` (`src/pyrite/montecarlo/case.py:551`). A `(0,0,0)` entry in
  `hkl_list` would count as own row, but the catalog never emits one.
- `planar_zone_line_parts` (`planar_zone_lines.py:76-110`) returns `None`
  when the record is missing, is not planar, or has no own-row member. That
  is the pass-through. Otherwise it refuses a multilayer stack and an
  untagged transport. It returns
  `[(view(other), segs), (view(own), rows(segs, ~zone))]`, and the first
  entry is omitted when there are no other reflections. `_rows` indexes
  every `_SEG_ARRAYS` key and `channeling_zone` with host indices through
  `electron_blocks._take`. Scalars (`Ne`) and per-electron arrays
  (`initial_*`) are kept.
- `_score_line_parts` (`src/pyrite/montecarlo/runner/case_driver.py:589-601`)
  calls `_cxr_for_segments` once per part, with the caller's `Ne`, coherence,
  audit, temporal and capture objects, and sums the spectra. Each view has
  `channeling=None`, so the recursion passes through to the ordinary route.
  That route includes GPU `iter_electron_blocks` for incoherent sums. The
  full-length `channeling_zone` left in those blocks is never read, because
  the view has no channeling record.
- `case_channeling_spectrum`
  (`src/pyrite/montecarlo/runner/channeling_emission.py:42-127`) passes
  `final_states="tracked"`. The replay/recompute path
  `rescore_channeling_spectrum` reaches the same producer.

### Every native line route

Every runner line route enters `_cxr_for_segments`: the emission adapter
(`emission_adapters.py:198`, for both `spec` and `spec_coherent`), line
roulette, `cxr_grid` coherent-window capture, and the energy-grid checks.
The only other `mc_cxr_spectrum` callers are validation figures and the
public `mc_cxr_spectrum_solid_angle`, which are not runner routes. The
routes that bypass the planar split are all refused for channeling cases:
line roulette (`_line_roulette.py:23`), live streaming
(`streamed_scoring.py:46`), trajectory capture and electron-block transport
(`transport_options.py:206-209`), photon aperture
(`photon_aperture.py:75`), and every transition-radiation model
(`case.py` `_validate_transition_radiation`). The two-beam DTR branch that
precedes the planar split in `_cxr_for_segments` is therefore unreachable
for a planar case. Grid seeding (`cxr_seeds`) still considers the zone
own-row resonance when placing nodes. That can only refine the grid where
no line remains, and it does not change any scored value.

### The `_LineCase` view

The view replaces only `hkl_list` and `channeling`. The kernel inputs read
from the case (`crystal`, `B_ang2`, `composition`, orientation keys,
mosaic, `sinc_cutoff`, `spec_chunk`, coherence and bunch keys,
`E_cut_lines_keV`, `layer_radiators`) are inherited unchanged. No case
array is aligned to `hkl_list` by position. The batched table cache keys on
the explicit `hkl_list` tuple (`_batched.py:239-252`), so the two parts
never share a table. `CoherentRowCollector` appends each part's rows with
call-local row indices and preserved electron IDs, so coefficient capture
accumulates correctly. `len`, iteration and `in` agree with the base case.
`"channeling" in view` is `True` with value `None`, and every consumer
tests `.get("channeling") is not None`.

## Numeric checks

These were run independently with a scratch probe, outside the repository.
The own row was classified geometrically, as $\lvert\hat{\mathbf g}\times
\hat{\mathbf g}_{(1\bar10)}\rvert<10^{-9}$, not with `is_own_row`. Each
expected part was scored by calling the kernel on explicitly masked rows.
Si, 20 µm, 12 line histories, $H=8$, 4 Bloch cells, tilt 3°.

| Case | Check | Result |
|---|---|---|
| 17 MeV, azimuth 46° ($\psi=0.913$ mrad), 5–400 keV | production `spec` vs $S[\text{other};\text{all}]+S[\text{own};\notin\text{zone}]$, incoherent | identical (0 of peak) |
| same | $S[\text{all};\text{all}]-S[\text{own};\in\text{zone}]-\text{spec}$, incoherent | $2.7\times10^{-14}$ of peak |
| same | removed zone own-row line / kept `spec`, band integral | $2.38\times10^{-2}$ / $2.99\times10^{-3}$ /sr |
| same, Gaussian bunch, coherent | production `spec_coherent` vs independent parts | identical |
| same, coherent | $S[\text{all}]-S[\text{own};\in\text{zone}]-\text{spec}_{\rm coh}$ | $3.3\times10^{-2}$ of peak (zone/non-zone interference, as derived) |
| 17 MeV, azimuth 45° ($\psi\approx10^{-17}$) | zone own-row line on 5–25 keV | exactly 0 |
| 5 MeV, azimuth 45° (fixture geometry), 1–10 keV | zone own-row line; production vs full sum | exactly 0; identical |
| 17 MeV, $\psi=0.913$ / 1.370 mrad | own-row zone line: PXR / CBS / total | $1.36\times10^{-5}$ / $2.27\times10^{-2}$ / $2.38\times10^{-2}$; $8.9\times10^{-6}$ / $1.53\times10^{-2}$ / $1.61\times10^{-2}$ |
| own-row part rows | full-length arrays left after `_rows` | none (26 of 38 rows kept) |

The own-row line on zone rows has $\lvert A_{\rm PXR}\rvert^2$ equal to
$6\times10^{-4}$ of the total. The PXR–CBS interference term is
4.1% (0.913 mrad) and 4.6% (1.37 mrad) of the total. These are the
undeclared omission 2. Scale check: at $\psi=0.913$ mrad,
$\hbar c\,\beta g_{220}\psi/(1-\beta)$ with $g_{220}=3.272$ Å$^{-1}$ and
$\gamma=34.3$ gives 13.8 keV. This matches the in-band line that the anchor
asserts, near 14 keV.

## Model marker and baseline

`CHANNELING_MODEL` is `planar-density-matrix-emission-v6`
(`src/pyrite/montecarlo/channeling/handoff.py:58`). It enters provenance
only when the case has a channeling record (`_physics_provenance.py:139-142`),
so channeling-off identities do not change. Axial cases carry the same marker
and are re-keyed although their physics is unchanged. That is harmless
over-invalidation. The fixture
(`tests/helpers/physics_baseline.py::_channeling_scene`) is Si at 5 MeV,
tilt 3°, azimuth 45°. The probe reproduces $\psi\approx0$ there, and the
zone own-row line is exactly zero, so the README explanation is right. The
pinned outputs are `spec`, `spec_characteristic`, `brem`, rows and fates.
`spec_channeling` is not pinned, which is why the tracked catalog leaves
them unchanged. The pinned `spec` is $[0,0,0]$ on the 1.5–2 keV grid, so the
fixture pins only identity. It would not detect a wrong own-row split. The
discriminating anchor is `test_zone_rows_score_no_own_row_lines_and_keep_every_other_row`.

## Anchors

`UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/montecarlo/channeling/test_planar_zone_lines.py tests/montecarlo/runner/test_physics_baseline.py::test_seeded_channeling_output_baseline`:
6 passed (rc 0, under an 8 GiB scope).

## Findings

1. **No implementation discrepancy.** The incoherent partition, the
   index-space own-row test, normalization, route coverage, refusals,
   pass-through cases and the `_LineCase` view all match the independent
   derivation. Production equals the independently assembled parts exactly,
   in both the incoherent and coherent routes.
2. **Claim text overstates completeness (non-blocking, claim edit needed).**
   The statement "omits only free→free above the tracked cut" is not exact.
   Zone rows also lose the own row's PXR term and its interference with CBS,
   about 4–5% of the removed native line at 0.9–1.4 mrad, almost all of it
   interference. In `spec_coherent`, they also lose the own-row coherent
   terms involving zone rows. Neither is double counting, and both follow
   from assigning the zone to an incoherent potential-emission source. They
   should be declared.
3. **Limiting-case wording.** The multilayer and untagged-transport refusals
   apply only when `hkl_list` holds an own-row reflection. Otherwise the case
   passes through unchanged, which is harmless. The "no zone rows" case
   equals the full sum to rounding, not bit for bit.
4. **Weak baseline pin (informational).** The baseline fixture's `spec` is
   identically zero, and it does not pin `spec_channeling`. It guards
   identity, not this rule.

## Verdict

`rederived`. The ownership partition and its implementation agree. Findings
2 and 3 are claim-text corrections, not code defects. With
`test_planar_zone_lines.py` green, the claim supports `anchored` once the
task owner applies `rederived` and amends the claim text.
