# `cr-axial-zone-row-native-lines`: native axial reflection ownership

**Status:** see the [ledger row](../ledger-channeling.md#cr-axial-zone-row-native-lines).
Fresh-context independent rederivation, 2026-11-12, #1148.

The native spectrum is the sum of nonzero axial-order reflections on all
rows and zero-order reflections on continuation rows only. Each part keeps
the same incident-electron normalization. The fibre source and its declared
in-cut free→free omission are unchanged. This supersedes #994's blanket zone-row
removal; that rule also removed off-zone reflections absent from the transverse
source, although their resonances lie outside the original admitted band.

## Sources consulted before implementation

Repository claim and derivation `cr-axial-2d-eigenproblem`, ADR-0018 emission ownership, current `cr-axial-zone-row-native-lines` claim and write-up. Primary supporting source: Wagner et al., Phys. Rev. Accel. Beams 22, 054502 (2019), DOI https://doi.org/10.1103/PhysRevAccelBeams.22.054502, continuum axial potential described as Fourier sum over reciprocal vectors perpendicular to the axis. Publisher abstract/search text accessible; full-text PDF retrieval failed, so no unverified equation number claimed. The elementary Fourier projection below independently establishes the selection.

## Derivation fixed before implementation inspection

Let the nonsingular direct-lattice matrix have columns $\mathbf a_i$, $A=(\mathbf a_1,\mathbf a_2,\mathbf a_3)$, and the reciprocal matrix be $B=2\pi A^{-\mathsf T}$. For integer index columns $h=(h,k,l)$ and $u=(u,v,w)$,

$$
\mathbf G=Bh,\quad \mathbf t=Au,\quad \mathbf G\cdot\mathbf t=2\pi h^{\mathsf T}u.
$$

The continuum projection along a full lattice translation is

$$
\overline U(\mathbf r)=\int_0^1 U(\mathbf r+s\mathbf t)\,ds
=\sum_h U_h e^{i\mathbf G\cdot\mathbf r}\int_0^1e^{2\pi i(h^{\mathsf T}u)s}\,ds
=\sum_{h^{\mathsf T}u=0}U_he^{i\mathbf G\cdot\mathbf r}.
$$

Thus the existing transverse source contains precisely the zero longitudinal-order Fourier sector; the nonzero-order sector is absent from its continuum Hamiltonian. The predicate is exact integer arithmetic and applies to every valid lattice metric, including oblique cells. The Euclidean dot product of Cartesian `hkl` and Cartesian axis would generally be wrong; fractional reciprocal/direct integer indices are dual. For an orthogonal frame rotation $R$, $(R\mathbf G)\cdot(R\mathbf t)=\mathbf G\cdot\mathbf t$, so sample tilt and crystal-cut rotations cannot change ownership. Beam tilt also cannot redefine the continuum lattice axis.

The required native row/reflection ownership table is:

| row | $h\cdot u=0$ | $h\cdot u\ne0$ |
| --- | --- | --- |
| entrance zone | source-owned; omit native | native-owned; score native |
| continuation after handoff | score native | score native |

Incident electron normalization $1/N_e$ remains unchanged; removing rows changes only the sum. No extra survival factor follows from this ownership derivation.

## Limits and accuracy boundary

- Axis reversal and any nonzero integer rescaling leave the zero-set invariant. $\pm h$ partners receive the same ownership.
- All-zero `hkl` is transverse but is normally excluded from physical reflection lists. Zero axis must be refused by axis validation.
- With no zone rows, the ordinary native spectrum is recovered. No channeling/planar case must keep its existing rule.
- At exact beam-axis alignment, nonzero-order reflections still have $\mathbf v\cdot\mathbf G\ne0$; their resonance must not be removed merely because the zero-order native line vanishes. Their absence in the old 800 eV–10 keV on-axis envelope was an energy-band fact, not a general ownership fact.
- Removing native zero-order CBS prevents the known perturbative duplicate of the transverse-force source. This does not prove that the complete PXR amplitude is represented by that source; omitting channeled-state PXR remains an accuracy limitation, especially off axis.
- Restored nonzero-order native PXR/CBS uses a straight mean-row/free-electron approximation inside the zone. The selection proves sector ownership, not quantitative accuracy for a channeled-state scattering amplitude, angular modulation, finite-depth interference, or cross-sector interference.
- Existing `free_free_upper` is a measured on-axis in-cut transverse-source omission bound for the admitted calibration cases. It cannot become a bound on restored nonzero-order native radiation, arbitrary detector angles/apertures, or all-energy total-spectrum accuracy.

## Implementation and regression anchors

`runner.axial.axial_zone_line_parts` classifies each reflection using the
integer pairing, with `native_line_rows` selecting continuation rows for the
zero-order part. Case views set `channeling=None` to prevent recursive
partitioning. The shared scorer adds the two incoherent reflection groups.
No transport, handoff, characteristic, bremsstrahlung, planar or off-case
algorithm is changed. `AXIAL_FIBRE_MODEL_MARKER` v3 prevents reuse of v2 output.

Independent numeric verification on an oblique direct cell and a rotated
reciprocal frame agreed with the integer predicate to less than $10^{-12}$;
the returned row groups and unchanged incident normalization were checked.

The anchors in `tests/montecarlo/channeling/test_axial_fibre_runner.py` check:

- synthetic partitions for three axes, with every reflection appearing once;
- actual si-fibre/A spectra equal the explicit two-part sum bit for bit;
- the restored nonzero-order zone contribution is positive at an oblique probe;
- zero-order reflections equal the native-continuation-only score;
- original exact-axis si-fibre/B output, source estimates, lookup bounds and
  checkpoint provenance remain covered by their existing tests.

The oblique probe tests the native scorer directly; it does not admit a
standard off-axis axial run or certify the fibre source there. The original
`free_free_upper` values (A 8.06%, B 18.9%) remain on-axis, in-cut, band-integral
source bounds; they do not bound restored native emission or detector totals.

Fresh-context verdict: **rederived** for the ownership equation. Regression
anchors establish the implementation of that equation, not the accuracy of
the straight-row channeled-state approximation. Human sign-off remains separate.

## Historical blanket-removal evidence (#994)

The following derivation and numerical records describe the previous v2 rule.
Its transverse-source free→free bounds remain evidence for that restricted
on-axis band. The blanket removal and former implementation assertions are
superseded by the reflection partition above.

## Inputs used before reading the implementation

- Ledger row and the `native_line_rows` / `_cxr_for_segments` docstrings.
- [`cr-axial-source-partition`](cr-axial-source-partition.md): the on-axis
  lowering current splits exactly into an axial block (final state below the
  emission barrier: bound→bound and free→bound) and a free→free block. There
  is no bound→free block, and the cross term vanishes under Hamiltonian
  evolution because the final subspaces are orthogonal.
- [`cr-axial-fiber-regression`](cr-axial-fiber-regression.md): the per-primary
  fibre estimator with a final-state boundary.
- Record `docs/validation/check-records/axial-source/free-free-994.json`
  (jobs 10497 and 10498, SHA `07c0f5f3e`, 20,000 primaries in 200 paired
  chunks per case).
- The [native-CBS section](../../research/physics/axial-material-source-992.md#native-cbs-and-the-freefree-block)
  of the #992 note and the #994 decision "A free→free ownership".
- The native line kernel's stated model (`mc_cxr_spectrum` docstring, Zhai SI
  Eqs. 5–7): each row is a free electron on a straight path with constant
  velocity, and the line sits at

$$
\omega_{\rm res}=\frac{\beta\,\hat{\mathbf v}\cdot\mathbf g\,c}{1-\beta\,\hat{\mathbf v}\cdot\hat{\mathbf n}}
$$

with amplitude $A_{\rm PXR}+A_{\rm CBS}$ and finite-time
$\operatorname{sinc}^2$ profile, normalized per incident line electron.

## Independent derivation

### Ownership partition of the zone

Let one line electron cross the zone, from the entrance face to its fibre
terminal depth. Expand its on-axis emission in the fibre model's finite
transverse eigensystem, with the boundary at the emission barrier
$U_{\max}$. Lowering currents need $E_i>E_f$, so by `cr-axial-source-partition`

$$
j=j_{\rm ax}+j_{\rm ff},\qquad
j_{\rm ax}=\Pi_{<}\,j,\quad j_{\rm ff}=\Pi_{>}\,j\,\Pi_{>},
$$

where $\Pi_{<}$ and $\Pi_{>}$ project the final state below and above
$U_{\max}$. The on-axis spectrum is quadratic in $j$, and the final subspaces
are orthogonal. The in-cut zone spectrum is therefore

$$
S_{\rm zone}^{\rm cut}(E)=S_{\rm ax}(E)+S_{\rm ff}(E)+X(E),
$$

with $X\equiv0$ under Hamiltonian evolution. `fiber_on_axis_source` with
`boundary="max"` estimates $S_{\rm ax}$. With `boundary="none"` it estimates
the whole retained in-cut current. The record's paired difference
`"none" − "max"` is therefore exactly the in-cut free→free block $F$ plus any
cross term $X$ the jump unraveling produces. The bound below is placed on
that measured difference, so it covers $X$ whether or not it vanishes.

The native line kernel on a zone row is the first-Born emission of a
plane-wave electron. Write the incident plane wave in the retained basis,
$\lvert\mathbf k\rangle=\sum_n c_n\lvert n\rangle$. To first order in the
string potential, the plane-wave line amplitude at reciprocal vector
$\mathbf g$ is the perturbative limit of the matrix elements
$\langle m\rvert j\lvert n\rangle$ summed over all retained $(n,m)$. That
includes bound→bound and capture ($n$ or $m$ below $U_{\max}$), not only
free→free. In the limit of a vanishing string potential, every retained
state is a plane wave, $S_{\rm ax}\to0$, and the native line equals the
free→free sum. With a finite potential, the native line approximates
$S_{\rm ax}+S_{\rm ff}$ on the same path. It is not an estimator of $S_{\rm ff}$
alone.

Scoring both the fibre source and the native line on a zone row would
count the zone's $S_{\rm ax}$ twice. Removing the native zone line removes an
overlapping estimate of the same emission, not a disjoint channel. After
removal, the persisted in-cut zone emission is $Y=\int S_{\rm ax}$. The true
in-cut value is $Y+F$.

Native rows outside the zone (outgoing launches) remain. Per the #992
ownership table, the fibre source does not represent emission after a
history leaves the retained subspace, so those rows do not overlap it.
Bremsstrahlung-only electrons (IDs $\ge N_e$) have no zone rows.

### Which native reflections the removal touches

The removal drops every reflection on a zone row, including those with
axial order $l\ne0$. These are native-owned under ADR-0018 and are not
scored by the fibre source. On axis, $\hat{\mathbf n}=\hat{\mathbf v}$, so
$1-\beta\hat{\mathbf v}\cdot\hat{\mathbf n}=1-\beta\approx1/(2\gamma^2)$. At
3.5 MeV, $\gamma=7.85$ and $2\gamma^2\approx123$. The smallest $l\ne0$
reflection of Si ⟨100⟩ is (111), with
$g_\parallel=2\pi/a=1.157$ Å$^{-1}$ and
$\hbar c\,g_\parallel=2.28$ keV. This puts the line near 280 keV, far above
the admitted 800 eV–10 keV band. For $l=0$, $\hat{\mathbf v}\cdot\mathbf g$
is set only by the tilt. With the A tilt of $1.536\times10^{-3}$ rad,
(220) at $\lvert\mathbf g\rvert=3.27$ Å$^{-1}$ gives
$\hbar\omega_{\rm res}\lesssim1.2$ keV, which is in band. At exact alignment
(B), $\hat{\mathbf v}\cdot\mathbf g=0$ and no line exists. Inside the band,
the removal therefore touches only the $l=0$ reflections, which are the
overlapping channel. The numeric probe below confirms this.

### Bound

From the record, the in-cut block relative to the `boundary="max"` control
yield over 800 eV–10 keV is:

| Case | $F/Y$ | SE | $u=F/Y+3\,\mathrm{SE}$ | lookup `free_free_upper` |
|---|---|---|---|---|
| si-fibre/A | 0.0783262 | 0.000751 | 0.0805795 | 0.08057945671257792 |
| si-fibre/B | 0.181937 | 0.002269 | 0.188744 | 0.18874404816708856 |

The absolute values are $F_A=4.17\times10^{-5}$ /sr against
$Y_A=5.32\times10^{-4}$ /sr, and $F_B=2.76\times10^{-5}$ /sr against
$Y_B=1.52\times10^{-4}$ /sr. The mean is about 100 SE (A) and 80 SE (B)
above zero, so $F\ge0$ holds empirically. It is also expected, because
$S_{\rm ff}$ is a power spectrum. This gives

$$
Y\le Y+F\le Y(1+u)
$$

for the band integral, in expectation, at the calibration numerics. This is
a statistical bound (3 SE of a paired mean), not a per-bin bound. It does
not cover free→free above the retained cut, which the ledger declares
unbounded.

### Normalization

The native spectrum is

$$
\mathrm{spec}(E)=\frac1{N_e}\sum_{r\notin\rm zone}\frac{dN_r}{dE\,d\Omega}.
$$

$N_e$ is the number of incident line electrons. It must not change when
rows are removed. Every array indexed by row must be filtered with the same
mask, and every array indexed by electron must be kept.

## Implementation comparison

- `native_line_rows` (`src/pyrite/montecarlo/runner/axial.py:166-200`)
  returns `segs` itself unless the `channeling` record has
  `model == "axial"`. It refuses an axial transport without a
  `channeling_zone` tag. Otherwise it returns a shallow copy in which every
  key in `_SEG_ARRAYS` and `channeling_zone` is indexed by
  `~channeling_zone`. Scalars (`Ne`, `thickness_ang`) and per-electron
  arrays (`initial_*`) are untouched, and the caller's `segs` is not mutated.
- The line kernel takes $N_e$ from the explicit `Ne` argument
  (`electron_limit`, `_setup.py:600-603`), which is `Ne_lines`, or from the
  scalar `segments["Ne"]`. Neither depends on row count, so normalization is
  per incident electron as derived.
- The row arrays the kernel reads (`r_mid`, `v_hat`, `L_ang`, `E_keV`,
  `E_repr_keV`, `t_*`, `elec_id`, `electron_id`, `flight_id`, `layer`,
  `track_id`) are all in `_SEG_ARRAYS`. The kernel's own row operations,
  `_clip_segments_to_cutoff` and `expand_escape_pieces`
  (`_formation.py:173`), also iterate `_SEG_ARRAYS`. The kernel creates
  `line_parent_L_ang` and `line_piece_fraction` itself, after the filter.
  `initial_r_ang` and `initial_t0_ang` are per-electron arrays, read
  `[:Ne]` only on coherent paths, and axial is incoherent-only. In the
  produced A and B transports, the probe found no array of the full row
  length left unfiltered.
- `_cxr_for_segments` (`case_driver.py:473-569`) applies the filter before
  both of its outputs: `mc_cxr_spectrum` (PXR+CBS lines) and the two-beam DTR
  branch. It feeds nothing else. Characteristic (`_characteristic_from_segments`)
  and bremsstrahlung (`emission.continuum`) read `batch.segments`, the
  unfiltered staged copy, through separate adapters
  (`emission_adapters.py`). The filter is local to the line call, so they are
  unchanged. Every native line route in the runner reaches the kernel
  through `_cxr_for_segments`: the emission adapter, line roulette,
  `cxr_grid`, and the energy-grid checks. The routes that subset rows by
  `_SEG_ARRAYS` alone are line roulette `_rows_of` and the GPU
  `iter_electron_blocks`. On such a route `channeling_zone` would be left
  full length, and the boolean index would fail loudly rather than silently.
  Both are unreachable for axial cases: roulette refuses `channeling`
  (`_line_roulette.py:23`), and axial is CPU-only.
- The bounds in `ADMITTED_AXIAL_CASES`
  (`src/pyrite/quantum/_axial_envelope.py:115-117,136-138`) equal the record
  values and mean + 3 SE to the last digit. The lookup test checks exact
  equality.

## Numeric checks

These were run independently with a scratch probe that is not part of the
repository. The cases were lowered from public objects as in the tests, each
with 8 line histories, 6 bremsstrahlung histories and a 47-point
800 eV–10 keV grid, and each marked runnable only for the probe.

| Check | si-fibre/A (0.088°) | si-fibre/B (0°) |
|---|---|---|
| zone rows / line electrons | 8 / 8 | 8 / 8 |
| zone `v_hat` | beam direction, tilted 1.536 mrad | $(0,0,1)$ |
| persisted `spec` equals kernel on non-zone rows | bit for bit | bit for bit |
| zone-only native line, in band | $2.006\times10^{-4}$ /sr | exactly $0$ |
| of which $l=0$ / $l\ne0$ | $2.006\times10^{-4}$ / $0$ (8 of 50 reflections are $l=0$) | $0$ / $0$ |
| `spec_channeling` in band | $4.02\times10^{-4}$ /sr | $9.50\times10^{-5}$ /sr |
| `spec_characteristic` equals recompute on full rows | bit for bit | bit for bit |
| full-length arrays left after filter | none | none |

For A, the removed zone-row native yield at 8 histories, $2.0\times10^{-4}$ /sr,
lies between the ledger's $1.56\times10^{-4}$ /sr at 4 histories and
$2.33\times10^{-4}$ /sr at 24 histories. It is about five times the measured
block, $4.17\times10^{-5}$ /sr. This is consistent with the native line
re-scoring the axial block rather than the free→free block alone. For B, the
zone-only line is exactly zero, so B's `spec` is unchanged.

## Limiting cases

- No channeling record, `None`, or a planar record: `native_line_rows`
  returns the same object (`is`), so output is unchanged byte for byte. An
  off case never enters, because `_cxr_for_segments` checks
  `case.get("channeling") is not None`.
- Exact alignment (B): the zone-only native line is exactly zero, so `spec`
  is unchanged. Confirmed numerically.
- Every electron retained: no native line rows remain, and `spec` is the
  empty sum, zero, divided by unchanged $N_e$.
- Vanishing string potential: $S_{\rm ax}\to0$, and the native line becomes
  the free→free estimate. The rule must not apply there, and it cannot,
  because only admitted lookup entries reach it.

## Anchors

`UV_CACHE_DIR=$HOME/.cache/pyrite-uv-cache uv run pyrite-dev test --slow tests/montecarlo/channeling/test_axial_fibre_runner.py tests/montecarlo/quantum/test_channeling_axial_envelope_lookup.py`:
45 passed (rc 0). The named anchors exist and pass:
`test_axial_zone_rows_score_no_native_lines`,
`test_native_line_rows_keeps_other_cases_and_refuses_an_untagged_axial_transport`,
`test_admitted_cases_persist_the_fibre_source_and_their_free_free_bound`, and
`test_free_free_values_equal_their_paired_record`.

## Findings

1. **No implementation discrepancy.** The filter, normalization, scope and
   bounds match the independent derivation.
2. **Anchor gap (minor).** The ledger's Checks list says "B is unchanged",
   but no named anchor asserts that B's zone-only native line is zero. The
   A test pins only the A removal. The probe shows the value is exactly
   $0$. A test assertion would turn this from evidence into an anchor. Separately, the #994 issue text
   gives B's zone-row native CBS as "≈1e-5 /sr", which disagrees with the
   exact zero found here for straight zone rows along the axis. The issue
   text should be corrected or its origin explained.
3. **Stale research note.** `docs/research/physics/axial-material-source-992.md:333`
   (table row "Zone row, tilted incidence … supplied natively") and lines
   355–357 ("A takes the block from native CBS on its tilted zone rows")
   still state the superseded ownership. The ledger Notes say they are
   superseded, but the note itself does not.
4. **Interpretation.** $Y\le Y+F\le Y(1+u)$ holds for the expected band
   yield at the calibration numerics. The persisted `spec_channeling` from a
   finite run carries its own statistical error on top of this.

## Verdict

`rederived`. The production anchor
`test_axial_zone_rows_score_no_native_lines` pins the removal in the
production runner for tilted A, against the native kernel on non-zone rows,
bit for bit. With the lookup test pinning the bounds to the record, this
supports `anchored` once the task owner applies `rederived`. Adding a
B-unchanged assertion (finding 2) would close the last unpinned Checks item.
