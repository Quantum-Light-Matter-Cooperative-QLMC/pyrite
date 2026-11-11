# `cr-planar-free-free-block`: in-cut planar free→free block of the lowering current

**Status:** see the [ledger row](../ledger-channeling.md#cr-planar-free-free-block). Fresh-context independent verification (2026-11-10, #1055, HEAD `b3cacadac`); [re-verified](#re-verification) at `fe8cb9909`.

The claim has four parts:

1. The diagnostic catalog `emission_lines(..., final_states="tracked")` adds
   every in-block transition between tracked above-barrier states as
   `KIND_FREE_FREE`. It uses the same operator, rate, widths, line groups and
   per-kind cut as the scored kinds.
2. The bound→bound and capture rows are unchanged. The tracked spectrum
   minus `free_free_spectrum` therefore equals the production spectrum.
3. Two limiting cases hold: zero for a flat potential, and the nearly-free
   $k\to k-g$ line far above the barrier.
4. Two auxiliary rules apply: the 1 keV forward floor and the zero
   $\gamma$-exponent fallback.

## Inputs used before reading the implementation

- The ledger rows `cr-planar-free-free-block`, `cr-dipole-rate`,
  `cr-capture-yield`, `cr-line-shape`, `cr-yield-assembly` and
  `cr-axial-source-partition`.
- From `cr-dipole-rate`: the generic first-order operator
  $A_{fi}=\langle f\rvert(\mathbf v\cdot\boldsymbol\epsilon^*/c)\,e^{-i\mathbf k\cdot\mathbf r}\lvert i\rangle$
  with $\mathbf v=(p_x/\gamma m,0,\beta c)$. Its path rate is
  $dN/dz=\tfrac43\alpha\gamma^2(\Delta E_\perp/\hbar c)^3\lvert x_{fi}\rvert^2/\beta$.
  The estimator is $\langle d_c^\dagger d_c\rangle_{\rho(z)}$ per cluster $c$
  of transitions that share a final state.
- From `cr-line-shape`: one secular generator per band-index pair $(i,f)$
  across Bloch blocks, with coherence transfer between blocks of the same pair.
- From `cr-axial-source-partition`: the on-axis lowering current splits by a
  final-state projector into blocks with orthogonal final subspaces and no
  cross term.
- The [#1055 research note](../../research/physics/planar-zone-row-cbs-1055.md)
  and the record `docs/validation/check-records/planar-zone/zone-cbs-1055.json`
  (17 MeV, 20 µm, `H = 16`, 16 Bloch cells, margin 300 eV).

## Independent derivation

### Transverse problem

Take $H_\perp=p_x^2/(2\gamma m)+U(x)$ with $U(x)=\sum_m U_m e^{imgx}$ and
$g=2\pi/d_p$. Its Bloch eigenstates are $\lvert n,\kappa\rangle$ with
energies $E_n(\kappa)$. Let $B$ be the emission barrier, $\max U$, and $M$
the margin. The solver tracks $S=\{E<B+M\}$. States above $B+M$ are a sink.

### Exact first-order operator on axis

On axis, $\hat{\mathbf n}=\hat{\mathbf z}$. Both polarizations are transverse
to $z$, so $\boldsymbol\epsilon\cdot\hat{\mathbf z}=0$. The photon has
$k_x=0$, so $e^{-i\mathbf k\cdot\mathbf r}$ acts only on the longitudinal
plane wave and gives the energy balance
$\hbar\omega(1-\beta)=E_i-E_f$. What remains is

$$
A_{fi}=\frac{\epsilon_x^*}{\gamma m c}\,\langle f\rvert p_x\lvert i\rangle ,
$$

which is diagonal in $\kappa$. With $[x,H_\perp]=i\hbar p_x/(\gamma m)$,

$$
x_{fi}=\frac{i\hbar\,(p_x)_{fi}}{\gamma m\,(E_i-E_f)}
=-\,\frac{i\hbar^2\,q_{fi}}{\gamma m\,(E_f-E_i)},\qquad p_x=\hbar q .
$$

The rate is $dN/dz\propto\Delta E^3\lvert x_{fi}\rvert^2\propto\Delta E\,\lvert q_{fi}\rvert^2$.
On axis this velocity form is the exact first-order operator, not a
long-wavelength approximation. It holds for any two eigenstates of the same
$H_\perp$, whether bound or free.

### Partition by the final state

Write the total lowering current in a block as
$j=\sum_{E_i>E_f}a_{fi}\lvert f\rangle\langle i\rvert$, over $i,f\in S$. Let
$\Pi_b$ and $\Pi_>$ project onto final states below and above $B$. Then
$j=\Pi_b j+\Pi_> j$. Because $E_f<E_i$, there is no bound→free term. The
block $\Pi_> j$ has $i,f$ both free:
$B\le E_f<E_i<B+M$. The operator $d_c^\dagger d_c$ of a cluster contains
$\lvert i'\rangle\langle f\vert f'\rangle\langle i\rvert$. When every cluster
shares one final state, a cluster lies wholly in one block, and
$\sum_c d_c^\dagger d_c$ splits into the bound-final part and the free-final
part with no cross term. The expectation is linear in $\rho$, and
$\rho(z)$ does not depend on the catalog. The per-line yields of the
bound-final rows are therefore unchanged.

The spectrum is the sum of the secular mode profiles of each line group.
It partitions exactly if, and only if, three conditions hold:

- **(P1)** the relative cut of the bound-final kinds does not depend on the
  free→free rows;
- **(P2)** clusters do not mix kinds, which holds automatically when they
  share a final state;
- **(P3)** every line group's generator and kind involve only one kind.

P3 is the non-trivial condition. A group is one band-index pair
$(u,l)$ across all blocks. If band $l$ is bound in some blocks and free in
others, the same pair $(u,l)$ is capture in the first set and free→free in
the second. The coherence-transfer generator then couples the two kinds.
This happens when the band "straddles" the barrier across the Bloch zone.
Nothing in the potential forbids it.

### Flat potential

When $U\equiv0$, the states of block $\kappa$ are the plane waves
$e^{i(\kappa+mg)x}$. Off its diagonal, $q_{fi}$ is zero, except inside an
exactly degenerate pair at $\kappa=0$ or $g/2$, where $\Delta E=0$. The rate
$\propto\Delta E\lvert q_{fi}\rvert^2$ vanishes in both cases. The block is
exactly zero.

### Nearly-free line

Far above the barrier, first-order perturbation theory in $U$ gives

$$
\lvert i\rangle\simeq\lvert k\rangle+\frac{U_{-g}}{E_k-E_{k-g}}\lvert k-g\rangle,\qquad
\lvert f\rangle\simeq\lvert k-g\rangle+\frac{U_{g}}{E_{k-g}-E_k}\lvert k\rangle .
$$

This gives

$$
\langle f\rvert p\lvert i\rangle=-\frac{\hbar g\,U_{-g}}{\Delta E_1},\qquad
\Delta E_1=E_k-E_{k-g}=\frac{\hbar^2(2kg-g^2)}{2\gamma m}.
$$

The rate $\propto\lvert U_g\rvert^2g^2/\Delta E_1$. The classical crossing
frequency is $v_xg=\hbar kg/(\gamma m)$. The quantum line is lower by the
factor $1-g/(2k)$, the transverse recoil. The second harmonic $k\to k-2g$
has $\Delta E_2=\hbar^2(4kg-4g^2)/(2\gamma m)$. At $k=6g$ its rate ratio is

$$
\frac{r_2}{r_1}=\Bigl\lvert\frac{U_{2}}{U_{1}}\Bigr\rvert^2\frac{4\cdot11}{20}.
$$

For Si(110), $\lvert U_2/U_1\rvert=0.341$ (profile coefficients 4.32 and
1.47 eV), so $r_2/r_1=0.26$. The $k\to k-g$ line is the strongest, as the
claim says. The other lowering partners need $U_{\ge3}$ or are not lowering.
For Si(110) at 17 MeV, $\gamma=34.268$ and $1-\beta=4.2587\times10^{-4}$.
With $k=6g$ ($\psi=2.21$ mrad, $E_\perp=42.9$ eV), $\Delta E_1=13.10$ eV,
so the forward line is $\Delta E_1/(1-\beta)=30.75$ keV. The classical value
is 33.55 keV, 9.1% higher. The test's 3% tolerance therefore separates the
quantum and classical energies. The units are $\hbar^2/2m=3.810$ eV Å$^2$ and
$g$ in Å$^{-1}$, which give eV. The forward energy is positive for $k>g/2$.

### Auxiliary rules

**(a) Floor.** The forward energy is the maximum over observation angle,
because $1-\beta\cos\theta\ge1-\beta$. A transition below 1 keV forward is
therefore below 1 keV in every direction. The floor corresponds to
$\Delta E_\perp<1000(1-\beta)=0.426$ eV at 17 MeV. These are the
$\kappa\approx0$ and $g/2$ doublets split by $2\lvert U_{2q}\rvert$, whose
rate $\propto\Delta E\,q^2$. Their only route into a band above 5 keV is
the Lorentzian tail. The dropped emission is non-negative, so the floor
biases the in-band block low by that tail.

**(b) $\gamma$-exponent fallback.** The exponent enters only the Gaussian
RMS $\sigma=\lvert E_m\rvert s\,\sigma_\gamma/\gamma$. The Voigt profile has
unit area, so the band integral cannot change. Only the edge leakage of a
line within about $\sigma$ of a band edge changes. The fallback keeps the
kinematic part of $s$ (about 2 forward).

### Validity of dipole lines and Bloch widths for free→free

- **On axis:** exact at first order, as shown above, for free→free as for
  capture.
- **Off axis:** $k_x=\omega\theta/c$ moves the final state to block
  $\kappa-k_x$. The catalog neglects this. For a free final state the line
  then moves by $(\partial E_f/\partial\kappa)k_x$, which is first order in
  $\theta$. For deep bound finals $\partial E_f/\partial\kappa\approx0$.
  Free→free therefore needs observation nearer the axis than capture does.
- **Widths:** the secular, same-pair coherence transfer drops transfer
  into other pairs of similar frequency. The free→free spectrum is denser
  than capture, so its shapes are approximate in the same way, more so.
  Per-line yields, and the group-mode sum $\sum_m w_m$, are exact. The block
  integral therefore does not depend on this approximation.

## Source-to-code agreement

All lines refer to `src/pyrite/montecarlo/channeling/emission.py`.

| Item | Derivation | Code | Agreement |
|---|---|---|---|
| Dipole | $x_{fi}=-i\hbar^2q_{fi}/(\gamma m(E_f-E_i))$ | `_block_dipoles`: `-1j*2*HBAR2_2M/gamma*momentum/gap`, `gap[b,a]=E_b-E_a` | equal |
| Rate | $\tfrac43\alpha\gamma^2(\Delta E/\hbar c)^3\lvert x\rvert^2/\beta$ | `dipole_rate_per_ang` | independent recomputation: max rel. diff $1.8\times10^{-15}$ on 4583 (Si) and 2398 (C) catalog members |
| Free→free set | $B\le E_f<E_i<B+M$ | `final_ok = bound or subspace` (l. 426); `free_final = not bound[lower]` | equal |
| Floor | forward $\Delta E/(1-\beta)<1$ keV dropped | l. 331, `_denominator(gamma, 0.0)` | equal |
| P1, per-kind cut | per block and kind | l. 455–459 | holds |
| P2, clusters | key (block, final) | l. 485–503: consecutive in `(block, lower, delta)` | holds; finals are disjoint by kind within a block |
| P3, line groups | must not mix kinds | `by_pair`/`by_rep` keyed on `(upper, lower)` only (l. 464, 538); a group's spectrum goes to `kind[members[0]]` (l. 300) | **fails when a band straddles $B$** (finding 1) |
| Fallback | $s\to$ kinematic part only | l. 353–357 | equal; never triggered on the tested grids |
| Pattern/Stokes | free→free is an $x$-dipole | same `boosted_dipole_pattern`/`dipole_stokes` | equal |

## Numeric checks

These were run with scratch probes outside the repository. Every
configuration is 17 MeV, 293.15 K and margin 300 eV, unless stated
otherwise.

| Check | Result |
|---|---|
| Fourier coefficients, Si(110) | $\lvert U_{1..3}\rvert=4.32,\,1.47,\,0.71$ eV, so the $k\to k-g$ line dominates ($r_2/r_1=0.26$) |
| Rows outside `KIND_FREE_FREE` identical (all fields), Si 4/8, Si 16/16, C 16/16 | yes; 184/389, 1777/5729, 864/3076 rows (default/tracked) |
| Bands straddling $B$ at 17 MeV, Si and C, 4/8 and 16/16 grids | none |
| Floor removed (1 µeV), Si 4/8, 2 µm, on axis, 5–60 keV free→free band | $\times1.0016$ at $\psi=0$; $\times1.0001$ at 2.2 mrad |
| Same, Si 16/16 production grid, $\sigma_E=17$ keV | $\times1.0004$ at $\psi=0$; $\times1.0000$ at 2.2 mrad |
| Zero-exponent fallback rows, floor in place or removed | 0 on every grid above |
| Free→free rate in clusters whose members spread by more than 1% of the representative $\Delta E$ | 0 (Si and C, 16/16) |
| Off-axis final-state shift $(\partial E_f/\partial\kappa)k_x/\Delta E$ per mrad, strongest free→free rows (31–38 keV) | $2.4$–$3.6\times10^{-3}$ |
| Band straddling $B$ on a 16-cell grid, 3–100 MeV (60 energies) | Si: 28 of 60 energies, e.g. 5.1–7.3 and 11.1–14.1 MeV; C: 33 of 60 |
| **Si 12.49 MeV, 16/16 (band 3 straddles)** | 27 mixed-kind groups. Capture `hwhm_eV` changes by up to 0.17 eV. `capture_spectrum` changes by 4.7% (on axis) and 1.7% (1 mrad) of its peak. $\int(\text{tracked}-\text{FF}-\text{base})$ over 2–60 keV is $-1.6\times10^{-5}$ /sr against an FF block of $1.5\times10^{-4}$ /sr (−11%) on axis, and $-2.8\times10^{-5}$ against $4.4\times10^{-4}$ (−6%) at 1 mrad |
| `include_capture=False` | default 0 capture rows; tracked 162 |

The anchors were run with
`UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/montecarlo/quantum/test_channeling_emission.py`:
37 passed (rc 0), including the four named anchors.

## Limiting cases

- **Flat potential:** exactly zero. The Hamiltonian is diagonal, so the
  eigenvectors are unit vectors and $q_{fi}=0$ off the diagonal. A
  degenerate pair has $\Delta E\le10^{-12}$ eV, which is excluded, and
  would in any case lie below the floor. The anchor passes. The catalog is
  empty, so the spectrum is identically zero.
- **Nearly free:** the test puts the free→free peak within 3% of 30.75 keV.
  The classical 33.55 keV would fail at that tolerance. The derivation also
  confirms that $k\to k-g$ is the strongest line for Si(110).
- **Margin 0:** no tracked free state, so the block is empty, as for
  capture.

## Findings

1. **Discrepancy: the exact partition fails when a band straddles the
   barrier across the Bloch blocks.** Line groups are keyed on the
   band-index pair alone (`emission.py:464` `by_pair`, `:538` `by_rep`).
   Kind is assigned per state and block (`bound = energies < barrier`,
   `:424`). When band $l$ is bound in some blocks and free in others, the
   tracked catalog puts capture rows $(u,l)$ and free→free rows $(u,l)$ in one
   group. This has three effects:
   - the capture rows' secular `hwhm_eV` changes (gain from line 481);
   - the group modes change;
   - the whole group's spectrum goes to `kind[members[0]]` (`:300`).

   At Si(110) 12.49 MeV on the production grid, `capture_spectrum` changes
   by 4.7% of its peak, and about 11% of the reported free→free band on axis
   is capture emission. The statements "bound→bound and capture rows are
   unchanged" and "tracked minus `free_free_spectrum` equals production" are
   false there. Straddling occurs at about half of the energies in 3–100 MeV
   for both Si and C. It does not occur at 17 MeV on the anchor or
   production grids, so the anchors and the #1055 measurement are
   unaffected. The driver's $10^{-10}$ partition guard (`checks/planar_zone_row_cbs.py:283`)
   would refuse a straddling case.
   - **Fix options** (the task owner chooses; the verifier did not edit code):
     (i) key `by_pair`/`by_rep` on `(upper, lower, kind)`. This restores the
     exact partition: inter-kind transfer is dropped, as production already
     drops it.
     (ii) Keep the coupled group, but split its modes by kind.
     (iii) Restrict the claim, and refuse `final_states="tracked"` when a
     final band straddles $B$.
   - **Anchor:** add a test at a straddling energy, for example Si(110)
     12.5 MeV with `n_cells=16`, `max_order=16`.
   - **Same mechanism in the default catalog:** an *upper* band straddling
     $B$ makes line/capture mixed groups. At 12.49 MeV, $(3,0)$ and $(3,2)$
     are assigned to capture. This affects the `capture_spectrum`/line
     split of `cr-capture-yield` and `cr-yield-assembly`, not their total.
     Report it against those claims.
2. **Scope (minor).** With `include_capture=False`, the tracked catalog
   turns capture on (`:425`; 0 → 162 capture rows). The partition holds only
   for the default `include_capture=True`. The docstring says this. The
   claim should too.
3. **Floor (a): negligible here.** Removing the floor raises the 5–60 keV
   block by at most 0.16% (4/8 grid) or 0.04% (production grid). The bias is
   one-sided (low), as derived. "Sub-eV intra-doublet pairs" should read
   "transverse splittings below $1000(1-\beta)$ eV, 0.43 eV at 17 MeV", so the
   threshold is explicit.
4. **Fallback (b): dormant.** It never triggered on any probed grid, even
   with the floor removed. It changes only the Gaussian width, never a band
   integral.
5. **Validity (item 4).** On axis, the operator is exact first order for
   free→free as for capture: the velocity form, with no long-wavelength step.
   Off axis, neglecting the final-state Bloch shift moves free→free lines by
   about $3\times10^{-3}$ per mrad at 30–40 keV. That is about 4% at
   $\theta\approx1/\gamma$, while bound finals are unaffected. "Near the beam"
   should be quantified: $\theta\ll1/\gamma$, or within the aperture used. Line
   shapes inherit the secular approximation of `cr-line-shape` and are less
   reliable than capture's. Block integrals do not depend on it. Row-level
   `hwhm_eV` of weak free→free rows reaches 26 eV (Si) and $3\times10^3$ eV
   (C) on the production grid, because the gain is normalized by a small
   $\lvert a_k\rvert^2$. It enters only the clustering tolerance, and no
   cluster merged members more than 1% apart in $\Delta E$.

## Initial verdict

`discrepancy`. The equations match the code: operator, rate, final-state
set, floor, per-kind cut, cluster key, flat and nearly-free limits, and the
17 MeV partition. The first divergent convention is the line-group key
`(upper, lower)` without kind (`emission.py:464,538`), together with
whole-group kind attribution (`:300`). It breaks the claimed exact partition
whenever a final band straddles the barrier across Bloch blocks. Once fix
(i) is applied, or the claim is restricted under (iii) with a refusal, plus
a straddling-energy anchor, the remaining claim supports `rederived`.

## Re-verification

Date: 2026-11-10, at `fe8cb9909`, which was rebased onto `main`. This round
checks the fix for finding 1 and the new claim text. It is not a new
independent derivation.

**Fix.** `_group_key` (`emission.py:308-310`) keys `by_pair` and `by_rep`
on `(upper, lower, kind == KIND_FREE_FREE)`. This is option (i), applied to
the free→free flag only. Within one block, clusters never mix free→free with
the other kinds, because their final states are disjoint. A row's
representative kind therefore carries the same flag as every one of its
members, and `by_rep` finds its `by_pair` generator. The default catalog has
no free→free rows, so its keys only gain a constant `False`. Its groups are
unchanged.

**Probes (same scratch scripts as above).**

| Check | Result |
|---|---|
| Si(110) 12.49 MeV, 16/16, margin 300 eV (band 3 straddles) | Every row field outside `KIND_FREE_FREE` is identical, including `hwhm_eV`, which differed by up to 0.17 eV before |
| Same, groups mixing free→free with other kinds | 0. Before the fix there were 27 mixed groups, all but the line/capture ones of #1091 involving free→free |
| Same, on axis: `capture_spectrum` | bitwise equal to production |
| Same, on axis: $\max\lvert\text{tracked}-\text{FF}-\text{base}\rvert$/peak | $1.3\times10^{-16}$; band difference $8\times10^{-20}$ /sr |
| Same, 1 mrad: `capture_spectrum` | bitwise equal to production |
| Same, 1 mrad: $\max\lvert\text{tracked}-\text{FF}-\text{base}\rvert$/peak | $1.4\times10^{-16}$; band difference $9\times10^{-20}$ /sr |
| Free→free band, 2–60 keV, before → after | on axis $1.496\to1.334\times10^{-4}$ /sr (−10.8%); 1 mrad $4.413\to4.118\times10^{-4}$ /sr (−6.7%). This is the capture emission that had been misattributed, and the change matches the earlier residual |
| 17 MeV, Si and C, 16/16 | rows identical outside free→free; no mixed groups, as before |
| Remaining mixed groups at 12.49 MeV | 3 line/capture groups ($(3,0)$, $(3,2)$, $(3,1)$). They are identical in both catalogs: the upper-band mislabel filed as #1091, outside this claim |
| `pyrite-dev test tests/montecarlo/quantum/test_channeling_emission.py` | 38 passed (rc 0), including `test_tracked_partition_is_exact_when_a_final_band_straddles_the_barrier` |

**Claim text.** The revised ledger row states the separate free→free
groups and restricts the partition to `include_capture=True`. It gives the
floor as a transverse splitting of $1000(1-\beta)$ eV with a one-sided bias
of at most 0.16%, and it requires $\theta\ll1/\gamma$. This covers findings
2, 3 and 5. The 0.16% figure comes from the 4-cell/`H = 8` grid at 2 µm.
The production grid gave 0.04%.

**Remaining approximation, not blocking.** Where a final band straddles
the barrier, the separate groups drop coherence transfer between the
capture blocks and the free→free blocks of the same band pair. Production
already drops transfer into the free→free blocks. This is the same
secular truncation as `cr-line-shape`. Per-line yields and every band
integral are unaffected, because group weights sum to their yields. Only
the mode shapes of those few groups are affected. The claim could say so
in a single sentence. It is optional.

### Verdict

`rederived`. With the fix, the derivation and the code agree on every item.
The exact partition holds to rounding at the straddling configuration that
failed before, both on axis and at 1 mrad, and it still holds at 17 MeV. A
straddling anchor pins it. Suggested ledger edit: the task owner sets
`rederived` and replaces "Re-verification pending" with this verdict. The
row's anchors would then support `anchored`. Sign-off remains with a human
(#277).

