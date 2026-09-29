# `line-grid-resonance-local-spacing`

Ledger row: [`line-grid-resonance-local-spacing`](../ledger-core-coherent-physics.md#line-grid-resonance-local-spacing). Status: rederived. Validation: `line-grid-resonance-local-spacing`.

## Resolution rule

The finite-flight line has first-zero width $w_i=\pi/a_i$. The opt-in `resonance-local` policy keeps a coarse backbone and places finer windows where measured lines resonate. For a line narrower than the backbone, a halo of half-width $D_i=w_i/(\pi^2\eta)$ follows from the one-sided sinc-squared tail bound: at most $\eta$ of that line's mass lies beyond each halo edge, so at most $2\eta$ outside the halo in total. Within the halo, the chosen power-of-two spacing is at most $w_i$ unless the global sinc spacing floor is coarser. One-hundred-eV bins take the finest overlapping requirement; adjacent bins at the same level merge into one window.

A spacing join breaks the uniform-grid exactness of node quadrature. The fresh-context verification below found that a population concentrated near a join loses about $(h/w)^2$ of its mass there, beyond the $10^{-3}$ intrinsic-source budget. Lines at least as wide as the backbone had no halo, so any join could sit inside them. Every line with $w_i<2h_{\rm backbone}$ therefore also requires spacing $\le\max(w_i/2,h_{\rm floor})$ within $\pm10\,w_i$ of its resonance (`LOCAL_CORE_WIDTHS`, `LOCAL_CORE_FRACTION`), so no piece coarser than that lies there. Joins between finer levels can still fall anywhere in the core. Under node quadrature the core lowers join error but bounds no join term: the re-check measured $6.3\times10^{-3}\to1.1\times10^{-3}$ for $w\approx3.1$ eV lines 3 eV below a join, up to $5\times10^{-3}$ per line when $w\approx h_{\rm backbone}$ at its own core edge, and about $10^{-2}$ for lines wider than twice the backbone. `resonance-local` is therefore refused unless it is paired with `bin-mean` quadrature ([`sinc-bin-integration`](../radiation-physics/sinc-bin-integration.md)), which the `high_energy` profile selects. Each bin then carries the closed-form mean of the line over that bin; the bin means telescope, so integrated yield depends on the axis only through its two outer bin edges and is exact at any spacing. Every re-check population measured bin-mean yield error equal to its band-edge tail truncation ($\le6.3\times10^{-5}$). The halo and core then change only how mass is distributed among bins. No shape bound is derived: at $h\le w/2$ the re-check measured peak and FWHM smoothing of 1–23% depending on resonance spread, so the ledger's line-shape comparison remains required.

The floor is the existing weighted global sinc rule. It can permit narrow lines to alias within its own budget. Under the required `bin-mean` quadrature the floor and halo change only how mass is distributed among bins, not integrated yield; their effect on line shape and detected counts is measured against a full-ceiling uniform reference on identical trajectories (`bandwidth_check.shape_and_counts`).

For one isolated line, the unbinned window spans $E_i\pm D_i$, clipped to the selected bandwidth, at spacing no larger than $w_i$ when the global floor permits it, and no larger than $w_i/2$ within $E_i\pm10\,w_i$. The policy changes only the identity of cases that select it; the `high_energy` profile does.

Production Ne=20,000 comparisons, float32/float64 agreement, detected counts, and peak memory remain outstanding. Human sign-off pending.

## Identical-trajectory comparison (SLURM 239)

`bandwidth_check reference --resolution local --compare-resolution uniform` (`shape_and_counts`), FP64 CUDA reference to the kinematic ceiling, identical segments, then FP32 replay. EEDL bremsstrahlung fallback on the remote host.

| case (5 MeV h-BN, seed 0) | nodes local / uniform / ceiling | line yield local | EagleXO counts local | Timepix3 counts local / uniform | Timepix3 detected L1 local / uniform | dominant FWHM local / uniform (ref) | FP32 yield | line eval s local / ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 um 10/100, Ne 2,000 | 8,744 / 10,736 / 1,415,387 | 2.4e-5 | -6.6e-7 | 1.7e-4 / 8.2e-5 | 1.7e-3 / 8.5e-4 | -15.8% / -4.3% (3.05 eV) | 1.4e-7 | 0.5 / 26 |
| 1 um 80/180, Ne 2,000 | 9,631 / 25,433 / 5,436,172 | 2.2e-5 | -1.5e-6 | 6.6e-6 / 1.1e-6 | 1.4e-5 / 1.4e-6 | -4.6% / +0.3% (2.52 eV) | -1.4e-6 | 0.6 / 31 |
| 1 mm 10/100, Ne 200 | 70,080 / 362,923 / 5,899,830 | -3.2e-5 | 1.4e-6 | -5.0e-7 / -6.4e-10 | 1.2e-5 / 6.3e-8 | +0.7% / +0.4% (1.92 eV) | -1.0e-6 | 60 / 191 |
| 1 mm 80/180, Ne 200 | 31,308 / 53,140 / 6,371,941 | 1.9e-5 | -4.5e-7 | 5.0e-7 / 3.1e-7 | 9.2e-7 / 4.0e-7 | -2.4% / -1.5% (0.87 eV) | -3.3e-6 | 40 / 245 |

Line yield differences are the measured-stop truncation (true fraction above the stop 1.9-2.4e-5; the 1 mm 10/100 local value includes -3.2e-5 from the 106 keV stop). Detected counts agree to <=1.7e-4 (Timepix3, grid term) and <=1.5e-6 (EagleXO) -- inside the 2e-4 interpolation share. The Timepix3 detected-shape L1 is <=6e-5 except 1 um 10/100 (1.7e-3, uniform 8.5e-4): there 3 eV lines on a 1.2 eV reference have a 0.35 sub-cell split bound at 100 eV, so part of it is sub-cell ambiguity in the 50 eV channel split, not grid error. The dominant-line FWHM is the one real shape limitation: bin-mean averaging at h ~ w/2 widens a 3 eV line by up to 16% (uniform axis 4%). The detector resolution is far coarser, so detected shape is unaffected, but an intrinsic FWHM finer than a few eV is not preserved by `resonance-local`. FP32 matches FP64 to <=3.3e-6 in yield. Peak host / device memory is 3.2 / 6.0 GiB. Separately, Timepix3 native-bin scoring (`apply_native`, used by physical-detector acquisition; `apply` already splits by overlap) assigns cells to channels by node and moves detected counts by up to 1.8e-3 depending on the axis (`node_rebin_counts_rel`) -- a detector-resampling term above its 2e-4 share, independent of this policy (#219).

## Independent verification

Fresh-context verifier, 2026-09-27, branch `issue-192-high-energy-line-grid`. This section is appended; the author's text and status above are unchanged. Verdict for the claim as worded: **`rederived`**. The halo half-width, the spacing rule, floor quantisation, binning, merging and the bandwidth edge all match an independent derivation. The Notes contain one stale statement (see [Findings](#findings)), and the checks below show that the claim does not bound integrated-yield error. It must not be read as an error bound.

### Independent derivation (before reading the implementation)

Take a line with profile $f(x)=\operatorname{sinc}^2(a x/\pi)=\sin^2(ax)/(ax)^2$, where $x=E-E_i$. Its first zero is at $w=\pi/a$ and its whole mass is $M=\int_{-\infty}^{\infty}f\,dx=\pi/a=w$. On one side,

$$
\int_D^\infty \frac{\sin^2(ax)}{(ax)^2}\,dx\;\le\;\int_D^\infty\frac{dx}{a^2x^2}=\frac{1}{a^2D},
\qquad
\frac{1}{a^2 D}\Big/\frac{\pi}{a}=\frac{1}{\pi a D}=\frac{w}{\pi^2 D}.
$$

Setting the one-sided fraction equal to $\eta$ gives

$$
\boxed{D=\frac{w}{\pi^2\eta}}\qquad\text{per side, so at most }2\eta\text{ of the mass lies outside }[E_i-D,\,E_i+D].
$$

Since $\sin^2$ averages to $\tfrac12$, the true far tail is about $\eta$ in total. The bound is therefore conservative by a factor of 2, and the two sides cancel that factor. Useful identity: $aD=1/(\pi\eta)$, so the envelope at the halo edge is $f(D)\le\pi^2\eta^2$.

Spacing: $\hat f(k)$ is a triangle supported on $\lvert k\rvert\le 2a$. By Poisson summation, a uniform infinite trapezoid of step $h$ at any offset is exact when $2\pi/h\ge 2a$, i.e. when $h\le\pi/a=w$. The aliased replica touches the support edge only where the triangle is zero. This is the reliance on `line-grid-sinc-convergence`, and it is used correctly.

Node quadrature on an isolated single-level halo: let $h_f\le w$ inside the halo and $h_c$ outside. Let $T$ be the nonuniform trapezoid, $S_f$ the infinite fine sum ($=M$ exactly), $S_{f,\rm out}$ its part beyond the halo edges, and $T_{\rm out}$ the coarse sum there. Then

$$
T-M=T_{\rm out}-S_{f,\rm out},\qquad
\lvert T-M\rvert\le\max\left(T_{\rm out},S_{f,\rm out}\right)\le 2\eta M\left(1+\frac{h_c}{2D}\right).
$$

So for a narrow line whose halo interior is at one spacing $\le w$, the halo argument **does** bound the node-quadrature error, not only the discarded mass. The ledger note is more pessimistic than necessary for this case. The bound fails in two cases: when the halo interior contains a join between two levels, and for lines the rule never refines (below).

### Cheap filters

| filter | result |
| --- | --- |
| units | pass — $w$, $D$ in eV, $\eta$ dimensionless. The collected width is `pi / a_width` (`_kernels.py::_accumulate_edge_truncation`), i.e. $2\pi\hbar c/(\text{denominator}\,t_L)$ in eV, and it is unpacked in the right `(energy, width, weight)` order in `_measured_line_grid` |
| limit: all lines $w\ge$ backbone | pass — `narrow` is empty, no seeds, uniform backbone (numeric: one step value $2.9999$ eV, from `resolution_num` rounding down) |
| limit: one narrow line | pass — one window $\supseteq[E_i-D,E_i+D]$, rounded out to 100 eV bins, spacing $\le w$ |
| limit: $w<$ floor | pass as specified — held at the floor (numeric: $w=0.2\to0.25$) |
| sign/convention | pass — halo symmetric, $\eta$ is a per-side share. Levels satisfy $\text{floor}\cdot2^k\le\max(w,\text{floor})$ (`line_seeds.py:999`); `resolution_num` makes the actual step $\le$ the requested one |

### Code comparison

- `line_seeds.py:1000` computes `halo = width / (pi**2 * halo_limit)`, which matches $D=w/(\pi^2\eta)$ per side. Bin indices use `floor` below and `ceil` above, so coverage only grows.
- `line_seeds.py:993` defines `narrow` as `width < backbone`. Weights are ignored, so zero, NaN or negative-weight lines that the bandwidth pass drops still get halos. This is conservative and costs only points.
- Lines outside $[\text{start},\text{stop}]$: bin indices are clipped to $[0,n_{\rm bins}]$ and halos lying fully outside are dropped. A halo that straddles `stop` refines the inside bins. The population is collected up to the ceiling, not the measured `stop`, which is conservative.
- `line_grid.py:262` sets `floor_spacing_eV=min(target_step, backbone)`, where `target_step` comes from `sinc_feature_spacing`. That uses the vacuum denominator and a $t_L^2$ proxy weight, while the halos use the production in-medium widths. The mismatch only shifts level quantisation.
- `build_window_plan` takes the finest covering spacing per elementary interval. Equal adjacent levels merge. The sliver clean-up can only act at the clipped last bin next to `stop`. The plan spans exactly $[\text{start},\text{stop}]$, and `windowed_coordinates` enforces this, so the bandwidth edge is unchanged.
- Production quadrature is `node` (default, `_line_grid_policy.line_quadrature_from_payload`); the `high_energy` profile sets no `quadrature`. Integrated yield is `sum(density * node_bin_widths)`, which equals the trapezoid at interior nodes and at every join.

### Numeric checks (CPU, synthetic populations, scratch scripts not committed)

Isolated line, floor $0.25$ eV, backbone 3 eV, $\eta=10^{-4}$, 200 random resonance offsets. In every case the halo covered $E_i\pm D$ and the maximum interior step was $\le\max(w,\text{floor})$.

| $w$ (eV) | $D$ (eV) | worst $\lvert T/M-1\rvert$ | bound $2\eta(1+h_c/2D)$ |
| --- | --- | --- | --- |
| 2.9 | 2938 | $1.2\times10^{-5}$ | $2.0\times10^{-4}$ |
| 2.0 | 2026 | $1.0\times10^{-4}$ | $2.0\times10^{-4}$ |
| 1.0 | 1013 | $3.5\times10^{-5}$ | $2.0\times10^{-4}$ |
| 0.3 | 304 | $5.5\times10^{-5}$ | $2.0\times10^{-4}$ |
| 0.05 (< floor) | 51 | $4.0$ | not applicable — floor aliasing |
| 0.01 (< floor) | 10 | $23.9$ | not applicable — floor aliasing |

Joins at a line's resonance. A line's resonance can sit close to a spacing join created by *other* lines' windows. The join can be between two levels that are both $\le w$, or between the 3 eV backbone and a fine window for a line with $w\ge 3$ eV (no halo). In both cases the uniform-grid exactness is lost. Per-line worst errors over offsets:

| line $w$ (eV) | join distance 1 eV | 10 eV | 30 eV |
| --- | --- | --- | --- |
| 2.9 (own 2 eV halo, 0.25 eV window inside) | $7.5\times10^{-2}$ | $7.3\times10^{-4}$ | $1.3\times10^{-4}$ |
| 3.0 (no halo) | $2.2\times10^{-1}$ | $1.3\times10^{-2}$ | $5.1\times10^{-3}$ |
| 6.0 (no halo) | $2.7\times10^{-2}$ | $3.2\times10^{-3}$ | $7.1\times10^{-5}$ |

When $h\approx w$, $\sin^2$ sampled at spacing $w$ is constant, so a half-line sum misses the $\tfrac12$ average. The error then decays only like $w/(2\pi^2 d)$.

Aggregate integrated yield, 3000 lines, local axis against a uniform floor axis:

- Broad population ($E$ uniform 8–40 keV, $w$ log-uniform 0.3–30 eV): the local and uniform results differ by $1.1\times10^{-10}$. Join errors average out, because the translation-averaged trapezoid is exact on any grid.
- Concentrated population ($E_{\rm res}$ within $\sigma=3$ eV, $w$ lognormal about 3.5 eV) with a join on the peak: $-1.9\times10^{-5}$.
- Concentrated, $w\approx3.3$ eV, join 10 eV away: $-1.1\times10^{-3}$.
- Concentrated, $w\approx6$ eV, join 10 eV away: $-1.3\times10^{-3}$.

The last two exceed the $10^{-3}$ intrinsic-source total, while the uniform axis is exact to $2\times10^{-5}$. These configurations are constructed. Whether a production population (a thin target at MeV, with a narrow resonance spread) meets them is exactly the identical-trajectory comparison the ledger already requires.

`tests/energy-grid/test_line_grid_local_spacing.py`: 12 passed.

### Findings

1. **Stale statement, ledger Notes and this page.** Both say the policy "is not selected by `high_energy`" (ledger `ledger-core-coherent-physics.md:213`; this page, "Resolution rule" section). Commit `0f78fef1` sets `resolution = "resonance-local"` in `src/pyrite/data/catalog/profiles/high_energy.toml:13`, and `tests/materials/test_profiles.py` asserts it. The policy is now a production default for that profile. This raises the stakes of item 2.
2. **Quadrature is not bounded for lines the rule does not refine, or for multi-level halos.** The halo rule protects narrow lines on single-level halos, within $2\eta(1+h_c/2D)$ as derived above. It does not protect lines with $w\ge$ backbone, or lines of any width whose resonance lies near a spacing join produced by other lines. Constructed concentrated populations exceed $10^{-3}$. Possible fixes (the implementer's choice):
   - use `bin-mean` for yield observables;
   - keep joins at least a few $w$ away from any significant resonance;
   - bound the join term, for example by requiring every piece within about $10w$ of a resonance to have $h\le w/2$, so that $\sin^2$ averages;
   - measure the term with the identical-trajectory uniform comparison and add a focused regression on a concentrated population straddling a join.
3. **Budget bookkeeping.** `_line_grid_policy.py:180-182` assigns $\eta=10^{-4}$ as "the quadrature-backbone row … split in two", i.e. $2\eta=2\times10^{-4}$, which is consistent with the per-side bound. But `tbl-line-budget-allocation` (`line-spectrum-error-budget.md:45`) lists that row as controlled by `sinc_feature_spacing` / `DEFAULT_MAX_SPACING_EV`, and it does not mention `resonance-local` or the halo. The row is therefore spent twice. The floor's `aliased_weight_limit` is the strictest rtol ($10^{-3}$), which is a weight fraction, not an error bound: a single aliased line can err by up to about $h/w$ of its mass (table above). The budget page should record the halo share, and the floor term should cite its own share.
4. **Wording.** The docstring at `line_seeds.py:950` ("keeps at most `halo_limit` of its mass") and the policy comment ("per-line tail share") read as a two-sided share. The bound is per side; state "per side, $\le2\eta$ in total".
5. Minor: weightless lines still seed halos (`line_seeds.py:993`). This is conservative, so no change is needed; note it in the Assumptions.

## Independent verification (re-check)

Fresh-context verifier, 2026-09-28, branch `issue-192-high-energy-line-grid` at `672c87a5` (after `bdb33e1b`, core rule, and `672c87a5`, `high_energy` bin-mean). This section is appended; the author's text, the earlier verification and the ledger status are unchanged. Verdict for the claim as worded: **`discrepancy`**, on wording and cited evidence, not geometry. The halo, the new core, floor quantisation, binning, merging and the bandwidth edge all match an independent derivation. Under the `high_energy` profile's `bin-mean` quadrature, integrated yield does not depend on any of them. The discrepancy has three parts: the claim clause "keeping spacing joins out of the line's centre"; the core regression, which does not discriminate and whose cited numbers do not reproduce; and the scope of the node-quadrature caveat (see [Re-check findings](#re-check-findings)).

### Derivation (before reading the implementation)

**Core rule, as claimed.** Each line with $w_i<2h_{\rm bb}$ ($h_{\rm bb}=3$ eV) asks for spacing $\le w_i/2$ on $[E_i-10w_i,\,E_i+10w_i]$. Levels are quantised to $h_{\rm floor}2^k<h_{\rm bb}$, so the delivered core step is $\le\max(w_i/2,\,h_{\rm floor})$. Each 100 eV bin takes the finest requirement among the halos and cores that overlap it. This removes every piece coarser than $\max(w_i/2,h_{\rm floor})$ from the core. It does **not** remove joins: a join between two spacings that are both finer can sit anywhere in the core, including on the resonance. This happens wherever a narrower line's halo or core ends at a bin edge.

**Why $h\le w/2$ helps, and what remains.** Write $f(x)=\sin^2(ax)/(ax)^2=(1-\cos 2ax)/(2a^2x^2)$, with $w=\pi/a$ and mass $M=w$. Sum $f$ by the trapezoid rule with step $h$ over the half-line $x\ge d$. The Poisson replica at frequency $2\pi/h$ beats against the $\cos 2ax$ component at $2\pi/h-2a$.

- $h=w$: the beat frequency is zero. The samples of $\sin^2$ are the constant $\sin^2(ax_0)$ rather than its mean $\tfrac12$. The half-sum error is then $(\sin^2 ax_0-\tfrac12)/(a^2d)$, i.e. a relative error of up to $w/(2\pi^2 d)$. That is $5.1\times10^{-3}$ at $d=10w$.
- $h\le w/2$: the beat frequency is at least $2a$, so the oscillatory term falls as $d^{-2}$. What remains is the Euler–Maclaurin endpoint term at the join, $(h_1^2-h_2^2)f'(d)/12$. It is of order $0.1\,(h/w)^2$ of the line mass near $d\lesssim w$.

The core therefore turns the $1/d$ tail into $1/d^2$. It does not remove the $(h/w)^2$ term of a join inside the core.

**Bin-mean yield.** The spectrum value at node $j$ is $\bar S_j=\Delta_j^{-1}\int_{e_j}^{e_{j+1}}S\,dE$, with edges from `bin_axis` (midpoints; outer edges a reflected half step). The yield $\sum_j\bar S_j\Delta_j$ uses the same edge array, and consecutive bins share an edge, so the sum telescopes:

$$
\sum_j \bar S_j\,\Delta_j=\int_{e_0}^{e_N}S\,dE
\quad\Longrightarrow\quad
\frac{\hat Y}{Y}-1=-\frac{1}{M}\left[\int_{-\infty}^{e_0}S+\int_{e_N}^{\infty}S\right],
$$

for any spacings, joins or levels. The only dependence on the grid is through $e_0=E_0-\tfrac12\Delta_0$ and $e_N=E_N+\tfrac12\Delta_N$. If the yield is instead taken as `np.trapezoid(spec, E)` (as in `energy_grid/convergence.py:329`), the interior weights equal $\Delta_j$ but the two end nodes get $\tfrac12\Delta$. That adds an end-bin term, which again is a band-edge effect only. So under `bin-mean` the halo and the core change only how the mass is distributed among bins, never the total.

**Bin-mean shape.** At the bin containing the peak, a single $\operatorname{sinc}^2$ averaged over width $h$ gives $2F(h/2w)\,w/h$, with $F$ the `sinc-bin-integration` antiderivative. That is $0.774$ at $h=w$ and $0.935$ at $h=w/2$ for a centred bin, and lower for other offsets.

### Filters

| filter | result |
| --- | --- |
| units | pass. $w$, $10w$ and $w/2$ are in eV; `core_fraction` and `core_widths` are dimensionless |
| limit: $w\ge2h_{\rm bb}$ | pass. No core and no halo (`line_seeds.py:1031`; test `test_lines_twice_the_backbone_seed_nothing`). The 3 eV backbone is already $\le w/2$ |
| limit: $h_{\rm bb}\le w<2h_{\rm bb}$ | pass. Core only, at level $\le w/2$ (test: $w=3\to1.0$ eV on one 100 eV bin) |
| limit: $w<h_{\rm bb}$ | pass. Halo at level $\le w$ plus core at level $\le w/2$ (test: levels 2, 1, 2 eV) |
| limit: $w/2<h_{\rm floor}$ | the core is held at the floor. The claim text omits "subject to the floor" for the core |
| convention: bins | pass. The core extent $\pm10w$ is rounded out to 100 eV; `floor`/`ceil`, clipped to $[0,n_{\rm bins}]$ |

Randomised coverage check (40 populations of 300 lines, $w$ log-uniform 0.05–8 eV, floors 0.1/0.25/0.4 eV, `stop` not bin-aligned at 31234.5 eV, 11,276 cored lines). The largest step inside $E_i\pm10w_i$, divided by $\max(w_i/2,h_{\rm floor})$, is $1.0000$. There are 0 halo violations. The axis spans exactly $[\text{start},\text{stop}]$.

### Node quadrature: what the core bounds (scratch scripts, CPU, not committed)

Per-line worst $\lvert T/M-1\rvert$ over 128 offsets, for a two-spacing join at distance $d$ from the resonance (exact reference from the antiderivative over the same finite span):

| line, join $h_1\vert h_2$ (eV) | $d=0$ | $w$ | $3w$ | $10w$ | $30w$ |
| --- | --- | --- | --- | --- | --- |
| cored $w=2.0$, $0.25\vert1.0$ inside core | $3.5\times10^{-2}$ | $4.2\times10^{-3}$ | $6.8\times10^{-4}$ | $7.1\times10^{-5}$ | $8.2\times10^{-6}$ |
| cored $w=3.3$, $0.25\vert1.0$ inside core | $1.1\times10^{-2}$ | $1.3\times10^{-3}$ | $2.1\times10^{-4}$ | $2.3\times10^{-5}$ | $2.6\times10^{-6}$ |
| cored $w=3.0$, $1.0\vert3.0$ core edge | — | — | — | $4.9\times10^{-3}$ | $1.6\times10^{-3}$ |
| cored $w=3.3$, $1.0\vert3.0$ core edge | — | — | — | $7.4\times10^{-4}$ | $9.1\times10^{-5}$ |
| wide $w=6$, $0.25\vert3.0$ (no core) | $3.7\times10^{-2}$ | $4.4\times10^{-3}$ | $7.1\times10^{-4}$ | $7.4\times10^{-5}$ | $8.7\times10^{-6}$ |
| wide $w=8$, $0.25\vert3.0$ (no core) | $2.0\times10^{-2}$ | $2.3\times10^{-3}$ | $3.6\times10^{-4}$ | $3.8\times10^{-5}$ | $4.4\times10^{-6}$ |

The core-edge value at $w=3.0$, $d=10w$ ($4.9\times10^{-3}$) matches $w/(2\pi^2d)=5.1\times10^{-3}$: when $w\approx h_{\rm bb}$ the backbone samples $\sin^2$ at spacing $\approx w$. Core-edge joins lie at $d\ge10w$ by construction, so those cells show "—".

Concentrated populations. The axes are built by `local_spacing_seeds` + `build_window_plan` (start 100 eV, stop 60 keV, floor 0.25 eV). The yield is $\sum\text{density}\times\Delta$; the reference is the whole-line mass. "No core" means `core_fraction=1`, `core_widths=1e-9`, which reproduces the pre-`bdb33e1b` rule exactly for these populations:

| population (400 lines, join at 20 keV from ten 0.4 eV lines) | node, core | node, no core | bin-mean | $-$truncation |
| --- | --- | --- | --- | --- |
| test `_join_population(3.3, +10)` | $-6.6\times10^{-5}$ | $+3.5\times10^{-5}$ | $-1.25\times10^{-5}$ | $-1.25\times10^{-5}$ |
| test `_join_population(8.0, -3)` | $+1.01\times10^{-2}$ | $+1.01\times10^{-2}$ | $-3.0\times10^{-5}$ | $-3.0\times10^{-5}$ |
| $w=3.1$, $\sigma=3$, offset $-3$ | $+1.1\times10^{-3}$ | $+6.3\times10^{-3}$ | $-1.2\times10^{-5}$ | $-1.2\times10^{-5}$ |
| $w=3.0$, $\sigma=3$, offset $-3$ | $+1.0\times10^{-3}$ | $+3.9\times10^{-3}$ | $-1.1\times10^{-5}$ | $-1.1\times10^{-5}$ |
| $w=2.0$, $\sigma=0.5$, offset $\pm1$ (join inside core) | $\mp4.9\times10^{-3}$ | $\mp2.5\times10^{-2}$ | $-7.6\times10^{-6}$ | $-7.6\times10^{-6}$ |
| $w=3.0$, tight ($\sigma=0.05$), 30.5 eV below own core edge | $-3.7\times10^{-3}$ | $-1.3\times10^{-5}$ | $-1.1\times10^{-5}$ | $-1.1\times10^{-5}$ |
| broad, 3000 lines, $w$ 0.3–30 eV | $-6.3\times10^{-5}$ | $-6.3\times10^{-5}$ | $-6.3\times10^{-5}$ | $-6.3\times10^{-5}$ |

The core lowers the node error by a factor of 3 to 6 for populations of cored lines near a join. It can still leave $10^{-3}$ to $5\times10^{-3}$, above the $10^{-3}$ intrinsic-source total, in three situations: $w\approx h_{\rm bb}$, a join between finer levels inside a core, and a tight population at its own core edge. That last row is a join the core itself creates; without the core it does not exist. For lines with $w\ge2h_{\rm bb}$ the error of about $10^{-2}$ is confirmed. The bin-mean column equals the band-edge truncation to every printed digit in every row. That confirms the yield identity derived above. The page's $4.6\times10^{-5}$ is of the same kind; I could not identify its exact population.

### Bin-mean shape (single line; Gaussian resonance ensembles)

| resonance spread $\sigma/w$ | $h/w$ | peak, bin-mean | FWHM, bin-mean | peak, node | FWHM, node |
| --- | --- | --- | --- | --- | --- |
| 0 (one line) | 1 | $-0.55$ | $+1.3$ | $-0.60$ | $+1.4$ |
| 0 (one line) | 0.5 | $-0.23$ | $+0.24$ | $-0.19$ | $+0.20$ |
| 0.5 | 0.5 | $-0.09$ | $+0.09$ | $-0.07$ | $+0.06$ |
| 2 | 0.5 | $-0.021$ | $+0.015$ | $-0.015$ | $+0.011$ |
| 5 | 0.5 | $-0.012$ | $+0.013$ | $-0.010$ | $+0.011$ |

Each entry is the worst relative error over 8 grid offsets, against a dense continuous reference. The FWHM is taken by linear interpolation on the samples. The bin-mean smoothing at $h\le w/2$ is comparable to node sampling at the same $h$; it is not small unless the resonance spread is several times $w$. Even at $\sigma=5w$ it exceeds the harness shape share of $10^{-2}$. No document derives or measures a shape bound for `resonance-local`. The statements "the core then limits only line-shape smoothing" (above) and "bounds only shape" (budget page, line 106) are therefore qualitative, not a bound.

### Profile compatibility with the bin-mean refusals

`lines/_setup.py:371-387` refuses `bin-mean` with coherent emission, with flight-grouped substeps, and with `sinc_cutoff`. I built every `high_energy` case (hbn 128, mose2 96, mos2 96) with `build_cases`. Every one has `coherent_emission=False`, `max_dE_frac=0`, `sinc_cutoff=None` and `line_quadrature="bin-mean"`, so the profile is compatible as shipped. Adding coherent emission (`--coherent` / `--emission both`) or `max_dE_frac>0` to this profile would raise only at spectrum time, after transport. No resolve-time guard exists (`campaign/sweep.py:970-975`).

### Budget bookkeeping

- The resolver refuses `windows` under `resonance-population` (`_line_grid_policy.py:663-668`). Measured-bandwidth cases therefore never spend the feature-window planner's share. Local seeds use only the planner's merge function, not its seeds or its measured campaign. Charging $\epsilon=10^{-4}$ and $2\eta=2\times10^{-4}$ to that row is consistent, and no row is spent twice.
- Under `bin-mean` the $2\eta$ charge is conservative for yield: the halo moves no mass. The floor-aliasing term also vanishes for yield. The earlier finding that `aliased_weight_limit` is the strictest rtol ($10^{-3}$, `_line_grid_policy.py:596,678`) rather than the backbone share of $2\times10^{-4}$ still stands for `node`.
- Under `node`, the join term above has no budget row. `resolve_line_grid_policy` accepts `resonance-local` with `node` without comment (`_line_grid_policy.py:656-662`).
- Minor: the budget paragraph (line 106) names `resonance-local`, but the bandwidth share applies to every `resonance-population` case, `sinc-nyquist` included. The table's "Controlled by" cell for the window row (line 54) does not mention the two new shares.

`tests/energy-grid/test_line_grid_local_spacing.py` and `tests/materials/test_profiles.py`: 64 passed.

### Re-check findings

1. **Claim wording (first divergent convention).** Ledger claim (`ledger-core-coherent-physics.md:207`, `domain-inventories.md:31`): "keeping spacing joins out of the line's centre". The code keeps only spacing coarser than $\max(w/2,h_{\rm floor})$ out of $E_i\pm10w_i$. Joins between finer levels remain anywhere in the core; in the committed test grid there is a 0.25/1.0 eV join 10 eV ($3w$) from the $w=3.3$ resonances. Suggested wording: "…at spacing no larger than $\max(w/2,\text{floor})$, so no piece coarser than that lies within $\pm10w$ of the resonance."
2. **Non-discriminating regression; unreproduced numbers.** `test_core_bounds_node_error_for_lines_near_a_join` (`tests/energy-grid/test_line_grid_local_spacing.py:281-284`) states "$-1.1\times10^{-3}$ before the core, $\sim1.4\times10^{-4}$ with it". Its own population gives $+3.5\times10^{-5}$ without the core and $-6.6\times10^{-5}$ with it, so the test passes with the core disabled. The page ("Resolution rule") and commit `bdb33e1b` cite the same numbers. A discriminating population, e.g. $w=3.1$, $\sigma=3$, offset $-3$ ($6.3\times10^{-3}\to1.1\times10^{-3}$) or $w=2.0$, $\sigma=0.5$, offset $\pm1$ ($2.5\times10^{-2}\to4.9\times10^{-3}$), should replace it.
3. **Scope of the node caveat.** "It cannot bound it for lines wider than twice the backbone" understates the residual. Cored lines also exceed $10^{-3}$ under `node`: $w\approx h_{\rm bb}$ at its own core edge ($\le w/(2\pi^2\cdot10)=5\times10^{-3}$ per line), and fine–fine joins inside a core (up to $\approx0.1(h/w)^2$ per line, $5\times10^{-3}$ for a constructed population). Suggested: state that under `node` the core reduces but bounds no join term, and either refuse `resonance-local` with `node` or add a budget row for the join term.
4. **Shape.** The bin-mean yield identity is proven, and all numeric rows confirm it. The "shape only" statements carry no bound. At $h\le w/2$, smoothing of peak and FWHM is $1$ to $23\%$ depending on resonance spread. The ledger Checks already require a line-shape comparison; until it exists, no shape claim should read as a bound.
5. Minor: `local_spacing_seeds` returns no seeds when `cored_lines == 0` (`line_seeds.py:1036`). With `core_fraction>1` that would drop narrow-line halos. The value is unreachable at the default of 0.5, but it is not validated. The core parameters are also absent from the policy payload and schema (`RESONANCE_LINE_GRID_POLICY_SCHEMA = 4`), so case identity does not record the core rule, though the run record's `local_spacing` summary does.

## Independent verification (second re-check)

Fresh-context verifier, 2026-09-28, branch `issue-192-high-energy-line-grid` at `5d01f415` (after `b01bc2f2`, which answers the re-check). This section is appended; the author's text, the earlier verifier sections and the ledger status are unchanged. I read the claim, the Notes and the "Resolution rule" section first, derived before reading `b01bc2f2` or the implementation, and only then read the earlier verifier sections.

- **Claim**: `line-grid-resonance-local-spacing` — `montecarlo/spectrum/line_seeds.py::local_spacing_seeds`; `_line_grid_policy.py::resolve_line_grid_policy` — the `finite-time-lineshape` first-zero width and the one-sided $\operatorname{sinc}^2$ tail bound
- **Filters**: units pass; limits pass; signs/conventions pass
- **Re-derivation**: `matches`
- **Verdict**: `rederived`, for the claim as now worded. The findings below are wording and guard-scope items; none of them changes a factor in the claim.
- **Write-up**: `docs/validation/beam-transport/line-grid-resonance-local-spacing.md`
- **Suggested ledger change**: status `unverified` → `rederived`. In Notes, replace `0f78fef1` with a reachable commit and state that the node refusal applies at policy resolution (findings 3 and 6). A human applies the change.

### Derivation (before reading the implementation)

Let $x=E-E_i$ and $f(x)=\sin^2(ax)/(ax)^2$, with $w=\pi/a$ and mass $M=\int f\,dx=w$. On one side, $\int_D^\infty f\,dx\le\int_D^\infty(ax)^{-2}dx=1/(a^2D)$. The mass fraction beyond the edge is then $1/(\pi aD)=w/(\pi^2D)$. Setting that to $\eta$ gives $D=w/(\pi^2\eta)$ per side, and at most $2\eta$ in total. This matches the claim and the docstring's "`2 halo_limit` in total".

Bin-mean yield: with edges $e_0<\dots<e_N$ and bin means $\bar S_j=\Delta_j^{-1}\int_{e_j}^{e_{j+1}}S\,dE$, consecutive bins share an edge, so

$$
\sum_j\bar S_j\,\Delta_j=\int_{e_0}^{e_N}S\,dE .
$$

The yield depends on the axis only through $e_0$ and $e_N$, whatever the spacings or joins inside. For an independent check I used the closed form $\int f\,dx=a^{-1}\left[\operatorname{Si}(2ax)-\sin^2(ax)/(ax)\right]$. Differentiating the bracket in $u=ax$ gives $2\sin u\cos u\cdot(-1/u)+\sin^2u/u^2+\sin 2u/u=\sin^2u/u^2$.

The core, as claimed, asks for spacing $\le w_i/2$ on $E_i\pm10w_i$ for every $w_i<2h_{\rm bb}$. After quantisation to $h_{\rm floor}2^k$, the delivered step is $\le\max(w_i/2,h_{\rm floor})$. A join between two finer levels can still sit anywhere inside the core, which is what the claim now says.

### Filters

| filter | result |
| --- | --- |
| units | pass. $w$, $D$, $10w$, $w/2$ in eV; $\eta$, `core_fraction`, `core_widths` dimensionless |
| limit: $w\ge2h_{\rm bb}$ | pass. No halo and no core (`line_seeds.py:1030,1035`) |
| limit: no cored line but narrow lines present (`core_fraction>1`) | pass. The halo survives (`line_seeds.py:1040`); see the mutation check below |
| limit: $w/2<h_{\rm floor}$ | pass. The core is held at the floor, as the claim now states |
| convention: `resonance-local` + `node` | pass. Refused (`_line_grid_policy.py:666-670`). An unspecified quadrature defaults to `node` (`_line_grid_policy.py:616`) and is refused as well |

### Reproduced numbers (CPU, scratch scripts not committed)

For the axes I used `local_spacing_seeds` and `build_window_plan`. The yields I computed independently: `np.trapezoid` for node quadrature, and the Si closed form over midpoint cells for bin-mean. The populations come from the test's `_join_population` (400 lines, $\sigma=3$ eV unless noted, plus ten 0.4 eV lines at 19,550 eV). "No core" means `core_fraction=1e9`.

| population | node, core | node, no core | bin-mean | $-$truncation |
| --- | --- | --- | --- | --- |
| $w=3.1$, offset $-3$ (test) | $+1.09\times10^{-3}$ | $+6.31\times10^{-3}$ | $-1.18\times10^{-5}$ | $-1.18\times10^{-5}$ |
| $w=3.3$, offset $+10$ | $-6.62\times10^{-5}$ | $+3.45\times10^{-5}$ | $-1.25\times10^{-5}$ | $-1.25\times10^{-5}$ |
| $w=8$, offset $-3$ (test) | $+1.01\times10^{-2}$ | $+1.01\times10^{-2}$ | $-3.04\times10^{-5}$ | $-3.04\times10^{-5}$ |
| $w=2$, $\sigma=0.5$, offset $+1$ | $-4.86\times10^{-3}$ | $-2.51\times10^{-2}$ | $-7.58\times10^{-6}$ | $-7.58\times10^{-6}$ |
| $w=2$, $\sigma=0.5$, offset $-1$ | $+4.73\times10^{-3}$ | $+2.45\times10^{-2}$ | $-7.58\times10^{-6}$ | $-7.58\times10^{-6}$ |

These values reproduce the page's $6.3\times10^{-3}\to1.1\times10^{-3}$, the $\sim10^{-2}$ figure for lines wider than twice the backbone, and "bin-mean error equals band-edge truncation" in every row. I did not re-run the $5\times10^{-3}$ core-edge figure for $w\approx h_{\rm bb}$. It agrees with the closed form $w/(2\pi^2\cdot10w)=5.1\times10^{-3}$.

Coverage: I ran 20 random populations of 300 lines, with $w$ log-uniform over 0.05–8 eV, floors 0.1, 0.25 and 0.4 eV, and `stop` $=31234.5$ eV (not bin-aligned). That gave 5,656 cored lines. The largest step inside $E_i\pm10w_i$ divided by $\max(w_i/2,h_{\rm floor})$ is $1+2\times10^{-11}$, which is float rounding. There are 0 halo violations. The axis spans exactly $[\text{start},\text{stop}]$.

Discrimination. In each mutant I exec'd a patched copy of the source in a scratch namespace; the repository was not modified.

- Restoring the old early return (`cored_lines == 0`) gives 0 seeds, so `test_narrow_line_halos_survive_without_cores` fails.
- Deleting the core `cover(...)` call makes cored equal to no core ($6.31\times10^{-3}$), so `test_core_reduces_but_does_not_bound_node_error_near_a_join` fails its `/4` assertion.
- Removing the refusal makes `test_local_resolution_needs_bin_mean_quadrature` fail.

`tests/energy-grid/test_line_grid_local_spacing.py` and `tests/materials/test_profiles.py`: 67 passed.

Profile: I built every `high_energy` case with `build_cases`: hbn 128, mose2 96, mos2 96. Each has `line_quadrature="bin-mean"`, no coherent emission, `max_dE_frac=0`, `sinc_cutoff=None`, a payload quadrature of `bin-mean` and resolution `resonance-local`, so there are 0 incompatible cases. `build_cases(..., coherent_emission=True)` is refused when the case is built, by `Case` validation (`montecarlo/case.py:327-331`). The re-check's remark that it "would raise only at spectrum time" no longer holds for coherent emission.

### Status of the re-check findings

1. **Claim wording: resolved.** The claim and inventory now read "no piece coarser than that lies within $\pm10w$ … joins between finer levels may remain" and "subject to the global sinc spacing floor". Both match the code and the coverage check.
2. **Non-discriminating regression: resolved.** The new test uses a discriminating population. Its docstring numbers reproduce, and it fails when the core is removed.
3. **Node caveat scope: resolved by the refusal, with a residual scope gap** (finding 3 below).
4. **Shape: turned into a stated limitation.** "No shape bound is derived" now appears in the ledger Notes, on this page and in the budget paragraph. Two stale sentences remain (finding 4).
5. **Early return and core identity: resolved or stated.** The early return is fixed and has a discriminating test. The core parameters remain outside the payload, and the Notes now say so.

### Second re-check findings

1. The halo, core, floor quantisation, binning, merge and bandwidth edge match the claim (`line_seeds.py:1027-1041`). No discrepancy.
2. The bin-mean yield identity holds to every printed digit in the five populations above, and it discriminates node from bin-mean (`tests/energy-grid/test_line_grid_local_spacing.py:320-328`).
3. **The refusal is resolve-time only.** The runner takes the quadrature from the case key (`montecarlo/runner/__init__.py:642`, default `node`). `_measured_line_grid` (`montecarlo/runner/line_grid.py:274`) and `Case` validation (`montecarlo/case.py:323-336`) do not check that a `resonance-local` payload is paired with `line_quadrature="bin-mean"`. The sweep path derives the key from the payload (`campaign/sweep.py:970-974`), so it stays consistent. A hand-built case, or a payload resolved before `b01bc2f2`, runs `node` without error. The claim "the policy requires `bin-mean`" is true of `resolve_line_grid_policy`. The fix is either a run-time guard or narrower wording.
4. **Stale trapezoid wording.** Three passages still refer to trapezoid error, which no longer exists for integrated yield because `bin-mean` is mandatory:
   - this page, "Resolution rule" paragraph 3: "does not bound nonuniform trapezoid error inside the window. That error needs an identical-trajectory comparison";
   - the ledger Notes' first sentence;
   - the docstring at `line_seeds.py:980-982`, "the window ladder measures it". The window ladder does not run for measured-bandwidth cases (`_line_grid_policy.py:673-678`).

   Under `bin-mean` the residual is line shape only.
5. The budget row's "Controlled by" cell (`line-spectrum-error-budget.md:54`) lists `DEFAULT_LOCAL_HALO_LIMIT` for all measured-bandwidth cases. The paragraph (`:106`) correctly restricts it to `resonance-local`. Minor.
6. The ledger Notes (`ledger-core-coherent-physics.md:213`) cite `0f78fef1`, which is not an ancestor of HEAD. The profile gained `resonance-local` in `508b7a74` and `bin-mean` in `8c6433f0`. Minor.
7. `core_fraction` and `core_widths` are not validated (`line_seeds.py:1035`). A non-positive `core_fraction` would hold every line at the floor, which costs only points. There is no reachable caller other than the defaults. Minor.
