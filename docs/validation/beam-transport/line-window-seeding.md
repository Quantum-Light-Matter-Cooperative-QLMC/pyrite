# `line-window-seeding`

Ledger row: [`line-window-seeding`](../ledger-core-coherent-physics.md#line-window-seeding).
Code under review: `src/pyrite/montecarlo/spectrum/line_seeds.py`
(`kinematic_line_seeds`, `absorption_edge_seeds`, `characteristic_line_seeds`)
and `src/pyrite/_line_windows.py::build_window_plan`. Acceptance evidence
the row cites:
[line-spectrum error budget](line-spectrum-error-budget.md).

Initial fresh-context verification, 2026-09-17. The derivation below was
written from the ledger row, the docstrings, and the cited upstream rows before
the function bodies were read. No implementation file was modified.

The initial `discrepancy` verdict records the pre-fix implementation. Section 6
records the independent re-verification after commits `16b9d38c` and `971648f1`;
the current verdict is **`anchored`**, with human sign-off pending.

## Initial verdict summary (pre-fix)

- **Claim**: `line-window-seeding`. Code: `montecarlo/spectrum/line_seeds.py::kinematic_line_seeds`,
  `::absorption_edge_seeds`, `::characteristic_line_seeds`, and
  `_line_windows.py::build_window_plan`. Source: the vacuum resonance of
  `line-energy-dispersion` (Zhai SI Eq. 10), the finite-time width of
  `finite-time-lineshape`, $\mu$ from `absorption-length`, and xraydb lines
  from `characteristic-radiation`.
- **Filters**: units pass; limiting cases pass (both); signs and conventions
  pass.
- **Re-derivation**: `differs` on two stated properties. The kernel
  conventions match exactly.
  1. The per-segment offset between the vacuum and in-medium resonance is
     $\Delta E\simeq -E\,\delta\,\beta\cos\Theta/(1-\beta\cos\Theta)$. The
     docstring calls this "$O(\delta)\sim10^{-5}$ relative, inside the
     `tail_widths` margin". Its ratio to the margin $\tau\,w_\varepsilon$ has
     no $O(\delta)$ bound. It is measured at up to $5.4\,w_\varepsilon$
     against a margin of $2\,w_\varepsilon$, above 300 eV, on hopg 300 keV
     tilt 85.
  2. The edge search picks the steepest bracket anywhere in $\pm5\%$ of the
     xraydb energy, and a bracket already seeded is then skipped. Together
     these leave out every secondary shell that lies within 5% of a stronger
     jump. It is neither seeded nor reported as `skipped`. WSe$_2$ loses Se L2
     ($\mu$ +35%), W M4 (+36%), and W L1 (+12.5%).
- **Verdict**: `discrepancy`. Both findings are grid-coverage policy and not
  radiation physics. Neither one shows up in the coverage measured here. It
  also does not show up in the twelve-case window campaign as reported.
- **Suggested ledger change**: set Status to `discrepancy`, and record the two
  diverging terms and this write-up in Checks and Notes.

## 1. Independent derivation

Units are $c=1$: lengths and times in Å, and $\omega$ in Å$^{-1}$, so
$E=\hbar c\,\omega$. A straight segment has velocity
$\mathbf v=\beta\hat{\mathbf v}$, length $L$, and flight time $t_L=L/\beta$.
The observation direction is $\hat{\mathbf n}$, and
$\cos\Theta=\hat{\mathbf v}\cdot\hat{\mathbf n}$.

### 1.1 Resonance, vacuum and in medium

The emitted field of a segment is proportional to
$\int_0^{t_L}e^{i\Phi(t)}\,dt$, with the phase
$\Phi(t)=\omega t-(\mathbf k+\mathbf g)\cdot\mathbf v\,t$ and
$\mathbf k=n(\omega)\,\omega\,\hat{\mathbf n}$. Stationarity,
$d\Phi/dt=0$, gives

$$
\omega\,\bigl(1-n\,\beta\cos\Theta\bigr)=\mathbf v\cdot\mathbf g .
$$

In vacuum ($n=1$) this is the ledgered resonance

$$
\omega_0=\frac{\mathbf v\cdot\mathbf g}{1-\mathbf v\cdot\hat{\mathbf n}},
\qquad E_0=\hbar c\,\omega_0 .
$$

A segment radiates only when $\omega_0>0$. The vacuum denominator satisfies
$1-\beta\cos\Theta\ge1-\beta>0$, so the sign test falls on
$\mathbf v\cdot\mathbf g$ alone.

With $n=1-\delta$,

$$
\omega_1=\frac{\mathbf v\cdot\mathbf g}{1-\beta\cos\Theta+\delta\,\beta\cos\Theta},
\qquad
\frac{\omega_1-\omega_0}{\omega_0}
=-\frac{\delta\,\beta\cos\Theta}{1-\beta\cos\Theta}+O(\delta^2).
$$

The relative shift is $O(\delta)$ times a geometric factor. That factor is
order unity for side emission, but it is enhanced for forward-going segments.
When $\beta\to1$ and $\Theta\to0$ it tends to
$2\gamma^2/(1+\gamma^2\Theta^2)$.

### 1.2 Feature width and the shift in feature widths

Near the vacuum root, the detuning rate is
$\Delta=(1-\beta\cos\Theta)(\omega-\omega_0)$. The segment integral is

$$
\Bigl|\int_0^{t_L}e^{i\Delta t}dt\Bigr|^2
=t_L^2\,\operatorname{sinc}^2\!\Bigl(\frac{\Delta t_L}{2}\Bigr),
\qquad \operatorname{sinc}x=\frac{\sin x}{x}.
$$

Its first zero is at $\Delta t_L/2=\pi$. The zero lies at
$|\omega-\omega_0|=\pi/a_w$ with $a_w=(1-\beta\cos\Theta)\,t_L/2$. In energy,
the segment's own feature width is

$$
w_{\rm seg}=\frac{2\pi\hbar c\,\beta}{(1-\beta\cos\Theta)\,L}.
$$

Dividing the first-order shift by this width removes the geometric
denominator:

$$
\frac{|E_1-E_0|}{w_{\rm seg}}
=\delta\,|\cos\Theta|\,\frac{E\,L}{2\pi\hbar c}
=\delta\,|\cos\Theta|\,\frac{L}{\lambda}.
$$

This ratio is the number of refractive phase cycles accumulated over the
segment. It is not $O(\delta)$: it is $\delta$ multiplied by $L/\lambda$,
which is about $10^2$–$10^5$ for these segments. The measured maximum
of the ratio is 0.13 at hopg 30 keV and 1.28 at hopg 300 keV.

The planner's margin is not $w_{\rm seg}$. It is
$\tau\,w_\varepsilon$, where $\tau$ is `tail_widths` $=2$ and
$w_\varepsilon$ is the $\varepsilon$-weighted lower-quantile width, which is
near the narrowest. The per-segment condition the docstring asserts is
therefore

$$
\boxed{\;
\frac{E\,\delta\,\beta\,|\cos\Theta|}{1-\beta\cos\Theta}\;\le\;\tau\,w_\varepsilon
\;}
$$

No property of the construction guarantees this condition. With the
free-electron estimate $\delta=r_e\lambda^2n_e/(2\pi)$, graphite has
$\delta\approx1.9\times10^{-5}$ at 5 keV. That grows as $E^{-2}$ to a few
$\times10^{-3}$ near 300 eV, where the C K edge adds its own anomalous
structure. Forward segments ($1-\beta\cos\Theta\ll1$) raise the left-hand
side further. The kernels' 300 keV feature widths are 0.2–0.3 eV. Violation
is therefore expected for forward-going segments with $E\lesssim1$ keV at high
beam energy.

The margin is not the only coverage mechanism. A window spans the
$[\varepsilon/2,1-\varepsilon/2]$ quantile band of the whole population, and a
shifted segment in the band interior stays covered. The quantity that
matters for coverage is the in-medium weight that falls outside the window.
Section 3 measures it.

### 1.3 Below ~300 eV

The tabulated Re $n$ can exceed 1 below roughly 300 eV (carbon: 6.24–285 eV,
per `_kernels.py::_in_medium_kinematics`). In that range $\delta<0$ and the
shift changes sign, so the root moves upward. The magnitude of $\delta$ is no
longer small, and $1-n\beta\cos\Theta$ can approach zero, which is a
Cherenkov-like pole. A vacuum root there can be many feature widths from the
kernel's root. It can also belong to a segment that the kernel rejects,
because the fixed-point iteration fails its convergence check and returns a NaN
denominator. Vacuum seeding therefore gives no margin guarantee below about
300 eV. Only the population band, or the backbone, covers that region. The
bandwidths of the hopg and wse2 cases start at 10–50 eV, so this region is
inside the band.

### 1.4 Dropped-reflection bound

Let $W_r$ be the radiating proxy weight of reflection $r$, and
$W=\sum_rW_r$. Reflections are sorted by ascending $W_r$, and they are dropped
while the cumulative dropped weight $D$ satisfies $D\le\varepsilon W$. Each kept
reflection is covered from its lower weighted quantile $q_-$ to its upper
weighted quantile $q_+$. The quantile is defined as the smallest ordered value
whose cumulative weight reaches the target. That definition leaves strictly
less than $\varepsilon W_r/2$ below $q_-$ and at most $\varepsilon W_r/2$
above $q_+$. The uncovered weight is therefore

$$
U\le D+\varepsilon\,(W-D)\le(2\varepsilon-\varepsilon d)\,W<2\varepsilon W,
\qquad d=D/W\le\varepsilon .
$$

The tail margin only shrinks $U$. The bound is relative to the total radiating
proxy weight. That total includes weight whose resonance lies outside
$[\texttt{start},\texttt{stop}]$, so relative to in-band weight alone the
bound can be looser by the in-band fraction.

### 1.5 Edges

The attenuation that the kernels apply is
$\mu=2r_e\lambda\sum_in_if_{2,i}$, with $f_2$ taken from the Chantler/FFAST
table (`absorption-length`). The kernel evaluates $\mu$ on a tabulation grid
that contains every native Chantler node, and it interpolates
$\log\mu_i$ against $\log E$. A step in $\mu$ therefore lies inside one or more
adjacent native Chantler brackets. That position is fixed by the Chantler node
energies, not by the xraydb edge energy; the two tables disagree. Every shell
whose own jump the table resolves needs its own bracket found. Taking the
steepest bracket in a window $\pm5\%$ wide cannot find a weaker jump when a
stronger one lies inside the same window.

### 1.6 Characteristic lines

`mc_characteristic_spectrum` deposits a line when two conditions hold: the line
energy exceeds the relaxation cutoff, and the line has a positive yield per
vacancy after the cutoff mask. The cutoff is the largest recommended cutoff
among the composition's elements. The seed set must be the same set, taken per
composition.

## 2. Diff against the implementation

### Check 1: seed rows and velocity convention match the kernels

| Item | Seeds (`line_seeds.py`) | Kernels | Match |
| --- | --- | --- | --- |
| energy field | `E_repr_keV` else `E_keV` (l. 159) | same (`lines/_setup.py:255`) | yes |
| velocity | `beta_from_keV(E)[:,None]*v_hat` (l. 166–167) | `beta_all[:, None] * seg_v` (`_setup.py:262–263`) | yes |
| electron subset | `elec_id < electron_limit` (l. 163–165); runner passes `Ne` (`runner/line_grid.py:95`) | `seg_elec_id < Ne` (`_setup.py:261`) | yes |
| orientation | `_orientation_R(lattice, beam_uvw, azimuth_rad, recip_miscut_rad, surface_hkl=)` (l. 187–189); same case/radiator fallbacks (l. 439–444) | same call (`_setup.py:203–209`); runner keys (`runner/__init__.py:651–655, 672–692`) | yes |
| row order | `mosaic_rotation @ (rotation @ g_hkl)`, weight `mosaic_weight` (l. 194–204) | `R_m @ (R_orient @ g_vec)`, `wm` (`_per_hkl.py:661–673`; `_batched.py:231–243`) | yes |
| mosaic quadrature | `_mosaic_quadrature(case["mosaic_mc_fwhm_rad"], case["mosaic_mc_nodes"])` | same helper and keys (`_setup.py:466`) | yes |
| resonance | $\hbar c\,(\mathbf v\cdot\mathbf g)/(1-\mathbf v\cdot\hat{\mathbf n})$, vacuum (l. 168, 201) | same, then in-medium fixed point (`_per_hkl.py:387–390`; `_kernels.py:460–520`) | vacuum by design; see check 2 |
| 10 eV floor | `resonance > _MIN_RESONANCE_EV` $=10$ (l. 72, 202) | `E_res > 10.0` on the in-medium root (`_per_hkl.py:399`; `_batched.py:431`) | yes, up to the in-medium shift |
| intensity proxy | $t_L^2=(L/\beta)^2$ (l. 169) | $t_L$ `= seg_L / beta_all` (`_setup.py:356`) | yes |

Numeric point (independent constants: $\hbar c=1973.269804$ eV Å,
$m_ec^2=510.99895$ keV, and $|\mathbf g_{002}|=4\pi/6.711$ Å$^{-1}$). The
inputs are hopg (002), 100 keV, flight along $\mathbf g$, and
$\hat{\mathbf n}$ at 60°. The seed centre and the closed form agree to
$2.2\times10^{-16}$ relative, both at 2790.5776 eV.

Differences that are disclosed or harmless:

- The seeds read unclipped segments. The kernels first apply
  `_clip_segments_to_cutoff` with `E_cut_lines_keV` (default 5 keV), which
  drops rows below 5 keV and shortens terminal flights. The docstring discloses
  this. The resulting band is wider, but the quantile is not strictly
  conservative: an extra heavy interior weight moves $q_\pm$ inward in
  relative terms of the true population. Not measured.
- Under `max_dE_frac` substeps, the kernels add the rows of one flight
  coherently (`_setup.py:264–290`). The per-row $t_L$ then overstates the width
  of the flight's feature. Both `feature_width_eV` and
  `narrowest_feature_width_eV` are per-row. This is opt-in only
  (`transport/api.py` default `max_dE_frac=0.0`), and it is outside the
  measured cases.

**Verdict: pass.**

### Check 2: vacuum root inside the tail margin of the in-medium root

The implementation seeds from the vacuum root (l. 201), exactly as documented.
The divergence is in the docstring's justification (l. 119–123) and in the
ledger Check. The derivation in §1.2 shows that the per-segment offset in units
of the margin is not $O(\delta)$. Measured results appear in §3. **Verdict:
fail as stated.** Coverage is intact in every measured case, because the
population band, not the margin, contains the shifted roots.

### Check 3: dropped-reflection rule

The rule is at l. 227–234: `share == 0.0 or dropped_weight + share <= epsilon * total`
over the reflections sorted ascending, stopping at the first failure. The
quantiles use `searchsorted(..., side="left")` (l. 80–85) at
$(\varepsilon/2,\,1/2,\,1-\varepsilon/2)$ (l. 240–244). This is exactly the
construction of §1.4, so $U<2\varepsilon W$. A randomized check ran 300
synthetic populations with six reflections and
$\varepsilon\in\{10^{-3},10^{-2},0.05,0.2\}$, using a margin of essentially
zero. The largest value of $U/(2\varepsilon W)$ was 0.93. The bound held and
was nearly tight. **Verdict: pass**, with the in-band normalization caveat of
§1.4.

A robustness finding, not physics: `_line_grid_policy` accepts
`tail_widths = 0` (l. 345–347). A reflection whose radiating population is a
single value, or all coincident values, then produces
`below_eV = above_eV = 0`, and `FeatureSeed` raises
`has an empty window and no anchor`. The failure was reproduced with
`tail_widths=0.0` in the randomized check.

### Check 4: Chantler bracket versus xraydb edge energy

The edge seeds read `load_henke`, which returns `xraydb.chantler_energies` and
`f2_chantler` (`materials/atomic.py:225–236`). This is the same table that
`absorption_length_ang` and the kernel's native-node tabulation read. Both
bracket nodes become anchors (l. 326–348). An independent recomputation with
xraydb gives the steepest adjacent $f_2$ ratio in $\pm5\%$, and the xraydb
energy lies above the bracket by:

| Edge | xraydb (eV) | Chantler bracket (eV) | xraydb above lower / upper node |
| --- | --- | --- | --- |
| N K | 409.9 | 401.41–401.60 | 2.11% / 2.07% |
| O K | 543.1 | 531.71–531.99 | 2.14% / 2.09% |
| F K | 696.7 | 685.00–685.39 | 1.71% / 1.65% |
| P L3 | 135.0 | 132.13–132.20 | 2.18% / 2.12% |
| C K | 284.2 | 283.67–283.80 | 0.19% / 0.14% |

The "up to 2.1%" statement holds when measured to the upper node. Measured to
the lower node it is 2.18%, and the 5% search window absorbs either figure.
The offset is not always positive: S L3 xraydb (162.5 eV) sits 1.3% *below*
its bracket (164.66–164.76 eV). The symmetric search handles that case.
**Verdict: pass** for the check as worded.

**Divergence in claim (2), which says "the absorption jumps of the Chantler
table":** the steepest bracket is chosen over the whole window (l. 308–315),
and a bracket index that has already been seeded is skipped (l. 319–322).
Suppose a shell's own jump lies within 5% of a stronger jump. The search then
returns the stronger bracket, and the shell is dropped. It does not appear in
the seeds or in `skipped`. The docstring describes such brackets as "shared by
two shells", but the two shells have separate brackets in the table:

| Shell | xraydb (eV) | Own Chantler jump (eV) | Modelled $\mu$ step | Search returns |
| --- | --- | --- | --- | --- |
| Se L2 | 1474.3 | 1474.16–1478.24 (ratios 1.13, 1.07, 1.11) | $\times1.35$ | Se L3 bracket 1433.81 |
| W M4 | 1872.0 | 1867.58–1878.66 (1.11, 1.09, 1.11) | $\times1.36$ | W M5 bracket 1805.32 |
| W L1 | 12100 | 12092.59–12107.01 (1.115) | $\times1.125$ | W L2 bracket 11538.43 |
| Mo L2 | 2625 | 2621.48–2628.72 (1.10, 1.11, 1.08) | — | Mo L3 bracket |
| Ga, As L2 | 1143.2, 1359.1 | 1140.8–1143.8, 1356.7–1360.5 | — | L3 bracket |
| Ti L2 | 460.2 | 461.02–461.98 (1.17, 1.14) | — | Ti L3 bracket |

The $\mu$ steps come from `absorption_length_ang` across each window. For
comparison, W L2 ($f_2$ ratio 1.28) and Mo M5 (1.30) *are* seeded. The
`absorption_edge_seeds(["W", "Se"], 10, 20000)` call returns only Se L3, Se K,
W M5, W L3, and W L2. The wse2 bandwidths (10–1700, 50–3100, and 50–10200 eV
at 30, 100, and 300 keV) contain Se L2 in every case and W M4 in two. The
single-bracket threshold `EDGE_MIN_F2_RATIO = 1.05` also skips W M3, whose
jump is split over three nodes (1.041, 1.039, 1.032), about 12% in total. That
behaviour is the stated policy.

**Divergent term:** `argmax` over the whole $\pm5\%$ window combined with
`if index in seen: continue`.

### Check 5: characteristic cutoff

The seeds use `max(table.recommended_cutoff_eV)` over the composition's
elements, then keep lines with `line_yield_per_vacancy.any(axis=0)` and
`centre > cutoff` (l. 378–388). The kernel uses the same maximum
(`characteristic.py:764–766`) and skips `line_energy <= relaxation_cutoff`.
It also requires `_xraydb_line_yields(table, cutoff) > 0`, which is
`line_yield_per_vacancy` with columns at or below the cutoff zeroed
(l. 692–694, 817–827). The two sets agree for non-negative yields.

The compositions match as well. The kernel uses `case["composition"]` for one
layer and `abs_layers[i][2]` for each layer (`runner/emission.py:96–123`).
The seeds use `abs_layers` compositions when present and `composition`
otherwise (l. 466–470).

An enumeration over all 26 catalog crystal elements compared the kernel's line
set with the seed set and found zero mismatches. The seeds' extra condition
`fwhm > 0` removes no line.

**Verdict: pass.**

### Limiting cases (`build_window_plan`)

- **Single straight flight.** The quantiles collapse to $E_0$. The window is
  $E_0\pm\tau w$ at spacing $w/8$, and the reflection is never dropped, since
  $W_r\le\varepsilon W_r$ is impossible for $\varepsilon<1$. This is confirmed
  by the independent numeric point above (half-extents $1.6=2\times0.8$ eV,
  spacing 0.1 eV) and by
  `test_single_flight_window_is_centred_on_the_closed_form_resonance`.
  **Pass.**
- **No seeds.** The breakpoints are $\{\texttt{start},\texttt{stop}\}$, giving
  one piece at the backbone spacing. `coordinates()` returns
  `np.linspace(start, stop, resolution_num(...))` (`_line_windows.py:159–165,
  228–235`). Across 2000 random $(\texttt{start},\texttt{stop},h)$ triples
  there were zero mismatches with `np.array_equal`. **Pass.**

### Tests

`uv run pyrite-dev test tests/energy-grid/test_line_seeds.py tests/energy-grid/test_line_windows.py`
gives 28 passed. No test covers a secondary shell near a stronger edge, and no
test covers the in-medium offset.

## 3. Measured vacuum-to-in-medium offsets

The script builds each case with `build_ladder_case` and one CPU transport
(`PYRITE_MC_BACKEND=cpu`, seed 7, small $N_e$). It computes the kernel's own
in-medium root per segment with `_in_medium_kinematics` over
`refractive_index(...).real` on a $3\times10^1$–$1.2\,E_{\rm stop}$ geometric
table. Seeds use `feature_width_eV` $=w_\varepsilon$ from
`sinc_feature_spacing` at $\varepsilon=10^{-3}$.

The first-order formula of §1.1 reproduces the kernel's shift to 1.1–1.5% of
the largest shift.

| Case ($N_e$) | $w_\varepsilon$ (eV) | max rel. shift, $E_0>300$ eV | max shift / $w_\varepsilon$ (worst per-reflection p99) | in-medium weight inside window vs vacuum |
| --- | --- | --- | --- | --- |
| hopg 30 keV tilt 5 (60) | 1.094 | $1.2\times10^{-3}$ | 0.41 (0.27) | identical |
| wse2 30 keV tilt 85 (40) | 10.76 | $3.8\times10^{-3}$ | 0.11 (0.09) | identical |
| hopg 300 keV tilt 85 (20) | 0.310 | $4.6\times10^{-3}$ | **5.4** (4.2) | identical to $<10^{-8}$ |
| wse2 300 keV tilt 5 (10) | 2.206 | $2.7\times10^{-2}$ | **3.8** (1.6) | differs by $\le6\times10^{-6}$ |

In the hopg 300 keV case, 0.7–2.7% of each reflection's proxy weight above
300 eV has an in-medium root more than $2w_\varepsilon$ from its vacuum root.
Those segments are forward-going ($\cos\Theta=0.76$–$0.99$), lie at
$E_0=300$–$1300$ eV, and are all shifted downward, as §1.1 predicts. Median
relative shifts are $2\times10^{-5}$–$3\times10^{-4}$, and the maxima reach
$10^{-3}$–$10^{-2}$. "$\sim10^{-5}$" is therefore an underestimate.

Below 300 eV, the largest offset is 2.7–3.2 $w_\varepsilon$ at hopg 30 keV
and 18–35 $w_\varepsilon$ at hopg 300 keV. The kernel rejects (returns NaN
for) $6\times10^{-5}$–$1\times10^{-2}$ of the vacuum-radiating weight.

Coverage survives all of this in these cases because the windows are very
wide. The hopg 300 keV (002) window is 9.8–3381 eV at 0.039 eV spacing.
Below 300 eV, the in-medium weight outside the window is 0 to
$9\times10^{-4}$ of that sub-band. That is at or below the vacuum figure in
every reflection. The protection comes from the breadth of the population,
not from the margin. A narrow population, such as a thin foil at low tilt,
would not have it.

## 4. Acceptance evidence, read critically

The row also requires the #101 window ladder. The
[error budget](line-spectrum-error-budget.md) reports that the finest windowed
rung of all twelve cases agrees with a uniform reference inside the window
share. Worst cases: yield $1.4\times10^{-4}$, FWHM $8.7\times10^{-4}$. These
numbers could not be reproduced here. The raw JSON
(`line_window_convergence_101b.json`) is deliberately untracked and lives on
the remote box. Reading the report and the harness
(`energy_grid/convergence_case.py`):

- **The ladder is blind to seed placement.** `window_ladder` rebuilds the
  seeds per rung, but only `samples_per_feature` changes. Window positions
  and extents are fixed, and edge spacing does not refine. The Richardson gate
  therefore cannot detect a mis-placed or missing window, including the two
  findings above. Only the comparison against the uniform reference can.
- **The reference is capped.** `reference_grid` uses
  `max(finest window spacing, span / 400000)`. For hopg 300 keV, with the band
  at 50–9100 eV, that gives about 0.023 eV. This is only 1.7× finer than the
  8-sample rung it is compared against, since the 16- and 32-sample rungs were
  dropped. It is still about 13 samples per $w_\varepsilon$, so it is a
  credible reference, but it is not "dense" relative to the rung. The report
  should state the reference spacing per case.
- **Windows degenerate to uniform at 100–300 keV.** The report says so for
  hopg 100 keV tilt 5, and it is reproduced here for hopg 300 keV. In those
  cases the comparison tests a fine uniform grid, not the window/backbone
  transition. The evidence that windowing proper holds accuracy rests mostly
  on the 30 keV cases.
- **Per-case gate outcomes are not tabulated.** The ledger Check asks that
  yield, centroid, and FWHM converge under window refinement. The report
  tabulates only the worst reference differences. It states one Richardson
  failure (wse2 100 keV tilt 85, line/background, which is not a required
  observable), and it states that hopg 300 keV rests on one 2/4/8 triple.
  Whether FWHM passes the gate in every case is asserted
  ("converges in … FWHM") only for wse2 100/85.
- **The width-tail argument is consistent with these runs.**
  `samples_at_narrowest` came out as exactly 8.0 in the hopg 30 keV ($N_e=60$)
  and hopg 300 keV ($N_e=20$) seeds here. It does not cover the substep case
  noted under check 1.

With these qualifications, the acceptance evidence is plausible. It does not
test the two coverage findings, because the ladder cannot see them, and the
reference comparison did not flag them in the cases run.

## 5. Initial adjudication and recommended actions (pre-fix)

Status: **`discrepancy`**. No radiation equation is wrong, and the kernel
conventions are reproduced exactly. The discrepancy is that two coverage
properties the row claims do not hold as stated.

1. **In-medium offset.** Either correct the docstring and ledger justification
   to the boxed condition of §1.2, or seed from the kernel's own root. Calling
   `_in_medium_kinematics` with the same `refractive_index` table is cheap on
   the host. It would also reproduce the kernel's NaN rejections and its 10 eV
   floor on the same root, which removes most of the sub-300 eV caveat.
2. **Secondary shells.** Search each shell in a window that excludes the
   bracket ramp of stronger shells already seeded, for example by taking the
   steepest bracket not adjacent to a `seen` index. Otherwise, report shells
   that resolve to an already-seen bracket instead of dropping them silently.
   Add a test on Se L2 or W M4.
3. **Minor items.** Reject `tail_widths = 0` in the policy, or handle a
   degenerate population. State that the $2\varepsilon$ bound is relative to
   all radiating weight. State that the per-row width is not a flight width
   under substeps.

Human sign-off remains required.

## 6. Fresh-context re-verification after remediation

Fresh-context re-verification on 2026-09-17 started again from the cited
source equations, units, assumptions, and limiting cases before inspecting the
remediated implementation. No implementation file was modified.

- **Claim**: `line-window-seeding` —
  `montecarlo/spectrum/line_seeds.py::kinematic_line_seeds`,
  `::absorption_edge_seeds`, and `::characteristic_line_seeds`. The governing
  equations are the in-medium resonance
  $E_{\rm res}=\hbar c\,(\mathbf v\cdot\mathbf g)/
  (1-\operatorname{Re}n(E_{\rm res})\,\mathbf v\cdot\hat{\mathbf n})$ and the
  finite-time first-zero width $2\pi\hbar c/(D t_L)$, together with the
  Chantler $f_2$ edge brackets and xraydb characteristic energies and widths.
- **Filters**: units pass; limiting cases pass; signs and conventions pass,
  conditional on the repository's inherited positive-$\mathbf g$ convention.
  This claim does not resolve the separate `line-energy-dispersion`
  harmonic-sign discrepancy.
- **Re-derivation**: `matches`. Production passes the case band and composition
  into kinematic seeding; seeding builds the kernels' padded native-node table,
  calls their `_in_medium_kinematics`, rejects the same unsettled roots, and
  derives feature widths from the same denominator as `a_width`. The
  independent pointwise-Brent regression pins the in-medium centre.
- **Secondary-edge remediation**: `matches`. Shells are processed strongest
  first, but each weaker shell selects the strongest qualifying bracket outside
  already claimed native-node windows. The Se L2 regression pins the previously
  omitted nearby edge; independent probes also found W M4/L1 and Mo/Ga/As/Ti L2
  anchors within 0.24% of their nominal edges.
- **Verdict**: `rederived`; existing regression anchors advance the ledger state
  to **`anchored`**. Human sign-off remains required.

Focused verification reported 54 passing seed/window tests and five passing
documentation tests. The built page contained rendered `math notranslate`
nodes with no surviving dollar delimiters. A supplementary Z=3--92 catalog
sweep was incomplete because the current xraydb/SciPy stack rejects Cs's
non-strictly-increasing Chantler coordinates; this does not affect the targeted
secondary-edge checks above.
