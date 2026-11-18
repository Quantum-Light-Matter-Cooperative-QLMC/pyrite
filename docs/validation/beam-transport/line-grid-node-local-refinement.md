# `line-grid-node-local-refinement`

Ledger row: [`line-grid-node-local-refinement`](../ledger-core-coherent-physics.md#line-grid-node-local-refinement). Status: anchored (fresh-context read-only verification, 2026-11-17, `ca9a7e7d0`: `rederived`; regression below). Validation: `line-grid-node-local-refinement`. Issue #1257.

## Claim

When the uniform `sinc-oversampled` axis (step $h_f=\min(W_q,h_\mathrm{max})/K$, with $W_q$ the sinc-Nyquist step, $K=8$, and $h_\mathrm{max}$ = `max_spacing_eV` = 3 eV) would exceed the point budget, the resolver builds a piecewise axis instead of refusing. The window step is $h_f$, the backbone step elsewhere is $h_b=h_\mathrm{max}$ ($W_q$ is a line-width quantile with no role where no line resonates, and a $W_q$ backbone alone exceeds the budget on a 1 MeV kinematic band), and the windows are seeded from the case's line population (`node_local_seeds`). The axis is accepted only if it fits the same budget. Where the uniform axis fits, the resolver output is unchanged.

Let a line have first-zero width $w \ge h_f$ and whole-line mass $M$, and let $\eta$ be `DEFAULT_NODE_JOIN_LIMIT` $=10^{-4}$. Under node (trapezoid) quadrature, every spacing join on one side of the line shifts its integrated yield by at most $\eta M$. The total join error per line is therefore at most $2\eta M$, beyond the out-of-band tail mass that a uniform axis on the same band also omits. The population-weighted yield error is at most $2\eta$.

The claim covers mass only. The windows give every line narrower than $K h_b$ the uniform axis's $h_f$ step over its resonance. A wider line keeps at least $K$ backbone nodes per first-zero width. No new peak or FWHM bound is derived; shape is qualified by measurement (#1257 Plan).

## Setup

Take one line from the finite-time lineshape ([`finite-time-lineshape`](../ledger-core-coherent-physics.md)), $f(x)=A\,\operatorname{sinc}^2(x/w)$ in NumPy's convention, with $x=E-E_\mathrm{res}$ and $M=\int_{\mathbb R}f=Aw$. Write $g(u)=\sin^2u/u^2$, so $f(x)=A\,g(\pi x/w)$. The node quadrature over coordinates $x_0<\dots<x_N$ uses the weights of `_grid_semantics.node_bin_edges_and_widths`. These are the trapezoid rule $T$ except at the two band-edge nodes, whose weight is the full first or last step. That band-edge term is shared with the uniform axis and never sits at a join.

Assumptions:

1. **Central piece.** The resonance $x=0$ lies in a uniform piece of spacing $h_c\le w$, with nearest joins $x_-=-d_-<0<x_+=d_+$. A side with no join reaches the band edge.
2. **Backbone cap.** No interval anywhere is longer than $h_b$.
3. **Absorption-free shape.** The line is the pure $\operatorname{sinc}^2$. Production node scoring uses the formation integral with absorption, which is not strictly band-limited; the uniform axis shares that error. Absorption scales the oscillating tail relative to the mass by $y e^{-y/2}/(1-e^{-y})\le1$ ($y=\mu L$). The remaining smooth part falls under the Peano bound, so the bound is expected to carry over, but it is not derived here.

## Derivation

**Exactness of the central lattice.** The Fourier transform of $f$ is a triangle supported on $|\nu|\le1/w$. By Poisson summation, $h_c\sum_n f(x_-+nh_c)=\int_{\mathbb R}f$, because every alias $k/h_c$ with $k\ne0$ lies at or beyond $1/w$, where the triangle vanishes. This holds for any lattice offset.

**Decomposition.** Write $T_{h_c}[x_+,\infty)$ for the trapezoid sum on the continued central lattice, with half weight at $x_+$, and likewise $T_{h_c}(-\infty,x_-]$. On the central piece,

$$
T_{h_c}[x_-,x_+]=\int_{\mathbb R} f - T_{h_c}(-\infty,x_-] - T_{h_c}[x_+,\infty).
$$

Adding the axis's own sums beyond the joins gives

$$
T-\int_{\mathbb R}f = e_- + e_+,\qquad e_+ = T_\mathrm{axis}[x_+,x_N]-T_{h_c}[x_+,\infty),
$$

and the mirror expression for $e_-$. Each side therefore depends only on its nearest join distance, the central step and the largest step beyond. The number of joins further out does not enter.

**Tail bound.** For $x\ne0$, $f\le \mathrm{env}(x)=Aw^2/(\pi^2x^2)$, which is decreasing on $x>0$. Take any partition of $[d,\infty)$ with intervals $\Delta_j\le h$. Its lower sum of $\mathrm{env}$ is at most $\int_d^\infty\mathrm{env}$. Its upper sum exceeds the lower by $\sum_j\Delta_j[\mathrm{env}(x_j)-\mathrm{env}(x_{j+1})]\le h\,\mathrm{env}(d)$. The trapezoid sum is the mean of the two, so

$$
0\le T[d,\infty)\le \int_d^\infty \mathrm{env}+\tfrac h2\mathrm{env}(d)=M\,\frac{w}{\pi^2d}\left(1+\frac{h}{2d}\right).
$$

Both terms of $e_+$ lie in this interval, with $h=h_b\ge h_c$, so

$$
|e_+|\le M\,\frac{w}{\pi^2 d_+}\left(1+\frac{h_b}{2d_+}\right).
\tag{tail}
$$

**Peano bound.** On one interval $[p,p+\Delta]$ the trapezoid error is $\tfrac12\int(x-p)(p+\Delta-x)f''(x)\,dx$, and its kernel is at most $\Delta^2/8$. Add and subtract $\int_{x_+}^{x_N}f$ and $\int_{x_+}^\infty f$:

$$
e_+ = \Big(T_\mathrm{axis}[x_+,x_N]-\!\int_{x_+}^{x_N}\!f\Big)-\Big(T_{h_c}[x_+,\infty)-\!\int_{x_+}^{\infty}\!f\Big)-\int_{x_N}^\infty f .
$$

The last term is the out-of-band tail. Since $\int_d^\infty|f''|\,dx = A(\pi/w)\,G(\pi d/w)$ with $G(u)=\int_u^\infty|g''|$,

$$
|e_+ + \textstyle\int_{x_N}^\infty f|\le M\,\frac{\pi\,(h_b^2+h_c^2)}{8w^2}\,G\!\left(\frac{\pi d_+}{w}\right).
\tag{Peano}
$$

Here $g''(u)=2\cos2u/u^2-4\sin2u/u^3+3(1-\cos2u)/u^4$, so $G(u)\le 2/u+2/u^2+2/u^3$. Over the whole line, $G(0)=\int_0^\infty|g''| = 1.4743$, by adaptive quadrature on $[0,2000]$ plus the tail bound $2/2000$; the code uses `SINCSQ_CURVATURE_INTEGRAL = 1.5`.

**Join distance.** `node_join_halo_eV` returns the smaller of two distances, each of which makes one bound at most $\eta$:

- Tail: $d=w/(\pi^2\eta)+h_b/2$. With $a=w/(\pi^2\eta)$, the bound equals $\eta\,a(a+h_b)/(a+h_b/2)^2\le\eta$.
- Peano: let $G^*=8w^2\eta/(\pi(h_b^2+h_c^2))$. If $G^*\ge1.5$, no distance is needed. If $1\le G^*<1.5$, take $d=3w/\pi$, where $G(3)\le0.963$. If $G^*<1$, take $d=3w/(\pi G^*)$, where $u\ge3$ and $G(u)\le(2/u)(1+1/3+1/9)<G^*$.

**Window assembly.** Each line with $w<Kh_b$ asks for $h_f$ over $E_\mathrm{res}\pm d$, rounded out to 100 eV, with $h_c=h_f$. Each wider line asks only that no join fall inside $E_\mathrm{res}\pm d$, with $h_c=h_b$ (conservative, since it may end up in a window). Intervals that overlap form components, and a component holding any fine window is refined whole (`_fine_union`). Spacing joins of `build_window_plan` sit only at the ends of fine runs. So every join lies outside every line's interval, which gives Assumption 1 with $d_\pm\ge d$: $h_c=h_f\le w$ for lines in windows, and $h_c\le h_b\le w/K$ for wide lines on the backbone. The plan's sliver rule only removes joins. Assumption 2 holds because every piece is at most $h_b$.

## Limiting cases

- **Uniform axis** (no windows, or one window over the whole band): there are no interior joins, $e_\pm$ reduce to the band-edge terms, and the axis is the uniform one.
- **Narrow line** ($w\ll h_b$): the tail distance $\approx w/(\pi^2\eta)$, the `resonance-local` halo of [`line-grid-resonance-local-spacing`](line-grid-resonance-local-spacing.md) at $\eta=10^{-4}$.
- **Wide line**: no distance is needed once $w\ge\sqrt{1.5\pi(h_b^2+h_c^2)/(8\eta)}$ (325 eV for $h_b=h_c=3$ eV). Such a line is integrated within $\eta$ whatever the joins.
- $h_b\to h_f$: the Peano bound vanishes as $(h_b^2+h_c^2)$ shrinks with the steps.

## Budget

The bound is per line and per side, so the population-weighted yield error is at most $2\eta=2\times10^{-4}$. Like the `resonance-local` halo share, it is charged to the feature-window row of `tbl-line-budget-allocation` ($5\times10^{-4}$), which an axis without feature windows does not otherwise spend. With the $10^{-4}$ bandwidth share, the row then holds $3\times10^{-4}$. Lines narrower than $h_f$ violate Assumption 1. They alias on the uniform axis too and lie inside its aliased-weight budget ($w<W_q$).

## Numerical check

`tests/energy-grid/test_line_grid_node_local.py::test_join_halo_bounds_single_line_node_mass` places one line, for $w$ from 0.4 eV to 2,000 eV, on its own local axis ($h_b=3$ eV, $h_f=0.375$ eV, ±60 keV band). Over 12 random centres it compares the trapezoid against the closed-form band mass. The worst sampled error is $6.8\times10^{-5}$ at $w=1$ eV and $2.6\times10^{-5}$ at $w=3$ eV; every other width stays below $3\times10^{-7}$. The bound is $2\times10^{-4}$. The attainable worst case is larger than this sample. When $h_b$ is a multiple of $w$, backbone nodes land on the $\sin^2$ maxima, and with the join exactly at the halo a phase scan gives about $0.47\eta$ per side at $w=0.5$ eV, roughly $9.4\times10^{-5}$ per line. The tail bound is tight to the expected factor of about 2 (the mean of $\sin^2$ is $1/2$). `test_join_error_grows_when_the_halo_shrinks` shows the halo is necessary: at $100\eta$, a $w=1$ eV line reaches $1.0\times10^{-3}$.

Measured axes (author probe, 2026-11-17, CPU float64):

| case | uniform $h_f$ nodes | local nodes | windows |
|---|---:|---:|---:|
| Si 1 MeV kinematic ceiling (20 histories, 196 lines at 5.6–6.8 keV) | 2,576,401 | 338,152 | 1 (6.9 keV) |
| Si 1 MeV measured band, `CxrReflections(n_families=4)`, 40 histories (482k lines, 33 eV–17.7 keV, band to 57.6 keV, $W_q=0.472$ eV) | 974,696 | 399,418 | 1 (22.9 keV) |

## Bin-mean quadrature (#1265)

Since #1265 the route also serves `bin-mean` `sinc-oversampled` axes. The window
step is `h_f = min(W_q, h_b) / K` with `K = 1/shape_spacing_factor(1e-2) = 10π/3
≈ 10.47` (`_cxr_grid_policy.py::uniform_step_divisor`), and lines narrower than
`K h_b ≈ 31.4 eV` get windows. Node-quadrature axes keep `K = 8` and are
unchanged.

Mass. Bin-mean bins on any node axis have contiguous midpoint edges
(`_grid_semantics.node_bin_edges_and_widths`). The exact bin mean times width is
`π{F(x_{i+1}) − F(x_i)}/a`, so the sum over bins telescopes to the outer-edge
mass. That makes the join error zero, and the node halos above are not needed
for mass, though they do no harm. Production scoring replaces the exact mean
with the averaged envelope beyond 64 first-zero widths
(`sinc-bin-far-envelope`). That term is at most about 1.2e-5 per side for any
contiguous edges, so the grid dependence is at most about 2.4e-5 per line,
below `2 eta`.

Shape. Lines left on the backbone have `h_b <= w/K`. Window lines with
`w >= W_q` have `h_f <= w/K`. So no peak bin is wider than
`shape_spacing_factor · w`, including at a join, and the isolated-line peak
bound of `line-grid-sinc-shape-spacing` holds on the local axis too.

Fresh-context verification of this section (2026-11-18): `rederived`. Probe:
exact bin masses over the local axis equal the outer-edge mass to 6.7e-16 for
50 lines. Anchor:
`tests/energy-grid/test_line_grid_node_local.py::test_bin_mean_axis_over_budget_takes_the_node_local_route_at_the_shape_step`.

## Open

- Shape (peak, FWHM, mass L1) and timing against a same-transport node 0.046875 eV reference: the remote qualification arm (#1257 Plan).
- Whether the local axis should become the default everywhere (#1257 Decisions).
