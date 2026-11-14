# Batched incoherent line path: layered and grooved escape (#1194)

Validation: `line-hkl-batch`.

Scope: the #1194 extension only. It covers incoherent layered and grooved
absorbers on the batched $(n_{\rm seg}, N_g)$ route, the folded EPDL
attenuation table `_escape_mu.exact_mu_table`, and the per-hkl loop's switch
to that table. The single-slab batched A/B recorded earlier in the row was not
re-run here.

## Independent derivation

This section was written from the ledger row, the reference host evaluator
(`photon_cross_sections._interpolate`, `attenuation._mu_total_inv_ang`), and
the escape-piece contract (`segment_escape_pieces`, `piece_mean_transmission`)
before reading `_escape_mu.py` or the route diff.

### Which escape quantities depend on the reflection

For segment $s$ and reflection (and mosaic orientation) $g$, the incoherent
line weight is $w_{sg}=P_{sg}\,\lvert A_{sg}\rvert^2\,T_{sg}$, with the escape
factor written as the segment mean of the Beer–Lambert transmission over
linear escape pieces $p$ and layers $k$:

$$
T_{sg}=\phi_s\sum_p f_{sp}\,
\overline{\exp\!\Big(-\sum_k \mu_k(E_{sg})\,\ell_{spk}(\xi)\Big)}^{\,\xi\in[0,1]},
\qquad
\ell_{spk}(\xi)=(1-\xi)\,\ell^{\rm start}_{spk}+\xi\,\ell^{\rm end}_{spk}.
$$

The photon leaves along the fixed observation direction $\hat{\mathbf n}$.
The geometry (slab, layer stack, blazed groove) therefore fixes the piece
fractions $f_{sp}$, the piece fraction $\phi_s$ and the per-layer path ends
$\ell^{\rm start,end}_{spk}$ per segment, independent of $g$. Only the line
energy $E_{sg}$ depends on $g$, through the resonance condition. Because the
exponent is linear in $\xi$, the mean is
$(e^{-\tau_0}-e^{-\tau_1})/(\tau_1-\tau_0)$ with
$\tau_{0,1}=\sum_k\mu_k(E_{sg})\,\ell^{\rm start,end}_{spk}$. So a batched
evaluation needs:

- `(N, P)` fractions and `(N, P, K)` path ends broadcast over $g$;
- a `(N, M, K)` tensor $\mu_k(E_{sg})$ with $M=N_g$ in the batch and $M=1$
  per row in the per-hkl loop;
- one table per layer composition in the stack order used to build the path
  ends, or one table (the crystal) for a groove.

No other layer- or groove-dependent quantity enters the incoherent weight.
Refraction and the escape gradient belong to the crystal and are shared by
both routes. The two routes therefore differ only in float reassociation if
they evaluate the same $\mu_k(E)$ elementwise.

### Folding the EPDL channels

Each channel $c$ (element $i$, one of five processes) has ascending knots
$e^c_0\le\dots\le e^c_{n_c-1}$, which may repeat at an edge. The host value is

$$
\sigma_c(E)=
\begin{cases}
0, & E<e^c_0,\\[2pt]
s^c_{i}+\dfrac{E-e^c_i}{e^c_{i+1}-e^c_i}\,(s^c_{i+1}-s^c_i), &
i=\min\!\big(\#\{m: e^c_m\le E\}-1,\;n_c-2\big),
\end{cases}
$$

and $\mu(E)=\sum_c n_{i(c)}\sigma_c(E)$ inside
$[E_{\min},E_{\max}]=[1\ \text{eV},10^{11}\ \text{eV}]$, NaN outside. Write
$u_0<u_1<\dots<u_{J}$ for the distinct union of all channel knots. On
$[u_j,u_{j+1})$ no channel has a knot in the open interval. The count
$\#\{m:e^c_m\le E\}$, the threshold test $E\ge e^c_0$ and the clip are then
constant for every channel. Each $\sigma_c$ is one affine function there, and
so is the weighted sum:

$$
\mu(E)=V_j+S_j\,(E-u_j),\qquad E\in[u_j,u_{j+1}),
$$

where $V_j$ is the right-continuous host value at $u_j$ and $S_j$ is the sum
of the selected channel slopes. On $[u_J,\infty)$ every channel keeps its
clipped last interval, so the same form holds with $j=J$. Evaluating with
$j=\#\{u\le E\}-1$ (`side="right"`) selects the post-edge branch at a repeated
knot, as the host does, and reproduces each channel's interval selection
exactly. The only condition is that the $(V_j,S_j)$ build selects, for each
channel, the interval that is live on $[u_j,u_{j+1})$. Any probe
$x\in[u_j,u_{j+1})$ achieves this. The probe $x=u_j$ is always valid. A
midpoint $\tfrac12(u_j+u_{j+1})$ computed in floating point is valid only if
it rounds strictly below $u_{j+1}$. If $u_{j+1}$ is the float successor of
$u_j$, round-half-even can return $u_{j+1}$.

Rounding: all channel values are non-negative and, on a selected interval,
lie between their two endpoint values. The terms of $V_j+S_j(E-u_j)$ are
therefore bounded by a small multiple of $\mu(E)$. The folded evaluation
then agrees with the host to a few float64 ulp, with no cancellation
amplification.

## Comparison with the implementation

| Item | Expected | Implementation | Result |
|---|---|---|---|
| Escape pieces $f_{sp},\phi_s,\ell_{spk}$ | $g$-independent, shared | `st.escape_pieces`, `st.piece_fraction` from setup; indexed by segment block `blk.sb` only | match |
| $\mu_k(E_{sg})$ shape | `(N, M, K)` | `escape_mu` stacks per-layer tables on the last axis: `(nb, N_g, K)` batched, `(n, 1, K)` per-hkl | match |
| Layer order | stack order of the path ends | `request.layers` order in both arms (the path ends come from the same `layers`) | match |
| Groove | one absorber, crystal composition | `(exact_mu_table(abs_comp),)`; $K=1$ goes through the fused `piece_transmission` (CuPy float32 only) or the generic path | match |
| Other uses of the slab `blk.mu` | none in the incoherent weight | `blk.mu` is used only by `_escape_weights` (single slab) and the coherent block | match |
| Routing | batch only incoherent, ungrouped, non-temporal, no capture | `_needs_per_hkl_route`: layered/grooved go per-hkl if `coherent` or the switch is off; `grouped`, `temporal`, `coefficient_capture` still force per-hkl; coherent layers are refused in setup | match |
| Union knots and interval selection | as derived | `np.unique` union; `searchsorted(side="right")` and clip in both build and evaluation | match |
| Build probe | any $x\in[u_j,u_{j+1})$ | midpoint `0.5*(knots[:-1]+knots[1:])` | **latent divergence**: see below |
| Above last knot | clipped last interval | probe `2*knots[-1]`; evaluation clip `size-1` | match (moot: every channel ends at $E_{\max}$) |
| Out-of-band | NaN | float64 comparison after casting the input | match |
| float32 input | host casts to float64 | `energy_eV.astype(np.float64)` before searching; no knot rounding to float32 | match |

### Latent divergence: ulp-adjacent union knots

`_escape_mu.py:73` builds the per-interval selection from a midpoint. Take a
synthetic composition with one channel constant at 1 from 1 eV, and a second
channel whose threshold is the float successor $b$ of
$a=\mathrm{nextafter}(1000,\infty)$, jumping from 0 to 5 at $b$. The midpoint
of $[a,b)$ rounds to $b$, the second channel is treated as live on
$[a,b)$, and the table returns $\mu(a)=6$ where the host returns $1$.

The current EPDL2025 data cannot trigger this case. Across the union of every
knot of all 100 elements (87,205 distinct knots), the smallest relative gap is
$1.05\times10^{-9}$, and no midpoint collides with an endpoint. Every
composition's union is a subset of that union, so no composition can trigger
it. Edges are stored as exact duplicates, which `np.unique` collapses.
Building with the probe $x=u_j$ (the knot itself, with `side="right"`) removes
the dependence on rounding at no cost.

## Numerical evidence

All runs used CPU and `REAL = float64` in this worktree at `ef26803d9`.

- `tests/montecarlo/spectrum/test_layered_batched_route.py` and
  `test_spectrum_phases.py`: 53 passed.
- Table against host, all 100 elements as single-element compositions plus
  six mixtures (including U/O and H/Pb/W). Each was evaluated at every union
  knot, its float64 neighbours, interior points at 0.5 and 0.7 of each
  interval, every knot rounded to float32 and that float32 value's
  neighbours, and the band ends. The NaN pattern was identical. The worst
  difference was 4.0 float64 ulp of $\mu$ (relative $\le 6.8\times10^{-16}$).
- EPDL survey: no channel has a duplicate last knot, every channel's last
  knot is $10^{11}$ eV (equal to $E_{\max}$), and the minimum first knot is
  1 eV (equal to $E_{\min}$). The below-threshold-zero branch below $u_0$ and
  the above-last-knot extrapolation are therefore unreachable inside the band.
- Batched against per-hkl spectra, with the row's fixture (hopg, four
  reflections, 48 rows, 500–4000 eV grid). Front-exiting stack, node and
  bin-mean: max abs $4.69\times10^{-16}$ and $7.82\times10^{-16}$ of peak,
  max rel $5.7\times10^{-16}$ and $8.2\times10^{-16}$. Back-exiting stack:
  $4.64\times10^{-16}$ and $6.19\times10^{-16}$ of peak, rel up to
  $1.10\times10^{-15}$. Groove: $4.83\times10^{-16}$ and
  $6.04\times10^{-16}$ of peak. Integral relative difference
  $\le 2.2\times10^{-16}$. None of the cases is bitwise identical.
- Device float32 (job 12551, not re-run): relative differences up to
  $1.2\times10^{-5}$ on bins above peak×$10^{-6}$, and integral differences
  $\le 7.9\times10^{-7}$ (about 7 float32 eps). These are consistent with
  float32 reassociation of the $E_{sg}$ kinematics and the per-bin sums. The
  table output is cast from float64, so the escape $\mu$ is the same function
  on both arms. The sibling `coherent-line-hkl-batch` measured
  $2.0\times10^{-5}$.

## Per-hkl loop semantics

The old code called `_mu_total_inv_ang(abs_comp, E_r)` for a groove (coherent
or not) and called it once per layer in `request.layers` order. It kept the
tabulated `mu_i` otherwise. The new `escape_mu_tables` returns the same
compositions in the same order and returns `None` for a single slab, so the
slab keeps `mu_i`. Values agree to $\le 4$ float64 ulp before the existing
cast to `REAL`. The coherent grooved formation path (`_formation_cxr`)
receives the same function.

## Verdict

The extension re-derives cleanly; the table's evaluation semantics match the
host to float64 rounding on all shipped data. A latent fragility exists in
the build probe for ulp-adjacent distinct knots, which does not occur in
EPDL2025.
