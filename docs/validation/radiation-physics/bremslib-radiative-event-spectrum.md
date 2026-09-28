# `bremslib-radiative-event-spectrum`

## Scope and source

Validation: `bremslib-radiative-event-spectrum`. This fresh-context verification used the [BremsLib v2.0.8 description](https://web.vu.lt/ff/a.poskus/files/2025/02/BremsLib_v2.0.pdf), the [BremsLib angular-model derivation](bremslib-angular-model.md), and the [energy-grid convention](../../physics/radiation-physics/energy-grid-semantics.md) before inspecting the scorer. BremsLib's scaled SDCS and DDCS have units mb and mb/sr. The intended output is photons/(eV sr incident electron) for a planar-slab observation direction, with hard photons already sampled by coupled transport.

## Independent derivation

For one hard event of energy $k$ emitted by an electron of pre-event energy $T$, let the physical differential cross sections from the same BremsLib table be $d\sigma/dk$ and $d^2\sigma/(dk\,d\Omega)$. Their ratio is the conditional directional density

$$
p(\Omega\mid T,k)
=\frac{d^2\sigma/(dk\,d\Omega)}{d\sigma/dk},
\qquad
\int_{4\pi}p(\Omega\mid T,k)\,d\Omega=1.
$$

The shared factors $10^{-27}Z^2/k$ cancel, leaving the normalized scaled DDCS divided by its SDCS. The ratio has units sr$^{-1}$. It is evaluated at $\cos\theta=\hat{\mathbf v}\cdot\hat{\mathbf n}$, with the incoming electron direction $\hat{\mathbf v}$ and observation direction $\hat{\mathbf n}$. For a planar slab and Beer–Lambert optical depth $\mu(k)L_{\rm escape}$, an event in photon bin $i$ of width $\Delta k_i$ contributes

$$
\frac{p(\Omega\mid T,k)\,\exp[-\mu(k)L_{\rm escape}]}{N_e\,\Delta k_i}
$$

to that bin's mean density. The event itself was sampled from the hard rate, so multiplying this contribution by a second cross section, atomic density, or flight length would double count the event frequency. When $\mu=0$, the expression reduces to $p/(N_e\Delta k_i)$.

Evaluation nodes $E_i$ imply midpoint bin edges $\epsilon_i=(E_{i-1}+E_i)/2$ internally, with one-sided outer edges. A cutoff $k_c=\epsilon_j>0$ separates soft bins $i<j$ from hard bins $i\ge j$. A hard photon at $k=k_c$ belongs to bin $j$ under the usual left-closed histogram convention; a terminal electron row still contains that photon. Soft track-length scoring uses the same BremsLib cross section at nodes below $k_c$, while the hard array stores event bin means above it. Thus the two arrays have disjoint bin support. Their sum is an approximate hybrid of node-sampled soft density and bin-mean hard density, as the energy-grid convention distinguishes.

## Source-to-code comparison

`brem_events.py::mc_hard_brem_event_spectrum` obtains the pre-event energy from `E_end_keV`, the emission point from the row endpoint, and the observation angle from the electron/observer dot product. It evaluates the SDCS and DDCS at that same $(T,k)$, divides them once, multiplies by escape transmission, histograms the event, and divides by incident-electron count and that bin's width. It accepts a terminal `CUTOFF` row carrying a photon. There is no extra rate, number-density, or path-length factor. `_check_transport_partition` compares the cutoff and table identity with coupled-transport metadata when present; missing or out-of-range tables fail during hard-event scoring.

`mc_soft_brem_spectrum` evaluates BremsLib track-length density and zeros bins on and above the split. `events.py::check_segment_event_contract` checks that nonterminal hard-event energy equals the subsequent electron energy jump and that only hard-radiative or terminal-cutoff rows carry a photon. The synthetic-table test anchors the exact DDCS/SDCS ratio, absence of an extra rate factor, disjoint support, and terminal cutoff scoring. The focused `test_hard_radiative.py` and `test_brem_events.py` run passed together (14 tests, 2026-09-25).

The verifier found that `brem_events.py::_cutoff_edge` accepted an edge within relative tolerance $10^{-12}$ of $k_c$, although the stated contract requires exact equality. The implementation was then changed to require exact equality, with a regression for the next representable value above an edge. The scorer does not model layered or finite-footprint escape or downstream photon transport; no independent full-track comparison of the directional scorer is available yet.

## Full-track cross-check (issue #182)

The Geant4 comparison (`checks/full_track_bremslib/README.md`) scores photons
at creation in W and Si at 300 and 800 keV. The sampled hard-photon energy
yield agrees with Geant4 on matched electron paths within counting error.
For photons of at least 10 keV, the sampled photon polar angle about the
parent electron agrees in mean cosine within 0.039 (at most 2.18 standard
errors). That angle is drawn from the same conditional DDCS this scorer
weights, so the result supports the shared angular law. It does not test the
point-detector weighting, escape factor or detected yield. Independent
full-track validation of those observables remains open.

## Verdict

- **Claim:** `bremslib-radiative-event-spectrum` — `montecarlo/spectrum/brem_events.py::mc_soft_brem_spectrum` and `::mc_hard_brem_event_spectrum`; `montecarlo/transport/events.py::check_segment_event_contract` — BremsLib v2.0.8 SDCS/DDCS relation and node-bin midpoint edges.
- **Filters:** units pass; limits pass; signs and conventions pass for the event weight and bin support; exact cutoff-edge equality passes after the regression fix.
- **Re-derivation:** matches for the scored conditional density and cutoff-bin boundary.
- **Verdict:** `rederived`.
- **Write-up:** `docs/validation/radiation-physics/bremslib-radiative-event-spectrum.md`.
- **Ledger change:** status `rederived`, with the cutoff-edge correction recorded in Notes; human sign-off remains pending.

## 2026-09-27 — issue #172 mid-bin cutoff

### Scope

Issue #172 lets $k_c$ fall inside a photon bin. Production brem grids use nodes on a 25 eV lattice with midpoint edges, so the default $k_c=1000$ eV is a node and falls mid-bin, in $[987.5,1012.5)$ eV. This fresh-context check derived the estimator from the node/bin convention of `energy-grid-semantics` and the partition of `bremslib-radiative-partition` before reading the changed bodies of `brem_events.py::_cutoff_edge`, `::mc_soft_brem_spectrum` and `::mc_hard_brem_event_spectrum`.

### Independent derivation

Let $n(k)$ be the expected escaping emission of the coupled tracks, in photons/(eV sr incident electron). By `bremslib-radiative-partition`, the soft and hard sides come from one BremsLib SDCS, so $n$ has the same functional form below and above $k_c$. Below $k_c$, the track-length estimator $f(k)$ has expectation $n(k)$ at any evaluation energy. Above $k_c$, hard events form a Poisson process on the same tracks, with $n_Z\,d\sigma/dk$ per unit length. A histogram of those events, weighted by $p(\Omega\mid T,k)\,e^{-\tau(k)}$ and divided by $N_e\Delta k_j$, is therefore an unbiased estimate of $\Delta k_j^{-1}\int n\,dk$ over the part of bin $j$ with $k\ge k_c$.

Nodes $E_i$ have midpoint edges $\epsilon_i$, and bin $j$ is $[\epsilon_j,\epsilon_{j+1})$ with width $\Delta k_j$. If $\epsilon_j<k_c<\epsilon_{j+1}$, the bin mean of the total spectrum is

$$
\bar n_j
=\frac{1}{\Delta k_j}\left[\int_{\epsilon_j}^{k_c}n\,dk+\int_{k_c}^{\epsilon_{j+1}}n\,dk\right].
$$

The hard term is estimated exactly in expectation. The soft array holds only the node value $f(E_j)$, so the natural estimator of the first term replaces $n$ with its node value over the soft sub-interval:

$$
\hat n_j
=f(E_j)\,\frac{k_c-\epsilon_j}{\Delta k_j}
+\frac{1}{N_e\Delta k_j}\sum_{\text{events }k\in[k_c,\epsilon_{j+1})}p(\Omega\mid T,k)\,e^{-\tau(k)}.
$$

Bins with $\epsilon_{j+1}\le k_c$ are soft only and keep the node value $f(E_i)$, as in the uncoupled estimator. Bins with $\epsilon_j\ge k_c$ are hard only. If $k_c=\epsilon_j$, the soft share is zero and the previous disjoint-support estimator is recovered exactly. This limit holds for every $j$, including $j=0$, where every bin is hard.

### Width-share error

Expand $n$ about the node. The bias of $\hat n_j$ is

$$
\hat n_j-\bar n_j
=-\frac{1}{\Delta k_j}\int_{\epsilon_j}^{k_c}\left[n(k)-n(E_j)\right]dk
=-\frac{n'(E_j)}{2\Delta k_j}\left[(k_c-E_j)^2-(\epsilon_j-E_j)^2\right]+O(\Delta k^3 n'').
$$

On a uniform grid of step $h$, $\epsilon_j=E_j-h/2$, and the bracket has magnitude at most $h^2/4$. The bracket reaches $h^2/4$ at $k_c=E_j$ and vanishes at either edge. The production default, $k_c$ on a node, is therefore the worst placement:

$$
\left|\frac{\hat n_j-\bar n_j}{\bar n_j}\right|
\le\frac{h}{8}\left|\frac{d\ln n}{dk}\right|+O\!\left(h^2\,\frac{n''}{n}\right),
\qquad
\left|\frac{\text{soft-part error}}{\text{soft part}}\right|\le\frac{h}{4}\left|\frac{d\ln n}{dk}\right|.
$$

For a $1/k$ spectrum, $d\ln n/dk=-1/k$. At $k=1$ keV and $h=25$ eV, the width share underestimates the soft half-bin by $h/(4k)=0.625\,\%$, and the whole straddling bin by $h/(8k)=0.3125\,\%$. The exact $1/k$ integrals give $0.626\,\%$ and $0.315\,\%$. The error is first order in $h$, unlike the $h^2/(24k^2)\approx2.6\times10^{-5}$ relative midpoint error of a whole $1/k$ bin. It affects only one bin, so the integrated yield bias is about $0.003\,n(k_c)h$.

The released BremsLib SDCS was checked at $k=1$ keV for C, Si and W at $T=2$, 5, 30 and 300 keV. Its local slope $d\ln n/d\ln k$ lies between $-0.61$ (W, 2 keV) and $-1.15$ (C, 2 keV). Direct quadrature over the straddling bin gave a relative bias of $-0.19\,\%$ to $-0.36\,\%$ for the bin and $-0.38\,\%$ to $-0.72\,\%$ for the soft half. Each agrees with $(h/8)\lvert d\ln n/dk\rvert$ to within 2 % of the bias.

This bound covers emission. The scored spectrum also contains escape, which adds $-\ell\,d\mu/dk$ to $d\ln n/dk$. Away from absorption edges, $\mu\propto k^{-3}$ gives $+3\tau/k$, so the general bound is $(h/8k)\lvert d\ln n_{\rm emit}/d\ln k+3\tau\rvert$. For optically thick escape, $\tau\gtrsim1$, it reaches about 1 %. An absorption edge in $[\epsilon_j,k_c)$ makes the error first order in the jump, as it is for any unrefined bin.

### Source-to-code comparison

- `_cutoff_edge` returns `split = searchsorted(edges, k_c, side="right") - 1` for $\epsilon_0<k_c<\epsilon_N$. That is the bin with $\epsilon_{\rm split}\le k_c<\epsilon_{\rm split+1}$, and it equals $j$ exactly on an edge. It returns 0 for $k_c\le\epsilon_0$ and $N$ for $k_c\ge\epsilon_N$.
- `mc_soft_brem_spectrum` sets `soft[split:] = 0` and, when `edges[split] < cutoff`, sets `soft[split] = full[split]*(cutoff-edges[split])/widths[split]`, where `full` is the node track-length estimate. This matches $f(E_j)(k_c-\epsilon_j)/\Delta k_j$ for $j\ge1$.
- `mc_hard_brem_event_spectrum` rejects any event with $k<k_c$, histograms on the same edges, and zeroes only `spectrum[:split]`. The straddling bin keeps events with $k\ge k_c$ divided by the full $\Delta k_j$, as derived.

| Case | Derived | Implementation |
| --- | --- | --- |
| $k_c=\epsilon_j$, $1\le j<N$ | soft $i<j$; bin $j$ hard only | `split` $=j$; share 0; hard zeroes $i<j$. Matches |
| $k_c=$ `nextafter`$(\epsilon_j)$ | share $\approx10^{-16}$; same as edge | Matches; anchor test at `atol=1e-12` |
| $k_c$ mid-bin, $j\ge1$ | $f(E_j)(k_c-\epsilon_j)/\Delta k_j$ + hard | Matches; $k_c=E_j$ gives share $1/2$ |
| $k_c\le\epsilon_0$ | all hard; events below $\epsilon_0$ leave the grid | `split` $=0$; soft zeros; hard unzeroed. Matches |
| $k_c\ge\epsilon_N$ | all soft; hard array zero | `split` $=N$; hard fully zeroed, including numpy's right-closed last bin. Matches |
| $\epsilon_0<k_c<\epsilon_1$ | $f(E_0)(k_c-\epsilon_0)/\Delta k_0$ + hard | **`split` $=0$ triggers the early `return zeros`; the soft share of bin 0 is dropped** |

The last row is a discrepancy. The early return in `mc_soft_brem_spectrum` (`if split == 0: return np.zeros(...)`) predates issue #172, when `split == 0` meant only $k_c\le\epsilon_0$. It now also covers a cutoff inside the first bin. The divergent term is the missing $f(E_0)(k_c-\epsilon_0)/\Delta k_0$. A direct call on the grid $[1000,1025,1050]$ eV with $k_c=1000$ eV and stubbed node values $[1,2,3]$ returned soft $[0,0,0]$ instead of $[0.5,0,0]$. The hard side does count the events at or above $k_c$ in bin 0, so that bin loses about half its mass for a node-aligned cutoff.

Installed catalog brem grids do not reach this case. `floored_lattice_start_eV(material, 25)` gives 50 or 75 eV for every catalog material, so $k_c=1$ keV lies far inside the grid. The case is reachable on any grid whose first node lies within half a step above $k_c$. Examples are a brem grid seeded from a line grid that starts at 1 keV (`campaign/sweep.py` and the `runner` fallback both use `E_grid[0]` as the start), a recompute with `brem_start_eV=1000`, and any direct call. The anchor test `test_cutoff_inside_a_bin_splits_it_between_soft_width_and_hard_events` covers only a straddling bin with $j=1$, so it does not catch this. `tests/montecarlo/test_brem_events.py` passes (12 tests, 2026-09-27).

A likely minimal fix is to take the early return only when $k_c\le\epsilon_0$, not whenever `split == 0`. A regression should cover $k_c$ inside the first bin. This verification does not change the code.

### Verdict

- **Claim:** `bremslib-radiative-event-spectrum`: `montecarlo/spectrum/brem_events.py::_cutoff_edge`, `::mc_soft_brem_spectrum`, `::mc_hard_brem_event_spectrum`; midpoint node bins from `energy-grid-semantics`, with the soft/hard partition from `bremslib-radiative-partition`.
- **Filters:** units pass (the width share is dimensionless); limits pass at an edge, just above an edge, below the grid and above the grid; convention fails for a cutoff inside the first bin.
- **Re-derivation:** differs. The implementation omits the soft width share $f(E_0)(k_c-\epsilon_0)/\Delta k_0$ when $\epsilon_0<k_c<\epsilon_1$, because of the `split == 0` early return in `mc_soft_brem_spectrum`. For $j\ge1$ the estimator matches, with a width-share bias of at most $(h/8)\lvert d\ln n/dk\rvert$ in the straddling bin. That is $0.31\,\%$ for $1/k$ at 1 keV and $h=25$ eV, and 0.19–0.36 % for released BremsLib C, Si and W.
- **Verdict:** `discrepancy`.
- **Write-up:** `docs/validation/radiation-physics/bremslib-radiative-event-spectrum.md`.
- **Ledger change:** status `discrepancy`, with the first-bin omission noted; human sign-off remains pending.

### Resolution (2026-09-27)

`mc_soft_brem_spectrum` now returns zeros early only when `cutoff_eV <= edges[0]`, which is the derived all-hard condition. A cutoff inside the first bin now reaches the width-share branch with `split` $=0$. Re-checked directly on the grid $[1000,1025,1050]$ eV with stubbed node values $[1,2,3]$:

| $k_c$ (eV) | Case | Soft result | Derived |
| --- | --- | --- | --- |
| 900 | $k_c<\epsilon_0$ | $[0,0,0]$ | all hard |
| 987.5 | $k_c=\epsilon_0$ | $[0,0,0]$ | all hard; the edge limit holds |
| `nextafter`(987.5) | just above $\epsilon_0$ | $[4.5\times10^{-15},0,0]$ | share $\approx 4\times10^{-15}$ of the bin, consistent |
| 1000 | $\epsilon_0<k_c<\epsilon_1$, on node | $[0.5,0,0]$ | $f(E_0)\cdot\tfrac12$; previously $0$ |
| 1012.5, 1025, 1050 | $j\ge1$ | unchanged | matches |
| 1062.5, 1070 | $k_c\ge\epsilon_N$ | $[1,2,3]$ | all soft |

The hard side is unchanged. With `split` $=0$ it zeroes no bin and keeps the events with $k\ge k_c$ in bin 0, as derived. The anchor test `test_cutoff_inside_a_bin_splits_it_between_soft_width_and_hard_events` now also asserts $k_c=10$ keV on the grid $[10,20,30]$ keV (bin 0 $=[5,15)$ keV), giving $[0.5,0,0]$. `tests/montecarlo/test_brem_events.py` passes (12 tests).

- **Re-derivation:** matches for every cutoff placement. The only approximation is the width-share bias bounded above, at most $(h/8)\lvert d\ln n/dk\rvert$, which is 0.31 % for a $1/k$ spectrum at 1 keV with $h=25$ eV.
- **Verdict:** `rederived`. The first-bin discrepancy recorded above is resolved; human sign-off remains pending.
