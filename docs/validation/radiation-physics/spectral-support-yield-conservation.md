# spectral-support-yield-conservation

Fresh-context verification for issue #954 (branch
`issue-954-spectral-support-integration`, commit `6244846cb`). The derivation
below was written from the ledger row, the
[energy-grid semantics](../../physics/radiation-physics/energy-grid-semantics.md)
contract (`eq-grid-midpoint-edges`, `eq-grid-linear-cumulative`,
`eq-grid-support-split`) and the public docstrings and signatures only; the
implementation bodies were read afterwards.

## Claim and inputs

The ledger claim is a quadrature and bookkeeping contract, not an emission
model. A stored component with density $f$ in photons / incident electron / eV
/ sr represents yield only over its declared support: $[E_0, E_{N-1}]$ for
node samples and $[\epsilon_0, \epsilon_N]$ for bin means. Window yields, cross-family
sums, node-to-bin integration, bin-mean overlap rebinning and unit conversion
must conserve that represented yield. Target-node interpolation makes no conservation promise.

Signatures under review: `node_yield(energy_eV, density, window_eV=None)`,
`bin_mean_yield(edges_eV, means, window_eV=None)`, `combine_yields(parts)`,
`nodes_to_bins(energy_eV, density, edges_eV, *, boundary)`,
`rebin_bin_means(edges_eV, means, target_edges_eV, *, boundary)`, and the
`SpectralDensity.integrate`, `SpectralDensity.rebin` and
`Result.integrated_yield` dispatchers.

## Independent derivation

### Piecewise-linear interpolant

Let nodes $E_0 < \dots < E_{N-1}$ ($N \ge 2$) carry samples $f_k$, with
$h_k = E_{k+1} - E_k > 0$. The linear interpolant on $[E_k, E_{k+1}]$ is

$$
\tilde f(x) = f_k + \frac{f_{k+1} - f_k}{h_k}\,(x - E_k).
$$

With $t = x - E_k \in [0, h_k]$,

$$
\int_{E_k}^{x}\tilde f\,dx' = f_k\,t + \frac{f_{k+1} - f_k}{2h_k}\,t^2,
\qquad
\int_{E_k}^{E_{k+1}}\tilde f\,dx' = \tfrac12 h_k\,(f_k + f_{k+1}).
$$

Summing completed intervals gives the cumulative yield from $E_0$:

$$
\boxed{
C(x) = \sum_{j<k}\tfrac12 h_j (f_j + f_{j+1})
+ f_k (x - E_k) + \frac{f_{k+1} - f_k}{2h_k}(x - E_k)^2,
\quad x\in[E_k, E_{k+1}]
}
$$

This is `eq-grid-linear-cumulative` term for term. $C$ is continuous at every
node, because at $x = E_{k+1}$ both adjacent branches give the same partial
sum. Evaluating at a node with either neighbouring $k$ is therefore
equivalent. $C(E_{N-1})$ is the full trapezoid. For a density linear in
$E$, $\tilde f \equiv f$ and $C$ is exact for any node placement.

### Piecewise-constant bin means

With edges $\epsilon_0 < \dots < \epsilon_M$ and means $\bar S_j$ ($M \ge 1$),

$$
C_{\rm b}(x) = \sum_{j<k}\bar S_j(\epsilon_{j+1} - \epsilon_j) + \bar S_k(x - \epsilon_k),
\quad x \in [\epsilon_k, \epsilon_{k+1}],
$$

which is exact if density is uniform within each bin. A window
inside one bin takes its overlap fraction of that bin's mass.

### Window split

Write the support as $[s_0, s_1]$ and the window as $[a, b]$ with $a \le b$.
Clip both window ends into the support, $a' = \min(\max(a, s_0), s_1)$ and
likewise $b'$. Then $s_0 \le a' \le b' \le s_1$, and

$$
N_{\rm in} = C(b') - C(a'),\qquad
N_{\rm below} = C(a') - C(s_0) = C(a'),\qquad
N_{\rm above} = C(s_1) - C(b').
$$

The sum telescopes to $C(s_1)$ for every window, which is
`eq-grid-support-split`. The part of the window outside the support is

$$
U = \bigl([a, \min(b, s_0)]\bigr) \cup \bigl([\max(a, s_1), b]\bigr),
$$

with empty or zero-length parts dropped. It is *uncovered*: the stored data
say nothing about emission there, so it is reported and never treated as zero
yield. Limits:

| Case | $N_{\rm in}$ | $N_{\rm below}$ | $N_{\rm above}$ | $U$ |
|---|---|---|---|---|
| $[a,b] = [s_0,s_1]$ | $C(s_1)$ | $0$ | $0$ | $\varnothing$ |
| $b < s_0$ | $0$ | $0$ | $C(s_1)$ | $[a,b]$ |
| $a > s_1$ | $0$ | $C(s_1)$ | $0$ | $[a,b]$ |
| $a < s_0 < s_1 < b$ | $C(s_1)$ | $0$ | $0$ | $[a,s_0]\cup[s_1,b]$ |

A window disjoint from the support returns zero inside mass and the full
integral as cropped mass, in agreement with the contract's limiting case. Its
side (below or above) follows from which side of the window the support lies.

### Combining families

Each family $c$ is integrated on its own support with its own declared
quadrature. Since integration is linear,
$N_{\rm in} = \sum_c N_{{\rm in},c}$, and likewise for below and above. No
family is interpolated onto another's grid. The combined support is
$\bigcup_c [s_{0,c}, s_{1,c}]$; the combined uncovered set is
$[a,b] \setminus \bigcup_c[s_{0,c}, s_{1,c}]$, the window part that *no*
family supports. A window part that only some families support is not
uncovered in the combination, but it is in those families' own $U_c$. The
total is then a lower-information statement: complete only in the sense of
union coverage.

### Node-to-bin and overlap rebinning

For target edges $e_0 < \dots < e_M$, set $e_i' = \min(\max(e_i, s_0), s_1)$:

$$
M_i = C(e_{i+1}') - C(e_i'),\qquad
N_{\rm below} = C(e_0'),\qquad
N_{\rm above} = C(s_1) - C(e_M').
$$

Then $\sum_i M_i + N_{\rm below} + N_{\rm above} = C(s_1)$ by telescoping, for
node ($C$) and bin-mean ($C_{\rm b}$) sources alike. The covered width of
target bin $i$ is $w_i = e'_{i+1} - e'_i$. A mean density $M_i/(e_{i+1} - e_i)$
is meaningful only when $w_i = e_{i+1} - e_i$; on a partly covered bin it
would silently dilute the mean with an unsupported region, so the bin is
undefined (`NaN`). With `boundary="raise"`, $e_0 \ge s_0$ and $e_M \le s_1$ are
required, which makes every $w_i$ full.

### Unit conversion

For $E' = \lambda E$ (for eV to keV, $\lambda = 10^{-3}$), a density per unit energy transforms as
$f' = f/\lambda$, and edges transform as $\epsilon' = \lambda\epsilon$. Then
$\int f'\,dE' = \int (f/\lambda)\,\lambda\,dE$ is invariant. The trapezoid and
bin-mean sums are bilinear in (width, density), so each discrete integral is
invariant as well. No resampling is involved.

### Native edges of bin-mean producers

From `eq-grid-midpoint-edges` with the 0 eV physical floor:

$$
\epsilon_0 = \max\!\left(0,\; E_0 - \tfrac12(E_1 - E_0)\right),\quad
\epsilon_i = \tfrac12(E_{i-1} + E_i),\quad
\epsilon_N = E_{N-1} + \tfrac12(E_{N-1} - E_{N-2}).
$$

### Fixture: native sum versus legacy union table

For CXR nodes $(100, 300)\,\mathrm{eV} \mapsto (2, 4)$, characteristic
$(100, 300) \mapsto (0.5, 0.5)$ *declared as node samples*, and brem
$(100, 200, 400) \mapsto (1, 1, 1)$, the native integrals are
$200\cdot 3 = 600$, $200 \cdot 0.5 = 100$ and $300 \cdot 1 = 300$, which sum to $1000$.
On the union nodes $\{100, 200, 300, 400\}$ with zero fill outside each
support, the total column is $(3.5, 4.5, 5.5, 1)$, with trapezoid
$100(4 + 5 + 3.25) = 1225$. The excess consists of the exterior triangles
$\tfrac12\cdot 100\cdot 4 = 200$ (CXR) and $\tfrac12\cdot 100 \cdot 0.5 = 25$
(characteristic) on $[300, 400]$.

If the characteristic component is instead declared as the producer does,
bin-mean on clamped edges $(0, 200, 400)$, its native integral is
$0.5 \cdot 400 = 200$ and the native total is $1100$. The legacy table still
gives $1225$, because it reads bin means as node samples. The contract text's
"100" for the characteristic component holds only under the node declaration
the test fixture uses.

## Implementation comparison

`results/spectral_support.py::_node_cumulative` evaluates

```text
cumulative = concat([0], cumsum(0.5 * widths * (density[:-1] + density[1:])))
x = clip(points, energy[0], energy[-1])
k = clip(searchsorted(energy, x, side="right") - 1, 0, energy.size - 2)
dx = x - energy[k]
return cumulative[k] + density[k] * dx + 0.5 * slope * dx * dx   # slope = Δf/h_k
```

This is $C(x)$ term for term with $x$ already clipped into the support. At
$x = E_{N-1}$, `searchsorted` gives $k = N-1$, which is clamped to $N-2$ with $dx = h_{N-2}$. That
is the full last trapezoid, consistent with the continuity argument above.
`_bin_cumulative` is $C_{\rm b}$ in the same form, with $k$ clamped to $M-1$.

`_yield` clips the window into the support, evaluates the cumulative at
$(a', b', s_1)$, and returns `inside`, `below`, `above` exactly as
$N_{\rm in}, N_{\rm below}, N_{\rm above}$. `_window` builds
$U = \{(a, \min(b, s_0)) \text{ if } a < s_0\} \cup \{(\max(a, s_1), b) \text{ if } b > s_1\}$
and drops zero-length parts, matching $U$. It refuses `NaN` or reversed
windows.

`combine_yields` sums `inside`, `below` and `above`. It requires one shared
window, merges supports with `_merge` (touching intervals coalesce), and
computes the uncovered set as `_subtract(window, union)`. This is the
union semantics derived above. `Result.integrated_yield` calls
`part.integrate(window)` per family and then `combine_yields`.
`SpectralDensity.integrate` dispatches on the *declared* `sampling`: `bin-mean`
goes to `bin_mean_yield(bin_edges_eV, …)`, `nodes` to `node_yield`, and `mixed`
refuses.

`_binned` (node-to-bin) returns `mass = diff(C(edges))`,
`below = C(edges[0])`, `above = C(s_1) - C(edges[-1])` and
`supported_width_eV = diff(clip(edges, s_0, s_1))`. Because the cumulative
clips internally, this equals $M_i$, $N_{\rm below}$, $N_{\rm above}$ and $w_i$.
`rebin_bin_means` delegates to
`_grid_semantics.py::rebin_piecewise_constant_density`. That function's `integral_to` is
$C_{\rm b}$ on clipped points (with explicit $0$ / total guards at the ends),
`below = C_b(e_0)` and `above = C_b(s_1) - C_b(e_M)`, so the same telescoping holds.
`BinnedSpectrum.mean_density` returns `mass / width` where
`supported_width_eV == widths`, else `NaN`, as derived.

`_target_edges` refuses edges beyond the support for `boundary="raise"` and
any other policy string. `_finite_array` and `_ordered` refuse masked, `NaN`,
infinite, non-1-D, duplicate or reordered inputs and fewer than two
coordinates, as the ledger's assumptions require.

`QuantifiedSpectrum.to_energy` stores eV values and converts both coordinate
(`eV` to `unit`) and density (`1/eV/sr` to `1/unit/sr`) through Pint on access.
Edges convert the same way. This is the $\lambda$, $1/\lambda$ pair derived above.

Producer declaration: `api.py::_line_sampling` builds edges with
`characteristic.py::_energy_bin_edges_and_widths`. That function implements the clamped
midpoint edges above exactly. The function assigns them to characteristic always
and to incoherent CXR only when `case["line_quadrature"] == "bin-mean"`.
`_bin_quadrature.py::bin_axis` calls the same edge function for the incoherent
bin-mean route. The characteristic kernel writes `weights / bin_widths`
(per-bin captured fraction over the same widths), which is a bin mean. Both
`_scalar_result` and the planar `Result` builder pass the same `E_grid` to
`_line_sampling` that the kernels used. Producer and declaration agree.

Result: **matches**, symbolically and dimensionally. Masses carry
photons / incident electron / sr (density per eV times eV).

## Numerical spot check

The script was run from the worktree with `uv run python -I`, against the worktree
`src/pyrite/results/spectral_support.py`, over 300 random trials. Each trial used
2 to 29 nonuniform nodes on $[0, 10^4]$ eV, signed Gaussian densities, random
windows reaching beyond both support ends, and 1 to 19 random target bins
extending beyond the support (`boundary="clip"`). References are
`scipy.integrate.quad` (relative tolerance $10^{-13}$) of `np.interp` or step
functions, split at the breakpoints. They are independent of the cumulative-sum
code.

| Check | Max relative error |
|---|---|
| node window: inside, below, above | $8.5\times10^{-16}$ |
| node split conservation $N_{\rm in}+N_{\rm below}+N_{\rm above}$ vs. full integral | $1.0\times10^{-15}$ |
| bin-mean window and total | $5.3\times10^{-16}$ |
| node-to-bin masses | $9.8\times10^{-16}$ |
| node-to-bin conservation $\sum M_i + N_{\rm below} + N_{\rm above}$ | $1.0\times10^{-15}$ |
| overlap-rebin masses | $3.6\times10^{-16}$ |
| overlap-rebin conservation | $4.6\times10^{-16}$ |
| linear density $c + sE$, nonuniform nodes, closed form | $3.7\times10^{-16}$ |

In every trial, `uncovered_eV` equalled the independently built $U$,
`supported_width_eV` equalled $\operatorname{diff}(\operatorname{clip}(e))$
exactly, and `mean_density` was `NaN` exactly on the partly covered bins.

Limiting cases on $E = (100, 150, 400, 1000)$: window = support gives
`inside == total`, $0$, $0$ and no uncovered parts. Window $(2000, 3000)$
gives inside $0$, `below == total` and uncovered $(2000, 3000)$. Window
$(10, 50)$ gives inside $0$, `above == total` and uncovered $(10, 50)$.

`combine_yields` with supports $[100,200]$ and $[300,400]$ and window
$[50,500]$ returns inside $300$, support
$((100,200),(300,400))$ and uncovered $((50,100),(200,300),(400,500))$. The
first part's own uncovered set is $((50,100),(200,500))$.

Fixture: with characteristic declared as nodes, the families give
$600, 100, 300$ and total $1000$, and the legacy `spectral_flux` trapezoid
gives $1225$. An independent `np.interp(left=0, right=0)` union sum gives
$1225$ with column $(3.5, 4.5, 5.5, 1)$. With characteristic declared
bin-mean on $(0, 200, 400)$, the total is $1100$ and the legacy table still
gives $1225$.

Unit conversion: the node spectrum $(200, 400, 1000) \mapsto (1, 3, 2)$ gives
$1900$ in eV and $1900$ after `to_energy("keV")` with a keV trapezoid. The bin-mean
spectrum on edges $(100, 300, 700, 1300)$ gives $2600$ in both units.

Edges: for grids $(10, 40, 50)$ and $(1000, 1010, 1500, 4000)$,
`_energy_bin_edges_and_widths`, `bin_axis` and `_line_sampling` all equal the
independently built clamped midpoint edges $(0, 25, 45, 55)$ and
$(995, 1005, 1255, 2750, 5250)$ bit for bit. `_line_sampling` declares CXR
`bin-mean` only with `line_quadrature="bin-mean"`.

The float-equality test in `mean_density` gives 0 spurious `NaN` for 24 interior
bins of a non-dyadic grid ($0.1$ to $0.7$ eV). Interior bins compare
`diff(clip(e))` with `diff(e)` on identical unclipped values, so equality is
exact rather than tolerance-dependent.

`uv run pyrite-dev test tests/results/test_spectral_support.py`: 20 passed.

## Edge cases and observations

1. **Contract fixture wording.** The energy-grid page's legacy-table bullet gives
   the characteristic native integral as 100 for $(100, 300) \mapsto (0.5, 0.5)$.
   That holds only if the component is declared as node samples, which is what
   the test fixture does. A producer-built characteristic is bin-mean on
   $(0, 200, 400)$ and integrates to 200, for a native total of 1100. This is a
   documentation precision point, not an implementation error.
2. **Union-only uncovered semantics.** `combine_yields(...).complete` is `True`
   whenever *some* family covers each part of the window, even if a family
   lacks support there. Per-family `uncovered_eV` carries that information, as
   documented. Callers that need every family covered must inspect the parts.
3. **Partial-bin `NaN` mean.** This is correct by derivation. `mass` and
   `supported_width_eV` remain finite, so no mass is hidden.
4. **Window = `None` combination.** Every part has zero below and above mass, and
   the combined support may contain gaps that are not reported as uncovered,
   because no window was requested. This is consistent with the contract.
5. **Signed densities** are accepted. The derivation is linear and does not
   need positivity.
6. `interpolate_nodes` and `SpectralDensity.interpolate` make no conservation
   claim. They carry no `Validation:` marker and are not in the ledger `Code` field,
   which is consistent.

## Adjudication

Re-derivation **matches** the implementation for every equation in the
claim: `eq-grid-linear-cumulative`, the piecewise-constant cumulative,
`eq-grid-support-split` with uncovered reporting, native-sum combination,
node-to-bin and overlap-rebin conservation, unit-conversion invariance, and
the producer sampling declarations with `eq-grid-midpoint-edges` (0 eV clamp).
Suggested status: `rederived`. Human sign-off is pending (#277).
