# Timepix source-cell bookkeeping

Validation: `detector-timepix`.

This independent verification covers only source-cell integration, response-input
channel aggregation, and native measured event-mass application. It does not
validate silicon absorption, diffusion, charge sharing, thresholds, Monte Carlo
calibration, or placeholder hardware parameters. The full ledger claim remains
`unverified`.

## Source, signature, and assumptions

The mathematical source is the exact integral of a piecewise-constant histogram
density over an interval intersection. The repository convention for converting
nodes to cells is [energy-grid semantics](../../physics/radiation-physics/energy-grid-semantics.md).
The reviewed signature is `TimepixResponse.apply_native(spec)`, with finite,
nonnegative density in events/eV on the last axis and arbitrary leading batch
dimensions. Source nodes are finite, strictly increasing, nonnegative, and number
at least two. Response-input channels cover every source cell. The fixed response
matrix maps input event mass to native measured event mass.

This derivation was written before inspecting implementation bodies. Only the
ledger record, class and method docstrings, intended signature, and owning
energy-grid semantics were consulted first.

## Independent derivation

For nodes $E_0,\ldots,E_{N-1}$, define source edges by

$$
s_0=\max\left(0,E_0-\frac{E_1-E_0}{2}\right),\qquad
s_i=\frac{E_{i-1}+E_i}{2}\quad(1\le i<N),\qquad
s_N=E_{N-1}+\frac{E_{N-1}-E_{N-2}}{2}.
$$

With density sample $f_i$, the explicit histogram reconstruction is

$$
f(E)=\sum_{i=0}^{N-1}f_i\mathbf{1}_{[s_i,s_{i+1})}(E).
$$

It is zero outside the source cells. In particular, a positive lower source edge
$s_0$ leaves the interval $[0,s_0)$ empty; widening the first cell to zero would
invent the event mass $f_0s_0$. The zero clamp removes negative energy support
when the reflected edge is negative.

Let input-channel edges be $a_j$. Integrating the reconstruction gives

$$
L_{ji}=\max\left(0,\min(s_{i+1},a_{j+1})-\max(s_i,a_j)\right),\qquad
\boxed{n_j=\sum_iL_{ji}f_i}.
$$

The overlap $L_{ji}$ has units eV; $n_j$ has units events. No division by source
or target width is appropriate for this mass result. Since input channels
partition and cover the source cells,

$$
\sum_jL_{ji}=s_{i+1}-s_i,\qquad
\sum_jn_j=\sum_if_i(s_{i+1}-s_i).
$$

This is conservation of the chosen histogram reconstruction, not an exact
integral of an unresolved underlying spectral function or a trapezoidal node
integral.

For dimensionless response probability $R_{kj}$, measured native mass is

$$
\boxed{m_k=\sum_jR_{kj}n_j}.
$$

Nonnegative $R_{kj}$ and column sums $q_j=\sum_kR_{kj}\le1$ give

$$
m_k\ge0,\qquad \sum_km_k=\sum_jq_jn_j\le\sum_jn_j.
$$

Equality requires unit column sums on occupied channels. Detector losses are
consistent with a conservative input rebin; they are not lost rebin mass. Zero
density gives zero input and measured mass. For any leading batch index $b$,

$$
n_{bj}=\sum_iL_{ji}f_{bi},\qquad m_{bk}=\sum_jR_{kj}n_{bj}.
$$

Thus batch shape is preserved and each spectrum is processed independently.
For coincident source and input edges the overlap is diagonal with cell widths,
recovering direct density-to-mass conversion.

## Numerical reference and refinement limits

For source nodes $(10,20,40)$ eV, edges are $(5,15,30,50)$ eV. Densities
$(2,1,3)$ events/eV and target edges $(0,20,40,60)$ eV give

$$
n=(2\cdot10+1\cdot5,\;1\cdot10+3\cdot10,\;3\cdot10)
=(25,40,30)\ \mathrm{events},\qquad \sum_jn_j=95\ \mathrm{events}.
$$

The target's interval below 5 eV contributes exactly zero.

Absolute input-channel lattice locations can remain fixed under source-node
refinement, while coverage may add channels at either end. Identical rebinned
masses follow when refinement represents the same piecewise-constant density
on the same support, for example exact subdivision of constant cells. Arbitrary
node refinement changes midpoint edges, reconstruction, and possibly support;
its masses need not be identical. A coarse response also approximates the
energy dependence within each input channel. This bookkeeping proof does not
establish detector-resolution accuracy or invariance of separately regenerated
Monte Carlo matrices.

The legacy `apply` projection converts native mass to a density at output-bin
centres and interpolates it onto source nodes. Linear interpolation, finite
source support, and endpoint conventions need not preserve the native mass
integral. Claims of exact total conservation belong to native mass aggregation,
not this sampled-density projection.

## Filters and implementation comparison

Units, zero-input and coincident-edge limits, nonnegativity, matrix orientation,
and positive-floor conventions pass for the independent expression above.
After that derivation, implementation bodies were inspected:

- `zero_based_detector_edges` uses midpoint cells and either clamps the reflected
  negative edge or prepends a distinct zero-support cell.
- `TimepixResponse.__init__` extends the absolute input-channel lattice when the
  source outer cells exceed its ordinary padded coverage.
- `_input_channel_masses` pads the density with zero when needed and calls
  `rebin_piecewise_constant_density`. That helper constructs a piecewise-linear
  cumulative integral and differences it at target edges. This is algebraically
  the overlap expression above, with saturation outside source support.
- `apply_native` applies that integral to each flattened batch row, multiplies
  by the transpose of `R`, and restores the leading shape. `apply` shares the
  same input integral before its output-density interpolation.

Independent numeric execution used a manually constructed response object,
without Monte Carlo generation or implementation helpers for the expected
values. The positive-floor example above returned exactly $(25,40,30)$ events.
The non-diagonal response

$$
R=\begin{pmatrix}0.5&0.1&0.2\\0.25&0.7&0.3\end{pmatrix}
$$

returned $(22.5,43.25)$ events to absolute tolerance $10^{-13}$. Six scaled
spectra with leading shape $(2,3)$ agreed to absolute tolerance $10^{-12}$;
zero input returned exact zero. A separate negative-reflection case with nodes
$(0,10,30)$ eV, clipped edges $(0,5,20,40)$ eV, and the same density and target
channels returned exactly $(25,60,0)$ events. These references exercise both
zero-boundary branches and response orientation independently.

Verdict: **rederived for source-cell and native-channel bookkeeping only**.
No divergent factor, sign, width, or channel convention was found. Floating-point
cumulative subtraction can lose relative precision in tiny masses following
very large accumulated masses. This floating-point limitation does not change
conservation in exact arithmetic; hardware accuracy remains outside this
verification.

Suggested ledger treatment: retain `unverified` for the full detector claim;
record this independent bookkeeping derivation in its Notes. No hardware
certification or human sign-off is implied.
