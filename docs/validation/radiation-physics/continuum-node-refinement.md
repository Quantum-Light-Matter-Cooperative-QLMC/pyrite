# Continuum-node refinement

Validation: `continuum-node-refinement`.

## Independence and scope

Fresh-context verification on 2026-11-04, scoped to the production default
adopted by issue #278. The derivation below was written from the ledger row,
the [energy-grid semantics](../../physics/radiation-physics/energy-grid-semantics.md)
anchors `continuum-node-refinement` and `eq-grid-quadrature-error`, and the
midpoint-edge rule `eq-grid-midpoint-edges`. Implementation bodies were read
only afterwards. The claim is a node-placement policy. No emission or escape
model is assessed here.

## Independent derivation

### Smooth band: equidistribution on a geometric grid

Let the nodes be $E_i = E_{\rm floor}\,r^i$ with $r = e^{\rho}$, where $\rho$ is
the step in $u = \ln E$. With midpoint bin edges in $E$, node $E$ owns the bin

$$
[a,b] = \left[\tfrac12 E(1+r^{-1}),\ \tfrac12 E(1+r)\right],
\qquad h = b-a = E\sinh\rho \simeq E\rho .
$$

The bin's centre is $E\cosh^2(\rho/2) \simeq E(1+\rho^2/4)$, not $E$. The stored
mass is $h\,n(E)$. Expanding the exact mass about the node gives two terms of
the same order:

$$
\frac{\int_a^b n\,dE}{h\,n(E)} - 1
\simeq \frac{h^2}{24}\frac{n''}{n} + \frac{E\rho^2}{4}\frac{n'}{n}.
$$

The first term is the usual midpoint-rule term (Press et al., *Numerical
Recipes* §4.1). The second term exists because a node on a geometric grid
sits below the centre of its $E$-midpoint bin. For a local power law
$n \propto E^{-p}$, $n''/n = p(p+1)/E^2$ and $n'/n = -p/E$. With $h = E\rho$,

$$
\boxed{\varepsilon_i \simeq \rho^2\left[\frac{p(p+1)}{24} - \frac{p}{4}\right]
= \rho^2\,\frac{p(p-5)}{24}},
$$

which does not depend on $E_i$. A SymPy series of the exact bin integral
reproduces this coefficient for $p = \tfrac12, 2, 3, -1$
($-3/32$, $-1/4$, $-1/4$, $+1/4$). Two conclusions follow.

* **Equidistribution holds.** Measured in $E$, the step is $h_i = E_i\rho$
  and $h_i^2\,n''/n$ is independent of $E$. Measured in $u$, the step is the
  constant $\rho$ and the integrand $g = nE \propto e^{(1-p)u}$ has constant
  $g''/g$. Either reading gives a relative error per bin that is uniform
  across the band.
* **The ledger's $\varepsilon_i \simeq (h_i^2/24)\lvert n''/n\rvert$ is
  incomplete at the same order.** It omits the node-offset term $-p\rho^2/4$.
  Both terms scale as $\rho^2 E^0$, so the omission does not affect
  equidistribution or the order. It does change the coefficient: for
  bremsstrahlung, $p \approx 1$, so the omitted term is three times the stated
  one and opposite in sign.

The smooth-band error is therefore **second order** in $\Delta\ln E$.

### Unresolved features: first order, and how anchors remove it

Suppose $n$ jumps by $J$ at $E^\ast$ inside a bin of width $h$, at offset
$\theta h$ from the bin's lower edge. The midpoint estimate assigns the whole
bin the value on the node's side. The mass error is therefore
$J h\,(\theta \text{ or } 1-\theta)$, which is $O(h) = O(\rho)$ relative to
the bin.

* **Kinematic endpoint.** Put the two nodes at $E^\ast \mp \tfrac12 s$, with
  nothing between them. Their midpoint bin edge is then exactly $E^\ast$, the
  jump coincides with a bin boundary, and the $O(\rho)$ term vanishes. The bin
  below holds a smooth tip, and the bin above holds zero.
* **EPDL medium edge.** EPDL stores each edge as an exact discontinuity in
  $\mu$. The same construction applies: a symmetric pair $E_{\rm edge} \mp
  \tfrac12 s$ places a bin edge on the jump.
* **Chantler Si edge.** The sensor response interpolates $f_2$ between native
  nodes, so the model's jump spans one native interval. Anchoring both native
  nodes reproduces the model's own representation. The residual is set by the
  native spacing and does not depend on $\rho$.

With every hard jump anchored, the source-side quadrature recovers second
order.

### Why detected counts converge at first order

Detected counts are $\int n(E)\,R(E)\,dE$. The Timepix3 response is evaluated
per detector input channel with a seeded Monte Carlo ($n_{\rm mc} = 32$), and
the source bins are rebinned onto those channels piecewise-constant. Within a
source bin, $R$ is therefore a staircase at the channel width, carrying
fixed-seed sampling structure. Its steps are not anchored and cannot be.
Whenever $h_i$ exceeds the channel width, the misallocation per source bin is
$O(h_i)\,\Delta R$. The error envelope is then $O(\rho)$ and not monotone in
$\rho$, because channel and node boundaries alias. The ledger's statement
that detected counts converge at "first order in $\Delta\ln E$" is consistent
with the midpoint analysis above. The lost order belongs to the response, not
to the source grid or its anchors, and anchor nodes cannot restore it. Eagle
XO counts use a smooth digitized QE and converge at about second order (see
the measurements below).

### Limiting case, units, and signs

* With no located edge and no interior endpoint, there are no marks, and the
  merge returns the baseline unchanged.
* All coordinates are in eV. $\rho$ is dimensionless. The endpoint pair
  spacing $s = E^\ast\rho$ is in eV.
* Ordering: $E_{\rm floor} < E^\ast < E_{\rm stop} = E^\ast e^{\rho}$, so the
  endpoint is strictly interior.

## Implementation comparison

| Claim element | Code | Result |
| --- | --- | --- |
| $\rho = 0.0025$, $r = e^{\rho}$ by a fixed 14-term Taylor sum | `src/pyrite/_continuum_grid.py:494`, `:497`–`:513` | matches; truncation $\rho^{15}/15! \approx 10^{-51}$ |
| Nodes $E_{\rm floor} r^k$ by repeated multiplication, closed exactly at $E_0 r$ | `_deterministic_geometric_nodes` `:524`–`:541`; `automatic_continuum_grid` `:568`–`:575` | matches; a final interval below half a step is absorbed |
| Floor is the medium floor | `automatic_continuum_grid` `:568` (`photon_continuum_floor_eV`) | matches |
| Endpoint pair $E^\ast \mp \tfrac12 E^\ast\rho$ | `_refine_baseline` `:423`–`:427`; `continuum_refinement_marks` `:254`–`:271` | matches |
| EPDL edge pair symmetric about the edge, at the shell's Chantler native spacing (else $10^{-3}E$) | `cxr_seeds.py:684`–`:698` | matches |
| Chantler Si pair equals the native bracket | `cxr_seeds.py:655`–`:668`; marks `:227`–`:236` | matches |
| Medium edges from EPDL, detector-path edges from Chantler | `_continuum_grid.py:211`–`:213` | matches |
| Band endpoints never moved | `refine_continuum_nodes` `:304`, `:339` | matches |
| No-feature limit is bit-identical | `refine_continuum_nodes` `:296`–`:297`, `:309`–`:310` | matches |

Numerical checks of the production grid:

* HOPG at 30 keV has 2772 nodes. The bin edge sits exactly on $E_0$, with a
  miss of `0.0` eV, between the adjacent nodes 29962.5 and 30037.5 eV.
* The C K (EPDL) and Si K anchors are adjacent nodes.
* For tungsten at 1 MeV, WSe₂ at 300 keV, and Si at 30 keV, every anchor pair
  is present and adjacent, and every EPDL or endpoint pair has a bin edge on
  its feature.

### Fixed-track measurements (HOPG, $10^4$ Å, 3 electrons, 30 keV)

The reference is the automatic grid at $\rho/32$. Values are relative errors
for yield, centroid, Timepix3 counts, and Eagle XO counts.

| $\rho$ | yield | centroid | Timepix3 | Eagle XO |
| --- | --- | --- | --- | --- |
| 0.000625 | 4.2e-7 | 1.0e-6 | 2.3e-5 | 1.6e-7 |
| 0.00125 | 5.6e-6 | 1.3e-5 | 3.0e-4 | 9.9e-8 |
| 0.0025 | 4.8e-5 | 8.1e-5 | 9.4e-5 | 2.9e-6 |
| 0.005 | 1.6e-4 | 2.8e-4 | 3.3e-4 | 1.1e-5 |
| 0.01 | 6.9e-5 | 8.0e-5 | 7.1e-3 | 2.3e-5 |

Yield, centroid, and Eagle XO converge at about second order in the
asymptotic range. Timepix3 is erratic, as expected for $O(\rho)$ with
aliasing.

## Anchor assessment

`tests/energy-grid/test_continuum_refinement.py` passed: 26 tests in 8.4 s.

`test_automatic_policy_matches_a_resolved_reference_on_fixed_tracks` compares
a grid at $\rho$ with the automatic grid at $\rho/4$, with a pass limit of
$10^{-3}$ for every observable. The table gives the worst observable at each
$\rho$:

| $\rho$ | automatic grid | plain baseline, no anchors |
| --- | --- | --- |
| 0.0025 | 8.2e-5 (passes) | 9.9e-5 (passes) |
| 0.005 | 2.9e-4 (passes) | 3.7e-4 (passes) |
| 0.0075 | 2.4e-3 (fails) | 2.3e-3 (fails) |
| 0.01 | 7.0e-3 (fails) | 7.0e-3 (fails) |

The test is a genuine reference-value regression on the step. It fails once
the baseline is coarsened by a factor of about 3 or more, it has roughly 12×
margin at the adopted step, and Timepix3 is the binding observable. It does
**not** detect dropped anchors in this case, because the plain baseline also
passes. Anchor placement is pinned structurally by the separate edge,
endpoint-bin-edge, and limiting-case tests. The two digest pins fix the node
bits: HOPG has 2772 nodes and tungsten 2406 at 30 keV.

## Findings

1. The ledger's error formula omits the node-offset term at the same order
   (see above). This is not a discrepancy in the claimed policy.
2. The ledger Notes say "node count unchanged for HOPG 300 keV, 2772". The
   2772-node grid is HOPG at **30 keV**, which the digest test pins. HOPG at
   300 keV has 3693 nodes.
3. The ledger `Source` field and the semantics page's "Absorption edges" bullet
   still say the escape $\mu$ edges come from Chantler $f_2$. Since #274 the
   medium edges come from EPDL. The Claim and Notes fields are already correct.
4. Some marks use a Chantler native spacing coarser than the local baseline
   step, up to 1.37× for W and WSe₂. The baseline nodes they displace are then
   replaced by a locally coarser lattice. This is consistent with "refine no
   further than the model", but the docstring says refinement "only adds
   nodes".
5. Two corner cases are not guarded. First, the cross-mark merge
   (`_continuum_grid.py:321`–`:329`) can drop an anchor of a later mark
   closer than $\tfrac14$ of the finer spacing. Second, a baseline node lying
   exactly at $E^\ast$ survives the strict-`<` displacement test at `:336`.
   Neither occurred in the probed media.

## Verdict

`rederived`. The policy, anchors, limiting case, and deterministic construction
match the derivation. The ledger text needs the documentation corrections in
findings 1–3.

## Addendum: cross-mark anchor merge (#1171, 2026-11-13)

Fresh-context verification of commit `c4fcd8347`
(`_continuum_grid.py::refine_continuum_nodes`, `_join_overlapping_pairs`,
`_split_mark_nodes`, `_strictly_inside_pairs`; `RefinementMark.centred`). The
derivation below was written from the ledger row and the semantics page before
the new function bodies were read.

### What an anchor must deliver

Bin edges are node midpoints ({eq}`eq-grid-midpoint-edges` on the semantics
page). The first-order term of the previous section vanishes for a jump at
$E$ exactly when some adjacent pair $x_k<x_{k+1}$ satisfies

$$
x_k+x_{k+1}=2E .
$$

For an EPDL edge and the kinematic endpoint the jump is exact at $E$, so this
condition is the whole requirement; the coordinates $E\mp\tfrac12 s$ are one
solution of it, not the requirement itself. For a Chantler edge the model's
jump is spread over one native interval $[a,b]$, so the requirement is that
$a$ and $b$ are adjacent nodes.

### Two overlapping centred pairs

Let $E_1<E_2$ with $E_2-E_1=d$, and let the pairs overlap,
$E_2-\tfrac12 s_2<E_1+\tfrac12 s_1$. Two bin edges need at least three nodes
$x_0<x_1<x_2$ with $x_0+x_1=2E_1$ and $x_1+x_2=2E_2$. Then

$$
x_1-x_0=2(x_1-E_1),\qquad x_2-x_1=2(E_2-x_1),
$$

so monotonicity requires $E_1<x_1<E_2$, and the smaller of the two intervals
is largest at $x_1=\tfrac12(E_1+E_2)$, where both equal $d$. The node $x_1$
then owns exactly the bin $[E_1,E_2]$, on which $\mu$ is smooth. A third
overlapping edge extends the chain by $x_{k+1}=2E_k-x_k$, which is valid while
$x_k<E_k$.

Comparison with the alternatives:

* Keeping one pair and dropping the other leaves a jump $J$ inside a bin at
  some offset, an error $O(J\,s)$, first order.
* Keeping both pairs as coordinates generally puts neither feature on a bin
  edge, so both edges keep first-order errors.
* The chain removes the first-order term for every feature. Its only cost is
  that the outer nodes sit off their bin centres. For smooth $n$, a node
  offset $\delta$ from its bin centre gives an error of $n'\delta w$ on a bin
  of width $w$, which is $O(h^2 n')$ on two bins, i.e. second order.

Moving an EPDL pair off $E\mp\tfrac12 s$ therefore does not weaken the
quadrature argument of `eq-grid-quadrature-error`. That argument needs a bin
edge on the jump, not the native spacing $s$. Nor does the chain refine below
the model's resolution. Its intervals are $d$, the separation the two tables
themselves place between the edges.

### Comparison with the implementation

`_join_overlapping_pairs` forms $x_1=\tfrac12(E_1+E_2)$,
$x_0=2E_1-x_1$, $x_2=2E_2-x_1$, and extends the chain by
`following = 2*feature - candidate[-1]`. This matches the derivation.
Samples and baseline nodes strictly inside any pair are removed
(`_strictly_inside_pairs`), and samples merge against the neighbouring anchors
and the last kept sample with tolerance $\tfrac14\min(s)$. Anchors are never
merged away. Chantler pairs are never re-anchored. A Chantler pair interleaved
with another pair is reported in `split_anchor_pairs` and is not resolved.

**Finding (tolerance factor 2).** The docstring, the ledger note and the brief
all state that the chain continues while every interval exceeds
$\tau=\tfrac14\min(s)$. The extension test is
`candidate[-1] < feature - tolerance`. That bounds the half-interval
$E_k-x_k$, so it requires intervals $>2\tau$. For $\tau<d\le 2\tau$ the
features are neither treated as one feature nor chained. Both pairs are then
kept as coordinates, and both features lie off a bin edge by $d/2$. A
synthetic probe with $s=1$ and $E_1=1000$ eV confirms this:

| $d$ (eV) | bin-edge offsets $E_1$, $E_2$ (eV) | outcome |
|---|---|---|
| 0.1–0.25 | 0, $d$ | one feature (shared pair) |
| 0.30 | 0.15, 0.15 | split, reported |
| 0.45 | 0.225, 0.225 | split, reported |
| 0.50 | 0.25, 0.25 | split; $E_1$ falls on a node |
| 0.55–0.99 | 0, 0 | chained |

The chain at $d=0.3$ would have intervals of $0.3>\tau$. No catalog case falls
in this window (scan below). Either the test should be `> tolerance / 2` on the
half-interval, or the documented rule should say $2\tau$. In the shared-feature
regime ($d\le\tau$), the later feature sits off its bin edge by up to $\tau$.
Centring the shared pair on $\tfrac12(E_1+E_2)$ would halve that.

### Independent scan

An independent script checked every catalog material, both bare and as shipped,
plus W, PtTe$_2$ and WS$_2$ on Au, MoS$_2$ on Si, HOPG and W, each at
$E_0\in\{10,30,100,300,1000\}$ keV (280 grids). It recomputed the marks with
`continuum_refinement_marks` and required:

* for each centred mark, a node midpoint within $10^{-9}E$ of the feature;
* for each Chantler mark, its two anchors as adjacent nodes;
* fixed band endpoints, strictly increasing nodes, and
  `validate_backend_coordinates(float32)`.

| | `main` (`bc35cdae1`) | branch (`c4fcd8347`) |
|---|---|---|
| feature violations | 170 in 85 grids (S $L_2$ in 15 film/stack combinations, Ta $M_2$ in TaS$_2$, W $M_3$ on Au, Pt $M_4$ on Au; worst offset $0.47$ of the local interval) | 0 |
| endpoint, monotonicity, FP32 failures | 0 | 0 |
| grids with a smaller minimum interval than `main` | — | 0 |
| node-count change | — | $-1$ to $-4$ in 80 grids |
| smallest interval | 0.0295 eV at 46.19 eV (ZrTe$_5$) | same (pre-existing) |

HOPG and W at 30 keV and 1 MeV are bit-identical between the two (SHA-256
prefixes `e1691b83…`, `9a73475f…`, `c8ee638b…`, `92254ebc…`; 2772 and 2406
nodes at 30 keV).

Local checks:

* PtTe$_2$ on Au gives nodes 2197.58, **2204, 2208, 2212**, 2214.42, with bin
  edges at exactly 2206 and 2210 eV.
* In MoS$_2$ the two EPDL pairs at 168 and 170 eV do not overlap. They were
  fixed by sample rejection alone, with bin edges at 168.0 and 170.0.
* In TaS$_2$ the S K sample at 2473.60 eV sits 0.36 eV from the Ta $M_2$
  anchor at 2473.24, above the 0.24 eV tolerance.
* On MoS$_2$/Si, the Chantler Si K pair (1837.98, 1838.82) and the EPDL Si K
  pair (1843.54, 1844.46) both survive.

Both EPDL S edges carry the label `S L2 (EPDL)`. The labels are cosmetic; the
anchors are distinct.

### Verdict (#1171)

`rederived` for the claimed behaviour on every catalog and named stack
configuration. One finding remains: the chain tolerance is $2\tau$, where the
documentation states $\tau$. It is reachable only in user-built media, and it
is reported through `split_anchor_pairs`, not hidden. The ledger note's
"smallest interval 0.36 eV" holds only for the overlap regions. The global
minimum is 0.0295 eV at 46.19 eV in ZrTe$_5$, unchanged from `main`.

**Resolution (implementation context, same PR).** The extension test now
bounds the half-interval by $\tau/2$, so the chain continues while every
interval exceeds $\tau$, as documented. The $d=0.3$ probe is pinned by
`test_close_centred_pairs_chain_down_to_the_merge_tolerance`. The ledger
note now qualifies the 0.36 eV figure as the overlap-region minimum. The
shared-feature offset ($d\le\tau$) is left as is.
