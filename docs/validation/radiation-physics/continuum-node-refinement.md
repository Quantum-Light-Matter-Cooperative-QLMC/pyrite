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
