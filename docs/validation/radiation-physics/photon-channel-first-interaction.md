# All-channel first photon interaction through layered prisms

`Validation: photon-channel-first-interaction` — independent verification of
`montecarlo/transport/photon_navigation.py::ray_layer_intervals` and
`::photon_first_interactions`.

## Source and independence boundary

Fresh verifier context, 2026-10-18. The following construction was written
before inspecting the implementation bodies; the verifier received the ledger
claim, API signatures, units, assumptions and docstrings only.

Source: [PENELOPE-2024, NEA/MBDAV/R(2024)1](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf),
§1.4.2–1.4.4, Eqs. 1.106–1.111 (macroscopic rate and exponential flight),
1.119 (interaction-type probability), and 1.124–1.125 (inverse flight
sampling and interface continuation). Direct PDF retrieval returned HTTP 403;
the publisher's indexed text supplied Eqs. 1.106–1.109. The remaining equation
numbers follow the ledger citation; the derivation below independently obtains
the needed inverse and conditional probabilities from the local rate law.
Cross-section values and interpolation belong to the separate
`narrow-beam-total-attenuation` claim, rather than this geometry/sampling claim.

Assume a fixed photon energy in eV, unit sample-frame direction, coordinates
and distances in Å, atomic densities in Å$^{-3}$ and partial cross sections
in Å$^2$. Homogeneous layers are ordered, non-overlapping z intervals;
gaps and empty compositions are vacuum. A common finite rectangle is centered
on the transverse origin, or the transverse extent is infinite. Two independent
uniforms are supplied explicitly. No outgoing scattering state is claimed.

## Independent construction before implementation inspection

### Geometry

Let the ray be $\mathbf r(s)=\mathbf r_0+s\hat{\mathbf n}$, $s\ge0$.
For each bounded axis $q$ with lower and upper faces $L_q<U_q$, if
$n_q\ne0$, define

$$
t_q^- = \min\left(\frac{L_q-r_{0q}}{n_q},\frac{U_q-r_{0q}}{n_q}\right),
\qquad
t_q^+ = \max\left(\frac{L_q-r_{0q}}{n_q},\frac{U_q-r_{0q}}{n_q}\right).
$$

For $n_q=0$, the axis imposes no restriction when
$L_q\le r_{0q}<U_q$ and rejects the ray otherwise. This half-open rule
assigns a ray lying along a shared face to exactly one layer. Unbounded
transverse axes impose no restriction. Intersecting all axis constraints with
the forward half-line gives

$$
a_l=\max(0,t_x^-,t_y^-,t_{z,l}^-),\qquad
b_l=\min(t_x^+,t_y^+,t_{z,l}^+).
$$

Only $b_l>a_l$ contributes; a tangency, backward interval or miss has zero
length. Empty intervals may be represented by coincident zero endpoints.
The origin must never be clamped: an outside ray must retain its vacuum travel
$a_l$, and a miss must stay a miss. At a nonparallel interface the material
entered in the forward direction owns the positive path, including the case
of an origin on the outer upper face pointing inward.

### Macroscopic rate and survival

For atomic species $i$ and channel $c$ in layer $l$, define

$$
w_{lic}(E)=n_{li}\sigma_{ic}(E),\qquad
\mu_l(E)=\sum_i\sum_{c=0}^{4}w_{lic}(E).
$$

The five channels are coherent scattering, incoherent scattering,
photoelectric absorption, nuclear-field pair production, and electron-field
pair production; the public channel-order tuple determines their integer
labels. The rate is nonnegative and has units Å$^{-1}$.

From the local collision probability $\mu(s)\,ds$, independent of previous
survival, $dS/ds=-\mu(s)S$, $S(0)=1$. Therefore

$$
S(s)=\exp[-T(s)],\qquad
T(s)=\sum_l\mu_l\,\max\{0,\min(s,b_l)-a_l\}.
$$

Here empty intervals contribute zero, as do vacuum intervals even when their
geometric length is infinite. In particular, zero times infinite length must
be evaluated as zero optical depth, not a NaN.

Sort occupied intervals by forward entry. With $D_l=\mu_l(b_l-a_l)$ and
$C_l$ their cumulative optical depth, draw

$$
\tau=-\ln(1-u_0),\qquad 0\le u_0<1.
$$

Escape occurs when no positive interval has $C_l>\tau$. Otherwise choose
the first such interval and return

$$
s=a_l+\frac{\tau-C_{l-1}}{\mu_l}.
$$

The strict inequality implements the ledger's deterministic tie convention:
an exact optical-depth boundary belongs to the following positive interval.
If that interval follows a vacuum gap, its entry distance is retained.
At $u_0=0$, zero-rate intervals are skipped and collision occurs at entry
to the first positive-rate interval. A final-boundary tie escapes.
These tie choices do not change a continuous random distribution, but do
matter for explicit uniforms and reproducibility.

### Joint atom and channel selection

Given collision in layer $l$, the probability of an atom/channel pair is

$$
P(i,c\mid l,E)=\frac{w_{lic}(E)}{\mu_l(E)}.
$$

Flatten the weights in element-major, channel-minor order. Set
$x=u_1\mu_l$ and choose the first cumulative weight strictly greater than
$x$. Thus zero weights cannot be selected, including when $u_1=0$.
For flattened index $j$, $i=\lfloor j/5\rfloor$ and $c=j\bmod5$.
Independence of the two supplied uniforms makes the joint density the
survival density times this conditional law. The returned element index
belongs to the original composition order, not a sorted atomic-number order.

### Filters and limiting cases

| Filter/case | Independent expectation |
|---|---|
| Dimensions | $w$ and $\mu$ are Å$^{-1}$; $T$ and $\tau$ dimensionless; $s$ in Å |
| Signs | Nonnegative cross sections/densities give nonincreasing survival and nonnegative collision distance |
| Homogeneous slab of path length $L$ | Escape probability $e^{-\mu L}$; interior inverse distance $a+\tau/\mu$ |
| Oblique slab | Geometric path includes $1/\lvert n_z\rvert$, with no extra angular factor in $\mu$ |
| Zero density, no layers, or complete miss | Escape; sentinel distance infinity and indices minus one |
| Infinite positive-rate lateral interval | Collision with certainty and mean distance from entry $1/\mu$ |
| Infinite zero-rate lateral interval | Escape; no indeterminate optical depth |
| Outside origin followed by vacuum gap | Entry and gap distances remain in $s$; vacuum contributes no $T$ |
| Backward z direction | Layers visited in decreasing z, determined by increasing forward entry |
| Parallel shared face | Exactly one half-open owner; no source displacement |
| Zero partial channel or atomic density | Never selected |
| Single positive atom/channel weight | That atom and channel always selected |

All cheap filters pass.

## Implementation comparison

The subsequent body inspection found the same construction term by term:

- `ray_layer_intervals` intersects axis near/far distances, starts at zero,
  accepts only positive-length intersections, and returns coincident zero
  endpoints for misses. Its parallel branch tests lower-inclusive,
  upper-exclusive membership on every bounded axis. It neither clamps nor
  displaces origins.
- `_coefficients` constructs density times each of the five public EPDL
  partial cross sections in element-major, channel-minor order. Empty
  compositions have zero columns and hence zero total rate. Invalid or
  negative individual rates are refused.
- `_collision_distances` assigns zero depth before handling zero-rate or
  empty intervals, avoiding a zero-rate infinite-path NaN. It sorts by
  forward entry and uses strictly greater cumulative depth with a
  positive-depth guard. Its distance is entry plus residual optical depth
  divided by the selected layer rate, exactly as derived.
- `photon_first_interactions` uses `-log1p(-u0)`, and chooses the first
  cumulative atom/channel weight strictly above the second threshold.
  Integer quotient/remainder by five return the original element and channel
  indices. A `nextafter` cap keeps a rounded threshold below the total for a
  uniform immediately below one; it does not change the mathematical law.
  Escape sentinels match the stated API.

The API validates finite unit rays, finite ordered non-overlapping layer
bounds and nonnegative finite densities, finite supported energies, and
uniforms in the half-open unit interval. Statistical independence of supplied
uniforms is a caller assumption, rather than something validation can infer.

### Independent numerical replay

The verifier ran a separate scalar implementation of ray/box constraints,
sequential residual-optical-depth inversion, and sequential atom/channel
selection. Only the public atomic cross-section getter and channel-order
tuple were shared input data; no navigator helper supplied expected geometry,
rates, inversion or selection. Random draws used fixed seeds.

| Check | Result |
|---|---|
| 3000 rays with outside origins and arbitrary directions through Si / vacuum / W–O layers; both infinite slabs and a finite centered rectangle; explicit outer/interface/parallel-face rays with zero first uniform | Zero interval, layer, channel or element mismatches in either geometry |
| 6000 rays per geometry through optically thick layers; energies log-uniform from 1 keV to 10 MeV; infinite transverse extent | 4576 collisions, zero escape/layer/atom/channel mismatches; maximum relative distance difference $4.10\times10^{-16}$ |
| Same 6000 rays with finite transverse extent | 2045 collisions, zero escape/layer/atom/channel mismatches; maximum relative distance difference $3.88\times10^{-16}$ |
| `pyrite-dev test tests/dev/test_docs.py` | 8 passed |

These replays validate the navigator independently of its vectorized
implementation; they do not independently certify the underlying EPDL data.

## Findings and verdict

No divergent factor, sign, unit, angular factor, interval ownership, optical
depth tie convention, or joint channel/atom probability was found.

A numeric robustness observation during review was reported to the task owner:
finite per-channel rates can still overflow when summed at unphysical finite
densities. The owner added an explicit finite-total-rate refusal. This does
not alter the derived law on its finite-rate domain.

**Verdict: rederived.** Suggested ledger edit: move
`photon-channel-first-interaction` from `unverified` to `rederived`, citing
this fresh-context derivation and numerical replay. Move to `anchored` only
once the task owner's independent reference regression anchors are green.
Only a human may mark it `signed-off`.

Source-access limitation: the exact contents of cited Eqs. 1.119 and
1.124–1.125 could not be reread from the PDF in this environment. The
collision/type law was independently derived here from the publisher-indexed
local-rate and survival equations. This limits citation-number auditing,
not the symbolic or numeric agreement demonstrated above.
