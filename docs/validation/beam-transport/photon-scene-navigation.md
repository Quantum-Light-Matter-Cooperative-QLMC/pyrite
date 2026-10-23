# Photon scene navigation: independent verification

Validation: `photon-scene-navigation`.

**Original independent verdict: discrepancy** (implementation at `3394ca95`).
The separating-axis overlap tolerance depended on absolute position and
could accept coincident material boxes. The original derivation and finding
are preserved below. The subsequent implementation repair is recorded
separately. Fresh-context verification of that repair on 2026-10-23 found a
different support-projection discrepancy, documented below. A second
implementation repair now retains every nonzero support contribution. The
verdict on that pre-repair code is **discrepancy**. A later independent pass
found a missing cross-axis case; after its repair, the final independent
verdict is **rederived**, recorded at the end of this page. Historical
findings below refer to their stated implementation snapshots.

## Scope and sources

This verification starts from the ledger contract before reading the
implementation. It covers homogeneous disjoint boxes, an analytic target
stack/prism, and an ideal rectangular aperture, in sample-frame Å. It does
not validate interaction cross sections, charged transport, or detector response.

The geometric source is David Eberly, [Dynamic Collision Detection using
Oriented Bounding Boxes](https://www.geometrictools.com/Documentation/DynamicCollisionDetection.pdf),
§§2.1–2.2.1, equation (1) and the stationary fifteen-axis test. Face contact
is permitted here, so separation includes equality, unlike closed-solid
intersection in that source. Optical-depth continuation follows the survival
law underlying [PENELOPE-2024](https://www.oecd-nea.org/jcms/pl_108003/penelope-2024-a-code-system-for-monte-carlo-simulation-of-electron-and-photon-transport),
NEA/MBDAV/R(2024)1, §1.4.4. The NEA landing page was accessible; its download
returned the landing page rather than the manual, so the subsection numbering
has not been independently audited. The continuation below is derived directly
from survival probabilities.

## Independent derivation

Let a box have centre $\mathbf c$, orthonormal axes $\mathbf e_a$,
and half sizes $h_a>0$. A unit-direction ray is
$\mathbf x(s)=\mathbf o+s\hat{\mathbf d}$, with $s\geq0$ in Å.
Taking dot products with each axis gives

$$
q_a=(\mathbf o-\mathbf c)\cdot\mathbf e_a,
\qquad v_a=\hat{\mathbf d}\cdot\mathbf e_a,
\qquad -h_a\leq q_a+s v_a<h_a.
$$

For $v_a\neq0$, the two roots $(-h_a-q_a)/v_a$ and
$(h_a-q_a)/v_a$ bound the axis interval; interchange their order for
negative velocity. Intersect all three intervals and the forward half-line.
A positive-length chord exists exactly when the upper endpoint exceeds the
lower. For $v_a=0$, the ray is admitted exactly when
$-h_a\leq q_a<h_a$. Endpoint ownership affects parallel surface rays;
an isolated endpoint has zero optical thickness.

Sort the disjoint material chords $[a_r,b_r]$ by entry distance. The
piecewise-constant attenuation coefficient $\mu_r$ has units Å$^{-1}$.
Independently multiplying survival factors through each chord gives

$$
P(S>s)=\exp[-H(s)],\qquad
H(s)=\sum_r\mu_r\max\{0,\min(s,b_r)-\max(0,a_r)\}.
$$

For $u\in(0,1)$, draw $\tau=-\ln u$. Subtract each full chord's
$\mu_r(b_r-a_r)$ until the remainder fits a positive-rate chord. The
collision distance there is $s=a_r+\tau_{\rm remaining}/\mu_r$.
Vacuum contributes no hazard regardless of gap length. Exhausting all
chords means escape. At an interface of zero measure either convention is
acceptable, provided no finite chord is traversed twice.

For a candidate separating direction $\hat{\mathbf u}$, a box's projected
radius follows by independently maximizing each local coordinate:

$$
\rho(\hat{\mathbf u})=\sum_a h_a
  |\mathbf e_a\cdot\hat{\mathbf u}|.
$$

Two boxes have overlapping interiors iff their projected centre separation
is strictly smaller than the sum of these radii on every nonzero member
of the six face normals and nine edge cross products. Zero cross products
are omitted. A laterally infinite slab has infinite projected radius
whenever the test direction has a lateral component; along its normal the
radius is its half thickness. This avoids undefined products of infinity
and zero. An aperture has one zero half size: positive projection overlap
tests whether it penetrates a material body, while face contact remains
allowed.

For the stated sample-to-lab rotation $R$, orthogonality implies

$$
\mathbf c_s=10^7 R^{\mathsf T}\mathbf c_{\rm lab},\qquad
\mathbf e_{a,s}=R^{\mathsf T}\mathbf e_{a,\rm lab}.
$$

Only positions and lengths gain the mm-to-Å factor. The inverse is the
transpose, and no translation is present because both origins are the
sample entrance. A common rigid rotation preserves every dot product,
chord length, separation decision, and optical depth.

For aperture centre $\mathbf c$, outward normal $\hat{\mathbf n}$,
and in-plane axes $\mathbf u,\mathbf v$, solve its plane crossing:

$$
s_{\rm ap}=\frac{(\mathbf c-\mathbf o)\cdot\hat{\mathbf n}}
                   {\hat{\mathbf d}\cdot\hat{\mathbf n}}.
$$

A front hit requires a positive denominator, $s_{\rm ap}>0$, and
both projected local coordinates within the rectangle. Absorption by the
aperture precedes material interaction iff $s_{\rm ap}<s_{\rm coll}$.

## Filters and exact limits

- Units pass: clipping roots and collision distances are Å, projected radii
  are Å, and $H$ is dimensionless. Pose directions remain unitless.
- Signs pass: reversing a nonparallel ray reverses and reorders roots;
  positive attenuation reduces survival; the inverse rotation is $R^{\mathsf T}$.
- Axis-centred chords have length $2h_a$. The corner-to-corner chord has
  length $2\sqrt{\sum_a h_a^2}$.
- Vacuum has $H=0$ and no collision. A tilted infinite plate of normal
  thickness $t$ gives transmission $\exp[-\mu t/|\cos\theta|]$.
- Removing all boxes must preserve the original stack navigator, including
  its explicit random draws and boundary ownership.

## Implementation comparison

The implementation was inspected only after the derivation above was written.

| Owner | Comparison |
| --- | --- |
| `ray_box_intervals`, `_clip_axes` | Row-vector transforms are the transpose equivalent of the axis dot products above; forward clipping and the half-open parallel rule match. |
| `scene_first_interactions`, `_interval_first_interactions`, `_collision_distances` | Chords are sorted spatially before cumulative optical-depth inversion; gaps and zero rates contribute no hazard. The draw uses $-\ln(1-u)$, distributionally equivalent to $-\ln u$. Exact interface depth belongs to the following interval. |
| `photon_box_from_pose`, `PhotonAperture.from_pose` | The transpose rotation, row-axis equivalent, full-size to half-size factor, and $10^7$ Å/mm conversion match. |
| `PhotonAperture.intersect` | Positive-facing, positive-distance plane intersection and rectangular local coordinates match. Rectangle edges are included, a measure-zero convention. |
| `_overlap` | The fifteen candidate axes and projected-radius formula match away from numerical thresholds. The tolerance subtraction changes the stated overlap predicate and fails the translation limit below. |

### Exact discrepancy: absolute-coordinate overlap tolerance

The first divergent term is the subtraction of `tolerance` from the support
radius sum in `_overlap`:

```python
tolerance = 1e-12 * max([1.0, *finite, *centers])
if abs(c1 - c2) >= r1 + r2 - tolerance:
    return False
```

Consider two **identical** axis-aligned Cu boxes, each with half sizes
$(1,1,1)$ Å. At centre $(0,0,0)$ Å the public
`validate_photon_regions([], [box, box])` correctly rejects them. Translate
both to $(0,0,3\times10^{12})$ Å. Their centre separation remains exactly
zero, each projected radius on a coordinate axis remains exactly one Å,
but the tolerance becomes three Å. The code now evaluates
$0\geq1+1-3$ and returns non-overlap, so the public validator accepts two
coincident bodies. All inputs in this example are exactly representable in
binary64; this failure does not require cancellation or noisy rotations.

The translation is deliberately extreme to make the contradiction exact.
There is no coordinate/extent restriction excluding it in the claim or
public geometry validation. Smaller translations can similarly hide small
penetrations. This finding does not establish a failure for ordinary
millimetre-scale filters with well-separated faces.

Overlapping material violates the disjoint-chord prerequisite of the
optical-depth inversion. The geometry gate must therefore refuse such input
rather than certify it. A repair should distinguish genuine coordinate
roundoff from body size and contact tolerance; unresolved geometry should
fail closed. It must preserve exact face contact and include common
translation, coincident-box, and thin-body regressions. No source repair was
made in this independent verification context.

Reproducer (inside the project runner's Python environment):

```python
from pyrite.montecarlo.transport.photon_navigation import (
    PhotonBox,
    validate_photon_regions,
)

axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
box = PhotonBox((0.0, 0.0, 3e12), axes, (1.0, 1.0, 1.0), (("Cu", 0.0849),))
validate_photon_regions([], [box, box])  # Returns; it must reject overlap.
```

## Remote evidence

On 2026-10-23 UTC, SLURM job 1701 ran on `qlmc-queen`, using a separate
checkout staged from `3394ca95fad5415947345f0b17eaf3d1f582e970`. It requested
two CPUs, 6 GiB RAM and fifteen minutes; no GPU was requested. Synced code
digest: `a1877afa83dded9b0efe8837288fcc0c39d5c60dcd3d69e8be29e555775a1c7a`.

An independent numeric probe enumerated all six face-plane crossings of a
rotated box and kept a crossing only when its point lay within the other
face bands. This reference did not call the production clipping helpers.
Across 6,000 seeded rays, including 326 positive chords, both entry and exit
distances matched within $7.11\times10^{-14}$ Å maximum absolute error.
The coincident-box counterexample reproduced: rejected at the origin,
accepted after the common translation.

The first existing-test run had 20 passes and three staging failures: the
test process could not resolve the synced differential EPDL data, and the
docs test needed `scripts/export_docs_demo.py`. The shared fixtures clear
`PYRITE_HOME`, so setting that variable alone did not repair dataset lookup.
These are test-environment failures, not sampler or geometry verdicts.

Corrected SLURM job 1703 supplied an isolated `XDG_DATA_HOME` whose PyRITE
dataset directory points at the pinned synced files, and included the
missing script. Both owning modules passed: **23 tests in 4.54 seconds**.
The 6,000-ray independent comparison and exact counterexample also repeated
with the same results. Focused command:

```bash
uv run pyrite-dev test tests/montecarlo/test_photon_scene.py tests/dev/test_docs.py --durations=5
```

The passing existing tests do not cover the coincident-box counterexample
and therefore do not resolve the discrepancy.

## Implementation repair

At the user's request, the implementation was repaired after the independent
report was checkpointed in `5adf824f`. This section is implementation-context
evidence, not an independent verification of the repair.

The separating-axis comparison now projects the relative displacement
directly, rather than subtracting two absolute centre projections. Each
finite axis uses its own projected radius sum $r=r_1+r_2$ and distance
$d=|\Delta\mathbf c\cdot\hat{\mathbf u}|$:

$$
\epsilon_{\rm contact}=32\epsilon_{64}\max(d,r),
\qquad d\geq r-\epsilon_{\rm contact}\ \Longrightarrow\ \text{separation/contact}.
$$

Here $\epsilon_{64}$ is binary64 machine epsilon. The small relative contact
allowance covers rounding across the short projection/support dot products;
32 is a conservative arithmetic allowance, not an empirical fit to an
observed penetration. It has no absolute Å floor. Infinite projected radii
cannot separate bodies and are skipped before tolerance arithmetic. Neither
the world origin nor a large extent orthogonal to the tested axis contributes
to this tolerance. Coincident positive-size boxes therefore retain positive
overlap even when tiny or far from the origin. Exact face contact remains
allowed. No algorithm can recover a displacement already lost when input
coordinates were rounded; common-translation tests use exactly representable
coordinates to distinguish this unavoidable limit from the original defect.

The new public-validator regressions cover coincident boxes at zero and
large common translations, tiny boxes, thin bodies with enormous lateral
extents, analytic 0.5 Å overlap/contact/gap cases, and apertures embedded in
thin boxes. Before the source repair, **10 failed and 10 passed** remotely;
this includes failures at the origin, so the defect was not confined to
extreme world coordinates. After repair, all 20 new regression cases pass.
The local focused run, including the photon history, coupling, source-scoring,
Scene handoff, docs and ledger modules, had 127 passes and one slow-test skip.
Its sole failure was the unrelated `cr-valence-directional-loss` owner-marker
coverage check, also reproduced on main. Adding the nine missing docstring
back-references subsequently restored all 18 ledger checks without changing
that model's implementation or validation status. The photon interaction
models, flight draws, and production default path are unchanged.

## Original verifier verdict

- **Claim:** `photon-scene-navigation` —
  `montecarlo/transport/photon_navigation.py::_overlap` and the owners above;
  analytic ray clipping, Eberly §§2.1–2.2.1 equation (1), optical-depth survival.
- **Filters:** units pass; limits fail (common-translation invariance of
  overlap validation); signs/conventions pass outside tolerance thresholds.
- **Re-derivation:** differs — the absolute-centre-dependent subtraction
  from the sum of projected support radii admits coincident boxes.
- **Verdict:** `discrepancy`.
- **Write-up:** `docs/validation/beam-transport/photon-scene-navigation.md`.
- **Original suggested ledger change:** change `photon-scene-navigation` from
  `unverified` to `discrepancy`, link this report, and record the tolerance
  counterexample. The task owner applies the edit; no human sign-off or
  full-track validation is implied. After the repair above, the owner retained
  `unverified` pending a fresh-context verdict on the changed implementation.

## Fresh-context repair verification (2026-10-23)

This pass did not implement the repair. Before reading its implementation,
the expected geometry was derived again from the ledger contract and Eberly's
stationary separating-axis construction. The preceding report supplies the
historical discrepancy, not evidence that the repair is correct.

### Independent expression before implementation inspection

The support function of a centred box in a direction $\mathbf u$ is

$$
S(\mathbf u)=\sum_{a=1}^3 h_a|\mathbf e_a\cdot\mathbf u|.
$$

Subtract centres before projecting. With $\Delta\mathbf c=\mathbf c_2-\mathbf c_1$,
interior separation on a nonzero candidate axis is

$$
|\Delta\mathbf c\cdot\mathbf u|\geq S_1(\mathbf u)+S_2(\mathbf u).
$$

Six face normals and nine nonzero edge cross products suffice. Axis
normalization is optional because both sides are homogeneous of degree one.
A contact allowance may depend on the magnitudes of the arithmetic terms
on this particular axis, but must not depend on the absolute world origin
or on an extent perpendicular to that axis. In particular, identical boxes
must overlap at every representable common translation, and a thin box's
normal overlap must not disappear as its lateral extent grows. A nonzero
projection of an infinite extent gives infinite support, so that axis cannot
separate the regions. Exactly zero projection contributes zero.

For ray clipping, solve each local band
$-h_a\leq (\mathbf o-\mathbf c)\cdot\mathbf e_a+s\hat{\mathbf d}\cdot\mathbf e_a<h_a$
and intersect the resulting
intervals with $s\geq0$. A parallel ray is admitted only on the half-open
band. The accumulated hazard is the sum of each nonnegative material chord
length times its attenuation coefficient. Vacuum gaps have zero hazard;
inverting the cumulative hazard gives the collision point without resetting
the optical-depth draw at interfaces. A common orthogonal transformation
preserves these dot products. The lab-to-sample transform is the transpose
of the stated sample-to-lab rotation, with $10^7$ Å/mm applied only to lengths.

Units, signs and exact vacuum, normal-incidence, common-translation and
common-rotation limits therefore pass at the contract level. Finite precision
must still be tested against the strict support inequality, especially for
very thin bodies and near-parallel axes.

### Implementation comparison and new discrepancy

The repaired `_overlap` subtracts centres before projection and bases its
contact allowance on each axis's projected lengths. Those changes address
the original defect. However, `_project` still thresholds the support sum:

```python
weights = np.abs(axes @ axis)
used = weights > 1e-12
return float(np.sum(half[used] * weights[used]))
```

The first divergent term is omission of $h_a|\mathbf e_a\cdot\mathbf u|$
when the angular factor is small but nonzero. A small angle does not bound
the omitted length independently of the half extent. `_overlap` also omits
cross-product axes with norm below `1e-9`; this was not needed to trigger
the counterexample and is not independently adjudicated here.

Take two boxes with half sizes $(10^{13},1,1)$ Å. The first is axis-aligned
at the origin. Put the second at $(0,0,3)$ Å with local axes

$$
\mathbf e_x=(\cos\theta,0,\sin\theta),\quad
\mathbf e_y=(0,1,0),\quad
\mathbf e_z=(-\sin\theta,0,\cos\theta),\qquad
\theta=5\times10^{-13}.
$$

The point $\mathbf p=(-6\times10^{12},0,0)$ Å lies strictly inside both.
Its local coordinates in each box are $(-6\times10^{12},0,0)$ to binary64
precision. The omitted long-axis contribution to the second box's support
along world $z$ is $10^{13}\sin\theta=5$ Å. The support radius sum is
seven Å, exceeding the centre separation of three Å. The implementation
drops the five Å contribution and uses a sum of two Å, falsely reporting
separation. The axes pass the public orthonormality check.

This deliberately extreme aspect ratio is allowed by the public contract;
the failure needs neither ambiguous face contact nor loss of displacement
under translation. It does not establish failure for ordinary millimetre
filters. This self-contained assertion reproduces the defect at `e1d2b98a`
using an independent strict-interior witness:

```python
import numpy as np
from pyrite.montecarlo.transport.photon_navigation import (
    PhotonBox,
    validate_photon_regions,
)

angle = 5e-13
c, s = np.cos(angle), np.sin(angle)
axes = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
first = PhotonBox(
    (0.0, 0.0, 0.0),
    tuple(map(tuple, np.eye(3))),
    (1e13, 1.0, 1.0),
    (),
)
second = PhotonBox(
    (0.0, 0.0, 3.0),
    tuple(map(tuple, axes)),
    (1e13, 1.0, 1.0),
    (),
)
point = np.array([-6e12, 0.0, 0.0])
for box in (first, second):
    local = np.asarray(box.axes) @ (point - np.asarray(box.center_ang))
    assert np.all(np.abs(local) < np.asarray(box.half_extent_ang))
try:
    validate_photon_regions([], [first, second])
except ValueError:
    pass
else:
    raise AssertionError("overlapping boxes were accepted")
```

A temporary test containing this assertion was run through
`UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test
/tmp/test_photon_navigation_reverification.py --durations=0`.
It failed at the final assertion after both strict-interior checks passed
(one test, 1.04 seconds). No implementation change was made in this verifier
context. A repair must retain significant finite support contributions and
treat nonzero infinite-extent projections conservatively; changing the
contact allowance does not address this term.

The existing scene module still passes all **35 tests** (1.76 seconds), and
`tests/dev/test_docs.py` passes all **eight tests**. These results distinguish
the uncovered geometry defect from failures of the existing regression set.

### Fresh verifier verdict

- **Claim:** `photon-scene-navigation` —
  `montecarlo/transport/photon_navigation.py::_overlap` through `_project`;
  Eberly §§2.1–2.2.1, equation (1), stationary support projections.
- **Filters:** units pass; limits fail (near-parallel long-body overlap);
  signs/conventions pass away from thresholded projections.
- **Re-derivation:** differs — support contributions with angular factor
  at most `1e-12` are discarded without regard to their projected length.
- **Verdict:** `discrepancy`.
- **Write-up:** `docs/validation/beam-transport/photon-scene-navigation.md`.
- **Suggested ledger change:** change `photon-scene-navigation` from
  `unverified` to `discrepancy`, reference this fresh verification and retain
  the original repair history. The task owner applies the edit; the verifier
  does not change the ledger or implementation.

## Support-projection repair (implementation context)

After the fresh verifier report, the implementation owner changed `_project`
to omit only exactly zero dot products. Small nonzero factors retain their
full finite support contribution; any nonzero factor on an infinite extent
produces infinite support. Exactly zero factors are excluded before
multiplication, avoiding undefined zero-times-infinity arithmetic. No clipping,
flight-draw, contact-allowance or cross-axis selection rule changed.

The public-validator regression `test_small_tilt_preserves_long_box_overlap`
uses the independently derived strict-interior witness above, both tilt signs,
and finite-box and infinite-stack counterparts. All four cases failed before
the repair. The companion contact/gap regression keeps valid geometries
accepted. After repair the scene, photon history, coupled history and Scene
handoff modules pass **93 tests**, with one existing slow test skipped; each
new regression case takes less than five milliseconds.

These are implementation regression checks. They do not independently verify
the changed code or advance the complete claim to `rederived`. In particular,
the separate near-parallel cross-axis threshold noted above remains outside
this repair's scope. Fresh-context verification remains required.

## Fresh independent verification at 3f19d652 (2026-10-23)

### Independence and derivation before implementation inspection

This verifier did not implement the claim. The ledger row, owner signatures and
AST-extracted docstrings were the only implementation information read before
writing this section; prior report sections and implementation bodies were withheld.
The complete claim includes finite boxes, finite or infinite target slabs,
zero-thickness surfaces, optical-depth continuation and pose conversion.

Source inspected: [Eberly, Dynamic Collision Detection using Oriented Bounding
Boxes](https://www.geometrictools.com/Documentation/DynamicCollisionDetection.pdf),
§§2.1–2.2.1, especially equation (1) and Table 1. Its closed-set intersection
criterion needs equality assigned to contact for this project's disjoint-interior
contract. Direct access to the cited PENELOPE-2024 PDF failed; the exponential
survival derivation below is independent, with §1.4.4 equation-number attribution
remaining unaudited.

For unit direction $\mathbf d$, write $\mathbf x(s)=\mathbf o+s\mathbf d$,
$s\geq0$. In local coordinates $q_a=\mathbf e_a\cdot(\mathbf o-\mathbf c)$,
$v_a=\mathbf e_a\cdot\mathbf d$, solve $-h_a\leq q_a+s v_a<h_a$.
For nonzero $v_a$, the two boundary quotients give the lower and upper crossing
distances after sorting. Intersect all bands and $s\geq0$; positive material
length requires exit strictly greater than entry. If $v_a=0$, the whole ray
satisfies this band iff $-h_a\leq q_a<h_a$. A merely small nonzero $v_a$
is not parallel: its crossing can be arbitrarily distant. Infinite lateral
extents impose no crossing constraint. With normalized directions the quotients
are Å; with nonunit directions they are ray parameters rather than lengths.

Independent support derivation: maximize the linear functional of
$\mathbf c+\sum_a t_a\mathbf e_a$, $|t_a|\leq h_a$, by choosing each sign.
Thus

$$
r(\mathbf u)=\sum_a h_a|\mathbf e_a\cdot\mathbf u|,\qquad
D(\mathbf u)=|(\mathbf c_2-\mathbf c_1)\cdot\mathbf u|.
$$

Disjoint interiors require some face normal or nonzero edge-cross-edge axis
with $D\geq r_1+r_2$. All 15 candidate directions must otherwise be retained.
Normalization is optional because both sides scale identically. An infinite
extent contributes infinity for every nonzero projection and zero for an exactly
zero projection. Small angles cannot be truncated independently of extent.
For a surface with one zero extent, test whether its relative interior enters
the material; contact on a material face is allowed, a plane through its interior
is forbidden. SAT against a full-dimensional body follows the same support rule.
Roundoff allowances may classify an unresolved boundary as contact, but cannot
justify finite geometric overlap far above the local arithmetic uncertainty.

For disjoint forward material intervals $[a_r,b_r]$ sorted by entry, the hazard
is piecewise constant, zero in vacuum. Integrating survival gives

$$
T(s)=\exp[-H(s)],\qquad
H(s)=\sum_r\mu_r\max(0,\min(s,b_r)-a_r),\qquad
\tau=-\ln U.
$$

Subtract completed interval depths from $\tau$; in the first region with
remaining depth smaller than $\mu_r(b_r-a_r)$, place the event at
$a_r+\tau_{\rm remaining}/\mu_r$. All $\mu_r$ have units Å$^{-1}$;
$H$ and $\tau$ are dimensionless. Zero rate means no event. Conditional channel
and atomic probabilities are their macroscopic partial rates divided by total
rate. A straight ray cannot re-enter a convex body, but a changed direction after
scattering requires recomputing intervals. Infinite depth absorbs with probability
one when the rate is positive. A finite tilted plate yields thickness divided
by the absolute direction-normal cosine, unless a lateral face truncates it.

Finally, $R$ maps sample vectors into lab vectors, hence the inverse pose is
$\mathbf c_s=10^7R^{\mathsf T}\mathbf c_{\rm lab}$ and
$\mathbf e_{a,s}=R^{\mathsf T}\mathbf e_{a,\rm lab}$, with sizes also multiplied
by $10^7$ Å/mm. Proper rotation preserves lengths, dot products and handedness;
no offset is permitted under the stated common-origin convention. Identity,
common rigid rotation/translation, axial chord $2h_a$, diagonal chord
$2\|\mathbf h\|$, vacuum survival and zero-width contact are the independent
filters. Units, signs and these analytic limits pass at the expression level.

### Comparison, new discrepancy, and reproducible evidence

`ray_box_intervals` transforms origins and directions with the row-axis matrix
and `_clip_axes` implements the independently derived interval intersection,
including exact-zero parallel tests and half-open parallel-face membership.
`scene_first_interactions` concatenates region intervals; the inversion helper
sorts by entry, accumulates positive material depths and uses strict greater-than
at an exact interface. Its draw convention is $\tau=-\ln(1-U)$, distributionally
equivalent to the derivation's survival uniform. `_interval_first_interactions`
selects flattened atom/channel partial rates. This comparison does not independently
validate the EPDL cross sections themselves. `PhotonBox` checks positive finite
sizes and orthonormal axes; pose conversion and `PhotonAperture.region` match the
unit conversion and zero-thickness support construction under proper rotations.
`_project` at this tip retains every nonzero projection, resolving the preceding
support-sum finding. Relative displacement and axis-specific contact allowance
also remove the original coordinate-scale failure.

**The complete claim still differs.** `_overlap` excludes candidate axes when
`norm < 1e-9`, instead of excluding only an exactly zero cross product. This is
an independent approximation to the separating-axis theorem: two long boxes may
have their only separating direction on an arbitrarily small edge cross product.
It produces false overlap, so `validate_photon_regions` refuses disjoint boxes,
finite-target counterparts, and disjoint scoring surfaces. It does not demonstrate
an accepted overlapping body; the observed failure is rejection of valid geometry.

Here is a public-API reproducer, executable with `uv run python` from this checkout:

```python
import numpy as np
from pyrite.montecarlo.transport.photon_navigation import (
    PhotonBox,
    validate_photon_regions,
)

angle = 1e-10
k = np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0)
K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
first = PhotonBox((0.0, 0.0, 0.0), tuple(map(tuple, np.eye(3))), (1e12, 1.0, 1.0), ())
second = PhotonBox(tuple(10 * k), tuple(map(tuple, R.T)), (1e12, 1.0, 1.0), ())
validate_photon_regions([], [first, second])
# Incorrect ValueError: photon boxes must not overlap ...
```

The first long axis is $\hat{\mathbf x}$; the second is
$\cos\theta\hat{\mathbf x}+\sin\theta(0,1,1)/\sqrt2$.
Their cross product has norm $10^{-10}$ and direction $\mathbf k$.
Along it, independently,

$$
D=10\ \text{Å},\qquad r_1+r_2=2\sqrt2\ \text{Å},\qquad
D-r_1-r_2=7.171572875\ \text{Å}>0.
$$

The measured binary64 values were `9.999999999999998` and
`2.8284271247461903`. This is a resolved positive gap, vastly larger than
$32\epsilon\max(D,r_1+r_2)$. The six face-axis tests cannot separate these rods:
the long-axis tilt projects about $70.71$ Å onto each transverse face direction,
whereas the centre displacement projects only $7.07$ Å. The missed cross axis
is therefore essential, despite its small unnormalized length. For a finite
stack counterpart, use the first body's axis-aligned geometry as a stack layer
from $-1$ to $1$ Å, with width $2\times10^{12}$ Å and height $2$ Å, and validate
`second` against it. For the zero-thickness version, replace the second box by
`surfaces=[(np.array(second.center_ang), np.array(second.axes),
np.array([1e12, 1., 0.]))]` and retain only `first` as a material box. Its support
is smaller and the same positive-gap certificate proves disjointness; the
validator nevertheless raises the scoring-surface error.

A separate input-validation limitation is `_rotation`: it checks orthogonality
but accepts determinant $-1$. A reflection is outside the stated physical
sample-to-lab rotation domain, so this is not used as the in-domain discrepancy
verdict. It is nevertheless accepted by the public constructor. With
`sample_to_lab=np.diag([-1.,1.,1.])`, identity-oriented lab pose centred at
`(0,0,1)` mm, the transformed lab normal should remain positive z, but the
aperture normal becomes negative z because a cross product changes sign under
an improper orthogonal map. The origin ray along positive z then misses rather
than hitting at $10^7$ Å. Rejecting reflections would make that domain explicit.
`PlanarPose` itself describes a target-reference origin, while this adapter's
ledger explicitly assumes that origin is the sample entrance; validation here
is conditional on that documented common-origin assumption.

### Checks and limits

Independent small CPU probes used no implementation helpers for the expected
geometry or optical-depth result. A six-plane boundary enumeration compared
2,000 seeded rays through a rotated translated box, with absolute and relative
tolerance $2\times10^{-12}$. Finite and infinite slab controls distinguish an
interior aperture, face contact, and an exterior aperture; both reject embedded
boxes and accept face contact. Synthetic partial-rate inversions cover unsorted
region storage, vacuum gaps, zero rates and an exact optical-depth interface.
Additional direct probes gave event distances `10.5`, `5.0`, and `inf` for
finite-gap, infinite-path, and all-vacuum cases. Proper-rotation pose checks and
a half-open parallel-face probe agree with the derived expressions. Numerical
claims are for representable finite coordinates; no algorithm can recover a
separation already lost when coordinates are rounded. Exhaustive extreme-range
IEEE overflow/underflow certification is not claimed.

The public reproducer is independent of cross sections and uses vacuum box
compositions solely to isolate region geometry: overlap validation applies to
all boxes regardless of composition. Replacing the empty compositions with a
positive material density leaves the SAT result unchanged. No source, tests or
ledger were edited by this verifier. Prior report sections remain historical.

### Fresh verifier verdict at 3f19d652

- **Claim:** `photon-scene-navigation` — the ledger's navigation, support,
  region-validation and pose owners — analytic clipping; Eberly equation (1)
  and §§2.1–2.2.1; exponential integrated-hazard continuation.
- **Filters:** units pass; limits failure (near-parallel finite boxes and
  zero-thickness surface disjointness); signs/conventions pass for proper rotations.
- **Re-derivation:** differs — `_overlap` discards nonzero cross axes below
  `1e-9`, eliminating a necessary separating direction.
- **Verdict:** `discrepancy`.
- **Write-up:** `docs/validation/beam-transport/photon-scene-navigation.md`.
- **Suggested ledger change:** owner changes `unverified` to `discrepancy`,
  records the cross-axis false-refusal witness and retains the successful
  support-projection repair history; never `signed-off`.

## Independent re-check of cross-axis repair (2026-10-23)

This is a re-check by the same independent verifier against the derivation
written above before implementation inspection. The reviewed tree is
`3f19d652` plus the task owner's uncommitted cross-axis/proper-rotation repair;
this verdict supersedes the preceding discrepancy for that repaired tree only.
No source, test, ledger or other report was edited by this verifier.

### Repair comparison against the independent expression

The source diff changes the candidate-axis norm from `np.linalg.norm(axis)` to
`np.hypot.reduce(axis)`, and the exclusion condition from `norm < 1e-9` to
`norm == 0.0`. Thus every represented nonzero face/cross axis is normalized
and tested, exactly as required by the independent SAT derivation. Repeated
hypotenuse evaluation avoids the squaring underflow that would otherwise
reintroduce an effective small-angle cutoff. `_project` still retains every
nonzero support weight, including infinite support for an unbounded slab.
The relative displacement, axis-local binary64 contact allowance and strict
interior/contact distinction remain the previously reviewed conventions.

The same public disjoint-box witness, finite-stack counterpart and zero-thickness
surface now pass validation. Independent positive-material probes also sweep
angles $10^{-8}$, $10^{-10}$, $10^{-50}$, $10^{-100}$, $10^{-200}$ and
$10^{-290}$ radians, with long half extent $100/\theta$ Å. At every angle the
$10$ Å separation is accepted for box/stack/surface, while coincident centres
are rejected for the two boxes. The separating-axis support and displacement
certificate remains independent of the implementation. Separate direct support
checks at $10^{-10}$, $10^{-100}$ and $10^{-200}$ retain the finite projected
long-side contribution and infinite slab contribution; an exactly perpendicular
projection onto the slab normal remains its finite thickness support.

`_rotation` now also requires determinant approximately positive one, in
addition to orthogonality. It rejects the accepted-reflection limitation
reported above for both plate and aperture construction. For proper rotations,
transpose inversion, the Å/mm factor and preservation of aperture handedness
remain correct. This makes the physical rotation input domain explicit; it
does not change the shared-origin assumption.

The complete claim comparison remains valid: analytic local-ray clipping with
exact-zero parallel treatment; forward finite/infinite slab intervals; convex-box
chords; sorted optical-depth continuation through vacuum gaps; joint partial-rate
selection; pose inversion; and zero-thickness scoring-surface refusal inside
material. No further divergent term or convention was found in these owners.
This is a derivation and bounded numerical verification, with the source-access,
representable-coordinate and extreme-range limits already recorded above. The
$32\epsilon$ contact allowance still intentionally treats numerically unresolved
penetration as contact; the verdict uses that declared finite-precision boundary
convention rather than asserting exact-real arithmetic at arbitrarily small gaps.

### Re-check evidence and verdict

The independent scratch suite passed **6 tests** through `pyrite-dev test`,
including the 2,000-ray plane-crossing oracle (241 positive chords), synthetic
optical-depth/channel inversion, finite/infinite slab and surface controls,
proper pose inversion, reflection refusal and the new angle/support sweeps.
The owning scene/navigation modules plus documentation tests passed **81 tests**.
After this report append, documentation tests were rerun separately. The task
owner owns the final combined docs build and rendered-math verification; the
preceding pre-repair build had already rendered this page with 109 math elements
and no raw dollar delimiters. This appended section uses the same math syntax.

- **Claim:** `photon-scene-navigation` — ledger navigation, region-validation,
  support and pose owners — analytic clipping; Eberly equation (1) and
  §§2.1–2.2.1; integrated exponential survival.
- **Filters:** units pass; limits pass; signs/conventions pass for proper
  rotations and the stated common-origin convention.
- **Re-derivation:** matches — every represented nonzero SAT axis and support
  projection is retained, with the documented binary64 contact allowance.
- **Verdict:** `rederived` for the repaired working tree.
- **Write-up:** `docs/validation/beam-transport/photon-scene-navigation.md`.
- **Suggested ledger change:** task owner changes the claim to `rederived`,
  links this repair verification and retains all historical discrepancy evidence;
  never `signed-off`.
