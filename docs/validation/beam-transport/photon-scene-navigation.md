# Photon scene navigation: independent verification

Validation: `photon-scene-navigation`.

**Original independent verdict: discrepancy** (implementation at `3394ca95`).
The separating-axis overlap tolerance depended on absolute position and
could accept coincident material boxes. The original derivation and finding
are preserved below. The subsequent implementation repair is recorded
separately; the complete claim remains `unverified` pending fresh-context
verification of the changed code.

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
    PhotonBox, validate_photon_regions,
)

axes = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
box = PhotonBox((0., 0., 3e12), axes, (1., 1., 1.), (("Cu", 0.0849),))
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
