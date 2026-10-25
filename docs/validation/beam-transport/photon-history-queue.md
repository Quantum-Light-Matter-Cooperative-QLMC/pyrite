# Photon history queue: independent verification

Validation: `photon-history-queue`.

**Current verdict: rederived**, after independent verification of the
bin-local tally and identifier-range repairs. The original discrepancy
and its reproducer below are historical; the final verdict is recorded
at the end of this page.

## Scope and source access

This fresh-context pass did not implement the photon queue. Its independent
expression below was written from the ledger, API signatures and docstrings
before reading implementation bodies. Navigation is a component dependency
under separate review. Cross-section and conditional interaction samplers
are reused claims, not independently rederived here.

The cited source is PENELOPE-2024, NEA/MBDAV/R(2024)1, §1.4 (detailed photon
histories and secondary stack), with reused pair kinematics in §2.4 and
ENDF-102 File 28 relaxation accounting. The
[NEA landing page](https://www.oecd-nea.org/jcms/pl_108003/penelope-2024-a-code-system-for-monte-carlo-simulation-of-electron-and-photon-transport)
confirms detailed photon tracking; its download returned HTML rather than
the manual during this pass. Exact subsection numbering is therefore not
independently audited. The queue construction and energy balance below are
derived directly from survival probabilities and local conservation.

## Independent derivation before implementation inspection

For a fixed photon energy in homogeneous region $r$, the total rate is the
sum of nonnegative partial rates $\mu_r=\sum_{a,c}n_{a,r}\sigma_{a,c}$.
Here number density is Å$^{-3}$ and cross section Å$^2$, giving a rate in
Å$^{-1}$. For piecewise straight material chords the cumulative hazard is

$$
H(s)=\int_0^s\mu(\mathbf x(t),E)\,dt,
\qquad P(S>s)=\exp[-H(s)].
$$

Inverting a single exponential optical-depth draw through all chords samples
the first collision; conditional channel and atom probability is the partial
rate divided by the total rate in the collision region. Vacuum gaps cost no
optical depth. A subsequent scattered photon starts a new flight with its
new position, direction and energy. All material chords and the ideal
absorbing aperture compete along the same ray: a front-facing rectangle
hit at strictly positive distance is terminal if it precedes the collision.
A miss with no collision is ordinary escape. Each photon contributes to
exactly one of escape, detection, local cutoff deposition or unresolved
energy, unless it first produces descendants.

At every physical interaction the parent's energy partitions into daughters
and local terminal terms. Rayleigh preserves photon energy. Compton gives
an outgoing photon and electron with $E=E'+T$. Photoabsorption gives a
photoelectron with $T=E-B$, followed by relaxation of the vacancy energy:

$$
E=(E-B)+\sum_j E_{\gamma,j}+\sum_j T_{e,j}
  +B_{\rm terminal}+\delta_{\rm transition}.
$$

The signed transition defect reconciles binding and tabulated transition
energies; it must not be clamped or counted twice. Pair creation gives

$$
E=T_-+T_++2m_ec^2.
$$

The photon-only queue hands off every positron with the rest-energy reservoir
pending; an electron below cutoff contributes kinetic energy locally.
A photon below cutoff deposits its energy; a photon stopped by a generation
cap remains unresolved. Neither rule may discard energy. Isotropic
relaxation emission uses uniform azimuth and uniform cosine, giving measure
$d\Omega/(4\pi)$; it does not multiply emitted energy by a solid angle.

Multiplying each local identity by the unchanged statistical weight and
summing over the tree cancels every internal parent/daughter term. Thus

$$
wE_0=E_{\rm escaped}+E_{\rm detected}+E_{\rm deposited}
 +B_{\rm terminal}+\delta_{\rm transition}+T_{\rm charged}
 +E_{\rm rest,pending}+E_{\rm unresolved}.
$$

Each terminal term already includes its particle's weight. Several source
photons may share a primary history: sum their weighted incident and terminal
energies by that history, without using it as an array position. Source event
and daughter lineage determine random streams independently of queue order,
batch position and chunking. An unchanged scattered photon continues its
own stream; distinct emitted daughters receive distinct ordinal keys.

For finite bin edges and fixed primary population, the ordinary aperture
spectrum is

$$
f_b=\frac{\sum_{i\in\mathrm{detected},\,E_i\in b}w_i}
              {N_{\rm primary}\,\Delta E_b}.
$$

There is exactly one primary and one bin-width denominator, no solid-angle
factor, and no normalization by the number of detected photons. All bins
must have positive width. Scaling source weights scales every terminal term
and bin tally linearly. Empty sources give empty rows and zero tallies.

Units, nonnegative probabilities, and signs pass at the contract level.
Vacuum, zero cross section, an aperture before all material, a one-generation
cap, and per-interaction conservation are the decisive implementation limits.

## Implementation comparison

| Owner | Comparison |
| --- | --- |
| `_flight`, `PhotonAperture.intersect` | Positive-facing plane hit, rectangular local bands and strict hit-before-collision ordering match. Escape and detection are mutually exclusive terminal terms. Exit positions intentionally retain the last interaction/source position, as documented. |
| `_interact`, `_photoabsorb` | The component samplers' outgoing photons, charged kinetic energies, terminal binding and signed transition defect enter the derived partition once. Every pair reserves its rest energy once and hands off its positron, even below the kinetic cutoff. |
| `_Queue.photon`, `_Queue.particle` | Daughter weights are unchanged; photon and electron cutoff energies deposit locally. Positrons are not treated as ordinary sub-cutoff electrons. |
| `track_photon_histories` | Generation-cap energy is unresolved. The loop processes each pending row once and creates new Philox streams from lineage keys for later flights. This differs from continuing a scattered photon's previous generator state in the independent construction, but is distributionally equivalent: each subsequent flight still receives fresh independent uniforms. |
| `photon_stream_key`, `daughter_stream_key` | Keys depend on seed/source ids and daughter ordinals, not queue position. Finite 64-bit hashing cannot guarantee collision-free streams over arbitrary trees; no statistical RNG-independence certification is claimed here. |
| `_assemble`, `ledger_residual_eV` | Sorted unique history ids label rows; incident and terminal lists are reduced with accurate summation before the residual is formed. The terminal terms match the telescoping identity. |
| `detected_spectrum` | The intended denominators match, but the weighted histogram can cancel a positive bin against a much larger weight in a different bin. This fails the bin-local ordinary-hit sum below. |

## Exact discrepancy: cancellation between independent energy bins

The first divergent operation is the weighted `np.histogram` reduction in
`detected_spectrum`. For explicitly supplied bin edges its cumulative weighted
reduction can round a small weight away next to a large one and subtract
identical cumulative sums. The resulting loss occurs between different bins;
roundoff in a large bin does not justify losing an independently representable
small-bin sum.

Two forward photons in vacuum, at 20 and 40 keV with weights $10^{20}$ and
one, both hit the ideal aperture. With two primaries and bin edges at 10, 30
and 50 keV, direct bin-local arithmetic gives

$$
(f_0,f_1)=\left(\frac{10^{20}}{2\times20000},
                   \frac{1}{2\times20000}\right)
          =(2.5\times10^{15},2.5\times10^{-5})\ \mathrm{eV}^{-1}.
$$

The actual result at `3f19d652` is $(2.5\times10^{15},0)$ eV$^{-1}$.
Both detected flags and retained exit weights are correct, so the first
lost contribution is in the histogram, not source transport or the aperture.
This is a 100% relative error in the second bin for finite supported inputs.

Reproducer, with no production helper in the independent expected result:

```python
import numpy as np
from pyrite.montecarlo.transport.photon_histories import (
    PhotonAperture,
    detected_spectrum,
    track_photon_histories,
)

histories = track_photon_histories(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    [2e4, 4e4],
    [1e20, 1.0],
    [7, 9],
    [0, 0],
    [],
    seed=341,
    photon_cutoff_eV=1e3,
    electron_cutoff_eV=1e3,
    aperture=PhotonAperture(
        (0.0, 0.0, 10.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        1.0,
        1.0,
    ),
)
assert histories.exits["detected"].all()
assert histories.exits["weight"].tolist() == [1e20, 1.0]
np.testing.assert_array_equal(
    detected_spectrum(histories, [1e4, 3e4, 5e4], 2),
    [2.5e15, 2.5e-5],
)
```

A temporary test ran through
`UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test
/tmp/test_photon_queue_independent.py --durations=0`.
It failed at the spectrum assertion after the hit and weight checks passed
(one test, 0.77 seconds). No code or regression tests were changed in this
verification context. A repair should accumulate each energy bin without
subtracting cumulative weights from other bins, preserving edge ownership
and the single primary/bin-width normalization.

## Verifier verdict

- **Claim:** `photon-history-queue` —
  `montecarlo/transport/photon_histories.py::detected_spectrum` and the queue
  owners above — PENELOPE-2024 §1.4 detailed histories and ordinary weighted
  aperture counting; component energy partitions reused.
- **Filters:** units pass; limits fail (two independent vacuum energy bins
  with unequal finite weights); signs/conventions pass outside cancellation.
- **Re-derivation:** differs — the second bin receives zero instead of its
  independently representable positive weighted count.
- **Verdict:** `discrepancy`.
- **Write-up:** `docs/validation/beam-transport/photon-history-queue.md`.
- **Suggested ledger change:** set `photon-history-queue` to `discrepancy`
  with this reproducer and report link. The task owner applies it; no
  independent full-track or RNG-statistical validation is implied.

The existing owning module still passes **22 tests** in 4.42 seconds, including
its original spectrum-normalization test. That regression set does not expose
the cross-bin dynamic-range failure. The docs and ledger static checks pass
**26 tests**. The source-access limitation above and navigation's separate
review remain explicit; neither is used to excuse the reproducible tally error.

## Fresh independent queue and changed-tally verification (2026-10-23)

### Independence boundary and derivation before implementation inspection

This verifier did not implement or inspect the queue bodies or the prior queue
report before writing this section. Inputs were the ledger rows, AST-extracted
signatures/docstrings, ADR-0020, the component-claim contracts and the cited
sources. The reviewed tree is `c8d9f4df` plus the task owner's uncommitted
bin-tally repair. The complete `photon-history-queue` claim is in scope;
`photon-continuous-source-scoring` is rechecked only for its changed ordinary-hit
bin reduction and category split, not its source proposal or coupled transport.

[ENDF-102 (2023), File 28](https://www.nndc.bnl.gov/endfdocs/ENDF-102-2023.pdf),
§§28.1–28.3, defines vacancy subshells, emitted transition energy and branch
probabilities. Direct access to the cited PENELOPE-2024 PDF failed; its exact
§1.4/§2.4 equation-number audit is unavailable. Conservation and ordinary-hit
integration below are derived directly; the separately ledgered conditional
interaction distributions are inherited assumptions rather than freshly
revalidated data/models in this queue audit.

Represent each pending photon by its position $\mathbf r$ [Å], unit direction,
energy $E$ [eV], positive statistical weight $w$, history, lineage and interaction
generation. A first flight samples an integrated optical depth and conditional
atom/channel, then either interacts at the forward collision distance or escapes
when no material interval can reach that optical depth. An absorbing aperture
with centre $\mathbf c$, normal $\mathbf n$ and orthonormal in-plane axes has
forward intersection

$$
t=\frac{(\mathbf c-\mathbf r)\cdot\mathbf n}{\mathbf d\cdot\mathbf n},\qquad
u=(\mathbf r+t\mathbf d-\mathbf c)\cdot\mathbf e_x,\quad
v=(\mathbf r+t\mathbf d-\mathbf c)\cdot\mathbf e_y.
$$

Detection requires positive denominator, $t>0$, rectangular membership including
edges, and $t<s_{\rm collision}$. An aperture before a material collision removes
the photon from further transport. Parallel/back-facing rays cannot hit. A
missed aperture with no collision escapes. Geometry validity is inherited from
the independently checked navigation claim.

For a Rayleigh event the outgoing photon has unchanged energy. For Compton,
$E=E'+T_e$; book or hand off the electron, then queue the scattered photon.
Photoelectric absorption partitions $E=(E-B_i)+B_i$. The relaxation identity
for the initial vacancy is

$$
B_i=\sum E_\gamma+\sum T_{\rm Auger}+\sum B_{\rm terminal}
       +\sum\delta_t,\qquad
\delta_t=B_{\rm parent}-\sum B_{\rm daughter}-E_{\rm transition}.
$$

The defect must retain its sign. Each emitted photon/electron receives the
parent weight, without dividing by daughter multiplicity. Isotropy follows
from solid-angle measure $d\Omega=d\phi\,d(\cos\theta)$, hence
$\cos\theta=1-2U_1$, $\phi=2\pi U_2$.
Pair production has $E=T_-+T_++2m_ec^2$: the pending positron carries the two
rest masses until a later charged/annihilation stage. Its rest term is booked
once even if its kinetic energy lies below the electron cutoff. Charged rows
are a handoff boundary, not deposited kinetic energy as well. Photons and
electrons at/below their respective cutoffs deposit their kinetic/photon
energy; terminal binding and transition defect remain separate.

At a generation cap, every pending descendant's full photon energy is unresolved,
not absorbed or escaped. Summing these local partitions cancels every internal
photon energy and gives, for each history,

$$
\sum_{\rm sources}wE=E_{\rm escaped}+E_{\rm detected}+E_{\rm deposited}
 +E_{\rm terminal\ binding}+E_{\rm defect}+E_{\rm charged\ kinetic}
 +E_{\rm positron\ rest}+E_{\rm unresolved}.
$$

All terms here already include their individual weights and have units eV.
No term is a rate. Independent lineage-keyed random streams must depend on the
root history/event and ordered daughters rather than batch position or queue
schedule, so permutation/chunk replay preserves each history. This is a
reproducibility requirement, not a mathematical assertion that finite uint64
hash keys can never collide for an unlimited set of possible lineages.

For bin edges $k_0<\cdots<k_B$, define $I_b=[k_b,k_{b+1})$ except that the
last interval includes $k_B$. Ordinary-hit counting is the bin-local sum

$$
C_b=\sum_{h\in\mathrm{det}}w_h\mathbf1(E_h\in I_b),\qquad
S_b=\frac{C_b}{N_{\rm primary}(k_{b+1}-k_b)}.
$$

Each hit contributes to exactly one bin or to the outside count; undetected
exits contribute nowhere. The result is aperture-integrated counts per primary
per eV, with no solid-angle denominator and no additional attenuation factor.
A large weight in another bin or outside the band cannot alter this bin's
mathematical count. Direct and redistributed counts use the same sum restricted
to complementary predicates: direct iff generation is zero and coupled round
is zero (round defaults to zero when absent). The outside-band count is the
sum of detected out-of-band weights divided only by primary count.
Empty hits yield zeros, a final-edge hit belongs to the last bin, and doubling
the original primary denominator halves every score. These expressions pass
units, sign, vacuum, cutoff, cap and normalization filters independently.

### Implementation comparison after the fresh derivation

| Owner | Independent comparison |
| --- | --- |
| `PhotonAperture.intersect`, `_flight` | Positive facing and strictly positive plane distance, closed rectangle membership and strict aperture-before-collision comparison match. Detection/escape are exclusive terminal outcomes. Exit positions retain the source/last interaction point by the documented record convention; `local_ang` carries the aperture coordinates. |
| `track_photon_histories` | Each active photon is processed once; newly produced photons enter the next generation. Capped photons enter unresolved energy before flight. Region compositions index layers followed by boxes. No subsequent flight exists after an exit or absorbed parent. |
| `_interact` | Rayleigh retains one outgoing photon; Compton additionally books/hands off its recoil electron. Both pair channels use the inherited pair sampler, book one rest reservoir and hand off the positron. Triplet-as-pair is an explicit model limitation, not full triplet kinematics. |
| `_photoabsorb` | Uses the smaller cutoff for inherited relaxation, queues each radiative transition and books/hands off each nonradiative electron. The terminal binding and signed defect are each booked once. Isotropic directions use the independently derived uniform-cosine/azimuth map. |
| `_Queue.photon`, `_Queue.particle` | Equality belongs to local deposition for photons/electrons. Above-cutoff daughter photons retain parent weight and increment generation. Positrons remain handed off even at zero kinetic energy. Charged kinetic energy and deposited energy are exclusive. |
| `_assemble`, `ledger_residual_eV` | Weighted sources and all eight terminal terms reduce by history using `fsum`; their residual is exactly the derived telescoping partition up to arithmetic roundoff. Multiple distinct source events may share a history. |
| `photon_stream_key`, `daughter_stream_key` | Root keys depend on seed/history/event, daughter keys on parent key/ordinal. New Philox generators depend on lineage rather than processing order. Distributional independence is inherited from the RNG construction, not established by replay tests or guaranteed collision-free over an unlimited uint64 domain. |
| `_weighted_photon_counts`, `detected_spectrum` | Filtering the closed reporting band then searching only internal edges with `side="right"` places an internal-edge hit in its right bin and a final-edge hit in the last bin. `bincount` adds weights within the selected bin. This matches the independent indicator sum and removes subtraction of unrelated cumulative weights. Only detected exits enter; the original-primary and bin-width denominators each appear once. |
| Runner `_score_exits` (bounded changed scope) | Total, direct and redistributed reductions use the same helper, with complementary hit masks and common normalization. Generation-zero/round-positive photons remain redistributed. Outside-band weights are summed independently and divided only by primary count. No source proposal, attenuation or RNG change occurs in the reviewed diff. |

The prior two-bin witness now returns exactly
$(2.5\times10^{15},2.5\times10^{-5})$ eV$^{-1}$ per primary. This repairs the
historical cancellation discrepancy without changing transport or the mathematical
estimator. Bin-local accumulation can still round small contributions in the
same bin, as ordinary binary64 positive summation does; it no longer transfers
that uncertainty into independent bins. Extreme arithmetic overflow is outside
the bounded finite-output checks here.

### Independent numerical evidence

The scratch suite `/tmp/test_queue_fresh_verifier_341.py` passed **7 tests**
through the project runner. Expectations were independently constructed rather
than copied from the queue's reduction helpers:

- A bin-by-bin `math.fsum` indicator oracle checks ten seeded input permutations,
  nonuniform bins, both outer edges, internal edges and adjacent representable
  energies, weights from one to $10^{30}$, undetected exits, direct/Compton-like
  and round-positive categories, independent outside counts and empty hits.
  The original two-bin witness is checked exactly.
- Controlled conditional sampler outputs exercise Rayleigh, Compton,
  photoelectric radiative/Auger daughters and pair handoff through the real
  queue. An independently written history ledger checks shared-history source
  aggregation, unchanged daughter weights, sub-cutoff electron deposition,
  negative relaxation defect, binding remainder, pair rest energy and
  sub-cutoff positron handoff. Both a one-generation cap and ordinary escape
  close the same partition; only the former books remaining photons unresolved.
  These are bookkeeping oracles, not validations of the mocked distributions.
- Direct cutoff-equality and isotropic endpoint probes check the relevant
  conventions. A rectangle probe checks interior, exterior, back-facing,
  parallel and exactly represented corner/adjacent-boundary rays. An initial
  normalized corner vector was correctly rejected: 80-digit arithmetic on
  its represented components puts the crossing at
  $3.000000000000000236\ldots$ Å, just outside the 3 Å edge. The oracle was
  corrected to distinguish exact represented contact from that exterior ray;
  no implementation change was needed.
- Real C/Si/Cu/W queues, 24 weighted source photons per material at 2 MeV,
  50 keV, 14 keV and 2 MeV respectively, close energy within
  $2\times10^{-13}$ of incident energy per history. Three permuted chunks
  reproduce all keyed exit energies, directions and history labels exactly;
  independently merged ledger terms agree within reduction roundoff.

The owning history and continuous-source-scoring modules passed **48 tests**,
including the existing attenuation, channel-frequency, fluorescence,
per-history conservation, pair-rest and stream replay anchors. Their green
result supports the fresh derivation and independent oracles; it is not the
basis of the derivation. Documentation tests are run after this append. The
task owner owns the final combined docs build and positive rendered-math check.

### Scope limits and proposed verdict

This verifies photon-only queue composition of the stated component models;
it does not rederive their atomic tables, conditional angular distributions,
charged transport, source proposal or full electron/positron shower. The
PENELOPE PDF access limitation remains. Physical validity of ignoring charged
feedback at MeV energies is not established by photon-only energy closure.
All geometry statements use the independently repaired navigation contract.

An auxiliary input-validation limitation was observed outside the energy/bin
formula: `_validate_ids` casts integer inputs to signed int64 without first
rejecting unsigned values above its maximum. Such inputs can change labels
(e.g. uint64 maximum becomes negative one). The numerical queue verification
uses nonnegative signed-int64-compatible history/event identifiers; this
observation does not change the energy partitions or repaired bin sum, but
callers should not infer a verified unrestricted-uint64 identifier contract.

- **Claim:** `photon-history-queue` — all ledger queue owners, including the
  repaired weighted-count helper — analog histories and secondary partition,
  ENDF-102 File 28 accounting, ordinary weighted aperture integration.
- **Filters:** units pass; limits pass; signs/conventions pass within the
  stated photon-only component assumptions and representable numerical domain.
- **Re-derivation:** matches — all local energy partitions and terminal terms
  telescope correctly; repaired bin counts implement the indicator sum.
- **Verdict:** `rederived` for `c8d9f4df` plus the reviewed bin-tally repair,
  with the identifier-range limitation above.
- **Write-up:** `docs/validation/beam-transport/photon-history-queue.md`.
- **Suggested ledger change:** owner changes `photon-history-queue` from
  `discrepancy` to `rederived`, retains the historical witness and links this
  fresh verification. Never `signed-off`.

For `photon-continuous-source-scoring`, the **bounded changed-tally verdict is
`rederived`**: total/direct/redistributed bin sums, edge convention, original
primary normalization and outside-band count match the independent expression.
Retain its existing claim status and record this bounded repair re-check; no
new independent verdict on the unchanged source proposal or coupled stage is
asserted here.

### Final identifier-boundary re-check

After the preceding observation, the task owner added an explicit signed-int64
representability guard for both history and event arrays before conversion,
and documented that accepted domain in the tracker. The verifier inspected this
guard independently: converting the maximum to a Python integer before comparison
avoids mixed signed/unsigned comparison ambiguity, so values above $2^{63}-1$
raise before casting. The preceding unsigned-wrap observation is historical
for the pre-guard tree, not a remaining discrepancy in the repaired tree.

Public independent probes check both identifier columns at $2^{63}$ and
$2^{64}-1$ (four refusals), and preserve $2^{63}-1$ exactly in both ledger and
exit labels. The boundary energy ledger also closes. These five probes and
the seven independent queue/tally tests pass together: **12 passed**. The
energy/bin derivation and the bounded scoring verdict remain unchanged.

**Final verdict:** `photon-history-queue` is `rederived` for `c8d9f4df` plus
the reviewed bin-tally and identifier-range repairs; the bounded changed
`photon-continuous-source-scoring` tally is `rederived`. No remaining
in-scope discrepancy was found. The source/model, finite-RNG and numerical
limits recorded above still apply. The task owner applies ledger changes
and owns the final build/render check; no human sign-off is asserted.
