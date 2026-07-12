# Parameter-space scanning: helping a user understand and prioritize the results

The present scan can generate far more valid spectra than a user can reasonably
inspect. The immediate problem is therefore not primarily how to compute fewer
points. It is how to turn a completed scan into a small number of understandable
views that answer:

1. Which parameters matter most?
2. Where are the promising regions?
3. What trade-offs distinguish one promising geometry from another?
4. Is a candidate a robust operating region, or a single fragile optimum?
5. Which spectra are actually worth opening and inspecting?

**Recommendation:** retain the current regular grids while they remain affordable,
and add an analysis hierarchy consisting of a coverage summary, parameter-effect
views, trade-off views, robustness checks, and a diverse shortlist. Evaluate sparse
sampling methods retrospectively against completed grids before changing how scans
are generated. The validation app should remain a physics-validation tool rather
than becoming the parameter-search interface.

This review is a design recommendation only. It does not change the physics or the
scan pipeline.

## 1. The user's problem

The current analysis offers useful individual views: polar/azimuth heatmaps,
one-dimensional metric scans, spectrum comparisons, and ranked tables of the top
geometries. These are good drill-down tools. They do not yet provide a coherent
answer to the higher-level question, "What did the scan teach us?"

A top-20 table alone is not enough. Its rows can be twenty nearly identical points
from one narrow region, and its ordering depends on one chosen score. A heatmap is
also not enough once thickness, energy, detector placement, and crystal miscut vary:
each heatmap hides or fixes the other dimensions. The user must mentally combine a
large stack of conditional views.

The analysis should therefore separate three tasks:

- **Overview:** summarize the whole space and identify influential parameters and
  promising regions.
- **Trade-off:** show why several regions are worth considering, rather than forcing
  them into one opaque ranking.
- **Drill-down:** open spectra and detector responses only for a short, deliberately
  varied set of candidates.

This prioritizes human attention. Reducing simulation count is a later optimization
if runtime becomes the limiting factor.

## 2. A physical mental model of the parameters

The scan parameters should be described by what they physically change, not only by
their field names.

| Parameter | Physical meaning | Current status | Recommended user-facing question |
| --- | --- | --- | --- |
| Polar sample tilt | Direction of the physical slab normal relative to the fixed beam and detector, tilted in a chosen plane | Swept as `tilt_deg` | "At what sample tilt does useful emission appear?" |
| Azimuthal sample tilt | Direction around the beam in which the slab is tilted | Swept as `tilt_azim_deg` | "Which rotation around the beam gives the best view of the reflection?" |
| Beam energy | Incident electron kinetic energy; changes transport and the emitted spectrum | Swept as `energy_keV` | "Which beam energy gives the desired line with acceptable background?" |
| Detector placement | Direction of the detector relative to the beam, currently represented chiefly by `theta_obs_deg`; size/distance also affect acceptance | Fixed within a `Sweep` | "Where should the detector be placed?" |
| Crystal thickness | Electron path length and photon escape depth | Swept as `thickness_ang` | "How thick can the sample be before scattering or absorption outweighs additional emission?" |
| Reciprocal-lattice miscut | Fixed angular offset between the crystal lattice (and therefore the selected reciprocal vector **g**) and the physical slab normal **n** | Physics hook exists as `recip_miscut_rad`, but is not exposed by `Sweep` or current grids | "What if the diffracting planes are not parallel to the cut surface?" |

### 2.1 Sample tilt and reciprocal-vector miscut are different

The physical slab normal **n** controls electron transport through the slab and the
photon escape path. The reciprocal vector **g** is normal to the diffracting crystal
planes and controls the coherent-emission condition.

Current scans assume **g** is parallel to **n** for the reflections of interest. In
that symmetric case, tilting the sample moves both together. This corresponds to a
crystal cut whose surface is parallel to the diffracting planes.

For an asymmetric cut or intentional miscut, the selected **g** and **n** have a
fixed offset. Tilting the mounted sample still rotates the physical slab, beam
geometry, and detector geometry. The miscut then applies an additional rotation to
the reciprocal lattice only, leaving **n** fixed. The existing hook rotates every
reciprocal vector together, as a real crystal cut would; it does not let each
reflection acquire an unrelated offset. This is already the meaning of the dormant
`recip_miscut_rad` hook in `montecarlo.geometry._orientation_R` and
`montecarlo.spectrum.mc_spectrum`.

A general three-dimensional miscut needs two numbers:

- its angular magnitude; and
- the in-plane direction toward which **g** is displaced.

It is reasonable to count miscut as a single sixth scan parameter only when its
direction is fixed, for example when the miscut is constrained to the beam-detector
scattering plane or is fixed by a known wafer cut. If both magnitude and direction
are unknown, the proposed six-dimensional scan is actually seven-dimensional.
The analysis must label this choice rather than silently assuming a direction.

### 2.2 Detector placement also needs a precise scope

"Detector placement" can mean detector polar angle, detector azimuth, distance,
active area, or all four. The current `Sweep` has a single fixed observation angle
and acceptance-related fields. Before this becomes a swept dimension, the study
should state which geometric degree of freedom is being optimized. The first useful
version should sweep central detector polar angle while holding detector size,
distance, and azimuth fixed. More detector degrees of freedom can be added only if a
specific experimental decision requires them.

## 3. What the repository already provides

The existing implementation is a strong base for analysis-first improvements:

- `Sweep` and `build_cases` generate a regular Cartesian product of thickness,
  energy, polar tilt, and azimuth.
- Most material grids contain 540 cases; thickness-heavy grids contain 2,160.
- Every record retains its case parameters, so analysis can reconstruct the scanned
  coordinates from a checkpoint.
- `line_metrics` already computes peak flux, integrated coherent flux, dominant-line
  flux and energy, background-related ratios, and a line-quality score.
- `selection_score` and `top_geometries` already produce quality-aware rankings.
- The renderer-neutral frame builders already reduce multi-dimensional results into
  one-dimensional scans and two-dimensional heatmaps.
- The analysis app already supports heatmap-to-spectrum drill-down and several
  pinned-parameter comparisons.

The main gaps are not missing spectra or missing scalar metrics. They are:

1. no whole-scan explanation of parameter influence;
2. no explicit trade-off or "several good answers" view;
3. no robustness measure around a high-scoring point;
4. no guarantee that a shortlist covers distinct parts of the space;
5. no first-class sweep/checkpoint identity for detector angle or reciprocal-vector
   miscut; and
6. no uncertainty estimate showing whether differences exceed Monte Carlo noise.

## 4. What should count as a good result?

There is no universal "best geometry." A geometry can produce a very tall but messy
line, a clean but weak line, or strong coherent emission at an inconvenient photon
energy. The user should select among physically meaningful trade-offs.

The overview should use a small set of quantities with plain-language labels:

| Quantity | Plain-language interpretation | Role |
| --- | --- | --- |
| Coherent flux | Total useful coherent photons in the modeled line window | Primary brightness measure that does not depend on selecting one peak |
| Peak flux density | Height of the brightest spectral point | Useful for detectability, but can favor narrow or noisy spikes |
| Line quality | Whether one dominant, narrow, high-contrast line exists | Reliability/interpretability measure, not brightness |
| Coherent-to-background ratio | How much coherent emission exists relative to bremsstrahlung | Spectral cleanliness measure |
| Dominant line energy | Where the selected line appears | Feasibility constraint set by the detector/application |
| Robustness | How much the result changes under small parameter perturbations | Alignment and manufacturing tolerance measure |

The default overview should begin with coherent flux, line quality, and
coherent-to-background ratio. Dominant-line flux and energy should be shown only
where line quality is adequate, because peak identity can jump between reflections
or become meaningless in a broad, multi-peak spectrum.

Before ranking, the user should be able to apply understandable eligibility rules,
for example:

- line energy lies within the detector's useful window;
- line quality exceeds a stated minimum;
- coherent-to-background ratio exceeds a stated minimum; and
- geometry lies within experimentally reachable tilt, thickness, and detector
  ranges.

These are filters, not preferences. Preferences are applied only after impossible or
unhelpful candidates are removed.

## 5. Recommended analysis hierarchy

### 5.1 Level 1: scan coverage and data quality

Start with a compact "what was scanned" card:

- parameter names, ranges, and number of distinct values;
- total requested, completed, missing, and failed cases;
- fixed parameters that materially affect interpretation;
- the detector model and photon-energy range;
- whether **g** is aligned with **n**, or the miscut magnitude and direction;
- electron counts and seed convention; and
- a warning when a requested axis is absent or contains only one value.

This prevents a user from interpreting a sparse or incomplete checkpoint as a full
study.

### 5.2 Level 2: parameter-effect summary

For each input, show how the outcomes change across its range after summarizing over
the other parameters. Use three complementary summaries:

1. **Best achievable:** the best qualifying outcome found at each value. This answers
   "Could this value ever work?"
2. **Typical:** the median qualifying outcome. This answers "Does this value usually
   work, or only under a special combination?"
3. **Spread:** an interquartile band or similar range. This answers "How strongly does
   this parameter interact with the others?"

These are familiar line-and-band plots rather than a statistical black box. They
should be available for coherent flux, quality, and background ratio. The title must
say when the curve is a best-over-other-parameters envelope, since that curve is not
the response of one fixed geometry.

Add a short influence table based on observable changes:

- range of the median outcome across the parameter;
- range of the best-achievable outcome;
- fraction of qualifying cases at each value; and
- rank stability under bootstrap resampling or Monte Carlo replicates.

Call this **observed influence**, not causal importance. Parameters can appear weak
because the scanned range was too narrow or because their effect occurs only through
an interaction.

### 5.3 Level 3: interaction maps

After the influence summary identifies important axes, show two-dimensional heatmaps
for the most informative pairs. Do not require the user to inspect every possible
pair. Select a small set using both domain knowledge and evidence of interaction.

Good initial pairs are:

- polar tilt × azimuth;
- polar tilt × reciprocal-vector miscut;
- polar tilt × detector angle;
- thickness × beam energy; and
- thickness × polar tilt.

Each heatmap should support three reductions over hidden parameters: best, median,
and qualifying fraction. The existing best-case reduction is useful but optimistic;
median and qualifying-fraction views show whether a bright cell represents a broad
region or a single special combination.

### 5.4 Level 4: trade-off view

Show a scatter plot with one desirable outcome on each axis, initially coherent flux
versus coherent-to-background ratio. Encode line quality by color and optionally
thickness or beam energy by symbol/facet.

Highlight the **Pareto front**. This term means the set of candidates for which no
other candidate is better in every displayed objective. For example, a point belongs
to the front if increasing brightness would require accepting more background, or
reducing background would require accepting less brightness. Points well behind the
front are dominated: another tested geometry is at least as good in all chosen ways.

This does not choose the answer for the user. It removes clearly inferior choices
and makes the remaining trade-offs visible. The interface should explain the idea in
those words rather than assuming the user knows the term "Pareto."

### 5.5 Level 5: a diverse, robust shortlist

Build a shortlist of roughly 6–12 cases, not simply the first 12 rows of a scalar
ranking. It should contain:

- the brightest qualifying case;
- the cleanest qualifying case;
- the best balanced trade-off;
- the most robust case;
- representatives of materially different energy, thickness, or angular regions;
  and
- any reference/validation geometry that the user should recognize.

For each candidate, report how rapidly the metrics change in its immediate scanned
neighborhood. A slightly lower-scoring plateau is usually more useful experimentally
than an isolated maximum that disappears after a fraction of a degree of
misalignment. When no neighboring points exist, label robustness as unknown rather
than good.

Only at this level should the app invite the user to inspect the full intrinsic and
detector-convolved spectra.

## 6. Plain-language guide to candidate sampling methods

The following methods answer the computation question: how can we learn about a
large space without evaluating every combination? They do not by themselves solve
the interpretation problem above.

### 6.1 Full Cartesian grid

Choose a list of values for every parameter and evaluate every combination. This is
the current approach.

**Analogy:** inspect every intersection on a rectangular street grid.

**Strengths:** simple, easy to plot, easy to resume, and provides real neighboring
points for robustness checks. **Weakness:** case count multiplies when a new axis is
added. It remains the preferred ground truth while four-dimensional grids are
affordable, but will scale poorly after detector angle and miscut are exposed.

### 6.2 Fractional factorial design

A full factorial study tests every combination of a few chosen levels, such as low
and high values. A fractional factorial tests a carefully selected subset that lets
one estimate broad main effects and some interactions.

**Analogy:** instead of tasting every combination of six recipe ingredients, choose
a balanced set of recipes designed to reveal which ingredients matter most.

**Strengths:** economical and interpretable for early screening. **Weakness:** it
works best when effects are reasonably smooth and can be summarized by low-order
interactions. Resonant spectra, changing peak identity, and narrow angular optima can
violate that assumption. Use it only as a coarse screening benchmark, not as the
final search strategy.

### 6.3 Latin hypercube sampling

Divide each parameter's range into equal-probability bands and choose points so every
band of every parameter is represented once. The selected coordinates are paired to
spread points throughout the multi-dimensional volume.

**Analogy:** place a small number of survey stations so that every north-south band
and every east-west band is visited, without visiting every intersection.

**Strengths:** good coverage with fewer points than a grid. **Weakness:** adding more
points later does not naturally preserve the same balanced design, and the irregular
points are less convenient for heatmaps and neighborhood robustness. It is a
reasonable comparison method, but not the first recommendation.

### 6.4 Sobol sequence

A Sobol sequence is another way to place points evenly through a multi-dimensional
space. It is a deterministic low-discrepancy sequence: each successive block is
constructed to fill gaps left by earlier points.

**Analogy:** repeatedly place the next survey station in a way that keeps the whole
territory evenly covered, rather than drawing stations randomly.

**Strengths:** it can be extended in stages without discarding earlier runs and
usually covers projections of the space more evenly than ordinary random sampling.
**Weakness:** it still produces irregular points, does not automatically concentrate
on promising regions, and requires explicit mapping of every parameter to a bounded
numeric interval. If sparse scanning becomes necessary, this is the recommended
global exploration method.

"Sobol screening" in this recommendation simply means running an initial, evenly
spread Sobol set and using its results to identify influential parameters and
promising regions. It is not a classifier and does not require the user to interpret
the sequence itself.

### 6.5 Sensitivity analysis

Sensitivity analysis asks how much an output changes when an input changes. A global
sensitivity analysis considers changes across the full scanned range, including
interactions with other parameters.

For this project the first sensitivity display should be the direct, observable
effect summaries in Section 5.2. Formal sensitivity indices can be evaluated later,
but should not replace plots a user can inspect. Sensitivities must be reported
separately for brightness, quality, and background ratio; a parameter can strongly
affect one and weakly affect another.

### 6.6 Pareto front

A Pareto front is not a sampling method. It is a way to summarize multiple goals
without inventing a single set of preference weights. Its role here is to help the
user understand trade-offs after results exist, regardless of how the points were
sampled.

## 7. Recommended staged strategy

### Stage A: improve understanding using existing checkpoints

1. Add the scan-coverage summary.
2. Add best/median/spread parameter-effect frames.
3. Add qualifying-fraction and selected interaction heatmaps.
4. Add the explained trade-off/Pareto view.
5. Replace the undifferentiated top-N list with a diverse, robustness-aware
   shortlist.
6. Add a small Monte Carlo repeat study at representative low, middle, and high
   scoring cases so the analysis can distinguish simulation variation from real
   parameter effects.

This stage addresses the user's stated bottleneck without changing the scan design.

### Stage B: evaluate sparse designs retrospectively

Use one or more completed regular-grid checkpoints as known truth. Hide most points,
then compare how well random, Latin-hypercube-like, Sobol-like, coarse factorial, and
coarse-to-fine selections recover what the full grid reveals.

Measure:

- **best-case regret:** how far the best sampled candidate falls below the true best;
- **trade-off coverage:** how much of the full-grid Pareto front is recovered;
- **region coverage:** whether distinct promising angular/thickness regions are
  represented;
- **influence stability:** whether the same parameters appear important; and
- **shortlist stability:** whether the recommended candidates remain similar when
  the sample or Monte Carlo seed changes.

This is more informative than choosing a method from general statistical reputation.
It tests the methods against this model's resonances, discontinuous peak identity,
and actual parameter ranges without paying for new simulations.

### Stage C: expose new axes safely

Before detector angle or miscut is swept:

1. define its units, allowed range, and coordinate convention;
2. decide whether reciprocal miscut direction is fixed or swept;
3. include all varying geometry in the stable case identity and checkpoint key;
4. derive simulation seeds from the parameter tuple plus a replicate identifier,
   rather than from iteration order;
5. add the new fields to result summaries and filters; and
6. validate the aligned limit: zero miscut must reproduce the current `g || n`
   result.

The zero-miscut check is essential, but this design does not introduce a new physics
claim or require a physics-validation-ledger change until the miscut is actually
wired into production sweeps.

### Stage D: use sparse global exploration only if needed

If the expanded grid is too expensive, use:

1. a small fixed set of replicated anchor cases to estimate Monte Carlo variation;
2. an extensible Sobol sample over bounded continuous parameters;
3. the same analysis hierarchy used for regular grids;
4. denser local grids around several promising trade-off regions; and
5. high-electron-count confirmation of the final shortlist.

Do not refine only around the single highest scalar score. Preserve several regions
with different brightness/background/robustness trade-offs so a narrow resonance or
Monte Carlo fluctuation cannot capture the whole follow-up budget.

## 8. Architecture implications for follow-on work

The analysis additions should remain renderer-neutral, following the existing
`plots._frames` pattern:

- pure functions turn result records into tidy data frames for coverage, effects,
  interactions, trade-offs, and shortlists;
- matplotlib and Altair render from those shared frames;
- the analysis app composes the views and owns only controls/presentation; and
- no sampling or ranking logic should live in the validation app.

Sparse designs cannot be represented by placing independently sampled coordinate
arrays into the current `Sweep`: `build_cases` would take their Cartesian product.
If Stage D is chosen, introduce a separate sampled-design-to-cases boundary rather
than overloading `Sweep`. Regular grids and irregular sampled designs should both
produce the same case schema and feed the same runner/results/analysis pipeline.

Checkpoint identity also needs attention. Current configuration names encode
material, thickness, polar tilt, and azimuth, while energy is the nested record key.
Detector angle and miscut would collide if varied without becoming part of a
canonical case identity. Address that before exposing either axis.

## 9. What not to do

- Do not choose Latin hypercube, Sobol, or fractional factorial merely because the
  names are associated with high-dimensional problems.
- Do not reduce every scientific preference to `quality × peak flux` and present the
  resulting order as objective truth.
- Do not show formal feature-importance numbers without direct effect plots and a
  statement of the scanned ranges.
- Do not treat an isolated maximum as robust.
- Do not mix fixed literature-validation anchors with exploratory optimization.
- Do not call a scalar miscut "general" unless its direction is fixed and labeled.
- Do not add new sweep axes until case identity, seeds, filtering, and summaries can
  distinguish them.

## 10. Decision and success criteria

The analysis-first direction is successful when a user unfamiliar with statistical
design methods can, without opening hundreds of spectra:

1. describe what parameter ranges were scanned;
2. identify which parameters have visibly large effects on each outcome;
3. locate several promising regions and understand their trade-offs;
4. distinguish robust plateaus from narrow optima;
5. select a small set of spectra for detailed inspection; and
6. understand whether reported differences are larger than Monte Carlo variation.

Only after those goals are met should the project decide whether sparse sampling is
needed. If full-grid computation remains affordable, keep it: regular neighborhoods
are valuable for explanation and robustness. If the detector-angle/miscut expansion
makes full grids impractical, adopt staged Sobol exploration plus local regular-grid
refinement only after the retrospective comparison demonstrates acceptable recovery
of the full-grid conclusions.
