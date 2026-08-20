# General 3D geometry engine evaluation

Branch: `docs/geometry-engine-evaluation`

## Problem and scope

The user asked for an evaluation of possible general 3D geometry engines for
this repository. That request lands directly on
[ADR-0008](../../../../docs/adr/0008-no-arbitrary-target-geometry.md), which
records "no general geometry system" as **Accepted**: no CSG, no surface or
region algebra, no navigator protocol, no imported meshes (STL, STEP, GDML), no
conditional escape hatch. `Target` is a closed variant set (`Slab | Stack`).

This task is therefore **a decision-support survey, not an implementation**. Its
output is a research write-up plus an explicit ADR outcome. Per the 2026-08-15
review, **reversing ADR-0008 is permitted**, so a superseding ADR-0011 is a live
outcome alongside reaffirmation; either way the implementation is a separate
task that does not exist yet.

Nothing in this task changes `src/` except an optional, explicitly bounded
throwaway benchmark under `scratch/` (see "Authorized spike" below), which is
never merged.

### The three consumers must not be conflated

"General 3D geometry engine" means three different things in this repository,
with three different risk profiles. The evaluation must answer them separately
and say so explicitly; a single verdict would be wrong for at least two of them.

**A. Target / electron-transport geometry (ADR-0008 non-goal).** The binding
constraint is the GPU. The transport hot path is one algorithm maintained as a
CPU/Numba twin and a `cupyx.jit` device core
(`src/pyrite/montecarlo/transport.py`,
`src/pyrite/montecarlo/transport_jit_kernel.py`,
[GPU transport design](../../../../docs/repo-design/compute/gpu-transport-rawkernel.md)).
Three specialized implementations of ADR-0008's named seam already exist:

- layer crossing by depth — `materials/attenuation.py:107` (`_layer_dz`),
  `:117` (`_layer_path_length`), plus the kernel-side layer search at
  `montecarlo/transport_jit_kernel.py:265`;
- prism face intersection — `montecarlo/transport.py:797`
  (`_first_prism_exit_scalar`), called from every CPU core variant
  (`:968`, `:1209`, `:1501`, `:1615`, `:1893`, `:2115`) and *inlined by hand*
  into the device core at `montecarlo/transport_jit_kernel.py:308`;
- groove facet intersection — `montecarlo/groove.py:259` (`first_surface_event`),
  `:303` (`escape_distance_ang`), `:87` (`surface_depth_ang`).

ADR-0008 states any general replacement **must** be a flat, bounded-depth,
device-representable region table interpreted by one loop on host and device —
never a polymorphic object graph. That sentence is the acceptance bar for any
candidate in this consumer class.

**B. Post-emission photon instrument geometry (already sanctioned, bounded).**
ADR-0008 explicitly permits reviewed analytic plane/box elements after emission,
outside `Target` and outside the electron navigator. `src/pyrite/instrument/`
landed under that clause: `geometry.py:86` (`ray_box_path_lengths`), `:153`
(`filter_path_lengths`), `model.py` (`PlanarPose`, `PixelGrid`, `FilterPlate`,
`PlanarDetector`, `PixelScorer`). Generalizing *this* surface — collimators,
apertures, non-rectangular foils, multi-chip detector mounts — needs no ADR
reversal, only a judgement about whether a library beats more closed types.
See `agentdocs/tasks/feature/positioned-photon-filters/README.md` and P3 item 3.

**C. Visualization and authoring only.** `plots/plotly/trajectories.py`,
`crystal_lattice.py`, `render.py`, and `apps/trace_app.py`. Zero physics risk; a
mesh library here carries only dependency weight. Must not be used to smuggle a
geometry representation into A.

## Implementation path and likely owners

Deliverables, in order:

1. `docs/research/architecture/geometry-engines.md` — the survey. New
   `architecture/` subdirectory under `docs/research/`; add a toctree caption to
   `docs/research/index.md`. Classification per
   [documentation policy](../../../../docs/repo-design/documentation.md):
   this is exploratory research, not current behavior, so it belongs in
   `docs/research/`, not `docs/physics/` or `docs/repo-design/`.
2. An ADR outcome — either `docs/adr/0011-*.md` superseding ADR-0008, or a dated
   amendment section on ADR-0008 recording the re-examination and its
   reaffirmation. Update `docs/adr/index.md` either way. If the verdict is
   "reaffirm", the ADR gains the concrete engine-by-engine evidence it currently
   lacks (it cites only the AdePT/VecGeom experience).
3. GitHub issue follow-ups on `main` for whatever the survey recommends —
   implementation tasks are **not** created by this task.

No `src/` owner. Read-only touchpoints for evidence gathering: `montecarlo/
transport.py`, `transport_jit_kernel.py`, `groove.py`, `_backend.py`,
`materials/attenuation.py`, `campaign/geometry.py`, `instrument/`,
`campaign/profiles.py` (identity digests).

## Candidates to evaluate

Screening list. The survey may add or drop entries with a recorded reason; it
may not silently omit one.

| # | Candidate | Class | Primary question |
|---|-----------|-------|------------------|
| 1 | Hand-rolled quadric/CSG region table in `cupyx.jit` + Numba | A | The ADR-named seam, built in-house. The baseline every external option must beat. |
| 2 | NVIDIA Warp | A | Python→CPU/CUDA kernel compiler with in-kernel `wp.mesh_query_ray` / `wp.bvh_query_ray`. Only serious candidate that is both Python-native and device-resident. |
| 3 | Celeritas ORANGE / VecGeom surface model | A | The reference GPU-geometry implementations. C++; evaluate as design reference and as an external transport target, not as an embeddable dependency. |
| 4 | Geant4 (`geant4_pybind`) / pyg4ometry | A, interop | Full engine. Adopting it means handing off transport, which PyRITE cannot do for coherent emission. Evaluate for GDML authoring/export only. |
| 5 | OpenMC CSG, DAGMC + MOAB (+ Embree) | A (CPU), interop | Established CAD/CSG navigation with real provenance. CPU-only for our purposes. |
| 6 | Embree (CPU) / OptiX (CUDA C++) | A, B | Raw acceleration structures with no physics semantics. What would still have to be written by us. |
| 7 | trimesh (+embreex), PyVista/VTK, CadQuery/build123d/OCP, gmsh | B, C | Mesh/CAD authoring, STEP/STL import, CPU ray queries, rendering. |
| 8 | Status quo: reaffirm ADR-0008, invest in the interop path instead | — | The `PhotonSource` contract from the ADR-0008 amendment and the Long-term-plans "Geometry interoperability" bullet. |

## Evaluation criteria

Every candidate scored against the same axes; the matrix goes in the write-up.

- **Device story.** Can it execute inside the existing `cupyx.jit` kernel, or
  does it force a rewrite of the transport core? What happens to the Numba CPU
  twin, and does the "one algorithm, two implementations" invariant survive?
  Warp specifically: CPU kernel launches run serially per the vendor docs, which
  must be measured against the current Numba path before it is called a
  replacement.
- **Backend portability.** `montecarlo/_backend.py` supports NumPy, CuPy
  (CUDA/ROCm), and Intel dpnp/SYCL. CUDA-only is accepted per the review, so
  this axis is no longer a veto — but each candidate must still say what happens
  to the dpnp/SYCL path and to local (Intel) development. See "Backend reality
  check".
- **Determinism and identity.** Per-electron RNG stream reproducibility
  (`stream_keys`, `_splitmix64`), and whether geometry changes perturb
  `case_content_key` / `dataset_identity` / checkpoint stems
  (`campaign/profiles.py`). An engine that makes results non-bitwise-reproducible
  across a version bump is disqualifying for a checkpointed campaign tool.
- **Coherent-emission fitness.** This is the axis generic geometry surveys miss.
  PyRITE sums segment amplitudes with phase; geometry must deliver exact path
  lengths and positions, not merely correct region membership. Tessellated CAD
  introduces faceting error in the path integral that feeds emission phase and
  self-absorption. Quantify or reject.
- **Validation burden.** New geometry needs source equations, limiting cases,
  `Validation: <id>` markers, and ledger rows per the repo physics contract. How
  much of the engine's behavior must PyRITE independently verify?
- **Packaging and licensing.** pip wheels, redistribution terms, offline
  install, wheel size, optional-extra fit, GPU-vendor lock, Python version
  support, release cadence, and who maintains it.
- **Migration cost.** `Target` closed set and its `__post_init__` validity
  rejection, `target_from_flat` / `target_flat_fields` / `target_replace`, the
  flat geometry vocabulary in catalog TOML and profile overrides, and the
  golden/regression data that pins current behavior.
- **Driver coverage.** How much of each driver geometry above does the candidate
  actually deliver? Score per driver, not overall. A candidate that navigates
  arbitrary meshes but cannot express a bent crystal's orientation field scores
  low on driver 1 however good its ray tracing is.

## Checklist

1. Read ADR-0008, the core-architecture RFC non-goal section
   (`docs/repo-design/core-architecture-rfc.md:769`), and the GPU transport
   design note. Restate the current decision and its stated binding constraint
   accurately before evaluating anything.
2. Inventory the existing seam implementations at the anchors listed above.
   Record what each one actually computes, its device duplication, and what a
   general engine would have to reproduce exactly.
3. Establish the consumer split (A/B/C) with the specific call sites that would
   change for each.
4. Settle the bent-crystal question in "Driver geometries" before scoring: is
   driver 1 a navigation problem, an orientation-field problem, or both? Check
   `docs/research/physics/channeling-radiation-physics.md`,
   `relativistic-electron-transport.md`, and `materials/crystal.py`. Record the
   answer prominently; it conditions the recommendation.
5. Screen the candidate table. For each: current status, license, packaging,
   device story, and whether it can satisfy ADR-0008's flat-region-table bar.
   Use Context7 for library facts; cite primary sources for the rest. Do not
   assert version-specific capability from memory.
6. Fill the criteria matrix, scoring driver coverage per driver. Mark unknowns as
   unknown rather than guessing.
7. Quantify the baseline: what would candidate 1 (in-house region table) cost,
   in the same units as the external options? An external engine that does not
   beat the in-house table on the axes above is not worth its dependency.
8. Run the authorized spike only if the matrix leaves a genuine tie or a load-
   bearing unknown that measurement can close. A spike is evidence, not a
   default step; skip it and say why if the paper comparison already decides.
9. Write the recommendation, separately for A, B, and C, staged against the
   driver list — including the sequencing answer, since driver 1 sits behind
   channeling and relativistic electrons in the backlog.
10. Draft the ADR outcome — a superseding ADR-0011 or a dated amendment on
    ADR-0008 — and update `docs/adr/index.md` and `docs/research/index.md`. If
    ADR-0011 supersedes, set ADR-0008's status to `Superseded by ADR-0011` per
    the ADR index's append-only rule, and carry forward the parts that survive:
    the named seam, the flat-region-table requirement, and the bounded
    post-emission carve-out.
11. Propose GitHub issue follow-ups for `main`; do not create implementation tasks.
12. Run `uv run pyrite-dev docs`.

## Decisions

Decided at triage:

- This task produces documentation and an ADR outcome. No merged `src/` change.
- The three-consumer split is mandatory structure, not a stylistic choice.
  ADR-0008 already draws the A/B line; the survey must respect it.
- Consumer B (instrument geometry) is *evaluated* here and *implemented* under
  `feature/positioned-photon-filters` / P3 item 3, not here.

Resolved by user review, 2026-08-15:

1. **ADR-0008 may be reversed.** A superseding ADR-0011 is an allowed outcome.
   The survey is genuinely open; it is not scoped to reach reaffirmation. It is
   also not scoped to reach adoption — "the engines evaluated do not pay for
   themselves yet" remains a valid finding, and must be reported as such if
   that is where the evidence lands.
2. **No present scientific driver; future support is wanted.** This is the
   awkward case: with no driver, "general" is underdetermined and the criteria
   have nothing to bite on. The survey resolves this by evaluating against the
   **latent drivers already in the backlog** rather than an abstract notion of
   generality. See "Driver geometries" below.
3. **CUDA-only is acceptable** where it is the most straightforward GPU route —
   but the cost must be stated, not assumed away. See "Backend reality check".
4. **A bounded spike is authorized** on the home setup or the lab box. See
   "Authorized spike".

### Driver geometries

The survey scores candidates against these, in priority order, instead of
against unbounded generality. Each is already in GitHub Issues or `docs/`:

1. **Bent crystals** — GitHub issue #14 (P1 "Paused / on hold"), sequenced after
   channeling and relativistic electrons. This is the clearest future
   target-geometry driver in the backlog and the canonical stress test.
2. **Finite/irregular target shapes** — targets that are not an axis-aligned
   rectangular prism. `_first_prism_exit_scalar` assumes exactly that
   (`transport.py:3082`); this is the cheapest possible generalization and the
   honest low bar.
3. **Multi-object scenes** — several materials at arbitrary relative pose, the
   literal ADR-0008 non-goal wording.
4. **Downstream instrument geometry** — collimators, apertures, non-rectangular
   foils, multi-chip detector mounts (consumer B; no ADR reversal needed).

**A finding to establish early, because it may decide the whole question:** a
bent crystal is not primarily a region-navigation problem. It needs a spatially
varying lattice orientation field feeding structure factors, reciprocal-lattice
vectors, and emission phase. No general geometry engine supplies that — it is a
materials/orientation concern, adjacent to `materials/crystal.py` and the
`>user<` "user-defined crystal cuts" inbox item, not to `locate` /
`distance_to_boundary`. If driver 1 is mostly an orientation-field problem, then
adopting a geometry engine buys much less than it appears to, and the survey
must say so plainly rather than letting the engine comparison carry the
conclusion. Verify this against the channeling and relativistic-transport
research notes (`docs/research/physics/`) before scoring candidates.

### Backend reality check

CUDA-only is accepted, with two consequences that go in the write-up:

- `pyproject.toml:37` ships an `intel` extra (`dpnp>=0.20.0`) and
  `montecarlo/_backend.py` supports dpnp/SYCL. A CUDA-only engine either strands
  that path or forces a permanent two-implementation split for target geometry —
  the exact "one algorithm" invariant ADR-0008 leans on. State which.
- **The development box is Intel-only**: `lspci` reports Intel UHD Graphics and
  an Arc A370M, with no `nvidia-smi`. If the "home setup" is this machine, a
  CUDA-only candidate cannot be exercised locally at all, and every GPU
  measurement must go to the lab box. Confirm which machine is meant before
  planning the spike; if the home setup does have an NVIDIA card, record it
  here and this constraint relaxes.

### Authorized spike

Bounded, and throwaway:

- Scope is a **standalone geometry microbenchmark** — ray/region queries at
  representative electron counts and step rates — not a port of the transport
  core and not a physics run. If it grows into a transport port, stop and return
  to the supervisor.
- Lives in ignored `scratch/`; results are transcribed into the write-up. No
  spike code merges to this branch.
- CPU-side spikes run locally. Any CUDA timing goes to the lab box through
  `pyrite remote` per the repo rule; use the `remote-gpu-jobs` skill. Never run
  GPU sweeps locally.
- Numbers must be reported with hardware, driver, and library versions, or they
  are not usable evidence.

## Delegation

- Owner: `lead-task`. The work is a judgement call against an accepted ADR that
  constrains every later geometry proposal; it is not a mechanical slice.
- Required skills: `repo-orientation`, `documentation-maintenance`,
  `scientific-library`; `performance` and `remote-gpu-jobs` for the spike.
- Slices:
  - S1 — seam inventory, consumer split, and the bent-crystal driver question
    (steps 1-4). **`one-shot`**: the inputs are named files at named anchors and
    the review resolved the framing. Deliverable is a draft section, not a
    verdict.
  - S2 — candidate screening and criteria matrix (steps 5-7). Not `one-shot`:
    library facts need Context7 lookups per candidate and the driver-coverage
    scoring depends on S1's answer.
  - S3 — spike, if S2 leaves a decidable unknown (step 8). Separate slice under
    `remote-gpu-jobs`; needs the machine question in "Backend reality check"
    answered first.
  - S4 — recommendation and ADR outcome (steps 9-12). Not `one-shot`: choosing
    between superseding and reaffirming ADR-0008 is the judgement this whole
    task exists to make.

## Acceptance checks

- The write-up states ADR-0008's current decision and binding constraint
  correctly, and every claim about it is traceable to the ADR or the RFC.
- Consumers A, B, and C are answered separately, each with the specific call
  sites that would change.
- Every candidate in the table appears with a verdict and a reason; additions and
  drops are recorded with justification.
- Every library capability claim carries a source; no version-specific behavior
  is asserted from memory.
- The criteria matrix is complete, with unknowns marked as unknown, and driver
  coverage is scored per driver rather than as a single verdict.
- The bent-crystal navigation-versus-orientation-field question is answered
  explicitly, with sources, and its effect on the recommendation is stated.
- Any spike number carries hardware, driver, and library versions; no spike code
  is merged.
- The in-house region table (candidate 1) is costed on the same axes as the
  external options.
- Coherent-emission fitness and identity/determinism are addressed for every
  candidate that reaches consumer A, not just the device story.
- The ADR outcome exists as a file: a superseding ADR-0011 or a dated amendment
  on ADR-0008, with ADR-0008's status updated if superseded.
  `docs/adr/index.md` and `docs/research/index.md` are updated.
- `uv run pyrite-dev docs` passes. No `src/` diff.
