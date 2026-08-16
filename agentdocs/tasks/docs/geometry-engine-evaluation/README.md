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
output is a research write-up plus an explicit ADR outcome. Reaffirming ADR-0008
with better-documented evidence is a legitimate and likely result; adopting an
engine requires a superseding ADR and a separate implementation task that does
not exist yet.

Nothing in this task changes `src/`.

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
3. `TODO.md` follow-ups on `main` for whatever the survey recommends —
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
  (CUDA/ROCm), and Intel dpnp/SYCL. A CUDA-only engine narrows that. State the
  cost explicitly rather than assuming the Intel path is expendable.
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
- **Scientific driver.** Which concrete experiment or user need requires
  non-slab target geometry? If none can be named, the survey says so and the
  answer is candidate 8.

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
4. Screen the candidate table. For each: current status, license, packaging,
   device story, and whether it can satisfy ADR-0008's flat-region-table bar.
   Use Context7 for library facts; cite primary sources for the rest. Do not
   assert version-specific capability from memory.
5. Fill the criteria matrix. Mark unknowns as unknown rather than guessing.
6. Quantify the baseline: what would candidate 1 (in-house region table) cost,
   in the same units as the external options? An external engine that does not
   beat the in-house table on the axes above is not worth its dependency.
7. Write the recommendation, separately for A, B, and C, with the scientific
   driver question answered explicitly.
8. Draft the ADR outcome (supersede or reaffirm-with-amendment) and update
   `docs/adr/index.md` and `docs/research/index.md`.
9. Propose `TODO.md` follow-ups for `main`; do not create implementation tasks.
10. Run `uv run pyrite-dev docs`.

## Decisions and open questions

Decided at triage:

- This task produces documentation and an ADR outcome only. No `src/` change, no
  prototype merged to a branch that touches transport.
- The three-consumer split is mandatory structure, not a stylistic choice.
  ADR-0008 already draws the A/B line; the survey must respect it.
- Reaffirming ADR-0008 is an acceptable and possibly correct outcome. The task
  is not scoped as "choose an engine".

Open — **user input needed before any slice can be called `one-shot`**:

1. **Is reversing ADR-0008 actually on the table**, or is the deliverable
   decision-support that may well reaffirm it? This changes the write-up's
   framing and the ADR deliverable.
2. **Is there a concrete scientific driver?** Which target geometry does a real
   planned measurement need that `Slab`/`Stack` cannot express? Without one, the
   survey can only conclude "no change, revisit when a driver exists".
3. **Is the Intel dpnp/SYCL backend expendable?** Several credible candidates are
   CUDA-only.
4. **Is a bounded prototype spike authorized?** If yes, on what hardware — a
   local spike is fine at NumPy scale, but any GPU timing must go through
   `pyrite remote` per the repo rule, and that needs explicit approval.
5. **Does consumer B (instrument geometry) get folded in here or stay with
   `feature/positioned-photon-filters` / P3 item 3?** Recommend: evaluate here,
   implement there.

## Delegation

- Owner: `lead-task`. The work is a judgement call against an accepted ADR that
  constrains every later geometry proposal; it is not a mechanical slice.
- Required skills: `repo-orientation`, `documentation-maintenance`,
  `scientific-library`. `performance` if a spike is authorized;
  `remote-gpu-jobs` if any GPU measurement is authorized.
- Slices:
  - S1 — seam inventory and consumer split (steps 1-3). Self-contained and
    evidence-driven, but not `one-shot`: it feeds the framing that open
    question 1 controls.
  - S2 — candidate screening and criteria matrix (steps 4-6). Not `one-shot`;
    open questions 2 and 3 change which candidates survive.
  - S3 — recommendation and ADR outcome (steps 7-10). Requires all open
    questions resolved.
- No slice is `one-shot` while questions 1-3 are open. Do not label one.

## Acceptance checks

- The write-up states ADR-0008's current decision and binding constraint
  correctly, and every claim about it is traceable to the ADR or the RFC.
- Consumers A, B, and C are answered separately, each with the specific call
  sites that would change.
- Every candidate in the table appears with a verdict and a reason; additions and
  drops are recorded with justification.
- Every library capability claim carries a source; no version-specific behavior
  is asserted from memory.
- The criteria matrix is complete, with unknowns marked as unknown.
- The in-house region table (candidate 1) is costed on the same axes as the
  external options.
- Coherent-emission fitness and identity/determinism are addressed for every
  candidate that reaches consumer A, not just the device story.
- The ADR outcome exists as a file: a superseding ADR or a dated amendment on
  ADR-0008. `docs/adr/index.md` and `docs/research/index.md` are updated.
- `uv run pyrite-dev docs` passes. No `src/` diff.
