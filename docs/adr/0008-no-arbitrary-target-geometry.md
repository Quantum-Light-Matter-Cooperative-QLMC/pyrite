# 0008 — No arbitrary target geometry

- **Status:** Accepted
- **Date:** 2026-08-13
- **Amended:** 2026-08-14 — interoperability remains the downstream strategy; MCPL is no longer preselected or required
- **Context source:** {ref}`core architecture RFC, Non-goals <sec-core-arch-nongoals>`

## Amendment: do not preselect an interchange format

The original decision named required MCPL export as the interoperability escape hatch. Implementation review found that PyRITE stores energy-binned spectral tallies, not photon phase-space records. A valid MCPL particle requires semantics for position, direction, energy, time, polarization, and statistical weight; inventing those fields during export would turn a format adapter into an unreviewed source model.

Interoperability remains the sanctioned downstream strategy, but no format is preselected or required. Work must start from a concrete consumer and define a format-neutral `PhotonSource` contract, including reference surface, joint phase-space distribution, normalization, provenance, and closure tests. Only then should the project select and test an adapter. MCPL remains a strong candidate for Geant4, OpenMC, MCNP/PHITS, and McStas/McXtrace workflows.

This does not weaken the geometry non-goal. An interchange adapter can hand a defined photon source to another transport code; it cannot make arbitrary target geometry affect electron transport and coherent photon production inside PyRITE. The MCPL-specific wording below records the original decision and is superseded by this amendment.

## Context

Consolidating the geometry surface behind `Target` raises the obvious next question: is this the first step toward a general geometry system? It is not, and the answer is a decision in its own right — it constrains every later proposal for meshes, CSG, or a navigator protocol, and it produces no code, so without a record it would be re-litigated.

The physics of interest is coherent emission from crystalline slabs and layer stacks. `Slab` and `Stack` (with `Layer`, `Footprint`, and `BlazedGrooves`) cover the current and foreseeable scientific programme.

## Decision

PyRITE will not gain a general geometry system: no constructive solid geometry, no surface or region algebra, no navigator protocol, no imported meshes (STL, STEP, GDML), and no conditionally available geometry escape hatch.

`Target` is a **closed variant set**. Adding a shape is an explicit, reviewed change to that set, not an extension point.

The binding constraint is the GPU. The transport hot path is a CUDA kernel maintained as one algorithm with its CPU twin. General geometry requires per-step boundary queries against an arbitrary region set, which on a device means an acceleration structure and a ray-tracing traversal. The AdePT/Celeritas experience is the reference point: porting realistic detector geometry to GPUs made geometry the bottleneck through thread divergence and register pressure, and the response was a [GPU-friendly surface model](https://www.epj-conferences.org/articles/epjconf/abs/2025/22/epjconf_chep2025_01207/epjconf_chep2025_01207.html) in VecGeom — a substantial project in its own right.

This restriction is specifically about target/electron-transport geometry. A bounded downstream photon instrument may contain reviewed analytic plane or box elements after emission, such as a finite attenuation plate and planar pixel detector. Those objects remain outside `Target`, use closed concrete types, and perform no boundary queries in the electron CPU/CUDA navigator. They do not create a general geometry protocol or weaken this decision.

Interoperability, not generality, was the sanctioned answer to a user who needs arbitrary geometry. The original decision specified required MCPL export so a code that has already solved general geometry could consume PyRITE's emitted photons. The 2026-08-14 amendment retains interoperability but defers the format until PyRITE has a scientifically defined photon-source boundary and a concrete consumer.

### The seam, named but not built

If the constraint ever changes, the transport core needs exactly three operations:

- `locate(r) -> region`
- `distance_to_boundary(r, v) -> (s, region)`
- `escape_path(r, v) -> per-region path lengths`

Three implementations of that interface already exist in specialized form — layer crossing by depth, prism face intersection, and groove facet intersection. Any future general implementation **must** be a flat, bounded-depth, device-representable region table interpreted by one loop on both host and device — never a polymorphic object graph. Naming the seam is the whole of the forward plan; nothing in the current design builds toward it.

## Consequences

- Geometry validity stays enumerable, so `Target.__post_init__` can reject an invalid configuration at construction time in the user's script.
- `Target.lower()` can stay a total function over a closed set with no dispatch protocol or plugin registry.
- The transport kernels keep their specialized, divergence-free boundary logic.
- Users needing arbitrary downstream geometry are routed to future interoperability work driven by a concrete consumer. Import of STL/STEP as PyRITE simulation geometry remains out of scope under this ADR.
- Reversing this decision requires a superseding ADR, and the seam above is the interface it would have to implement.
