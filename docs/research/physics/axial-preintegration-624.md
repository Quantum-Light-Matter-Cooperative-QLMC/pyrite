# #624 preliminary axial integration: canonical recipe and refusal seams

This is a configuration-only change. **No standard axial run is enabled.**
Issue [#623](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/623)
still lacks an anchored physical matching surface, incident-normalized material
photon source, and calibrated per-case return-flux/depth bound. Periodic FFT
wrapping and finite-grid numerical sinks are *not* physical escape events.

## Proposed public selector

The `Physics` facade takes `channeling_model="axial"` together with a
`channeling_axial` mapping; the matching profile numerics keys have the same
names. Both go through `campaign.axial_spec.resolve_axial_spec` before run
admission. The versioned canonical mapping is JSON-safe, strictly tagged,
and rejects unknown keys and nonfinite/scalar-array values. A minimal proposed
configuration is:

```yaml
channeling_model: axial
channeling_axial:
  axis_uvw: [0, 0, 1]
  zone_max_ang: 1500
  entry:
    version: 1
    kind: gaussian
    waist_ang: 0.5
```

The resolver expands defaults for the 2D impact, tilt offset, rectangular
projected supercell, sampling grid, depth step, seed, ensemble size and
isotropic TDS-plus-electronic research mechanisms. It reduces direct-lattice
`[uvw]` using the positive gcd but retains direction sign. Incoherent mixtures
retain component ordering and normalize positive weights; no cross-component
amplitude interference is implied. Unsupported weighted divergence, impact
quadrature, custom state arrays and callable entry fields are refused.

`boundary` and `source` are explicitly tagged
`{version: 1, kind: pending-physical-validation}`; no physically supported
alternative is offered. They do not substitute for the #623 contracts.

`axial_spec_identity` is a **candidate scientific-configuration digest**,
not a transport cache key. It includes the axial model marker, normalized
axis, recipe, grid/step and both stochastic count and seed. Execution-only
admission limits do not perturb scientific identity, though changing the
limits can refuse a job. Material data fingerprints, an anchored physics
model marker and explicit source-binning identity must be added when #623
lands. Off/planar identities remain untouched.

## Early refusal and next work

The Scene lowering entry point calls the central axial refusal before case
construction; the raw per-case resolver refuses before planar profile/zone
work. An axial call should never reach the planar density-matrix solver or
reuse its cache. CLI/profile/remote codepaths must converge on the same
`Physics` and `scene_physics` admission gate.

Raw compatibility mappings have an independent production guard in
`_channeling_admission.refuse_axial_channeling`, also exposed through
`montecarlo.case`. The data-only owner keeps configuration/table tools
independent of Monte Carlo imports. It recognizes an axial
`channeling.model` record as well as unlowered `channeling_model` /
`channeling_axial` selectors. Batch scheduling, transport (including capture,
blocks and adaptive precision), spectrum scoring, full/streamed artifact
replay, planar source reconstruction and result insertion refuse before their
setup or output mutation. Artifact replay reports a `TrajectoryArtifactError`;
the other paths report `ValueError`, all identifying the missing #623 contract.
The low-level artifact/checkpoint serializers remain storage primitives;
these guards do not establish a supported axial persistence format.

Local scan resolution checks the effective profile settings before resolving
material sweeps or checkpoint identities. Remote run submission resolves that
same inventory before busy/live-job probes, synchronization or staging, for
full, quick and CPU-profiling runs. A dry run applies the same admission.
Direct `run_sweep` and `run_cases` calls preflight the entire batch before
checkpoint-directory creation, cache/resume processing or worker dispatch.
Cached results cannot bypass admission.

Checkpoint brem/line repair checks every record while building its pure plan,
including records already at the requested parameters. An unsupported later
record therefore refuses the whole operation before earlier cases are
retuned, scored, saved or reported through progress callbacks. Trajectory
checkpoint planning similarly checks metadata before provenance/table work,
large-file hashing or target reads. Remote trajectory scoring requests this
metadata preflight before staging its SLURM job; ordinary inventory remains
available for inspecting unsupported captures. The host repeats planning
before scoring.

Human local-run, remote-run and checkpoint-repair commands report axial
admission as a runtime error without a traceback. Local run and repair JSON
modes retain one parseable failure envelope; remote run JSON remains
unsupported under its existing CLI contract.

The scan and trace apps currently construct their own default settings and
use `build_configured_cases`; they expose no standard-run axial selector.
The separate channeling app's axial diagnostic is a research solver, not
production admission. Stored-result analysis/compare/pixel views do not
establish a physical source or transport contract.

Next changes needed after #623 is independently anchored: material/axis frame
resolution and eligibility evidence, `Case.channeling` discriminated variant,
photons/eV/sr/incident electron source registration, joint native continuation,
model-specific cache/provenance marker, atomic bounded optional storage, and
explicit replay/reline, detector, CUDA, adaptive, capture and remote refusal
coverage. This preliminary change does not claim those acceptance criteria.
