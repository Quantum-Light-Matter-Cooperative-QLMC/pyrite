# Core simulation

Build immutable scenes, select physics and numerical controls, and run simulations with `simulate`. Use domain namespaces for specialized objects.

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.photon_energies
   pyrite.Beam
   pyrite.Target
   pyrite.Scene
   pyrite.Sweep
   pyrite.Physics
   pyrite.Numerics
   pyrite.simulate
```

## Scene simulation

Build a beam, target and detector, then call {py:func}`pyrite.simulate` with
those three components, or pass a complete `Scene` as `simulate(scene)`.
The canonical `Scene` dataclass fields are the physical `beam`, `target`,
`detector` geometry and ordered `filters`. Legacy constructor inputs for
`physics`, `numerics`, `quantum` and `acquisition` remain accepted through a
compatibility adapter, but are excluded from the physical dataclass value and
its `asdict()` representation. Prefer passing `physics=` and `numerics=` to
`simulate()` and acquisition through `Observation`. Pass an independent
{py:class}`pyrite.quantum.QuantumSetup` with `quantum=`; omission inherits a
legacy Scene setup and explicit `quantum=None` disables it.
`pyrite.api.build_case` uses the same precedence and validation rules, and
unsupported or contradictory quantum settings fail before case preparation or
transport. Effective quantum solver controls are recorded in result numerics.
Target tilt
and azimuth retain full floating-point precision in scenes and profile sweeps.
`Physics` selects models and `Numerics` controls sampling and integration.
Beam transverse and
longitudinal descriptions and target footprints are explicit inputs.
{py:func}`pyrite.config.load_physics` and {py:func}`pyrite.config.load_numerics` return
named catalog presets (for example the bundled `electron-standard` and
`default`, or ones made with `pyrite physics|numerics preset create`) as
frozen `Physics` and `Numerics`; adjust them with `dataclasses.replace`
instead of restating every setting.

Pass {py:class}`pyrite.observations.Observation` with `observation=` to select
response, acquisition and pixel sampling independently of the scene. Its
`detector="detector"` reference selects `Scene.detector`; a supplied planar
value must match that physical geometry. A scalar `Detector` value requests a
virtual angular observation, including on a detector-free source scene.
`sampling=` currently uses the supported `pyrite.observations.PixelScorer`.
Omitted observation fields inherit legacy embedded settings. Differing explicit
settings raise; start from a detector without embedded response/scorer and a
scene without acquisition to reuse one physical definition across observations.
Names are labels and do not change scientific identity.
`result.provenance["input_scene"]` retains the original scene for both legacy
and independently composed calls. The `"physics"`, `"quantum"` and
`"acquisition"` provenance keys expose the effective independent configuration;
`"numerics"` retains the effective numerical controls. These values do not
change case or observation identity. Meanwhile,
`result.scene` remains the resolved compatibility view used by read-time methods.
The [observation example](../../reference/python-scene-recipes.md#compose-independent-observations)
shows how to reuse the same geometry.

Use `simulate(scene, executor=executor)` to select an existing
`pyrite.execution.Executor` independently of the physical scene. For example,
`LocalExecutor()` uses the established batch runner for the single case.
It can start CPU worker processes, so scripts should invoke it inside an
`if __name__ == "__main__":` guard, as with other batch-runner calls.
Omitting `executor` retains the original in-process simulation path. Scalar,
planar and counting observations use the same result assembly and scientific
identity in both forms. The executor must return one completed result;
interrupted or missing directional results fail explicitly.

This opt-in route currently requires `Numerics.transport_core="auto"`, since
batch executors do not accept a per-call transport-core override. Kernel
chunk settings remain on `Numerics`; scheduler settings remain on the executor.
Dask and SLURM executors keep their existing caller-configured artifact stores
and table-aware case keys. Their returned artifacts are checked against the
resolved source key and requested directions before use. Such executors can
write worker artifacts; `simulate` does not write simulation checkpoints.
`result.provenance["execution"]` records the executor class and its available
typed resource, scheduler and retry settings separately from scientific
identity. These describe requests, not measured worker hardware. A shared scientific identity does not promise bitwise
agreement across backends or scheduling strategies.

`simulate` returns an in-memory `Result` without reading or writing checkpoints.
Radiation families remain separate: `cxr.incoherent`, optional `cxr.coherent`,
`brem` and optional `characteristic`. Each spectrum carries its energy grid
and normalization. Scalar-detector densities are response-free photons per
incident electron per eV per sr; planar-detector densities are filtered,
solid-angle-weighted observation averages, still per sr. Use `spectral_flux()`
for current and acceptance scaling, or `acquire()` for registered pixel counts.

The workflow covers [results](../../reference/python-scene-recipes.md#interpret-the-result),
[detector response](../../reference/python-scene-recipes.md#apply-detector-response-explicitly),
[sweeps](../../reference/python-scene-recipes.md#expand-a-path-addressed-sweep),
[adaptive sampling](../../reference/python-scene-recipes.md#choose-adaptive-sampling) and
[shared target/filter bodies](../../reference/python-scene-recipes.md#reuse-bodies-as-targets-and-filters).
For correlated imported beams, see [GDF beam import](../../guides/gpt-gdf-beams.md).

Experimental photon source observations and their separate normalization and
persistence rules are documented under [photon aperture interfaces](experimental.md#photon-aperture-interfaces).
