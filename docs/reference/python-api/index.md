# Python API reference

Start with `Beam`, `Target`, `Detector`, `Physics`, `Numerics`, and `simulate`. Use the canonical public namespaces for specialized types. Each section lists generated class and function documentation (including source links).

Public API stability is governed by `api-surface.toml`. It is **not** equivalent to scientific validation: see the [capability matrix](../../physics/capabilities.md) and [validation status](../../validation/status-summary.md).

| Reference | Purpose |
| --- | --- |
| [Core simulation](core.md) | Scene construction and simulation entry points |
| [Beams and particle distributions](beams.md) | Spots, bunch timing and imported particles |
| [Geometry and targets](geometry.md) | Slabs, stacks, bodies, cuts and filters |
| [Materials and crystallography](materials.md) | Catalogs, authoring and material response |
| [Detectors and response](detectors.md) | Acceptance, pixel geometry and response |
| [Observations and acquisitions](observations.md) | Measurement requests and acquisition |
| [Physics and numerical configuration](config.md) | Models, sampling and integration controls |
| [Results and analysis](results.md) | Spectra, counts, provenance and reductions |
| [Quantum and channeling](quantum.md) | Portable settings and bounded solver diagnostics |
| [Electron transport](transport.md) | Trajectory simulation and saved-capture reductions |
| [Radiation sources](radiation.md) | Source spectra, units and normalization |
| [Experimental and provisional interfaces](experimental.md) | Provisional optimization, photon and optical interfaces |
| [Execution and scheduling](execution.md) | Executors and scheduling policies |
| [Units and quantities](units.md) | Pint and canonical units |
| [Persistence and storage](storage.md) | `pyrite.storage` |
| [Compatibility](compatibility.md) | Root aliases and migration paths |

```{toctree}
:maxdepth: 1

core
beams
geometry
materials
detectors
observations
config
results
quantum
transport
radiation
experimental
execution
units
storage
compatibility
Legacy section links <../../api>
```

## Supplementary reference contracts

```{toctree}
:maxdepth: 1

../quantum-api
../experimental-api
../directional-sources
../gdf-beams
```
