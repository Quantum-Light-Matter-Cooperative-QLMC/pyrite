# Electron transport

Reviewed trajectory and capture operations. For everyday simulations prefer the high-level `simulate` entry point.

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.transport.Case
   pyrite.transport.simulate_trajectories
   pyrite.transport.summarize_trajectories
   pyrite.transport.iter_primary_summaries
   pyrite.transport.depth_loss_profile
   pyrite.transport.capture_secondary_energy_balance
   pyrite.montecarlo.runner.run_case
   pyrite.montecarlo.runner.run_cases
```

## Advanced transport and radiation

Use `pyrite.transport` for trajectory simulation and saved-capture reductions;
use `pyrite.radiation` for source-level spectra. These lazy facades resolve to
exactly the existing implementation objects: signatures, errors, numerical
outputs, RNG streams, provenance, serialization and scientific identities are
unchanged. Importing either namespace loads no engines or material datasets;
resolving a function loads its implementation, and calling it may read tables,
select an accelerator or perform the documented artifact I/O.

```python
from pyrite.transport import Case, simulate_trajectories, summarize_trajectories
from pyrite.radiation import photon_energies, mc_cxr_spectrum, mc_brem_spectrum

# Provisional directional API; no stability promotion:
from pyrite.radiation import directional_source, directional_sources
from pyrite.results.directional import DirectionalSpectrum
```

`pyrite.simulate(scene)` remains the supported high-level one-scene operation.
`Case` is the frozen, validated lowered record with `to_dict()` serialization;
it remains advanced-stable for existing advanced callers. Prefer `Scene` for
new scene construction. No engine package becomes stable as a whole.

## Scientific contracts and migration

All transport facade entries below are **advanced-stable**. Units in keyword
and table-column names are literal: angstroms, keV and radians. Optional model
controls should be passed by keyword. The linked function docstrings define
full defaults, supported combinations and exceptions.

The [transport-output contract](../../physics/beam-transport/transport-outputs.md)
defines returned segment fields, event codes, units and normalization. For
stored trajectories, see the
[saved-capture observables](../../physics/beam-transport/transport-outputs.md#saved-capture-observables)
and [capture workflow](../trajectory-workflows.md).

| Canonical import (`pyrite.transport.`) | Previous import | Input/output and effects |
| --- | --- | --- |
| `Case` | `pyrite.montecarlo.Case` | Validated lowered simulation/configuration record; field-specific units, frozen mapping serialization. Invalid fields raise validation errors; no transport on construction. |
| `simulate_trajectories` | `pyrite.montecarlo.transport.simulate_trajectories` | Incident energy/cutoff in keV, thickness/segment positions and lengths in angstroms; returns segment arrays and incident `Ne`/fate metadata. Samples transport with the supplied seed; may resolve elastic/stopping tables and backend. Unsupported model combinations/table coverage raise errors; incomplete step-limited histories raise. No detector response or photon tally. |
| `summarize_trajectories` | `pyrite.montecarlo.trajectory_summary.summarize_trajectories` | Reads a saved capture and explicit depth/energy/angle edges; returns histogram DataFrames. Fate fractions divide by stored incident `Ne`, including missed entries. Depth loss is keV/bin without primary, mass or volume normalization. Invalid edges/missing row fields raise. |
| `iter_primary_summaries` | `pyrite.montecarlo.trajectory_summary.iter_primary_summaries` | Streams capture DataFrames: primary fate, depth/range (angstrom), terminal pre-event energy (keV), angle (rad). Descendants do not determine primary fate; missed entries have no material rows. Artifact validation and file errors propagate. |
| `depth_loss_profile` | `pyrite.montecarlo.trajectory_summary.depth_loss_profile` | Stored segment mapping plus depth edges (angstrom) to continuous-loss keV/bin by layer. Excludes cutoff residuals/hard transfers; no primary normalization or file I/O. Invalid edges raise. |
| `capture_secondary_energy_balance` | `pyrite.montecarlo.trajectory_balance.capture_secondary_energy_balance` | Reads secondary-transport capture, reuses its canonical energy ledger; returns `term, energy_keV`. Requires secondary metadata and positive block limit; propagates artifact/file errors. No resimulation. |

For source grids, normalization and radiation import compatibility, see the
[radiation contracts](radiation.md#source-contracts-and-migration).

## Other documented engine operations

These remain at their current owners; listing them does not promote other
engine internals. All retain their existing support level.

| Entry | Owner / status | Scientific contract and effects |
| --- | --- | --- |
| `run_case`, `run_cases` | Runner orchestration, advanced-stable compatibility paths | Lowered `Case`/case mappings, field-specific units; execute transport and scoring, resolving tables/backend and retaining existing result/provenance contracts. Batch scheduling/collection and existing errors remain runner concerns; use `simulate` for public scene orchestration. No new transport facade aliases. |

See [detector response contracts](detectors.md#response-contracts),
[saved beam diagnostics](beams.md#saved-beam-diagnostics), and
[quantum solver diagnostics](quantum.md#solver-diagnostic-contracts) for
other reviewed engine operations.


## Coupled radiative moment evaluation

`simulate_trajectories(..., radiative_moments="table")` is the default for
coupled BremsLib transport on CPU and CUDA. Prepared float64 tables interpolate
soft stopping in eV/Å and hard-event rate in Å⁻¹ using linear weights in log
kinetic energy, with relative budget `2e-4` and explicit coverage checks.
Use `radiative_moments="exact"` for reference quadrature. This keyword-only
switch preserves the existing positional call interface. It has no effect on
uncoupled radiation. Exact conditional photon sampling and the sampled photon
energy debit remain in both modes.

New coupled cases include `radiative_moment_model` in content/replay identity;
legacy cases without that field retain exact evaluation. Advanced callers can
use `dataclasses.replace(case, radiative_moment_model="exact")` to select an
exact case explicitly. Result `radiative` metadata include `moment_model` and,
for tables, `moment_table_sha256`. The latter pins the actual packed numerical
buffer, including coverage and physical plasma values. The existing source
and physical table identities are retained separately.

Default activation was explicitly requested before ensemble qualification was
complete. See [radiative moment evidence](../../validation/radiation-physics/radiative-moment-interpolation.md)
for passed component/replay checks and the remaining scope of issue #788.

## Transport scheduling

The production process defaults to the `auto` transport schedule. Every
supported exact-CUDA pass, whatever its size, uses bounded continuation with
256 rows per round. Other passes retain full-history with an INFO-level
reason. Set `PYRITE_MC_TRANSPORT_SCHEDULE=full-history` before importing
PyRITE to force the reference schedule; `continuation` forces continuation
and rejects unsupported modes. `pyrite remote` forwards this override.

Advanced callers can pass `PerElectronTransportConfig` to
`simulate_trajectories`; separately constructed configs keep the existing
`schedule=None` rule (continuation exactly when a continuation config is
provided). Scheduling does not change scientific case/checkpoint identity.
Live scratch is bounded; retained trajectories still scale with total rows
and can exhaust host RAM. See the
[production qualification and limits](../../validation/beam-transport/equal-precision-pilot.md#production-continuation-qualification-799).
