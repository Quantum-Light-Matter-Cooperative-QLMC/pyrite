# Grouped numerics and execution controls (#680)

The Python API now has optional typed sections for *transport discretization*,
*radiation numerical settings*, and *local kernel execution settings*. These
three sections do **not** replace the #681 executor infrastructure, which
handles local/SLURM dispatch, resources and work units. No scheduler object is
stored on ``Numerics`` and no ``ResourceRequest`` is duplicated.

```python
import pyrite as pr

numerics = pr.Numerics(
    sampling=pr.FixedSampling(400, 200),
    transport=pr.TransportNumerics(energy_model="midpoint", max_dE_frac=0.01),
    radiation=pr.RadiationNumerics(
        cxr_reflections=pr.CxrReflections(n_families=6)),
    execution=pr.ExecutionConfig(
        transport_core="per-electron", cxr_chunk=256, brem_chunk=256),
)
print(numerics.describe_numerics())
```

- **Scientific input:** ``TransportNumerics`` and ``RadiationNumerics``.
  The legacy ``energy_model``, ``max_dE_frac``, ``cxr_reflections``, and
  ``energy_bins`` fields are still the canonical case-lowering fields.
- **Kernel execution:** ``ExecutionConfig`` owns ``transport_core`` and
  kernel batches. The preexisting ``spec_chunk``/``brem_chunk`` case fields
  are **not removed** and no claim is made that batch settings are safely
  excluded from scientific content identity until a separate numerical audit.
- **Orchestration:** ``pyrite.execution.Executor``, ``LocalExecutor``,
  ``SlurmArrayExecutor`` and ``ResourceRequest`` stay separate (#681). For
  example, the executor is passed to ``run_sweep(..., executor=...)``.

Legacy ``pr.Numerics(cxr_chunk=..., energy_model=...)`` and all existing
profile/preset/CLI inputs remain valid and unchanged. New nested sections
are opt-in; they simply set the established flat fields before validation.
A non-default flat setting that disagrees with a typed section raises an
error instead of being ignored. ``Numerics()`` produces its original defaults.

To update a **typed section** on a configured object, use the explicit helpers
``with_transport()``, ``with_radiation()`` and ``with_execution()``; these
also reset the corresponding compatibility mirrors before calling
``dataclasses.replace``. Replacing an unrelated field normally works:

```python
from dataclasses import replace
updated = numerics.with_transport(
    replace(numerics.resolved_transport, max_dE_frac=0.005))
assert updated.max_dE_frac == 0.005
```

This is the **Python API bridge**, not the schema-v2 migration. Named grouped
CLI presets, cross-route provenance, execution resource negotiating and
possible deprecation of flat fields are future work. Nothing changes an
existing Case payload, RNG stream, checkpoint, hash or validated science
model merely by selecting equivalent typed sections.
