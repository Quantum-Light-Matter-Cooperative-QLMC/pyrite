# Explicit sampling policies (issue #680)

The Python API now offers **fixed** and **adaptive** policies as separate frozen
objects. Existing ``Numerics(cxr_trials=..., brem_trials=...)`` and
``Numerics(precision=Precision(...))`` remain valid, with unchanged case and
checkpoint identity. This slice adds an opt-in Python API; named CLI catalog
policies and schema migration are **not** implemented yet.

```python
import pyrite as pr

fixed = pr.Numerics(sampling=pr.FixedSampling(cxr_electrons=450, brem_electrons=100))
adaptive = pr.Numerics(
    sampling=pr.AdaptiveSampling(
        target_rse=0.05,
        min_electrons=200,
        max_electrons=20_000,
        block_electrons=100,
        observables=("cxr", "brem"),
    )
)

assert fixed.resolved_sampling == pr.FixedSampling()
print(adaptive.describe_sampling())
```

Adaptive stopping is currently validated only for CXR/brem incoherent scoring;
coherent emission, physical pixel detectors, cascades, and other unsupported
routes continue to refuse. Adaptive component counts are *maximum budgets*, not
realized electron counts. Independent brem-prefix stopping is unchanged. The
historical ``Precision`` name remains supported; it means *statistical*
precision, not floating-point precision.

A typed policy cannot be combined with different legacy fixed counts or
with a different legacy adaptive policy. Legacy-only adaptive construction
still carries inactive 450/100 trial defaults for compatibility; use
``describe_sampling()`` to see this explicitly. The old serialized ``line``
observable is deliberately preserved for checkpoints and content hashes.

When using ``dataclasses.replace`` to **switch** sampling modes, clear the
old legacy compatibility field as appropriate, e.g.:

```python
from dataclasses import replace

updated = replace(adaptive, sampling=pr.FixedSampling(100, 100), precision=None)
```

Current API-default fixed counts (450/100) and CLI-default adaptive policy
remain intentionally different until explicit versioned profile migration.
Do not assume changing to this API implicitly migrates existing CLI profiles.
