---
name: regression-testing
description: Use when the task is to design or add a fast PyRITE regression test for numerical, stochastic, plotting, checkpoint, or physics behavior; domain skills own implementation review.
---

# Regression Testing

Freeze the smallest externally meaningful behavior that would have caught the
regression.

## Test design

1. Reproduce the failure with a focused test before changing implementation.
2. Use a physically meaningful, minimal input; fix RNG seeds for stochastic code.
3. Assert public behavior: values, shapes, units, normalization, invariants, or
   serialized compatibility. Avoid duplicating implementation details.
4. Prefer analytic or independently derived expectations. Otherwise record the
   provenance of reference data.
5. Choose tolerances from numerical conditioning and backend precision, not from
   the observed error alone.
6. Keep CPU tests fast; place expensive external or publication comparisons in
   `checks/`. Time new tests (`--durations=0`); mark any >5 s wall or with a
   large peak RSS `@pytest.mark.slow` (CI still runs them on every PR).

Report original symptom, pre-fix failure, seed/tolerance rationale, and focused
command:

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test \
  tests/path/to/test.py -k test_name
```

- Snapshotting a large array when a physical invariant is the real contract.
- Using a tolerance wide enough to hide the reported regression.
- Sharing implementation helpers with the expected-value calculation.
