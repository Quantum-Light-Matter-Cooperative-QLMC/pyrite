# Execution and scheduling

Execution profiles and executor interfaces are independent of physical scene identity. Optional scheduler backends have their own runtime dependencies; see [cluster execution](../cluster-execution.md).

`pyrite.execution.Executor`, `ExecutionProfile`, `LocalExecutor`,
`ResourceRequest` and `RetryPolicy` are provisional execution interfaces, outside
the governed supported API inventory in `api-surface.toml`. See the
[cluster execution reference](../cluster-execution.md#run-and-inspect-a-profile) for profile fields and
[cluster execution](../cluster-execution.md) for executor usage and limitations.
API support and scientific validation remain separate contracts.

## Governed sweep execution and cache preparation

`run_sweep` is the governed entry point for running and persisting a sweep.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.execution.run_sweep
   pyrite.execution.warm_cache
```

The [native SLURM Python interface](../cluster-execution.md#native-slurm-python-api)
and [retry/cancellation contracts](../cluster-execution.md#retry-cancellation-and-resumption)
cover experimental array orchestration. Worker persistence belongs to
[`pyrite.storage`](storage.md).

`warm_cache()` optionally prepares compiled kernels before a simulation. It uses
the configured backend and cache directories, reports failures per case, and
does not promise coverage of every later specialization.

```python
from pyrite.execution import warm_cache

report = warm_cache("minimal")
for case in report["cases"]:
    print(case["case"], case["status"])
```
