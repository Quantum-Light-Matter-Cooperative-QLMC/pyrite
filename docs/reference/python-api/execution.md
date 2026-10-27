# Execution and scheduling

Execution profiles and executor interfaces are independent of physical scene identity. Optional scheduler backends have their own runtime dependencies; see [cluster execution](../cluster-execution.md).

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.execution.Executor
   pyrite.execution.ExecutionProfile
   pyrite.execution.LocalExecutor
   pyrite.execution.ResourceRequest
   pyrite.execution.RetryPolicy
```
