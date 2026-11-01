# Physics and numerical configuration

Select physical models, sampling and integration settings, and local kernel controls. Scheduler profiles instead belong to `pyrite.execution`.

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.config.load_physics
   pyrite.config.load_numerics
   pyrite.config.CxrReflections
   pyrite.config.Precision
   pyrite.config.AdaptiveSampling
   pyrite.config.FixedSampling
   pyrite.config.RadiationNumerics
   pyrite.config.TransportNumerics
   pyrite.config.Timepix3ResponseNumerics
```

`KernelTuning` is the preferred public spelling of the existing
`ExecutionConfig` class. It selects local kernel controls; executor scheduling
is documented in [execution and scheduling](execution.md).

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :template: autosummary/public-alias.rst
   :nosignatures:

   pyrite.config.KernelTuning
```

`TransportKernelTuning` (experimental) is `KernelTuning.transport`: the CUDA
transport schedule, continuation row quantum, electron launch batch, device
scratch budget (`scratch_mib`) and launch threads per block (`kernel_threads`)
for in-process runs. It takes precedence over `PYRITE_MC_TRANSPORT_SCHEDULE` and
`PYRITE_MC_ELECTRON_BATCH_MAX` and never changes case identity or results.
Its `precision` field (`"fp64"`, `"mixed"`, or `None` for
`PYRITE_MC_TRANSPORT_PRECISION`, then the `mixed` default; #838) selects the exact CUDA kernel's
arithmetic. It changes trajectories at rounding scale, is recorded in run
provenance as `transport_precision`, and stays out of case identity.
`KernelTuning.cxr_threads` and `KernelTuning.cxr_energies_per_block`
(experimental) select the CUDA CXR line-reduction launch: threads per block
and energy nodes per block. They regroup the float sum only and never enter
case identity.
`KernelTuning.stream_block_electrons` (experimental, positive integer or
`None`) opts `pyrite.simulate` into live streamed scoring in whole-electron
blocks; see [Stream spectra in electron blocks](../../guides/streamed-scoring.md).
It never enters case identity; spectra agree with the default path to rounding.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.config.TransportKernelTuning
```

## Angular sampling and legacy kernel controls

`AngularSampling` is the advanced-stable semantic alias for `PixelScorer`;
it controls angular quadrature, independently of physical detector geometry.
`ExecutionConfig` retains its root compatibility import; prefer
`pyrite.config.KernelTuning` for the same local kernel controls.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :template: autosummary/public-alias.rst
   :nosignatures:

   pyrite.config.AngularSampling
   pyrite.ExecutionConfig
```
