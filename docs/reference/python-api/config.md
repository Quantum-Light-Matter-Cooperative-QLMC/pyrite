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
transport schedule, continuation row quantum and electron launch batch for
in-process runs. It takes precedence over `PYRITE_MC_TRANSPORT_SCHEDULE` and
`PYRITE_MC_ELECTRON_BATCH_MAX` and never changes case identity or results.

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
