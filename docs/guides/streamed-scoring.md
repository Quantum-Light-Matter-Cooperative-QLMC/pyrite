# Stream spectra in electron blocks (experimental)

Use live streamed scoring when a scene's joined trajectories do not fit in
device or host memory. Transport runs in whole-electron blocks; each block is
scored as soon as it finishes and then released, so peak memory follows the
block size rather than the electron count.

Where the joined run fits, the default path is faster. Keep streaming off
unless the default runs out of memory.

## Turn it on

Set `KernelTuning.stream_block_electrons` on the scene's `Numerics`:

<!-- verify: skip (illustrative; runs a full simulation) -->
```python
import numpy as np

import pyrite as pr
from pyrite.config import KernelTuning
from pyrite.detectors import EnergyBins

numerics = pr.Numerics(
    cxr_trials=64_000,
    brem_trials=64_000,
    # Streaming needs a final line grid: give explicit energy bins.
    energy_bins=EnergyBins(
        cxr=np.linspace(8000.0, 9000.0, 400), brem=np.arange(1000.0, 300001.0, 1000.0)
    ),
    execution=KernelTuning(stream_block_electrons=4096),
)
result = pr.simulate(scene, numerics=numerics)
```

`result` has the same fields, case and identity digest as the default run.
Spectra agree with it to floating-point summation order; transport counts
(backscatter and hit fractions, segment count) are exact.

## Choose the block size

The value counts electrons, not bytes. Peak memory scales with block size
times the trajectory segments per electron, which grows with beam energy,
target thickness and atomic number, so the same block size uses different
memory on different scenes. For reference, the #611 measurements streamed a
300 keV W slab in 4096-electron blocks on a 16 GiB GPU. Start smaller than
the largest electron count the default path fits, then raise it while memory
allows: larger blocks amortize per-block launch and staging costs.

## What it supports

Streaming runs on the per-electron or CUDA transport core, in-process, through
`pyrite.simulate` with a scalar `Detector`. It covers additive spectra: the
incoherent CXR line spectrum, characteristic lines, and the EEDL or BremsLib
continuum, including coupled BremsLib radiative transport.

These raise `ValueError` before any transport:

- coherent emission, temporal profiles, photon transport and transition
  radiation;
- channeling, adaptive precision, automatic line-grid resolution (no explicit
  `energy_bins`), GDF beams, and secondary, pair or positron cascades;
- the lockstep transport core;
- a `PlanarDetector`, `pyrite.phases.simulate_transport`, directional source
  scans, and batch executors.

The design and its eligibility table are in
[Live streamed scoring](../repo-design/compute/streamed-scoring.md).
