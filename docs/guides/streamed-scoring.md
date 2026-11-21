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
from pyrite.config import FixedSampling, KernelTuning
from pyrite.detectors import EnergyBins

numerics = pr.Numerics(
    sampling=FixedSampling(cxr_histories=64_000, brem_histories=64_000),
    # Optional: explicit energy bins skip the grid-resolution pass.
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

## Automatic line grids

Without explicit `energy_bins`, the line grid is resolved automatically from
every electron's trajectories. Streaming then transports the scene twice: a
first pass keeps only the compact per-segment data the grid needs and resolves
the same grid as the default run, and a second pass scores on it. The extra
pass costs one transport, plus one line evaluation on a two-point axis under
the `resonance-population` bandwidth. A grid already in the line-grid cache
skips it. That pass's data grows with the electron count, at about 16 bytes
per line segment, plus about 12 bytes per production line under the
`resonance-population` bandwidth (24 with `PYRITE_FP64=1`).

Once that data passes 2 GiB, or is projected to over the run, it moves to
disk, so host memory stays bounded and the grid is unchanged. It goes under
`<user cache>/line-grid-spill/` (`PYRITE_CACHE_DIR` moves it) and is deleted when the
pass ends, however it ends. Peak disk use is about twice the line-segment data
plus the production lines: roughly 170 kB per electron for h-BN at 5 MeV, 1 mm,
so 10^6 primaries need about 170 GB free. If the projection does not fit, the
run stops at the first block with a `LineGridSpillError` naming the space
needed; free space there or point the spill at a larger disk. Explicit
`energy_bins` or a cached grid need neither the memory nor the disk.

To spill to a large scratch file system, or at a different size, set
`KernelTuning.stream_spill_dir` (a path) and `stream_spill_mib` (MiB of held
or projected evidence; `0` always spills; default 2048) next to the streaming
option:

<!-- verify: skip (illustrative; runs a full simulation) -->
```python
numerics = numerics.with_kernel_tuning(
    KernelTuning(stream_memory_mib=32_768, stream_spill_dir="/scratch/pyrite", stream_spill_mib=0)
)
```

Both require `stream_block_electrons` or `stream_memory_mib`. `pyrite.simulate`
creates the directory and refuses an unwritable one before transport. Each run
uses its own subdirectory, removed when the pass ends. The grid, case identity
and line-grid cache are unchanged.

## Choose the block size

The value counts electrons, not bytes. Peak memory scales with block size
times the trajectory segments per electron, which grows with beam energy,
target thickness and atomic number, so the same block size uses different
memory on different scenes. For reference, the #611 measurements streamed a
300 keV W slab in 4096-electron blocks on a 16 GiB GPU. Start smaller than
the largest electron count the default path fits, then raise it while memory
allows: larger blocks amortize per-block launch and staging costs.

On CUDA, fixed blocks of at least 262,144 electrons, with two or more of
them in the run, overlap transport of the next block with scoring of the
current one. Results are bitwise identical to streaming without overlap; wall
time drops by up to about 30% when transport is a large share of the run, and
peak device memory grows by about one block. Set `PYRITE_STREAM_OVERLAP=0` to
turn it off, or `1` to force it for smaller fixed blocks.

## Set a memory target instead

Use `KernelTuning.stream_memory_mib` to give a device-memory budget in MiB
rather than an electron count. It excludes `stream_block_electrons`:

<!-- verify: skip (illustrative; runs a full simulation) -->
```python
numerics = numerics.with_kernel_tuning(KernelTuning(stream_memory_mib=32_768))
```

The target covers the whole card. At the start the run reads the device memory
in use outside the CuPy pool, including this process's CUDA context and any
other process, and caps the pool at the target minus that and a 0.5 GiB
allowance, at most 95% of the target. It restores the previous cap afterwards.
The pool always keeps half of the target, so targets too small for that overhead
are best effort. Memory other processes take after the start is not covered.

Transport counts are exact. CUDA scoring runs in float32 by default, and these
blocks split the continuum population differently from a joined run, so
continuum bins can differ more than the lines: in W at 1 MeV, up to about 7e-5
per bin against about 5e-7 for the lines, and about 7e-7 for the Si continuum.
With `PYRITE_FP64=1` the difference falls to rounding. An
allocation over the cap goes through the normal block recovery and can raise
an out-of-memory error if the target is too small for the scene.

## What it supports

Streaming runs on the per-electron or CUDA transport core, in-process, through
`pyrite.simulate` with a scalar `Detector`. It covers additive spectra: the
incoherent CXR line spectrum, characteristic lines, and the EEDL or BremsLib
continuum, including coupled BremsLib radiative transport and secondary,
pair or positron cascades (#923), whose histories run whole inside each block.
Adaptive sampling streams too (#1280): a measure-only pass finds the realized
electron counts, then those counts stream. Scans switch an adaptive case to
this route on their own when its retained rows would not fit in memory
(`PYRITE_STREAMED_ADAPTIVE=off` disables that).

These raise `ValueError` before any transport:

- coherent emission, temporal profiles, photon transport and transition
  radiation;
- channeling, windowed automatic line grids, and GDF beams;
- adaptive sampling with reflection selection or batch means;
- the lockstep transport core;
- a `PlanarDetector`, `pyrite.transport.simulate_trajectories`, directional
  source scans, and batch executors.

The design and its eligibility table are in
[Live streamed scoring](../repo-design/compute/streamed-scoring.md).
