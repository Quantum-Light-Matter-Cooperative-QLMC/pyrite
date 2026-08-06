# Streaming coherent line RawKernel stage

## Purpose

The coherent batched path already removes the old per-hkl setup loop, but after
its rebase onto the newer line machinery it still constructs and retains eight
mask-compacted arrays for every accepted `(segment, g)` pair, transfers per-row
counts to the host, and concatenates/slices those arrays before launching the
existing per-g coherent reduction.

This change replaces that CUDA-fp32 path with a bounded streaming pipeline.
CPU, fp64, `sinc_cutoff`, layered, grooved, and explicitly disabled paths retain
the prior implementations.

## Algebra

For reflection/orientation row `g`, polarization `p`, and photon energy `E`,

```text
F[g,p,E] = sum_j c[g,p,j] sinc(a[g,j] (E-E_r[g,j]))
                         exp(i (s[j] E - phi[g,j]))
```

and the spectrum contribution remains

```text
sum_g w_g ( |F[g,s,E]|^2 + |F[g,p,E]|^2 ).
```

The key streaming identity is

```text
F[g,p,E] = sum_blocks F_block[g,p,E].
```

Intensity is *not* reduced per block. Four persistent field planes are updated
across segment blocks and are squared only after the final block. Different `g`
rows are never added as fields, so reflection and mosaic-orientation coherence
semantics are unchanged.

## CUDA stages

### 1. Coherent prologue

`run_coherent_prologue_kernel` owns one `(g, segment)` pair and performs:

- resonance kinematics;
- shared-bracket interpolation of chi/U/mu;
- both complex polarization amplitudes;
- Beer-Lambert amplitude attenuation;
- finite-time width;
- propagation phase slope and `g.r` phase.

Output is fixed-order **g-major** scratch. Rejected pairs have zero field
coefficients; no compaction is required.

### 2. Field accumulation

`run_coherent_field_accumulation_kernel` owns one `(g, energy-group)` CUDA
block. Threads stride over the current segment block, reduce sigma/pi real and
imaginary fields in shared memory, and add one partial field to each persistent
`(g,E)` cell. There is exactly one writer to a field cell per launch, so no
atomics are needed.

### 3. Finalize

`finalize_coherent_fields` owns photon-energy bins, squares each completed `g`
row, applies its mosaic weight, and adds the incoherent row sum to `spec`.

## Memory behavior

The former batched coherent path retained approximately eight REAL values for
all kept `(segment,g)` pairs until the end of the case. The streaming path holds
approximately

```text
8 * pair_target * sizeof(float32)
+ 4 * N_g * N_E * sizeof(float32)
```

plus ordinary input/table storage. `_JIT_COHERENT_PAIR_TARGET` defaults to
1,000,000 pair slots, so prologue scratch is about 32 MB. Scratch is explicitly
released after each field-accumulation launch so the memory pool can reuse it
for the next block.

## Numerical behavior

The physics equations are unchanged, but two float32 reassociations remain
validation debt:

1. the RawKernel interpolation/compiler expression tree may round differently
   from the previous CuPy elementwise path;
2. cases larger than one internal segment block reduce each block's field first
   and then add block partials, rather than performing one monolithic reduction.

These should stay at float-rounding scale and are covered by the existing
`coherent-line-hkl-batch` tolerance philosophy, but require CUDA goldens and
A/B validation before human sign-off.

## Required CUDA validation

Run at minimum:

```text
pytest tests/montecarlo/test_coherent_emission.py
pytest tests/montecarlo/test_chunk_invariance.py
```

The updated coherent tests include:

- single-segment coherent == incoherent self-term;
- in-phase N^2 limit;
- on-resonance phase cancellation;
- batched coherent vs forced legacy per-hkl route;
- multiple reflections remain incoherent;
- streaming RawKernel vs pre-existing batched coherent fallback;
- one internal streaming block vs forced multiple segment blocks.

Then benchmark `hopg_coherent` with NVTX ranges
`cxr.lines.coherent_prologue`, `cxr.lines.coherent_field`, and
`cxr.lines.coherent_finalize`. Tune `CoherentStreamKernelConfig` and
`_JIT_COHERENT_PAIR_TARGET` from measurements rather than assumptions.
