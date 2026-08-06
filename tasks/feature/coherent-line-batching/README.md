# Batched line path for `coherent_emission=True`

Branch: `feature/coherent-line-batching`
Worktree: `worktrees/coherent-line-batching`

TODO scope: P1 Ready item "Batched coherent line accumulation."

## Problem

`mc_spectrum` routes `coherent or groove is not None or layers is not None` to
the legacy per-hkl `_accumulate` loop (`src/cxr_mc/montecarlo/spectrum.py`,
`if coherent or groove is not None or layers is not None:`). The batched
`(n_seg, N_g)` path in the `else` branch — built to "collapse the tiny-kernel
launch storm that starved the GPU" — therefore never runs for coherent
profiles, which is the entire `hopg_coherent` workload.

## Evidence (ALEX-DESKTOP: 24 cores, 23.4 GB RAM, RTX 3060 Ti 8 GB)

Workload `hopg_coherent` / `hopg`, Ne=10000, 864 line bins, 4 reflections.
cProfile over 89 distinct cases, cold, 16.1 s of GPU-phase wall:

| symbol | calls | tottime | share |
|---|---|---|---|
| `_accumulate` (per-hkl loop) | 356 (4/case) | 6.14 s | 38% tottime, 54% cumtime |
| `cupy...core.array` (H2D asarray) | 8010 (90/case) | 2.43 s | 15% |
| `free_all_blocks` | 89 (1/case) | 0.88 s | 5% |
| `.get()` (blocking D2H sync) | 534 (6/case) | 0.46 s | 3% |

~17 ms of Python + launch dispatch per g-vector. Same geometry, both branches:

- `coherent_emission=True` → 0.1225 s/case (per-hkl loop)
- `coherent_emission=False` → 0.0462 s/case (batched) → 2.65x

The two branches compute different physics, so 2.65x is an upper bound on the
prize; the structural cause (4 serial passes of tiny kernels vs one vectorized
pass) is what the profile shows. The dispatch floor is also why raising Ne does
not help: Ne=1000 → 51.2 ms/case, Ne=40000 → 270.6 ms/case (40x electrons,
5.3x time).

The H2D share above is already partly addressed: the per-hkl branch now hoists
all its host→device transfers into a single stacking prologue (bit-for-bit,
~8% on a coherent microbench). The remaining cost is the loop structure itself.

## Scope

1. Extend the batched `(n_seg, N_g)` accumulation to `coherent=True`. The
   coherent path sums *phased complex fields* per polarization within a
   reflection/orientation and squares afterwards, so batching must keep
   reflections and mosaic orientations incoherent while vectorizing across g.
2. Decide whether the JIT coherent reduction kernel
   (`coherent_jit_kernel.run_coherent_reduction_kernel`, fp32 + CuPy +
   `sinc_cutoff is None`) takes a g-batched signature or stays per-g.
3. Out of scope unless cheap: the grooved and layered branches.

## Physics gate (blocking)

Batching reassociates float reductions exactly as the incoherent batch did
(`Validation: line-hkl-batch` precedent, noted at the `else` branch comment).
Required before sign-off:

- source equation + assumptions + limiting case in the code comment,
- new `Validation: <id>` marker and a physics-ledger row
  (`docs/physics-validation-ledger.md`, method `docs/validation/README.md`),
- golden regeneration (`regen-golden` skill),
- fresh-context verification; only a human marks `signed-off`.

Limiting cases to keep green: `tests/montecarlo/test_coherent_emission.py`
(single-segment coherent == incoherent self-term, N^2 in-phase limit,
on-resonance phase cancellation) and `tests/montecarlo/test_chunk_invariance.py`.

## Related, already landed

- Transport-only worker budget `_PIPELINE_WORKER_MEM_MB` (the GPU-pipeline pool
  was capped at 2 workers on a 24-core box by the full-case budget).
- `conservative` resource policy `release_every` 1 → 4 (per-case
  `free_all_blocks` cost ~4% of GPU-phase wall).
- Per-hkl H2D stacking prologue described above.
