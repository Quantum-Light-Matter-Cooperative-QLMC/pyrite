# JIT spectrum-kernel walkthrough

This reference follows the arithmetic and CUDA ownership model of the two segment-reduction kernels used by the line-spectrum path:

- [`line_jit_kernel.py`](../../../src/pyrite/montecarlo/spectrum/line_jit_kernel.py) adds independent segment **intensities**;
- [`coherent_jit_kernel.py`](../../../src/pyrite/montecarlo/spectrum/coherent_jit_kernel.py) adds complex segment **fields** and squares the completed field.

The examples use small, artificial arrays so every operation is visible. They explain evaluation of already-ledgered equations; they do not introduce or independently validate physics. The physical derivations and validation status remain owned by [Validation: `coherent-line-spectrum`](../../validation/radiation-physics/coherent-line-spectrum.md), [Validation: `finite-time-lineshape`](../../validation/radiation-physics/finite-time-lineshape.md), and [Validation: `coherent-emission`](../../validation/radiation-physics/coherent-emission.md). The optional branches described below additionally inherit [Validation: `xray-in-medium-propagation-phase`](../../validation/radiation-physics/xray-in-medium-propagation-phase.md), [Validation: `coherent-inter-electron-decoherence`](../../validation/radiation-physics/coherent-inter-electron-decoherence.md), and the currently filtered [Validation: `cross-reflection-coherence`](../../validation/radiation-physics/cross-reflection-coherence.md) boundary.

## Common execution model

Both modules define specialized CuPy `jit.rawkernel` functions. A CUDA block owns `energies_per_block` consecutive photon-energy bins. Every thread in that block visits a strided subset of the input lines:

```text
line = threadIdx.x
while line < n_lines:
    accumulate line
    line += blockDim.x
```

A thread therefore does two independent kinds of batching:

1. it handles lines `tid`, `tid + nthreads`, `tid + 2*nthreads`, and so on;
2. for each visited line, it handles every energy bin owned by its block.

The second point is easy to miss. With `energies_per_block = 3`, thread 17 does not own only one energy. It keeps three private accumulators and evaluates line 17 at all three energies before advancing to line `17 + nthreads`.

For $N_E$ energy bins and $B_E$ energies per block, the host wrapper launches

$$
N_{\rm blocks}=\left\lceil\frac{N_E}{B_E}\right\rceil.
$$

Block $b$ owns energy indices $bB_E$ through $bB_E+B_E-1$. Bounds flags guard the unused lanes of the final block when $N_E$ is not divisible by $B_E$. Because no two blocks own the same energy index, thread 0 can write the reduced answer without an atomic operation.

## Incoherent line reduction

### Quantity evaluated

The prologue in `lines.py` has already reduced each accepted segment/reflection/orientation pair $j$ to three `float32` arrays:

- `E_r[j]`: resonance energy $E_{r,j}$ in eV;
- `aw[j]`: width coefficient $a_j$ in eV$^{-1}$;
- `w[j]`: all energy-independent intensity factors $w_j$.

For energy bin $E_k$, the reduction kernel evaluates

$$
I_k \mathrel{+}= \sum_j w_j
\left[\frac{\sin x_{jk}}{x_{jk}}\right]^2,
\qquad
x_{jk}=a_j(E_k-E_{r,j}).
$$

This is the source expression `wt * s * s`. `w == 0` is a rejected prologue pair and is skipped before loading its geometry. At exact resonance, the code replaces $x=0$ with `F32_TINY = 1.0e-20`; numerically this implements the limit $\sin(x)/x\to1$.

The default `SpectrumKernelConfig` is 512 threads and three energies per block, so the live specialization is `_kernel_3e`. The one-, two-, and four-energy specializations have the same ownership and reduction pattern with fewer or more private accumulators.

### Worked three-energy block

Use one default-shaped block with 512 threads, three energy bins $E=[0,1,2]$ eV, and 33 input slots. Only lines 0, 1, 2, and 32 are live:

| line $j$ | $E_{r,j}$ (eV) | $a_j$ (eV$^{-1}$) | $w_j$ |
|---:|---:|---:|---:|
| 0 | 0 | $\pi/2$ | 1 |
| 1 | 1 | $\pi/2$ | 2 |
| 2 | 2 | $\pi/2$ | 3 |
| 32 | 0 | $\pi$ | 4 |

All omitted slots have zero weight. Since `n_lines = 33` is less than 512, threads 0, 1, 2, and 32 each visit one live line. The important multi-energy mapping is:

| thread | visited line | calculations performed by that one thread |
|---:|---:|---|
| 0 | 0 | $j=0$ at $E_0$, $E_1$, and $E_2$ |
| 1 | 1 | $j=1$ at $E_0$, $E_1$, and $E_2$ |
| 2 | 2 | $j=2$ at $E_0$, $E_1$, and $E_2$ |
| 32 | 32 | $j=32$ at $E_0$, $E_1$, and $E_2$ |

For line 0, the three detunings are $x=[0,\pi/2,\pi]$. Its private contributions are therefore

$$
\mathbf a_0
=1\left[1^2,\left(\frac{2}{\pi}\right)^2,0^2\right]
=\left[1,\frac{4}{\pi^2},0\right].
$$

Repeating the same operations for the other live threads gives:

| thread | private `acc0` | private `acc1` | private `acc2` |
|---:|---:|---:|---:|
| 0 | $1$ | $4/\pi^2$ | $0$ |
| 1 | $8/\pi^2$ | $2$ | $8/\pi^2$ |
| 2 | $0$ | $12/\pi^2$ | $3$ |
| 32 | $4$ | $0$ | $0$ |
| all others | $0$ | $0$ | $0$ |

Each thread writes those values into three contiguous shared-memory regions:

```text
[energy 0 partials: 512 float32 values]
[energy 1 partials: 512 float32 values]
[energy 2 partials: 512 float32 values]
```

After `syncthreads`, all three regions are tree-reduced simultaneously. The strides are 256, 128, 64, 32, 16, 8, 4, 2, and 1. At stride 32, thread 0 folds thread 32's partial into its own. At the final stride, thread 0 folds thread 1's remaining partial. The same additions happen independently in every energy region.

Thread 0 finally writes

$$
\begin{aligned}
I_0 &\mathrel{+}= 5+\frac{8}{\pi^2}=5.810569,\\
I_1 &\mathrel{+}= 2+\frac{16}{\pi^2}=3.621139,\\
I_2 &\mathrel{+}= 3+\frac{8}{\pi^2}=3.810569.
\end{aligned}
$$

The wrapper permits an existing `out` array because `lines.py` streams multiple deterministic prologue batches through the kernel. Each launch adds its block totals to the already accumulated spectrum.

### Strided-line variant

If the same example used `n_lines = 513` and line 512 were live, thread 0 would first evaluate line 0 at all three energies, advance by 512, then evaluate line 512 at all three energies. It would add both contributions into its three private accumulators before shared memory is touched. This is how one thread handles both multiple lines and multiple energy bins without any synchronization inside the line loop.

## Coherent field reduction

### Quantity evaluated

The coherent prologue provides, for every line $j$:

- the same $E_{r,j}$ and $a_j$ line-shape inputs;
- `phase_slope[j]` $=p_j$ and `g_phase[j]` $=g_j$;
- complex coefficients $C_{s,j}$ and $C_{p,j}$, split into real and imaginary arrays for the two photon polarizations.

For vacuum propagation, the kernel phase and real line shape are

$$
\phi_{jk}=p_jE_k-g_j,
\qquad
q_{jk}=\frac{\sin[a_j(E_k-E_{r,j})]}
{a_j(E_k-E_{r,j})}.
$$

With in-medium dispersion enabled, the launch-uniform branch changes only the phase:

$$
\phi_{jk}=p_jE_k-g_j-L_{{\rm esc},j}\,\delta\omega_k.
$$

For each energy, every thread keeps four private accumulators: real and imaginary parts of the $s$- and $p$-polarized fields. For polarization $r\in\{s,p\}$ it performs the explicit complex multiply

$$
F_{r,k}=\sum_j q_{jk}C_{r,j}e^{i\phi_{jk}}.
$$

Only after the four field components have been reduced across the block does thread 0 add the intensity

$$
I_k\mathrel{+}=w_m\left[
(\operatorname{Re}F_{s,k})^2+(\operatorname{Im}F_{s,k})^2+
(\operatorname{Re}F_{p,k})^2+(\operatorname{Im}F_{p,k})^2
\right],
$$

where $w_m$ is the mosaic-orientation weight. Reflections and mosaic orientations remain separate squaring boundaries and are incoherent with one another. A normal flat-field launch contains a whole reflection/orientation row. When inter-electron decoherence is active, the grouped path can instead launch once for each electron's subset of that row, sum those per-electron squares, and blend the grouped result with the flat-row square.

The default `CoherentKernelConfig` is 256 threads and two energies per block, so `_kernel_2e` gives each thread eight private accumulators: four field components for each of two energies.

### Worked two-energy block

Use $E=[0,1]$ eV, vacuum phase, $w_m=1/4$, and two lines:

| $j$ | $E_{r,j}$ (eV) | $a_j$ (eV$^{-1}$) | $p_j$ | $g_j$ | $C_{s,j}$ | $C_{p,j}$ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | $\pi/2$ | 0 | 0 | $1+0i$ | $2+0i$ |
| 1 | 1 | $\pi/2$ | 0 | $-\pi/2$ | $1+0i$ | $2+0i$ |

Thread 0 processes line 0 at **both** energies. Its phase is zero, and its line-shape values are $[1,2/\pi]$, so its eight private values are

$$
\begin{array}{c|rrrr}
 & \operatorname{Re}F_s & \operatorname{Im}F_s &
   \operatorname{Re}F_p & \operatorname{Im}F_p\\ \hline
E_0 & 1 & 0 & 2 & 0\\
E_1 & 2/\pi & 0 & 4/\pi & 0
\end{array}
$$

Thread 1 processes line 1 at both energies. Since $\phi=-g_1=\pi/2$, the complex phase rotates its real coefficients onto the imaginary axis. Its line-shape values are $[2/\pi,1]$:

$$
\begin{array}{c|rrrr}
 & \operatorname{Re}F_s & \operatorname{Im}F_s &
   \operatorname{Re}F_p & \operatorname{Im}F_p\\ \hline
E_0 & 0 & 2/\pi & 0 & 4/\pi\\
E_1 & 0 & 1 & 0 & 2
\end{array}
$$

The kernel lays these partials into eight shared-memory regions of 256 `float32` values and tree-reduces them with strides 128 through 1. After the last stride, both energies have the same squared field norm:

$$
\lVert F_k\rVert^2
=1+\frac{4}{\pi^2}+4+\frac{16}{\pi^2}
=5+\frac{20}{\pi^2}
=7.026424.
$$

Thread 0 applies the mosaic weight only after squaring and adds

$$
I_0\mathrel{+}=I_1\mathrel{+}=
\frac14\left(5+\frac{20}{\pi^2}\right)=1.756606.
$$

This placement of the square is the defining difference from the incoherent kernel. If two equal scalar fields were in phase, coherent reduction would produce $\lvert1+1\rvert^2=4$ while incoherent reduction would produce $\lvert1\rvert^2+\lvert1\rvert^2=2$. With the $\pi/2$ relative phase used above, their cross term vanishes instead.

## Tail blocks and memory sizing

For $N_E=5$ with two energies per block, the coherent wrapper launches three blocks. Blocks 0 and 1 own `(0, 1)` and `(2, 3)`. Block 2 sets `k0 = 4` and `k1 = 5`; `has1` is false, so no thread reads or writes energy 5. The incoherent three-energy specialization uses corresponding `has_k1` and `has_k2` guards.

Dynamic shared memory is allocated exactly as

$$
\begin{aligned}
M_{\rm incoherent} &= B_E N_t\operatorname{sizeof}(\texttt{float32}),\\
M_{\rm coherent} &= 4B_E N_t\operatorname{sizeof}(\texttt{float32}).
\end{aligned}
$$

At the defaults, those are $3\times512\times4=6144$ bytes and $4\times2\times256\times4=8192$ bytes per block, respectively. The coherent factor of four is the real/imaginary pair for each of two polarizations.

## Reading and changing these kernels safely

- All arithmetic and shared storage are `float32`. Reassociation from the tree reduction means CPU scalar sums need tolerances, not bitwise comparison.
- Supported thread counts are powers of two from 32 through 1024; the halving reduction assumes that shape.
- The energy specializations are deliberately explicit. Changing one requires checking its siblings and the `_REDUCTION_KERNELS` dispatch table.
- Incoherent zero-weight lines are skipped. The coherent arrays are already filtered by their prologue and have no analogous `w` input.
- Every coherent launch contains lines from at most one reflection/orientation row. Its squaring boundary is either that whole row or, under inter-electron decoherence, one electron's subset of it. Moving the square across reflection/orientation rows would introduce interference between them and would be a physics change, not a kernel refactor.
- `L_esc` and `delta_omega` must be supplied together. Omitting both preserves the vacuum-phase path behind a launch-uniform branch.
- Focused behavioral anchors live in [`test_coherent_emission.py`](../../../tests/montecarlo/test_coherent_emission.py) and the CUDA spectrum tests under `tests/montecarlo/`. The broader physical limits and outstanding validation boundaries are documented in [Coherent-emission tracking](../../physics/radiation-physics/coherent-emission.md).
