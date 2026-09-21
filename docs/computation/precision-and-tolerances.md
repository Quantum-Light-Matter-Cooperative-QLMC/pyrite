# Precision and tolerances

Transport and spectrum calculations use different floating-point policies.
When results differ, distinguish changes in type, math-library implementation,
summation order, and sampled trajectories. Each requires a different comparison.

## The working real type

```{list-table} Floating-point policy by stage.
:name: tbl-precision-policy
:header-rows: 1

* - Stage
  - Host
  - Device
  - Configurable
* - Electron transport (all three cores)
  - `float64`
  - `float64`
  - no
* - Transport lookup tables
  - `float64`
  - `float64`
  - no
* - Spectrum kernels (line, coherent, bremsstrahlung)
  - `float64`
  - `float32`
  - `PYRITE_FP64=1`
* - Segment payload as produced by transport
  - `float64`
  - `float64`
  - no
```

The spectrum kernels share one module-level working type, resolved once at
import: `float32` on an accelerator, `float64` otherwise, with `PYRITE_FP64=1`
forcing `float64` everywhere. That single binding is what the chunk-sizing
arithmetic reads for its item size, which is why a device run admits roughly
twice the chunk a host run does from the same byte budget — see
{eq}`eq-memory-chunk` in
[memory, chunking, and scheduling](memory-and-scheduling.md).

Two process-level overrides interact with it. A worker pinned to the CPU
rebinds the working type to `float64` in its own process only, never touching
the driver's globals, so a pooled run's CPU workers compute in double even on a
machine with a GPU. The driver has a matching scoped context for the same
purpose. Both are local mutations by construction; a run does not end up with
two types live in one reduction.

### Why transport is not configurable

Transport arithmetic and lookup tables use `float64` on every core.
The CPU per-electron reference and CUDA kernel use the same algorithm and
precision, allowing comparisons of their math-library behavior.

Small rounding differences in scattering angles can grow over hundreds of
events. Keeping transport precision fixed avoids adding a type difference to
those comparisons. Segment arrays also remain `float64` when retained on the
device for spectrum calculations.

### Why spectrum precision is configurable

Spectrum reductions have large working sets. Using `float32` halves the
storage per value and the transient footprint used to size chunks. Device
staging casts each shared segment array to the working type once for reuse
across kernels.

The global coherent sum is sensitive to cancellation. Its conditioning is
a property of the estimator and is discussed
in [statistical methods](statistical-methods.md#coherent-versus-incoherent-statistics).

## Four kinds of difference

When two runs disagree, the useful first question is which of these applies.

**Type.** The same expression evaluated in `float32` and `float64`. The ratio of
machine epsilons is about $2 \times 10^{7}$, so a well-conditioned reduction that
agrees to $10^{-14}$ in double should be expected near $10^{-7}$ in single. This
is predictable and can be checked by rerunning with `PYRITE_FP64=1`.

**Library.** The same expression in the same type, evaluated by different
implementations of `log`, `exp`, `pow`, `sin`, `cos`. Host and device libraries can round these functions differently even at the
same precision.

**Reassociation.** The same terms summed in a different order or grouping.
Exact in real arithmetic, not in floating point. This is what chunking, block
streaming, and expression-tree differences produce.

**Realization.** A different sample of the same distribution, because the random
draws were consumed in a different order. Not a floating-point effect at all,
and not bounded by any tolerance — it is bounded by
$1/\sqrt{N}$ statistics instead. See
[random number streams](random-streams.md) and
[statistical methods](statistical-methods.md).

Higher precision can reduce arithmetic differences. It does not remove
differences between independently sampled trajectories.

## Float32 reassociation

The streaming coherent kernel has two sources of rounding differences compared
with an elementwise, unblocked reduction. Both remain tracked as validation debt.

1. **Expression-tree rounding.** The RawKernel's interpolation and compiler
   expression tree may round differently from the elementwise path.
2. **Block partials.** A case larger than one internal segment block reduces
   each block's field first and adds the block partials afterwards, rather than
   performing one monolithic reduction.

Block accumulation uses {eq}`eq-execution-streaming-identity` from
[execution and acceleration](execution-and-acceleration.md). The identity is
exact in real arithmetic, but floating-point addition is not associative.

The kernel accumulates fields in four persistent planes and forms intensity
once after the final block. Forming intensity per block would lose cross-block
interference and change the physical calculation.

The measured envelope: a `float64` A/B against the non-streaming path agrees to the
level expected of double precision, and the on-device `float32` sibling
comparison measured $2.0 \times 10^{-5}$ relative. That is consistent with the
epsilon-ratio expectation for a sum with real cancellation in it, and it is the
figure to compare a future change against.

Both reassociations require CUDA golden regeneration and an A/B comparison
before human sign-off. These checks remain outstanding; the measured envelope
does not replace them.

## Tolerance vocabulary

Different claims are checked at different tolerances, and the number alone does
not say what was tested. The four that recur:

`rtol=1e-12`, single step
: Used for **first-step agreement** between the CPU per-electron core and the
  CUDA kernel. It is a tight tolerance applied to a single evaluation, before
  rounding differences have accumulated along a trajectory. Even this is not exact, because the first
  recorded segment already passes through a logarithm.

Bit-for-bit
: Reserved for claims that are *structurally* exact rather than numerically
  close: the random generator across host and device (integer arithmetic only),
  invariance to launch geometry and batch size and capacity replay, the
  zero-limit of every optional distribution, the default/explicitly-disabled
  straggling path, and staging changes that only move where an identical cast
  happens. Where the documentation says bit-for-bit, a test asserts equality
  rather than closeness. This does not claim bit identity between a straggled
  and deterministic trajectory: the physical inputs differ.

4σ, aggregate over seeds
: Used to compare two cores or two configurations that produce **different
  realizations**. It is a statistical statement about ensemble means with their
  Monte Carlo errors, not a numerical one, and it is the only meaningful form of
  comparison across a stream change.

Envelope figures
: Quoted relative agreements such as the $2.0 \times 10^{-5}$ above. These
  describe a measured difference at a stated configuration; they are evidence,
  not thresholds, and they do not gate anything on their own.

Choose the tolerance to match the claim. Comparisons across different random
streams require statistical errors, not a looser numerical tolerance.

### Straggling does not relax numerical tolerances

The Urban sampler adds physical variance; it does not turn arithmetic error
into Monte Carlo error. Its key derivation and integer uniforms are exact. The
channel rates, Poisson inversion and continuum marks use floating-point
transcendentals, so host/device evaluation may differ by libm ulps even when the
address is identical. After a loss changes the energy, transport chaos can
amplify that rounding into a different later trajectory. Compare one draw or
first-row state numerically, but compare whole straggled runs statistically.

Conversely, a straggling on/off comparison at matched seeds is a paired physics
comparison, not a numerical tolerance test. The disabled-path claim is the one
that is bit-for-bit: the sampler, its keys and its output field are absent.
CUDA straggling tests cover disabled-path
identity, deterministic replay, finite energy bookkeeping, first-row transport
state, and ensemble agreement. They do not yet compare the first row's applied
loss (`E_end_keV`) against the host, so the `rtol=1e-12` loss-parity claim is not
promoted beyond the existing transport-state check.

## Practical guidance

* To distinguish a type difference from anything else, rerun with
  `PYRITE_FP64=1`. If the disagreement collapses toward `float64` levels, it was
  precision.
* To distinguish a library difference from a realization difference, pin
  `PYRITE_MC_TRANSPORT_CORE=per-electron` and compare against `cuda`. Those two
  address identical random streams, so what remains is arithmetic. Comparing
  either against `lockstep` mixes in a realization change.
* To rule out a chunking reassociation, pin an explicit `spec_chunk` on both
  arms. Chunk size never changes physics, so a difference that moves with it is
  a rounding artifact.
* A device run whose backend fell back to NumPy computes in `float64` and will
  not reproduce a `float32` device golden. Check the fallback reason in the
  runtime plan before concluding anything about the numbers.

## Validation

Precision policy carries no ledger row of its own; it is a property of the
implementation rather than a physical claim. Where precision affects a ledgered
claim, the row records it: `coherent-line-hkl-batch` documents the `float32`
envelope and the owed golden regeneration, and `gpu-transport-core` documents
the host/device arithmetic difference and its statistical consequence.
`energy-loss-straggling` records disabled-path identity, address replay, and
the statistical observable comparison without weakening any existing numeric
tolerance.

The tolerance conventions themselves serve the evidence standard in
[validation methodology](../validation/methodology.md), which is what decides
when a measured agreement is sufficient for sign-off.
