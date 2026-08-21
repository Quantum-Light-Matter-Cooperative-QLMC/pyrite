# Precision and tolerances

Which floating-point type each stage of a run uses, why the choices differ, and
what vocabulary the tests and validation records use when they say two numbers
agree.

This page exists because "the GPU result differs" is not one question but four,
and they have different answers: a *type* difference, a *library* difference, a
*reassociation* difference, and a *realization* difference. Only the last is
about physics.

## The working real type

Transport and the spectrum kernels have different precision policies, and the
split is deliberate.

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

Transport arithmetic is `float64` on every core, and the device kernel is
textually identical to the host core so that **any divergence is attributable to
the math library alone**. That attribution is the whole verification strategy
for the port; a type difference layered on top would destroy it.

`float32` would roughly double throughput on a consumer card, and it is still
not offered. Transport is chaotic — a scattering-angle difference compounds over
the several hundred events an electron undergoes — so single precision would
change trajectories *qualitatively* rather than at rounding scale. That is a
different physical sample, not a cheaper one, and it would not be visible as a
tolerance failure in any test that looks at aggregates.

The segment payload therefore stays `float64` even when it is left resident on
the device for the spectrum kernels, which is why residency costs the device
memory it does. Compacting the join to the spectrum working type would nearly
halve that, at the price of changing the dtype the transport function documents.

### Why the spectrum kernels are

The spectrum reductions are sums of many comparable-magnitude terms into energy
bins. Their conditioning is good, their working set is large, and `float32`
halves both the bandwidth and the transient footprint that bounds the chunk
size. The cast to the working type happens exactly **once**, at device staging,
rather than once per kernel — three kernels over one case previously performed
the same cast independently, and hoisting it left the spectra bit-for-bit
identical while removing two thirds of the uploads.

The exception is the global coherent sum, whose conditioning is *not* good. That
is a property of the estimator rather than of the precision, and it is treated
in [statistical methods](statistical-methods.md#coherent-versus-incoherent-statistics).

## Four kinds of difference

When two runs disagree, the useful first question is which of these applies.

**Type.** The same expression evaluated in `float32` and `float64`. The ratio of
machine epsilons is about $2 \times 10^{7}$, so a well-conditioned reduction that
agrees to $10^{-14}$ in double should be expected near $10^{-7}$ in single. This
is predictable and can be checked by rerunning with `PYRITE_FP64=1`.

**Library.** The same expression in the same type, evaluated by different
implementations of `log`, `exp`, `pow`, `sin`, `cos`. CUDA's are accurate to a
few ulp but are not the host libm's. This difference is irreducible: it cannot
be engineered away, only bounded.

**Reassociation.** The same terms summed in a different order or grouping.
Exact in real arithmetic, not in floating point. This is what chunking, block
streaming, and expression-tree differences produce.

**Realization.** A different sample of the same distribution, because the random
draws were consumed in a different order. Not a floating-point effect at all,
and not bounded by any tolerance — it is bounded by
$1/\sqrt{N}$ statistics instead. See
[random number streams](random-streams.md) and
[statistical methods](statistical-methods.md).

The first three shrink as precision improves. The fourth does not, and reporting
it as a numerical discrepancy is the most common category error in reading a
core comparison.

## Float32 reassociation

The streaming coherent kernel is where single precision meets a genuinely
reordered reduction, so it gets its own accounting.

The streaming coherent path introduces two reassociations against its
predecessor, and both are tracked as validation debt rather than assumed
harmless.

1. **Expression-tree rounding.** The RawKernel's interpolation and compiler
   expression tree may round differently from the elementwise path it replaced.
2. **Block partials.** A case larger than one internal segment block reduces
   each block's field first and adds the block partials afterwards, rather than
   performing one monolithic reduction.

The second is the interesting one, because it is the streaming identity
{eq}`eq-execution-streaming-identity` in
[execution and acceleration](execution-and-acceleration.md) doing exactly what
it promises — and floating-point addition is not associative, so "exactly" holds
in real arithmetic only.

What makes this acceptable is that the partitioning happens on the **field**,
not on the intensity. Intensity is formed once, after the last block, from four
persistent field planes. Had the kernel reduced intensity per block, the
cancellation structure of the coherent sum would have been destroyed rather than
merely reordered, which is a physics change and not a rounding one.

The measured envelope: a `float64` A/B against the previous path agrees to the
level expected of double precision, and the on-device `float32` sibling
comparison measured $2.0 \times 10^{-5}$ relative. That is consistent with the
epsilon-ratio expectation for a sum with real cancellation in it, and it is the
figure to compare a future change against.

Both reassociations require CUDA goldens and an A/B before human sign-off, and
that regeneration is still owed. Until then, the `float32` coherent spectrum is
correct-in-envelope rather than pinned.

## Tolerance vocabulary

Different claims are checked at different tolerances, and the number alone does
not say what was tested. The four that recur:

`rtol=1e-12`, single step
: Used for **first-step agreement** between the CPU per-electron core and the
  CUDA kernel. It is a tight tolerance applied to a single evaluation, before
  chaos has anything to amplify. Even this is not exact, because the first
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

The rule that ties them together: **choose the tolerance from what is claimed,
not from what passes.** A comparison across a stream change cannot be rescued by
loosening a numerical tolerance, because no tolerance is the right instrument
for it.

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
CUDA straggling remains hardware-unverified, so no host/device tolerance is
promoted here beyond the existing hardware-gated tests.

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
