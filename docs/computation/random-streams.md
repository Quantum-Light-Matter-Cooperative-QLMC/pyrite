# Random number streams

Reproducibility here is a design constraint, not a convenience, and it is
achieved through **stream structure** rather than through a single global seed.

The property the structure buys is unusually strong and worth stating up front:
enabling an optional physical input — a finite beam spot, an energy spread, a
bunch length, or energy-loss straggling — takes no draw from the pre-existing
transport stream. For the input distributions this leaves otherwise unaffected
arrays bit-for-bit identical. Straggling deliberately changes the later energy
history, but its first-row free-path and scattering draws remain identical and
the disabled path is bit-for-bit historical. These are exact, tested contracts,
not rounding-scale approximations.

## Two independent mechanisms

The package uses two different constructions, for two different problems.

```{list-table} The two stream mechanisms and what each solves.
:name: tbl-streams-mechanisms
:header-rows: 1

* - Mechanism
  - Where
  - Problem it solves
* - `SeedSequence` child tree
  - Beam and bunch sampling, host side
  - Keeping each optional physical input's draws off every other input's stream
* - Counter-addressed SplitMix64
  - Per-electron transport cores and the Urban straggling sampler
  - Letting a thread or row compute its own draws with no shared generator or
    draw-order coupling
```

They compose: the child tree partitions fixed-shape host-side input sampling,
while counter addressing partitions in-core randomness across electrons and,
for straggling, across physical flights and numerical substeps. The straggling
namespace is derived from the counter-addressed electron key; it is not a
`SeedSequence` child and never advances the run generator.

## The `SeedSequence` child tree

Each optional beam or bunch distribution draws from its own child of the run
seed, obtained by spawning a fixed number of children and taking the last. The
index assignment is fixed and documented in the code, because it is part of the
reproducibility contract rather than an implementation detail.

```{list-table} Child-stream assignment by physical input.
:name: tbl-streams-children
:header-rows: 1

* - Child
  - Owner
  - Draws
* - `spawn(2)[1]`
  - Transverse beam spot
  - Gaussian entry offsets, per plane
* - `spawn(3)[2]`
  - Groove lateral phase
  - Uniform phase over one groove period
* - `spawn(4)[3]`
  - Longitudinal bunch
  - Arrival-time offsets
* - `spawn(5)[4]`
  - Courant–Snyder transverse distribution
  - Correlated positions and slopes
* - `spawn(6)[5]`
  - Relative energy spread
  - Per-electron energy deviation
```

The main free-path and scattering-angle draws use the run generator directly and
are never spawned from, which is what makes the children disjoint from transport
rather than merely different.

### Why disjoint children make inertness provable

Consider enabling a finite beam spot. If the offsets were drawn from the
transport generator, every subsequent free path and scattering angle would shift
— not because the physics changed, but because the stream position did. The run
would be a different realization, and no test could separate the geometric
effect from the stream effect.

With a disjoint child, the transport stream is untouched. In the laterally
infinite limit the spot is a rigid per-electron translation, so every returned
array except the segment midpoints is **identical bit-for-bit** to the run with
no spot at all, and the midpoints differ only by that constant offset.

The same argument gives every optional distribution a clean zero limit:

* `beam_fwhm_mm → 0` recovers the point source exactly, because
  $\sigma \to 0$ gives identically zero offsets;
* `bunch_length_fs → 0` and an unset distribution both give an all-zero
  arrival-offset array — the legacy point bunch, bit-for-bit;
* zero emittance gives zero slopes, so every direction is the shared beam
  direction *exactly*, which is what makes the collimated limit bit-for-bit
  rather than merely close;
* an unset energy spread leaves the monoenergetic beam untouched.

There is one deliberate subtlety in the elliptical-spot case. Drawing a standard
normal of shape $(N_e, 2)$ and scaling each column afterwards consumes the
stream identically to drawing at a scalar width, so an isotropic beam stays
bit-for-bit compatible with the historical scalar-spot path even though the code
now supports independent widths per plane.

### Where inertness stops

The guarantee is scoped, and the scope matters.

**A finite crystal footprint breaks it.** Sampled transverse positions classify
missed entries and can cause side-face exits, and segment positions feed the
six-face escape attenuation. With a finite footprint, beam size changes the
emitted spectrum.

**Coherent emission breaks it.** The incoherent policy reads only $|A|^2$ per
segment and never reads transverse position, which is why a spot is inert there.
The coherent phase reads segment position directly, so a constant per-electron
transverse offset enters every cross-electron term. Turning on a micron-scale
spot drops the coherent peak height by a large factor against the point source,
and the residual is one speckle realization whose peak scatters substantially
seed to seed — a contrast that does **not** fall as the electron count grows,
because the transverse form factor that should average those terms away is not
implemented. This is tracked as an open discrepancy; consult
[coherent emission](../physics/radiation-physics/coherent-emission.md) before
using a finite spot with coherent emission.

## Counter-addressed streams

The lockstep transport core is written as an outer step loop over an inner
electron loop for one reason: a single generator serves every electron, so the
draws must be consumed in a fixed interleaved order. That order is exactly what
a device cannot reproduce, since one thread per electron runs its electron to
completion and therefore consumes its stream contiguously.

The two conventional fixes both make the result unverifiable. Per-thread cuRAND
states make results depend on how threads were assigned; an atomic output
counter makes segment order depend on scheduling. Either would force a full
golden regeneration on every launch-configuration change.

Randomness is therefore **addressed** rather than streamed. Draw $c$ of electron
$e$ is

```{math}
:label: eq-streams-counter-address

k_e = \operatorname{splitmix64}\!\bigl(\text{seed} + \Phi\,(e+1)\bigr),
\qquad
u(e, c) = \bigl(\operatorname{splitmix64}(k_e + \Phi\,(c+1)) \gg 11\bigr)\,2^{-53},
```

with $\Phi = \texttt{0x9E3779B97F4A7C15}$ and SplitMix64's finalizer, which is a
bijection on 64 bits and passes BigCrush in counter mode{cite:p}`steele2014`.

Two implementation choices follow from the requirement that host and device
agree exactly. Keys are built on the host, so device code needs no 64-bit
integer casts and both cores provably address the same streams. And the whole
construction is integer arithmetic plus one exact `uint64 → double` conversion —
the shifted value is below $2^{53}$, so the conversion is lossless.

The generator is therefore **bit-for-bit identical on host and device**, and it
is asserted against an independent pure-Python SplitMix64 rather than against
itself.

### What counter addressing makes invariant

Because a draw is a pure function of $(\text{seed}, e, c)$ and nothing else, a
run is independent of:

* **CUDA launch geometry** — the same result at 32, 128, or 512 threads;
* **batch size** — how many electrons share a launch;
* **capacity replay** — an overflowing batch replayed at larger capacity
  reproduces itself exactly.

The third is what makes the [capacity policy](execution-and-acceleration.md#output-addressing-and-capacity-replay)
safe: an overflow costs time and nothing else. It is also what makes the
device-OOM fallback safe, since replaying the same seed with segments downloaded
is exact rather than merely equivalent.

### The straggling namespace

Urban straggling consumes a variable number of uniforms per material row, so a
single per-electron counter would make the draw for a later row depend on every
earlier Poisson count. Instead it creates two more address levels:

1. `_urban_stream_key_scalar` hashes the electron's `stream_keys(seed, Ne)` key
   with a fixed straggling salt. This makes the namespace disjoint from the
   free-path and scattering counter while retaining the same per-electron root.
2. `_urban_flight_key_scalar` packs `(flight_id, substep_id)` into a collision-
   free 32/32-bit index and hashes it again. Each row starts its local draw
   counter at zero.

Thus a straggling variate is a pure function of
`(seed, electron_id, flight_id, substep_id, local_counter)`. Variable Poisson
and continuum loop lengths are confined to that row. They cannot shift another
row, another electron, or a pre-existing transport draw. The 32-bit fields are
far wider than any reachable `max_steps` or substep count.

This addressing also defines the cross-core contract. Given the same row state
and `(seed, electron, flight, substep)`, every host core calls the same sampler;
the exact CUDA kernel carries a transcription of the same key derivation. Five
hardware-gated tests pass on an RTX 5080 (driver 610.47, CuPy 14.1.1), covering
disabled-path identity, replay, finite energy bookkeeping, first-row transport
state, and ensemble agreement. The first-row test currently compares the state
*entering* the loss update, not `E_end_keV`, so exact host/device parity of one
sampled loss remains an anchor gap; whole trajectories retain only the
distributional contract because libm rounding can change a Poisson quantile.

## Draw order as a contract

The lockstep and per-electron cores implement the same models with the same draw
semantics, and consume differently-ordered streams. They therefore realize
**different samples of the same distribution**.

The straggling namespace does not make whole trajectories equal across those
cores: their elastic streams already produce different rows. It guarantees the
narrower property that the Urban draw attached to an otherwise identical row
has the same address and that enabling straggling does not consume a free-path
or scattering variate. Once the first random loss changes the energy, later
trajectory states may diverge physically.

This is structural and cannot be fixed. It also is not a defect — but it does
determine what may be compared how:

* `per-electron` versus `cuda`: same stream addressing, so they may be compared
  numerically on the first-row entering state and statistically over complete
  straggled trajectories. The current hardware anchor bounds summed sampled
  loss to 2% and cutoff counts to 5% (or two electrons); it does not establish
  field-by-field equality. What separates the implementations before
  trajectory divergence is math-library rounding alone.
* Either versus `lockstep`: different stream order, so they may be compared only
  **statistically**, with ensemble means and error bars. The measured difference
  is not small — of order a percent on line spectra and tens of percent on
  small-population coherent observables — and every one of those figures tracks
  $1/\sqrt{N}$ for its own population, which is the signature of a realization
  change rather than a physics change.

[Statistical methods](statistical-methods.md) covers how such a comparison is
actually performed.

## Seeding and provenance

A seed reaches a case as an ordinary case parameter and is recorded with the
result, so a checkpoint carries the seed that produced it.

The important asymmetry: **the transport core is not part of a case's physical
parameters.** Two runs with the same seed and the same physics on different
cores produce different realizations while hashing identically. That is correct
— the parameters really are the same — but it means a pinned spectrum taken at
more than 1000 electrons on a CUDA machine is not reproduced by the lockstep
core, and must be regenerated or compared statistically.

`PYRITE_MC_TRANSPORT_CORE` exists for exactly this. Pinning it for a process is
how a run reproduces a pre-threshold result, or bisects a host/device difference
by holding the stream order fixed while changing only the arithmetic.

## Adding a new random input

Choose the mechanism from where the randomness is consumed:

* a fixed-shape host-side input distribution takes a new `SeedSequence` child;
* a variable-trip in-core process takes a new salted counter namespace, with
  stable physical identifiers in its address.

Neither shares an existing namespace or draws from the transport generator.

The consequences the rule protects are worth restating, since sharing a stream
is the tempting shortcut and its damage is invisible until someone tries to
verify something:

* sharing with transport destroys inertness for *every* run that enables the new
  input, and cannot be undone by a later fix without changing results again;
* sharing with a sibling distribution couples two physically independent inputs,
  so enabling one silently changes the other's sample.

The accompanying test shape has two parts, and both are required:

1. **A zero-limit test asserting bit-for-bit identity**, not closeness — the
   disabled or zero-parameter case must reproduce the previous run exactly,
   array by array.
2. **A limiting-case test** for the physics itself, per the
   [validation methodology](../validation/methodology.md).

An in-core counter namespace additionally needs an offline address-replay test,
plus invariance under batching and capacity replay. If it has a device twin,
the host and device must reconstruct the same keys independently.

One correlation caveat: when a single policy owns more than one moment of the
same distribution, it must own them from **one** stream. The Courant–Snyder
policy draws positions and slopes together for this reason — splitting them
across two children would break the position–slope correlation the emittance
encodes. "One child per physical input" is the rule, not "one child per array".

## Validation

The claims on this page are pinned by the ledger rows `gpu-transport-core`
(counter addressing, host/device generator identity, invariance to launch
geometry and batching), `beam-phase-space-injection` (the transverse child
streams and their zero limits), `longitudinal-bunch-sampling` (the bunch child
stream and its point-bunch limit), and `energy-loss-straggling` (the salted
per-row namespace, disabled-path inertness, and offline replay).

The generator's own properties, the three invariances, and the pure-Python
cross-check live in `tests/montecarlo/test_transport_per_electron.py`;
straggling replay and inertness live in
`tests/montecarlo/test_straggling_rng_plumbing.py` and the remaining-core replay
tests.
