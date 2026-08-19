# Random number streams

Reproducibility here is a design constraint, not a convenience, and it is
achieved through **stream structure** rather than through a single global seed.

The property the structure buys is unusually strong and worth stating up front:
enabling an optional physical input — a finite beam spot, an energy spread, a
bunch length — takes no draw from the transport stream and leaves every other
array bit-for-bit identical. That is not an approximation that holds to
rounding. It is exact, and it is tested as exact.

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
  - Per-electron transport cores
  - Letting a thread compute its own draws with no shared generator
```

They compose: the child tree partitions randomness *across physical inputs*, and
counter addressing partitions it *across electrons* within the transport stream.

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
bijection on 64 bits and passes BigCrush in counter mode.[^splitmix]

[^splitmix]: Steele, Lea & Flood, *Fast splittable pseudorandom number
    generators*, OOPSLA 2014. [DOI:10.1145/2660193.2660195](https://doi.org/10.1145/2660193.2660195)

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

## Draw order as a contract

The lockstep and per-electron cores implement the same models with the same draw
semantics, and consume differently-ordered streams. They therefore realize
**different samples of the same distribution**.

This is structural and cannot be fixed. It also is not a defect — but it does
determine what may be compared how:

* `per-electron` versus `cuda`: same stream addressing, so they may be compared
  **numerically**, and at production scale they agree to 0.000 % on every
  recorded field. What separates them is math-library rounding alone.
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

The rule is short: **a new distribution takes a new child stream.** It never
shares an existing child, and it never draws from the transport generator.

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

One correlation caveat: when a single policy owns more than one moment of the
same distribution, it must own them from **one** stream. The Courant–Snyder
policy draws positions and slopes together for this reason — splitting them
across two children would break the position–slope correlation the emittance
encodes. "One child per physical input" is the rule, not "one child per array".

## Validation

The claims on this page are pinned by the ledger rows `gpu-transport-core`
(counter addressing, host/device generator identity, invariance to launch
geometry and batching), `beam-phase-space-injection` (the transverse child
streams and their zero limits), and `longitudinal-bunch-sampling` (the bunch
child stream and its point-bunch limit).

The generator's own properties, the three invariances, and the pure-Python
cross-check live in `tests/montecarlo/test_transport_per_electron.py`.
