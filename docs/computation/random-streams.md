# Random number streams

PyRITE separates random draws by physical input and transport process. Enabling a beam spot, energy spread, bunch length, or energy-loss straggling does not consume draws from the free-path and scattering stream.

For input distributions, this leaves otherwise unaffected arrays bit-for-bit identical. Straggling changes the subsequent energy history, but preserves the first row's free-path and scattering draws. Disabling it preserves the deterministic-loss path exactly.

## Two independent mechanisms

The package uses two stream mechanisms:

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
  - Per-electron transport cores, the Urban straggling sampler, and the
    per-electron draws inside each beam and bunch child
  - Letting a thread, row, or electron block compute its own draws with no
    shared generator or draw-order coupling
```

They compose: the child tree partitions host-side input sampling by physical input, while counter addressing partitions randomness across electrons — inside each beam and bunch child as well as in the cores — and, for straggling, across physical flights and numerical substeps. The straggling namespace is derived from the counter-addressed electron key; it is not a `SeedSequence` child and never advances the run generator.

## The `SeedSequence` child tree

Each optional beam or bunch distribution draws from its own child of the run seed, obtained by spawning a fixed number of children and taking the last. The index assignment is fixed and documented in the code, because it is part of the reproducibility contract rather than an implementation detail.

Inside a beam or bunch child, draws are counter-addressed per electron (`child_stream_root`, `counter_normals` in `montecarlo/transport/kinematics.py`). The child's first 64-bit state word roots the stream, and draw $c$ of electron $e$ is the counter construction {eq}`eq-streams-counter-address` with a 52-bit mantissa, $u = (\lfloor z/2^{12} \rfloor + \tfrac12)\,2^{-52} \in (0, 1)$. Gaussian draws are $\Phi^{-1}(u)$, one uniform per normal; the representable range truncates the tails at $|z| \approx 8.2$. A per-electron draw therefore depends on $(\text{seed}, e, c)$ only, never on the electron count. The groove phase, which only the lockstep core consumes, stays on a sequential generator.

```{list-table} Child-stream assignment by physical input.
:name: tbl-streams-children
:header-rows: 1

* - Child
  - Owner
  - Draws
* - `spawn(2)[1]`
  - Transverse beam spot
  - Gaussian entry offsets: counter draws 0 (x) and 1 (y)
* - `spawn(3)[2]`
  - Groove lateral phase
  - Uniform phase over one groove period (sequential; lockstep only)
* - `spawn(4)[3]`
  - Longitudinal bunch
  - Arrival-time offsets: counter draws 0 (Gaussian or train centre),
    1 microbunch, 2 jitter, 3 envelope, 4 mixture uniform
* - `spawn(5)[4]`
  - Courant–Snyder transverse distribution
  - Correlated positions and slopes: counter draws 0, 1 (x plane) and 2, 3
    (y plane)
* - `spawn(6)[5]`
  - Relative energy spread
  - Per-electron energy deviation: counter draw 0
```

The main free-path and scattering-angle draws use the run generator directly and are never spawned from, which is what makes the children disjoint from transport rather than merely different.

### What separate child streams preserve

Drawing beam offsets from the transport generator would shift every later free-path and scattering draw. Separate child streams let comparisons isolate the effect of the input distribution.

With a disjoint child, the transport stream is untouched. In the laterally infinite limit the spot is a rigid per-electron translation, so every returned array except the segment midpoints is **identical bit-for-bit** to the run with no spot at all, and the midpoints differ only by that constant offset.

The same argument gives every optional distribution a clean zero limit:

* `beam_fwhm_mm → 0` recovers the point source exactly, because $\sigma \to 0$ gives identically zero offsets;
* `bunch_length_fs → 0` and an unset distribution both give an all-zero arrival-offset array and reproduce the point bunch exactly;
* zero emittance gives zero slopes, so every direction is the shared beam direction *exactly*, which is what makes the collimated limit bit-for-bit rather than merely close;
* an unset energy spread leaves the monoenergetic beam untouched.

For an elliptical spot, both widths scale the same per-electron standard normals (draw 0 for $x$, draw 1 for $y$). Equal widths therefore use the same draws as the scalar-width case and produce bit-for-bit identical offsets.

The counter-addressed beam and bunch streams replaced sequential `Generator` draws in #361, a one-time change to the seed realizations of finite-spot, Twiss, energy-spread, and bunched runs. The sampling laws and limits are unchanged, and point-beam runs are unaffected.

### Where inertness stops

**A finite crystal footprint breaks it.** Sampled transverse positions classify missed entries and can cause side-face exits, and segment positions feed the six-face escape attenuation. With a finite footprint, beam size changes the emitted spectrum.

**Coherent emission breaks it.** The incoherent policy reads only $|A|^2$ per segment and never reads transverse position, which is why a spot is inert there. The coherent phase reads segment position directly, so a constant per-electron transverse offset enters every cross-electron term. Turning on a micron-scale spot drops the coherent peak height by a large factor against the point source, and the residual is one speckle realization whose peak scatters substantially seed to seed — a contrast that does **not** fall as the electron count grows, because the transverse form factor that should average those terms away is not implemented. This is tracked as an open discrepancy; consult [coherent emission](../physics/radiation-physics/coherent-emission.md) before using a finite spot with coherent emission.

## Counter-addressed streams

The lockstep core consumes one generator in a fixed order across steps and electrons. The per-electron and CUDA cores instead compute each draw from its address, so electrons can run independently to completion.

Draw $c$ of electron $e$ is

```{math}
:label: eq-streams-counter-address

k_e = \operatorname{splitmix64}\!\bigl(\text{seed} + \Phi\,(e+1)\bigr),
\qquad
u(e, c) = \bigl(\operatorname{splitmix64}(k_e + \Phi\,(c+1)) \gg 11\bigr)\,2^{-53},
```

with $\Phi = \texttt{0x9E3779B97F4A7C15}$ and SplitMix64's finalizer, which is a bijection on 64 bits and passes BigCrush in counter mode{cite:p}`steele2014`.

Keys are built on the host so both cores address the same streams. `stream_keys(seed, Ne, start=s)` returns the keys of electrons $[s, s + N_e)$, which equal `stream_keys(seed, s + Ne)[s:]` exactly; the hard-collision and radiative key domains take the same offset. The construction uses integer arithmetic and one exact `uint64 → double` conversion: the shifted value is below $2^{53}$, so the conversion is lossless.

The generator is therefore **bit-for-bit identical on host and device**, and it is asserted against an independent pure-Python SplitMix64 rather than against itself.

### What counter addressing makes invariant

Because a draw is a pure function of $(\text{seed}, e, c)$ and nothing else, a run is independent of:

* **CUDA launch geometry** — the same result at 32, 128, or 512 threads;
* **batch size** — how many electrons share a launch;
* **capacity replay** — an overflowing batch replayed at larger capacity reproduces itself exactly.

Exact replay supports the [capacity policy](execution-and-acceleration.md#output-addressing-and-capacity-replay) and device out-of-memory recovery: a batch can be rerun with more capacity or with downloaded segments without changing its result.

### Electron blocks and prefix stability

Adaptive electron counts (#361) extend a run block by block, so a run over $[0, N)$ must equal the same run assembled from blocks $[kB, (k+1)B)$. Every input the per-electron and CUDA cores consume is prefix-stable:

* transport, hard-collision, radiative, and straggling keys take the block's start offset;
* spot, Twiss, and energy-spread draws are counter-addressed inside their children;
* the cores emit rows electron-major, so concatenating blocks reproduces the single call's row order;
* every block builds its shell, radiative, and lookup tables over the population's energy range (`_energy_range_keV`), not its own, so interpolation is identical.

Two population-wide steps are redone once after the join (`montecarlo/runner/block_transport.py`). Bunch offsets are centred on the realized population's centroid, which no block knows; each raw offset is still a pure function of $(\text{seed}, e)$. Hard-photon directions come from one sequential generator over photon rows.

The lockstep core shares one generator across electrons and is not supported. Grooves, GDF beams, and secondary cascades are rejected in blocks.

### The straggling namespace

Urban straggling consumes a variable number of uniforms per material row, so a single per-electron counter would make the draw for a later row depend on every earlier Poisson count. Instead it creates two more address levels:

1. `_urban_stream_key_scalar` hashes the electron's `stream_keys(seed, Ne)` key with a fixed straggling salt. This makes the namespace disjoint from the free-path and scattering counter while retaining the same per-electron root.
2. `_urban_flight_key_scalar` packs `(flight_id, substep_id)` into a collision- free 32/32-bit index and hashes it again. Each row starts its local draw counter at zero.

Thus a straggling variate is a pure function of `(seed, electron_id, flight_id, substep_id, local_counter)`. Variable Poisson and continuum loop lengths are confined to that row. They cannot shift another row, another electron, or a pre-existing transport draw. The 32-bit fields are far wider than any reachable `max_steps` or substep count.

This addressing also defines the cross-core contract. Given the same row state and `(seed, electron, flight, substep)`, every host core calls the same sampler; the exact CUDA kernel carries a transcription of the same key derivation. Hardware-gated tests cover disabled-path identity, replay, finite energy bookkeeping, first-row transport state, and ensemble agreement. The first-row test currently compares the state *entering* the loss update, not `E_end_keV`, so exact host/device parity of one sampled loss remains an anchor gap; whole trajectories retain only the distributional contract because libm rounding can change a Poisson quantile.

## Draw order as a contract

The lockstep and per-electron cores implement the same models with the same draw semantics, and consume differently-ordered streams. They therefore realize **different samples of the same distribution**.

The straggling namespace does not make whole trajectories equal across those cores: their elastic streams already produce different rows. It guarantees the narrower property that the Urban draw attached to an otherwise identical row has the same address and that enabling straggling does not consume a free-path or scattering variate. Once the first random loss changes the energy, later trajectory states may diverge physically.

Comparisons must account for those different stream assignments:

* `per-electron` versus `cuda`: same stream addressing, so they may be compared numerically on the first-row entering state and statistically over complete straggled trajectories. The current hardware anchor bounds summed sampled loss to 2% and cutoff counts to 5% (or two electrons); it does not establish field-by-field equality. What separates the implementations before trajectory divergence is math-library rounding alone.
* Either versus `lockstep`: different stream order, so compare ensemble means with statistical error bars.

[Statistical methods](statistical-methods.md) covers how such a comparison is actually performed.

## Seeding and provenance

A seed reaches a case as an ordinary case parameter and is recorded with the result, so a checkpoint carries the seed that produced it.

The transport core is not part of a case's physical parameters. Runs with the same seed and physics can therefore hash identically while producing different realizations on different cores.

Pin `PYRITE_MC_TRANSPORT_CORE` when reproducing a result. A comparison between `per-electron` and `cuda` preserves stream addressing; a comparison with `lockstep` also changes draw order.

## Adding a new random input

Choose the mechanism from where the randomness is consumed:

* a fixed-shape host-side input distribution takes a new `SeedSequence` child;
* a variable-trip in-core process takes a new salted counter namespace, with stable physical identifiers in its address.

Neither shares an existing namespace or draws from the transport generator.

Sharing with transport would shift free-path and scattering draws whenever the input is enabled. Sharing with a sibling distribution would make one input's sample depend on whether the other is enabled.

Tests must cover:

1. **A zero-limit test asserting bit-for-bit identity**, not closeness — the disabled or zero-parameter case must reproduce the previous run exactly, array by array.
2. **A limiting-case test** for the physics itself, per the [validation methodology](../validation/methodology.md).

An in-core counter namespace additionally needs an offline address-replay test, plus invariance under batching and capacity replay. If it has a device twin, the host and device must reconstruct the same keys independently.

Correlated quantities from one distribution must share a stream. The Courant–Snyder policy draws positions and slopes together for this reason — splitting them across two children would break the position–slope correlation the emittance encodes. "One child per physical input" is the rule, not "one child per array".

## Validation

The claims on this page are pinned by the ledger rows `gpu-transport-core` (counter addressing, host/device generator identity, invariance to launch geometry and batching), `beam-phase-space-injection` (the transverse child streams and their zero limits), `longitudinal-bunch-sampling` (the bunch child stream and its point-bunch limit), and `energy-loss-straggling` (the salted per-row namespace, disabled-path inertness, and offline replay).

The generator's own properties, the three invariances, and the pure-Python cross-check live in `tests/montecarlo/test_transport_per_electron.py`; straggling replay and inertness live in `tests/montecarlo/test_straggling_rng_plumbing.py` and the remaining-core replay tests. Offset keys, the beam and bunch counter streams, and bit-for-bit block invariance live in `tests/montecarlo/test_block_transport.py`.
