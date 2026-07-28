# Beam phase space & discrete longitudinal bunch support

> **Branch-scoped working doc** for `feature/discrete-bunch-support` (see
> [`tasks/README.md`](README.md)). On landing this branch, promote the durable
> beam-model design (§1–4, §6–7) into a proper `docs/beam-phase-space.md` note,
> then `git rm` this file. The checklist (§5), delegation plan (§9), and
> in-progress decisions (§8) are ephemeral and do not migrate to `docs/`.

## 1. Motivation

Every electron the simulation launches today is transversely distributed (a
Gaussian spot) but **longitudinally a delta function**: all electrons enter at
the same instant, with the same energy and the same direction. That is adequate
for the *incoherent* PXR/CBS observable we currently compute, but it blocks
three things the project wants:

- **Superradiant / coherent PXR/CBS** (TODO On-Hold #3) — coherent emission
  *across* a bunch requires knowing where each electron sits *along* the beam,
  because the coherent enhancement is set by the longitudinal bunch form factor
  `|F(ω)|²`.
- **Rep-rated flux estimates** for pulsed sources (REGAE-scale beams, TODO
  On-Hold #1: 3–5 MeV, ~50 fs, ~100 fC) — reporting detected flux per second
  needs charge-per-bunch × rep-rate, not the present per-nA-current scaling.
- **Standard beam-quality metrics** (emittance, energy spread, Twiss
  parameters, bunch length) that let us describe and compare beams the way an
  accelerator physicist would.

The **near-term deliverable is plumbing + metrics only**: initiate particles
with a specified longitudinal distribution (in addition to the existing
transverse one), thread it through the pipeline, and expose beam metrics.
Coherent emission is explicitly deferred (§6). Because the current segment sum
is incoherent, a per-electron longitudinal offset has **zero effect on the
emitted spectrum today** — exactly analogous to `beam_fwhm_mm` in the laterally
infinite-slab limit (ledger `finite-beam-size`). The value of landing it now is
that the distribution and its metrics become first-class, reproducible inputs,
ready for the coherent sum to consume.

## 2. Current state: the 6-D phase space audit

A beam is a distribution over the 6-D phase space
`(x, x', y, y', z/t, δ)` where `x'=dx/dz` etc. are divergence angles, `z/t` is
longitudinal position (equivalently arrival time), and `δ = ΔE/E` is the
fractional energy offset. Here is what `simulate_trajectories`
(`montecarlo/transport.py`) populates today:

| Coord | Meaning | Today | Knob | Status |
|-------|---------|-------|------|--------|
| `x, y` | transverse position | Gaussian spot | `beam_fwhm_mm` | ✅ ledgered `finite-beam-size` |
| `x', y'` | divergence / angle | all identical (`dirs = tile(beam_dir)`) | — | ❌ zero divergence → **transverse emittance gap** |
| `z / t` | longitudinal / arrival time | all at `t=0` (`clock = zeros(Ne)`) | — | ❌ **this task** |
| `δ = ΔE/E` | energy spread | mono-energetic (`E = full(Ne, E0)`) | — | ❌ zero spread → **longitudinal emittance gap** |

The 6-D coordinates are the *shape* of the beam; its central energy `E0` is the
zeroth-order beam property that sets `β`, every resonance, and the whole
spectrum. Today `E0` lives on `Sweep.energy_keV`; §4.1.1 moves it onto
`BeamSpec` so the beam is described in one object. The `δ` column above is the
*spread about* that central energy (future).

Key source anchors (`montecarlo/transport.py::simulate_trajectories`):

- `pos[:, :2] = project_beam_entry(offsets, …)` — transverse spot sampled from
  an independent RNG child stream `SeedSequence(seed).spawn(2)[1]`.
- `dirs = np.tile(beam_dir, (Ne, 1))` — one shared direction, zero divergence.
- `E = np.full(Ne, float(E0_keV))` — one shared energy.
- `clock = np.zeros(Ne)` — per-electron clock, **the injection point for a
  longitudinal offset**. `clock` accumulates `Σ L/β` and is recorded at each
  segment start as the electron *age*; it drives the penetration / lifetime
  plots (→ fs via `c = 2997.92 Å/fs`).

**Critical subtlety.** `clock` is currently *relative age since entry*. The
coherent phase later needs *absolute arrival time* `t_e^abs = Δt_e + age`. These
must be kept separate: folding the bunch offset `Δt_e` into `clock` would shift
every existing lifetime/penetration plot and break the `finite-beam`-style
bit-for-bit guarantee. So the bunch offset is stored as its **own per-electron
array** (`t0_ang`, propagated to a per-segment `t0_ang` in the returned dict),
never added into `clock`.

## 3. Physics: why it is a no-op now, and what coherent needs later

`mc_spectrum` sums intensities incoherently (Zhai SI Eqs. 5–7):

```
d²N/dE dΩ = Σ_segments |A|² t_L² sinc²[…] T_abs
```

There is no complex phase carrying the electron's absolute time, so shifting an
electron in `t` cannot change the sum. This is the same structural reason
`beam_fwhm_mm` is a pure geometry refinement in the infinite-slab limit.

The coherent extension (deferred) multiplies the coherent-line intensity of a
bunch of `N` electrons by a **longitudinal bunch form factor**

```
|F(ω)|² = |(1/N) Σ_e exp(i ω Δt_e)|²          (0 ≤ |F|² ≤ 1)
```

so the *coherent* yield scales like `N + N(N-1)|F(ω)|²`. `|F(ω)|² → 1` for a
bunch short compared to the radiation period (superradiance); `→ 1/N` for a
bunch long compared to it (the incoherent limit we compute today). A
pre-modulated (microbunched) beam yields `|F|²` peaks at the modulation harmonics
— the superradiance/FEL driver. **Landing the per-electron `Δt_e` now is exactly
the input this factor consumes later**; nothing about the incoherent result
changes until `mc_spectrum` grows the coherent branch.

## 4. Design

### 4.1 `BeamSpec` — one dataclass for the whole phase space

Introduce a frozen `BeamSpec` that owns the full beam description and **absorbs
the existing `beam_fwhm_mm`**. Staged fields (only the first two are wired now;
the rest are declared as documented extension points with inert defaults, so the
dataclass shape is stable across the roadmap):

```python
@dataclass(frozen=True)
class BeamSpec:
    # central energy (moved out of Sweep.energy_keV — §4.1.1) -------------
    energy_keV: ScalarOrSeq = (30.0, 45.0, 60.0)  # beam kinetic energy; the primary swept axis
    # transverse (already modelled today; now per-plane — decision 8) ----
    transverse_fwhm_x_mm: float | None = 1.0   # ← today's Sweep.beam_fwhm_mm (x)
    transverse_fwhm_y_mm: float | None = 1.0   # y; equal to x → isotropic spot
    # (an isotropic `transverse_fwhm_mm=…` convenience ctor sets both.)
    # longitudinal (THIS task) -------------------------------------------
    bunch_length_fs: float | None = None       # RMS σ_t; None → legacy point bunch
    long_shape: str = "gaussian"               # "gaussian" | "uniform" | "file"
    long_offsets_fs: tuple[float, ...] | None = None  # explicit per-particle Δt (overrides shape)
    rep_rate_hz: float = 5000.0                # detected-flux multiplier only (§4.5)
    bunch_charge_pc: float = 1.0               # single-bunch charge → peak-current / flux metrics
    # avg current I = bunch_charge_pc·rep_rate_hz is DERIVED here (decision 7),
    # superseding results.Settings.beam_current_na as the source of truth.
    # emittance / spread (FUTURE, §6; inert defaults keep runs bit-for-bit) --
    divergence_mrad: float | None = None       # x',y' RMS  → transverse emittance
    energy_spread_frac: float | None = None    # δ RMS      → longitudinal emittance
```

`BeamSpec` is the **single home for every beam property**: central energy,
transverse size, longitudinal bunch, rep-rate/charge, and (future) divergence
and energy spread. Whatever describes the *electrons before they hit the
crystal* belongs here. Note two "beam"-named fields that do **not** move:
`Sweep.beam_uvw` is a *crystal* axis (orientation), and `beam_dir` is derived
from sample tilt geometry — neither is a beam-quality knob.

#### 4.1.1 Energy moves out of `Sweep`

`Sweep.energy_keV` becomes `Sweep.beam.energy_keV`. It stays a `ScalarOrSeq`
and remains the main swept axis — `build_cases` takes the Cartesian product over
`sweep.beam.energy_keV` exactly as it does today over `sweep.energy_keV`, and
each case still resolves a scalar `E0_keV`. Only the *ownership* changes; the
sweep semantics do not. Consumers that read the energy grid re-point to
`sweep.beam.energy_keV`:

- `profiles.SweepProfile.apply_sweep` — `_centered_sample(energy_keV, max_energies)`
  and the `energy_values` filter that gates `E_grid_line_by_energy`.
- `profiles.high_energy_floor_identity` / `scan._resolved_run` — the
  `high_energy_materials` energy floor.
- `config.material_sweep` — builds the grid from the catalog `ScanSpec`.
- `dataset_identity` — see §4.4 (kept flat for hash compatibility).

The **photon** grids (`E_grid_line`, `E_grid_line_by_energy`, `E_grid_brem`)
stay on `Sweep`: they are detector/analysis grids, not beam properties, and
`E_grid_line_by_energy` is *keyed by* beam energy — a cross-reference from
`Sweep` into `beam.energy_keV`, documented at both fields. (Relativistic /
MeV-scale beams for REGAE remain future scope; `energy_keV` carries them as
larger numbers, no schema change.)

Parametrization decisions (locked with the user):

- **Longitudinal knob = RMS bunch length in femtoseconds, Gaussian by default.**
  `c = 1` in the transport clock (units of Å), so `σ_z[Å] = σ_t[fs] · 2997.92`
  and `Δt_e ∼ Normal(0, σ_t)` converts cleanly. `long_shape="uniform"` and
  `"file"`/`long_offsets_fs` cover flat-top and measured/arbitrary profiles;
  the analytic shapes are a convenience layer over an explicit offset array.
- **One primary bunch only.** A bunch *train* is **not** modelled as distinct
  bunches; instead `rep_rate_hz` multiplies detected flux (§4.5). This matches
  the user's intent (train = flux × rep-rate, no per-bunch structure).
- **Microbunched comb** is a future extension point (§6), reachable through
  `long_shape` / explicit offsets without reshaping `BeamSpec`.

### 4.2 Placement: standard profile value, catalog-profile-overridable

Per the user: the beam distribution is a *standard value carried alongside
transverse beam size*, defaulting in the `standard` profile but settable so a
catalog profile (e.g. `sub_100keV`) can specify its own transverse/longitudinal
distribution.

Concretely:

- `BeamSpec` lives on `Sweep` as `Sweep.beam: BeamSpec = BeamSpec()`, replacing
  **both** `Sweep.beam_fwhm_mm` and `Sweep.energy_keV`. **Hard cut**
  (decision 1): all call sites migrate to `Sweep.beam.*` in one commit — no
  read-through shim — consistent with the pre-1.0 posture. Affected:
  `sweep.build_cases` (products over `beam.energy_keV`), `profiles.apply_sweep` /
  `high_energy_floor_*`, `config.material_sweep`, `scan._resolved_run`,
  `montecarlo/runner.py::{_transport_case,_brem_for_case}`,
  `transport.simulate_trajectories`, `dataset_identity` (via the flat projection,
  §4.4), and every test / notebook that reads `sweep.energy_keV` or
  `sweep.beam_fwhm_mm`.
- **Catalog profiles** (`data/materials.toml` `[profiles.*]`, resolved into
  `ScanSpec` beside the existing `n_electrons` / `n_electrons_brem` per-profile
  knobs) may carry a `[profiles.<name>.beam]` block. `config.material_sweep`
  already resolves a `catalog_profile`; it injects the profile's `BeamSpec`
  (falling back to the `standard` default) into the constructed `Sweep`. So
  `standard` gets the 1 mm / point-bunch default, `sub_100keV` can override.
  **Scope of the `[profiles.*.beam]` block: the beam *distribution* fields**
  (transverse size, bunch length/shape, rep-rate, charge, and future
  divergence/spread). `beam.energy_keV` keeps resolving from the material's
  per-material `ScanSpec.energy_keV` scan grid, not from the beam block — the
  scan energy grid is already a per-material catalog concept and stays the
  source of truth; the beam block would only ever *floor/select* energies, which
  the profile fidelity policy (`SweepProfile.apply_sweep`) and
  `high_energy_materials` floor already own. (If a preset ever needs to *set*
  energy — e.g. a fixed-energy `regae` preset — that is an explicit later
  extension, flagged not silently allowed.)
- CLI: a `--beam-*` option group (or a single `--beam-profile` name once we have
  named beam presets like `regae`) sets/overrides fields on the resolved
  `BeamSpec`. Routed through the `cli-ui-ux` skill per repo policy.

### 4.3 Sampling in transport

In `simulate_trajectories`, after the transverse draw:

```python
t0_ang = np.zeros(Ne)                      # absolute bunch arrival offset [Å, c=1]
if bunch_length_fs is not None:
    bunch_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[3])
    sigma_ang = bunch_length_fs * C_ANG_PER_FS      # 2997.92 Å/fs
    dt = bunch_rng.normal(0.0, sigma_ang, size=Ne)  # or uniform / explicit offsets
    dt -= dt.mean()                                  # center the bunch (t=0 = centroid)
    t0_ang = dt
```

- **New independent RNG child stream** `spawn(4)[3]`, matching the proven
  `beam_rng` (`spawn(2)[1]`) / `phase_rng` (`spawn(3)[2]`) pattern, so enabling
  the bunch **never perturbs** the main free-path / scattering draws.
- `t0_ang` is broadcast to a **per-segment** `t0_ang` array in the returned
  dict (constant per electron, like `elec_id`), leaving `clock`/`t_ang`
  untouched. Downstream `t_abs = t_ang + t0_ang` is what the coherent sum will
  read; nothing reads it today.
- **Bit-for-bit limiting case:** `bunch_length_fs = None` (default) leaves
  `t0_ang` all-zero and adds one inert array — the returned `t_ang`, `E_keV`,
  `v_hat`, etc. are identical to today. `bunch_length_fs → 0` recovers the point
  bunch (`σ → 0`). This is the `finite-beam-size` compatibility contract, reused.

### 4.4 Reproducible identity

`BeamSpec` must join `dataset_identity`'s hashed payload so a bunched run and a
point-source run never resume into one checkpoint. **But nesting `beam` under
`Sweep` in the payload would renest `energy_keV`/`beam_fwhm_mm` and change every
existing run's `parameter_sha256` — silently orphaning all current checkpoints.**
Decision 6: keep the hashed payload **flat and legacy-shaped** — the in-memory
code gathers everything into `BeamSpec`, but `dataset_identity` *projects* it
back to the historical keys:

- `energy_keV` and the transverse size (at its old `beam_fwhm_mm` key) are
  emitted at their **historical top-level `sweep` positions**, byte-for-byte, so
  every pre-existing digest — and stem — is preserved bit-for-bit.
- The **new** longitudinal / rep-rate / emittance fields join the payload only
  when they diverge from their inert defaults — the exact compatibility rule
  already used for `catalog_profile` and the electron-count grids.

So identity serialization stays a flat legacy view while the code model unifies
under `BeamSpec`; the two concerns are deliberately decoupled. Any sweep- or
identity-level golden fixtures are re-derived and re-verified with the
`regen-golden` path if touched; the material-catalog golden is unaffected (it is
catalog-shaped, not sweep-shaped).

### 4.5 Rep-rate → detected flux

Detected-flux scaling lives in `results` (`store.PER_NA`, `case["scale"] =
domega_sr * PER_NA`, per-e-per-sr → per-s-per-nA). For a pulsed source the
physical flux is `charge_per_bunch × rep_rate × yield_per_electron`, i.e.
electrons/s `= (bunch_charge_pc · 1e-12 / e) · rep_rate_hz`.

Decision 4: **report photons/s at the rep-rate *alongside* the existing per-nA
scaling** (not a replacement). `BeamSpec` defaults `rep_rate_hz = 5000.0`
(5 kHz) and `bunch_charge_pc = 1.0` (1 pC); at those values one bunch carries
`1e-12/1.602e-19 ≈ 6.24e6` electrons and the source runs `5e3` bunches/s. The
per-nA number stays the primary intrinsic figure; the rep-rated photons/s is an
additional reporting column. Reporting/units layer, no physics change.

### 4.6 Beam metrics

New small module (`beam_metrics.py`, leaf under `montecarlo`/`results`)
computing standard descriptors from the sampled phase-space arrays, for
diagnostics and `trace_app`:

- **RMS sizes**: σ_x, σ_y (transverse), σ_t / σ_z (bunch length), σ_δ (energy
  spread).
- **Geometric emittance** `ε = √(⟨x²⟩⟨x'²⟩ − ⟨xx'⟩²)` per plane; **normalized**
  `ε_n = βγ·ε`. Zero today (no divergence), wired for §6.
- **Twiss** α, β, γ per plane from the same second moments.
- **Longitudinal emittance** `ε_z = √(⟨t²⟩⟨δ²⟩ − ⟨tδ⟩²)`.
- **Peak current** `I_pk ≈ Q / (√(2π) σ_t)` and **bunch charge** from
  `bunch_charge_pc`; **average current** `= bunch_charge · rep_rate_hz`.

All are pure functions of the initial phase-space sample (and are exact
diagnostics regardless of the incoherent/coherent question).

## 5. Plumbing checklist (near-term)

Ordered so each step builds on a green tree. Step E is the largest/riskiest
(it moves the primary swept axis) and gates everything after it.

- **A. `BeamSpec` dataclass** — new frozen type + `Sweep.beam: BeamSpec` field;
  fields per §4.1. No behavior yet.
- **E. Energy + transverse migration (hard cut)** — move `energy_keV` and
  `beam_fwhm_mm` onto `BeamSpec`; re-point every reader
  (`sweep.build_cases`, `profiles.apply_sweep`/`high_energy_floor_*`,
  `config.material_sweep`, `scan._resolved_run`, runner, transport, notebooks,
  tests). **Generalize transverse to per-plane `fwhm_x/_y` (decision 8)** in the
  same cut. Tree must be green with **identical** results for default inputs
  (energy relocated; `x==y==old` transverse is bit-for-bit). ← biggest blast
  radius.
- **G. Relativistic ceiling guard (decision 9)** — raise on
  `energy_keV > ceiling` in `BeamSpec` validation/transport, explanatory error;
  placeholder ceiling + ledger flag until the physics call sets it.
- **H. `dataset_identity` flat projection** (§4.4) — prove existing digests are
  bit-for-bit unchanged after A+E via a stored-digest regression.
- **L. Longitudinal sampling** — `t0_ang` per-segment array in
  `simulate_trajectories` (new RNG child `spawn(4)[3]`; `clock` untouched;
  `bunch_length_fs=None` bitwise-legacy).
- **T. Thread `BeamSpec`** through `build_cases` → `case` dict → runner →
  transport (bunch fields reach the sampler).
- **P. Catalog `[profiles.*.beam]`** resolution in `config.material_sweep`
  (distribution fields only; `standard` default preserved).
- **M. `beam_metrics.py`** + surface in `trace_app` / diagnostics.
- **F. Rep-rate/charge flux + current reconciliation** — photons/s alongside
  per-nA; **derive `beam_current_na` from `bunch_charge_pc · rep_rate_hz`
  (decision 7)**, verifying default-input count rates are unchanged.
- **C. CLI `--beam-*` group** (via `cli-ui-ux`).
- **D. Docs**: this file, `README` beam section, generated CLI/API reference,
  `docs/repo_map.md` entry; validation ledger row(s) + derivation docstrings;
  branch `TODO.md` + `main` one-liner (`todo-sync`).

## 6. Roadmap / staging

```
transverse spot (done, `finite-beam-size`)
    └── longitudinal bunch sampling  ← THIS task (plumbing + metrics)
            ├── coherent longitudinal form factor |F(ω)|²  (superradiant PXR/CBS)
            │       └── microbunched comb (modulated |F|² harmonics)
            └── divergence (x',y') + energy spread (δ)  → full 6-D emittance
```

- **Coherent sum**: add a coherent branch to `mc_spectrum` that carries the
  complex phase `exp(iω t_abs)` over electrons within a coherence class, using
  `t0_ang`. Feeds superradiant-yield studies. Requires its own ledger row +
  fresh-context validation.
- **Emittance (transverse + longitudinal)**: populate `dirs` with a
  position-correlated divergence draw and `E` with an energy-spread draw,
  parametrized by Twiss/emittance. `BeamSpec` already reserves the fields.
- **Microbunch comb**: `long_shape="comb"` / explicit offsets; feasibility in
  our setup is uncertain (user), so it stays an extension point, not a
  near-term build.

### 6.1 Adjacent beam cleanups this change should absorb or flag

Gathering the beam into `BeamSpec` exposes other beam properties that are
currently stranded or mismodelled. Ranked by whether they belong in *this*
branch.

**In scope, promoted to §8 decisions 7–9:**

- **`beam_current_na` reconciliation** (decision 7). Average beam current lives in
  `results.Settings.beam_current_na` (default 5 nA; drives `PER_NA` count-rate
  scaling in `metrics`/`tables`). It is a *beam property* stranded outside the
  beam object, and it is now **derivable**: `I_avg = bunch_charge_pc · rep_rate_hz`.
  Make `BeamSpec` the single source of truth — either compute `beam_current_na`
  from the beam (recommended) or keep it as an explicit override that *defaults*
  to the derived value, and document the identity so the per-nA and per-rep-rate
  reports (§4.5) can never silently disagree.
- **Elliptical transverse (`σ_x`, `σ_y`)** (decision 8). Today the spot is a
  single isotropic `beam_fwhm_mm`. The metrics module wants **per-plane** σ /
  Twiss / emittance, and future per-plane divergence needs a per-plane size to
  correlate with. Since the hard cut already rewrites every transverse call site,
  generalize now to independent `transverse_fwhm_x_mm` / `_y_mm` (isotropic
  scalar stays the convenience default). Doing it later means a second hard cut
  through the same files.
- **Relativistic-regime guard** (decision 9). `beta_from_keV` + the Zhai PXR/CBS
  kernels are **nonrelativistic**; REGAE-scale energies (3–5 MeV, On-Hold #1)
  silently leave the model's validity. Now that `BeamSpec` centralizes energy,
  **refuse** (raise) when `energy_keV` crosses the nonrelativistic ceiling, with
  an explanatory error naming the limit and pointing at the future
  relativistic/channeling work — a MeV beam must not quietly produce wrong
  numbers. Refusal (not a warning) is deliberate: no valid result exists above
  the ceiling today, so silence would be a correctness trap. The ceiling itself
  needs a defensible value (§8 note).

**Flag / guard (cheap, low-risk):**

- **Mean-vs-spread framing.** State the pattern explicitly in `BeamSpec` docs:
  each phase-space axis has a *mean* set elsewhere and a *spread* owned by
  `BeamSpec` — energy mean = `energy_keV`, spread = `energy_spread_frac`;
  direction mean = geometry (`beam_dir` from tilt), spread = `divergence_mrad`;
  position mean = origin (future pointing offset), spread = transverse size. Keeps
  the future emittance work from re-deriving where each knob lives.

**Explicitly keep OUT / defer:**

- **`n_electrons` / `n_electrons_brem` stay out of `BeamSpec`.** They are
  *numerical sampling* (macro-particle counts), not a physical beam property.
  Document the split so nobody folds them in: physical beam = `BeamSpec`,
  Monte-Carlo sampling = `n_electrons` (+ `seed`).
- **`beam_uvw` rename (separate branch).** `sweep.beam_uvw` / catalog `beam_uvw`
  is a *crystal* axis along the surface normal — nothing to do with the electron
  beam, and doubly confusing now that a real beam object exists. A rename to
  `surface_uvw` / `normal_uvw` is worthwhile but spans ~9 modules + `materials.toml`
  + the catalog golden fixture, so it is its own `refactor(...)` branch, not this
  one. Noted so it is not forgotten.
- **Beam pointing offset** (centroid position + mean-angle misalignment relative
  to the crystal) — a real beam DOF, out of scope; extension point on the
  position/direction *mean* side of the mean-vs-spread split above.
- **Ledger grouping.** Point the scattered beam rows (`finite-beam-size`,
  `grazing-beam-projection`, `finite-transverse-crystal`) at this doc as the
  canonical beam reference and group them under a "beam" heading — a doc nicety
  for the §5-D docs step.

## 7. Validation plan

- New ledger row(s), e.g. `longitudinal-bunch-sampling` — Gaussian/uniform
  sampling convention, fs→Å (`c = 2997.92 Å/fs`) and FWHM/RMS conventions,
  limiting case `bunch_length_fs → 0`/`None` recovers the point bunch
  bit-for-bit, RNG-independence of the spawned child stream. Like
  `finite-beam-size`, this is a *geometric/parametric convention* (no single
  citable source equation), so expect a `rederived`-style row, not a paper
  reproduction. The eventual coherent form factor gets its own, physics-bearing
  row.
- Derivation docstring on the new sampling code: convention, units,
  limiting case, `Validation:` marker.
- Tests (fast, seeded — per `regression-testing`/`monte-carlo` skills):
  `bunch_length_fs=None` bitwise-legacy; sampled `std(Δt)` matches σ_t; RNG
  independence (main draws unchanged); `dataset_identity` digest stability for
  legacy-inert defaults; metrics correctness on analytic inputs.

## 8. Resolved decisions

1. **`beam_fwhm_mm` migration** — hard cut. All call sites move to `Sweep.beam`
   in one commit; no read-through shim (§4.2).
2. **Beam energy into `BeamSpec`** — `Sweep.energy_keV` → `Sweep.beam.energy_keV`
   so every beam property lives in one object; sweep semantics unchanged
   (§4.1.1). Photon grids stay on `Sweep`.
3. **Bunch centroid** — center on centroid (`Δt -= Δt.mean()`), so `|F(ω)|`
   phase references the bunch center and metrics stay unbiased at finite N
   (§4.3).
4. **Rep-rate reporting** — report photons/s at `rep_rate_hz` *alongside* the
   per-nA scaling. Defaults `rep_rate_hz = 5000.0` (5 kHz),
   `bunch_charge_pc = 1.0` (1 pC) (§4.5).
5. **Beam presets** — named presets live as catalog `[profiles.*.beam]` rows
   (e.g. a `regae` profile), the same mechanism by which `sub_100keV` can carry
   its own distribution. Distribution fields only; energy resolves from the
   material scan grid. No separate registry (§4.2).
6. **Identity hash stays flat** — `dataset_identity` projects `BeamSpec` back to
   the historical top-level keys so every existing `parameter_sha256` and
   checkpoint stem is bit-for-bit preserved; new fields join only when they
   diverge from inert defaults (§4.4).
7. **`beam_current_na` derived from the beam** — `BeamSpec` is the source of
   truth; `I_avg = bunch_charge_pc · rep_rate_hz`. `results.Settings.beam_current_na`
   becomes a derived value (or a default-from-derived override), so the per-nA
   and per-rep-rate reports can't disagree (§6.1). Preserve the 5 nA default
   number via the `1 pC × 5 kHz` defaults' implied current, or keep an explicit
   override for exact back-compat — decide at implementation, test the count-rate
   outputs are unchanged for default inputs.
8. **Elliptical transverse** — replace isotropic `beam_fwhm_mm` with per-plane
   `transverse_fwhm_x_mm` / `_y_mm` during the same hard cut; isotropic scalar is
   a convenience ctor. Bit-for-bit legacy when `x == y == old value` (§6.1).
9. **Relativistic ceiling = refuse** — raise (not warn) when `energy_keV` exceeds
   the model's validity ceiling, with an explanatory error naming the limit and
   the future relativistic/channeling work (§6.1). The concern is the *radiation*
   model, not `β`: `beta_from_keV` is already relativistic (the model runs fine at
   the current 30–100+ keV range), but the **Zhai PXR/CBS derivation itself is
   nonrelativistic** and breaks toward REGAE MeV energies. **Open sub-decision:**
   the exact ceiling — set it where the nonrelativistic PXR/CBS treatment is still
   defensible (well above today's 30–100+ keV working range, well below the
   3–5 MeV REGAE regime; likely a few hundred keV). Needs a physics call before
   coding; until set, use a conservative placeholder and flag it in the ledger.

## 9. Delegation plan (subagents + models)

Which checklist steps (§5) to hand to subagents, which to keep on the main
thread, and at what model tier. Repo rule: **no Explore agent** for code
research (tokensave first); use `cavecrew-investigator` when a compressed
file:line inventory is wanted. Physics-validation **must** run in fresh context,
never the author's — so it is always a separate agent.

| Step | Owner | Agent type | Model | Why |
|------|-------|-----------|-------|-----|
| A — `BeamSpec` dataclass | main | — | Opus | Small but design-defining; sets the contract every other step consumes. Keep in-hand. |
| E — energy+transverse hard-cut migration | main, with a scout | inventory via `cavecrew-investigator`; edits on main | Opus (edits); Haiku (scout) | Cross-cutting, >3 files, touches the identity hash and the primary swept axis — highest risk, must not fan out blindly. Scout the full `energy_keV`/`beam_fwhm_mm` call-site list first (cheap, compressed), then edit on main. |
| H — identity flat projection + digest regression | main | — | Opus | Correctness-critical; a wrong projection silently orphans every checkpoint. Author + a stored-digest test on main. |
| L — longitudinal sampling in transport | delegate | `general-purpose` | Sonnet | Self-contained numeric edit to one function with a crisp bit-for-bit contract and seeded tests; paste the exact target source into the prompt (per repo file-split token rule). |
| T — thread `BeamSpec` through cases | delegate | `cavecrew-builder` | Sonnet | Mechanical wiring, 1–2 files per hop, format-preserving — the builder's sweet spot. |
| P — catalog `[profiles.*.beam]` resolution | delegate | `general-purpose` | Sonnet | Localized to `config`/catalog decode + a TOML fixture; moderate but bounded. |
| M — `beam_metrics.py` + trace_app surface | delegate | `general-purpose` | Sonnet | New leaf module, standard closed-form formulas (RMS/emittance/Twiss), pure functions + unit tests. Notebook edit follows `notebook-workflow`. |
| F — rep-rate/charge flux reporting | delegate | `cavecrew-builder` | Sonnet | Small `results`/`Settings` reporting add, no physics. |
| C — CLI `--beam-*` group | delegate | `general-purpose` | Sonnet | Must run the `cli-ui-ux` skill (design+impl+tests); mechanical once the option surface is fixed. |
| D — docs (README, repo_map, CLI/API ref, TODO-sync) | delegate | `general-purpose` | Haiku→Sonnet | Prose + generated-reference regen; `documentation-maintenance` + `todo-sync` skills. Low reasoning load. |
| V — validation: derivation docstring + ledger row + independent check | split | author on main; verify via `physics-validator` | Opus (verify) | Repo law: the implementer writes the `Validation:` marker/row; a **fresh-context** `physics-validator` re-derives the `longitudinal-bunch-sampling` convention and may not be the author. |

Rules of thumb applied above: **Opus** for anything touching the identity hash,
the swept-axis migration, or physics correctness; **Sonnet** for bounded,
well-specified edits with tests; **Haiku** for pure location/inventory and
low-reasoning prose. Delegated code steps get the target source pasted into the
prompt rather than told to re-read shared files (repo file-split token rule).
Everything stays gated by `verifying-changes` before any completion claim.
