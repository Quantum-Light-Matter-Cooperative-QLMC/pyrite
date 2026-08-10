# Beam phase space

How PyRITE specifies the electron beam **as an input**, and how that
specification becomes the per-electron initial conditions the Monte Carlo
transports.

Companion to [`docs/sweep-profiles.md`](sweep-profiles.md) (where the beam TOML
block is documented) and the `beam-phase-space-metrics` ledger row (which
covers the *output* side, `beam_metrics.sampled_beam_metrics`).

Every key below is spelled the same wherever the beam is written. Beams are now
named catalog objects — a top-level `[beams.NAME]` table attached by
`beam = "NAME"` on a profile and managed with `pyrite beam ...` — rather than an
inline `[profiles.NAME.beam]` sub-table written by `pyrite profile` flags. The
inline spelling still decodes and still means exactly this, and the nine
`pyrite profile` beam flags still work while warning; nothing about the physics,
the units, or the mutual exclusions changed with the move. Because a reference
resolves to values before hashing, converting an inline block to a named beam
leaves `parameter_sha256` bit-for-bit. See
[`docs/sweep-profiles.md`](sweep-profiles.md) "Named beams".

## Reference plane

Every quantity here describes the beam **at the crystal entrance face** — the
plane where transport begins.

This matters because the numbers a user has to hand are usually gun-exit or
source numbers. At 1 pC in 200 fs at 30–100 keV, space charge is not negligible
over a realistic source-to-target drift, and PyRITE does not model it. There
is no beamline transport, no space charge, and no envelope evolution: the
distribution written in a profile is the distribution sampled. Feeding gun-exit
emittance into a profile and reading the output as physical is a user error the
code cannot detect.

## What was already there

The beam object is not new. `BeamSpec` (`src/cxr_mc/sweep.py`) has owned central
energy, transverse spot FWHM, the longitudinal policy, bunch charge and
repetition rate for some time, and profile plumbing decodes all of it.

Two fields were declared, decoded and hashed but never read by anything under
`src/cxr_mc/montecarlo/`: `divergence_mrad` and `energy_spread_frac`. The
transport fanned a single `beam_dir` and a scalar `E0_keV` out across all
electrons, and `geometry.py` stated the assumption in as many words —
"perfectly collimated lab beam (zero divergence)".

This document covers making the transverse and energy phase space live.

## Canonical parameterization

Per transverse plane there are exactly three independent second moments:
`<x²>`, `<x x'>`, `<x'²>`. Equivalently the Courant–Snyder (Twiss) triplet
`(eps, beta, alpha)`. Any parameterization that lets a user set more than three
numbers per plane is over-determined.

**Canonical input is the Twiss triplet per plane**, with *normalized*
emittance:

| symbol | field | unit |
| --- | --- | --- |
| `eps_n` | `normalized_emittance_mm_mrad` | mm·mrad |
| `beta`  | `beta_twiss_m` | m |
| `alpha` | `alpha_twiss` | dimensionless, signed |

### Why normalized, not geometric

`energy_keV` is the primary swept axis, and this repo spans 30 keV
(`beta*gamma ≈ 0.34`) to the REGAE-scale 3–5 MeV case (`beta*gamma ≈ 7`) in the
backlog. Geometric emittance is not invariant under acceleration, so a single
geometric value attached to a multi-energy `BeamSpec` silently means a
*different beam at every energy* — the sweep would confound emittance with
energy. Normalized emittance is the invariant, so it is what gets stored.

Geometric emittance is derived per case:

```
gamma_rel  = 1 + T_keV / 510.99895
beta*gamma = sqrt(gamma_rel² − 1)
eps_geom   = eps_n / (beta*gamma)
```

with `T_keV` the case kinetic energy. This is the same relation
`beam_metrics.py` already uses in the opposite direction when it reports
`normalized_emittance_mm_rad = beta_gamma * geometric_emittance_mm_rad`.

### Units

The input units above are the accelerator-conventional ones. The diagnostics
module works in mm·rad and mm/rad, so the round trip carries two conversions:

| quantity | input | `beam_metrics` output |
| --- | --- | --- |
| normalized emittance | `mm·mrad` | `normalized_emittance_mm_rad` = input × 1e−3 |
| Twiss beta | `m` | `beta_mm_per_rad` = input × 1e3 |
| Twiss alpha | — | `alpha`, same value |

`alpha` is sign-preserved and its sign convention is `alpha = −<x x'>/eps`,
matching `beam_metrics._plane_metrics`. Negative `alpha` is a diverging beam
past its waist and is legitimate input.

## Mutual exclusion

Spot FWHM, divergence and emittance are three numbers that over-determine two
and drop the correlation entirely. Allowing all three with a precedence rule
produces a plausible-looking beam that is not the one the user asked for, with
no diagnostic. So:

| spelling | status | meaning |
| --- | --- | --- |
| `transverse` block (Twiss triplet per plane) | canonical | full three-moment description |
| `transverse_fwhm_x_mm` / `_y_mm` | legacy convenience | zero-emittance waist: `alpha = 0`, zero divergence |
| `divergence_mrad` | derived, read-only | RMS slope at the case energy |
| `energy_spread_frac` | canonical | longitudinal only; no Twiss equivalent |

Setting the `transverse` block **and** a spot FWHM is a hard error, not a
precedence resolution. Same rule the `longitudinal` policy already applies
against the flat legacy bunch fields.

`divergence_mrad` becomes a derived property rather than a stored input,
because as a stored input it is energy-independent, which contradicts the
normalized-emittance argument above: the physical RMS slope of a fixed beam
falls as `1/sqrt(beta*gamma)`, so a constant `divergence_mrad` across an energy
sweep is not one beam.

`energy_spread_frac` survives as a real input. It is not a transverse quantity
and has no Twiss equivalent.

## Sampling

Given the resolved per-case `(eps_geom, beta, alpha)` for a plane, and two
independent standard normal draws `u1, u2 ~ N(0,1)`:

```
sigma_x = sqrt(eps_geom * beta)
x       = sigma_x * u1
x'      = sqrt(eps_geom / beta) * (u2 − alpha * u1)
```

which reproduces the three target moments exactly:

```
<x²>   = eps_geom * beta
<x'²>  = eps_geom * (1 + alpha²) / beta = eps_geom * gamma_twiss
<x x'> = −eps_geom * alpha
```

so `sqrt(<x²><x'²> − <x x'>²) = eps_geom` as required. The `x` and `y` planes
are sampled independently — there is no `<x y>` coupling term and no
skew/solenoid model.

Slopes become directions by tilting the nominal `beam_dir` about the two
transverse axes of the beam frame. For the small slopes involved (`x'` of order
1e−3), the direction is normalized after tilting rather than approximated.

### Energy spread

```
delta_i ~ N(0, energy_spread_frac)
E_i     = E0_keV * (1 + delta_i)
```

drawn **independently and uncorrelated** with the longitudinal offsets, for
every `long_shape` including `compressed`.

This is a decision, not an oversight. `compressed` is terminology for a short
bunch, not a chirp model, so `<t delta> = 0` holds by construction.
`beam_metrics` will therefore report a longitudinal emittance of exactly
`sigma_t * sigma_delta` with no correlation term. A future chirp model is an
additive change — a correlation parameter that defaults to zero — not a
reinterpretation of what these fields already mean.

### RNG placement

Transverse and energy draws take their own RNG children, following the
`_sample_bunch_offsets` / `spawn` precedent. This is what makes the zero-spread
limit bit-for-bit rather than statistically identical: when the distribution is
inert, no draw is taken and the transport stream is untouched.

## What stays inert

`bunch_charge_pc` and `rep_rate_hz` answer "how many electrons per second", not
"where each electron is". They are normalization, not phase space, and nothing
under `montecarlo/` reads them.

They stay that way, and a test pins it: with the coherent path off, changing
either must leave every sampled array (`initial_r_ang`, `initial_v_hat`,
`t0_ang`, and the segment arrays) bit-for-bit identical at a fixed seed.

The test exists rather than a docstring because the tempting next step is
wrong in a specific, expensive way. Once emittance is a real input, `BeamSpec`
looks like a complete physical beam, and the obvious "add realism" move is to
derive MC statistics from charge — `Ne = Q/e`, or weighting the incoherent sum
by `N_phys`. A 1 pC bunch is 6.24e6 electrons against ~300 macro-particles.
That coupling would change every RNG draw, risks double-counting against
downstream normalization, and wrecks runtime.

Coherent/superradiant emission genuinely does scale with `N_phys` (`N²` vs
`N`), so there is a legitimate future path where charge enters the physics. The
test makes that a deliberate, ledgered decision instead of drift.

Separately: charge × rep rate does **not** currently normalize the spectrum to
absolute flux. There are no photons/second anywhere, only the diagnostics
struct. An explicit absolute-flux multiplier is a real missing feature and a
separate backlog item.

## Interaction with the line-energy grid

`montecarlo/spectrum/lines.py` consumes per-segment `v_hat` and per-segment `E_keV`,
so once injection is randomized the emitted line spectrum picks up divergence
and energy spread with no further work. Two consequences are worth stating
explicitly:

1. **The `E_grid_line` window is derived from the nominal case energy**, so an
   energy-spread-broadened line is displaced from where the grid was cut.
   Checked, and it does not clip. Differentiating the resonance
   `omega = v.g / (1 - n.v)` in `beta` gives a fractional line shift of
   `S * delta` with

   ```
   S = (gamma - 1) / (gamma^3 beta^2 (1 - beta cos(theta_obs)))
   ```

   (`energy_grid.bounds.line_shift_fraction`). `S` is largest at the *low*
   energy end -- 0.46 at 30 keV, falling to 0.24 at the 300 keV model ceiling,
   with the nonrelativistic limit `S -> 1/2` -- and `margined_stop` cuts the
   window 15% above the measured coverage energy. The margin is therefore only
   consumed once the beam spread reaches ~33% RMS, which is far outside both
   any real photoinjector and the first-order expansion `S` is derived under.
   No gate on `energy_spread_frac` is warranted. The full kernel is measured
   against `S` in `tests/montecarlo/test_beam_energy_spread_grid.py` (a 5% beam
   energy step moves the 30 keV hopg line by 2.28%, against 2.30% predicted).
2. **The analytic broadening helpers do not widen.** `mosaic_fwhm_eV`,
   `aperture_fwhm_eV` and `mosaic_psi_rad` (`montecarlo/detector.py`) stay at
   the nominal `beam_dir` and `E0_keV` by design. They are diagnostics, not the
   spectrum. Their not widening is correct and must not be read as the sampled
   result being wrong.

## Limiting cases

- `eps_n → 0`: zero spot, zero divergence, collimated. Reproduces the previous
  behaviour bit-for-bit.
- `energy_spread_frac → 0`: monoenergetic, bit-for-bit.
- `alpha = 0`: at the waist, `<x x'> = 0`, and the beam matches the legacy
  FWHM spelling with `beta = sigma_x² / eps_geom`.
- Two energies differing in `beta*gamma` with the same `eps_n` give geometric
  emittances in the inverse ratio of `beta*gamma` — this is the observable
  signature of the normalized convention and is what the round-trip test pins.

## Out of scope

Space charge, source-to-crystal beamline transport, the coherent form factor,
`<x y>` coupling, chirp, and absolute flux normalization.

## Validation

`Validation: beam-phase-space-injection` — see the row in
[`docs/physics-validation-ledger.md`](physics-validation-ledger.md).
