# Beam phase space

How PyRITE specifies the electron beam **as an input**, and how that specification becomes the per-electron initial conditions the Monte Carlo transports.

Companion to the [sweep profiles guide](../../guides/sweep-profiles.md), which documents the beam TOML block, and to the `beam-phase-space-metrics` ledger row, which covers the *output* side, `beam_metrics.sampled_beam_metrics`.

A beam is a named catalog object: a top-level `[beams.NAME]` table attached by `beam = "NAME"` on a profile and managed with `pyrite beam create` / `pyrite beam set`. The inline `[profiles.NAME.beam]` sub-table decodes to the same keys with the same meaning. A beam reference resolves to values before hashing, so converting an inline block to a named beam leaves `parameter_sha256` bit-for-bit. See the [sweep profiles guide](../../guides/sweep-profiles.md) under "Named beams".

## Reference plane

Every quantity here describes the beam **at the crystal entrance face**, the plane where transport begins.

PyRITE models no beamline transport, no space charge, and no envelope evolution: the distribution written in a profile is the distribution sampled. At 1 pC in 200 fs at 30–100 keV space charge is not negligible over a realistic source-to-target drift, so feeding gun-exit or source emittance into a profile and reading the output as physical is a user error the code cannot detect.

## Canonical parameterization

Per transverse plane there are exactly three independent second moments: `<x²>`, `<x x'>`, `<x'²>`. Equivalently the Courant–Snyder (Twiss) triplet `(eps, beta, alpha)`. Any parameterization that lets a user set more than three numbers per plane is over-determined.

**Canonical input is the Twiss triplet per plane**, with *normalized* emittance:

| symbol  | field                          | unit                  |
| --------- | -------------------------------- | ----------------------- |
| `eps_n` | `normalized_emittance_mm_mrad` | mm·mrad              |
| `beta`  | `beta_twiss_m`                 | m                     |
| `alpha` | `alpha_twiss`                  | dimensionless, signed |

### Why normalized, not geometric

`energy_keV` is the primary swept axis, and this repo spans 30 keV ({math}`\beta\gamma \approx 0.34`) to the REGAE-scale 3–5 MeV case ({math}`\beta\gamma \approx 7`) in the backlog. Geometric emittance is not invariant under acceleration, so a single geometric value attached to a multi-energy `BeamSpec` means a *different beam at every energy*, confounding emittance with energy across the sweep. Normalized emittance is the invariant, so it is what gets stored.

Geometric emittance is derived per case:

```{math}
\gamma  = 1 + \frac{T}{510.99895} \\
\beta\gamma = \sqrt{\gamma_{\mathrm{rel}}^2 − 1} \\
\epsilon_{\mathrm{geom}}   = \epsilon_n / (\beta\gamma)
```

with {math}`T` the case kinetic energy in keV. `beam_metrics.py` uses the same relation in the opposite direction when it reports `normalized_emittance_mm_rad = beta_gamma * geometric_emittance_mm_rad`.

### Units

The input units above are the accelerator-conventional ones. The diagnostics module works in mm·rad and mm/rad, so the round trip carries two conversions:

| quantity             | input     | `beam_metrics` output                         |
| -------------------- | --------- | --------------------------------------------- |
| normalized emittance | `mm·mrad` | `normalized_emittance_mm_rad` = input × 1e−3 |
| Twiss beta           | `m`        | `beta_mm_per_rad` = input × 1e3               |
| Twiss alpha          | —         | `alpha`, same value                            |

`alpha` is sign-preserved under the convention `alpha = −<x x'>/eps`, matching `beam_metrics._plane_metrics`. Negative `alpha` is a diverging beam past its waist and is legitimate input.

## Mutual exclusion

Spot FWHM, divergence and emittance are three numbers that over-determine two and drop the correlation entirely. Allowing all three with a precedence rule produces a plausible-looking beam that is not the one the user asked for, with no diagnostic. So:

| spelling                                     | status             | meaning                                           |
| -------------------------------------------- | ------------------ | ------------------------------------------------- |
| `transverse` block (Twiss triplet per plane) | canonical          | full three-moment description                     |
| `transverse_fwhm_x_mm` / `_y_mm`             | legacy convenience | zero-emittance waist:`alpha = 0`, zero divergence |
| `divergence_mrad`                            | derived, read-only | RMS slope at the case energy                      |
| `energy_spread_frac`                         | canonical          | longitudinal only; no Twiss equivalent            |

Setting the `transverse` block **and** a spot FWHM is a hard error, the same rule the `longitudinal` policy applies against the flat legacy bunch fields.

`divergence_mrad` is a derived property rather than a stored input. A stored value is energy-independent, which contradicts the normalized-emittance argument above: the physical RMS slope of a fixed beam falls as {math}`1/\sqrt(\beta\gamma)`, so a constant `divergence_mrad` across an energy sweep is not one beam.

`energy_spread_frac` is a real input. It is not a transverse quantity and has no Twiss equivalent.

## Sampling

Given the resolved per-case {math}`(\epsilon_\mathrm{geom}, \beta, \alpha)` for a plane, and two independent standard normal draws {math}`u_1, u_2 ~ N(0,1)`:

```{math}
\sigma_x = sqrt(\epsilon_\mathrm{geom} \beta) \\
x       = \sigma_x u_1 \\
x'      = \sqrt{\frac{\epsilon_\mathrm{geom}}{beta}} (u_2 − \alpha u_1)
```

which reproduces the three target moments exactly:

```{math}
\langle x^2 \rangle   = \epsilon_\mathrm{geom} \beta \\
\langle x'^2 \rangle  = \epsilon_\mathrm{geom} \frac{1 + \alpha^2}{\beta} = \epsilon_\mathrm{geom} \gamma_\mathrm{twiss} \\
\langle x x'\rangle = − \epsilon_\mathrm{geom} \alpha
```

so {math}`\sqrt{\langle x^2 \rangle \langle x'^2\rangle − \langle x x' \rangle^2} = \epsilon_\mathrm{geom}` as required. The $x$ and $y$ planes are sampled independently: there is no {math}`\langle x y \rangle` coupling term and no skew/solenoid model.

Slopes become directions by tilting the nominal `beam_dir` about the two transverse axes of the beam frame. For the small slopes involved ({math}`x'` of order 1e−3), the direction is normalized after tilting rather than approximated.

### Energy spread

```text
delta_i ~ N(0, energy_spread_frac)
E_i     = E0_keV * (1 + delta_i)
```

drawn **independently and uncorrelated** with the longitudinal offsets, for every `long_shape` including `compressed`, so {math}`\langle t\,\delta\rangle = 0` holds by construction and `beam_metrics` reports a longitudinal emittance of exactly `sigma_t * sigma_delta`. A chirp model would enter as an additive correlation parameter defaulting to zero; see [Longitudinal bunch structure](longitudinal-structure.md).

### RNG placement

Transverse and energy draws take their own RNG children, following the `_sample_bunch_offsets` / `spawn` precedent, and are counter-addressed per electron inside them ([random streams](../../computation/random-streams.md)). That makes the zero-spread limit bit-for-bit rather than statistically identical: when the distribution is inert, no draw is taken and the transport stream is untouched.

## What stays inert

`bunch_charge_pc` and `rep_rate_hz` answer how many electrons per second, not where each electron is. They are normalization, not phase space, and nothing under `montecarlo/` reads them. A test pins it: with the coherent path off, changing either must leave every sampled array (`initial_r_ang`, `initial_v_hat`, `t0_ang`, and the segment arrays) bit-for-bit identical at a fixed seed.

Deriving Monte Carlo statistics from charge — `Ne = Q/e`, or weighting the incoherent sum by `N_phys` — would change every RNG draw, risks double-counting against downstream normalization, and wrecks runtime: a 1 pC bunch is 6.24e6 electrons against ~300 macro-particles. Coherent/superradiant emission does scale with `N_phys` (`N²` vs `N`), so there is a legitimate future path where charge enters the physics; the test makes that a ledgered decision.

Charge × rep rate does not normalize the spectrum to absolute flux. There are no photons/second anywhere, only the diagnostics struct. An explicit absolute-flux multiplier is a separate backlog item.

## Interaction with the line-energy grid

`montecarlo/spectrum/lines.py` consumes per-segment `v_hat` and per-segment `E_keV`, so once injection is randomized the emitted line spectrum picks up divergence and energy spread with no further work. Two consequences:

1. **The `detector.energy_bins.line` window is derived from the nominal case energy**, so an energy-spread-broadened line is displaced from where the grid was cut. It does not clip. Differentiating the resonance `omega = v.g / (1 - n.v)` in `beta` gives a fractional line shift of `S * delta` with

   ```{math}
   S = \frac{\gamma - 1}{\gamma^3 \beta^2 (1 - \beta \cos\theta_\mathrm{obs})}
   ```

(`energy_grid.bounds.line_shift_fraction`). Over the measured 30–300 keV range, `S` is largest at the *low* energy end: 0.46 at 30 keV, falling to 0.24 at 300 keV, with the nonrelativistic limit `S -> 1/2`. `margined_stop` cuts the window 15% above the measured coverage energy, so the margin is only consumed once the beam spread reaches ~33% RMS, far outside both any real photoinjector and the first-order expansion `S` is derived under. No gate on `energy_spread_frac` is warranted. The full kernel is measured against `S` in `tests/montecarlo/test_beam_energy_spread_grid.py` (a 5% beam energy step moves the 30 keV hopg line by 2.28%, against 2.30% predicted).
2. **The analytic broadening helpers do not widen.** `mosaic_fwhm_eV`, `aperture_fwhm_eV` and `mosaic_psi_rad` (`montecarlo/detector.py`) stay at the nominal `beam_dir` and `E0_keV` by design. They are diagnostics, not the spectrum.

## Limiting cases

- `eps_n → 0`: zero spot, zero divergence, collimated.
- `energy_spread_frac → 0`: monoenergetic, bit-for-bit.
- `alpha = 0`: at the waist, `<x x'> = 0`, and the beam matches the legacy FWHM spelling with `beta = sigma_x² / eps_geom`.
- Two energies differing in `beta*gamma` with the same `eps_n` give geometric emittances in the inverse ratio of `beta*gamma`, the observable signature of the normalized convention and what the round-trip test pins.

## Out of scope

Space charge, source-to-crystal beamline transport, the coherent form factor, `<x y>` coupling, chirp, and absolute flux normalization.

## Validation

`Validation: beam-phase-space-injection` — see the row in [physics validation ledger](../../validation/physics-validation-ledger.md).

## Native GPT snapshots

`source="gpt_gdf"` replaces analytic sampling with correlated time-output or screen records.See [GDF beam import](../../guides/gpt-gdf-beams.md) for units, explicit ray-plane projection, normalization, and limitations, and the [independent derivation](../../validation/beam-transport/gpt-gdf-injection.md).
