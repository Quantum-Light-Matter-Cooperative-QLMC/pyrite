# `finite-beam-size`

**Claim.** The incident electron beam's transverse entry point is drawn from an azimuthally-symmetric 2D Gaussian spot of a given FULL WIDTH AT HALF MAXIMUM (`beam_fwhm_mm`, mm), applied as a rigid per-electron `(x0, y0)` offset to every segment that electron emits; `None`/`0` (falsy) is a strict no-op reproducing the old point-source beam bit-for-bit; the offset is drawn from an RNG stream independent of the main transport `rng`. Its zero-spectrum-effect conclusion applies only when both finite-footprint dimensions are `None`.

**Code.** `src/pyrite/montecarlo/transport.py::simulate_trajectories` (`beam_fwhm_mm=` parameter, branch at `if beam_fwhm_mm:` near `pos = np.zeros((Ne, 3))`). **Source.** Standard Gaussian beam-spot parametrization (no specific paper equation — this is a geometric convention, not a derived physical law). **Verifier context.** Independent session; did not author the implementation.

## Scope of the claim

- **Signature:** `beam_fwhm_mm: float | None`, mm. Output: an additive `(x0, y0, 0)` offset baked into `pos[:, :2]` before the transport loop, which propagates unchanged into every `r_mid[:, :2]` that electron emits (transport is a rigid translation of the trajectory's starting point; the physics of free-path sampling, elastic scattering, and stopping power are all local/relative and don't care about an absolute lateral origin).
- **All-`None` footprint scope:** when `crystal_width_mm` and `crystal_height_mm` are both `None`, the crystal is laterally infinite and the detector direction `n_hat` is a fixed far-field unit vector. In this limiting model, no downstream physics reads transverse position, so changing beam size has *zero* effect on the emitted spectrum.
- **Finite-footprint exclusion:** when both transverse dimensions are finite, sampled `(x0, y0)` determines whether an incident electron misses the crystal, can affect lateral electron escape, and changes photon self-absorption through the first of six faces. Those effects belong to the separate, unverified `finite-transverse-crystal` ledger row; this write-up does not validate them.
- **Limiting case:** `beam_fwhm_mm → 0` or `None` recovers the point source exactly.

## 1. Docstring vs. implementation, unit and conversion factors

Read `simulate_trajectories` in full (`src/pyrite/montecarlo/transport.py`, lines 201–486 on `feature/finite-electron-beam-size`). Confirmed line-for-line:

```python
pos = np.zeros((Ne, 3))
if beam_fwhm_mm:
    MM_TO_ANG = 1.0e7
    sigma_ang = float(beam_fwhm_mm) * MM_TO_ANG / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    beam_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
    pos[:, :2] = beam_rng.normal(0.0, sigma_ang, size=(Ne, 2))
```

- **mm → Å:** 1 mm = 1e7 Å (1 mm = 1e-3 m = 1e7 Å since 1 Å = 1e-10 m). ✓ `MM_TO_ANG = 1.0e7` is correct.
- **FWHM → σ:** standard Gaussian relation `FWHM = 2√(2 ln 2) · σ`, so `σ = FWHM / (2√(2 ln 2))`, `2√(2 ln 2) ≈ 2.35482`. Independently computed `2*sqrt(2*log(2)) = 2.3548200450309493` — matches the code's inline formula exactly (it is evaluated the same way, not a hardcoded magic number, so there's no drift risk). ✓
- The offset is added **once**, before the transport loop (`pos[:, :2] = ...` happens before the `for _ in range(max_steps):` loop and before `beam_dir` is even normalized), and is never touched again except through the ordinary `pos[grp] = p + step[:, None] * d` advance — i.e. it acts exactly as a fixed additive translation of the whole trajectory, not a resampled per-segment quantity. Confirmed by reading the loop body: `p = pos[grp]` reads the running position (offset baked in from step 0 onward), and nothing resets or overwrites `pos[:, :2]` afterward outside the ordinary flight/collision update.

## 2. Central claim, restricted to the all-`None` footprint

The following source inspection and grep were valid for the laterally infinite, all-`None` footprint branch. They must not be generalized to a finite rectangular crystal.

For that branch, grepped `r_mid`/`seg_r`/`pos[:, 0|1|:2]` across all of `src/pyrite`:

```
src/pyrite/montecarlo/spectrum/lines.py:173:   seg_r = xp.asarray(segments["r_mid"], ...)
src/pyrite/montecarlo/spectrum/lines.py:283:   z_mid = seg_r[idx, 2]
src/pyrite/montecarlo/spectrum/lines.py:521:   seg_r = xp.asarray(segments["r_mid"], ...)
src/pyrite/montecarlo/spectrum/lines.py:524:   z_mid = seg_r[:, 2]
src/pyrite/montecarlo/transport.py: (definition + docstring only)
src/pyrite/plots/mpl/trajectories.py:227: L, v, r = segs["L_ang"], segs["v_hat"], segs["r_mid"]
```

- `spectrum.py::mc_spectrum` (coherent PXR/CBS line spectrum): in the all-`None` branch, `seg_r` is sliced **only** at `[..., 2]` (`z_mid`), used solely to compute the Beer–Lambert escape path length to the entrance/exit face (`L_esc = z_mid / (-n_hat[2])` or `(thickness - z_mid) / n_hat[2]`, and the layered-stack equivalent `_stack_tau`). The amplitude physics (`A_PXR`, `A_CBS`, resonance energy `E_res = ħc·(v·g)/(1 − v·n̂)`, sinc² lineshape) depends only on `v_hat`, `E_keV`, `L_ang`, `t_ang`, the fixed reciprocal vector `g`, and the fixed far-field `n_hat` — never on the segment's absolute lateral position. This is dimensionally sound: `n_hat` is a unit vector fixed for the whole detector geometry (far-field approximation), so a lateral translation of the source point does not change the direction to the detector, hence doesn't change any dot product `v·n̂`, `v·g`, etc.
- `spectrum.py::mc_brem_spectrum` (bremsstrahlung background): the all-`None` branch likewise passes `z_mid = seg_r[:, 2]` to `_escape_length`/`_layer_dz` for self-absorption; the bremsstrahlung cross-section (`_brem_dsigma_dk`) depends only on `Z_i`, `seg_E`, `E_grid`.
- `plots/mpl/trajectories.py` (the only other consumer of `r_mid`): uses the full 3-vector `r` for plotting trajectory projections (`start @ e1`, `start @ e2`) — visualization only, not physics.

I independently confirm the central claim only in that limit: **under the fixed-far-field, laterally-infinite all-`None` model, a nonzero `beam_fwhm_mm` cannot change a single bin of the emitted spectrum** — it only relabels which `(x, y)` each electron's z-trajectory sits at.

This is now an explicit model boundary. A finite footprint consumes transverse position in `simulate_trajectories` (missed incidence and lateral exits) and in `mc_spectrum`/`mc_brem_spectrum` (the six-face escape distance). It can therefore change yields and spectra. Its ray-prism and attenuation physics is not adjudicated by this historic beam-size re-derivation.

## 3. Limiting case: `beam_fwhm_mm = None` vs `0.0`

`if beam_fwhm_mm:` — Python falsiness: `None` is falsy, `0.0` is falsy (`bool(0.0) is False`), any nonzero float is truthy. So both `None` and `0.0` skip the block entirely — no RNG draw happens for either (not even a zero-variance draw that would still advance `beam_rng`, which doesn't matter here since `beam_rng` is independent, but it does mean the falsy-zero case is truly a no-op path, not a "draw with σ=0" path). Confirmed by inspection and by `tests/montecarlo/test_montecarlo.py::test_beam_fwhm_mm_zero_and_none_are_equivalent`, which asserts `np.array_equal(none_segs["r_mid"], zero_segs["r_mid"])` — this is a real bit-for-bit check on the actual output array, not just a code-path inspection, and it passed.

Design judgment: falsy-zero is a deliberate and reasonable choice here (0 mm FWHM is physically a point source, so identifying it with `None` is correct), not an oversight — though it is worth noting for the record that this means a caller cannot pass a distinct sentinel to force "draw with σ=0" vs "skip the branch"; the two are indistinguishable, but since they're mathematically identical (delta function either way) this has no physical consequence.

## 4. RNG-independence

Verified independently (not just re-reading the code) with a fresh Python snippet:

```python
import numpy as np

seed = 42
rng_main = np.random.default_rng(seed)
draws_main_before = rng_main.random(10)

beam_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
_ = beam_rng.normal(size=(50000, 2))  # heavy draw from the "beam" stream

draws_main_after = rng_main.random(10)

rng_ref = np.random.default_rng(seed)  # never touched by beam_rng
ref_before = rng_ref.random(10)
ref_after = rng_ref.random(10)

assert np.array_equal(draws_main_before, ref_before)
assert np.array_equal(draws_main_after, ref_after)
```

Both assertions pass: drawing 100,000 numbers from `beam_rng` does not perturb `rng_main`'s sequence at all, confirming `rng` and `beam_rng` are two independent `Generator` objects backed by independent bit generators (`PCG64` seeded from two different, uncorrelated `SeedSequence` children — `spawn(2)[1]` is entropy-mixed from the parent seed and the spawn-key index, not a re-seed of the same stream). Also confirmed `beam_rng`'s stream differs from a `Generator` built directly from the same integer seed (i.e. `spawn` is not silently just returning the same-seeded generator).

This matches numpy's documented `SeedSequence.spawn` contract: each spawned child gets an independent, statistically uncorrelated substream, and `Generator` objects backed by different bit-generator states cannot affect each other's draw sequence — there is no shared mutable state between `rng` and `beam_rng`.

## 5. Statistical spot-check (independent, not just trusting the repo test)

```python
seed = 999
fwhm_mm = 2.5
sigma_ang_expected = fwhm_mm * 1e7 / (2 * np.sqrt(2 * np.log(2)))  # 1.06165e7 Ang
beam_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
draws = beam_rng.normal(0.0, sigma_ang_expected, size=(100000, 2))
# sampled std: 1.06522e7 (x), 1.06138e7 (y); rel err 0.34%, well within tolerance
```

Matches the formula to <0.5% at N=100,000 (independent draw, different seed and FWHM value than the repo's own test). The repo's own `test_beam_fwhm_mm_matches_gaussian_sigma` (N=20,000, `rel=0.05`) also passed in a live run (see §6) — corroborating, not identical, evidence.

## 6. Existing regression tests — read bodies, not just names

Ran on `feature/finite-electron-beam-size` (already checked out, matches described branch):

```
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/montecarlo/test_montecarlo.py -k beam_fwhm -v
...4 passed in 1.50s
```

Read all four test bodies (`tests/montecarlo/test_montecarlo.py` lines 69–137):

- `test_beam_fwhm_mm_zero_and_none_are_equivalent` — asserts `np.array_equal(none_segs["r_mid"], zero_segs["r_mid"])`. Directly tests the falsy-zero claim on the real output array. ✓ does what it says.
- `test_beam_fwhm_mm_offsets_transverse_position_only` — asserts every non-positional output (`E_keV`, `v_hat`, `L_ang`, `t_ang`, `elec_id`, `layer`, backscatter/transmit/stopped counts) is bit-for-bit identical between `beam_fwhm_mm=None` and `beam_fwhm_mm=1.0` runs at the same seed, that `r_mid[:, 2]` (depth) is untouched, and that the `r_mid[:, :2]` difference is a single constant vector per `elec_id` (not just per electron's first segment — checked across every segment that electron emits, via the per-electron-offset dict built while iterating all segments). This is exactly the RNG-independence + rigid-translation claim, tested against the real code path (not a toy reimplementation). ✓ does what it says.
- `test_beam_fwhm_mm_matches_gaussian_sigma` — reconstructs each electron's raw entry offset by inverting the first segment's midpoint (`r_mid - 0.5*L*v_hat`, i.e. undoing the first flight to recover `pos` at t=0), at N=20000, and checks `std(x0y0) ≈ FWHM/(2√(2ln2))` to 5%. This is a legitimate, non-circular statistical check (it doesn't just re-read `sigma_ang` from the implementation — it reconstructs the sampled points and computes their empirical std independently). ✓ does what it says.
- `test_beam_fwhm_mm_seed_reproducible` — same seed twice ⇒ bit-for-bit identical `r_mid`, confirming the beam-offset draw is deterministic given `seed` (not e.g. accidentally reading system entropy). ✓ does what it says.

All four tests pass and are not misleadingly named — they test what their names and the ledger notes claim.

## Filters

- **Units:** mm → Å conversion factor 1e7 correct; FWHM → σ conversion 2√(2ln2) correct; both independently recomputed above. ✓
- **Limiting case:** `beam_fwhm_mm ∈ {None, 0.0}` → bit-for-bit point source (verified by falsy-branch reasoning + the passing regression test). ✓
- **Sign/symmetry:** isotropic (azimuthally symmetric) 2D Gaussian, mean zero — no directional bias, consistent with "standard Gaussian beam-spot" convention; no sign convention to get wrong here (offset enters additively, symmetric about 0). ✓
- **RNG independence:** confirmed both by code inspection (separate `Generator` object from a `SeedSequence.spawn` child) and an independent numeric test (100,000-draw perturbation of `beam_rng` does not change `rng`'s next 10 draws). ✓
- **Central claim (all-`None` footprint only):** the fixed-far-field, laterally-infinite branches do not consume `x,y`, so beam size is spectrum-neutral there. A finite footprint is expressly excluded: it consumes `x,y` for missed incidence, lateral exits, and six-face self-absorption, under the separate unverified `finite-transverse-crystal` claim. ✓

## Adjudication

No discrepancy was found for the beam-size claim in its corrected scope: the unit/FWHM conversions are correct, the falsy-zero limiting case is intentional and verified bit-for-bit, and the RNG-independence design is correctly implemented and independently confirmed (spawned `SeedSequence` child ⇒ genuinely uncorrelated, non-state-sharing `Generator`). For an all-`None` footprint, a nonzero `beam_fwhm_mm` remains a no-op on the emitted spectrum by construction. The existing regression tests (`tests/montecarlo/test_montecarlo.py::test_beam_fwhm_mm_*`, 4/4 passing) genuinely exercise these beam-size claims against the real code path, not a reimplementation.

This adjudication does **not** extend to finite transverse dimensions. In that model, beam position changes missed incidence and lateral transport, while the spectra use the nearest of six escape faces; those claims remain `unverified` under `finite-transverse-crystal` pending an independent fresh context write-up.

One point for the record (not a discrepancy, since it is explicitly and correctly caveated in both the docstring and the ledger row): "standard Gaussian beam-spot parametrization" is not a physics law with a citable source equation the way most other rows are — it is a geometric convention. Verification here is necessarily about internal consistency (units, limiting case, code-physics decoupling) rather than agreement with an external derivation, since there is no external formula to re-derive against. This is appropriately reflected by the row not claiming a specific paper equation.

Recommended status: **`rederived`** (upgrade from `filtered`) — an independent fresh-context re-derivation/re-verification of every claim in the row (units, limiting case, RNG independence, and the central zero-physics-effect claim) matches the implementation, and the cited regression tests were confirmed (by reading their bodies, not just names) to actually pin these claims. Final `signed-off` is a human decision.

## Counter-addressed spot draws (#361 re-verification, 2026-10-07)

The spot offsets are now $(x_e, y_e)=(\sigma_x Z_{e,0},\,\sigma_y Z_{e,1})$, with $Z_{e,c}=\Phi^{-1}(u_{e,c})$ in the `spawn(2)[1]` child. The uniform and normal laws and their independence are derived in [beam-phase-space-injection §6](../beam-transport/beam-phase-space-injection.md). Independent checks at FWHM $0.05$ and $0.02$ mm with $N=2\times10^5$ gave std$/\sigma$ of $0.99866$ and $0.99919$ and $\rho(x,y)=-8.9\times10^{-4}$. Rows $e=0$ and $99999$ equal $\sigma\,\Phi^{-1}(u_{e,0/1})$ from a pure-Python SplitMix64 reference exactly. The block `start=1234`, $n=50$ equals the slice, and both widths `None` give exact zeros. A point-beam run is bit-identical to `main` on both cores. A default 1 mm spot changes only `initial_r_ang` and `r_mid`, the one-time realization change. No discrepancy.
