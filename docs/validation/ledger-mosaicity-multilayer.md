# Mosaicity & multilayer (code-cross-checked; need sign-off + measured data)

Part of the [physics validation ledger](physics-validation-ledger.md). See the [validation methodology](methodology.md) for the status lifecycle and the [domain inventories](domain-inventories.md) for a claim-by-claim index.

## `mosaic-analytic`

- **Claim:** analytic broadening `FWHM = E·\|tan ψ\|·η`
- **Code:** `montecarlo/detector.py::mosaic_fwhm_eV`
- **Source:** `docs/physics/materials/crystal-mosaicity.md`
- **Status:** rederived
- **Checks:** units, zero/spread/normal-incidence limits, even symmetry, FWHM convention, and exact prefactor checked; unlinearized spot check differs by `2.03e-6` relative
- **Anchor:** `checks/mosaic_mc_check.py`
- **Notes:** energy-shift only; grazing divergence marks model breakdown; [validation write-up](materials/mosaic-analytic.md)

## `mosaic-mc`

- **Claim:** exact per-orientation incoherent average (2-D Gauss–Hermite)
- **Code:** `montecarlo/spectrum/lines.py::mc_spectrum`; `montecarlo/geometry.py::_mosaic_quadrature`
- **Source:** `docs/physics/materials/crystal-mosaicity.md`
- **Status:** rederived
- **Checks:** `1/π` normalization, positive symmetric weights, radial variance, proper rotations, η→0, constant-spectrum preservation, and incoherent intensity average checked
- **Anchor:** `checks/mosaic_mc_check.py`; `tests/montecarlo/test_mosaic_mc.py`
- **Notes:** broadens PXR+CBS; no grazing divergence; [validation write-up](materials/mosaic-mc.md)

## `multilayer-stack`

- **Claim:** film-on-substrate transport + absorption
- **Code:** `montecarlo/transport/api.py::simulate_trajectories` (`layers=`)
- **Source:** `../physics/materials/multilayer-materials.md` §(3) option A (CASINO-style boundary-aware multilayer transport)
- **Status:** rederived
- **Checks:** units; limits (one-layer reduction, identical-sublayer invariance, substrate backscatter); signs/conventions (entrance-first ordering, outer-face-only termination); one-layer `layers=` vs `composition=` bit-for-bit on every segment array and counter; 16-seed subdivision-invariance matrix
- **Anchor:** `checks/multilayer_validation_check.py`
- **Notes:** substrate-dominance prediction lives here. Fresh-context derivation: boundary truncation with a fresh per-layer exponential draw is the exact sampler for the inhomogeneous collision density `exp[−∫Σ(z(s))ds]` by memorylessness; implementation matches term-for-term (right-side `searchsorted` layer lookup, layer-indexed rates/stopping, no collision at an internal crossing, outer-face-only exit, per-segment `layer` tag). Artificial subdivision of a 3 µm C slab into three identical sub-layers leaves backscatter/transmission/stopped/path-length/path-weighted-depth invariant while segment count `+0.77%` and mean segment length `−0.76%` move oppositely at fixed path length (the expected extra-truncation signature). The one physical residual — transmission `−0.73%` (`z=−2.38`) under `energy_model="frozen"` — vanishes under `"midpoint"` (`−0.06%`, `z=−0.17`), attributing it to the left-endpoint energy rule (`transport-midpoint-stopping`, `energy-step-convergence`), not boundary handling. Segment midpoints never straddle their labelled layer; 100 nm C on Si raises `η` 0.0335→0.1090 vs 0.0330 for bulk C. Cross-stack escape factor is the separate `self-absorption` row. Human sign-off pending. [validation write-up](materials/multilayer-stack.md)
