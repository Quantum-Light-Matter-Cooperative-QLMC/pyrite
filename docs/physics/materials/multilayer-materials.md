# Multilayer film-on-substrate materials

Lab samples are typically a thin vdW **film** (MoSe₂ / MoS₂ / WS₂ / MoTe₂, tens of nm) grown or transferred onto a **substrate** (amorphous SiO₂, crystalline Si, or sapphire Al₂O₃, hundreds of µm). The measurable line flux therefore depends on the whole stack: each crystalline layer radiates its own lines, and every photon is attenuated by all layers on its way to the detector.

The pipeline models an ordered stack of layers. A single-crystal slab is the one-layer special case.

## Coupled effects

1. **Cross-stack self-absorption.** A line born in the film travels out through the rest of the film and the substrate (or the substrate and then the film, depending on exit face). For soft lines (≤4.5 keV) the substrate is optically thick, so its attenuation is often the dominant correction to the film flux.
2. **Per-layer radiation.** Each crystalline layer is its own PXR/CBS source with its own structure factor, reflections, and B-factor. The spectra add incoherently because the layers are physically separate crystals. An amorphous layer (fused-silica SiO₂) radiates no coherent lines; it only absorbs and produces bremsstrahlung.
3. **Multilayer electron transport.** Free path, stopping power, and scattering element change at a layer boundary. This matters when the film is not thin relative to the electron range, and for the substrate's own emission.

## Stack path

| Concern | Representation | Code site |
|---|---|---|
| Sample definition | catalog material + optional ordered `stack`/`substrate`; the resolved case keeps the film's scalar fields plus `abs_layers`/`layer_radiators` | `materials.catalog`, `sweep.build_cases` |
| Transport | composition-aware flights truncated at each internal boundary | `montecarlo.simulate_trajectories(layers=...)` |
| Line spectrum | one radiator per crystalline layer, summed incoherently | `montecarlo.runner._spectrum_case`, `mc_spectrum` |
| Self-absorption | piecewise optical depth through every crossed layer | `materials.attenuation._stack_tau` |
| Bremsstrahlung | per-layer emitting segments with the same cross-stack escape depth | `mc_brem_spectrum` |
| Geometry | whole stack shares one normal/tilt | `montecarlo.tilted_geometry` |
| Compound µ | `composition=[(el,n),…]` | `_normalize_composition`, `_mu_total_inv_ang` |
| Crystalline Si/sapphire | bundled phase-specific CIFs + catalog crystal rows | `data/cifs/`, `data/catalog/crystals/` |

Substrate elements (O, Al for SiO₂ / sapphire) need no hand-added atomic data: `henke_dispersion`/`load_henke` resolve any element, and `composition`-based compound absorption handles amorphous layers. Sapphire is represented as crystalline corundum, so it carries its own PXR/CBS radiator.

## Data model

A **Stack** is an ordered list of **Layers**, beam-entrance first. The film is the catalog material's crystal; `stack` lists the layers behind it:

```toml
[materials.mos2-on-sio2-si]
label = "MoS2 on SiO2/Si"
profile = "standard"
crystal = "mos2"
thickness_layers = { values = [3, 4, 5, 6] }
stack = [
  { material = "sio2", thickness_ang = 2850.0 },
  { material = "silicon", thickness_ang = 5000000.0 },
]
```

- A layer referencing `[media.<key>]` is amorphous: it absorbs and produces bremsstrahlung but has no coherent radiator.
- A layer referencing `[crystals.<key>]` receives that crystal's composition, default cut (`surface_hkl` for every packaged crystal), and reflection policy. An inline `beam_uvw` replaces the layer orientation and clears the inherited `surface_hkl` rather than merging with it.
- A single-layer stack reproduces the single-material result bit-for-bit, so the scalar `crystal`/`thickness_ang`/… path is internally promoted to a one-layer stack.

**Flow through the pipeline:**

- `MaterialCatalog.resolve_stack` resolves catalog references to immutable, ordered physical layers with cumulative boundaries and number densities.
- `sweep.build_cases` keeps the film's scalar `crystal`/`composition`/`hkl_list`/`B_ang2` fields and adds `abs_layers` plus aligned `layer_radiators` when a stack is present.
- `montecarlo._transport_case` passes the layer stack to `simulate_trajectories`.
- `montecarlo._spectrum_case` loops over crystalline layers, accumulating `mc_spectrum` per layer with the cross-stack `T_abs`; `mc_brem_spectrum` is summed per layer likewise.
- `results.store_result` / `plots` label records by stack name. Per-record metrics are unchanged, since `spec`/`brem` remain one array per case.
- Checkpoints store `abs_layers` and aligned `layer_radiators` for stacked cases; single-material checkpoints without those fields still load.

## Geometry and conventions

The beam enters at `z=0` along `+z`. Layer boundaries are `0 = z₀ < z₁ < … < z_N = Σ tᵢ`; layer *i* occupies `[z_{i-1}, z_i]`, layer 0 is the entrance film, and the substrate is the deepest. A point at depth `z` belongs to the layer whose interval contains it. The whole stack shares one normal and tilt (vdW films are conformal to the substrate).

## Cross-stack self-absorption

A single-material escape

```python
L_esc = z_mid / (-n_hat[2]) if n_hat[2] < 0 else (thickness - z_mid) / n_hat[2]
T_abs = exp(-L_esc * mu(E))
```

generalizes to a piecewise optical depth along the straight ray `r(s) = r_mid + s·n̂`. The ray runs in `z` from `z_mid` to the exit face (`z=0` if `n̂_z<0`, else `z=z_N`). Within each crossed layer *i* it travels `ℓ_i = Δz_i / |n̂_z|`, where `Δz_i` is the overlap of `[z_mid → z_exit]` with `[z_{i-1}, z_i]`:

```
T_abs(E) = exp( − (1/|n̂_z|) · Σ_i  μ_i(E) · Δz_i )
```

`μ_i(E)` is `_mu_total_inv_ang(layer_i.composition, E)`, vectorized over `E_res`. With few layers, the per-segment cost is a small fixed loop, negligible next to the sinc² matmul. `N=1` collapses to the single-material formula exactly.

## Per-layer radiation

`_spectrum_case` loops over the stack's crystalline layers and sums their spectra incoherently. Each layer:

- is assigned its segments by midpoint depth `z_mid` via `montecarlo._segments_in_layer`. Segments are short relative to layer thickness, so boundary-straddling is neglected.
- radiates with its own `crystal`/`hkl_list`/`B_ang2`/`beam_uvw` through `mc_spectrum`, applying the cross-stack `T_abs` (`layers=abs_layers`, the optical depth through every layer on the escape path).
- is described by `case["layer_radiators"]`, a per-layer list aligned with `abs_layers` and built in `sweep.build_cases`: a `{crystal, hkl_list, B_ang2, beam_uvw}` dict for a crystalline layer (the film, and a crystalline substrate via `sweep.substrate_radiator`), or `None` for an amorphous one. `layer_radiators=None` (no substrate) is the single-slab path.

Amorphous layers are skipped for lines, but their segments still feed bremsstrahlung and they still absorb. Bremsstrahlung (`mc_brem_spectrum`) is summed per layer with the same cross-stack `T_abs`.

A crystalline substrate is another material key: `Stack.on_substrate(film, thickness_ang, "silicon")` makes the substrate radiate its own (hkl) lines. The substrate radiates on the film's line bins, so `detector.energy_bins.line` must be wide enough to bracket both materials' lines. It uses the substrate crystal's default `beam_uvw`. Emission from deep in a thick substrate is strongly self-absorbed, so visible substrate lines come from near the interface.

## Multilayer electron transport

`simulate_trajectories(layers=…)` performs per-layer transport. Free path, stopping power, and scattering element switch with the electron's current layer. Flights truncate at internal boundaries (no collision there; the electron continues into the neighbor layer), so material is constant within a flight. Each segment carries its emitting `layer`. A single layer reproduces single-material transport bit-for-bit.

## Adding a stack to the catalog

N-layer stacks live in `data/catalog/materials/`; `substrate = "key"` is two-layer shorthand. To add a named stack runnable as `pyrite run standard -m <key>`:

1. If a crystalline phase is absent, add its bundled CIF under `data/cifs/` and a `crystals/<key>.toml` object. Add amorphous number densities in `media/<key>.toml`.
2. Add one `materials/<run-key>.toml` object with the film `crystal`, a profile or scan overrides, and either `substrate` or an ordered inline `stack` (never both).
3. Run `pyrite material validate`. Transport support errors are fatal. Mott tables are not checked here; `elastic_model="mott"` needs user-supplied SRD 64 tables and fails at run time when one is missing.

The material run key is the CLI/checkpoint name; the film crystal key drives crystallography. No transport, radiation, absorption, or plotting registry edit is required.

## Invariants

- A one-layer stack reproduces the single-material `spec`/`brem` bit-for-bit; mosaic, tilt, and checkpoint paths are unchanged (`checks/multilayer_check.py`, `tests/montecarlo/test_multilayer.py`).
- An amorphous substrate adds no lines. On a front (high-flux) exit it does not attenuate film segments, so the result equals the film-only spectrum bit-for-bit (`checks/multilayer_slice3_check.py`, `checks/multilayer_check.py`).
- With a crystalline substrate, the pipeline spectrum equals the per-layer incoherent sum exactly.
- The integrated film-line flux ratio with versus without substrate equals `exp(−μ_sub(E)·t_sub/|n̂_z|)` evaluated at the line's resonance energy `E_res`. The per-segment identity `τ_stack − τ_film = μ_sub·t_sub/|n̂_z|` is pinned in `tests/montecarlo/test_multilayer.py::test_stack_tau_back_exit_substrate_closed_form`.
- The depth-dose centroid and maximum penetration follow Kanaya–Okayama scaling: `z_max ≈ R_KO`, `E^1.67` with energy, and `A/(Z^0.889 ρ)` with material (`checks/multilayer_validation_check.py`).
- Positive tilt points the entrance-face normal toward the detector (front exit), leaving film lines unattenuated by the substrate. Negative tilt is the back-exit geometry, where the substrate attenuates film lines more as the exit path lengthens.
