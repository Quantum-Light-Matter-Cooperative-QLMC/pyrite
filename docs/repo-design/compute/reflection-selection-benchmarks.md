# Reflection-selection scoring benchmarks

[Machine-readable measurements](data/reflection-selection-2026-11-15.json).
The selector and its tolerance are described in the
[tolerance selector derivation](../../validation/radiation-physics/reflection-yield-tolerance-selector.md).

Issue #1200 measured the scoring cost of the tolerance-selected incoherent CXR
default from #1173 (`line_yield_tolerance=1e-3`) and reduced it without changing
the tolerance, the estimator or the selected sets. Two changes landed:

1. **Whole-table coupling fill.** `reflection_coupling_tables` evaluates all
   reflections of a call with array operations in the same order as the former
   per-row fill, and polarization pairs are vectorized. The output is
   bit-identical (code ade4f6d9f).
2. **Table-sized selection calls.** Each selection line call scores as many
   reflections as fit in 2 GiB of coupling tables (4 × `n_tab` × REAL per
   reflection), clamped to 64–8192. The count depends only on the tabulation
   length `n_tab` and the precision, never on the device, so the summation
   order and the output bits are the same on every machine. The admission check
   runs before any scoring, so a device that cannot hold one call is refused up
   front. The record stores `reflections_per_call` for each radiator, and
   `CXR_REFLECTION_SELECTION_MODEL` is `measured-shell-cutoff-v3` (code
   274a2c2fb).

`n_tab` grows with the output **span**, not its spacing. The internal
tabulation is a 1 eV mesh over the span ±20 %, joined with the Henke and EPDL
edge energies. A 0.5–10 keV band has 13,616 points, so a call holds 8,192
reflections. A 50 eV–300 keV automatic axis has 362,289 points, so a call holds
370 reflections. #1201 tracks a coarser tabulation away from the edges.

## Protocol

The profiler is `checks/cxr_selection_profile_1200.py`:

```bash
uv run python checks/cxr_selection_profile_1200.py silicon si.pkl out/si-band-node \
    --axis band --quadrature node --max-reflections 512 auto --repeats 2
```

- **Workload.** It replays frozen #642 transports (seed 642501) through the
  production `select_case_reflections`, using one untimed warm-up selection and
  then round-robin arms with two passes each.
- **Phase timing.** Phase timers synchronize the device. The JSON keeps the
  exclusive time of each phase.
- **Memory.** "Pool" is the device memory-pool reservation after the arm. The
  pool is emptied before each arm. This is a reservation, not the peak in use.
- **Hardware.** One NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB) on a
  hosted notebook runtime, using the CUDA backend.
- **Statistics.** Two passes give a range, not a confidence interval.

| Scene | Transport SHA-256 (prefix) | Segment rows |
|---|---|---:|
| Si slab, 300 keV, 10 µm, 64 primaries | `34e8391152b4` | 5,825 |
| MoS₂ on Si, 30 keV, 64 primaries | `a68a38be8946` | 30,118 |
| Si slab, 300 keV, 10 µm, 1,024 primaries | `10668ba9452f` | 96,603 |

The band axis is 0.5–10 keV with 0.09375 eV nodes (101,335 nodes). The
automatic axis is the case's own axis: 420,113 nodes for Si and 27,530 for the
layered scene.

## Final A/B (code 274a2c2fb)

Both arms run the same code in the same session. `512` is the former fixed call
size; `auto` is the new table-sized call.

| Scene / axis / quadrature | 512 (s) | auto (s) | Reflections per call | Speedup | Pool 512 → auto (GiB) |
|---|---:|---:|---:|---:|---:|
| Si, band, node | 3.34–3.37 | 3.02–3.05 | 8192 | 1.10× | 0.7 → 4.3 |
| Si, band, bin-mean | 9.90–9.94 | 6.33–6.34 | 8192 | 1.57× | 1.0 → 3.7 |
| MoS₂/Si, band, node | 25.81–25.89 | 24.35–24.37 | 8192 | 1.06× | 2.5 → 9.8 |
| MoS₂/Si, band, bin-mean | 37.58–37.81 | 23.72–23.93 | 8192 | 1.58× | 2.0 → 9.9 |
| Si, automatic, node | 35.10–35.49 | 36.80–36.88 | 370 | 0.96× | 8.1 → 4.8 |
| MoS₂/Si, automatic, node | 13.39–13.41 | 11.95–12.02 | 3480 / 3541 | 1.12× | 2.3 → 9.2 |
| Si 1,024 primaries, band, node | 28.16–28.21 | 28.04–28.10 | 8192 | 1.00× | 3.9 → 5.5 |

Every arm selects the same set as its warm-up, with the same cutoffs: Si G ≤
30 Å⁻¹ (13,688 reflections). The layered band case selects 40,198 + 13,688
reflections. The layered automatic case selects 40,198 + 16,624 (G ≤ 32 Å⁻¹ for
Si). The tail fractions match, and spectra differ by at most 4e-8 of the peak,
which comes from float32 summation order. On the Si automatic axis the
table-sized call is 4 % slower than 512, but it reserves 41 % less memory. The
512 call there already held 3.4 GB of tables at peak, and a 2048 call held
12.9 GB.

## Against the pre-#1200 code

The baseline is code 10735e82e with fixed 512-reflection calls. It ran in a
separate session on the same device.

| Scene / axis / quadrature | Baseline (s) | Table fill, 512 (s) | Final auto (s) | Total speedup |
|---|---:|---:|---:|---:|
| Si, band, node | 6.28 | 3.32 | 3.04 | 2.07× |
| Si, band, bin-mean | 12.81 | 9.86 | 6.33 | 2.02× |
| MoS₂/Si, band, node | 39.77 | 25.81 | 24.36 | 1.63× |
| MoS₂/Si, band, bin-mean | 52.36 | 38.23 | 23.82 | 2.20× |
| Si, automatic, node | 37.56 | 34.75 | 36.84 | 1.02× |
| MoS₂/Si, automatic, node | 27.73 | 13.36 | 11.99 | 2.31× |

The baseline's table and node-reduction phases were the largest, at 3.4 s and
1.6 s of the 6.3 s Si band node selection. After the table fill, the reduction
kernels dominate the remaining cost (see #1191 and #1192). The call-size sweep
at ade4f6d9f saturates near 4096–8192 reflections on the band axis: 6.6 / 6.3 /
6.3 s for Si bin-mean at 2048 / 4096 / 8192. On the Si automatic axis, smaller
calls cost more: 43.2 / 37.4 / 34.8 s at 128 / 256 / 512.

## Limits

- **Refused at the default grid budget.** The 1,024-primary Si transport on its
  automatic axis needs 615,662 nodes, above the default 600,000-node energy-grid
  budget, so selection is refused before scoring. A run with
  `PYRITE_ENERGY_GRID_MAX_POINTS=700000` took 667 s for its warm-up selection on
  this device, and selected the same 13,688-reflection set as the band axis.
- **Warm-up versus warm passes.** The first selection in a process includes
  table loading and JIT/kernel compilation. Warm-up times are in the JSON:
  13–53 s on the band axis, against 3–38 s for the warm passes.
- **Scope.** These runs cover CUDA only. CPU and ROCm call sizes follow the same
  device-independent rule, but they were not timed here.
- **Noise.** The 64-primary Si transport is Monte Carlo noisy: 50 eV block
  yields scatter by 24 % rms. The 1,024-primary transport selects the same set,
  with tail 3.71e-4 against 3.68e-4.
