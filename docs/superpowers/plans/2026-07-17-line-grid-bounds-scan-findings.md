# Findings: full line-grid-bounds scan (SLURM job 207) — diagnostic grid too narrow at high beam energy

## Status

Task 3's empirical scan (`scripts/analyze_line_grid_bounds.py`, all 48 standard-profile
materials × all 7 standard beam energies) finished on the lab box (`qlmc`) at
2026-07-17T10:25:02-07:00, after ~16h. Output: `line_grid_bounds.json` (pulled to
`/tmp/claude-1000/-home-alexa-dev-cxr-mc/a02c4243-8651-4fee-90f6-ec0a94abf03f/scratchpad/line_grid_bounds.json`,
full log at `.../scratchpad/full_scan_log.txt`).

**`materials.toml` has NOT been patched.** The scan's results for the two highest beam
energies are not trustworthy as-is — see below. Do not proceed to Task 3 Step 2 with the
current JSON until this is resolved.

## Raw result table

```
  energy     raw_eV   stop_eV    num         driver   tilt    azim  spot?
-------------------------------------------------------------------------
      30     4515.0    5200.0   1731        diamond   9.89  100.00     no
      50     5825.0    6700.0   2231        diamond   9.89  100.00     no
     100     8590.0    9900.0   3284        diamond   9.89  100.00     no
     150     9540.0   11000.0   3651 mos2-on-sio2-si   9.89  100.00     no
     200     9825.0   11300.0   3751 mos2-on-sio2-si   9.89  150.00     no
     250       10.0     100.0     18          mote2   0.00   90.00     no
     300       10.0     100.0     18          pdse2   0.00   90.00     no
```

## The problem

`WIDE_GRID_EV = np.arange(10.0, 10000.0, 5.0)` (script's diagnostic histogram grid,
ceiling ~9995 eV) is too narrow at high beam energy.

- **250 keV and 300 keV rows are degenerate.** `raw_eV=10.0` is literally
  `coverage_energy`'s floor return value for a zero-intensity spectrum
  (`src/cxr_mc/line_grid_bounds.py`'s documented behavior). The driver for both rows is
  the *exact* tilt=0 geometry — the same geometric degeneracy the design spec already
  warned about (confirmed present at every energy via the script's own
  `[warn] <material>: zero coherent-line intensity at tilt=0 deg ...` lines, ~48 per
  energy in the log). For a degenerate zero-intensity candidate to *win* the `max()`
  over all refined candidates, every other candidate (real, non-degenerate tilt) must
  also have come back at the same 10.0 floor — i.e. literally nothing scanned at
  250/300 keV (48 materials × 2 near-zero tilts × 10 azimuths, plus 2 spot-check tilts)
  produced a single coherent-line photon inside the grid.

- **150 keV and 200 keV rows are suspect.** Their 99%-coverage points (9540 eV, 9825 eV)
  sit at 95% and 98% of the grid ceiling respectively. `coverage_energy` integrates only
  what's inside the grid — if the true spectrum extends past ~9995 eV, the computed 99%
  point is silently truncated low, the opposite failure mode from the degenerate case
  but the same root cause.

- **30/50/100 keV rows look safe** — 4515, 5825, 8590 eV, all comfortably clear of the
  ceiling.

## Why this is physically expected, not a code bug

Dispatched a fresh-context check of `E_res`'s beam-energy dependence
(`src/cxr_mc/montecarlo/spectrum.py`, `mc_spectrum()` → `_accumulate()`, "Eq. 10
resonance"):

```
E_res = HBARC_EV_ANG * (v·g) / (1 - v·n)      # v = beta*v_hat, Doppler-like denominator
```

`E_res` scales with β (electron speed), not γ directly. β increases from 0.6954 (200
keV) to 0.7409 (250 keV) to 0.7765 (300 keV) — a modest ~5% and ~9% relative increase
over 200 keV. But the Doppler denominator `1 - β·cosθ` amplifies this: for geometries
where `cosθ` is close to `1/β` (near-forward/grazing configurations, which do occur
across 48 materials × tilt/azimuth combos), the same β increase produces a much larger
— potentially large — jump in `E_res`. Given the 200 keV driver was already at 98% of
the grid ceiling, this modest β increase is fully sufficient to explain the 250/300 keV
rows going degenerate across essentially every material/geometry simultaneously.

Conclusion: this is the diagnostic tool repeating, one level down, the exact mistake
this whole plan was written to fix in the catalog itself — a fixed intensity-capture
window that's too narrow for high-energy beams, silently discarding (here, more
severely: completely losing) real radiated intensity.

## Recommended fix (not yet applied — pending your call)

1. Widen `WIDE_GRID_EV` in `scripts/analyze_line_grid_bounds.py` to a safely larger
   ceiling, e.g. `np.arange(10.0, 30000.0, 5.0)`.
2. Re-run the scan for the at-risk energies. Two scope options:
   - **Targeted (recommended):** re-run only 150/200/250/300 keV (the 4 rows that are
     either degenerate or within 5% of the old ceiling), keep 30/50/100 keV's results
     as-is. Saves ~3/7 of the full runtime (~16h → ~9-10h), and re-running 150/200 lets
     us confirm whether their original values were actually truncated or just
     coincidentally close to the old ceiling.
   - **Full re-run:** all 7 energies with the wider grid, for total self-consistency.
     Costs the full ~16h again; 30/50/100 keV are very unlikely to change given how far
     below the old ceiling they landed.
3. Once a trustworthy 7-row table exists, proceed to Task 3 Step 2 (patch
   `materials.toml` / `test_material_catalog.py` / the golden fingerprint file) as
   originally planned.

## Next action needed from the user

Confirm which re-run scope to use (targeted vs. full) and the new grid ceiling, then
I'll widen the script, submit the re-run job to the lab box, and resume the check-in
loop.
