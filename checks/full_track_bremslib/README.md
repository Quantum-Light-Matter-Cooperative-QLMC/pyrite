# Coupled BremsLib full-track comparison (issue #182)

This is an opt-in, slow comparison against Geant4 11.4.2 TestEm5, pinned to
source commit `8cc04f65977807f1848da7b958c421cd5e162f26`. It does not
sign off either BremsLib ledger claim. All raw inputs and outputs are retained.

## Reference setup

Apply `geant4-testem5.patch` with `git apply --unidiff-zero` to the pinned
Geant4 source. The patch adds
histogram 62 for photons whose creator process is `eBrem`; histogram 3 scores
all photons. The two histograms agree bin for bin in these runs. The upstream
TestEm5 source and patch are also preserved in `~/dev/geant4/issue-182/`.

Build TestEm5 against Geant4 11.4.2 with CMake `Release`, then run
`w_300kev.mac`, `si_300kev.mac`, `w_800kev.mac` and `si_800kev.mac` in the
build directory. The reference
environment used conda-forge Geant4 11.4.2, its packaged data and
`gxx_linux-64` compiler. See `GEANT4_BUILD.md` for exact
commands. A single thread and Geant4 random seeds `12345 67890` are fixed.
`geant4-conda-explicit.txt` pins the original Linux package set.

All cases use normal-incidence electrons in a 1 cm lateral slab:

| Case | Element | Energy | Density | Thickness | Primaries |
|---|---|---:|---:|---:|---:|
| `w_300kev` | W | 300 keV | 19.3 g/cm³ | 10 µm | 10,000 |
| `si_300kev` | Si | 300 keV | 2.329 g/cm³ | 100 µm | 10,000 |
| `w_800kev` | W | 800 keV | 19.3 g/cm³ | 10 µm | 10,000 |
| `si_800kev` | Si | 800 keV | 2.329 g/cm³ | 100 µm | 10,000 |

The 800 keV macros differ from the 300 keV macros only in gun energy and in
photon histograms extended to 80 ten-keV bins.

Geant4 uses `empenelope` (PenIoni, PenBrem), Goudsmit–Saunderson multiple
scattering, no loss fluctuations or fluorescence, a 10 keV electron stop
energy, a 1 nm global production range cut, and a 1 mm electron production
range cut. Production cuts do **not** stop electron tracking; the
`lowestElectronEnergy` command sets that limit. Photon histograms are scored
at creation in 10 keV bins, so photon transport and escape do not enter the
comparison. The gzip-compressed raw logs and CSV files in `reference/` are
the complete output needed for the tabulated observables.

## PyRITE setup

`test_benchmark.py` runs the exact CPU per-electron core with the released
BremsLib W or Si table, 10 keV electron cutoff, midpoint energy, no stopping
straggling, no transport LUT, and PyRITE seed 12345. Number density is
calculated from the density and atomic weight in the driver. The runs were
recorded as PyRITE commit `fa20e0ddc4aa1dcd412dd9837e295f22210a09ae` with its
locked dependencies. On 2026-09-26 the compute-node checkout was found to
differ from that commit outside `montecarlo/transport/` (including
`xsgen/bremslib/tables.py`). Its transport modules are identical, and the
BremsLib table digests in these records match the verified-pin emission runs
below, so these results are retained. These 10,000-primary records predate
batching; batch 0 of the current driver uses the same seed. The Slurm script
`run_cpu.sbatch` runs photon cutoffs of 1, 5 and 10 keV: 10,000 primaries at
1 keV and 3,000 at each higher cutoff. It requires a staged checkout and uv
environment on the remote CPU node. The exact JSON summaries and compressed CPU log are
in `results/`; table keys and digests are in each JSON record.

All four `.sbatch` scripts are site-neutral. Submit them from the staging
directory (`sbatch` with no path assumptions beyond the following):

- `PYRITE_BENCH_ROOT`: staging directory holding the pinned `pyrite/` checkout
  (`pyrite-fa20e0dd/` for `run_emission.sbatch`) and `test_benchmark.py`. It
  defaults to the submission directory; `run_emission.sbatch` requires it.
- `PYRITE_REMOTE_UV`: `uv` executable, default `$HOME/.local/bin/uv`. Use an
  absolute path; a literal `~` is not expanded here.
- Slurm logs (`--output`) are written to the submission directory. The scripts
  pin no node; add `sbatch -w NODE` (and `-p PARTITION` if your site does not
  use `cpu`) to target specific hardware.
The driver checks the segment event contract and calculates the vector recoil
residual for every hard photon. The target takes the residual momentum with
negligible recoil energy, as in the ledgered convention. All six reruns pass
the contract; maximum recorded componentwise recoil residual is
`1.42e-10 eV/c`. This is an internal accounting check, not an independent
Geant4 momentum comparison.
`run_sr_control.sbatch` repeats W at 1 keV photon cutoff with PyRITE's
screened-Rutherford elastic option; all other settings are held fixed.
`run_800kev.sbatch` historically ran W and Si at 800 keV with Mott elastic scattering and
the W screened-Rutherford control, 10,000 primaries each at a 1 keV hard
cutoff (log `results/benchmark_800kev_117.log.gz`). The Si 800 keV
screened-Rutherford control (`results/si_800kev_sr_1000ev.json`, about 10 s)
was run on the local CPU with the same driver, commit and seed.

The archived 800 keV Mott rows below are **invalid configurations**: they
exceed the standard SRD 64 table's 300 keV ceiling. Issue #318 adds a coverage
guard, so these rows cannot be reproduced with the current implementation
and those tables. They remain here as evidence of the former clamp defect;
use ELSEPA for new transport comparisons above the Mott table range.

To repeat one case in a prepared remote PyRITE checkout, set
`PYRITE_BENCH_CASE`, `PYRITE_BENCH_NE`, `PYRITE_BENCH_CUTOFF_EV`,
`PYRITE_BENCH_ELASTIC` (`mott` or `sr`) and `PYRITE_BENCH_OUTPUT`, then run:

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test checks/full_track_bremslib/test_benchmark.py -s -q
```

Run the comparison after all eleven JSON files are present:

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test checks/full_track_bremslib/test_compare.py -s -q
```

The comparison asserts three-standard-error screening tolerances on the
radiative observable, the photon creation yield above 10 keV, and on its
cutoff stability. Electron terminal fractions are printed, not asserted: they
test elastic transport, which is outside both BremsLib claims (see below).
These tolerances are not calibrated model-validation criteria. The cutoff runs reuse a PyRITE seed, so the printed independent
Poisson differences for those runs are approximate.

## Results and disposition

All rates are per incident electron at a 1 keV PyRITE hard cutoff unless
stated. `z` is the difference in independent standard errors.

| Case | Photons ≥10 keV PyRITE / Geant4 | z | Transmitted PyRITE Mott / Geant4 | z | Backscattered PyRITE Mott / Geant4 | z |
|---|---|---:|---|---:|---|---:|
| `w_300kev` | 0.0426 / 0.0372 | +1.91 | 0.4522 / 0.5526 | −14.27 | 0.4964 / 0.4200 | +10.87 |
| `si_300kev` | 0.0059 / 0.0058 | +0.09 | 0.8544 / 0.8505 | +0.78 | 0.1058 / 0.1059 | −0.02 |
| `w_800kev` | 0.0225 / 0.0230 | −0.23 | 0.9181 / 0.9204 | −0.60 | 0.0813 / 0.0795 | +0.47 |
| `si_800kev` | 0.0056 / 0.0047 | +0.89 | 0.9793 / 0.9948 | −9.72 | 0.0206 / 0.0052 | +9.67 |

Screened-Rutherford controls (only the elastic option changed):

| Case | Photons ≥10 keV z | Transmitted PyRITE SR / Geant4 | z | Backscattered PyRITE SR / Geant4 | z |
|---|---:|---|---:|---|---:|
| `w_300kev` | +0.04 | 0.5694 / 0.5526 | +2.39 | 0.3999 / 0.4200 | −2.89 |
| `w_800kev` | +0.09 | 0.9362 / 0.9204 | +4.33 | 0.0634 / 0.0795 | −4.42 |
| `si_800kev` | +0.30 | 0.9915 / 0.9948 | −2.83 | 0.0084 / 0.0052 | +2.75 |

**Radiative source yield.** Photon creation yields above 10 keV agree with
Geant4 within 1.91 standard errors in all 7 PyRITE runs, across both
energies, both elements and both elastic models. The coarse windows
10–50, 50–100, 100–300 and (800 keV) 300–800 keV agree within 2.10
standard errors at the 1 keV cutoff; sparse high-energy Si bins carry few
counts. At 300 keV, the 5 and 10 keV hard cutoffs stay within 1.1 standard
errors of each case's 1 keV photon yield.

**Electron terminal fractions.** Neither PyRITE elastic model tracks
Geant4's Goudsmit–Saunderson multiple scattering across the grid. Mott
agrees for Si 300 keV and W 800 keV but fails for W 300 keV and Si 800 keV;
screened Rutherford is marginal (<3σ) for W 300 keV and Si 800 keV but fails
for W 800 keV. The photon yield is insensitive to these changes. The
disposition is an elastic-transport discrepancy outside the BremsLib
radiative claims. The Si 800 keV Mott backscatter is about 4 times the
Geant4 and 2.5 times the PyRITE screened-Rutherford value. Issue #318 traced
the #183 excess to screening clamped at the 300 keV SRD 64 endpoint while
the Browning total kept falling: at 800 keV, Si's transport cross section
was 1.72 times ELSEPA and its cross section above 90° was 3.83 times ELSEPA
(W: 1.83 and 1.75 times). Current Mott transport rejects these out-of-table
energies. The historical W 800 keV agreement does not validate that invalid
configuration.

**Energy accounting.** PyRITE's total primary energy debit (initial energy
less terminal electron energy) is 70.28, 69.58 and 69.11 keV per W primary
at 1, 5 and 10 keV hard cutoffs at 300 keV; Si gives 78.30, 78.29 and 78.16
keV. These totals combine collisional and radiative loss. The isolated
radiative and recoil comparison is in the emission benchmark below.

**Scope differences.** The dumped Geant4 production thresholds are 990 eV
for photons and 2.31 MeV (W) or 546 keV (Si) for electrons, so Geant4 makes
no delta rays here. Its secondary electrons come from transported photons;
PyRITE does not transport photons or secondaries. BremsLib omits
electron–electron bremsstrahlung. Photon histograms are scored at creation,
so the >800 keV escape attenuation gap (#171) does not enter. No
detected-yield claim is made here.

## Emission and recoil benchmark (100,000 primaries)

This second benchmark isolates radiative energy loss from elastic transport
and compares the recoil and photon angle with Geant4. Its tolerances were
fixed in `test_compare_emission.py` before the PyRITE 100k results were
inspected.

### Reference

`geant4-testem5-emission.patch` replaces `geant4-testem5.patch` (it includes
histogram 62). For the primary track only, it adds:

- histogram 63: step length in µm by mean step energy
  `(T_pre + T_post + k)/2`, in 1 keV bins, where `k` is the step's eBrem
  photon energy;
- one `EBREM` log line per primary eBrem photon: event, post-emission
  primary energy, photon energy, post-emission primary direction and photon
  direction.

`*_emission.mac` are the four macros with 100,000 events, the histogram 63
setting, `/run/dumpCouples` and new output names; seeds are unchanged. See
`GEANT4_BUILD.md`. Adding the histogram left the logged emissions
bit-identical. The four gzip logs and `*_emission_h1_h63.csv` files in
`reference/` are the complete Geant4 inputs.

PenBrem's photon threshold is 990 eV. In the pinned
`G4PenelopeBremsstrahlungModel::SampleSecondaries`, the electron leaves along
`unit(p_in u - k n)` with kinetic energy `T - k`; the target takes the
collinear remainder. PyRITE keeps `u` and assigns
`(p(T) - p(T-k)) u - k n` to the target. Both neglect the recoil energy.
The comparison inverts Geant4's rule to recover the pre-emission direction
`u` from each logged emission.

### PyRITE

`test_benchmark.py` now transports 10,000-primary batches with seeds
`12345 + b`, which bounds its segment buffer. It records per-photon
`(primary, T, k, cos theta)`, the track-length spectrum by row-mean energy
in 1 keV bins, and a soft-radiative estimate. Soft radiation is included in
continuous stopping, so the driver re-evaluates `n S_soft(T; kc) L` for each
row at the row-mean energy. `run_emission.sbatch` runs every case at 1, 5 and
10 keV cutoffs with 100,000 primaries, from a checkout verified file by file
against `fa20e0dd`. The twelve `results/*_emission.json.gz` files are its
output. Geant4 TestEm5 uses Si at 2.330 g/cm³ and 28.09 g/mol, while PyRITE
uses 2.329 g/cm³ and 28.085 g/mol (a 0.04 % areal-density difference).

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test checks/full_track_bremslib/test_compare_emission.py -s -q
```

### Acceptance tolerances

A check passes when `|Δ| ≤ rel·|reference| + abs + 3σ`, with σ the combined
standard error of per-primary sums, or Poisson error for counts.

| Check | Allowance |
|---|---|
| Isolated model: Geant4 photons compared with BremsLib moments on Geant4's primary track-length spectrum (energy and count, `k > 1 keV`, `k ≥ 10 keV`) | 5 % (BremsLib compared with the scaled Seltzer–Berger DCS in PenBrem) |
| Implementation: PyRITE hard photons compared with BremsLib moments on PyRITE's own track-length spectrum | 1 % (1 keV bins at centre; hazard sampled at flight start) |
| Cutoff convergence: PyRITE hard + soft at 5 and 10 keV, compared with 1 keV | 0.5 % (soft estimator) |
| Full track: PyRITE photon energy `k ≥ 10 keV` (every cutoff) and `k > 1 keV` compared with Geant4 | 5 % |
| Photon angle to parent electron, `k ≥ 10 keV`: mean cosine | 0.03 absolute |
| Recoil in PyRITE's convention from both samples, `k ≥ 10 keV`: mean `|q|` and `q·u` | 5 % |

The cutoff runs share batch seeds and are treated as independent, which is
approximate.

### Results

All 76 checks pass. Rates are per primary; `z` is the difference in
standard errors.

| Case | Geant4 path, BremsLib / Geant4 `E(k>1keV)` eV | z | PyRITE path, realized / BremsLib `E_hard` (1 keV) eV | z | Hard+soft 1 / 5 / 10 keV eV | max \|z\| |
|---|---|---:|---|---:|---|---:|
| `w_300kev` | 2379 / 2372 | +0.15 | 2614 / 2602 | +0.22 | 2632 / 2682 / 2647 | 0.66 |
| `si_300kev` | 376 / 401 | −1.29 | 379 / 366 | +0.70 | 383 / 399 / 383 | 0.58 |
| `w_800kev` | 2695 / 2599 | +1.11 | 2765 / 2768 | −0.03 | 2772 / 2729 / 2776 | 0.34 |
| `si_800kev` | 475 / 510 | −0.93 | 542 / 510 | +0.85 | 543 / 554 / 546 | 0.20 |

Photon counts on the Geant4 path agree within 1.94σ, and the worst case is Si
800 keV at `k > 1 keV` (−6.5 %). On PyRITE's path, hard counts and energies at
all three cutoffs agree within 1.48σ; W agrees within 2.4 %. The 5 and 10 keV hard + soft totals agree with
1 keV within 4.1 % and 0.66σ. The soft estimate grows from 18 to 177 eV
for W 300 keV as the cutoff rises from 1 to 10 keV.

| Case | Mean cos θγ PyRITE / Geant4 | z | Mean \|q\| PyRITE conv. PyRITE / Geant4 keV/c | z | Geant4 native mean \|q\| keV/c | Geant4 electron deflection mean / p99 |
|---|---|---:|---|---:|---:|---|
| `w_300kev` | 0.498 / 0.484 | +1.34 | 72.8 / 72.6 | +0.09 | 62.5 | 4.6° / 22° |
| `si_300kev` | 0.558 / 0.519 | +1.48 | 59.3 / 62.9 | −0.97 | 53.8 | 4.0° / 21° |
| `w_800kev` | 0.726 / 0.750 | −2.18 | 71.9 / 70.0 | +0.54 | 49.0 | 3.0° / 27° |
| `si_800kev` | 0.730 / 0.769 | −1.66 | 58.6 / 58.9 | −0.05 | 39.8 | 2.6° / 27° |

The direct full-track photon energies pass only through the 5 % allowance
for W 300 keV (`k ≥ 10 keV`: +10 %, z up to +3.68). The PyRITE Mott primary
path is 25.3 µm, compared with Geant4's 22.8 µm (+11 %), and both isolated
checks agree for this case. The excess is therefore elastic transport (see #183), not radiation.

### Disposition

- **Isolated radiative loss.** BremsLib and PenBrem agree on a shared
  electron path within statistics; W resolution is about 2–3 % and Si about
  5–7 % (1σ). PyRITE's coupled transport realizes BremsLib's hard rate and
  first moment on its own paths. The measured cutoff convergence is limited
  to about 2–3 % (1σ) by statistics, not by the 0.5 % estimator allowance.
- **Photon angle.** Mean parent-relative cosines differ by at most 0.039
  (Si 300 keV, +1.48σ) and at most 2.18σ (W 800 keV, −0.025), within the
  0.03 + 3σ allowance. The models differ: BremsLib DDCS and PENELOPE shape
  functions.
- **Recoil.** With a common convention, the target recoil derived from the
  joint `(T, k, θ)` samples agrees within 6 % (≤1.01σ). The conventions
  differ. Geant4 deflects the electron by 2.6–4.6° on average per photon
  of at least 10 keV, while PyRITE does not. Its target recoil is 14–32 %
  smaller than in PyRITE's convention. These rare deflections are
  small compared with multiple scattering in these slabs. The difference is a
  documented convention, not a defect. Neither model gives a
  BremsLib-derived electron deflection.
- **Energy accounting.** Both codes debit exactly `k` for each photon and
  neglect recoil energy; Geant4 prints `edep + eleak = E0` for every case.

Remaining limits: the Mott elastic discrepancy (#183), photon transport and
detected yield (#171 above 800 keV), electron–electron bremsstrahlung, and
the directional point-detector scorer. Only a human may mark the ledger rows
`signed-off`.
