# MoS₂ frozen-phonon oracle (#535)

Quantitative layered-crystal comparison for `cr-density-matrix-split-operator`.
This extends the [Si oracle](cr-density-matrix-split-operator-abtem.md) to
2H-MoS₂ with sourced uniaxial displacements. It records the remaining
discrepancy; it does not enable end-to-end MoS₂ transport or promote the
thermal/potential claims to `anchored`.

## Case and observable

- **Crystal:** catalog `mos2`, basal `(001)`, 295 K, 16.92 MeV kinetic
  energy, normal incidence, 1 µm depth, eight true periods across the planes.
- **Displacements:** independent Gaussian atoms, Cartesian covariance
  `diag(u_b, u_b, u_c)` about the hexagonal c axis. Mo uses
  `(0.0048, 0.0048, 0.0077)` Å² and S uses
  `(0.0038, 0.0038, 0.0049)` Å², from Schönfeld et al. (1983), Table 3,
  as recorded under `cr-uniaxial-thermal-projection`.
- **Beam:** `[1 10 0]`, checked against `[1 13 0]`; the cell retains the
  actual in-plane crystal correlations. `[1 7 0]` is insufficiently converged.
- **Oracle:** abTEM 1.0.10, ASE 3.29.0, NumPy 2.5.4; Lobato potential,
  GPU multislice on cluster-head, eight configurations (indices 0–7, seed 504).
  The interaction constant agrees with `e/(ħv)` to 8.06e-9 relative.
- **Solver:** TDS only, no electronic excitation, `n_cells = 8`, no sink
  (`margin_eV = 1e9`), harmonic order H = 48, dz = 12.5 Å.

The first eight eigenstates form six population groups:
`[0], [1], [2], [3, 4], [5], [6, 7]`. Gaps below 0.001 eV are merged.
The mirror-well pairs have arbitrary individual eigenvectors, which can
exchange order as H changes. Comparing their summed populations avoids a
spurious 0.177 entry-population difference. The grouping includes the whole
pair when the requested band count cuts through it.

## Convergence and remaining discrepancy

All differences below are maximum absolute grouped-population changes over
the shared recorded depths, 0–10000 Å. Coarse runs record every 500 Å; the
fine-grid run records every 1000 Å.

| Change | Maximum absolute difference |
|---|---:|
| Beam `[1 7 0]` → `[1 10 0]`, 0.02 Å sampling | 0.02217 |
| Beam `[1 10 0]` → `[1 13 0]`, 0.02 Å sampling | 0.001677 |
| Slice 1 → 0.5 Å, `[1 10 0]`, 0.02 Å sampling | 0.000996 |
| Sampling 0.02 → 0.0125 Å, `[1 10 0]`, 1 Å slice | 0.002149 |
| Solver dz 25 → 12.5 Å, H = 48 | 0.0000229 |
| Solver H = 48 → 64, dz = 12.5 Å | 0.000240 |
| Fine-grid oracle vs solver H = 48, dz = 12.5 Å | **0.002736** |

Maximum fine-grid SEM is 0.000338. At 1 µm the six oracle populations are
`0.010083 / 0.010538 / 0.010686 / 0.054200 / 0.011394 / 0.045512`;
the solver gives
`0.009143 / 0.009619 / 0.009994 / 0.052942 / 0.010749 / 0.045250`.

The residual exceeds sampling statistics and solver convergence errors.
The measured grid, slice and beam sensitivities describe its numerical
scale; they are not independent statistical error bars. The potential
parametrizations also differ: Lobato versus Waasmaier–Kirfel Mott–Bethe
factors differ by about 6% for Mo and 2% for S at s = 0.05 Å⁻¹, falling to
about 1% above s = 0.2 Å⁻¹. This comparison does not isolate their share of
the residual. No tolerance was fixed before inspecting these layered runs,
so no new pass/fail record is asserted.

This basal case exercises unequal transverse variances and a multi-element
sandwich well. Its transverse covariance has zero cross term; oblique-plane
cross terms are covered by the operator quadrature tests and
[independent anisotropic derivation](cr-lindblad-jump-operators-anisotropic.md),
not by this oracle. Directional electronic loss remains a separate gate
(#536). hBN, WSe₂ and graphite remain without admitted thermal rows;
general inequivalent-site tensors and temperature series remain outstanding.

## Reproduction and artifacts

Harness: `checks/abtem_channeling_oracle.py`. Recovered raw datasets remain
on cluster-head at `~/pyrite-535/results/oracle/` (`n7.npz`, `n10.npz`,
`n10half.npz`, `n10fineA.npz`, `n13.npz`).
[Machine-readable evidence](../check-records/abtem_mos2_535.json) retains
their SHA-256 hashes, configurations, revisions, grouped means/SEMs and the
three solver convergence arrays. Most GPU data were generated at
`eb7fc5e37c194fe84e60a118b0a23eed2e7e821c`; n13 and the comparison use
`57c2b4d453a2e7d055ffc52c2ff645620b91f2f6`.

Run heavy work remotely. From the staged checkout on cluster-head, the fine case is:

```bash
uv run --group oracle python checks/abtem_channeling_oracle.py run \
  --device gpu --crystal mos2 --hkl 0 0 1 --temperature 295 \
  --energy 16.92e6 --cells 8 --thickness 10000 --exit-every 1000 \
  --orders 48 --beam 1 10 0 --sampling 0.0125 --slice 1 \
  --configs 0 8 --seed 504 --out results/oracle/n10fineA.npz
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --group oracle python checks/abtem_channeling_oracle.py compare \
  results/oracle/n10fineA.npz --bands 8 --merge-eV 0.001 --dz 12.5 \
  --out results/oracle/n10fineA.json
```

The original local dz/H checks left no saved results. Recovery job 1511
on cluster-head's CPU partition recomputed H = 48 at dz = 25 and 12.5 Å and
H = 64 at dz = 12.5 Å, recording every 500 Å. BLAS/OpenMP used one thread;
each result was saved independently. Regression tests in
`tests/materials/test_channeling_oracle.py` pin the n13 orthogonal cell and
the grouped entrance populations across H = 48/64.
