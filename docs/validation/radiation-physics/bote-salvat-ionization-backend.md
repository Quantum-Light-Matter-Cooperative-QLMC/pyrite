# `bote-salvat-ionization-backend`

## Scope

Validation: `bote-salvat-ionization-backend`. Implementation-context record for #92: the opt-in `ionization_model="bote-salvat"` backend of characteristic scoring. EEDL stays the default and the only model the runner uses. Physics and equations are in [Characteristic radiation](../../physics/radiation-physics/characteristic-radiation.md#ionization-model-option); the transcribed fit itself was independently re-derived under [`eedl-shell-ionization-comparison`](eedl-shell-ionization-comparison.md).

## Source and provenance

- Formula: Bote et al., *At. Data Nucl. Data Tables* **95**, 871 (2009), Eqs. (87)–(88), fitted to the DWBA/PWBA calculations of Bote and Salvat, *Phys. Rev. A* **77**, 042701 (2008).
- Parameters: NIST public-domain [BoteSalvatICX.jl `src/xione.jl`](https://github.com/usnistgov/BoteSalvatICX.jl/blob/8520cf5d002b11c3cf6669ebd5fedbb3de8d1fdb/src/xione.jl), commit `8520cf5`, SHA-256 `d0bd0d3d…41599`. Its `edge` field is the fit's own shell edge energy; it is used as-is, not replaced by EEDL's binding energy.
- Redistribution: the file is a U.S. Government work and is fetched from its pinned upstream URL, not packaged or mirrored. It is an opt-in dataset (`pyrite tables fetch bote-salvat`): bare `tables fetch`, `tables verify` and `pyrite remote sync` skip it, so it never blocks a default workflow.
- Coverage: K, L1–L3, M1–M5 for Z = 1–99. NIST SRD 164 tabulates the same model to 1 GeV; PENELOPE-2024 also carries N-shell tables, which this backend does not. N and higher shells keep EEDL.

## Method and checks

| Check | Result |
|---|---|
| Default marker | `characteristic_model_marker()` is byte-identical to the pre-#92 marker; `CHARACTERISTIC_MODEL` and every stored identity are unchanged |
| Identity separation | the Bote–Salvat marker appends `/bote-salvat-ionization-<sha[:12]>`; per-(element, yields, model) table cache entries are distinct |
| Threshold | each replaced row opens at the fit edge with `σ = 0`; scoring returns zero below it |
| Coverage | each row ends at the EEDL row's last projectile energy (1e11 eV) |
| Interpolation | lin-lin on the tabulated grid within 6×10⁻⁴ of the analytical fit at every interval midpoint above U = 1.001 (observed maximum 5.0×10⁻⁴) for Z = 3–99; the U = 16 node pair preserves the fit's branch discontinuity, at most 1.15 % |
| Fit values | Si K at U = 2, 16, 100 matches the independent transcription of `eedl-shell-ionization-comparison` to 10⁻¹² |
| Scoring linearity | with zero attenuation the scored C spectrum scales exactly as σ_BS/σ_EEDL at the segment energy |
| Cascade isolation | line set and `line_yield_per_vacancy` are the same objects as the EEDL table's |

Synthetic-parameter tests run in every environment; pinned-data tests skip when the dataset is not installed.

## Model disagreement

`ionization_model_disagreement` gives Bote–Salvat/EEDL − 1 for thin-target line production with a shared cascade:

| Line | 10 keV | 30 keV | 100 keV | 1 MeV | 100 MeV |
|---|---:|---:|---:|---:|---:|
| Si Kα1 | +0.093 | +0.042 | −0.019 | −0.064 | +0.095 |
| Cu Kα1 | +0.394 | +0.216 | +0.048 | −0.074 | −0.016 |
| Cu Lα1 | +0.207 | +0.127 | +0.049 | −0.030 | +0.263 |
| W Lα1 | — | +1.127 | +0.385 | +0.055 | +0.067 |
| W Kα1 | — | — | +0.330 | −0.116 | −0.056 |

"—" is below the edge. These agree with the inverse ratios of `eedl-shell-ionization-comparison` (W L3 family EEDL/BS 0.472 at 30 keV and 0.721 at 100 keV). The two differ only in edge convention.

## Status and open items

Status `filtered`: units, limits, sign conventions and source-to-code checks pass in implementation context. A fresh-context derivation of the tabulation, identity and diagnostic is required for `rederived`. Comparison with experimental production data near threshold and at high overvoltage (#92 plan item 3) is not yet done. Only a human may mark this row signed off.

Anchors: `tests/montecarlo/test_bote_salvat_backend.py`.
