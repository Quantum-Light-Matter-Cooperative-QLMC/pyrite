# `bote-salvat-ionization-backend`

## Scope

Validation: `bote-salvat-ionization-backend`. Record for #92: the opt-in `ionization_model="bote-salvat"` backend of characteristic scoring, selected for runs by `Physics.shell_ionization_model`. EEDL stays the default. Physics and equations are in [Characteristic radiation](../../physics/radiation-physics/characteristic-radiation.md#ionization-model-option); the transcribed fit itself was independently re-derived under [`eedl-shell-ionization-comparison`](eedl-shell-ionization-comparison.md).

## Source and provenance

- Formula: Bote et al., *At. Data Nucl. Data Tables* **95**, 871 (2009), Eqs. (87)–(88), fitted to the DWBA/PWBA calculations of Bote and Salvat, *Phys. Rev. A* **77**, 042701 (2008).
- Parameters: NIST public-domain [BoteSalvatICX.jl `src/xione.jl`](https://github.com/usnistgov/BoteSalvatICX.jl/blob/8520cf5d002b11c3cf6669ebd5fedbb3de8d1fdb/src/xione.jl), commit `8520cf5`, SHA-256 `d0bd0d3d…41599`. Its `edge` field is the fit's own shell edge energy; it is used as-is, not replaced by EEDL's binding energy.
- Redistribution: the file is a U.S. Government work and is fetched from its pinned upstream URL, not packaged or mirrored. It is an opt-in dataset (`pyrite tables fetch bote-salvat`): bare `tables fetch` and `tables verify` skip it, and `pyrite remote sync` ships it only when it is installed locally, so it never blocks a default workflow.
- Coverage: K, L1–L3, M1–M5 for Z = 1–99 in the fit; Es (Z = 99) does not load under either model because of an existing xraydb core-hole-width gap. NIST SRD 164 tabulates the same model to 1 GeV; PENELOPE-2024 also carries N-shell tables, which this backend does not. N and higher shells keep EEDL.

## Method and checks

| Check | Result |
|---|---|
| Default marker | `characteristic_model_marker()` is byte-identical to the pre-#92 marker; `CHARACTERISTIC_MODEL` and every stored identity are unchanged |
| Identity separation | the Bote–Salvat marker appends `/bote-salvat-ionization-<sha[:12]>`; per-(element, yields, model) table cache entries are distinct |
| Threshold | each replaced row opens at the fit edge with `σ = 0`; scoring returns zero below it |
| Coverage | each row ends exactly at the EEDL row's last projectile energy (1e11 eV) |
| Interpolation | lin-lin on the tabulated grid within 6×10⁻⁴ of the analytical fit at every interval midpoint above U = 1.001 (observed maximum 5.0×10⁻⁴) for Z = 3–99; the U = 16 node pair preserves the fit's branch discontinuity, at most 1.15 % |
| Fit values | Si K at U = 2, 16, 100 matches the independent transcription of `eedl-shell-ionization-comparison` to 10⁻¹² |
| Scoring linearity | with zero attenuation the scored C spectrum scales exactly as σ_BS/σ_EEDL at the segment energy |
| Cascade isolation | line set and `line_yield_per_vacancy` are the same objects as the EEDL table's |
| Run selection | `Physics`/`Settings` default `"eedl"`; an EEDL case omits `shell_ionization_model` and keeps its dataset identity, content key and provenance marker; a Bote–Salvat case selects descriptor `emission.characteristic/bote-salvat` and the forked marker in all three |
| End to end | 30 keV on 1 µm Si, 200 electrons, same seed: integrated Si K line BS/EEDL = 1.044 |

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

## Fresh-context verification (2026-10-23)

An independent verifier derived Eqs. (87)–(88a–c) from the NIST review (J. Phys. Chem. Ref. Data 43, 013102) and the pinned `xione.jl` before reading the code, then compared at commit `6c598f5e`:

- **Formula.** An independent parser and scalar transcription agree with the code's parameters for all 99 elements and 731 shells. Values agree from U = 0.5 to 10³ and at 10⁹–10¹² eV, including U = 1 + 10⁻¹², 16 and 16(1 + 10⁻¹²). A 40-digit check bounds the code's error at 2×10⁻¹⁶. The code is zero for U ≤ 1 and switches branch at U = 16. An edge override changes only U.
- **Grid.** All 714 rows for Z = 1–98 are strictly increasing, start at the edge with σ = 0 and reproduce the formula at every node. Lin-lin error, sampled at 7 points per interval, is at most 5.0×10⁻⁴ above U = 1.001 (Mo M5) and 4.2×10⁻⁴ below it. The U = 16 jump, at most 1.153 % (Cr M1), is preserved. In float32 builds the node pair rounds to one value, so the jump becomes a step at a single point.
- **Scoring.** Replaced rows go through the EEDL interpolation and zero-outside mask. Scored σ matches the analytical value to 3.2×10⁻⁴ for Si, Cu and W. Cascade objects are shared. A single-segment Si spectrum gives BS/EEDL = 1.04156, against 1.04142 from the analytical σ_K ratio.
- **Identity.** The default marker is byte-identical to the parent commit for both yield sources. The cache keys cannot alias the two models.
- **Diagnostic.** Every table entry reproduces to three decimals from the analytical σ.

The verifier made two findings, now fixed. The physics page understated the fit-edge versus EEDL-edge difference; it is up to 643 eV (Cf K) and up to 15 % relative (Mn M3). The last grid node could fall one ulp short of the EEDL row's end; it is now set exactly and tested.

## Status and open items

Status `rederived`. Comparison with experimental production data near threshold and at high overvoltage (#92 plan item 3) is not yet done. Only a human may mark this row signed off.

Anchors: `tests/montecarlo/test_bote_salvat_backend.py`, `tests/montecarlo/test_shell_ionization_model_selection.py`.
