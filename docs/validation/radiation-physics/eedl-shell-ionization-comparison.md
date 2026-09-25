# Validation: `eedl-shell-ionization-comparison`

## Scope and reference

Validation: `eedl-shell-ionization-comparison`. This is an implementation-context comparison for #86, not independent fresh-context validation. It compares PyRITE's packaged EEDL MF=23 subshell ionization with the Bote–Salvat model for the 24 catalogue transport elements. The reference parameters are NIST's public-domain [BoteSalvatICX.jl `xione.jl` at pinned commit `8520cf5`](https://github.com/usnistgov/BoteSalvatICX.jl/blob/8520cf5d002b11c3cf6669ebd5fedbb3de8d1fdb/src/xione.jl), fetched on demand and SHA-256 checked; they are not packaged. The fit follows Bote et al., *Atomic Data and Nuclear Data Tables* **95**, 871 (2009), and the distorted-wave/plane-wave calculation of Bote and Salvat, *Physical Review A* **77**, 042701 (2008).

## Equations and method

For overvoltage $U=E/E_i$, the Bote–Salvat low-energy branch used here is

$$
\sigma_i(E)=4\pi a_0^2(U-1)\left[\frac{A_1+A_2U+(A_3+(A_4+A_5/(1+U)^2)/(1+U)^2)/(1+U)}{U}\right]^2,\quad 1<U\le16.
$$

Above $U=16$, `shell_ionization.BoteSalvatElement.cross_section_cm2` uses the model's relativistic Bethe branch with its shell-specific $A_{nlj}$, $B_e$ and four $g$ parameters. Cross sections are in cm²; incident and edge energies are in eV. Both branches return zero at or below the shell edge. The transcription reproduces the pinned `xion.f` reference values to a maximum relative error of 0.572% over 12 supplied cases, below the 1% gate.

EEDL and Bote–Salvat edge energies differ. The check therefore reports ratios at a common incident energy and at the same *overvoltage relative to the EEDL edge*, evaluating Bote–Salvat with that edge. EEDL uses the same linear interpolation as production. For each queried energy the check compares its ratio with the ratios at the two adjacent native EEDL nodes. The local gate permits an 8% excursion because the Bote–Salvat denominator curves between EEDL's sparse nodes; the largest observed excursion is 6.05%. The gate tests interpolation behavior, not agreement between the two physical models. Ratios whose reference production is zero are undefined; below-edge K entries are labelled explicitly.

For characteristic production, both ionization models use the **same** PyRITE xraydb relaxation and L-shell Coster–Kronig transfer. Per line family, the check compares $\sum_i\sigma_iY_{i,\mathrm{line}}$. Shells without Bote–Salvat parameters retain EEDL in both sums. An illustrative thick-target ratio integrates each production cross section against approximate Bethe stopping, $\int\sigma_{\mathrm{prod}}(E)\,dE/S(E)$, from threshold to 30, 100 or 262.4 keV. It omits backscatter, photon absorption and secondary vacancies; it is not a measured yield prediction.

## Results (2026-09-25)

The pinned comparison covers 153 common K/L/M shells over 24 elements, at overvoltages 1.1–1000. The EEDL/Bote–Salvat shell ratios span 0.125–1.170. Representative ratios at the EEDL edge convention:

| Shell | $U=1.1$ | $U=2$ | $U=10$ | $U=100$ |
|---|---:|---:|---:|---:|
| Si K | 0.880 | 0.923 | 0.937 | 1.038 |
| W L3 | 0.228 | 0.368 | 0.723 | 0.932 |

The W L3 deficit is strongest near threshold and persists after accounting for the 3.8 eV edge difference (EEDL 10,209 eV; Bote–Salvat 10,205.2 eV). It feeds characteristic production: W L3-family ratios at 30, 100 and 300 keV are 0.472, 0.721 and 0.865. Approximate thick-target ratios for 30, 100 and 262.4 keV beams are 0.394, 0.605 and 0.740. W K production is undefined at 10 and 30 keV below its edge; the check prints “below K edge” and writes JSON `null`. Si K production ratios at 10, 30, 100 and 300 keV are 0.913, 0.961, 1.020 and 1.113.

## Verdict and limits

The transcription and local interpolation gates pass. The EEDL/Bote–Salvat disagreement is a source-model difference, especially for high-Z L-shell production near threshold; it is not an interpolation defect. The W L3 and Si K bands are committed as fast regression anchors. This evidence should inform #92's planned optional Bote–Salvat backend and its default-source decision; this validation slice does not change production physics.

Bote–Salvat is a theoretical reference, not an independent measured-data validation for all shells. The calculation uses free-atom cross sections, EEDL binding energies for shape isolation, and PyRITE's existing relaxation model. M/N relaxation incompleteness and secondary-vacancy cascades remain outside this comparison. The status is `filtered` pending a fresh-context source-to-code review; only a human can mark it signed off.

Run `PYRITE_MC_BACKEND=cpu uv run python checks/shell_ionization_comparison.py --download --output report.json`. The fast anchors are in `tests/montecarlo/test_shell_ionization_comparison.py`.
