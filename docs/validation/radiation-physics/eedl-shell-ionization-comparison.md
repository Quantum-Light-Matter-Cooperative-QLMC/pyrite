# `eedl-shell-ionization-comparison`

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

Bote–Salvat is a theoretical reference, not an independent measured-data validation for all shells. The calculation uses free-atom cross sections, EEDL binding energies for shape isolation, and PyRITE's existing relaxation model. M/N relaxation incompleteness and secondary-vacancy cascades remain outside this comparison. The initial implementation-context status was `filtered`; the fresh-context review below supports `rederived`. Only a human can mark it signed off.

Run `PYRITE_MC_BACKEND=cpu uv run python checks/shell_ionization_comparison.py --download --output report.json`. The fast anchors are in `tests/montecarlo/test_shell_ionization_comparison.py`.

## Fresh-context derivation and adjudication (2026-09-25)

This derivation was made from the [Bote et al. formula as reproduced in the NIST review, Eqs. (87)–(88)](https://www.nist.gov/system/files/documents/2018/03/01/1cross_sections_for_inner-shell_ionization_by_electron_impact.pdf) and the [pinned NIST coefficient source](https://github.com/usnistgov/BoteSalvatICX.jl/blob/8520cf5d002b11c3cf6669ebd5fedbb3de8d1fdb/src/xione.jl), before inspecting the Python implementation. The quantities are electron energy $E$ and shell binding energy $B_i$ in eV, overvoltage $U=E/B_i$, and ionization cross section in cm². The fit is for isolated neutral atoms and K, L, and M subshells; it combines a low-energy distorted-wave fit with a high-energy plane-wave Born fit. Substituting the EEDL binding energy for $B_i$ is a declared comparison convention, not a new fit.

For $1<U\leq16$, Eq. (87) gives

$$
\sigma_i(E)=4\pi a_0^2\frac{U-1}{U^2}
\left(a_1+a_2U+\frac{a_3}{1+U}
+\frac{a_4}{(1+U)^3}+\frac{a_5}{(1+U)^5}\right)^2.
$$

For $U>16$, Eqs. (88a)–(88c) give

$$
\begin{aligned}
\sigma_i(E)&=4\pi a_0^2\frac{A_i}{\beta^2}\frac{U}{U+b_i}
\left\{[\ln X^2-\beta^2]\left(1+\frac{g_1}{X}\right)
+g_2+g_3(1-\beta^2)^{1/4}+\frac{g_4}{X}\right\},\\
\beta^2&=\frac{E(E+2m_ec^2)}{(E+m_ec^2)^2},\qquad
X=\frac{\sqrt{E(E+2m_ec^2)}}{m_ec^2}.
\end{aligned}
$$

Here $a_0=5.291772108\times10^{-9}$ cm and $m_ec^2=5.10998918\times10^5$ eV, matching the pinned reference implementation. All fitted coefficients, $U$, $\beta$, and $X$ are dimensionless; $4\pi a_0^2$ supplies cm². The cross section is zero for $U\leq1$. The low branch approaches zero linearly from above threshold, and its square keeps it nonnegative. At high energy, $\beta^2\to1$, $U/(U+b_i)\to1$, and the leading fit grows logarithmically, as in the reference formula. The finite switch at $U=16$ is the source's branch convention.

For native EEDL points $(E_j,\sigma_j)$, the declared linear interpolation is

$$
\sigma^{\rm EEDL}(E)=\sigma_j+
\frac{E-E_j}{E_{j+1}-E_j}(\sigma_{j+1}-\sigma_j),
\qquad E_j\leq E\leq E_{j+1}.
$$

EEDL source values in barns become cm² through $1\ \mathrm{barn}=10^{-24}\ \mathrm{cm}^2$. At a native node this expression returns the source value. With the EEDL binding energy $B_i^{\rm EEDL}$ in the Bote–Salvat denominator, the shell ratio is defined only when that denominator is positive:

$$
R_i(E)=\frac{\sigma_i^{\rm EEDL}(E)}
{\sigma_i^{\rm BS}(E;B_i^{\rm EEDL})}.
$$

For each line family $\ell$, the model-dependent production is $P_\ell(E)=\sum_i\sigma_i(E)Y_{i\ell}$, with the *same* dimensionless relaxation yield $Y_{i\ell}$ in both sums. It therefore has cross-section units. The illustrative thick-target ratio integrates $P_\ell(E)/S(E)$ under continuous slowing down; the same stopping weight cancels its absolute normalization in the ratio. It does not model backscatter or photon absorption.

**Source-to-code diff.** `parse_bote_salvat` maps the pinned datum's five fields to the expected shell vectors and $n\times4$ and $n\times5$ matrices; the fetched bytes have SHA-256 `d0bd0d3ddca915a785a2562dfe29d6158b02d4a0e0330f54b784f8bba0b41599` and parse to 99 elements. `BoteSalvatElement.cross_section_cm2` matches both expressions, including the factor $(U-1)/U^2$, the high-branch factor $U/(U+b_i)$, the sign before $\beta^2$, and $g_3\sqrt{m_ec^2/(E+m_ec^2)}=(1-\beta^2)^{1/4}$. Its edge override changes $U$ while retaining the actual $E$ in $\beta$ and $X$, as the comparison requires. `compare_shells` uses the EEDL-edge convention and adjacent native-node ratios; `compare_production` multiplies both model vectors by the same line-yield matrix, retains EEDL for shells absent from Bote–Salvat, and returns an undefined ratio when reference production is zero.

An independent scalar transcription using the pinned Si K coefficients gives $8.393129394945435\times10^{-21}$ cm² at $U=2$, $5.356385964758975\times10^{-21}$ cm² at $U=16$, and $1.818776286011186\times10^{-21}$ cm² at $U=100$. The Python function agrees at these points and at $U=1.001$, $16.0001$, and $1000$ to at most $4.5\times10^{-16}$ relative. The comparison grid begins at $U=1.1$; over all 24 catalogue elements, its same-energy $U=1.1$ queries are above their EEDL thresholds and first tabulated energies. `compare_shells` uses NumPy's endpoint-held interpolation outside a native table, so that threshold conclusion applies to the stated grid, not arbitrary extrapolated queries.

**Adjudication:** units, threshold and high-energy limits, branch and edge conventions, and source-to-code terms pass. The published EEDL/Bote–Salvat ratio differences remain model differences, not transcription errors. The independent source-to-code claim is **rederived**; measured-data comparison and human sign-off remain open. The earlier implementation-context result record above is retained unchanged.
