# SBETHE material inputs

Validation: `sbethe-material-inputs`.

## Independent derivation from the cited inputs

The ledger cites ICRU Report 37's independent-atom Bragg approximation, the
mass-density definition, and the elemental data in `materials/_transport_data.py`.
The vendored Salvat `sbethe.f` independently implements the same input rule:
its keyboard-composition path sums `ZT`, `AW`, and `XPOT` from stoichiometric
counts, then sets `EXPOT=EXP(XPOT/ZT)` and `VMOL=AVOG*RHO/AW` (lines
1170–1230). The ICRU report itself was not available for direct inspection.

Let $n_i$ be the catalog number density in atoms/Å$^3$, $A_i$ the atomic
weight in g/mol, $Z_i$ the atomic number, and $I_i$ the elemental mean
excitation energy in eV. Because 1 cm$^3=10^{24}$ Å$^3$, the number of
moles of element $i$ per cm$^3$ is $10^{24}n_i/N_A$. Its mass contribution
is therefore $10^{24}n_iA_i/N_A$ g/cm$^3$, giving

$$
\rho=\frac{10^{24}}{N_A}\sum_i n_i A_i.
$$

Bragg additivity applies to the electron-weighted stopping logarithm:
$Z_i\ln I_i$ per atom. Dividing the sum by the electron count gives

$$
\ln I=\frac{\sum_i n_iZ_i\ln I_i}{\sum_i n_iZ_i},
\qquad
I=\exp\!\left(\frac{\sum_i n_iZ_i\ln I_i}{\sum_i n_iZ_i}\right).
$$

The common factor converting number density to stoichiometric population
cancels from the ratio. For one element, $I=I_1$ and
$\rho=10^{24}n_1A_1/N_A$. Permuting elements changes neither expression;
scaling every $n_i$ by $c>0$ scales $\rho$ by $c$ but leaves $I$ and the
stoichiometric ratios unchanged. Every positive $n_i$, $Z_i$, and $I_i$
makes $I$ positive and bounded by the minimum and maximum elemental values.

## Implementation comparison

`material_inputs_from_composition` accumulates `mass_sum` as $\sum_i n_iA_i$,
`electron_sum` as $\sum_i n_iZ_i$, and `log_i_sum` as
$\sum_i n_iZ_i\ln I_i$. It converts the stored elemental `J_keV` to eV by
the factor $10^3$, then computes `mass_sum / (Avogadro * 1.0e-24)` and
`exp(log_i_sum / electron_sum)`. These are term-for-term the two derived
expressions, with no divergent factor, exponent, unit, or weighting.
Positive finite number densities and known elemental inputs are required.

The generated SBETHE deck needs relative atom counts. The function groups by
atomic number and divides every number density by the smallest; this common
normalization preserves $n_i/n_j$ and $I$. Ratios within $10^{-8}$ of an
integer are snapped to that integer, so the deck composition can differ by
at most that absolute ratio tolerance from the supplied composition. Density
and $I$ are computed from the original densities, before snapping. This is
the only non-exact composition convention in the mapping.

As an independent numeric check, the bundled silicon CIF lists eight atoms
in a cubic cell of side $5.4309$ Å. It gives
$n_{\rm Si}=8/(5.4309\,\mathrm{Å})^3=0.0499429934\,\mathrm{Å}^{-3}$.
Using the listed $A_{\rm Si}=28.085\,\mathrm{g\,mol^{-1}}$ and
$I_{\rm Si}=0.173\,\mathrm{keV}$ gives
$\rho=2.32915341\,\mathrm{g\,cm^{-3}}$ and $I=173\,\mathrm{eV}$.
These satisfy the test's $2.33\,\mathrm{g\,cm^{-3}}$ (2% tolerance) and
$173\,\mathrm{eV}$ anchors. For a separate two-element point with
$n_{\rm Mo}=0.01\,\mathrm{Å}^{-3}$ and
$n_{\rm S}=0.02\,\mathrm{Å}^{-3}$, the source elemental constants give
$\rho=2.65802488\,\mathrm{g\,cm^{-3}}$,
$I=292.725545\,\mathrm{eV}$, and deck atom counts $1:2$.

`material_request` passes the deck composition, density, and $I$ to
`material_identity`, which hashes all three; it includes a supplied band gap
in the material identity and the deck model record. The parametrized identity
test changes each input separately and confirms a different table key. The
catalog alias test confirms that a material alias uses its film crystal.

Verdict: **rederived** for the equations and source-to-code mapping. Direct
inspection of ICRU Report 37 was unavailable; Salvat's vendored keyboard
composition path independently confirms the same logarithmic $I$ rule.
Human sign-off remains separate.
