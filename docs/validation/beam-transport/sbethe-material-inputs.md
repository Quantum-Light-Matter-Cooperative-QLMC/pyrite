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

Pending independent comparison with `material_inputs_from_composition`,
`material_request`, and their tests.
