# SBETHE positron collision stopping

Validation: `sbethe-positron-stopping`. Independent verdict: rederived for
the selection and closure claim. The row was then anchored by a
value-pinning test against the positron Bethe formula (see Anchor
below). The independent tabulated stopping and CSDA-range
comparison was split to #331. Human sign-off remains pending (#277).

## Source and intended quantity

Salvat's SBETHE computes the collision stopping power of a material for the
projectile chosen by `IPROJ` (`vendor/xsgen/sbethe/sbethe.f`, lines
173–186): `IPROJ=2` is the positron, with $z_1=+1$ and unit mass. In that
mode SBETHE reads the positron shell-correction tables `pshcor-ZZ.tab`
(lines 340–347) instead of the electron `eshcor-ZZ.tab`. The density-effect
correction (`DENSIT`) is computed once per material and does not depend on
the projectile.

At high energy the result must reach the positron Bethe formula, PENELOPE-2024
Eq. 3.120 with Eq. 3.122:

$$
S^{(+)}=N\frac{2\pi e^4}{m_ev^2}Z\left[\ln\frac{E^2(\gamma+1)}{2I^2}+f^{(+)}(\gamma)-\delta_F\right].
$$

ICRU Report 37 uses the same Bethe expression for positrons, with its own
Sternheimer $\delta$. The Eq. 3.120 comparison checks the same high-energy
physics but is not a substitute for tabulated ICRU 37 values (#331).

## Checks

- **Deck and selection.** `SbetheDeck(projectile="positron")` answers the
  `IPROJ` prompt with `PROJECTILES["positron"]`$=2$. The request key and
  manifest `model` carry `projectile: positron`, so the electron key does
  not change. `resolve_catalog_table(key, projectile="positron")` resolves
  it and raises with a positron-specific hint if the table is missing.
  Installed `silicon`, `mos2` and `ws2` positron tables have manifest
  `projectile: positron` and the same 391-node energy grid as the electron
  tables.
- **Transport use.** `secondaries._positron_overrides` passes the positron
  `stopping_tables` to positron steps. `build_shell_inelastic_tables(projectile="positron")`
  checks that their grid equals the positron catalog grid.
  `adopted_stopping_cs(projectile="positron")` feeds
  `catalog_shell_rate_closure`, and the closed first moment equals the
  positron `stp.dat` value to $\le2.2\times10^{-16}$ at 5 keV, 20 keV and
  1 MeV for all three materials, and so does the soft plus hard sum at
  $W_c=500$ eV.
- **Physics sanity against the positron Bethe formula.** Ratio of SBETHE positron stopping to
  my Eq. 3.120 with $f^{(+)}$ (catalog $I$, oscillator $\delta_F$):

  | $E$ | silicon | MoS₂ | WS₂ |
  | --- | --- | --- | --- |
  | 10 keV | 0.981 | 0.915 | 0.897 |
  | 100 keV | 0.998 | 0.986 | 0.982 |
  | 1 MeV | 1.007 | 1.000 | 1.004 |
  | 10 MeV | 1.003 | 1.000 | 1.006 |
  | 100 MeV | 0.999 | 0.994 | 0.993 |

  The low-energy deficit is SBETHE's positron shell correction. Its factor
  $S/S_{\rm no\ shell}$ is 0.981 (Si) and 0.897 (WS₂) at 10 keV, which is
  exactly the residual. The positron/electron SBETHE ratio follows the Bethe
  ratio $(L+f^{(+)}-\delta)/(L+f^{(-)}-\delta)$ to within 0.6% from 100 keV
  to 100 MeV: e.g. silicon 1 MeV, 0.9762 against 0.9768. That rules out a
  mislabeled electron table.

## Scope limits

- No ICRU 37 positron stopping or CSDA-range data is in the repository.
  The independent tabulated stopping and CSDA-range comparison moved to
  #331 (decision 2026-10-05) and does not gate #276. This row does not
  claim ICRU 37 agreement or CSDA-range validation.
- `tests/xsgen/test_positron_tables.py` tests release plumbing and species
  separation only; it pins no stopping value.

## Ledger wording finding

The original **Assumptions** said that "the electron-table shell and
density-effect corrections apply unchanged". The density effect does, but
SBETHE positron mode applies its own positron shell corrections
(`pshcor-ZZ.tab`), which differ from the electron ones by up to 10% at
10 keV in WS₂. The ledger row now says so. The code was correct.

## Verdict

Rederived: transport uses the SBETHE `IPROJ=2` table, and the positron
partition closes on it. Recommended status: `rederived`, with `anchored`
once a value-pinning test exists.

## Anchor

`tests/montecarlo/test_sbethe_positron_stopping.py` pins the
recommended test: Si, MoS₂ and WS₂ SBETHE positron stopping lies within 1%
of Eq. 3.120 with $f^{(+)}$ at 1–100 MeV, and the positron/electron ratio
lies within 1% of the Bethe ratio. On that basis the row is `anchored`
(2026-10-05).
