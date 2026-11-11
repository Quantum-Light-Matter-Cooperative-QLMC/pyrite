# EEDL material shell rates

Validation: `eedl-material-shell-rates`. Status: rederived (fresh-context, 2026-11-11). This is a host-side
rate preparation step; no electron transport mode consumes it.

## Source and construction

The packaged EPICS2025 EEDL ENDF tape supplies electron-impact subshell
ionization cross sections in MF=23, MT=534–572. The tape is pinned by its SHA-256
digest in `montecarlo/eedl_ionization.py`. Each section carries a shell
designator, binding energy, incident-energy grid, and cross section. The parser
accepts only ENDF linear-linear interpolation for these sections and converts
barns to cm². The catalog supplies each element's number density in atoms/Å³.

For element $i$ and shell $s$, the host rate is

$$
\lambda^{-1}_{i,s}(E)
  = n_i\,\sigma^{\rm EEDL}_{i,s}(E)\,10^{16}\quad {\rm \AA}^{-1},
\qquad
\lambda^{-1}_{\rm total}(E)=\sum_{i,s}\lambda^{-1}_{i,s}(E).
$$

The factor $10^{16}$ converts cm² to Å². The independent-atom mixture rule
assumes a homogeneous material and adds shell rates without a compound
correction. A shell below its binding energy has zero rate. An accessible shell
outside its EEDL projectile grid is rejected rather than extrapolated. Thus
the all-closed-shell limit has zero rate, and a one-element composition reduces
to its elemental rate multiplied by the element number density.

`material_shell_ionization_rates` preserves element, EEDL shell designator,
binding energy, and the native projectile-energy limits on every channel, so
a future hard event can identify the physical vacancy and its data coverage.
The total rates do not specify an energy-transfer distribution, secondary
kinematics, or a soft stopping moment. Those require the separately validated
GOS model and a positive stopping-closure procedure before any transport use.
The shared EEDL parser is a sibling of `transport` and `spectrum`, allowing
both to read the same pinned shell tables without importing either subsystem.

## Validation still needed

Compare interpolated per-shell and material rates with independently tabulated
EEDL values and the Bote–Salvat K/L/M reference across representative elements
and energies. Check that any later transfer spectrum reproduces selected shell
rates and closes to corrected SBETHE stopping without changing the inner-shell
vacancy rates. Fresh-context source-to-code verification remains pending.

## Independent verification

Fresh-context verification (2026-11-11), written before reading the
implementation bodies or the text above. Inputs were the ledger row, the
signatures and docstrings of `material_shell_ionization_rates` and
`load_eedl_shell_ionization`, the ENDF-6 formats manual (MF=23
electro-atomic sections), and the fetched EPICS2025 tape `EEDL.endf`. Its
SHA-256 `f3ef54f6…d1769c` matches the pinned digest.

### Source format

ENDF-6 MF=23 stores electro-atomic cross sections. MT=522 is total
electroionization, and MT=534–572 are subshell ionizations in the fixed order
K, L1–L3, M1–M5, N1–N7, O1–O9, P1–P11, Q1–Q3. The subshell designator is
therefore $d = \mathrm{MT}-533$, so that MT=534 is K ($d=1$), MT=537 is L3
($d=4$), MT=558 is O9 ($d=25$) and MT=559 is P1 ($d=26$). Each section is a
TAB1 record. Its $C_1$ field (EPE) is the subshell binding energy $B_s$ in eV.
The NBT/INT pairs give interpolation laws, and the points $(E_k,\sigma_k)$
give $E$ in eV and $\sigma$ in barns.

### Independent expression

In a homogeneous medium of independent atoms, channel $(i,s)$ has partial
inverse mean free path $\lambda^{-1}_{i,s}=N_i\sigma_{i,s}$, where $N_i$ is
the number of atoms of element $i$ per unit volume. Given $n_i$ in atoms/Å$^3$,
$N_i = 10^{24} n_i$ cm$^{-3}$. Also, $1\ {\rm cm}^{-1} = 10^{-8}$ Å$^{-1}$
and $1\ {\rm b} = 10^{-24}$ cm$^2$. Therefore

$$
\lambda^{-1}_{i,s}\,[{\rm \AA}^{-1}]
= n_i\,\sigma_{i,s}[{\rm cm}^2]\,10^{16}
= n_i\,\sigma_{i,s}[{\rm b}]\,10^{-8},
\qquad
\lambda^{-1}_{\rm tot}=\sum_i\sum_s\lambda^{-1}_{i,s}.
$$

The factor $10^{16}$ is the number of Å$^2$ per cm$^2$. This agrees with the
ledger equation.

ENDF interpolation law INT=2 (lin-lin) on $[E_k,E_{k+1}]$ gives

$$
\sigma(E)=\sigma_k+\left(\sigma_{k+1}-\sigma_k\right)\frac{E-E_k}{E_{k+1}-E_k}.
$$

At a node, $E=E_k$ returns $\sigma_k$ exactly. Each value is a convex
combination of nonnegative nodes, so $\sigma\ge0$. With $n_i>0$, every rate is
nonnegative.

The intended domain logic is as follows:

- $E\le B_s$: the channel is energetically closed, and $\sigma=0$.
- $B_s<E<E_1$ or $E>E_{N_P}$: the shell is accessible but the energy is outside
  the table. Reject it rather than extrapolate.
- Otherwise, use lin-lin interpolation.

Below $\min_{i,s}B_{i,s}$, every channel is closed and $\lambda^{-1}_{\rm tot}=0$.
For a single element, $\lambda^{-1}_{\rm tot}=n\sum_s\sigma_s$. The rate is
linear in each $n_i$.

### Cheap filters

| Filter | Result |
|---|---|
| Units: atoms/Å$^3$ × cm$^2$ × $10^{16}$ → Å$^{-1}$; barn→cm$^2$ uses $10^{-24}$ | pass |
| Below all binding energies → $\lambda^{-1}_{\rm tot}=0$ | pass (Si at $0.99\,B_{\min}$ and at $B_{\min}=8.15$ eV) |
| Accessible shell outside the grid is rejected | pass (Si at $E>10^{11}$ eV raises) |
| Nonnegativity | pass (parser rejects $\sigma<0$; $n_i\le0$ is rejected) |
| Linearity in $n_i$ | pass (doubling $n$ doubles the rate) |
| MT→designator ($d=\mathrm{MT}-533$, O8/O9 before P1) | pass |

### Whole-tape structural scan (independent fixed-column parser)

I parsed all 1612 MF=23 subshell sections in 100 materials with a standalone
reader of ENDF columns 1–66, MAT, MF and MT. The scan found:

- Every section declares INT=2 only.
- Every energy grid is strictly increasing, and no $\sigma$ is negative.
- For H, Si, Cu and Au, the sum of the interpolated subshells reproduces MT=522
  on its own grid to within $5\times10^{-9}$ relative. This independently
  confirms the MT range and the lin-lin semantics.
- In 1609 sections, the first node is $(E_1,\sigma_1)=(B_s,0)$. In the other
  three (Ni M4 and M5, MT=541/542, and Ag N5, MT=547), $B_s=10$ eV but
  $E_1\approx10.48$, 10.62 and 10.74 eV, with $\sigma_1$ of order $10^{6}$ b.
  For $B_s<E<E_1$ these shells are accessible but untabulated, so they are
  correctly rejected. The result is a narrow window of about 10–10.7 eV in
  which any Ni- or Ag-bearing composition cannot be evaluated. It lies far
  below the 1 keV transport cutoff.

### Comparison with the implementation

The code computes each channel as `density * sigma * _CM2_PER_ANG2`, with
`_CM2_PER_ANG2 = 1.0e16`. The parser stores `cross_section_barn * 1.0e-24`.
Interpolation uses `np.interp` on the native grid, and the parser
requires a strictly increasing grid and INT=2 throughout. Together these
reproduce the lin-lin formula above, including exact node values. The closed
test is `energy_ev <= shell.binding_energy_eV`, which gives $\sigma=0$, the
same value as the tabulated $(B_s,0)$ node. The out-of-range test
`energy_ev < grid[0] or energy_ev > grid[-1]` raises. Each channel carries the
element symbol, `shell_designator = mt - 533`, EPE and the grid limits. The
total is a plain sum over channels. The symbolic and dimensional comparison
found no divergent term.

The numeric reference values were computed only by the standalone parser and
interpolation above. Number densities come from the catalog.

| Case | Reference $\sigma$ (b) | Code $\sigma$ (cm$^2$) | Reference rate (Å$^{-1}$) | Code rate (Å$^{-1}$) |
|---|---|---|---|---|
| Si K, 30 keV, $n=0.0499430$ | 5035.649872 (between 19952.6 and 31622.8 eV) | $5.035649872\times10^{-21}$ | $2.514954\times10^{-6}$ | $2.514954\times10^{-6}$ |
| Si K, node 31622.8 eV | 4843.95 | $4.84395\times10^{-21}$ | $2.419214\times10^{-6}$ | $2.419214\times10^{-6}$ |
| Si K, $E=B_K=1844$ eV | 0 | 0 | 0 | 0 |
| ZrTe5 Zr K, 30 keV, $n_{\rm Zr}=0.00503052$ | 63.41435874 | $6.341435874\times10^{-23}$ | $3.190073\times10^{-9}$ | $3.190073\times10^{-9}$ |
| ZrTe5 Te L3, 30 keV, $n_{\rm Te}=0.0251526$ | 2188.745596 | $2.188745596\times10^{-21}$ | $5.505265\times10^{-7}$ | $5.505265\times10^{-7}$ |
| ZrTe5 Te K, 30 keV ($B_K$ above $E$) | 0 | 0 | 0 | 0 |

All agree to within one unit in the last place. The catalog `silicon` density,
0.0499430 Å$^{-3}$, is consistent with $8/a^3$ for $a\approx5.431$ Å.

### Findings (not repairs)

1. No test calls `material_shell_ionization_rates` directly. Its unit factor,
   closed-shell rule, out-of-grid rejection and identity fields are exercised
   only indirectly, through `transport/shell_rates.py::eedl_inner_cross_sections`
   in `tests/montecarlo/collisions/test_shell_rates.py`. The ledger row has no
   regression anchor of its own.
2. The ledger note says no production scheduler consumes these rates.
   However, `transport/shell_rates.py::eedl_inner_cross_sections`, which falls
   under `penelope-shell-rate-closure`, now calls this function for per-atom
   cross sections. The note should be brought up to date.
3. The ledger says "nonnegative". The code rejects $n_i=0$, so it
   requires $n_i>0$. This is stricter than the claim but consistent with it.
4. Ni and Ag have an untabulated threshold window, described in the
   structural scan above. It is harmless above 1 keV.
5. The name `_CM2_PER_ANG2` is inverted: the value $10^{16}$ is Å$^2$ per
   cm$^2$. The value and its use are correct.

### Verdict

`rederived`: the independent expression, interpolation semantics,
designator mapping, limits and numeric values match the implementation.
Human sign-off remains pending.
