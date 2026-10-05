# `inelastic-angular-deflection`

## Scope and independence

Validation: `inelastic-angular-deflection`. Fresh-context validation on 2026-10-04 in the `issue-317-inelastic-angular-deflection` worktree at commit `0b826c10`. The expressions below were derived from the ledger row, the physics page, EGSnrc PIRS-701 §2.4.7 (Eqs. 4.7.5–4.7.8 and 4.7.20–4.7.23, read from the PDF page image), and the Geant4 11.4.2 sources `G4eDPWAElasticDCS.cc` and `G4eDPWACoulombScatteringModel.cc`. Bethe (1953) and Kawrakow (1997) were not available. The implementation bodies were read only afterwards. This validates the size of the rate correction and how it is applied to the elastic rate coefficients. It does not validate the elastic tables, the shell soft/hard sampler, or the backscatter change this correction causes. It does not authorize human `signed-off` status.

## Independent derivation

Work in units $mc^2=1$ and $2\pi r_0^2=1$, with $\tau=T/mc^2$ and $\beta^2=1-(\tau+1)^{-2}$.

### Screened-Rutherford moment, $g_R$

Per atom, the screened-Rutherford nucleus is

$$
\frac{d\sigma_R}{d\mu}=\frac{Z^2}{\beta^2\tau(\tau+2)}\,\frac{1}{(1-\mu+2\eta)^2}.
$$

At $\eta=0$ this is Rutherford's $d\sigma/d\Omega=Z^2r_0^2/[(pc)^2\beta^2(1-\mu)^2]$ with $(pc)^2=\tau(\tau+2)$. Put $x=1-\mu$, $a=2\eta$ and $u=x+a$. Then $1-\mu^2=x(2-x)=-u^2+(2+2a)u-a(2+a)$, and

$$
\int_{-1}^{1}\frac{1-\mu^2}{(1-\mu+a)^2}\,d\mu
=\int_a^{2+a}\left[-1+\frac{2+2a}{u}-\frac{a(2+a)}{u^2}\right]du
=(2+2a)\ln\!\left(1+\frac2a\right)-4 .
$$

With $a=2\eta$ this is $2\left[(1+2\eta)\ln(1+1/\eta)-2\right]$, so

$$
\int(1-\mu^2)\,d\sigma_R=\frac{2Z^2\,g_R(\eta)}{\beta^2\tau(\tau+2)},\qquad
g_R(\eta)=(1+2\eta)\ln\!\left(1+\frac1\eta\right)-2 .
$$

The PDF typesets Eq. 4.7.22 as $(1+2\eta)\left[\ln(1+1/\eta)-2\right]$. That form differs by $4\eta$. It is negative for $\eta\gtrsim0.16$, whereas a moment of a positive density must be positive. The derived form tends to $1/(6\eta^2)>0$ as $\eta\to\infty$ (numerically $1.5\times10^{-3}$ at $\eta=10$). The PDF bracket is therefore a typesetting error, and the bracket placement used by Geant4 is the correct one.

### Møller moment, $g_M$

Per atomic electron, with $\varepsilon=w/\tau$,

$$
\frac{d\sigma_M}{dw}=\frac{1}{\beta^2\tau^2}\left[\frac1{\varepsilon^2}+\frac1{(1-\varepsilon)^2}+\left(\frac{\tau}{\tau+1}\right)^2-\frac{2\tau+1}{(\tau+1)^2}\,\frac1{\varepsilon(1-\varepsilon)}\right].
$$

Energy and momentum conservation for a primary that loses $w$ give $\cos^2\theta=(\tau-w)(\tau+2)/[\tau(\tau-w+2)]$, so

$$
\sin^2\theta=\frac{2w}{\tau(\tau-w+2)} .
$$

For a hard collision, $w\in[\tau_c,\tau/2]$, where the primary is the faster outgoing electron. I did not reduce $\int_{\tau_c}^{\tau/2}\sin^2\theta\,(d\sigma_M/dw)\,dw$ to closed form by hand. Instead I checked numerically that it equals

$$
\int_{\tau_c}^{\tau/2}\sin^2\theta\,\frac{d\sigma_M}{dw}\,dw=\frac{2\,g_M(\tau,\tau_c)}{\beta^2\tau(\tau+2)}
$$

with $g_M$ as printed in PIRS-701 Eq. 4.7.21. Adaptive quadrature at relative tolerance $10^{-13}$ gives the ratio $1.000000000000$ at $(T,T_c)=$ (300 keV, 50 eV), (300 keV, 1 keV), (10 keV, 1 keV), (1 MeV, 100 keV), (2 keV, 0.5 keV) and (5 keV, 2.4 keV). At $\tau=2\tau_c$, every logarithm in Eq. 4.7.21 has argument 1 and the last term has the factor $\tau-2\tau_c$. So $g_M\to0$ continuously, and $g_M=0$ for $\tau\le2\tau_c$.

### Scattering-power bookkeeping

Both moments carry the same prefactor $2/[\beta^2\tau(\tau+2)]$. Measure scattering power in units of the nuclear $\sin^2$ moment per $Z^2$, $2g_R/[\beta^2\tau(\tau+2)]$. Then:

- the nucleus contributes $Z^2$;
- Bethe's prescription sets the deflection by all atomic electrons, soft and hard, to $Z\xi_0$;
- the hard Møller collisions that the transport samples explicitly contribute $Z\cdot g_M/g_R$ (there are $Z$ electrons, each with moment $2g_M/[\beta^2\tau(\tau+2)]$).

The share left to fold into the elastic rate is $Z\xi_0-Z\,g_M/g_R$. The elastic weight is therefore

$$
Z^2+Z\left(\xi_0-\frac{g_M}{g_R}\right)=Z(Z+\xi),\qquad
\xi=\xi_0-\frac{g_M}{g_R}\ \text{clipped to}\ [0,\xi_0].
$$

The per-element rate factor is $Z(Z+\xi)/Z^2=1+\xi/Z$. Clipping at 0 means that once the explicit Møller share exceeds the assumed total, nothing is subtracted from the nucleus. The upper bound $\xi\le\xi_0$ is automatic, because $g_M\ge0$ is a moment of a positive density.

### Adjudicating PIRS-701 Eq. 4.7.20

Eq. 4.7.20 as printed gives $\xi=\xi_0\left[1-g_M/((\bar Z+\xi_0)g_R)\right]$. For one element with $\xi_0=1$, the weight is $Z(Z+1)-Z\,g_M/[(Z+1)g_R]$. That removes only a fraction $1/(Z+1)$ of the explicit hard share, which the derivation above does not support. Read the bracket instead as a multiplier on the whole weight $Z(Z+\xi_0)$:

$$
Z(Z+\xi_0)\left[1-\frac{g_M}{(Z+\xi_0)g_R}\right]=Z\left(Z+\xi_0-\frac{g_M}{g_R}\right).
$$

This reproduces the derivation exactly. It is also what Geant4 computes. `InitSCPCorrection` sets `scpCorr = 1 - gm*z0/(z0*(z0+1))` with `gm = min(g_M/g_R, 1)`, and `ComputeCrossSectionPerAtom` multiplies by `scpCorr*(1+1/Z)`. The printed Eq. 4.7.20 thus mislabels the multiplier of $Z(Z+\xi_0)$ as $\xi$. Using $\xi=\xi_0-g_M/g_R$ is the correct choice. For compounds, Geant4 uses one $Z_{\rm eff}$ in `scpCorr` and the element's own $Z$ in $1+1/Z$, so it agrees with $Z_i(Z_i+\xi)$ only for a single element. The physics page states that qualifier.

### Molière screening parameter

Eqs. 4.7.6–4.7.8 give $b_c=7821.6\,\rho Z_Se^{Z_E/Z_S}/(Ae^{Z_X/Z_S})$ in cm$^{-1}$ and $\chi_{cc}^2=0.1569\,\rho Z_S/A$ in MeV$^2$ cm$^{-1}$. The screening parameter is $\eta=\chi_{cc}^2/[4b_c\,m^2\tau(\tau+2)]$. In the ratio, $\rho$, $A$ and the common $Z_S$ cancel:

$$
\eta=\frac{0.1569\ \mathrm{MeV^2}}{7821.6}\,
\frac{\exp\!\left[(Z_X-Z_E)/Z_S\right]}{4\,(mc^2)^2\,\tau(\tau+2)} .
$$

This is dimensionless only with the $(mc^2)^2$ in MeV$^2$. Geant4's `A = moliereXc2/(4*tau*(tau+2)*moliereBc)` omits it. In Geant4 internal units (MeV = 1), its $\eta$ is $1/(0.511)^2=3.83$ times smaller. For one element, $\chi_{cc}^2/b_c=2.006\times10^{-5}\,Z^{2/3}(1+3.34\alpha^2Z^2)$ MeV$^2$. For Si at 300 keV ($\tau=0.5871$), this gives $\eta=7.601\times10^{-5}$. As a cross-check, Eq. 4.7.5, $\eta_0(1.13+3.76\alpha'^2)$, gives $7.8\times10^{-5}$ for the same case, which is consistent.

## Cheap filters

| check | result |
| --- | --- |
| units | $g_M$, $g_R$, $\eta$ and $\xi$ are dimensionless; $\chi_{cc}^2/b_c$ [MeV$^2$] over $(mc^2)^2$ [MeV$^2$]; the rate factor $1+\xi/Z$ is dimensionless — pass |
| limits | $T\le2T_c\Rightarrow g_M=0,\ \xi=1$; $T_c=\infty\Rightarrow\xi=1$, factor $(Z+1)/Z$; $Z\to\infty\Rightarrow$ factor $\to1$ (W: $75/74=1.0135$); $g_M\ge g_R\Rightarrow\xi=0$; $g_R>0$ for all $\eta>0$ — pass |
| signs/conventions | $\xi$ falls with $T$ and rises with $T_c$; $\sin^2\theta$ moments in a common normalization; $\tau_c=T_c/mc^2$ — pass |

## Source-to-code comparison

`atomic_electrons.py`:

- `moliere_screening_eta` (lines 49–64) uses atom fractions $p_i$ and weights $p_iZ_i(Z_i+\xi_0)$. It forms $Z_S$, $Z_E$ with $-\tfrac23\ln Z$, and $Z_X$ with $\ln(1+3.34\alpha^2Z^2)$. It returns $(0.1569/7821.6)\,e^{(Z_X-Z_E)/Z_S}/[4m^2\tau(\tau+2)]$ with $m^2=(0.51099895)^2$ MeV$^2$. This is identical to the derivation, including the $(mc^2)^2$ factor.
- `screened_rutherford_g` (line 70) is $(1+2\eta)\,\mathrm{log1p}(1/\eta)-2$, which is the derived bracket placement.
- `moller_g` (lines 73–91) is Eq. 4.7.21 term by term. The coefficient $\tfrac14(\tau+2)^2+(\tau+2)(\tau+\tfrac12)/(\tau+1)^2$ equals Geant4's $\tfrac14(\tau+2)\left[\tau+2+2(2\tau+1)/(\tau+1)^2\right]$. The function returns 0 where $\tau\le2\tau_c$.
- `atomic_electron_xi` (line 107) returns $\xi_0\left[1-\min(r/\xi_0,1)\right]=\xi_0-\min(r,\xi_0)$, with $r=g_M/g_R$. `cutoff_eV=None` returns $\xi_0$. `elastic_rate_scale` (line 112) is $1+\xi/Z$.

`layer_tables.py::build_layer_tables`:

- Lines 41–46 reject `kawrakow` when a shell cutoff is set and `elastic_model != "elsepa"`. `validate_inelastic_args` (`hard_inelastic.py:57-63`) forbids a cutoff in continuous mode. The analytic scaling at lines 115–120 therefore only ever runs with $\xi=\xi_0$, and there the constant factor $1+1/Z_i$ is exact.
- Only the numerators `L_sr_rate_numer` and `L_mott_numer` are scaled. Both enter $\sigma$ linearly (`scattering.py:105`, `:215`; `lut.py:182-190`). The Browning denominators and the Joy–Luo screening are untouched, so the angular law is unchanged.
- Lines 121–130 evaluate $1+\xi(E_k)/Z_i$ on each element's own ELSEPA energy nodes, using the layer composition and `hard_cutoff_eV` (the `inelastic_cutoff_eV` passed from `api.py:559`, or `None` in continuous mode).

`scattering.py::pack_elsepa_tables` (lines 412–416) multiplies `n_cm3 * sigma` by the node scale before the log. Each stored log-rate node therefore shifts by exactly $\ln(1+\xi/Z_i)$, and the pdf/CDF rows are built from `dcs_cm2_sr` and are unchanged. The flight rate and the element choice read the same per-element interpolants (`cores.py:317-334`), so $P(i)=\Sigma_i/\Sigma$ carries the factor consistently.

No divergent factor, sign, exponent, unit, or convention was found in the code.

## Numerical checks (independent of implementation helpers)

These were run on 2026-10-04 with scratch scripts that do not import PyRITE for the reference values.

- **$g_R$:** quadrature of $\tfrac12\int(1-\mu^2)(1-\mu+2\eta)^{-2}d\mu$ matches the formula to $\le5\times10^{-13}$ relative for $\eta=10^{-6}$ to $10$. At $\eta=0.1$, the PDF bracket gives 0.477 against the true 0.877.
- **$g_M$:** at the six $(T,T_c)$ points above, the quadrature of the Møller DCS times $\sin^2\theta$ equals $2g_M/[\beta^2\tau(\tau+2)]$ to 12 digits.
- **Code against the independent evaluation:** `atomic_electron_xi` for Si on 300 energies from 60 eV to 1 MeV, at $T_c=$ 50 eV, 1 keV and 10 keV, agrees to $\max\lvert\Delta\xi\rvert=2.4\times10^{-11}$. For W at $T_c=1$ keV it agrees to $1.2\times10^{-10}$. `moliere_screening_eta` for SiO$_2$ agrees to $3\times10^{-11}$ relative. For Si at 300 keV it gives $7.6012\times10^{-5}$.
- **Si, $W_c=50$ eV:** $\xi=1$ at $T\le100$ eV, 0.360 at 150 eV, 0.164 at 200 eV, 0.012 at 300 eV, and 0 from **315 eV** up to at least 1 MeV ($g_M/g_R=1.05$–$1.12$).
- **Si, $W_c\ge1$ keV, 2.5 keV–1 MeV:** $\xi\in[0.288,0.925]$ at 1 keV, and the minimum rises with the cutoff (0.36, 0.47 and 0.54 at 2, 5 and 10 keV). This is consistent with the page's "between 0.29 and 1".
- **Example factors:** Si at 300 keV, $T_c=1$ keV: $\xi=0.352$, factor 1.0252. The printed Eq. 4.7.20 would give $\xi=0.957$. W at 300 keV, $T_c=1$ keV: $\xi=0.155$, factor 1.0021.
- **Anchors:** `tests/montecarlo/test_atomic_electron_deflection.py` gives 28 passed. `tests/materials/test_profiles.py` with `-k "atomic_electron_deflection_forks or mott_shell_cases_keep"` gives 2 passed.

## Findings in the prose (not the code)

1. The physics page ("from about 500 eV upward") and the ledger row's limiting case ("from about 500 eV up") understate where the clipping starts. For Si at $W_c=50$ eV, $\xi$ reaches 0 at 315 eV ($\xi=0.012$ at 300 eV). The statement holds at 500 eV and above, but the onset is about 315 eV. Suggested wording: "from about 300 eV upward".
2. Eq. 4.7.22 as typeset in the PDF has the bracket misplaced, as derived above. The page and the code already use the correct form. The ledger note mentions only Eq. 4.7.20, so it could also record the Eq. 4.7.22 typesetting point.

Inherited modelling assumptions, recorded but not checked against Kawrakow (1997):

- The share is weighted by the $\sin^2\theta$ moment, not the first transport moment $1-\cos\theta$.
- The nuclear reference moment is screened Rutherford with Molière's $\eta$ rather than the active ELSEPA DCS.
- Møller kinematics are those of free electrons.

All three are the EGSnrc and Geant4 conventions, and the row's Assumptions already list them.

## Evidence and verdict

- **Filters:** units pass; limits pass; signs and conventions pass.
- **Re-derivation:** matches. $\xi=\xi_0-g_M/g_R$ follows from a common $\sin^2$ normalization. The printed PIRS-701 Eq. 4.7.20 is a mislabelled multiplier of $Z(Z+\xi_0)$, and the code's deviation from it is correct. The code uses the correct bracket placement for Eq. 4.7.22, and $\eta$ correctly includes $(mc^2)^2$.
- **Verdict:** `rederived`.
- **Human sign-off:** pending.
