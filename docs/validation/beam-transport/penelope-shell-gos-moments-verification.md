# PENELOPE shell GOS moments: independent verification

Validation: `penelope-shell-gos-moments`. This is the fresh-context verifier
record for branch `issue-93-soft-hard-inelastic-transport` at `dbc81be1`
(issue #93). It does not change the ledger status. Only a human may mark the
claim `signed-off`.

## Summary

- **Claim**: `penelope-shell-gos-moments`, covering
  `src/pyrite/montecarlo/transport/shell_gos.py::{shell_gos_moments,density_effect_correction,bethe_stopping_cs,formula_units_per_angstrom3,path_moments}`
  and the reused `src/pyrite/montecarlo/transport/inelastic.py::_qmin_ev`.
  Source: PENELOPE-2024 (NEA/MBDAV/R(2024)1) §§3.2.1–3.2.4, Eqs. 3.51, 3.56,
  3.59, 3.70–3.80, 3.83, 3.85–3.88, 3.94–3.110 and 3.115–3.121. The input
  oscillators come from `penelope-shell-oscillators` and are taken as given.
- **Filters**: units pass. Limits pass. Signs and conventions pass.
- **Re-derivation**: `matches`. My own code agrees with the implementation
  for every oscillator, channel and moment for Si, SiO₂ and MoS₂ at 1, 10 and
  100 keV and 1 MeV. The agreement is $\le1.3\times10^{-15}$ relative when
  both codes use a stable $Q_-$. With a naive $Q_-$ in my code the agreement
  is $\le4\times10^{-9}$, and all of that difference is cancellation in my
  $Q_-$.
- **Verdict**: `rederived`, with caveats. The main caveat is a manual
  inconsistency that the derivation doc does not list: Eq. 3.96 uses
  $\Theta(W-Q_k)$, but Eqs. 3.86, 3.92 and 3.106 use $Q'_k$ (item 1f). The
  other caveats are small numerical-wording errors in the derivation doc and
  the xfail reason.
- **Suggested ledger change**: set the status to `rederived` and add this
  record to **Anchor**. In **Notes**, add the Eq. 3.96 versus Eq. 3.106
  close-cutoff inconsistency and its impact. Also correct the Eq. 3.81 impact
  ranges to the dense-grid values: $\sigma^{(0)}$ up to 1.4% (at 10 MeV) and
  $\sigma^{(2)}$ up to 6.1% (Si, 1.24 keV). A human applies the change.

| item | subject | verdict |
| --- | --- | --- |
| 1 | equations and numbers (3.56, 3.59, 3.70–3.80, 3.83, 3.85–3.88, 3.94/3.95, 3.104–3.110, 3.115–3.121) | verified; the code follows Eq. 3.104 as decided; the manual's 3.96/3.106 inconsistency is flagged |
| 2 | interpretation choices | verified with caveats (per-choice table below) |
| 3 | independent numerics, ratio table, Bethe limit, factorisation | verified; three wording deviations |
| 4 | limits and units | verified; there is a threshold jump that the manual implies (item 4c) |
| 5 | tests | verified with caveats (assertions can fail; the xfails are justified but weak) |

## Independent derivation

I wrote this derivation from the manual pages (PDF pp. 116–131) before I read
the body of `shell_gos.py`.

**Prefactor and kinematics.** The prefactor is
$\mathcal P=2\pi e^4/(m_ev^2)=2\pi r_e^2m_ec^2/\beta^2$ in eV cm², with
$\gamma=1+E/m_ec^2$ and $\beta^2=1-\gamma^{-2}$. The moments are per formula
unit:

$$
\sigma^{(n)}_{\rm in}=\sum_k\big(\sigma^{(n)}_{{\rm dis,l},k}+\sigma^{(n)}_{{\rm dis,t},k}+\sigma^{(n)}_{{\rm clo},k}\big),\qquad n=0,1,2
\quad(\text{Eqs. 3.99, 3.103}).
$$

**Density effect (Eqs. 3.70–3.72).** With the oscillator OOS (Eq. 3.59),

$$
\mathcal F(L)=\frac{\Omega_p^2}{Z}\sum_k\frac{f_k}{W_k^2+L^2}=1-\beta^2,\qquad
\delta_F=\frac1Z\sum_kf_k\ln\!\Big(1+\frac{L^2}{W_k^2}\Big)-\frac{L^2}{\Omega_p^2}(1-\beta^2),
$$

and $\delta_F=0$ if $1-\beta^2\ge\mathcal F(0)$. The sum includes the
conduction band.

**Bound shell $U_k>0$, $E>U_k$.** The modified resonance and cutoff are

$$
W'_k=\begin{cases}W_k&E>3W_k-2U_k\\[2pt] (E+2U_k)/3&\text{otherwise}\end{cases},\qquad
Q'_k=\begin{cases}U_k&E>3W_k-2U_k\\U_kE/(3W_k-2U_k)&\text{otherwise}\end{cases}.
$$

The triangle end is $W_{\rm dis}=3W'_k-2U_k$ and
$W_{\max}=(E+U_k)/2$. $Q_-$ is Eq. 3.83 evaluated at $W'_k$. If
$Q_-<Q'_k$,

$$
\sigma^{(n)}_{{\rm dis,l},k}=\mathcal Pf_k\ln\!\Big[\frac{Q'_k(Q_-+2m_ec^2)}{Q_-(Q'_k+2m_ec^2)}\Big]M_n,\qquad
\sigma^{(n)}_{{\rm dis,t},k}=\mathcal Pf_k\Big[\ln\frac1{1-\beta^2}-\beta^2-\delta_F\Big]M_n,
$$

$$
M_n=\int_{U_k}^{\min(W_{\rm dis},W_{\max})}W^{n-1}\,\frac{2(W_{\rm dis}-W)}{(W_{\rm dis}-U_k)^2}\,dW
\quad(\text{Eqs. 3.76, 3.104, 3.105}).
$$

The antiderivatives are elementary. With $u=\min(W_{\rm dis},W_{\max})$ and
$D=W_{\rm dis}$,

$$
M_0=\tfrac{2}{(D-U)^2}\big[D\ln\tfrac uU-(u-U)\big],\quad
M_1=\tfrac{2}{(D-U)^2}\big[D(u-U)-\tfrac{u^2-U^2}2\big],\quad
M_2=\tfrac{2}{(D-U)^2}\big[\tfrac{D(u^2-U^2)}2-\tfrac{u^3-U^3}3\big].
$$

A sympy check of the untruncated triangle ($u=D$) gives $M_1=1$,
$M_2=(D+2U)/3=W'_k$ and $\langle W^2\rangle=(D^2+2DU+3U^2)/6$.

**Conduction band ($U=0$, Eq. 3.95 remark).** The band uses
$p_{\rm dis}\to\delta(W-W_{cb})$ and $Q_{cb}=W_{cb}$. It therefore contributes
$M_n=W_{cb}^{n-1}$ if $W_{cb}<W_{\max}=E/2$ and $Q_-(W_{cb})<W_{cb}$, and
nothing otherwise.

**Close collisions (Eqs. 3.86–3.88, 3.106–3.110).** With $E'=E+U_k$ and
$a=[E/(E+m_ec^2)]^2$,

$$
\sigma^{(n)}_{{\rm clo},k}=\mathcal Pf_k\big[J_n(W_{\max})-J_n(Q'_k)\big],\qquad
W^{n-2}F^{(-)}=W^{n-2}\Big[1+\Big(\tfrac{W}{E'-W}\Big)^2-\tfrac{(1-a)W}{E'-W}+\tfrac{aW^2}{E'^2}\Big].
$$

For the conduction band the lower limit is $Q_{cb}=W_{cb}$. Sympy confirms
three things. The expanded Eq. 3.87 equals the Eq. 3.107 integrand. Also
$dJ_n/dW=W^{n-2}F^{(-)}$ holds exactly for the manual's Eqs. 3.108–3.110,
and these remain valid with $E\to E'$.

**Bethe limit (Eqs. 3.120–3.121).**
$\sigma^{(1)}\to\mathcal PZ\{\ln[E^2(\gamma+1)/(2I^2)]+f^{(-)}(\gamma)-\delta_F\}$,
where
$f^{(-)}=1-\beta^2-(2\gamma-1)\gamma^{-2}\ln2+\tfrac18[(\gamma-1)/\gamma]^2$.

**Density (Eq. 3.51).** $N Z=\epsilon_0m_e\omega_p^2/e^2$ with
$\hbar\omega_p=\Omega_p$. The path moments are $N\sigma^{(n)}$ with
1 cm² $=10^{16}$ Å².

## Item 1: equations and their numbers

a. **Eqs. 3.56 and 3.59.** For bound shells $Q_k=U_k$. For the band,
$Q_{cb}=W_{cb}$ with a δ resonance. These are
`shell_gos.py:175-186`, and they match.

b. **Eqs. 3.70–3.73.** These are `shell_gos.py:69-95`, and they match. The
bracket `upper = Omega_p^2/(1-beta^2)` is valid because
$\mathcal F(\text{upper})<1-\beta^2$ strictly. For a single oscillator
($f=Z$) the root has a closed form,

$$
\delta_F=\ln\frac{\Omega_p^2}{(1-\beta^2)W^2}-1+\frac{(1-\beta^2)W^2}{\Omega_p^2}.
$$

The implementation reproduces it to $4\times10^{-16}$ at 1 MeV, 10 MeV and
1 GeV. The ultrarelativistic limit Eq. 3.73 holds at 10 GeV to
$6.5\times10^{-6}$ (Si), $9.3\times10^{-6}$ (SiO₂) and
$5.1\times10^{-5}$ (MoS₂).

c. **Eqs. 3.76–3.80 and 3.83.** These are `shell_gos.py:98-108` and
`176-183`. `_qmin_ev` (`inelastic.py:84-96`) rationalises
$p_0-p_1=W[2(E+m_ec^2)-W]/(p_0+p_1)$ and
$\sqrt{\Delta p^2+m^2c^4}-mc^2=\Delta p^2/(\sqrt{\cdot}+mc^2)$. It is
algebraically identical to Eq. 3.83 and is evaluated at $W'_k$ as the
manual requires. Both match.

d. **Eqs. 3.94/3.95 and 3.104/3.105.** These are `shell_gos.py:187-194`, and
they match. The code uses the Eq. 3.104 $W^{n-1}p_{\rm dis}$ form, as the
owner decided. I did not relitigate the choice. The Eq. 3.81/3.82
($p_{\rm dis}/W_k$) versus Eq. 3.94 ($p_{\rm dis}/W$) conflict is real.
Integrating Eq. 3.81 over $Q$ does not give Eq. 3.94. At the derivation
doc's sample points (1, 2, 5, 10, 20, 50, 100 keV) I reproduce its impact
numbers exactly: $\sigma^{(0)}$ $-0.16\%$ to $-1.24\%$, $|\sigma^{(1)}|\le0.69\%$,
and $\sigma^{(2)}$ $-0.53\%$ to $+5.47\%$. A dense grid gives larger extremes.
$\sigma^{(2)}$ reaches $+6.07\%$ for Si near 1.24 keV. $\sigma^{(0)}$ reaches
$-1.35\%$ (Si) and $-1.40\%$ (MoS₂) at 10 MeV, because the $\langle1/W\rangle$
difference persists at high energy. The ledger note "up to 1.2% / 5.5%"
therefore understates both extremes slightly.

e. **Eqs. 3.85–3.88 and 3.106–3.110.** These are `shell_gos.py:111-124` and
`195-200`, and they match. The code's $J_2$ differs from Eq. 3.110 by a
constant that cancels in the difference. Its value is $-E'$, not the
"$2E'$" stated in the derivation doc. I checked this symbolically and
numerically: at $E'=5000$ eV, code $-$ manual $=-5000.0$. This is a
documentation error only.

f. **Manual inconsistency that the derivation doc does not list.** Eq. 3.96
(the energy-loss close DCS) prints $\Theta(W-Q_k)$ with the unmodified
$Q_k=U_k$. Eqs. 3.86, 3.92 and 3.106 use $Q'_k$. The code follows
Eq. 3.106 (`shell_gos.py:195-199`, lower limit `q_mod`). Below
$E=3W_k-2U_k$, $Q'_k=U_kE/(3W_k-2U_k)<U_k$. So close losses with
$Q'_k\le W<U_k$ are allowed there. For those losses a knock-on energy
$W-U_k$ would be negative (for Si K at $E\to U_k^+$, $Q'_k=400$ eV while
$U_k=1844$ eV). The alternative, a close lower limit of $U_k$, changes the
total moments by the following amounts.

| material | $E\ge1$ keV: $\Delta\sigma^{(0)}$, $\Delta\sigma^{(1)}$, $\Delta\sigma^{(2)}$ | worst below 1 keV |
| --- | --- | --- |
| silicon | $-0.18\%$, $-3.3\%$, $-14.1\%$ at 1.85 keV (K edge) | $-16\%$, $-32\%$, $-55\%$ at 105 eV (L edge) |
| sio2 | $-0.57\%$, $-3.7\%$, $-12.1\%$ at 1.0–1.85 keV | $-14\%$, $-16\%$, $-25\%$ at 105 eV |
| mos2 | $-0.11\%$, $-2.6\%$, $-12.5\%$ at 1.0–2.9 keV | $-4.9\%$, $-16\%$, $-36\%$ at 237 eV |

Following three equations over one is defensible, and it is consistent with
the manual's stated purpose, "lowers an excessively high peak". However,
the choice should be recorded as an interpretation choice. A later sampler
also needs a rule for $W<U_k$ secondaries.

g. **Eqs. 3.115–3.121.** These are `shell_gos.py:204-226`, and they match
Eq. 3.121 term by term with the same $\delta_F$.

## Item 2: interpretation choices

| choice | manual | numeric impact of the alternative (dense grid 150 eV–10 MeV) |
| --- | --- | --- |
| Triangle broadening for all $U_k>0$ shells | **Supports (implicitly).** The Eq. 3.95 remark splits the terms only into "inner shells" ($p_{\rm dis}$) and the conduction band (δ). The EABS inner/outer split (p. 118) is introduced for ionisation and relaxation (§3.2.6), not for the DCS | A δ resonance at $W_k$ with $Q_k=U_k$ for *all* bound shells changes $\sigma^{(0,1,2)}$ by up to 1.4%, 1.4% and 8.9% in magnitude above 1 keV (MoS₂; SiO₂ $\sigma^{(2)}$ is $+5.1\%$), and by $-11\%$ and $-26\%$ in $\sigma^{(1,2)}$ for Si near 350 eV. With δ only for non-K/L/M shells, only MoS₂ is affected (Mo N shells): $\sigma^{(0)}$ $-0.8\%$ at 10 MeV, and $\sigma^{(1,2)}$ unchanged above 1 keV |
| $W_{\max}$ truncates $p_{\rm dis}$ (no renormalisation) | **Supports.** Eq. 3.88 covers "collisions (close and distant)", and Eqs. 3.98, 3.104 and 3.105 integrate to $W_{\max}$ | With no truncation, $\sigma^{(1)}$ rises by up to $+0.45\%$ (Si, 8.5 keV), $+0.85\%$ (SiO₂, 1 keV) and $+1.2\%$ (MoS₂, 1 keV), and $\sigma^{(2)}$ by up to $+6\%$. Near threshold $\int_U^{W_{\max}}p_{\rm dis}\to3/4$ |
| $a$ evaluated at $E$ | **Silent or ambiguous.** Eq. 3.85 defines $a(E)$. Eq. 3.87 primes only the ratio terms, but the text says $F^{(-)}$ "is calculated with the energy $E'$". I could not check the PENELOPE Fortran here | $a(E')$ changes $\sigma^{(1)}$ by $\le3.6\times10^{-5}$ and $\sigma^{(2)}$ by $\le3.3\times10^{-4}$ (MoS₂, about 220 keV). Negligible |
| Transverse bracket clipped at zero | **Silent.** Eq. 3.68 does not address $\delta_F>\ln\gamma^2-\beta^2$ | Active only for SiO₂ ($\mathcal F(0)=1.0067$; Si 0.9967 and MoS₂ 0.918 are never clipped) below **4.67 keV**, where $\delta_F\le1.64\times10^{-4}$. The derivation doc says "about 4.3 keV" and "up to $1.4\times10^{-4}$". Unclipped moments differ by $\le5.9\times10^{-6}$ relative |
| $N$ from $\Omega_p$ (Eq. 3.51) | **Supports.** Eq. 3.51 defines $\Omega_p$ from $N Z$. Inverting it is exact given the same $\Omega_p$ used in Eqs. 3.62–3.63 | Against the SBETHE-table density ($\rho N_A/M$ implied by `stopping_eV_per_angstrom / stopping_cs_eV_cm2`) the difference is $+1.6\times10^{-5}$ (Si), $+1.9\times10^{-5}$ (SiO₂) and $-5.5\times10^{-5}$ (MoS₂), consistent with the doc's "within $6\times10^{-5}$" |

## Item 3: independent numerics

My code is ephemeral and lives outside the repository, in the session
scratchpad. It takes only the oscillator set from `build_shell_oscillators`,
`catalog_material`, `plasma_energy_eV` and `load_conduction_bands`, and only
`stopping_*` arrays from `resolve_catalog_table`. It implements the formulas
above with the manual's Eqs. 3.108–3.110, my own $\delta_F$ root-finder and my
own triangle antiderivatives. I also cross-checked the close moments by direct
quadrature of Eq. 3.87 (agreement $\le10^{-11}$).

Total moments per formula unit from the independent code (the implementation
agrees to $\le1.3\times10^{-15}$ relative when the same $Q_-$ is used):

| material | E | $\sigma^{(0)}$ (cm²) | $\sigma^{(1)}$ (eV cm²) | $\sigma^{(2)}$ (eV² cm²) | $\delta_F$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| silicon | 1 keV | 1.07445e-16 | 3.84178e-15 | 5.50224e-13 | 6.0009e-08 |
| silicon | 10 keV | 1.55541e-17 | 7.87186e-16 | 4.90091e-13 | 1.8116e-04 |
| silicon | 100 keV | 2.46978e-18 | 1.52347e-16 | 5.40762e-13 | 1.6313e-02 |
| silicon | 1 MeV | 9.93244e-19 | 7.01380e-17 | 2.12685e-12 | 3.8386e-01 |
| sio2 | 1 keV | 2.95401e-16 | 9.79414e-15 | 9.79954e-13 | 3.0013e-05 |
| sio2 | 10 keV | 4.23917e-17 | 1.82466e-15 | 1.02048e-12 | 5.4467e-04 |
| sio2 | 100 keV | 6.72117e-18 | 3.42289e-16 | 1.15346e-12 | 3.1551e-02 |
| sio2 | 1 MeV | 2.64026e-18 | 1.53145e-16 | 4.55471e-12 | 6.9327e-01 |
| mos2 | 1 keV | 3.54012e-16 | 1.67453e-14 | 2.72579e-12 | 0 |
| mos2 | 10 keV | 5.25636e-17 | 3.67693e-15 | 2.69815e-12 | 0 |
| mos2 | 100 keV | 8.44000e-18 | 7.40115e-16 | 2.98468e-12 | 8.9765e-03 |
| mos2 | 1 MeV | 3.42212e-18 | 3.49126e-16 | 1.13407e-11 | 3.4648e-01 |

The path moments $N\sigma^{(n)}\times10^{16}$ also agree. For example, Si at
10 keV gives $7.768\times10^{-3}$ Å⁻¹, 0.3931 eV/Å and 244.8 eV²/Å.

**Ratio table.** I recomputed every row of the derivation doc's comparison
table. That includes GOS $\sigma^{(1)}$, `stp.dat`, GOS/stp, GOS/Bethe,
Bethe/no-shell and no-shell/stp for the three materials at 1, 2, 5, 10, 20,
50 and 100 keV, 1 MeV and 1 GeV. Every entry agrees at the printed
precision, except for one last-digit rounding (MoS₂ at 1 GeV, no-shell/stp:
1.0071 versus 1.0070). The pinned 1, 5, 10 and 100 keV ratios in the test
therefore reproduce.

**Bethe limit.** $|\sigma^{(1)}/\text{Bethe}-1|$ at 100 keV is
$2.6\times10^{-4}$ (Si), $9.2\times10^{-5}$ (SiO₂) and
$8.5\times10^{-4}$ (MoS₂), all $\le10^{-3}$ as claimed. On a dense
1 MeV–1 GeV grid the maximum is $1.3\times10^{-5}$ (Si), $9.3\times10^{-6}$
(SiO₂) and $4.19\times10^{-5}$ (MoS₂, about 19 MeV). The MoS₂ value slightly
exceeds the claimed "$4\times10^{-5}$ above 1 MeV". This is a wording issue,
not physics. The residual is the expected $O(Q_k/2m_ec^2)$ from the Mo K
shell ($U=20$ keV).

**Factorisation.** GOS/stp $=$ (GOS/Bethe)(Bethe/no-shell)(no-shell/stp) is a
telescoping identity, so "the columns multiply exactly" carries no
information by itself. The decomposition is sound and not circular:

- The Bethe value is computed from $Z$, $I$, $\gamma$ and $\delta_F$ alone,
  with no GOS moment in it.
- The SBETHE columns are external data.
- Bethe/no-shell $=1.0000$ at 1–5 keV, where both density effects vanish.
  This confirms that SBETHE's no-shell column uses the same $I$ and
  $f^{(-)}$, so the factor isolates the $\delta_F$ (oscillator versus
  continuous OOS) difference.

GOS/Bethe uses the same $\delta_F$ on both sides, so it isolates the
model-versus-Bethe excess.

## Item 4: limits and units

a. **High-energy Bethe asymptote.** This passes; see item 3. For a single
δ oscillator ($U=0$, $f=Z=4$, $W=I=50$ eV, $\Omega_p=30$ eV), the
implementation equals my closed form (log term plus transverse term plus
direct Møller quadrature) to $\le2\times10^{-14}$ at 1 keV–1 GeV. It
reaches Bethe to $\le2\times10^{-6}$ above 1 MeV.

b. **One-shell hydrogen-like fixture ($W=I$).** This passes, via the test
and my own code.

c. **$W_{\max}\to U$ near threshold.** For $E\to U_k^+$,
$W'_k\to(E+2U)/3\to U$ and $W_{\rm dis}=E$. The truncated triangle weight is
$\int_U^{W_{\max}}p_{\rm dis}=3/4$ exactly. However, $Q_-(W'_k\approx E)\approx
E>Q'_k$, so the distant terms vanish; I confirmed zero distant moments for Si
K up to $E=1.5U_K$. The close term does *not* vanish. It runs from
$Q'_k=U^2/(3W_k-2U)$ to $W_{\max}\approx U$ and is finite and positive: for
Si K, $\sigma^{(0)}_K=1.15\times10^{-19}$ cm², which is $1.8\times10^{-3}$ of
the total. So the shell moments jump from 0 at $E=U_k$ to a finite value at
$E=U_k^+$. This follows literally from Eqs. 3.80 and 3.106 (see item 1f),
and the implementation is faithful to it. The ledger limit "$E\le U_k$
closes a shell" is true, but a reader could take it to imply continuity at
threshold, which does not hold. Continuity at $E=3W_k-2U_k$ holds.

d. **Conduction band.** The band is a δ resonance with $Q_{cb}=W_{cb}$. It
contributes nothing when $W_{cb}\ge E/2$, including the close term, since the
lower limit equals $W_{\max}$ (`shell_gos.py:186,195`).

e. **Positivity and finiteness.** For $W\le E'/2$, write $r=W/(E'-W)\le1$.
Then $F^{(-)}\ge1-r+r^2\ge3/4$, so the close moments are positive. The
triangle moments are positive for $u>U$. $\ln(b/U)$ and $\ln Q'_k/Q_-$ are
finite under the guards `w_mod < E`, `loss[1] > 0` and `q_minus < q_mod`.
The transverse channel is non-negative by the clip.

f. **Units.** The prefactor is eV cm² (from $r_e^2$ in cm² times $m_ec^2$ in
eV). $\sigma^{(0,1,2)}$ are therefore in cm², eV cm² and eV² cm². $N$ is in
Å⁻³, and `path_moments` multiplies by $10^{24}\times10^{-8}=10^{16}$
Å²/cm², giving Å⁻¹, eV/Å and eV²/Å. These are correct. The Eq. 3.51
density agrees with SBETHE to $\le5.5\times10^{-5}$.

## Item 5: tests

Command:
`PYRITE_MC_BACKEND=cpu pyrite-dev test tests/montecarlo/test_shell_gos.py -q`.
Result: 32 passed and 3 xfailed.

- **Assertions can fail.** Every `pytest.approx` on a cm²-scale quantity
  passes `abs=0.0` explicitly. The remaining non-zero `abs` values apply to
  dimensionless quantities: $\delta_F$ (`abs=1e-3`, O(10)) and the stopping
  ratios (`abs=2e-4`, O(1)). The `np.allclose` calls use `atol=0.0`. The
  `abs=1e-12` trap is absent.
- **What catches what.**
  - The triangle test (`row[0] = <1/W> row[1]`) would catch a swap to
    Eq. 3.81.
  - The pinned-ratio test at 1, 5 and 10 keV would catch removal of the
    $W_{\max}$ truncation (up to 1.2% in $\sigma^{(1)}$) and the Eq. 3.96
    cutoff alternative.
  - The close-quadrature test mirrors the implementation's own choices of
    $a(E)$ and $Q'_k$. It checks the algebra, not the interpretation.
- **Coverage gaps.**
  - $\delta_F$ at intermediate $\beta$ is pinned only through Eq. 3.73 at
    10 GeV. The Bethe comparisons share the same function, so an error
    common to both would cancel. Recommend a single-oscillator closed-form
    $\delta_F$ test (item 1b formula).
  - The SiO₂ clip region has no test.
  - The threshold jump (item 4c) has no test.
- **Strict xfails** (`test_fetched_raw_stopping_matches_sbethe_below_10_kev`,
  ×3). These are justified. The raw δ-oscillator stopping genuinely exceeds
  the shell-corrected `stp.dat`, and the manual claims ICRU agreement only
  for $E\ge10$ keV (Fig. 3.10). However, each case fails at the first loop
  energy (1 keV), so the xfail records nothing about 2–10 keV; the
  pinned-ratio test is the real record. The reason string "7–26% at
  1–10 keV" is inaccurate for Si, whose excess is 13.3%, 10.6%, 4.0% and
  1.4% at 1, 2, 5 and 10 keV. It also does not cover the 1.4% at Si 10 keV.
  Recommend replacing the xfail with a positive assertion, such as
  GOS/stp $>1.01$ at 1–5 keV per material, or rewording the reason to
  "1.4–26%".

## Recommended fixes (description only; not applied)

1. Record the Eq. 3.96 versus Eq. 3.106 close-cutoff inconsistency as
   interpretation choice 7 in the derivation doc and the ledger. Include the
   item 1f impact table and the resulting threshold jump and $W<U_k$ close
   losses. Decide before any sampler consumes these moments.
2. In the derivation doc, correct the following:
   - the $J_2$ constant is $E'$, not $2E'$;
   - the SiO₂ clip region is $E<4.67$ keV with $\delta_F\le1.64\times10^{-4}$;
   - the high-energy Bethe residual is $\le4.2\times10^{-5}$ (MoS₂);
   - the Eq. 3.81 impact extremes on a dense grid are $\sigma^{(0)}$ to
     $-1.4\%$ at 10 MeV and $\sigma^{(2)}$ to $+6.1\%$ at 1.24 keV.
3. Tests: add a single-oscillator closed-form $\delta_F$ check. Optionally
   add a SiO₂ zero-transverse check below 4.6 keV. Reword the xfail reason or
   replace it with positive assertions.
4. Rephrase the ledger limiting case "$E\le U_k$ closes a shell" so that it
   does not imply continuity at $E=U_k$.

## Commands

The following commands ran in `/tmp/pyrite-issue-93` with
`PYRITE_MC_BACKEND=cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache`:

- `pdftotext -layout` of the PENELOPE-2024 PDF, and page images of PDF
  pp. 140–146 and 150–151 for Eqs. 3.68–3.114 and 3.115–3.123;
- `uv run python` running sympy checks of Eqs. 3.107–3.110 and the triangle
  moments;
- `uv run python` running the independent-moment, ratio-table, variant,
  limit and Bethe scripts (session scratchpad, not committed);
- `pyrite-dev test tests/montecarlo/test_shell_gos.py -q`.

## Author resolution

The author applied fixes 2–4 (doc numbers, closed-form $\delta_F$ test,
xfail wording, ledger threshold wording). For fix 1, the owner chose
Eq. 3.96: bound-shell close collisions start at $Q_k=U_k$, while distant
terms keep $Q'_k$. The implementation, the close-quadrature test and the
recorded ratios were updated. A new test checks that close moments vanish
continuously as $E\to U_k^+$ and that the mean close loss exceeds $U_k$. This
changes the verified close-channel lower limit, so a follow-up fresh check of
that channel is advisable. The ledger status stays `rederived` on the
strength of this record plus the updated tests.

## Eq. 3.96 re-check (2026-09-24)

Fresh-context targeted re-check of the close-channel change in commit
`a27044d9` (verified at `a85d836b`), requested in the author resolution above.
Scope: the close (Møller) channel of `shell_gos.py::shell_gos_moments` and the
re-pinned GOS/`stp.dat` ratios. The distant channels, $\delta_F$ and the Bethe
limit were not re-derived beyond the regression comparison below.

### Source

PENELOPE-2024 is internally inconsistent here. Eq. 3.56 and the text after it
("close interactions are allowed for energy transfers $W$ larger than $Q_k$;
for bound shells, we set $Q_k=U_k$"), Eq. 3.75 and Eq. 3.96 use
$\Theta(W-Q_k)$ with $Q_k=U_k$. Eqs. 3.86, 3.92 and 3.106 use $Q'_k$. Eq. 3.80
introduces $Q'_k$ inside §3.2.2.1, "Distant interactions with inner-shell
electrons", right after $W'_k$ is applied "in all formulas pertaining to the
distant excitations of inner shells". It is motivated by the threshold
behaviour of the distant peak. The section context supports, but does not by
itself prove, the owner choice of $U_k$ for close and $Q'_k$ for distant. The
note after Eq. 3.95 keeps $Q_{cb}=W_{cb}$ for the conduction band.

### Independent derivation

For a bound shell $k$ at $E>U_k$:

$$
\sigma^{(n)}_{{\rm clo},k}=\frac{2\pi e^4}{m_ev^2}f_k\int_{U_k}^{(E+U_k)/2}W^{n-2}F^{(-)}(E',W)\,dW,\qquad E'=E+U_k,
$$

$$
F^{(-)}(E',W)=1+\Big(\frac{W}{E'-W}\Big)^2-(1-a)\frac{W}{E'-W}+a\frac{W^2}{E'^2},\qquad a=\Big(\frac{E}{E+m_ec^2}\Big)^2 .
$$

The upper limit follows from taking the faster final electron as the
primary. The primary keeps $E-W$ and the knock-on has $W-U_k$, so
$E-W\ge W-U_k\iff W\le(E+U_k)/2$ (Eq. 3.88; sympy `solve` gives
$W=E/2+U/2$). The interval $[U_k,(E+U_k)/2]$ is non-empty exactly when
$E>U_k$, and its length $(E-U_k)/2\to0$ as $E\to U_k^+$. On it,
$W\ge U_k>0$ and $E'-W\ge E'/2$, so the integrand is bounded and the moments
vanish linearly in $E-U_k$. Since $W\ge U_k$, the knock-on energy is never
negative and $\langle W\rangle_{\rm clo}\ge U_k$. For the conduction band
($U=0$) the limits are $[W_{cb},E/2]$ with $E'=E$.

Relative to Eq. 3.106, the two limits coincide for $E>3W_k-2U_k$
($Q'_k=U_k$). Below that, $Q'_k=U_kE/(3W_k-2U_k)<U_k$, and Eq. 3.106 adds
$\int_{Q'_k}^{U_k}$. That extra piece never vanishes at threshold (Si K:
$Q'_k\to400$ eV as $E\to U_k$).

### Comparison with the code

- Lower limit: `q_close = u if u > 0.0 else q_mod`, and for the band
  `q_mod = w`. Upper limit `w_max = 0.5 * (energy_eV + u)`. `prime = energy_eV + u`.
  `a` is evaluated at $E$ (the prior owner decision). This matches the
  derivation exactly.
- `_moller_integrals`: sympy confirms
  $\frac{d}{dW}J_n=W^{n-2}F^{(-)}(E+U,W)$ exactly for $n=0,1,2$. The code's
  $J_2$ differs from Eq. 3.110 (with $E\to E'$) by the constant $E+U$, which
  cancels in the difference.
- Independent adaptive quadrature of every channel for the catalog Si, SiO₂
  and MoS₂ oscillators at 1, 2, 5, 10, 100 and 1000 keV agrees to
  $\le5.9\times10^{-9}$ relative, with an identical zero pattern.
- Si K near threshold ($U=1844$ eV, $3W_k-2U_k=8495$ eV):
  $\sigma^{(0)}_{\rm clo}=1.60\times10^{-21}$, $1.89\times10^{-22}$,
  $1.92\times10^{-23}$ and $1.93\times10^{-26}$ cm² at
  $E/U-1=10^{-1},10^{-2},10^{-3},10^{-6}$ (code identical), with
  $\langle W\rangle_{\rm clo}=1890\to1844$ eV. The $Q'_k$ variant stays at
  $\approx1.15\times10^{-19}$ cm², a finite jump.

### Pinned ratios

The quadrature, together with an independent $S$ from the mass-stopping column
and xraydb molar masses, reproduces the new pins in
`test_fetched_raw_to_sbethe_stopping_ratios_are_recorded` (1, 5, 10, 100 keV).
Switching only the close lower limit back to $Q'_k$ reproduces the old pins.
The pin change is therefore due to the lower limit alone:

| material | $U_k$ (independent) | new pin | $Q'_k$ (independent) | old pin |
| --- | --- | --- | --- | --- |
| Si | 1.1333, 1.0309, 1.0139, 1.0003 | 1.1333, 1.0309, 1.0139, 1.0003 | 1.1333, 1.0399, 1.0139, 1.0003 | 1.1333, 1.0398, 1.0139, 1.0003 |
| SiO₂ | 1.2143, 1.0636, 1.0320, 0.9997 | 1.2142, 1.0635, 1.0320, 0.9996 | 1.2606, 1.0724, 1.0350, 0.9997 | 1.2606, 1.0724, 1.0349, 0.9996 |
| MoS₂ | 1.1525, 1.1149, 1.0674, 1.0051 | 1.1525, 1.1150, 1.0675, 1.0052 | 1.1626, 1.1274, 1.0674, 1.0051 | 1.1627, 1.1274, 1.0675, 1.0052 |

The residual $\le1\times10^{-4}$ equals the xraydb-versus-SBETHE molar-mass
difference, well inside the test's absolute tolerance of $2\times10^{-4}$.
The ledger note's 1 keV excesses (13%, 21%, 15%) and 10 keV excesses (1.4%,
3.2%, 6.8%) match. The quadrature test and the new threshold test pass
(`pyrite-dev test tests/montecarlo/test_shell_gos.py`).

### Findings

1. **Code: matches** the Eq. 3.96 reading. No factor, limit or kinematic
   discrepancy.
2. **Stale documentation (low severity).** The ledger **Equation** field for
   `penelope-shell-gos-moments` (`ledger-transport-background.md:125`) still
   writes $\int_{Q'_k}^{(E+U_k)/2}$ for $\sigma_{\rm clo}$. The display equation in
   `penelope-shell-gos-moments.md:95` still writes
   $\int_{Q'_k}^{W_{\max}}W^{n-2}F^{(-)}(E,W)\,dW$, citing Eqs. 3.86/3.106. It
   uses $Q'_k$ and $F(E,W)$ instead of $Q_k=U_k$ (band: $W_{cb}$) and
   $F(E+U_k,W)$. Item 7 of that document records the decision, but its first
   sentence ("The code follows Eq. 3.106") now reads as current. Suggested
   fix (author): change both to $\int_{Q_k}^{(E+U_k)/2}W^{n-2}F^{(-)}(E+U_k,W)\,dW$
   with $Q_k=U_k$ ($W_{cb}$ for the band), citing Eqs. 3.56 and 3.96, and put
   the item 7 history in the past tense.
3. **Verdict:** `rederived` stands for the updated close channel. Only a human
   may mark the row `signed-off`.
