# PENELOPE shell GOS soft/hard partition

Validation: `penelope-shell-soft-hard-partition`. Status: unverified. The
partitioned moments are host-side only. Nothing samples them, and no
transport mode uses them.

## Source and intended quantity

PENELOPE-2024 mixed (class II) simulation treats inelastic collisions with
energy loss above a cutoff $W_{cc}$ as discrete hard events and the rest as
a continuous soft energy loss with straggling (NEA/MBDAV/R(2024)1, §3.2.5 and
Chapter 4). Per atom or formula unit, with $d\sigma_{\rm in}/dW$ the closed shell
GOS DCS of
[the rate closure](penelope-shell-rate-closure.md):

$$
\sigma(W_{cc})=\int_{W_{cc}}^{W_{\max}}\frac{d\sigma_{\rm in}}{dW}\,dW
=\sum_k\sigma_k(W_{cc})
\qquad(\text{Eqs. 3.124, 4.44}),
$$

$$
\sigma_s^{(n)}=\int_0^{W_{cc}}W^n\frac{d\sigma_{\rm in}}{dW}\,dW,\quad n=1,2
\qquad(\text{Eqs. 4.46, 4.47}).
$$

Hard events select oscillator $k$ with probability
$p_k=\sigma_k(W_{cc})/\sigma(W_{cc})$ (Eq. 3.124), then a branch in
proportion to its restricted cross section. The code writes $W_c$ for
$W_{cc}$.

## Restricted moments

Each oscillator channel of the unrestricted moments
([shell GOS moments](penelope-shell-gos-moments.md)) is a $W$-independent
factor times a one-dimensional loss integral. The distant factor
$f_k\{\ln[Q'_k(Q_-+2mc^2)/(Q_-(Q'_k+2mc^2))]\}$ or
$f_k[\ln\gamma^2-\beta^2-\delta_F]_+$ uses the (modified) resonance
$W'_k$ through $Q_-$, not the sampled $W$. So a loss window
$(W_a,W_b]$ restricts only the loss integral:

- bound-shell distant, Eq. 3.76 triangle:
  $\int_{\max(U_k,W_a)}^{\min(W_{\rm dis},W_{\max},W_b)}W^{n-1}p_{\rm dis}(W)\,dW$
  with $p_{\rm dis}=2(W_{\rm dis}-W)/(W_{\rm dis}-U_k)^2$;
- conduction-band distant, $\delta(W-W_{cb})$: $W_{cb}^{n-1}$ when
  $W_a<W_{cb}\le W_b$ and $W_{cb}<W_{\max}$, else 0;
- close Møller, Eqs. 3.87, 3.96, 3.106–3.110:
  $\int_{\max(Q_k,W_a)}^{\min((E+U_k)/2,W_b)}W^{n-2}F^{(-)}(E+U_k,W)\,dW$,
  $Q_k=U_k$ ($W_{cb}$ for the band).

The triangle integrals are closed-form:
$\int_a^bW^{-1}(W_{\rm dis}-W)dW=W_{\rm dis}\ln(b/a)-(b-a)$,
$\int_a^b(W_{\rm dis}-W)dW=W_{\rm dis}(b-a)-(b^2-a^2)/2$,
$\int_a^bW(W_{\rm dis}-W)dW=W_{\rm dis}(b^2-a^2)/2-(b^3-a^3)/3$. The Møller
integrals are differences of the same antiderivatives $J_n$ used for the
full range. `windowed_shell_gos_moments(material, E, W_a, W_b)` returns these
windowed moments; `shell_gos_moments` is the window $(0,\infty)$ and is bitwise
unchanged by the refactor.

## Partition of the closed moments

The rate closure multiplies every channel of oscillator $k$ by one factor
$s_k$: $\rho_i\sigma_{{\rm si},i}/\sigma_i^{(0)}$ for inner shells, $\mathcal
N(E)$ otherwise. It keeps each oscillator's loss PDF. The partition is
therefore

$$
\text{soft}_k=\mathcal N\,\sigma^{(n)}_k\big|_{(0,W_c]},\qquad
\text{hard}_k=\mathcal N\,\sigma^{(n)}_k\big|_{(W_c,\infty)}
\qquad(\text{outer }k),
$$

$$
\text{soft}_i=0,\qquad\text{hard}_i=s_i\sigma^{(n)}_i\big|_{(0,\infty)}
\qquad(\text{inner }i).
$$

For every $W_c$, soft plus hard equals the closed moments, and
$\sigma_s^{(1)}+\sigma_h^{(1)}$ equals the adopted `stp.dat` stopping.

## Interpretation choices

1. **Inner shells always hard (deviation).** PENELOPE's soft DCS
   (Eq. 4.113) includes inner shells whose $U_i<W_{cc}$. §3.2.6.1 creates
   vacancies only in hard collisions ("Hard inelastic collisions with inner
   shells are assumed to ionize the target atom"), so those soft ionisations
   create no vacancy. Here every inner-shell loss is a hard event, with the
   full closed GOS loss PDF ($W\ge U_i$). The hard inner rate is then
   $\rho_i\sigma_{{\rm si},i}$, the substituted EEDL vacancy rate, for any
   $W_c$. The two rules agree when $W_c\le\min_iU_i$: 104 eV for Si and
   SiO₂ (Si L2/L3), 68 eV for MoS₂ (Mo N1).
2. **Boundary.** A loss exactly at $W_c$ is soft ($W>W_c$ is hard). This only
   matters for the conduction-band delta; it matches the OOS-bin partition
   in `inelastic.py`.
3. **Branch probabilities.** `hard_channel_probabilities` gives the joint
   (oscillator, branch) point probabilities, the product of Eq. 3.124's
   $p_k$ and the branch selection of §3.2.5.

## Limits and checks

- $(0,\infty)$ is bitwise `shell_gos_moments`.
- $(0,W_c]$ plus $(W_c,\infty)$ equals the full moments ($10^{-12}$) for
  cutoffs below, at and above every resonance, edge and $W_{\max}$.
- Windowed triangle and Møller moments match direct quadrature
  ($10^{-10}$, $10^{-9}$).
- $W_c=0$ leaves no soft moment; $W_c\to\infty$ leaves only inner-shell
  hard events, whose rate equals $\sum_i\rho_i\sigma_{{\rm si},i}$.
- The hard rate is non-increasing, and soft stopping and straggling are
  non-decreasing in $W_c$.
- Catalog Si, SiO₂ and MoS₂ at 1, 10 and 100 keV with $W_c=10$, 50 and
  1000 eV reproduce `stp.dat` ($10^{-12}$) and the EEDL vacancy rates.
- Tests: `tests/montecarlo/test_shell_partition.py`.

## Diagnostics

Soft share of stopping and hard-event mean free path (Å), from
`catalog_shell_partition` and the Eq. 3.51 formula-unit density:

| material | $E$ | $W_c=50$ eV | $W_c=100$ eV | $W_c=1$ keV |
| --- | ---: | --- | --- | --- |
| silicon | 1 keV | 0.486, 201 | 0.534, 263 | 0.632, 338 |
| silicon | 10 keV | 0.349, 1045 | 0.376, 1242 | 0.463, 1491 |
| silicon | 100 keV | 0.267, 6149 | 0.284, 6941 | 0.340, 7848 |
| sio2 | 1 keV | 0.611, 241 | 0.677, 397 | 0.812, 744 |
| sio2 | 10 keV | 0.432, 1414 | 0.468, 1975 | 0.583, 3042 |
| sio2 | 100 keV | 0.338, 9148 | 0.360, 11791 | 0.435, 15926 |
| mos2 | 1 keV | 0.515, 105 | 0.651, 237 | 0.820, 578 |
| mos2 | 10 keV | 0.354, 555 | 0.445, 1051 | 0.583, 1817 |
| mos2 | 100 keV | 0.285, 3196 | 0.356, 5835 | 0.450, 8994 |

$W_c=10$ eV lies below every catalog loss (the lowest is $W_{cb}\ge16.7$ eV),
so it is purely detailed: soft share 0 and hard mean free path equal to the
closed IMFP. Even at $W_c=1$ keV, inner-shell events and the Møller tail keep
18–66% of the stopping hard.

## Not covered

Hard-event sampling of $W$ (Eq. 3.125, the Møller branch), recoil and
secondary kinematics, soft angular deflection (Eq. 4.113), positrons,
and any transport integration. The partition inherits the rate closure's
unvalidated IMFP and straggling.
