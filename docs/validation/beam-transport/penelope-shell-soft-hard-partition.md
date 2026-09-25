# PENELOPE shell GOS soft/hard partition

Validation: `penelope-shell-soft-hard-partition`. Status: rederived. The
partitioned moments and their hard-event sampler are host-side only. No
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
\text{soft}_k=s_k\,\sigma^{(n)}_k\big|_{(0,W_c]},\qquad
\text{hard}_k=s_k\,\sigma^{(n)}_k\big|_{(W_c,\infty)}
\qquad(\text{every }k),
$$

For every $W_c$, soft plus hard equals the closed moments, and
$\sigma_s^{(1)}+\sigma_h^{(1)}$ equals the adopted `stp.dat` stopping.
For an inner shell, the explicit vacancy rate is its hard zeroth moment.
It equals the EEDL-substituted rate when $W_c\le U_i$ and decreases when the
cutoff condenses losses above $U_i$.

## Interpretation choices

1. **Inner-shell cutoff.** PENELOPE's soft DCS (Eq. 4.113) includes inner
   shells whose $U_i<W_{cc}$. §3.2.6.1 creates explicit vacancies only in
   hard collisions, so condensed inner-shell losses produce no vacancy. The
   full EEDL-substituted vacancy rate is recovered when $W_c\le\min_iU_i$:
   104 eV for Si and SiO₂ (Si L2/L3), 68 eV for MoS₂ (Mo N1).
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
- $W_c=0$ leaves no soft moment; $W_c\to\infty$ leaves no hard event or
  explicit vacancy.
- The hard rate is non-increasing, and soft stopping and straggling are
  non-decreasing in $W_c$.
- Catalog Si, SiO₂ and MoS₂ at 1, 10 and 100 keV with $W_c=10$, 50 and
  1000 eV reproduce `stp.dat` ($10^{-12}$). At each cutoff, explicit and
  condensed inner rates sum to the EEDL-substituted rates.
- Tests: `tests/montecarlo/test_shell_partition.py`.

## Diagnostics

Soft share of stopping and hard-event mean free path (Å) at $W_c=50$ eV,
from `catalog_shell_partition` and the Eq. 3.51 formula-unit density:

| material | $E$ | soft share, hard mean free path |
| --- | ---: | --- |
| silicon | 1 keV | 0.486, 201 |
| silicon | 10 keV | 0.349, 1045 |
| silicon | 100 keV | 0.267, 6149 |
| sio2 | 1 keV | 0.611, 241 |
| sio2 | 10 keV | 0.432, 1414 |
| sio2 | 100 keV | 0.338, 9148 |
| mos2 | 1 keV | 0.515, 105 |
| mos2 | 10 keV | 0.354, 555 |
| mos2 | 100 keV | 0.285, 3196 |

$W_c=10$ eV lies below every catalog loss (the lowest is $W_{cb}\ge16.7$ eV),
so it is purely detailed: soft share 0 and hard mean free path equal to the
closed IMFP. Above the binding energy of an inner shell, its soft losses
reduce its explicit vacancy rate.

## Not covered

Soft angular deflection (Eq. 4.113), positrons, and transport integration.
The partition inherits the rate closure's unvalidated IMFP and straggling.

## Independent verification (fresh context, 2026-09-24)

Verified at commit `2158f43d` on branch `issue-93-soft-hard-inelastic-transport`
in a context that did not write the implementation. §§V1–V2 were written from
PENELOPE-2024 (NEA/MBDAV/R(2024)1), the ledger row and the function
signatures before the implementation bodies and the sections above were read.
The implementer's text above is unchanged.

### V1. Independent derivation

**Per-oscillator DCS.** For oscillator $k$ at kinetic energy $E$, Eqs.
3.94–3.96 give

$$
\frac{d\sigma_k}{dW}
=\underbrace{\big(D_{k,l}+D_{k,t}\big)}_{W\text{-independent}}\frac{p_{{\rm dis},k}(W)}{W}
+\frac{2\pi e^4}{m_ev^2}\,f_k\,\frac{F^{(-)}(E+U_k,W)}{W^2}\,
\Theta(W-Q_k)\,\Theta(W_{\max}-W),
$$

with $p_{\rm dis}=2(W_{\rm dis}-W)/(W_{\rm dis}-U_k)^2$ on $[U_k,W_{\rm dis})$
(Eq. 3.76), $W_{\rm dis}=3W'_k-2U_k$ (Eqs. 3.77–3.79), $W_{\max}=(E+U_k)/2$
(Eq. 3.88), and $p_{\rm dis}\to\delta(W-W_{cb})$ for the conduction band.
The longitudinal factor
$D_{k,l}\propto f_k\ln[Q'_k(Q_-+2m_ec^2)/(Q_-(Q'_k+2m_ec^2))]$ and the
transverse factor $D_{k,t}\propto f_k[\ln\gamma^2-\beta^2-\delta_F]$ do not
depend on the sampled $W$. $Q'_k$ is Eq. 3.80 and $Q_-$ is evaluated at the
modified resonance $W'_k$ (Eq. 3.83), not at $W$. A loss window therefore
restricts only the one-dimensional loss integral of each channel.

**Closure.** Eq. 3.142 multiplies each oscillator's total cross section by a
$W$-independent factor $s_k$ "without altering details of the PDFs". So the
closed DCS is $s_k\,d\sigma_k/dW$.

**Partition.** Eqs. 4.44, 4.46, 4.47 and 3.124 give, per formula unit,

$$
\sigma_s^{(n)}=\sum_k s_k\int_0^{W_{cc}}W^n\frac{d\sigma_k}{dW}dW,\qquad
\sigma_h^{(n)}=\sum_k s_k\int_{W_{cc}}^{W_{\max,k}}W^n\frac{d\sigma_k}{dW}dW .
$$

Additivity of the integral over $(0,W_{cc}]\cup(W_{cc},W_{\max}]$ gives
$\sigma_s^{(n)}+\sigma_h^{(n)}=\sum_ks_k\sigma_k^{(n)}$ for each $k$, $n$ and
$W_{cc}$. With the rate closure $\sum_ks_k\sigma^{(1)}_k=S_{\rm stp}/\mathcal N_{\rm fu}$,
we have $\mathcal N_{\rm fu}(\sigma_s^{(1)}+\sigma_h^{(1)})=S_{\rm stp}$, the
corrected `stp.dat` stopping. Here $\mathcal N_{\rm fu}$ is the formula-unit
density.

**Units.** $2\pi e^4/(m_ev^2)=2\pi r_e^2m_ec^2/\beta^2$ has units of
eV cm²; $f_k$ is dimensionless; $p_{\rm dis}$ is in eV⁻¹. So $\sigma^{(n)}$ is in
cm² eV$^n$, and $s_k$ is dimensionless.

**Inner shells.** Inner-shell support is
$[U_i,W_{\rm dis}]\cup[Q_i,W_{\max}]$ with $Q_i=U_i$ (Eq. 3.96, adopted
upstream). If $W_{cc}\le U_i$, the hard window contains the whole support:
$\sigma_{h,i}^{(0)}=s_i\sigma_i^{(0)}=\rho_i\sigma_{{\rm si},i}$, the
EEDL-substituted rate. If $W_{cc}>U_i$, the integrand is strictly positive on
$(U_i,\min(W_{cc},W_{\max}))$. This follows because $F^{(-)}>0$: its quadratic
in $r=W/(E'-W)$ has discriminant $(1-a)^2-4<0$. Also
$p_{\rm dis}>0$ below $W_{\rm dis}$. So the hard rate is strictly smaller and
non-increasing in $W_{cc}$. §3.2.6.1 says "Hard inelastic collisions with
inner shells are assumed to ionize the target atom", and Eq. 4.113 includes
in the soft DCS every inner shell with $U_k<W_{cc}$. Together these give
explicit vacancy rate $=\sigma^{(0)}_{h,i}$; the condensed part
$s_i\int_0^{W_{cc}}d\sigma_i$ creates no vacancy.

**Limits.** If $W_{cc}\ge\max_kW_{\max,k}$, no hard support remains, so there
are no hard events and no explicit vacancies. If $W_{cc}=0$, the soft moments
vanish. The window $(0,\infty)$ gives the unrestricted moments.

**Sampler interval (derived).** For an inner shell above its edge, the hard
conditional densities are $p_{\rm dis}(W)/W$ on
$(\max(U_i,W_{cc}),\min(W_{\rm dis},W_{\max}))$ and $F^{(-)}/W^2$ on
$(\max(U_i,W_{cc}),W_{\max})$.

### V2. Source conflicts and conventions

- **Boundary at $W_{cc}$.** Eq. 4.49 defines soft as $\Theta(W_{cc}-W)$, and
  Eq. 4.113 sums oscillators with $W_k<W_{cc}$. So PENELOPE puts a δ-loss
  exactly at $W_{cc}$ in the hard class. The implementation puts it in the soft
  class ($W\le W_c$), as a declared assumption. This affects only the
  conduction-band δ when $W_{cb}=W_c$ exactly, which is a measure-zero choice
  of $W_c$. It is not a discrepancy; it is a documented deviation from
  PENELOPE's strict inequality.
- **Eq. 3.125 versus Eq. 3.94.** This conflict belongs to
  `penelope-shell-hard-loss-sampling`, not to this claim. The partition
  moments use $p_{\rm dis}/W$ (Eq. 3.94/3.104) consistently.
- **§3.2.5.4 versus §3.2.6.1.** §3.2.5.4 says that PENELOPE deposits $U_i$
  locally and does not follow relaxation after hard GOS collisions. §3.2.6.1,
  which describes the 2014+ scheme, sends hard inner-shell collisions to
  RELAX. The vacancy rule verified here follows §3.2.6.1.

### V3. Source-to-code comparison

`windowed_shell_gos_moments` implements V1 exactly. The triangle is integrated
over $[\max(U,W_a),\min(W_{\rm dis},W_{\max},W_b)]$. The δ is included when
$W_a<W_{cb}\le W_b$ and $W_{cb}<W_{\max}$. The Møller term is integrated over
$[\max(Q_k,W_a),\min(W_{\max},W_b)]$. The distant log factors do not depend on
the window. `partition_shell_rates` multiplies both windows by
`closure.scale` for every oscillator, inner or outer, which matches
$s_k$ in V1. `_loss_bounds` in `shell_sampling.py` returns lower bound
$\max(U,W_c)$ for bound distant and close branches, and $\max(W_{cb},W_c)$ for
band close. Its upper bounds are $\min(W_{\rm dis},W_{\max})$ and $W_{\max}$,
and its near-threshold $W_{\rm dis}=E$ equals $3W'_k-2U_k$ with
$W'_k=(E+2U_k)/3$. This matches the derived sampler interval.

### V4. Independent numeric checks

The checks used an independent DCS from Eqs. 3.76–3.80, 3.83, 3.85, 3.87–3.88
and 3.94–3.96, integrated with `scipy.integrate.quad`. They used only upstream
inputs from the code: oscillators, `closure.scale`, $\delta_F$, and the
`stp.dat` interpolation. The inputs covered catalog Si, SiO₂ and MoS₂ at
$E=1$, 10 and 100 keV.

| check | cutoffs $W_c$ (eV) | result |
| --- | --- | --- |
| windowed moments vs quadrature, every oscillator, channel, $n=0,1,2$, both windows | 16.7, 23, 50, 60, 104, 154, 200, 1000 | worst relative difference $1.6\times10^{-9}$ (quadrature-limited) |
| $(\sigma_s^{(1)}+\sigma_h^{(1)})/S_{\rm stp}-1$ | 0, 16.7, 50, 60, 68, 104, 154, 200, 1000, $10^9$ | $\le4.4\times10^{-16}$ in all 90 cases |
| soft/hard $\sigma^{(0,1)}$ vs quadrature × $s_k$ | same | $\le1.6\times10^{-9}$ |
| explicit inner rate $=\rho_i\sigma_{{\rm si},i}$ for $W_c\le U_i$ | same | exact ($10^{-12}$) in all cases |
| explicit inner rate $<\rho_i\sigma_{{\rm si},i}$ for $W_c>U_i$ | same | all true; Si at 10 keV, $W_c=154$: L2/L3 at 0.627; $W_c=200$: L2/L3 0.42, L1 0.74; $W_c=1000$: 0.022/0.022/0.035 |
| explicit + condensed inner $=\rho_i\sigma_{{\rm si},i}$ | same | $10^{-12}$ in all cases |
| $W_c=10^9$ (and $W_c=1000$ at 1 keV) | — | $\sigma_h^{(0)}=0$ and vacancy rates 0 |
| δ at $W_c=W_{cb}$ (Si, 16.7 eV) | $W_{cb}^-$, $W_{cb}$, $W_{cb}^+$ | hard, soft, soft |

For the edge-straddling cutoffs, Si at 60 eV lies below every inner edge. At
104 eV, it lies at L2/L3, and the hard rates stay full. At 200 eV, it lies
above L1, L2 and L3, and their explicit rates fall to the fractions above.

**Sampler versus partition, inner shells above the edge.** The check drove
`sample_shell_hard_loss` with the channel uniform at the centre of each inner
channel's probability interval and $N$ midpoint loss uniforms. It compared
sampled $\langle W\rangle$ and $\langle W^2\rangle$ with
$\sigma^{(1)}_h/\sigma^{(0)}_h$ and $\sigma^{(2)}_h/\sigma^{(0)}_h$ from the
partition. Cases were Si at 3 and 10 keV with $W_c=200$ eV, MoS₂ at 10 keV
with $W_c=200$ eV (Mo N1, S L2/L3), and SiO₂ at 10 keV with $W_c=600$ eV
(Si L-shells, O K).

- With $N=4000$, every distant and close case agrees to $\le5.2\times10^{-6}$
  relative.
- For O K at $N=16000$, the distant residual falls from $-1.1\times10^{-6}$ to
  $-1.4\times10^{-7}$. The close residual falls from $-6.5\times10^{-8}$ to
  $-4.1\times10^{-9}$. These are the discretisation rates, so the residual is
  midpoint-rule error, not a distribution mismatch.
- Every sample has $W>W_c$, $W(u=0)=W_c$, and an inner vacancy label.

The sampler and the partition describe one distribution.

**Diagnostics.** The table above at $W_c=50$ eV was recomputed and reproduces
all nine rows. The existing test file passes: 100 tests.

### V5. Findings and verdict

- **Filters:** units pass; limits pass ($W_c=0$, $W_c\to\infty$, full
  window, $W_c\le U_i$); signs pass (non-negative integrands, monotone in
  $W_c$).
- **Re-derivation:** matches.
- **Documentation drift (non-physics):** the `ShellSoftHardPartition` class
  docstring in `shell_partition.py` still says "Inner shells
  (`closure.inner`) are hard at every `W_c`", which contradicts the
  implementation after `2158f43d`. It should be corrected in a code-owning
  context.
- **Convention note:** the boundary δ-loss convention differs from PENELOPE's
  strict inequality (V2). It is declared and measure-zero.
- **Verdict:** `rederived`. Only a human may mark `signed-off`.
