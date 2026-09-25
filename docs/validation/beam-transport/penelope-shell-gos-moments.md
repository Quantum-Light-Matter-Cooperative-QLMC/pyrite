# PENELOPE shell GOS moments

Validation: `penelope-shell-gos-moments`. Status: unverified. These are raw,
uncalibrated host-side moments. Nothing samples them, and no transport mode
uses them.

## Source and intended quantity

PENELOPE-2024 §§3.2.2–3.2.4 (NEA/MBDAV/R(2024)1, pp. 119–131) define the
electron inelastic DCS of the δ-oscillator GOS. The inputs are the oscillators
$(f_k,U_k,W_k)$ from
[`penelope-shell-oscillators`](penelope-shell-oscillators.md). The code
evaluates the integrated moments

$$
\sigma^{(n)}_{\rm in}=\int_0^{W_{\max}}W^n\frac{d\sigma_{\rm in}}{dW}\,dW
=\sigma^{(n)}_{\rm dis,l}+\sigma^{(n)}_{\rm dis,t}+\sigma^{(n)}_{\rm clo}
\qquad(\text{Eqs. 3.99, 3.103})
$$

for $n=0,1,2$ per formula unit and per oscillator. $N\sigma^{(0)}$,
$N\sigma^{(1)}$, and $N\sigma^{(2)}$ are the inverse IMFP, stopping power,
and straggling parameter (Eqs. 3.100–3.102). The code uses
$\mathcal P=2\pi e^4/(m_ev^2)=2\pi r_e^2m_ec^2/\beta^2$ and SI constants from
`scipy.constants`.

### Distant interactions

With the modified resonance $W'_k$ and cutoff $Q'_k$ (below),

$$
\sigma^{(n)}_{\rm dis,l}=\mathcal P\sum_k f_k
\ln\!\left[\frac{Q'_k}{Q_-}\,\frac{Q_-+2m_ec^2}{Q'_k+2m_ec^2}\right]
\int_0^{W_{\max}}W^{n-1}p_{\rm dis}(W)\,dW\qquad(\text{Eq. 3.104}),
$$

$$
\sigma^{(n)}_{\rm dis,t}=\mathcal P\sum_k f_k
\left[\ln\frac{1}{1-\beta^2}-\beta^2-\delta_F\right]
\int_0^{W_{\max}}W^{n-1}p_{\rm dis}(W)\,dW\qquad(\text{Eq. 3.105}).
$$

Both require $Q_-<Q'_k$ (the $\Theta(Q'_k-Q_-)$ of Eqs. 3.94–3.95). $Q_-$ is
the zero-angle recoil energy for $W'_k$ (Eq. 3.83, equivalently App. Eq. A.31).
It is computed with the cancellation-free form already validated as
`gos-distant-response` (`inelastic.py::_qmin_ev`).

A bound shell ($U_k>0$) uses the triangle distribution (Eq. 3.76)

$$
p_{\rm dis}(W)=\frac{2(W_{\rm dis}-W)}{(W_{\rm dis}-U_k)^2},
\qquad U_k\le W<W_{\rm dis},\qquad W_{\rm dis}=3W'_k-2U_k
\quad(\text{Eqs. 3.77, 3.79}),
$$

with

$$
W'_k=\begin{cases}W_k & E>3W_k-2U_k\\ (E+2U_k)/3 & \text{otherwise}\end{cases},
\qquad
Q'_k=\begin{cases}U_k & E>3W_k-2U_k\\ U_kE/(3W_k-2U_k) & \text{otherwise}\end{cases}
\quad(\text{Eqs. 3.78, 3.80}).
$$

With $b=\min(W_{\rm dis},W_{\max})$, the exact truncated moments are

$$
\int W^{-1}p\,dW=\tfrac{2}{(W_{\rm dis}-U)^2}\big[W_{\rm dis}\ln\tfrac{b}{U}-(b-U)\big],\quad
\int p\,dW=\tfrac{2}{(W_{\rm dis}-U)^2}\big[W_{\rm dis}(b-U)-\tfrac{b^2-U^2}{2}\big],
$$

$$
\int W\,p\,dW=\tfrac{2}{(W_{\rm dis}-U)^2}\big[\tfrac{W_{\rm dis}(b^2-U^2)}{2}-\tfrac{b^3-U^3}{3}\big].
$$

For $b=W_{\rm dis}$ these are $1$ and $W'_k$ for $n=1,2$. The conduction band
uses $\delta(W-W_{cb})$ and $Q_{cb}=W_{cb}$ (text after Eq. 3.95). It
contributes $W_{cb}^{n-1}$ when $W_{cb}<W_{\max}$.

The density-effect correction uses the unmodified oscillator OOS (Eq. 3.59).
$L^2$ is the root of

$$
\frac{\Omega_p^2}{Z}\sum_k\frac{f_k}{W_k^2+L^2}=1-\beta^2,
\qquad
\delta_F=\frac1Z\sum_kf_k\ln\!\left(1+\frac{L^2}{W_k^2}\right)-\frac{L^2}{\Omega_p^2}(1-\beta^2)
\quad(\text{Eqs. 3.70–3.72}),
$$

and $\delta_F=0$ when $1-\beta^2\ge F(0)$.

### Close interactions

$$
\sigma^{(n)}_{\rm clo}=\mathcal P\sum_kf_k\int_{Q_k}^{(E+U_k)/2}W^{n-2}F^{(-)}(E+U_k,W)\,dW
\qquad(\text{Eqs. 3.56, 3.86, 3.96, 3.106}),
$$

$$
F^{(-)}=1+\left(\frac{W}{E'-W}\right)^2-\frac{W}{E'-W}
+a\left(\frac{W}{E'-W}+\frac{W^2}{E'^2}\right),\qquad
a=\left(\frac{E}{E+m_ec^2}\right)^2\qquad(\text{Eqs. 3.85, 3.87}),
$$

with $E'=E+U_k$ and $W_{\max}=E'/2=(E+U_k)/2$ (Eq. 3.88). The antiderivatives
are Eqs. 3.107–3.110 with $E\to E'$. Re-deriving them gives the manual's $J_0$
and $J_1$ exactly. The re-derived $J_2$ differs from Eq. 3.110 by the constant
$-E'$, which cancels in the definite integral. The code evaluates this
$J_2$:

$$
J_2=(3-a)W+\frac{E'^2}{E'-W}+(3-a)E'\ln(E'-W)+\frac{aW^3}{3E'^2}.
$$

## Interpretation choices

1. **Eq. 3.81 versus Eqs. 3.94/3.104.** The double-differential inner-shell
   DCS (Eq. 3.81) prints $p_{\rm dis}(W)/W_k$. The integrated DCS (Eq. 3.94)
   and moments (Eq. 3.104) use $p_{\rm dis}(W)/W$. The code follows the
   integrated equations, as the slice requires. The two forms give the same
   $\sigma^{(1)}$ for an untruncated triangle, but different $\sigma^{(0)}$
   and $\sigma^{(2)}$. Measured with the Eq. 3.81 form on the three catalog
   materials, the total $\sigma^{(0)}$ changes by $-0.2\%$ to $-1.4\%$,
   $\sigma^{(1)}$ by $\le0.7\%$ (from $W_{\max}$ truncation below 100 keV), and
   $\sigma^{(2)}$ by $-0.5\%$ to $+6.1\%$ (dense-grid extremes: $\sigma^{(0)}$
   at 10 MeV, $\sigma^{(2)}$ in Si at 1.24 keV). Owner decision
   (2026-09-24): keep Eq. 3.104. It is the equation the manual gives for the
   integrated moments, and replacing $\delta(W-W_k)$ by $p_{\rm dis}(W)$ in
   a GOS proportional to $f(W)/W$ gives $p_{\rm dis}(W)/W$. PENELOPE's
   source code was not available to check which form it implements.
2. **Which shells are broadened.** §3.2.2.1 describes "inner shells". §3.2.3
   says all terms are written as inner shells, except that the conduction band
   uses a δ resonance. The code broadens every $U_k>0$ shell. The EABS-based
   inner/outer split (p. 118) concerns ionisation substitution (§3.2.6), so it
   is not used here.
3. **$W_{\max}$ bounds distant losses.** Eq. 3.88 applies to "collisions
   (close and distant)", and Eq. 3.104 integrates to $W_{\max}$. The triangle is
   therefore truncated when $W_{\rm dis}>(E+U_k)/2$. This happens near
   threshold and for $3W_k-2U_k<E<6W_k-5U_k$. There, $\langle W\rangle<W'_k$.
4. **$a$ in $F^{(-)}$.** Eq. 3.87 primes $E$ only in the ratio terms, so $a$
   is evaluated at $E$. The $a$ terms are $O(E/m_ec^2)^2$ at keV energies.
5. **Transverse clipping.** When $F(0)>1$, $\delta_F>0$ at every energy.
   The free-electron Eq. 3.62 band alone gives $F(0)\ge1$, and SiO₂'s
   measured $W_{cb}=22$ eV lies just below its Eq. 3.62 value. Below about
   4.67 keV in SiO₂, $\delta_F$ (up to $1.64\times10^{-4}$) exceeds the
   Eq. 3.68 bracket $\ln\gamma^2-\beta^2\approx\beta^4/2$, so the
   transverse DCS would be negative. The manual does not address this. The
   code clips the bracket at zero. The clip changes the moments by at most
   $5.9\times10^{-6}$ relative. Si and MoS₂ are not clipped.
6. **Formula-unit density.** $N$ is recovered from $\Omega_p$ with Eq. 3.51,
   $NZ=\epsilon_0m_e\omega_p^2/e^2$, so it shares $\Omega_p$ with the
   oscillators. It agrees with $\rho N_A/M$ from the SBETHE tables within
   $6\times10^{-5}$.
7. **Close-collision lower limit (Eq. 3.96 versus Eq. 3.106).** Eq. 3.96
   prints $\Theta(W-Q_k)$, with $Q_k=U_k$ (Eq. 3.56); Eqs. 3.86, 3.92 and
   3.106 use the modified $Q'_k$ (Eq. 3.80). The code first followed Eq. 3.106.
   Below $E=3W_k-2U_k$, $Q'_k<U_k$, so this admits close losses
   $Q'_k\le W<U_k$, whose knock-on energy $W-U_k$ would be negative (Si K
   near threshold: $Q'_k\approx400$ eV, $U_k=1844$ eV). With $U_k$ as the
   lower limit, $\sigma^{(1)}$ changes by up to $-3.7\%$ and $\sigma^{(2)}$
   by up to $-14\%$ above 1 keV, and by up to $-32\%$ and $-55\%$ near the
   L edges (fresh-context verification). **Owner decision (2026-09-24):
   follow Eq. 3.96.** Bound-shell close collisions start at $Q_k=U_k$, so
   no close loss leaves a negative knock-on energy, and the close moments
   vanish continuously as $E\to U_k^+$. The distant terms keep $Q'_k$, as
   in Eqs. 3.94 and 3.104. The conduction band keeps $Q_{cb}=W_{cb}$. The
   comparison table below uses this cutoff; it lowers raw stopping only
   near inner-shell thresholds (Si 2–5 keV, SiO₂ 1–10 keV, MoS₂ 1, 5 and
   50 keV).

## Limiting cases

- **Bethe limit (Eqs. 3.115–3.121).** For $E\gg U_k$, the total
  $\sigma^{(1)}$ reaches $\mathcal PZ[\ln(E^2(\gamma+1)/2I^2)+f^{(-)}(\gamma)-\delta_F]$
  within $10^{-3}$ at 100 keV and $4.2\times10^{-5}$ above 1 MeV for all
  three materials. The Si fixture test uses $2\times10^{-4}$ at 1 MeV–1 GeV.
- **Ultrarelativistic $\delta_F$ (Eq. 3.73).** At 10 GeV,
  $\delta_F=\ln[\Omega_p^2/((1-\beta^2)I^2)]-1$ to $10^{-3}$.
- **Mean loss of an untruncated triangle (Eq. 3.77).**
  $\sigma^{(2)}_{\rm dis}=W_k\sigma^{(1)}_{\rm dis}$ to $10^{-12}$.
- **One-shell atom (Eqs. 3.60–3.61).** $W=I$. At 50 keV, the distant stopping
  equals $\mathcal P\{\ln[\ldots]+\ln\gamma^2-\beta^2\}$ from independently
  computed $Q_-$ to $10^{-6}$.
- **Close quadrature.** The analytic close moments match direct quadrature of
  Eq. 3.87 with $E'=E+U_k$ to $10^{-9}$ at 2, 20, and 300 keV.
- **Threshold.** A shell with $E\le U_k$ contributes zero. Just above
  $E=U_k$ the close term now starts at $U_k$ (choice 7), so it vanishes as
  $E\to U_k^+$. The Eq. 3.106 cutoff $Q'_k$ produced a finite jump there (Si K:
  $1.8\times10^{-3}$ of total $\sigma^{(0)}$). The moments are
  continuous across $E=3W_k-2U_k$ to $10^{-6}$, as intended by Eq. 3.78.
- **Conduction band.** It contributes nothing below $E=2W_{cb}$. Above that,
  it dominates $\sigma^{(0)}$, with more than half of it at 10 keV in Si.
  Moving $W_{cb}$ from 14 to 20 eV changes $\sigma^{(1)}$ by less than 1% but
  $\sigma^{(0)}$ by more than 5% (manual, Fig. 3.11 discussion).

## Comparison with corrected SBETHE `stp.dat`

The corrected SBETHE stopping (`stopping_cs_eV_cm2`, catalog tables) compared
with raw GOS $\sigma^{(1)}$ (eV cm² per formula unit). Nothing was tuned.

| material | E (keV) | GOS $\sigma^{(1)}$ | `stp.dat` | GOS/stp | GOS/Bethe | Bethe/no-shell | no-shell/stp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silicon | 1 | 3.842e-15 | 3.390e-15 | 1.1333 | 1.1016 | 1.0000 | 1.0288 |
| silicon | 2 | 2.471e-15 | 2.303e-15 | 1.0728 | 1.0368 | 1.0000 | 1.0347 |
| silicon | 5 | 1.307e-15 | 1.268e-15 | 1.0309 | 1.0057 | 1.0000 | 1.0251 |
| silicon | 10 | 7.872e-16 | 7.764e-16 | 1.0139 | 0.9990 | 1.0000 | 1.0149 |
| silicon | 20 | 4.716e-16 | 4.676e-16 | 1.0086 | 1.0013 | 0.9999 | 1.0074 |
| silicon | 50 | 2.415e-16 | 2.409e-16 | 1.0026 | 1.0005 | 0.9997 | 1.0024 |
| silicon | 100 | 1.523e-16 | 1.523e-16 | 1.0003 | 1.0003 | 0.9990 | 1.0010 |
| silicon | 1000 | 7.014e-17 | 7.069e-17 | 0.9922 | 1.0000 | 0.9917 | 1.0005 |
| silicon | 10⁶ | 9.789e-17 | 9.778e-17 | 1.0012 | 1.0000 | 1.0000 | 1.0012 |
| sio2 | 1 | 9.434e-15 | 7.770e-15 | 1.2142 | 1.0811 | 1.0000 | 1.1231 |
| sio2 | 2 | 5.928e-15 | 5.152e-15 | 1.1507 | 1.0336 | 1.0000 | 1.1133 |
| sio2 | 5 | 3.047e-15 | 2.865e-15 | 1.0635 | 1.0030 | 1.0000 | 1.0604 |
| sio2 | 10 | 1.820e-15 | 1.763e-15 | 1.0320 | 1.0014 | 0.9999 | 1.0306 |
| sio2 | 20 | 1.075e-15 | 1.061e-15 | 1.0130 | 0.9994 | 0.9998 | 1.0138 |
| sio2 | 50 | 5.459e-16 | 5.439e-16 | 1.0036 | 1.0001 | 0.9993 | 1.0042 |
| sio2 | 100 | 3.423e-16 | 3.424e-16 | 0.9996 | 1.0001 | 0.9980 | 1.0015 |
| sio2 | 1000 | 1.531e-16 | 1.559e-16 | 0.9825 | 1.0000 | 0.9816 | 1.0010 |
| sio2 | 10⁶ | 2.102e-16 | 2.097e-16 | 1.0023 | 1.0000 | 1.0000 | 1.0023 |
| mos2 | 1 | 1.660e-14 | 1.440e-14 | 1.1525 | 1.2436 | 1.0000 | 0.9268 |
| mos2 | 2 | 1.105e-14 | 9.408e-15 | 1.1741 | 1.0994 | 1.0000 | 1.0680 |
| mos2 | 5 | 5.991e-15 | 5.373e-15 | 1.1150 | 1.0261 | 1.0000 | 1.0866 |
| mos2 | 10 | 3.677e-15 | 3.445e-15 | 1.0675 | 1.0093 | 1.0000 | 1.0576 |
| mos2 | 20 | 2.234e-15 | 2.153e-15 | 1.0375 | 1.0057 | 1.0000 | 1.0316 |
| mos2 | 50 | 1.162e-15 | 1.146e-15 | 1.0133 | 1.0014 | 0.9999 | 1.0119 |
| mos2 | 100 | 7.401e-16 | 7.363e-16 | 1.0052 | 1.0009 | 0.9995 | 1.0048 |
| mos2 | 1000 | 3.491e-16 | 3.504e-16 | 0.9963 | 1.0000 | 0.9946 | 1.0016 |
| mos2 | 10⁶ | 5.044e-16 | 5.009e-16 | 1.0070 | 1.0000 | 1.0000 | 1.0070 |

"Bethe" is Eq. 3.120 with the oscillator $\delta_F$. "No-shell" is SBETHE's
`stopping_no_shell` column: the same Bethe formula with SBETHE's Fano
$\delta_F$ from its continuous OOS and no shell correction. The columns
multiply exactly to GOS/stp, so the disagreement separates into three
documented factors:

1. **Model versus Bethe (GOS/Bethe).** The δ-oscillator model exceeds Bethe
   by 7–25% at 1–2 keV. The excess falls below 1% by 10 keV for Si/SiO₂ and
   by 20–50 keV for MoS₂. It falls below $4\times10^{-5}$ above 1 MeV. The
   manual claims agreement with ICRU only at $E\ge10$ keV (Fig. 3.10).
2. **Density effect (Bethe/no-shell).** The formulas are identical
   (Eq. 3.69), but the OOS differ: δ oscillators versus SBETHE's continuous
   OOS. The difference is at most 1.8% (SiO₂, 1 MeV). It vanishes as $\beta\to1$,
   where both reach Eq. 3.73.
3. **SBETHE shell correction (no-shell/stp).** The DHFS shell correction in
   `stp.dat` lowers stopping by 3–11% at 1–2 keV (MoS₂ raises it by 7% at
   1 keV). It is 0.1–0.7% at 1 GeV. The GOS has no independent shell
   correction.

Tests assert agreement within 2% at 0.1–1000 MeV and within 0.6% at 100 keV.
A strict expected failure records the 1–10 keV disagreement. The 1, 5, 10, and
100 keV ratios are pinned to $2\times10^{-4}$. PENELOPE itself rescales
outer-shell rates to restore the stopping (Eqs. 3.141–3.142); this slice does
not.

## IMFP and straggling observations

Values come from the Eq. 3.51 density $N$.

| material | IMFP 1 keV (Å) | IMFP 10 keV (Å) | IMFP 100 keV (Å) | $\langle W\rangle$ 10 keV (eV) |
| --- | ---: | ---: | ---: | ---: |
| silicon | 18.6 | 128.7 | 810.7 | 50.6 |
| sio2 | 15.4 | 107.0 | 674.8 | 42.9 |
| mos2 | 15.0 | 101.1 | 629.8 | 70.0 |

These IMFPs depend strongly on $W_{cb}$ and have not been compared with
measured IMFPs. SBETHE `asymptotic.dat` contains uncorrected free-atom
moments. Against it, GOS $\sigma^{(0)}$ is 0.46 (Si), 0.81 (SiO₂), and
0.46 (MoS₂) at 10 keV. At high energy the ratio falls further because
`asymptotic.dat` omits $\delta_F$. SBETHE's OOS also has sub-plasmon strength
that no band gap removes (see the 2026-09-24 task notes). GOS $\sigma^{(2)}$ is
0.59–0.70 of `asymptotic.dat` at 10 keV and 0.93–0.97 at 100 keV. It reaches
$1.000$ above 10 MeV, where close collisions dominate and depend only on $Z$.
Neither comparison is asserted.

## Checks

`tests/montecarlo/test_shell_gos.py`: finiteness and non-negativity at
40 eV–1 GeV; channel and shell partition; the $N\sigma$ conversion; close
quadrature; the triangle mean; the one-shell closed form; threshold and
Eq. 3.78 continuity; conduction-band cutoff and $W_{cb}$ sensitivity; the
Bethe limit; the Eq. 3.73 limit; the Eq. 3.51 density; energy validation; and
the fetched SBETHE comparison described above.

Fresh-context re-derivation: [verification record](penelope-shell-gos-moments-verification.md).
