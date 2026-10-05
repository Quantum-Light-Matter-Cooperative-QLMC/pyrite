# Bhabha close collisions for positrons

Validation: `bhabha-close`. Independent verdict: rederived (the existing
tests also anchor it). Human sign-off remains pending (#277).

## Source and intended quantity

PENELOPE-2024 (NEA/MBDAV/R(2024)1) §3.2.2.3 gives the close energy-loss DCS
of a positron of kinetic energy $E$ on oscillator $k$ (Eq. 3.92):

$$
\frac{d\sigma^{(+)}_{\rm clo}}{dW}
=\frac{2\pi e^4}{m_ev^2}\,f_k\,\frac{F^{(+)}(E,W)}{W^2},
\qquad Q_k\le W\le E,
$$

with the Bhabha factor (Eq. 3.91), $x=W/E$,

$$
F^{(+)}(E,W)=1-b_1x+b_2x^2-b_3x^3+b_4x^4,
$$

and, writing $g=\left((\gamma-1)/\gamma\right)^2$ (Eq. 3.90),

$$
b_1=g\,\frac{2(\gamma+1)^2-1}{\gamma^2-1},\quad
b_2=g\,\frac{3(\gamma+1)^2+1}{(\gamma+1)^2},\quad
b_3=g\,\frac{2\gamma(\gamma-1)}{(\gamma+1)^2},\quad
b_4=g\,\frac{(\gamma-1)^2}{(\gamma+1)^2}.
$$

The manual states that the largest loss of a positron is $W_{\max}=E$ (the
sentence after Eq. 3.92), replacing the electron's $(E+U_k)/2$ of Eq. 3.88,
and evaluates $F^{(+)}$ at $E$ (no Coulomb shift $E'=E+U_k$, which Eq. 3.87
introduces for Møller only). The prefactor $2\pi e^4/(m_ev^2)$ has units
eV cm$^2$, so the DCS is cm$^2$/eV per formula unit and $\sigma^{(n)}$ has
units eV$^n$ cm$^2$.

## Independent derivation

**Coefficients.** $g/(\gamma^2-1)=(\gamma-1)/[\gamma^2(\gamma+1)]$, so $b_1$ is
finite as $\gamma\to1$. As $\gamma\to1$ every $b_k\propto(\gamma-1)\to0$ and
$F^{(+)}\to1$ (Rutherford $1/W^2$). As $\gamma\to\infty$, $g\to1$ and
$(b_1,b_2,b_3,b_4)\to(2,3,2,1)$. At $W=E$,
$F^{(+)}(E,E)=(\gamma^4-3\gamma^2+6)/[\gamma^2(\gamma+1)^2]>0$ for all
$\gamma$; a scan of $10\ {\rm eV}\le E\le10\ {\rm GeV}$, $0<x\le1$ gives
$\min F^{(+)}=0.267$, so the DCS is non-negative on $(0,E]$.

**Moments.** With $J^{(+)}_n=\int W^{n-2}F^{(+)}\,dW$, term-by-term
integration gives

$$
\begin{aligned}
J^{(+)}_0&=-\frac1W-\frac{b_1\ln W}{E}+\frac{b_2W}{E^2}-\frac{b_3W^2}{2E^3}+\frac{b_4W^3}{3E^4},\\
J^{(+)}_1&=\ln W-\frac{b_1W}{E}+\frac{b_2W^2}{2E^2}-\frac{b_3W^3}{3E^3}+\frac{b_4W^4}{4E^4},\\
J^{(+)}_2&=W-\frac{b_1W^2}{2E}+\frac{b_2W^3}{3E^2}-\frac{b_3W^4}{4E^3}+\frac{b_4W^5}{5E^4},
\end{aligned}
$$

identical to Eqs. 3.112–3.114; SymPy confirms
$dJ^{(+)}_n/dW-W^{n-2}F^{(+)}=0$ for $n=0,1,2$. The close moments of
oscillator $k$ are
$\sigma^{(n)}_{{\rm clo},k}=\frac{2\pi e^4}{m_ev^2}f_k\,[J^{(+)}_n]_{Q_k}^{E}$.

**Distant terms.** The first-Born distant longitudinal and transverse DCS
(Eqs. 3.76–3.81) depend on the projectile charge only through $e^2$, so
they are identical for $e^\pm$; only the upper truncation changes to
$W_{\max}=E$. Wherever $(E+U_k)/2$ does not truncate the electron distant
loss interval, the distant moments coincide.

**Bethe limit.** For $E\gg U_k$, $[J^{(+)}_1]_{Q_k}^{E}\to\ln(E/Q_k)-b_1+b_2/2-b_3/3+b_4/4$
(Eq. 3.119). Adding the distant stopping (Eq. 3.116) and using
$\sum_kf_k\ln W_k=Z\ln I$, $2m_ec^2\beta^2\gamma^2E=2E^2(\gamma+1)$:

$$
S^{(+)}=N\frac{2\pi e^4}{m_ev^2}Z\left[\ln\frac{E^2(\gamma+1)}{2I^2}
+2\ln2-\beta^2-b_1+\frac{b_2}{2}-\frac{b_3}{3}+\frac{b_4}{4}-\delta_F\right].
$$

SymPy shows
$2\ln2-\beta^2-b_1+b_2/2-b_3/3+b_4/4-f^{(+)}(\gamma)\equiv0$ with

$$
f^{(+)}(\gamma)=2\ln2-\frac{\beta^2}{12}\left[23+\frac{14}{\gamma+1}+\frac{10}{(\gamma+1)^2}+\frac{4}{(\gamma+1)^3}\right],
$$

so Eqs. 3.119, 3.120 and 3.122 are mutually consistent.

**Sampling.** The hard close loss on $[\max(Q_k,W_c),E]$ has the normalized
CDF $\left[J^{(+)}_0(W)-J^{(+)}_0(W_{\rm lo})\right]/\left[J^{(+)}_0(E)-J^{(+)}_0(W_{\rm lo})\right]$.
Binary kinematics with a free electron at rest do not involve the charge, so
the primary and knock-on cosines (Eq. 3.134 and its partner) are the
electron ones with $E\to E-W$ for the projectile. The positron is
distinguishable from the target electron and stays the primary for every
$W\le E$.

## Source-to-code comparison

| Item | Derivation | Code | Result |
| --- | --- | --- | --- |
| $b_1..b_4$ | Eq. 3.90 | `shell_gos.py::bhabha_coefficients`; Numba and CUDA `_bhabha_j0` repeat it | match (relative difference $\le5\times10^{-9}$, from my rounded $m_ec^2$) |
| $J^{(+)}_{0,1,2}$ | above | `shell_gos.py::_bhabha_integrals` | match, term by term |
| $W_{\max}$ | $E$ | `max_energy_loss_eV`; Numba `_hard_loss_bounds`; CUDA inline | match, close and distant bounds |
| $F^{(+)}$ argument | $E$, no $E+U_k$ | `_bhabha_integrals(energy_eV, w)` | match |
| close lower limit | $Q_k=U_k$ (bound), $W_{cb}$ (band) | `q_close = max(u if u>0 else q_mod, lower)` | match |
| $f^{(+)}$ | Eq. 3.122 | `bethe_stopping_cs(projectile="positron")` | match |
| kernel CDF | $J^{(+)}_0$ | `hard_inelastic.py::_sample_hard_transfer_eV`, branch $+3$ | match |
| cosines | charge-independent | `_hard_primary_cosine`, `_hard_secondary_cosine` strip the tag | match |

Branch tagging: `_split_branch` maps codes 3–5 to 0–2 plus a positron flag;
the CUDA twin inlines the same test against `I32_POSITRON_BRANCH`. Every
kernel use of `il_ch_branch` (cores and `_jit_kernel`) goes through the
three functions above, all of which strip the tag. `channel_code` stays
$3k+{\rm branch}$, so binding and inner-shell lookups are species
independent.

## Numerical checks (independent of implementation helpers)

Scratch scripts, not committed:

- Close moments of every oscillator of `silicon`, `mos2` and `ws2` at
  6 keV, 20 keV and 1 MeV versus adaptive quadrature of
  $W^{n-2}F^{(+)}$ from my own $b_k$: largest relative difference
  $2.6\times10^{-9}$ (constant rounding).
- Summed positron GOS stopping versus my Bethe formula with $f^{(+)}$ at
  10 MeV–1 GeV: $-1.2\times10^{-5}$ (silicon), $-4\times10^{-5}$ (MoS₂),
  $-7.7\times10^{-5}$ (WS₂).
- Distant moments: equal to the electron ones except where the electron's
  $(E+U_k)/2$ truncates a near-threshold triangle, e.g. Si K at 6 keV.
- Numba `_sample_hard_transfer_eV` with branch 5: inverting my quadrature
  CDF at the sampled $W$ reproduces $u$ to $\le8\times10^{-12}$; a KS test of
  20 000 draws against that CDF gives $p=0.33$–$0.85$ for
  $(E,U)=(20\ {\rm keV},104\ {\rm eV})$, $(1\ {\rm MeV},1844\ {\rm eV})$,
  $(1\ {\rm MeV},{\rm band})$ and $(5\ {\rm keV},1844\ {\rm eV})$; draws reach
  $0.98E$, beyond the electron limit. Tagged and untagged primary/secondary
  cosines agree exactly, and the primary cosine matches
  $\sqrt{(E-W)(E+2m_ec^2)/[E(E-W+2m_ec^2)]}$.
- Positron closure (`catalog_shell_rate_closure(projectile="positron")`):
  total first moment equals positron `stp.dat` to $\le2.2\times10^{-16}$;
  the soft/hard partition at $W_c=500$ eV sums to it to the same precision;
  every open inner shell has scale exactly 1.
- Repository tests: `test_shell_gos_positron.py`,
  `test_positron_shell_transport.py`, `test_positron_cascade.py` and
  `test_positron_runner.py` pass. The three CUDA tests in
  `test_positron_cuda.py` skip (no device); the CUDA twin was compared by
  reading the code only.

## Documented deviations (accepted, not discrepancies)

- **Inner shells keep the raw Bhabha GOS.** EEDL is electron-impact only, and
  PENELOPE's positron DWBA tables (`pdpsi`) are not vendored, so
  `gos_inner_cross_sections` passes each inner oscillator's own
  $\sigma^{(0)}$, giving scale 1. Near threshold the GOS can differ from DWBA
  positron ionization by tens of percent; this affects inner-shell vacancy
  yields in positron tracks, not total stopping, which closes on SBETHE.
- **Kawrakow $\xi$ unchanged.** `atomic_electron_deflection="kawrakow"`
  subtracts the Møller, not the Bhabha, hard share from the atomic-electron
  angular term for positrons. It is a second-order angular effect, but not
  the positron physics.
- **$2m_ec^2$ per pair** is booked as `positron_rest_pending` whatever the
  positron's fate, since annihilation (#295) is absent. The balance closes
  by construction; deposited dose omits annihilation photons.

## Observation (no discrepancy)

The hard transfer is drawn at the row-start energy $E_j$ up to
$W_{\max}=E_j$ and subtracted from the row-end energy $E_{{\rm end},j}<E_j$.
For electrons $W\le(E_j+U)/2$ makes $W>E_{{\rm end},j}$ unreachable. For
positrons a draw in $(E_{{\rm end},j},E_j]$ is possible. The cutoff branch
then takes it and `cutoff_residual` $=E_{{\rm end}}-W$ is negative for
that row, though the global balance still closes. The probability per close
event is about $(\Delta E_{\rm soft}/E)\,(W_c/E)\,F^{(+)}(E,E)$; in 20 seeded
cascade runs (300 positron hard rows, soft fraction $\le9\times10^{-4}$)
none occurred. Clamping $W$ to $E_{{\rm end},j}$, or a test that asserts
non-negative per-row residuals, would close the gap.

## Verdict

Matches. Coefficients, antiderivatives, $W_{\max}$, the Bethe limit
$f^{(+)}$, closure, kernel CDF inversion and kinematics all agree with the
source. Recommended status: `anchored`, with the anchors already in the row.
