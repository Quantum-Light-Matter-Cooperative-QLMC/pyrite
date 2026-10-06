# In-flight two-photon positron annihilation (Heitler)

Validation: `heitler-annihilation`. Independent verdict: rederived. The
manual's rejection bound is wrong and PyRITE's replacement is correct. Human
sign-off remains pending (#277).

## Source and intended quantity

PENELOPE-2024 (NEA/MBDAV/R(2024)1) §3.4 treats the target electrons as free
and at rest. A positron of kinetic energy $E$, $\gamma=1+E/m_ec^2$,
annihilates into two photons. The lower photon fraction is (Eq. 3.183)

$$
\zeta=\frac{E_-}{E+2m_ec^2},\qquad \zeta_{\min}\le\zeta\le\tfrac12,\qquad
\zeta_{\min}=\frac{1}{\gamma+1+\sqrt{\gamma^2-1}}\quad\text{(Eq. 3.186)}.
$$

The photons leave at polar angles about the positron direction (Eqs. 3.184–3.185)

$$
\cos\theta_-=\frac{\gamma+1-1/\zeta}{\sqrt{\gamma^2-1}},\qquad
\cos\theta_+=\frac{\gamma+1-1/(1-\zeta)}{\sqrt{\gamma^2-1}},
$$

with azimuths $\phi$ and $\phi+\pi$. The DCS per electron is (Eqs. 3.187–3.188)

$$
\frac{d\sigma_{\rm an}}{d\zeta}=\frac{\pi r_e^2}{(\gamma+1)(\gamma^2-1)}\,[S(\zeta)+S(1-\zeta)],\qquad
S(\zeta)=-(\gamma+1)^2+\frac{\gamma^2+4\gamma+1}{\zeta}-\frac{1}{\zeta^2}.
$$

The total is (Eq. 3.189)

$$
\sigma_{\rm an}=\frac{\pi r_e^2}{(\gamma+1)(\gamma^2-1)}
\left\{(\gamma^2+4\gamma+1)\ln\!\left[\gamma+\sqrt{\gamma^2-1}\right]-(3+\gamma)\sqrt{\gamma^2-1}\right\}.
$$

The inverse mean free path is $\lambda_{\rm an}^{-1}=\mathcal NZ\sigma_{\rm an}$ (Eq. 3.190).
To sample $\zeta$, the manual draws $\upsilon$ from $P(\upsilon)=S(\upsilon)$ on
$[\zeta_{\min},1-\zeta_{\min}]$ and factors it as $\pi(\upsilon)g(\upsilon)$ with
$\pi\propto1/\upsilon$ (Eqs. 3.192–3.196):

$$
g(\upsilon)=-(\gamma+1)^2\upsilon+(\gamma^2+4\gamma+1)-\frac1\upsilon,\qquad
\upsilon=\zeta_{\min}\left(\frac{1-\zeta_{\min}}{\zeta_{\min}}\right)^{\xi}.
$$

A draw is rejected if $\xi g(\zeta_{\min})>g(\upsilon)$, and the sampler
returns $\zeta=\min(\upsilon,1-\upsilon)$ (Eq. 3.197).

Units: $r_e=2.8179403262\times10^{-5}$ Å, so $\sigma$ is in Å². $\mathcal NZ$
is in Å⁻³ and $E$ in keV with $m_ec^2=510.99895$ keV.

## Independent derivation (before reading the implementation)

- **Eq. 3.189 from Eq. 3.187.** The antiderivative of $S$ is
  $H(\zeta)=-(\gamma+1)^2\zeta+(\gamma^2+4\gamma+1)\ln\zeta+1/\zeta$. The
  integral is $\int_{\zeta_{\min}}^{1/2}[S(\zeta)+S(1-\zeta)]\,d\zeta=[H(\tfrac12)-H(\zeta_{\min})]-[H(\tfrac12)-H(1-\zeta_{\min})]$.
  SymPy evaluates the difference from the closed form as exactly zero at
  $\gamma=3$ and $\gamma=1.01$. With $\zeta_{\min}(1-\zeta_{\min})^{-1}=(\gamma+\sqrt{\gamma^2-1})^{-1}$,
  the log term is $(\gamma^2+4\gamma+1)\ln(\gamma+\sqrt{\gamma^2-1})$. A
  30-digit mpmath quadrature of Eq. 3.187 agrees with Eq. 3.189 to
  $8\times10^{-30}$ (relative) from 1 keV to 100 MeV.
- **Low-energy limit.** Write $\gamma=1+\epsilon$ and $s=\sqrt{2\epsilon}$.
  The brace tends to $6s-4s=2s$ and the prefactor to $\pi r_e^2/(4\epsilon)$,
  so $\sigma_{\rm an}\to\pi r_e^2/s=\pi r_e^2/\beta$. Numerically
  $\sigma\beta/\pi r_e^2=1-2.3\times10^{-18}$, $1-2.3\times10^{-12}$ and
  $1-2.3\times10^{-6}$ at $E=10^{-6}$, $10^{-3}$ and 1 keV.
- **High-energy limit.** The brace tends to $\gamma^2(\ln2\gamma-1)$, so
  $\sigma\to\pi r_e^2(\ln2\gamma-1)/\gamma$ (Dirac). The ratio to Dirac is
  1.0163, 1.00016 and $1+1.6\times10^{-6}$ at 0.1, 10 and 1000 GeV.
- **Magnitude.** $\sigma_{\rm an}=3.99$, 1.28, 0.447, 0.172, 0.0383 and
  0.00641 b at 1 keV, 10 keV, 100 keV, 1 MeV, 10 MeV and 100 MeV. This
  matches Fig. 3.20 (right).
- **$\zeta_{\min}$.** Substituting $\zeta_{\min}$ into Eq. 3.184 gives
  $\cos\theta_-=-1$ exactly (SymPy), as the manual states. At rest,
  $\zeta_{\min}\to\tfrac12$.
- **Kinematics.** In units of $m_ec^2$ the photon energies are
  $\zeta(\gamma+1)$ and $(1-\zeta)(\gamma+1)$. The longitudinal momentum
  $\zeta(\gamma+1)\cos\theta_-+(1-\zeta)(\gamma+1)\cos\theta_+-\sqrt{\gamma^2-1}$
  simplifies to 0. The transverse balance
  $[\zeta(\gamma+1)]^2\sin^2\theta_--[(1-\zeta)(\gamma+1)]^2\sin^2\theta_+$
  also simplifies to 0. Equal transverse magnitudes at azimuths $\phi$ and
  $\phi+\pi$ therefore cancel, and Eqs. 3.184–3.185 conserve four-momentum
  exactly.
- **Envelope (the manual's claim).** $g'(\upsilon)=-(\gamma+1)^2+\upsilon^{-2}$
  and $g''=-2\upsilon^{-3}<0$, so $g$ is concave with its single maximum at
  $\upsilon^*=1/(\gamma+1)$. There
  $g(\upsilon^*)=\gamma^2+4\gamma+1-2(\gamma+1)=\gamma^2+2\gamma-1$. For
  $\gamma>1$, $\upsilon^*>\zeta_{\min}$ because
  $\sqrt{\gamma^2-1}>0$, and $\upsilon^*<\tfrac12<1-\zeta_{\min}$. So $g$ rises on
  $[\zeta_{\min},\upsilon^*]$ and **is not monotonically decreasing**. The
  manual's statement is false, and $g(\zeta_{\min})<g_{\max}$ for every
  $\gamma>1$:

  | $E$ (keV) | $\zeta_{\min}$ | $1/(\gamma+1)$ | $g(\zeta_{\min})$ | $g_{\max}$ | $g(1-\zeta_{\min})$ |
  | --- | --- | --- | --- | --- | --- |
  | 10 | 0.4508 | 0.4952 | 2.0608 | 2.0787 | 2.0570 |
  | 511 | 0.2113 | 0.3333 | 6.366 | 7.000 | 4.634 |
  | 1000 | 0.1484 | 0.2527 | 12.509 | 13.657 | 7.063 |
  | 10 000 | 0.02374 | 0.04636 | 453.2 | 463.2 | 51.16 |

  With the manual's bound, $g(\upsilon)/g(\zeta_{\min})>1$ near $\upsilon^*$.
  The acceptance saturates there and the sample is biased. My own sampler
  with that bound fails a KS test against the exact CDF at 1 MeV
  ($D=0.0109$, $p=10^{-10}$, $10^5$ samples). At 10 MeV the excess is only 2%
  and goes undetected at this sample size ($p=0.23$). Because $g$ is concave,
  its minimum on the support is at an endpoint, and both endpoint values
  above are positive. So $g\ge0$ and $g/g_{\max}\in[0,1]$, a valid
  acceptance probability. The manual's "$\sim80\%$ efficiency at 10 MeV"
  remark does not affect this.
- **Exact CDF for the sampler test.**
  $F(\zeta)=\{[H(\zeta)-H(\zeta_{\min})]-[H(1-\zeta)-H(1-\zeta_{\min})]\}/F_0$,
  where $F_0$ is the same bracket at $\zeta=\tfrac12$.
- **Thinning.** Annihilation is terminal. Suppose the other channels'
  hazards do not include it and its uniform is drawn from a stream
  independent of the transport. Then the joint law of (trajectory up to $s$,
  no annihilation before $s$) factorises as
  $P_{\rm transport}(\text{path})\exp[-\int_0^s\mathcal NZ\sigma_{\rm an}(E(s'))ds']$.
  That is the law of a process that competes annihilation with the other
  channels. Cutting the transported track where
  $\int\mathcal NZ\sigma_{\rm an}ds=-\ln(1-\xi)$ samples exactly this law.
  Every hard event, secondary and photon past the cut must be removed. The
  remaining approximation is the within-row dependence $E(s)$, which is not
  known inside a condensed-history row.

## Comparison with the implementation

`annihilation.py` and the cascade hooks in `secondaries.py` were read only
after the derivation above.

- **Cross section.** `heitler_cross_section_ang2` is Eq. 3.189 term for term.
  It agrees with my mpmath reference to $2.2\times10^{-16}$ (relative) on 61
  energies from 0.1 keV to 100 MeV. `_CLASSICAL_RADIUS_ANG` is CODATA $r_e$ in Å.
  `electron_density_per_ang3` for silicon gives 0.69920 Å⁻³, against
  $2.329\,{\rm g\,cm^{-3}}\,N_A\,Z/A=0.69914$ Å⁻³.
- **Density.** `heitler_zeta_density` is Eqs. 3.187–3.188 divided by Eq. 3.189.
  It integrates to $1\pm1.5\times10^{-4}$ (trapezoid on 20 001 points).
- **Sampler.** `sample_heitler_zeta` uses exactly Eq. 3.196, the acceptance
  test $\xi g_{\max}\le g(\upsilon)$ with $g_{\max}=\gamma^2+2\gamma-1$, and
  Eq. 3.197. I ran a KS test of $10^5$ samples per energy against my analytic
  CDF $F$:

  | $E$ | $D$ | $p$ | sample min / $\zeta_{\min}$ |
  | --- | --- | --- | --- |
  | 10 keV | 0.00325 | 0.24 | 0.45078 / 0.45078 |
  | 1 MeV | 0.00189 | 0.87 | 0.14838 / 0.14838 |
  | 10 MeV | 0.00321 | 0.26 | 0.023745 / 0.023745 |
  | 100 MeV | 0.00171 | 0.93 | 0.0025357 / 0.0025356 |

- **Kinematics.** `sample_heitler` sets $E_-=\zeta(E+2m_ec^2)$ and
  $E_+=E+2m_ec^2-E_-$, so energy is conserved by construction. The polar
  cosines come from Eqs. 3.184–3.185, clamped to $[-1,1]$ against rounding,
  and both photons are rotated by `_rotate` about the same positron direction
  with $\phi$ and $\phi+\pi$. Over 20 000 random energies (1 keV–100 MeV) and
  directions, the maximum relative energy error is $2.1\times10^{-16}$. The
  maximum $\lvert\mathbf p_-+\mathbf p_+-\mathbf p_{e^+}\rvert/(E+2m_ec^2)$ is
  $1.5\times10^{-12}$, and unit norms hold to $2.2\times10^{-16}$.
- **Thinning on synthetic tracks.** I built 40 000 shuffled 12-row tracks.
  Each row loses 200 keV linearly over $2\times10^4$ Å, with a 30 keV hard
  drop at each row end, from 3 MeV down, and $\mathcal NZ=1.5\times10^3$ Å⁻³.
  `truncate_in_flight` cut 41.21% of the tracks. My `scipy.quad` value was
  $1-e^{-0.53278}=41.30\%$, a difference of $z=-0.39$. Re-integrating
  independently to each recorded cut point reproduces the budget $\tau$ to
  $1.4\times10^{-10}$ (relative). The cut energy equals the linear
  interpolation at the cut point. The cut-track budgets pass a KS test
  against $\mathrm{Exp}(1)$ truncated at the total depth ($p=0.42$).
- **Truncated-row contract.** On the same data the crossing row is the
  track's last and only `ANNIHILATION` row. Its `E_end_keV` is
  $E_{\rm ann}$, its end point is `r_mid` $+\tfrac12$`L_ang``v_hat` and lies
  at the cut, `t_end_ang` is interpolated linearly, `hard_channel` is $-1$,
  and `hard_W_keV` and the radiative payload are 0. Later rows are dropped,
  and uncut tracks keep all rows. This follows the events.py contract: the
  event sits at the far end of a straight row, and `E_end_keV` is the
  pre-event energy. A hard event at the end of the crossing row therefore
  correctly never happens.
- **Gauss–Legendre.** The 8-point row rule is exact to $10^{-13}$ when
  $E_0/E_1\le2$. It reaches $3\times10^{-4}$ (relative) only for a single
  row spanning 3 MeV to 100 keV. In the cascade runs below, positron rows
  have $E_0/E_1\le1.0009$, so the quadrature error is at rounding.
- **Cascade.** I ran `_cascade` (3 MeV, Si, `pair_scale=1e9`) with seeds 11
  and 5 at $10^6$ Å and seed 23 at $3\times10^6$ Å, with $\sigma_{\rm an}$
  scaled by 300 to force in-flight events and also unscaled. Results:
  - Each in-flight track ends in exactly one `ANNIHILATION` row, at the
    recorded point and energy.
  - No child of a positron track is launched after its cut clock.
  - The photon energies sum to $E_{\rm ann}+2m_ec^2$ exactly.
  - Photon momentum matches $\sqrt{E(E+2m_ec^2)}\,\hat{\mathbf v}$ to
    $1.2\times10^{-15}$.
  - Recounting terminal kinds per generation gives `n_backscattered`,
    `n_transmitted`, `n_side_exited`, `n_cutoff_stopped` and `n_annihilated`
    equal to `counts_per_generation`. So the decrement of the original exit
    or cutoff count is right.
  - Global and per-history residuals are $\le3.6\times10^{-12}$ keV against
    18 MeV incident.
  - Repeated runs are bit-identical.

  `tests/montecarlo/test_positron_annihilation.py` and
  `test_positron_cascade.py` pass (26 tests).
- **Hazard not double counted.** The positron transport overrides only
  stopping, elastic and BremsLib tables. No transport kernel references
  annihilation, so the cut is the only annihilation channel.
- **RNG.** The budget uses `annihilation_stream_keys(..., in_flight=True)`
  with salt `0x5851F42D4C957F2D`. Emission and photon scoring use
  `0xA0761D6478BD642F`. Both are keyed on (seed, parent track, photon
  ordinal), the pair event's unique identity. Both salts differ from every
  other stream salt in the package. The budget stream is independent of the
  transport stream (`_PAIR_POSITRON_STREAM_SALT`), as thinning requires.

## Findings

- **Envelope claim: confirmed.** The manual's text bound $g(\zeta_{\min})$ is
  below $\max g$ for every $\gamma>1$ and biases the sample measurably at
  1 MeV. PyRITE's $g_{\max}=\gamma^2+2\gamma-1$ is the exact supremum. I have
  no PENELOPE Fortran source in this environment, so I could not check
  whether the released `PANaR` routine uses the text bound. The ledger note
  should say that the *manual text* states that bound, rather than implying
  the code does.
- **Within-row energy.** Linear $E(s)$ inside a row is a modelling
  assumption. It is declared in the ledger row.
- **Latent: `vacuum_*` arrays.** `truncate_in_flight` and
  `_annihilate_positrons` do not truncate the result's `vacuum_*` flight
  arrays (`secondaries.py::_annihilate_positrons`). This is unreachable today because
  `_validate_pair_model` refuses a groove, so those arrays are empty. It
  would need handling if positron transport ever admits grooves.
- **Annihilation photons above $2m_ec^2$.** These never pair-convert. Their
  energy is booked `annihilation_absorbed` at the first interaction. This is
  a declared assumption, not a defect.

## Verdict

`rederived`. Eqs. 3.183–3.190 and the sampler match an independent
derivation, both limits hold, and the corrected envelope is right. Thinning
is equivalent to a competing channel, with the stated within-row
approximation, and the truncated-row bookkeeping and energy closure are
exact. No blocking discrepancy.

## Independent device-port verification (2026-10-06)

A fresh verifier derived the equations and rejection envelope before reading
`_jit_annihilation.py` or the integration diff. Primary-source §3.4 excerpts
were available through indexed searches of the [PENELOPE-2024 manual](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf);
direct retrieval of the full PDF failed. This review used those excerpts,
the transcribed equations above, and independent conservation algebra.

Writing $p=\sqrt{\gamma^2-1}$, energy and longitudinal/transverse momentum
conservation yield Eqs. 3.184–3.186. Integrating $S$ over the full unfolded
support with its antiderivative $H$ gives Eq. 3.189. The identity needed for
the logarithm is

$$
\frac{1-\zeta_{\min}}{\zeta_{\min}}=\gamma+p,
\qquad
g'(\upsilon)=-(\gamma+1)^2+\upsilon^{-2},
\qquad
\max g=\gamma^2+2\gamma-1.
$$

`_heitler_cross_section_ang2` implements Eq. 3.189 in Å² per free electron,
using the shared rest energy and classical-radius constants. `_sample`
draws the logarithmic proposal, applies the true maximum above, folds
the accepted fraction, and rotates the Eq. 3.184–3.185 photon pair with
opposite transverse azimuths. Units, signs, both asymptotic limits, and
the photon-exchange convention agree. The shared rotation helper constructs
an orthonormal basis about the normalized positron direction.

Each event starts its keyed SplitMix64 counter at zero: two uniforms per
rejection attempt, one azimuth uniform after acceptance, then four scoring
uniforms. Reordering or chunking changes neither the event key nor its
counter sequence. This establishes the stated stream contract; it does not
assert bitwise equality with the legacy host Philox sampler.

The integration diff selects device emission for a CUDA cascade, checks
the validity flags, and transfers the photon payload at the existing host
EPDL first-interaction scoring boundary. The default host branch is
unchanged. Hazard integration and fate accounting remain the existing
host implementation; this port introduces no second annihilation hazard.

An independent 70-digit mpmath evaluation of Eq. 3.189 compared with the
device expression evaluated in host float64 gives relative errors
$1.05\times10^{-13}$, $1.08\times10^{-15}$, $2.82\times10^{-16}$ and
$6.36\times10^{-17}$ at 0.1, 10, 1000 and 100000 keV. These are arithmetic
checks of the expression, not execution of compiled CUDA kernels.

The final device kernels reject a Lorentz factor rounded to one or an
overflowed $\gamma^2-1$: the cross section returns NaN, while emission
retains its initial NaN payload and false validity flag. This resolves the
initial zero-denominator finding at $10^{-14}$ keV. Float64 conditioning
still limits accuracy near rest: evaluating the direct formula at
$10^{-12}$ keV differs from the high-precision reference by about 1.04%,
well below the tracking regime. This verification covers the stated
physical model and ordinary tracking energies, not uniform numerical
accuracy across every representable positive float64 value.

Device-port verdict: `rederived`; no remaining source equation, unit,
sign, or rejection-envelope discrepancy. Compiled-device distribution,
momentum and stream checks require the task owner's CUDA test execution;
this verifier ran no GPU workload and makes no human-sign-off claim.
