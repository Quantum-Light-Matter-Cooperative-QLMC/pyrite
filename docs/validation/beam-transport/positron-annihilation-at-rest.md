# Positron annihilation at rest

Validation: `positron-annihilation-at-rest`. Independent verdict: rederived.
Human sign-off remains pending (#277).

## Source and intended quantity

PENELOPE-2024 (NEA/MBDAV/R(2024)1) §3.4 treats target electrons as free and
at rest. A positron whose kinetic energy falls below the absorption energy
annihilates at rest. The pair then has four-momentum $(2m_ec^2,\mathbf 0)$,
so two-photon annihilation gives

$$
E_1=E_2=m_ec^2,\qquad \hat{\mathbf v}_2=-\hat{\mathbf v}_1 .
$$

Nothing in the initial state picks a direction, so the common axis is
isotropic:

$$
\hat{\mathbf v}_1=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta),\qquad
\cos\theta=2\xi_1-1,\quad \phi=2\pi\xi_2 .
$$

Two limits act as checks. The in-flight Heitler kinematics reduce to this
case, since $\zeta_{\min}\to\tfrac12$ as $\gamma\to1$ and then
$E_\pm\to m_ec^2$. The total photon momentum is zero.

The residual kinetic energy, below $T_s$ at birth or at the tracking cutoff,
is deposited locally. If a positron leaves the stack, its $2m_ec^2$ leaves
with it. Per primary history the energy then closes as

$$
k=E_{\rm dep}+E_{\rm esc}^{e^\pm}+E_{\rm esc}^{\gamma}+E_{\rm bind}
+E_{\rm ann,esc}+E_{\rm ann,abs}+2m_ec^2N_{e^+,\rm esc}.
$$

Here the annihilation photon terms carry $2m_ec^2$ per non-escaping positron,
plus $E_{\rm ann}$ for each in-flight one.

## Comparison with the implementation

- **Directions.** `sample_at_rest_directions` builds
  $\cos\theta=2\xi_1-1$ and $\phi=2\pi\xi_2$ and returns the axis and its
  negative. Over 50 000 draws, $\max\lvert\hat{\mathbf v}_1+\hat{\mathbf v}_2\rvert=0$
  exactly. The KS $p$-values are 0.85 for $\cos\theta$ against
  $U[-1,1]$ and 0.33 for $\phi$ against $U[-\pi,\pi]$. The mean axis is
  $(-2.0,0.7,1.3)\times10^{-3}$, consistent with 0 at
  $\sigma=2.6\times10^{-3}$.
- **Energies.** `score_annihilation_photons` sets both photons to
  `ELECTRON_REST_KEV`$=510.99895$ keV for `ANNIHILATION_AT_REST`. In every
  cascade run each at-rest photon equals $m_ec^2$ exactly.
- **Fates.**
  - Positrons born at or below $T_s$: `_init_positron_fates` marks them
    `FATE_SUBTHRESHOLD`. Their annihilation point and clock are those of
    the pair event, and their kinetic energy is booked
    `positron_subthreshold` (part of deposited).
  - Positrons reaching the cutoff: `_annihilate_positrons` marks a
    transported positron whose (possibly truncated) last row is `CUTOFF` as
    `FATE_AT_REST`. Its point is that row's end,
    `r_mid` $+\tfrac12$`L_ang``v_hat`, and its residual is booked by the
    existing `cutoff_residual`.
  - Escaping positrons are marked `FATE_ESCAPED`. They get no photons and
    are booked $2m_ec^2$ (`positron_escaped_rest`). Their kinetic energy at
    exit is in `escaped`.
  - Any other terminal kind raises.

  In the cascade runs below every non-escaping pair had exactly two photons
  and every escaping pair none, and the fates partition the pairs.
- **Closure.** `secondary_energy_balance` drops `pair_rest_mass` and
  `positron` when positrons are transported. It adds `positron_subthreshold`
  (to deposited), `positron_escaped_rest`, `annihilation_escaped` and
  `annihilation_absorbed`, where the last two split each scored photon by
  whether its EPDL first interaction lies in the stack. These terms enter
  the residual once each. I ran `_cascade` (3 MeV, Si, `pair_scale=1e9`):
  seeds 11 and 5 at $10^6$ Å and seed 23 at $3\times10^6$ Å, both with the
  real $\sigma_{\rm an}$ and with it scaled by 300. In all runs the global
  residual is $\le3.6\times10^{-12}$ keV and every per-history residual
  $\le2.7\times10^{-12}$ keV, against 18 MeV incident. With the real
  $\sigma_{\rm an}$ all 10 non-escaping transported positrons of seeds 11
  and 5 annihilated at rest. The `positron_rest_pending` term no
  longer exists.
- **RNG.** At-rest photons use the emission salt `0xA0761D6478BD642F` keyed
  on (seed, parent track, photon ordinal), the same key as in flight. The
  first two uniforms set the axis and the next four set the two photons'
  first-interaction draws. The runs are bit-reproducible.
- **Off switch.** `positron_transport=False` leaves the results unchanged.
  This is covered by
  `test_positron_cascade.py::test_positron_mode_is_reproducible_and_off_is_unchanged`,
  which passes.

## Findings

No discrepancies. The model omits positronium, three-photon decay and Doppler
broadening of the 511 keV line. These are declared assumptions.

## Verdict

`rederived`. The photon energies, back-to-back isotropic directions, fate
partition and per-history energy closure match the independent derivation.
