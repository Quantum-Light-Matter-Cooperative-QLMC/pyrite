# Shell soft/hard secondary electron transport

`Validation: shell-secondary-transport` — independent re-derivation for the
[ledger claim](../ledger-transport-background.md#shell-secondary-transport).

## Source, scope, and assumptions

Source: [PENELOPE-2024, NEA/MBDAV/R(2024)1](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf),
§3.2.5.4, Eqs. 3.137–3.138 (secondary energy and direction), and §4.2
(mixed transport). The ledger also depends on
`penelope-shell-secondary-direction` and `penelope-shell-hard-loss-sampling`.
The intended quantity is the full electron shower returned by
`simulate_trajectories(secondary_threshold_eV=...)`, starting with one or more
primaries in shell soft/hard mode. The cutoff $T_s$ is a kinetic energy in eV;
row energies are in keV, positions and clocks in Å (with $c=1$). Shell
vacancies reserve binding energy without relaxation; hard bremsstrahlung is
debited only in the coupled radiative mode. The target electron is initially
at rest for the emission direction.

## Independent construction before code inspection

At a hard collision the projectile loses $W$. For an explicitly ionized inner
shell of binding energy $U_k$, the emitted electron takes $T=W-U_k$ and the
vacancy holds $U_k$. For an outer shell or conduction electron, $T=W$. With a
single production and tracking threshold,

$$
T = \begin{cases}W-U_k,&\text{inner-shell vacancy},\\W,&\text{otherwise},\end{cases}
\qquad
\text{launch iff }T>T_s.
$$

If $T\le T_s$, its kinetic energy is locally deposited. A launched electron
starts at the collision endpoint, $\mathbf r_{\rm mid}+L\hat{\mathbf v}/2$,
at the endpoint clock, in the collision layer. Its inherited bunch offset is
the primary's. It is tracked with the same medium and transport model, with
its cutoff set to $T_s$. The same rule applies recursively. An absorbing
primary collision still produces its eligible secondary.

For projectile momentum magnitude $p(E)=\sqrt{E(E+2m_ec^2)}/c$ and recoil
momentum $q=p(Q)$, momentum conservation gives the secondary polar angle in
the incident flight frame:

$$
\cos\theta_s=\frac{p(E)^2+q^2-p(E-W'_k)^2}{2p(E)q},
\qquad \phi_s=\phi_p+\pi\pmod{2\pi}.
$$

The longitudinal channel uses its sampled recoil $Q$ and modified resonance
$W'_k$; the close channel uses $Q=W$; the transverse channel uses its stated
fixed cosine $1/2$. All are dimensionless cosines. A physical recoil triangle
requires $-1\le\cos\theta_s\le1$. Opposite azimuths put the projectile and
secondary transverse momenta on opposite sides of the incident direction.

For one collision, let $E^-$ and $E^+$ be the projectile kinetic energies
immediately before and after it, so $E^--E^+=W$. The locally accounted
energy is $U_k+T=W$ if the secondary is below threshold, or $U_k$ plus the
launched track's initial $T$ otherwise. Summing this identity and continuous
losses over every track cancels all launched energies between a parent's loss
and a child's start. Thus, for incident primaries that enter the slab,

$$
\sum E_0=E_{\rm escaped}+E_{\rm deposited}
+E_{\rm binding}+E_{\rm radiated}.
$$

Escaped energy is the kinetic energy of tracks exiting through any face;
deposited energy includes continuous loss, subthreshold secondary kinetic
energy, and cutoff residuals; binding counts each explicit vacancy once;
radiated energy counts energy-debiting hard photons. The equation holds
separately for each primary history. Each track ends once, and the number of
generation-$g+1$ tracks equals the number of qualifying generation-$g$ hard
events. Unique child identity can be assigned from its parent track and hard
event ordinal; the random stream must be keyed by these plus the run seed, so
processing order and batch size cannot change a child's draws.

The limiting checks are direct: `secondary_threshold_eV=None` selects the
original primary-only path without additional draws; $T_s$ above every
secondary energy launches no tracks; a collision at $T=T_s$ deposits $T$;
and vacancy bookkeeping does not change a primary track's characteristic
path-length estimator.

## Implementation comparison

The host and CUDA hard-collision paths compute the secondary direction from
the same channel, transfer, recoil, and azimuth draws as the primary; the host
uses `_hard_secondary_direction` and CUDA uses `_hard_secondary_cosine` plus
`_store_rotated`. Both retain the direction even when the parent collision
ends in `EVENT_CUTOFF`. The sampled longitudinal cosine has the derived
$p(E)^2+q^2-p(E-W'_k)^2$ numerator; the close branch reduces to its $Q=W$
expression; the transverse branch returns $1/2$. Both use the opposite
azimuth. The input and output angles are dimensionless.

`hard_event_energy_accounting` gives $T=W-U_k$ for an inner shell
and $T=W$ otherwise, in keV. `_harvest` launches only when
$T>T_s$ after converting eV to keV. It takes the recorded row endpoint,
direction, clock, and layer; a generation inherits its parent's primary
history and bunch offset. `transport_secondary_cascade` replaces beam
sampling with that launch state, sets every child's cutoff to $T_s$, and
repeats until no qualifying collision remains. Its child stream key mixes
the seed, parent track ID, and parent hard ordinal under a separate salt.
Generation zero still enters the pre-existing `simulate_trajectories` path
when the option is `None`. A finite-threshold run reproduces its primary
rows.

`secondary_energy_balance` includes exit energies, continuous row loss,
cutoff residuals, subthreshold $T$, reserved binding, and coupled hard
photons in keV. Its per-row `E_start_keV - E_end_keV` term excludes
the discrete hard transfer, which is distributed to those other terms.
Consequently the implementation realizes the all-track telescoping
identity above. In uncoupled mode, post-hoc radiation does not enter that
identity. A track has one terminal count; each launched hard event produces
one next-generation track. Characteristic scoring uses the ordinary
track-length estimator for each track; vacancy bookkeeping introduces no
extra source.

The focused regression anchors check host direction against the independent
`shell_collision_world_directions` sampler, primary-row invariance,
exact launch states, batch-size invariance, event and particle balance, and
energy residual below $10^{-9}$ of incident energy for Si and MoS₂, with and
without straggling. They also cover coupled radiative loss, absorbing hard
collisions, characteristic yield, and the no-launch limit. These are
implementation anchors; they do not replace the derivation.

**Verdict: rederived.** No divergent factor, sign, unit, threshold convention,
or energy term found. The ledger may advance from `unverified` to
`rederived` after review of this record; only a human may set
`signed-off`.

## Empirical checks

`checks/shell_secondary_transport_observables.py` ran at full size on the
lab box CPU partition at revision `454d1d46` (collector record in
`docs/validation/check-records/shell_secondary_transport_observables.jsonl`;
full per-case report, the source of the table below, in
[`shell-secondary-transport-454d1d46.json`](shell-secondary-transport-454d1d46.json)).
Si and MoS₂ slabs of 0.2 and 1.2 CSDA ranges at 20 and 100 keV;
$T_s$ = off, 10 (100 keV only), 5, 2 and 1 keV with the primary cutoff at the
1 keV SBETHE floor; 2000 (20 keV) or 600 (100 keV) primaries per seed, five
seeds. All gates pass.

| Quantity | Gate | Observed |
|---|---|---|
| Aggregate energy residual / incident | $\le10^{-9}$ | $\le2.4\times10^{-16}$ |
| Worst per-history residual / $E_0$ | $\le10^{-9}$ | $\le2.7\times10^{-15}$ |
| Primary backscatter fraction vs $T_s$ | identical | identical in every case |
| Energy backscatter, transmission, characteristic yield, $T_s$ 2→1 keV | $\le1\%$ | $\le0.1\%$ |
| Bremsstrahlung yield, $T_s$ 2→1 keV | $\le1\%$ | $\le0.63\%$ (Si, 20 keV) |
| Depth-dose L1 distance, $T_s$ 2→1 keV | $\le0.005$ | $\le0.0023$ |
| Generation-1 launches / bound Møller, 2–8 keV | $1\pm(0.2+3\sigma_{\rm Poisson})$ | 0.88–1.18 |

Every observable approaches its lowest-threshold value monotonically. Turning
secondary transport on raises characteristic yield by up to 7% (Si, 100 keV)
and bremsstrahlung by up to 4%, and it moves energy backscatter and
transmission by at most 0.2% of the incident energy.

The cascade reference integrates the relativistic Møller DCS along every
generation-0 row. A free-electron reference puts every electron at rest and
unbound, $T=W$; it gives ratios of 0.85–1.03 for Si and 0.71–0.89 for MoS₂.
The MoS₂ deficit is the binding shift: 14 of its 74 electrons per formula unit
have $U\ge2.5$ keV, so an inner-shell secondary with $T$ in a 2–8 keV bin needs
$W=T+U$. The bound reference keeps Møller in $W$ for each catalogue shell and
shifts each explicit inner shell to $T=W-U$. It gives 0.88–1.09 for MoS₂ and
1.03–1.18 for Si. The reference covers close collisions only. It omits
distant hard collisions on deep inner shells (Si K, S K, Mo L, Mo K), which
also launch secondaries in these bins. It evaluates the Møller DCS at $E$ with
cap $E/2$ rather than $E+U$ and $(E+U)/2$, and it ignores the transport's
EEDL-substituted inner-shell rates and stopping closure. The residual ratio is
therefore consistent with the reference's approximations, not a test of the
launch spectrum to better than about 20%. Across bins the ratio varies by less
than about 15% in every case.

The threshold sweep is limited below by the 1 keV SBETHE table floor, which is
also the primary cutoff here; convergence is shown down to that floor, not
beyond it.

## Outstanding empirical checks

- Run the CUDA anchors (`tests/montecarlo/test_shell_secondary_cuda.py`) on a
  device and compare CPU/CUDA aggregate observables. The CUDA implementation
  was inspected here, but not executed by the verifier.
