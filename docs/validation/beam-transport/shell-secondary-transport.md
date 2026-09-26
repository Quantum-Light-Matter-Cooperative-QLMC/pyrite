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

## Outstanding empirical checks

- Run `checks/shell_secondary_transport_observables.py` remotely at full size
  and assess threshold convergence of backscatter, transmission, depth dose,
  characteristic line yields, and bremsstrahlung. The same run compares the
  cascade launch spectrum with the Møller prediction.
- Check energy balance separately for each primary history; the current fast
  anchor tests the aggregate identity.
- Run the CUDA anchors on a device and compare CPU/CUDA aggregate observables.
  The CUDA implementation was inspected here, but not executed by the verifier.
