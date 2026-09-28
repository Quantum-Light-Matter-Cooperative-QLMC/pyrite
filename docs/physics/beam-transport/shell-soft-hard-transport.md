# Shell soft/hard inelastic transport (opt-in)

The default transport removes the whole corrected collision stopping continuously (see [Stopping power](stopping-power.md)). `simulate_trajectories(inelastic_model="shell-soft-hard")` instead follows PENELOPE's mixed scheme{cite:p}`salvat2024penelope`: energy losses $W\le W_c$ stay continuous, and losses $W>W_c$ become discrete hard collisions that lower the primary energy by $W$ and deflect it by the sampled recoil. Both parts come from one stopping-closed shell generalized-oscillator-strength (GOS) model, so neither is added on top of the other. The mode is opt-in and requires `energy_model="midpoint"`. With the default `inelastic_model="continuous"`, transport is bit-for-bit unchanged.

## Model

For each layer the host builds the PENELOPE-2024 shell oscillators, their distant (longitudinal and transverse) and close Møller moments, EEDL-substituted inner-shell rates, and one outer-shell factor $\mathcal N(E)$ that closes the first moment to the corrected SBETHE `stp.dat` stopping $S(E)$ (see the validation records `penelope-shell-rate-closure` and `penelope-shell-soft-hard-partition`). Split at $W_c$, that closure gives soft moments $\sigma_s^{(1)},\sigma_s^{(2)}$ and hard zeroth and first moments $\sigma_{h,k}^{(0)},\sigma_h^{(1)}$ for each hard channel $k$ (oscillator × branch), with $\sigma_s^{(1)}+\sigma_h^{(1)}=\sigma^{(1)}$.

At every node $E_i$ of the layer's SBETHE table, with the transport's own stopping $S(E_i)$:

```{math}
:label: eq-shell-soft-hard-tables

S_s(E_i)=S(E_i)\,\frac{\sigma_s^{(1)}}{\sigma_s^{(1)}+\sigma_h^{(1)}},\qquad
\mu_k(E_i)=S(E_i)\,\frac{\sigma_{h,k}^{(0)}}{\sigma_s^{(1)}+\sigma_h^{(1)}},\qquad
\Omega_s^2(E_i)=S(E_i)\,\frac{\sigma_s^{(2)}}{\sigma_s^{(1)}+\sigma_h^{(1)}} .
```

$S_s$ (keV Å$^{-1}$) is interpolated log-log like the full table; the hard rates $\mu_k$ (Å$^{-1}$) and the soft straggling parameter $\Omega_s^2$ (keV² Å$^{-1}$) are interpolated linearly in $\ln E$. Because only ratios enter, soft plus mean hard loss per path equals the transport's own $S$ at every node, whatever number density the table uses.

**Scheduling.** Each physical flight draws an elastic optical depth, as before, and an independent hard optical depth $\tau_h=-\ln u$. Its step is the shorter of $\tau_{\rm el}\lambda_{\rm el}$ and $\tau_h/\mu_h$ with $\mu_h=\sum_k\mu_k$, then truncated by geometry, the cutoff and the `max_dE_frac` cap as usual. Both optical depths are consumed across numerical substeps at each substep's start-energy hazard and redrawn after every flight-closing event. Two independent Poisson processes with piecewise-constant hazards are exactly their superposition, so this is equivalent to sampling the combined rate and selecting the type.

**Hard collision.** At a hard event the channel is chosen with probability $\mu_k/\mu_h$. $W$ is drawn by inverting that channel's restricted loss CDF on $W>W_c$ (bound distant $p_{\rm dis}(W)/W$, close $F^{(-)}(E+U,W)/W^2$), and the primary polar angle follows PENELOPE Eqs. 3.126–3.129 (longitudinal), 3.134 (close) or no deflection (transverse), with uniform azimuth. These are scalar twins of the host sampler in `shell_sampling.py`. The collision uses the row's start energy, where its hazard was evaluated. The primary continues at $E_{\rm end}-W$; if that is at or below its cutoff, the row becomes a terminal `CUTOFF` row at the collision point.

**Soft losses.** Without straggling, the row loses $\int S_s\,ds$ under the midpoint rule. With `straggling=True`, it loses a sample from PENELOPE's two-moment distribution (Eqs. 4.54–4.63) with mean $S_s(E)s$ and variance $\Omega_s^2(E)s$: a truncated Gaussian, uniform, or delta-plus-uniform law depending on $\langle\omega\rangle^2/{\rm var}(\omega)$. The Urban sampler is not used in this mode, because it models the unrestricted loss spectrum and would count the explicit hard tail twice.

## Requirements and limits

- $W_c$ must be strictly above every layer's conduction-band resonance $W_{cb}$ (Si 16.7, SiO₂ 22, MoS₂ 23 eV). The closed model's total inelastic mean free path is 26–35% below independent full Penn calculations. That excess sits in the conduction-band distant loss at exactly $W_{cb}$, which is soft for $W_c>W_{cb}$; below that cutoff it would drive hard events, so it is rejected.
- Each layer needs a catalog key with a packaged conduction band (currently `silicon`, `sio2`, `mos2`), and the transport table must share that key's SBETHE energy nodes.
- The hard and soft DCS are frozen at the row's start energy; PENELOPE's Eq. 4.65 energy-dependence correction is not applied. `max_dE_frac` bounds the resulting error.
- Soft-fraction tables inherit the kinks that EEDL node interpolation leaves in $\mathcal N(E)$. $S_s/S$ has a 1–3% sawtooth between those nodes, so the transport LUT may report a stopping interpolation error of about $2\times10^{-4}$.
- Each hard row records its transfer `hard_W_keV` and channel `hard_channel` ($3\times$oscillator + branch). `shell_transport.hard_event_energy_accounting` splits $W$ into local deposit, emitted secondary energy and reserved inner-shell binding, following the host sampler's rules. Without `secondary_threshold_eV` secondaries are not transported: the primary's energy balance closes exactly, and the secondary's energy leaves the primary without being deposited or radiated further. With it, secondaries above the threshold are transported as described in [Secondary electron transport](#secondary-electron-transport). Vacancies are never transported (#91).
- It runs on the lockstep and per-electron CPU cores (exact and LUT) and on the exact CUDA core. The CUDA LUT core and grooved transport raise; production runs on CUDA disable the LUT to reach the exact kernel.

## Secondary electron transport

`simulate_trajectories(secondary_threshold_eV=T_s)` transports the electrons that hard collisions emit. It needs `inelastic_model="shell-soft-hard"`, the only mode with hard collisions, and is rejected otherwise. The default `None` transports no secondaries, and the run is then bit-for-bit the primary-only transport: no launch field is allocated and no random number is drawn.

**One threshold.** $T_s$ is both the production cut and the tracking cutoff of every secondary. A hard row with transfer $W$ gives the secondary kinetic energy $T=W-U_k$ for an EEDL-substituted inner shell and $T=W$ for an outer shell or the conduction band, following the host sampler (`penelope-shell-hard-loss-sampling`). The secondary is launched when $T>T_s$. Otherwise $T$ is deposited at the collision point, as `hard_event_energy_accounting` already does. Every launched electron is tracked until it leaves the target or its energy reaches $T_s$, which replaces `E_cut_keV` for all secondaries. Primaries keep their own `E_cut_keV`. Launched secondaries emit their own secondaries under the same rule.

**Launch state.** A secondary starts at the end point of its parent's hard row, $\mathbf r_{\rm mid}+\tfrac12L\,\hat{\mathbf v}$. It takes the row's layer, the clock `t_end_ang` of the collision, and its primary's bunch offset `t0_ang`. Its direction is the PENELOPE-2024 §3.2.5.4 emission direction of `penelope-shell-secondary-direction`, rotated about the parent's pre-collision direction $\hat{\mathbf v}$ in the same frame as the primary's recoil. Its polar cosine comes from the momentum transfer (longitudinal: modified resonance $W'_k$ and sampled recoil $Q$; close: $Q=W$; transverse: fixed cosine 0.5), and its azimuth is the primary's plus $\pi$. The core evaluates this from the four hard draws it already makes, so it adds no draw. A collision that absorbs the primary (a terminal `CUTOFF` row) still emits its secondary.

**Identity.** Every trajectory is a track.

- `electron_id` names the primary history. Every track carries its primary's index in $[0,N_e)$, so `electron_limit` selections, per-primary normalization by `Ne`, and the bunch offset act on whole showers.
- `generation` is 0 for primaries and $g+1$ for a secondary of a generation-$g$ track.
- `track_id` is unique per track. Primaries use their electron index. The tracks of generation $g+1$ are numbered after all earlier tracks, in order of (parent `track_id`, parent hard ordinal).
- The parent hard ordinal $k$ counts the parent's rows with `hard_channel >= 0` in flight order from 0, including an absorbing `CUTOFF` row.
- `parent_id` is the parent's `track_id`, and $-1$ for primaries.

**Generation batching.** The primaries run first, on the core the run resolves. Their launched secondaries form generation 1, and each generation is transported with the same configuration, tables and cores from its explicit launch state; beam sampling is skipped. Its launched secondaries form the next generation. The loop ends when a generation launches nothing. Each generation reuses the per-electron batching of `batching.py`, so accelerator memory is bounded per launch as for primaries. Since $W\le(E+U_k)/2$ on every hard channel, a secondary carries at most about half its parent's energy. The number of generations is therefore at most about $\log_2(E_0/T_s)+1$. `max_secondary_generations` (default 64) and `max_secondary_tracks` (default $1000\,N_e$) bound the loop. Exceeding either raises; nothing is truncated silently.

**Random numbers.** Generation 0 uses the streams of the primary-only run. A secondary's stream key is a SplitMix64 hash of (seed, parent `track_id`, parent hard ordinal) under a salt of its own, so it does not depend on batch size or on the order in which tracks are processed. The hard, soft-straggling and coupled-radiative streams derive from that key exactly as they derive from a primary's key. The lockstep core draws from one shared `Generator` and cannot address a stream per track. Secondaries therefore always run on a per-electron core: the CUDA core when generation 0 ran on CUDA, the per-electron CPU core otherwise. Photon directions of coupled hard radiative events are completed on the host for each generation, from a generation-keyed stream over that generation's deterministic row order.

**Energy balance.** Summed over all tracks of a run, with $E_0$ the initial energy of every primary that entered the target,

```{math}
:label: eq-secondary-energy-balance

\sum E_0 = E_{\rm escaped} + E_{\rm deposited} + E_{\rm binding} + E_{\rm radiated}.
```

- $E_{\rm escaped}$ is the kinetic energy at the exit row of every track that leaves the target through any face.
- $E_{\rm deposited}$ collects the continuous (soft) loss $E_{\rm start}-E_{\rm end}$ of every row, the residual energy of every track that ends at its cutoff, and the kinetic energy of every secondary at or below $T_s$. In the coupled radiative mode the continuous loss also carries the soft radiative share below `radiative_cutoff_eV`.
- $E_{\rm binding}$ is the inner-shell binding $U_k$ reserved at every explicit vacancy (`binding_reserved`). Vacancies stay bookkeeping-only until #91; see below.
- $E_{\rm radiated}$ is the energy of hard photons in the coupled `bremslib-soft-hard` mode. With the uncoupled model, radiation is scored after transport and debits no electron energy.

The identity also holds per primary history. Launched secondary energy appears in no term directly: it is the next generation's initial energy. $E_{\rm binding}$ is not small: with $W_c=50$ eV the L shells of Si are explicit vacancies, and reserved binding is about 12% of the incident energy of a 20 keV beam stopped in Si. A dose tally that omits it undercounts local deposition until #91 relaxes the vacancies; `checks/shell_secondary_transport_observables.py` deposits it at the collision point.

**Particle balance.** Each track ends exactly once, by backscatter, transmission, side exit or cutoff. The number of tracks in generation $g+1$ equals the number of generation-$g$ hard rows with $T>T_s$.

**Radiation.** Secondary rows are ordinary transport rows. The track-length bremsstrahlung and characteristic estimators and the coupled hard-photon scoring therefore score them with no change. Characteristic emission keeps the track-length estimator as its only source: explicit inner-shell vacancies at hard collisions emit no photon, and their binding counts as $E_{\rm binding}$. A secondary's segments ionize inner shells along its own path. Scoring them is not a double count: each track's estimator covers only that track's path. Consumers keep their own validity floors. Characteristic scoring clips every track at 1 keV, so a lower $T_s$ adds no characteristic path. Coherent spectra sum phases within one `electron_id`, which would make a whole shower one emitter. They are not supported with secondary transport and raise.

**Coverage at the threshold.** $T_s$ must lie inside every table the secondaries use. It is checked against each floor like a primary cutoff, and an out-of-coverage value raises; nothing clamps or falls back.

| model | coverage | at $T_s$ |
| --- | --- | --- |
| SBETHE soft stopping and shell soft/hard tables | catalog nodes, 1 keV–1 GeV | range check; sets the floor $T_s\ge1$ keV |
| ELSEPA elastic (`elastic_model="elsepa"`) | 100 eV–100 MeV | `check_elsepa_coverage` |
| Mott (Browning fit) | stated 0.1–30 keV | below the SBETHE floor, so never reached |
| screened Rutherford | analytic | no table |
| BremsLib (`bremslib-soft-hard`) | per-element SDCS tables | `radiative_cutoff_eV` $\le T_s$ and table range checks |

**Supported paths.** The lockstep, per-electron (exact and LUT) CPU cores and the exact CUDA core. Grooved transport and the CUDA LUT core already reject the shell mode. `collect_diagnostics` is rejected with secondaries, because its per-electron summaries assume one track per electron.

`Validation: shell-secondary-transport` (rederived). See [secondary transport validation](../../validation/beam-transport/shell-secondary-transport.md).

## Outputs

The result adds per-row `hard_W_keV` (0 on rows without a collision) and `hard_channel` (−1 without a collision), plus an `inelastic` metadata dict with the cutoff, the layer materials and each layer's channel table. `stopping_tables` on the result are the soft tables, so post-transport cutoff clips use the continuous rule the transport ran with. A clip also marks a hard row `CUTOFF` when the collision drops the primary below the consumer's floor. Rows ending in a hard collision carry `EVENT_HARD_INELASTIC` and satisfy `check_segment_event_contract`: they close the flight, energy drops by exactly `hard_W_keV`, and position and clock stay continuous.

## RNG layout

Hard draws use a counter-addressed SplitMix64 stream keyed by the electron's stream key XOR a dedicated salt, re-hashed. It is disjoint from the free-path/scattering stream and from the straggling key domain. The per-electron counter advances by one per flight for $\tau_h$ and by four per hard event (channel, $W$, recoil, azimuth). Soft fluctuations use the existing per-(flight, substep) straggling keys. The lockstep cores keep their shared `Generator` for elastic draws only. With $W_c$ above every channel endpoint the mode is bitwise the continuous transport; the regression test checks this.

## Configuration

Profiles and `Numerics` carry `inelastic_model` and `inelastic_cutoff_eV`; `pyrite profile numerics set --inelastic-model shell-soft-hard --inelastic-cutoff-ev 50 --energy-model midpoint` sets them. Cases carry them only when the mode is on, so dataset identities and case content keys of continuous runs are unchanged.

`Validation: shell-soft-hard-transport` (unverified). See [shell soft/hard transport validation](../../validation/beam-transport/shell-soft-hard-transport.md).
