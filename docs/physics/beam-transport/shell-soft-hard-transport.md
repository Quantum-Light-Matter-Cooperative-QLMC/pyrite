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
- Secondaries and vacancies are not transported (#94). Each hard row records its transfer `hard_W_keV` and channel `hard_channel` ($3\times$oscillator + branch). `shell_transport.hard_event_energy_accounting` splits $W$ into local deposit, emitted secondary energy and reserved inner-shell binding, following the host sampler's rules. The primary's energy balance closes exactly; the secondary's energy leaves the primary without being deposited or radiated further.
- It runs on the lockstep and per-electron CPU cores (exact and LUT) and on the exact CUDA core. The CUDA LUT core and grooved transport raise; production runs on CUDA disable the LUT to reach the exact kernel.

## Outputs

The result adds per-row `hard_W_keV` (0 on rows without a collision) and `hard_channel` (−1 without a collision), plus an `inelastic` metadata dict with the cutoff, the layer materials and each layer's channel table. `stopping_tables` on the result are the soft tables, so post-transport cutoff clips use the continuous rule the transport ran with. A clip also marks a hard row `CUTOFF` when the collision drops the primary below the consumer's floor. Rows ending in a hard collision carry `EVENT_HARD_INELASTIC` and satisfy `check_segment_event_contract`: they close the flight, energy drops by exactly `hard_W_keV`, and position and clock stay continuous.

## RNG layout

Hard draws use a counter-addressed SplitMix64 stream keyed by the electron's stream key XOR a dedicated salt, re-hashed. It is disjoint from the free-path/scattering stream and from the straggling key domain. The per-electron counter advances by one per flight for $\tau_h$ and by four per hard event (channel, $W$, recoil, azimuth). Soft fluctuations use the existing per-(flight, substep) straggling keys. The lockstep cores keep their shared `Generator` for elastic draws only. With $W_c$ above every channel endpoint the mode is bitwise the continuous transport; the regression test checks this.

## Configuration

Profiles and `Numerics` carry `inelastic_model` and `inelastic_cutoff_eV`; `pyrite profile numerics set --inelastic-model shell-soft-hard --inelastic-cutoff-ev 50 --energy-model midpoint` sets them. Cases carry them only when the mode is on, so dataset identities and case content keys of continuous runs are unchanged.

`Validation: shell-soft-hard-transport` (unverified). See [shell soft/hard transport validation](../../validation/beam-transport/shell-soft-hard-transport.md).
