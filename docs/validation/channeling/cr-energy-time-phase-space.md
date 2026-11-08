# `cr-energy-time-phase-space` — correlated classical entrance phase space and linear chirp

- **Claim**: `cr-energy-time-phase-space` — `montecarlo/channeling/energy_time.py::{LinearEnergyChirp,IncidentPhaseSpace.covariance,IncidentPhaseSpace.with_linear_chirp,IncidentPhaseSpace.from_gdf}` — additive linear chirp of epic #620 Stage III, $T(t_0) = T_0 + h(t_0 - \bar t_0) + \delta T_u$, and bilinearity of covariance.
- **Verifier**: fresh context (issue #628 verification); derivation written from the ledger row, design note and docstrings before the function bodies were read.

## Independent derivation

Records $i = 1..N$ carry $(t_i, T_i^{(0)}, x_i, y_i, x'_i, y'_i)$ with normalized weights $w_i$, $\sum_i w_i = 1$. Weighted population moments are

$$
\langle q\rangle = \sum_i w_i q_i, \qquad
C_{ab} = \sum_i w_i (q_{ia} - \langle q_a\rangle)(q_{ib} - \langle q_b\rangle),
$$

with no Bessel correction, because the records are the distribution, not a sample from it.

The chirp is $T_i = T_i^{(0)} + h(t_i - t_{\rm ref})$. Then

$$
\langle T\rangle = \langle T^{(0)}\rangle + h(\langle t\rangle - t_{\rm ref}),
$$

so the centroid is unchanged exactly when $t_{\rm ref} = \langle t\rangle$. The constant $h\,t_{\rm ref}$ drops out of every central moment. Bilinearity gives, for **any** $t_{\rm ref}$,

$$
\operatorname{Cov}(t,T) = \operatorname{Cov}(t,T^{(0)}) + h\operatorname{Var}t,
\qquad
\operatorname{Var}T = \operatorname{Var}T^{(0)} + 2h\operatorname{Cov}(t,T^{(0)}) + h^2\operatorname{Var}t .
$$

These are algebraic identities on the records, so they hold to rounding. The regression slope $\operatorname{Cov}(t,T)/\operatorname{Var}t = h + \operatorname{Cov}(t,T^{(0)})/\operatorname{Var}t$ equals $h$ exactly only when the *sample* $\operatorname{Cov}(t,T^{(0)})$ vanishes; for independent analytic draws that holds in expectation.

Sign: $t$ larger means later, at the bunch tail. With $h > 0$ the tail is more energetic. $h = 0$ is the identity.

Units and adapters:

- $c = 2.99792458\times10^{8}\ {\rm m/s} = 2.99792458\times10^{18}\ \text{Å/s} = 2997.92458\ \text{Å/fs}$, so the transport clock gives $t\,[\rm fs] = (ct)[\text{Å}]/2997.92458$.
- Energies convert as $T\,[{\rm eV}] = 10^3\,T\,[{\rm keV}]$.
- Slopes are $x' = u_x/u_z$ and $y' = u_y/u_z$, which needs $u_z > 0$.

Whole-record resampling draws index $i$ with probability $w_i$. Every joint moment of the resampled set is therefore an unbiased estimate of the weighted moment, but it is not exact for finite $n$.

## Source-to-code diff

| Item | Derivation | Code | Result |
|---|---|---|---|
| $c$ | $2997.92458$ Å/fs | `C_ANG_PER_FS = _C_M_PER_S * 1e-5` (energy_time.py:36) $= 2997.92458$ | match |
| covariance | $\sum_i w_i \delta q_a \delta q_b$, population | `(centred * w[:, None]).T @ centred` (:186-187), weights normalized at :148 | match |
| chirp | $T^{(0)} + h(t - t_{\rm ref})$, default $t_{\rm ref} = \langle t\rangle$ | :229-234 | match |
| $T \le 0$ | refuse | raises (:235-236) | match |
| other coordinates | unchanged | `_replace(kinetic_energy_eV=...)` only (:239) | match |
| clock adapter | $t = (ct)/c$ | `arrival_ang / C_ANG_PER_FS` (:294) | match |
| slopes | $u_{x,y}/u_z$, $u_z > 0$ | `direction[:, :2] / direction[:, 2:3]`, $u_z > 0$ checked (:291, :297) | match |
| analytic sampler | entrance arrival offset $t_0$, not start clock | unpacks `(pos, dirs, E_keV, _clock, t0_ang, _)` (:412); `sample_primary_entry` returns `clock, t0` with face-arrival delay folded into `t0` (beam_entry.py:416-418) | match |
| GDF | whole records, crossing times as $c\,t$ Å | `beam.sample(...)` → `from_entry_arrays` (:324-334) | match |
| resample | inverse CDF over $w$ | `searchsorted(cdf, u*cdf[-1], 'right')` (:255-256) | match |

No term diverges.

## Numerical checks

Script: the verifier's disposable `check_phase_space.py` (not committed), run with the project runner.

- $N = 4000$ weighted records with a base $T^{(0)}$ deliberately correlated with $t$ (slope 1.5 eV/fs):
  - covariance matrix against an independent weighted formula: max relative error $4.4\times10^{-16}$;
  - $h = 5$ eV/fs with default $t_{\rm ref}$: Cov identity $5\times10^{-15}$, Var identity $2.5\times10^{-15}$, centroid shift $0$;
  - $h = -3$ eV/fs with $t_{\rm ref} = 123$ fs: identities $\le 2.3\times10^{-14}$, centroid shift $377.6$ eV, equal to $h(\langle t\rangle - t_{\rm ref})$;
  - $h = 0$: bitwise identity;
  - other coordinates and weights bitwise unchanged.
- Nonpositive chirp ($h = 10^6$ eV/fs) is refused.
- Adapter round trip ($ct$ Å, unit directions, keV): $|\Delta t| \le 1.1\times10^{-13}$ fs, $|\Delta x'| \le 5\times10^{-20}$, $\Delta T = 0$.
- Anchor `tests/montecarlo/channeling/test_channeling_energy_time.py` is green. It covers the exact transport sampler, the GDF screen fixture and the JSON round trip; all 19 tests of the three anchor modules pass.

## Verdict

- **Filters**: units pass; limits pass ($h = 0$ identity, $T \le 0$ refused, centroid kept at $t_{\rm ref} = \langle t\rangle$); signs/conventions pass (larger $t$ is the tail; $h > 0$ makes the tail more energetic).
- **Re-derivation**: `matches`.
- **Verdict**: `rederived`.

## Findings (non-blocking)

1. *Wording.* The limiting case "uncorrelated base records give a $t$–$T$ regression slope $h$" holds in expectation, not exactly for the records. The sample $\operatorname{Cov}(t,T^{(0)})$ is nonzero. Suggest "in expectation".
2. *Convention guard.* `sample_incident_phase_space` forwards `beam_dir`, `tilt_polar_rad` and `tilt_azim_rad`. With those, `from_entry_arrays` gives slopes about the slab normal with a nonzero mean, and $z$ is no longer the mean beam axis. The design note states that convention, and `cr-retarded-photon-arrival` relies on it. The axial MCWF path is immune because it subtracts $\langle x'\rangle$. The retarded timing is not. Suggest documenting this or refusing it.
3. *Provenance.* An existing GDF correlation is not detected as "already chirped" (only the provenance key is checked). That is by design but worth one sentence.

## Resolution (implementer, 2026-11-07)

1. The ledger limiting case now says the regression slope is `h` *in expectation*.
2. `sample_incident_phase_space` and the design note state that the retarded timing needs the slab `z` to be the mean beam axis (no `beam_dir`/tilt kwargs); the axial MCWF path subtracts the centroid slope and is unaffected.
3. `with_linear_chirp` documents that only an explicit earlier chirp is refused; a correlation already in imported records adds.
