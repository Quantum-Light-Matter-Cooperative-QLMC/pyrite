# issue-117 — coherent line-grid aliasing (working scratch)

Disposable branch-local scratch. Canonical plan: GitHub #117. Durable results
belong in `docs/validation/` + the ledger, not here.

## Milestone 1 — the coherent band limit (derived, numerically confirmed)

The coherent route builds, per (reflection, orientation) row, the complex field

```
A(E) = sum_j c_j * sinc(a_width_j (E - E_r,j) / pi) * exp[i phi_j(E)]
phi_j(E) = d_j * omega(E) - g.r_j - L_esc,j * delta_omega(E)
```

with `omega(E) = E / HBARC_EV_ANG`, `d_j = t_mid,j - n_hat . r_j` [Ang, c=1]
(`lines/_setup.py:403,442`), and the spectrum is `|A(E)|^2`
(`lines/_per_hkl.py:_accumulate_reflection_coherent`).

Squaring produces cross terms `j,k` carrying `exp[i(phi_j - phi_k)]`. Their
oscillation rate in `E` is the **phase-slope difference**

```
s_j = dphi_j/dE = d_j / HBARC_EV_ANG - L_esc,j * d(delta_omega)/dE
```

so the narrowest coherent fringe has period `2 pi / max_{j,k}|s_j - s_k|` and the
Nyquist step is

```
h_coh = pi / (s_max - s_min)  ~=  pi * HBARC_EV_ANG / D_span,
D_span = max_j d_j - min_j d_j
```

The in-medium term is negligible here: `1 - n_re ~ 1e-5` for x-rays, so
`L_esc * (1-n_re) << d`.

### Why this is not the incoherent width

Along one segment the increment of `d` is exactly
`t_L (1 - beta v.n) = dnm * t_L`, which is precisely the quantity setting the
incoherent first zero `2 pi HBARC_EV_ANG / (dnm t_L)` in
`diagnostics.py:sinc_feature_spacing`. So:

- incoherent Nyquist <- **per-segment** `d` increment
- coherent Nyquist   <- **total** `d` span across everything summed coherently

The ratio is therefore of order the coherently-summed path extent divided by one
segment's contribution, i.e. it grows with segment count. The incoherent
estimator cannot bound it in principle, not merely in calibration.

### Measured (`scratchpad/dspan.py`, Ne=200, seed 7, tilt 5 deg)

| case | sinc est [eV] | D_span all-e [Ang] | h_coh all-e [eV] | h_coh per-e [eV] | ratio all-e / per-e |
|---|---|---|---|---|---|
| hopg 30 keV  | 0.974 | 2.53e7 | 2.45e-4 | 1.49e-2 | 3975x / 65x |
| hopg 100 keV | 0.481 | 2.65e7 | 2.34e-4 | 2.81e-3 | 2055x / 171x |
| wse2 30 keV  | 7.85  | 2.50e7 | 2.48e-4 | 3.41e-2 | 3.2e4x / 230x |

The all-electron `D_span` is ~2.5e7 Ang ~ 2.5 mm in every case and is set by the
transverse beam extent projected on `n_hat` (`beam_fwhm_mm = 1.0`), not by the
material or the beam energy. The per-electron span is the decoherence-grouped
floor and does vary with case.

## Milestone 1b — decoherence changes which span applies

`decoherence_active` (`lines/_setup.py:447`) is **True** for the standard ladder
cases, because `beam_fwhm_mm = 1.0` gives nonzero transverse offsets. The
spectrum is then the blend
`(1-F) * sum_e |S_e|^2 + F * |sum_e S_e|^2` on the offset-free geometric phase.

Consequences:

- The `sum_e |S_e|^2` floor needs only the **per-electron** span (65-230x finer
  than the incoherent estimate).
- The `|sum_e S_e|^2` term needs the **all-electron** span (2000-32000x finer),
  but those inter-electron fringes are one realization of the sampled offsets --
  speckle, which `_setup.py` itself notes "does not shrink with electron count".
- So the open question is NOT only "how fine", it is **whether the all-electron
  fringes are an observable to resolve or a nuisance to average**. If they must
  be averaged, the correct rule is a bias/averaging statement, not a Nyquist
  step: no affordable uniform grid resolves 2.4e-4 eV over a keV-scale band
  (~1e7 points).

Standard ladder cases carry no `bunch_length_fs`, so `coherent=True` currently
**refuses** on them (`longitudinal_rms_fs` required once decoherence is active).
The ladder harness must set it; it is a real modelling input (cf. #57).

## Milestone 2 — harness switch (done)

`CaseLadder(..., coherent=True)` added in `energy_grid/convergence_case.py`;
`lines()` follows `self.coherent`. One injected transport serves both routes, so
both ladders run on identical trajectories.

## Milestone 3 — ladder (in progress)

Local CPU ladder is too slow past coarse rungs (coherent cost is
O(n_seg * n_points)); confirms the issue's "heavy ladders via `pyrite remote`".

## Open decision for the issue owner

Whether the default all-electron coherent sum is physically intended at
`beam_fwhm_mm = 1.0`. If yes, automatic resolution must refuse `coherent=True`
rather than silently alias, since the required step is unaffordable. That is the
acceptance fork in #117.
