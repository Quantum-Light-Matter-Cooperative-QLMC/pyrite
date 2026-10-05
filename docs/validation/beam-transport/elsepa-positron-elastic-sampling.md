# ELSEPA positron elastic sampling

Validation: `elsepa-positron-elastic-sampling`. Independent verdict:
rederived (the existing tests also anchor it). Human sign-off remains
pending (#277).

## Source and intended quantity

ELSEPA 2020 (`vendor/xsgen/elsepa/elscata.f`) selects the projectile with
`IELEC` ($-1$ electron, $+1$ positron; line 68). For positrons it forces
`MEXCH=0` (no exchange, line 363). `MCPOL=2` selects the LDA
correlation–polarization potential (line 70). Above 10 keV ELSEPA turns
correlation–polarization off for either species (lines 441–447). For the
muffin-tin LDA-II absorption default gap, ELSEPA uses
$\max(0,E_{\rm ion}-6.8\ {\rm eV})$ for positrons (line 398) and the first
excitation energy for electrons.

The sampler is the one of `elsepa-elastic-sampling`. Given a node's
tabulated DCS on the $\mu=(1-\cos\theta)/2$ grid, it inverts the
piecewise-linear CDF. Nothing in it depends on the projectile, so the only
positron-specific requirement is that positron tracks receive positron
tables.

## Checks

- **Deck.** `ElsepaDeck(projectile="positron")` writes `IELEC 1`, defaults
  `exchange_model=0` and `polarization_model=2`, and rejects a nonzero
  exchange model. Electron decks keep `IELEC -1`, `MEXCH 1`, `MCPOL 0` and
  their historical keys. The muffin-tin constructor forwards `projectile`
  and leaves the gap at ELSEPA's species default.
- **Installed tables.** The W free-atom positron table manifest records
  `projectile: positron`, `exchange_model: 0`, `polarization_model: 2`.
  At 1 keV the positron/electron ratios are $\sigma_1$ 0.337 and $\sigma$
  0.845 for W, and $\sigma_1$ 0.503 for Si. At 10 MeV, $\sigma_1$ is 0.767
  for W and 0.953 for Si. This is the expected sign of the charge effect:
  nuclear repulsion reduces large-angle positron scattering, more so for
  heavy $Z$, and the Mott ratio stays below 1 at high energy for large
  $Z\alpha$.
- **Routing.** `resolve_layer_tables`/`resolve_stack_tables` pass `projectile`
  to both the free-atom and the muffin-tin requests; electron and positron
  keys are disjoint. `secondaries._positron_overrides` puts the positron
  `elastic_tables` into positron steps only.
- **Sampler.** `tests/montecarlo/test_elsepa_positron_elastic.py` (Si and W
  at nodes near 1 keV, 100 keV and 10 MeV) passes against the installed
  positron tables. The KS distance of 50 000 draws is below the 99% critical
  value, the first moment is within 1% of ELSEPA's $\sigma_1/\sigma$, and W
  1 keV $\sigma_1^{(+)}<0.8\,\sigma_1^{(-)}$. `tests/xsgen/test_positron_tables.py`
  passes.

## Verdict

Matches: the positron deck reproduces ELSEPA's positron input contract, and
positron tracks sample the unchanged inverse-CDF sampler from positron
tables only. Recommended status: `anchored`.
