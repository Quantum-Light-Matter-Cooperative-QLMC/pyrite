# Positron bremsstrahlung scaling $F_p(Z,E)$

Validation: `positron-brems-scaling`. Independent verdict: rederived (the
existing test also anchors it). Human sign-off remains pending (#277).

## Source and intended quantity

PENELOPE-2024 Eq. 3.153 takes the positron radiative DCS as the electron
one times a factor that does not depend on the photon energy:

$$
\frac{d\sigma^{(+)}_{\rm br}}{dW}=F_p(Z,E)\,\frac{d\sigma^{(-)}_{\rm br}}{dW},
$$

where $F_p$ is Kim et al.'s (1986) positron/electron radiative stopping
ratio, fitted to about 0.5% by (Eqs. 3.154–3.155)

$$
F_p=1-\exp\!\left(-1.2359\times10^{-1}t+6.1274\times10^{-2}t^2-3.1516\times10^{-2}t^3
+7.7446\times10^{-3}t^4-1.0595\times10^{-3}t^5+7.0568\times10^{-5}t^6-1.8080\times10^{-6}t^7\right),
$$

$$
t=\ln\!\left(1+\frac{10^6E}{Z^2m_ec^2}\right).
$$

$t$ is dimensionless because $E$ and $m_ec^2$ share units. SBETHE
`EBRSTP` (`vendor/xsgen/sbethe/sbethe.f`, lines 2850–2855, `IPROJ=2`)
evaluates the same polynomial in nested form,
$-T(a_1-T(a_2-T(a_3-T(a_4-T(a_5-T(a_6-Ta_7))))))$ with positive $a_j$. That
expands to the alternating signs above. SBETHE uses
`REV`$=510998.95$ eV and, for a compound, $Z^2\to Z_{\rm eq}^2=\sum_i
s_iZ_i^2/\sum_is_i$ (`ZBR2`, line 2793).

## Independent checks

- **Limits.** $t\to0$ as $E\to0$, so $F_p\to1-e^{0}=0$. Over
  $Z=1$–99 and 10 eV–1 GeV my transcription gives
  $2.5\times10^{-4}\le F_p\le1$. $F_p$ increases monotonically in $E$ and
  decreases in $Z$ on the whole grid. $F_p\to1$ at high $E$, e.g.
  $F_p(14,10\ {\rm MeV})=0.97705$ and $F_p(74,1\ {\rm GeV})=0.99441$.
- **Transcription.** An independent Python transcription of SBETHE's nested
  form differs from `hard_radiative.py::positron_brems_factor` by at most
  $1.4\times10^{-15}$ (absolute) for $Z=1$–99 and $E=10$ eV–1 GeV. The
  code's tuple $(a_1..a_7)$ and its Horner loop expand to
  $\sum_{j=1}^7a_jt^j$ with the signs of Eq. 3.154, and `_ELECTRON_REST_EV`
  $=510998.95$ eV matches `REV`.
- **Application.** `positron_bremslib_tables` multiplies each element's
  `scaled_sdcs_mb[i, :]` and `scaled_ddcs_mb_sr[i, :, :]` by
  $F_p(Z,T_{1,i})$. $T_1$ is the incident kinetic energy (the
  `incident_energy_keV` docstring), converted to eV. The table has no other
  integrated field, so the radiative stopping, straggling, hard rate and
  photon/angle shapes derived from these arrays are all $F_p$-scaled
  consistently. The `+positron-fp` key suffix keeps run identity distinct.
- **Tests.** `tests/montecarlo/test_positron_brems.py` passes.

## Documented deviations (accepted, not discrepancies)

- **Per element.** PyRITE applies $F_p(Z_i,E)$ to each element's DCS, while
  SBETHE (and PENELOPE's molecular DCS, Eqs. 3.150–3.152) use a single
  $F_p(Z_{\rm eq},E)$. The two agree for elemental targets. For compounds the
  per-element choice is arguably the better physics, but it does not
  reproduce SBETHE's positron radiative stopping exactly.
- **Node scaling.** $F_p$ is applied at the $T_1$ nodes, and the transport
  interpolates the scaled product. Between nodes this differs from
  $F_p(E)\times$ interpolated electron DCS by a second-order interpolation
  error, well below the fit's 0.5%.

## Verdict

Matches. Recommended status: `anchored` (`test_positron_brems.py`).
