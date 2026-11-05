# Explicit spectral rate normalization

## Claim and source

Claim `spectral-explicit-rate-normalization` covers explicit mean-current
transformations of scalar, directional, spatial and aperture photon spectra,
matching scalar acceptance, and paired energy-density conversion. It changes
no radiation scoring, response law, numerical default or checkpoint identity.

The [BIPM SI ampere](https://www.bipm.org/en/si-base-units/ampere) fixes
$e=1.602176634\times10^{-19}\,\mathrm C$ exactly, with
$\mathrm A=\mathrm C/\mathrm s$. The following derivation was supplied by a
fresh-context read-only verifier, who derived it from that source and the
intended quantities before inspecting implementation bodies.

## Independent derivation

For directional photon yield per incident electron,

$$
Y_E=\frac{dN_\gamma}{dN_e\,dE\,d\Omega},\qquad
\dot N_e=\frac{I}{e},\qquad I\ge0,
$$

linearity of expectation at fixed stored yield gives

$$
R_E=\frac{I}{e}Y_E,\qquad K_E=\frac{Y_E}{e}.
$$

$R_E$ has units photons/(s energy sr); $K_E$ has units
photons/(s A energy sr), with ampere as the canonical current denominator.
$K_E$ is defined directly, including at zero beam current; it is never
computed by dividing a rate by the current. At zero current finite supported
rate values vanish; unsupported NaNs and masks retain their meaning.
For a unit yield, 1 nA gives $6.241509074460763\times10^9$ photons/(s energy sr).

For pulsed beams, the established mean-current convention is $I=Q_b f_b$.
Independent incident electrons suffice for incoherent yields. For coherent
CXR, the stored yield already includes its resolved physical bunch state:
$Y_{\rm coh}(N_b)=N_{\gamma,b}/N_b$. Independently repeated bunches give

$$
f_bN_{\gamma,b}=f_bN_bY_{\rm coh}=\frac{I}{e}Y_{\rm coh}.
$$

This is a transformation at fixed stored bunch population and shape, such as
changing repetition rate. Changing bunch charge/shape can change coherent
radiation and requires a new evaluation; this API does not predict it.

For uniform directional source density over the matching case acceptance,

$$
Y_{E,\rm acc}=\Omega_{\rm case}Y_E.
$$

The user must acknowledge this assumption explicitly. For an observation
average formed from pixels with already-applied transmission $T_p$,

$$
\bar Y_E=\frac{\sum_p\Delta\Omega_p T_pY_{E,p}}{\Omega_{\rm pix}},\qquad
\Omega_{\rm pix}=\sum_p\Delta\Omega_p.
$$

Multiplying by that same $\Omega_{\rm pix}$ recovers accepted yield, without
another transmission or detector-response factor. It does not recover
intrinsic source spectra. Spatial/aperture spectra already include acceptance;
measured spatial spectra also include response. They receive only current
scaling. Registered acquisition counts remain exposure/realization-owned.
The scalar API requires positive finite matching solid angle; it refuses
missing geometry. The formal zero-acceptance limit is zero accepted yield,
but a zero-solid-angle observation average has no defined denominator.

For $E_{\rm eV}=1000E_{\rm keV}$, conservation requires

$$
Y_{\rm eV}\,dE_{\rm eV}=Y_{\rm keV}\,dE_{\rm keV},\qquad
Y_{\rm keV}=1000Y_{\rm eV}.
$$

Energy coordinates and bin edges convert together with reciprocal density;
node integrals and bin masses remain invariant. Scalar current/acceptance
factors commute with this Jacobian. Native grids, sampling and unsupported
windows remain distinct. Alternative coherent/incoherent modes are never summed.

## Implementation comparison and anchors

`results/_spectral_rate.py` validates the explicit current at the existing
scalar-quantity boundary. Its per-current coefficient is `1/elementary_charge`;
the per-second coefficient multiplies that by mean current in amperes.
The existing spectral carriers multiply their detached arrays once and retain
separate measure, provenance, acceptance, response and normalization state.
Repeated current transformations fail. `Result.accepted_spectra` uses the
matching case or pixel acceptance and no filter or response replay.

`tests/results/test_explicit_spectral_rates.py` uses the hard-coded exact SI
charge independently of the implementation helper. It covers equivalent units
and foreign registries, zero current, per-current output, invalid currents,
duplicate transforms, native-grid integration and bin masses, masks/NaNs,
matching observation acceptance (0.3 sr versus case 0.1 sr), preserved measured
response, legacy schemas and detached read-back ownership. Float tolerances
cover only arithmetic/conversion roundoff; no experimental uncertainty is
claimed. Existing quantity tests pin Pint-lazy numeric imports and snapshots.

This extends [directional rate normalization](directional-source-rate-normalization.md)
and preserves the separate [pixel acquisition counting claim](../ledger-detector-forward-models.md#pixel-acquisition-counting).
No new transport or detector-response validation is claimed. Human sign-off
remains in #277.
