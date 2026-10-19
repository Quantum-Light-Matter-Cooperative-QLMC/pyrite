# Directional source rate normalization

## Claim and source

Claim `directional-source-rate-normalization` concerns
`src/pyrite/results/directional.py::DirectionalSpectrum.rate(current_na=None)`.
The input density is photons per incident electron per eV per sr; the output
is photons per second per eV per sr at a nonnegative mean electron current
in nA. The default current comes from the resolved beam normalization.

The [BIPM definition of the ampere](https://www.bipm.org/en/si-base-units/ampere)
fixes the elementary charge exactly at
$e = 1.602176634\times10^{-19}\,\mathrm{C}$ and identifies one ampere
with $1/e$ elementary charges per second. This is a unit conversion and
independent-particle expectation argument, not a new radiation model.

## Independent derivation

This derivation was recorded from the source, intended quantity and method
signature/docstring before inspecting the implementation body.

Let the mean yield per incident electron at photon energy $E$ and direction
$\hat{\mathbf n}$ be

$$
Y(E,\hat{\mathbf n}) = \frac{d^2\langle N_\gamma\rangle}
{dE\,d\Omega\,N_{\mathrm{incident}}}.
$$

For independent incident electrons, linearity of expectation gives
the same per-electron mean for every primary. A nonnegative electron current
magnitude $I$ corresponds to incident electron rate

$$
\dot N_{\mathrm{incident}} = \frac{I}{e}.
$$

Writing the configured numerical current in nA as $I_{\mathrm{nA}}$ gives

$$
\boxed{R(E,\hat{\mathbf n}) = Y(E,\hat{\mathbf n})
\frac{I_{\mathrm{nA}}\,10^{-9}}{1.602176634\times10^{-19}}.}
$$

For bunch charge magnitude $Q_{\mathrm{pC}}$ in pC and repetition rate
$f_{\mathrm{kHz}}$ in kHz, the time-averaged current in nA is

$$
I_{\mathrm{nA}} = Q_{\mathrm{pC}} f_{\mathrm{kHz}},
$$

because one pC times one kHz is $10^{-9}$ C/s. This describes the mean rate,
not the instantaneous rate inside a bunch. A source with inter-electron
coherent terms would require its own charge weighting; it is outside this claim.
Interference between the PXR and CBS amplitudes of each primary remains
inside the already-computed $Y$ and does not alter this conversion.

## Cheap filters and independent numerical anchors

- Units: photons/(electron eV sr), multiplied by electrons/s, yields
  photons/(s eV sr).
- Limits: zero current gives zero at every supported finite source cell;
  doubling current doubles the rate without changing its spectral shape.
- Sign: current is a magnitude. Negative or nonfinite current is invalid;
  electron charge's negative sign cannot make the photon rate negative.
- Direction convention: the result is still a density per sr. Multiplying
  by angular-bin width, $\sin\theta$, or solid angle would change the
  observable and is absent. This remains true at forward/backward directions.
- For a unit source density, 1 nA yields approximately
  $6.241509074460763\times10^9$ photons/(s eV sr).
- A bunch charge of 2 pC at 3 kHz corresponds to 6 nA, hence approximately
  $3.744905444676458\times10^{10}$ photons/(s eV sr) for a unit density.
- Unsupported direction cells remain explicitly masked and are outside the
  zero-current statement; a mask must not be interpreted as a physical zero.

## Implementation comparison

After recording the derivation, inspection of `DirectionalSpectrum.rate`
found the following expression:

```python
return self.density * (current * 1e-9 / elementary_charge)
```

`elementary_charge` is the SI charge magnitude. The multiplier agrees
term for term with the boxed expression: the nA conversion occurs once,
the division is by elementary charge, and no direction spacing or solid
angle is present. The method rejects negative and nonfinite currents.
Its default resolves through `beam_current_na(self.case)`; for a case
with bunch charge 2 pC and repetition rate 3000 Hz, this gives 6 nA.

A separate spot check constructed a unit-density `DirectionalSpectrum`
without transport or a detector. The 0, 1, 2 and default 6 nA outputs
agreed with the numerical anchors above to relative tolerance
$2\times10^{-15}$. Negative, NaN and infinite overrides raised
`ValueError`. The reference values used the BIPM elementary charge,
independently of the implementation's constant import.

## Verdict

`rederived`: units, limits and sign conventions pass; the implementation
matches the independent expression. The task owner may advance the ledger
to `anchored` once the regression tests named in its row pass. This claim
does not validate the underlying PXR/CBS spectrum or establish human sign-off.
