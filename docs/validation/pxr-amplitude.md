# Validation: `pxr-amplitude`

## Claim and source

- Claim: `materials/crystal.py::chi_g` returns the dimensionless Fourier
  component of the X-ray electric susceptibility.
- Source: Feranchuk--Spence (2000), Eq. (3), as cited by the ledger and
  derivation docstring.
- Signature: `chi_g(crystal, hkl, photon_E_eV, B_ang2=0.0,
  use_henke=False)`.

## Independent derivation

For a unit-cell structure factor

\[
S_{\mathbf g}=\sum_j f_j(\mathbf g,E)
  \exp(i\mathbf g\cdot\mathbf r_j)\exp(-W_j),
\]

the Fourier component of the electron number density is
`S_g / V_cell`. In Gaussian units the driven-electron response gives

\[
\chi_{\mathbf g}
=-\frac{4\pi e^2}{m\omega^2}\frac{S_{\mathbf g}}{V_{cell}}.
\]

Using `r_e=e^2/(mc^2)`, `k=omega/c=2 pi/lambda`,

\[
\chi_{\mathbf g}
=-\frac{4\pi r_e}{k^2V_{cell}}S_{\mathbf g}
=-\frac{r_e\lambda^2}{\pi V_{cell}}S_{\mathbf g}.
\]

No extra factor of `2`, `pi`, or unit-cell multiplicity remains when `S_g`
is the full unit-cell sum. A complex anomalous structure factor is allowed;
the overall minus sign maps a positive electron-density amplitude to the
usual X-ray susceptibility convention.

## Cheap filters

- Units: `r_e lambda^2 / V_cell` is `angstrom^3 / angstrom^3`; `S_g` is in
  electrons, treated as a dimensionless scattering amplitude. `chi_g` is
  dimensionless.
- Limits: `S_g -> 0` for an extinct reflection gives `chi_g -> 0`;
  `lambda -> 0` gives `chi_g -> 0` as `lambda^2`; doubling identical unit-cell
  contents and volume leaves `chi_g` unchanged.
- Sign/convention: the leading minus sign follows the negative-electron
  plasma response. With `F=f0+f'+i f''` and the repository's wave convention,
  the same minus sign produces the passive-medium susceptibility sign used by
  its absorption model.

## Implementation comparison

Production computes `S` through `structure_factor`, sets
`lambda = HC_EV_ANG / photon_E_eV`, and returns

```text
-R_E_ANG * lambda**2 / (pi * V_cell) * S
```

This matches the independent expression term-for-term. `structure_factor`
applies the Debye--Waller factor once inside the site sum; production does not
apply a second factor. The docstring's first displayed form writes both `S(g)`
and `exp(-W)`, while its second form and implementation treat `S(g)` as already
Debye--Waller-weighted. This is notation ambiguity, not an implementation
factor error.

Independent LiF (200), 3890 eV spot check: the unit-cell sum was rebuilt from
the eight catalog sites with direct xraydb calls and `g=4 pi/a`, without using
`structure_factor`. It gives

```text
reference chi_g = -4.304121622460900e-05 - 1.259695312646876e-06 i
production chi_g = -4.304121622460899e-05 - 1.259695312646876e-06 i
absolute difference = 6.78e-21
```

The magnitude, about `4.306e-5`, also reproduces the recorded Feranchuk LiF
anchor.

Follow-up applied after independent verification: the function docstring now
contains `Validation: pxr-amplitude`, and its Debye--Waller wording now makes
the single application explicit.

## Verdict

`rederived`. Units, limits, sign, every prefactor, and an independent numeric
point match. Ledger status advanced from `unverified` to `rederived` after
verification.
