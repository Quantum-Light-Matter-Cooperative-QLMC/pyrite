# Validation: characteristic-radiation

## Source model

PyRITE reads electron-impact subshell-ionization cross sections from the
attached 2025 Livermore Evaluated Electron Data Library (EEDL), distributed in
ENDF-6 form as NDS-IAEA-226. The parser accepts ENDF File 23, MT 534--572 TAB1
sections and converts their tabulated cross sections from barns to cm$^2$.
Each section declares interpolation law 2, so the incident-energy dependence
is evaluated piecewise linearly and set to zero outside the tabulated range.

EEDL supplies vacancy-production cross sections, but not the relaxation data
used here. Line energy $E_{a i\ell}$, fluorescence yield $\omega_{a i}$, and
conditional radiative intensity $I_{a i\ell}$ come from the Elam tables exposed
by xraydb. For a segment $j$ in an emitting material, the bin-integrated
track-length estimator is

```{math}
:label: eq-characteristic-track-length

\left.\frac{d^2N}{dE\,d\Omega}\right|_b
=\frac{1}{4\pi N_e\,\Delta E_b}
\sum_{j,a,i,\ell}
n_a L_j\,\sigma_{a i}(T_j)\,
\omega_{a i}I_{a i\ell}\,
\exp[-\tau_j(E_{a i\ell})]\,
\mathbf{1}[E_{a i\ell}\in b].
```

Here $a$ is an element, $i$ an initially ionized subshell, $\ell$ a line from
that vacancy, $T_j$ the representative electron energy, and $b$ an energy bin.
The result is photons eV$^{-1}$ sr$^{-1}$ per incident electron. The factor
$1/(4\pi)$ is isotropic emission; $\tau_j$ is the existing PyRITE
Beer--Lambert optical depth along slab, finite-prism, groove, or multilayer
escape geometry.

## Units and numerical conventions

- $n_a$ is stored in $\AA^{-3}$ and multiplied by $10^{24}$ to obtain
  cm$^{-3}$; $L_j$ is stored in $\AA$ and multiplied by $10^{-8}$ to obtain cm.
  Thus $n_aL_j\sigma_{ai}$ is a dimensionless expected vacancy count.
- Dividing a bin-integrated delta line by $\Delta E_b$ produces the spectral
  density represented on PyRITE's line-grid centres. Natural linewidth and
  detector broadening are not applied here; detector response remains a
  downstream operation.
- A line exactly on an internal bin edge is assigned to the upper bin. Lines
  outside the requested line grid do not contribute, so a study of a specific
  shell must include its xraydb line energy in the configured line range.
- Packaged EEDL bytes are verified before first use against SHA-256
  `ce37912435e0b8002f85878f98ccf7c5840cb168f1d46af3c9e915cd16c70ccc`.
  The resolved xraydb package version is included in the characteristic-model
  checkpoint marker.

## Assumptions and scope

The estimator treats independent atoms and isolated, directly created
vacancies. It multiplies the xraydb edge fluorescence yield by the conditional
line intensity for that same initial shell. It does not invent Auger-fed
daughter vacancies, Coster--Kronig redistribution, multiple-vacancy shifts, or
Auger-electron transport because xraydb's line API is not a complete cascade
model. An EEDL shell with a nonzero fluorescence yield but no xraydb line list
is retained, emits zero photons, and raises a `RuntimeWarning` rather than being
silently approximated.

Characteristic emission uses the bremsstrahlung electron population and its
default 1 keV transport cutoff, rather than the PXR/CBS population's default
5 keV cutoff. This preserves more low-energy ionization path while sharing the
same trajectories as the rest of a case. The remaining path below 1 keV is not
modeled unless the case lowers `E_cut_brem_keV`. The 50 eV relaxation cutoff is
a photon-line data cutoff and does not override the electron transport cutoff.

For multilayers, each layer emits using its own elemental composition and all
layers attenuate the escaping photon. Passive absorber elements are not loaded
as EEDL emitters for another layer. Atomic relaxation is incoherent, so the
same characteristic component is added once to both PyRITE's incoherent and
optional coherent PXR/CBS totals. It is also stored separately as
`spec_characteristic` for audit and plotting.

## Limits and regression evidence

- If density, segment length, cross section, fluorescence yield, or line
  intensity tends to zero, {eq}`eq-characteristic-track-length` tends to zero.
- With zero attenuation and one line, the implementation reduces to
  $nL\sigma\omega I/(4\pi N_e\Delta E_b)$.
- Splitting a constant-energy segment into collinear subsegments preserves the
  total yield because the estimator is linear in path length.
- Increasing optical depth suppresses the line monotonically through
  $\exp(-\tau)$.

`tests/montecarlo/test_characteristic.py` anchors the packaged carbon K-shell
EEDL value, the unattenuated one-line analytic reduction, segment-subdivision
invariance, absorber/emitter separation, and runner composition into both
emission modes. Checkpoint/reline and identity tests cover persistence and
reuse boundaries. A small end-to-end HOPG simulation also exercises ENDF
parsing, electron transport, self-absorption, line binning, and result assembly.

## Validation status

This is an implementation-context derivation and regression record dated
2026-08-24. The source-to-code mapping, units, limiting cases, and integration
paths were reviewed while implementing the feature. It has not received the
repository's required independent fresh-context re-derivation, so the ledger
status is `filtered`; human sign-off remains pending.
