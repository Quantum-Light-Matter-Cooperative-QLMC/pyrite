# Validation: characteristic-radiation

## Source model

PyRITE reads electron-impact subshell-ionization cross sections from the
packaged 2025 Livermore Evaluated Electron Data Library (EEDL), distributed in
ENDF-6 form as NDS-IAEA-226. The parser accepts ENDF File 23, MT 534--572 TAB1
sections and converts their tabulated cross sections from barns to cm$^2$.
Each section declares interpolation law 2, so the incident-energy dependence
is evaluated piecewise linearly and set to zero outside the tabulated range.

EEDL supplies vacancy-production cross sections, but not the relaxation data
used here. Line energy $E_{ai\ell}$, fluorescence yield $\omega_{ai}$, and
conditional radiative intensity $I_{ai\ell}$ come from the Elam tables exposed
by xraydb. Natural initial- and final-hole widths come from xraydb's compiled
Krause--Oliver and Keski-Rahkonen--Krause tables. For segment $j$ in an
emitting material, the bin-averaged track-length estimator is

```{math}
:label: eq-characteristic-track-length

\left.\frac{d^2N}{dE\,d\Omega}\right|_b
=\frac{1}{4\pi N_e\,\Delta E_b}
\sum_{j,a,i,\ell}
n_a L_j\,\sigma_{ai}(T_j)\,
\omega_{ai}I_{ai\ell}\,
\exp[-\tau_j(E_{ai\ell})]\,\widehat q_{ai\ell b}.
```

Here $a$ is an element, $i$ an initially ionized subshell, $\ell$ a line from
that vacancy, $T_j$ the representative electron energy, and $b$ an energy bin.
The factor $\widehat q_{ai\ell b}$ is the natural Lorentzian mass in bin $b$,
renormalized over the requested grid:

```{math}
q_{\ell b}=\frac{1}{\pi}\left[
\tan^{-1}\!\frac{2(E_b^+-E_\ell)}{\Gamma_\ell}
-\tan^{-1}\!\frac{2(E_b^--E_\ell)}{\Gamma_\ell}
\right],\qquad
\widehat q_{\ell b}=\frac{q_{\ell b}}{\sum_c q_{\ell c}},\qquad
\Gamma_\ell=\Gamma_{\rm initial}+\Gamma_{\rm final}.
```

The result is photons eV$^{-1}$ sr$^{-1}$ per incident electron. The factor
$1/(4\pi)$ is isotropic emission; $\tau_j$ is the existing PyRITE
Beer--Lambert optical depth along slab, finite-prism, groove, or multilayer
escape geometry. Exact CDF differences avoid point-sampling line shapes much
narrower than a bin. The grid renormalization preserves the full transition
yield whenever its centre is in the requested window, matching the previous
delta-line window convention.

## Units and numerical conventions

- $n_a$ is stored in $\AA^{-3}$ and multiplied by $10^{24}$ to obtain
  cm$^{-3}$; $L_j$ is stored in $\AA$ and multiplied by $10^{-8}$ to obtain cm.
  Thus $n_aL_j\sigma_{ai}$ is a dimensionless expected vacancy count.
- Dividing the analytically integrated Lorentzian mass by $\Delta E_b$
  produces the spectral density represented on PyRITE's line-grid centres.
  Detector broadening remains a separate downstream operation.
- A line contributes only when its centre lies inside the requested line grid.
  Its Lorentzian tails are then normalized on that grid so truncation cannot
  change the integrated vacancy yield.
- The transition FWHM is the sum of the pertinent initial- and final-hole
  widths. Combined final labels such as `M4,5` use the mean available
  component width. A missing final width contributes zero; a missing initial
  width fails closed.
- Packaged EEDL bytes are verified before first use against SHA-256
  `f3ef54f66efaa606a4a5ea7afb3cfe10e35a22b543887dafb3fc7ec830d1769c`.
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
optional coherent PXR/CBS totals. It is retained as `spec_characteristic` for
audit and plotting, persisted in its own `characteristic.h5` component, and can
be hidden in the analysis app without modifying stored totals. The control
defaults to showing the component.

A single natural-width Lorentzian is used per xraydb transition. The empirical
multi-Lorentzian fits of Hölzer et al. demonstrate satellite and asymmetric
structure in 3d-transition-metal lines, but do not supply a universal
parameterization for the full EEDL element/shell domain. That finer structure,
chemical shifts, and multiple-vacancy broadening are intentionally excluded.

## Limits and regression evidence

- If density, segment length, cross section, fluorescence yield, or line
  intensity tends to zero, {eq}`eq-characteristic-track-length` tends to zero.
- With zero attenuation and one line, summing $\Delta E_b$ times the returned
  density recovers $nL\sigma\omega I/(4\pi N_e)$ to floating-point
  precision.
- Splitting a constant-energy segment into collinear subsegments preserves the
  total yield because the estimator is linear in path length.
- Increasing optical depth suppresses the line monotonically through
  $\exp(-\tau)$.

`tests/montecarlo/test_characteristic.py` anchors the packaged carbon K-shell
EEDL value and natural carbon K-alpha width, exact Lorentzian bin integration
and yield preservation, segment-subdivision invariance, absorber/emitter
separation, and runner composition into both emission modes. Checkpoint/reline
tests cover independent `characteristic.h5` persistence, missing-component
compatibility, and identity boundaries. A small end-to-end HOPG simulation
also exercises ENDF parsing, electron transport, self-absorption, line
profiles, and result assembly.

## Validation status

This is an implementation-context derivation and regression record updated
2026-09-10. The source-to-code mapping, units, Lorentzian normalization,
limiting cases, and integration paths were reviewed while implementing the
feature. It has not received the repository's required independent
fresh-context re-derivation, so the ledger status is `filtered`; human sign-off
remains pending.
