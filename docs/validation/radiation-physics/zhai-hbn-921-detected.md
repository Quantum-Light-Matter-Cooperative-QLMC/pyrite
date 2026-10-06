# Zhai h-BN 921 nm detected spectrum

Validation: `zhai-hbn-921-detected`.

Independent verifier; source derivation below written before reading implementation
bodies. Source: [Zhai et al. (2025), publisher Supplementary Information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-025-66063-6/MediaObjects/41467_2025_66063_MOESM1_ESM.pdf),
Eqs. (2), (5)–(9), (13), (14), (16), Sections S3, S4, S6, S7,
Fig. S5b and Table 4. The locally downloaded publisher PDF and extracted text were
the source inspected; the web reader could not open the publisher URL.

## Independent source derivation

The incident-electron-normalized angular photon density in Eq. (2) is

$$
\frac{d^2N}{d\omega\,d\Omega}
=\frac{\alpha\omega}{4\pi^2c^2N_e}
\sum_{j=1}^{N_e}\sum_i\sum_s
\left|\int_0^{t_{L,i}}\mathbf v_i(t)\cdot
\mathbf E_{\mathbf k s}[\mathbf r_i(t),\omega]
e^{-i\omega t}\,dt\right|^2.
$$

The polarization sum gives an unpolarized photon count. The collision-segment
sum is a sum of intensities; reciprocal vectors and PXR/CBS amplitudes inside
one segment are summed before taking the modulus square. A sum of amplitudes
over different collisions computes a different quantity.

Let photon energy $E$ be in eV, let $H=\hbar c$ be in eV Å, and define the
length-valued segment amplitude $\mathcal A_{is}$ by the integral above using
velocities in Å/s. Since $d\omega/dE=1/\hbar$,

$$
\boxed{\frac{d^2N}{dE\,d\Omega}
=\frac{\alpha E}{4\pi^2H^2N_e}
\sum_{j,i,s}|\mathcal A_{is}|^2.}
$$

The prefactor has units $\mathrm{eV}^{-1}\mathrm{Å}^{-2}$; the result has
units photons/eV/sr/incident electron. Equivalently, parameterizing a straight
segment by path length gives $\mathbf v\,dt=\hat{\mathbf v}\,d\ell$.
There is **no additional** overall $\beta^2$, $1/\beta^2$, $\gamma^2$, or
$1/\gamma^2$ in this length-amplitude convention. The speed remains in the
phase mismatch

$$
\Delta=\frac{E}{H\beta}-(\mathbf k+\mathbf g)\cdot\hat{\mathbf v},
\qquad \mathbf k=\frac{E}{H}\hat{\mathbf n},
\qquad
E_p=H\frac{\boldsymbol\beta\cdot\mathbf g}
{1-\boldsymbol\beta\cdot\hat{\mathbf n}}.
$$

The Lorentz factor belongs to the CBS amplitude in Eq. (6), and its
longitudinal contraction in Eq. (7); these are not global source normalization
factors. For a vacuum, constant-velocity segment of length $L$,

$$
\left|\int_0^L e^{-i\Delta\ell}\,d\ell\right|^2
=L^2\operatorname{sinc}^2(\Delta L/2).
$$

Thus the resonance limit is $L^2$ and the vanishing-length limit is zero.
All photon densities are nonnegative. Reversing the convention for both
reciprocal vectors and their phase factors leaves the physical result unchanged.

For a central-ray source density $S(E)$, the small-aperture approximation to
the measured spectrum is

$$
D(E)=\frac{10^{-9}}{e}\,\Omega
\int S(E')\,q(E')\,G(E-E';\sigma(E'))\,dE',
\qquad \Omega=0.066\ \mathrm{sr}.
$$

Here $10^{-9}/e=6.241509074\times10^9$ incident electrons/s/nA,
$q$ is any explicitly assumed detector quantum efficiency, and $G$ has unit
area. The output is photons/eV/s/nA. The SI explicitly prescribes Gaussian
broadening, but S3/S4 do not specify a numerical window-QE correction or whether
the experimental spectra have already been efficiency-corrected. Setting or
removing $q$ therefore needs a separately documented detector convention.

From Eqs. (13), (14), (16),

$$
\sigma^2(E_p)=\frac{2.52E_p+988}{8\ln2}
+\left(\frac{\partial E_p}{\partial\theta_{\rm obs}}
\frac{\Delta\theta_{\rm obs}}{3}\right)^2,
\qquad \Delta\theta_{\rm obs}=16.6\frac{\pi}{180}.
$$

The angle is the **full stated variation**; its divisor is 3, rather than
$\sqrt3$ or 6. Differentiating the peak-energy denominator gives

$$
\frac{\partial E_p}{\partial\theta_{\rm obs}}
=E_p\frac{\boldsymbol\beta\cdot\partial_\theta\hat{\mathbf n}}
{1-\boldsymbol\beta\cdot\hat{\mathbf n}}.
$$

This matches Eq. (14), including its outer minus sign and the signs inside its
numerator. For the initial beam along $+z$, it becomes
$-E_p\beta\sin\theta_{\rm obs}/(1-\beta\cos\theta_{\rm obs})$.
Only its magnitude enters the variance. The aperture contribution vanishes as
the angle span tends to zero. A unit-area response preserves integrated photon
number on an unbounded energy grid; truncating tails can reduce it.

S3 supplies observation angle 119°, aperture 0.066 sr, negligible beam divergence
of about 1 mrad, negligible current fluctuation below 1%, live-time acquisition,
and experimental bremsstrahlung subtraction. S4 supplies the empirical EDS
resolution and energy calibration, not a new source normalization. S6 identifies
approximately 921 nm h-BN at tilt 17° and azimuth 130°; the tilt was fitted by
RSS against multiple spectra, rather than independently measured. S7 identifies
17.5, 20, 22.5 and 25 keV and applies the same processing. No thickness standard
deviation is listed for the approximately 921 nm specimen in Table 4.

For anisotropic Debye–Waller parameters in Å², the amplitude factor is

$$
e^{-W},\qquad
W=\frac{B_{11}(g_x^2+g_y^2)+B_{33}g_z^2}{16\pi^2}.
$$

An isolated basal $(00l)$ intensity therefore changes as
$\exp[-\Delta B_{33}l^2/(2c^2)]$ and is independent of $B_{11}$.
For h-BN $c\simeq6.66$ Å, changing $B_{33}$ from 0.6 to 3.45 Å²
predicts factors about 0.879 for $(002)$ and 0.598 for $(004)$.
Off-basal reflections depend on both components; interference prevents assigning
the same isolated-reflection factor to an entire mixed spectrum. The SI states
the Debye–Waller factor but does not give these two h-BN parameter values.

## Implementation comparison

The inspected owners are `anchor_figures.py::_supplementary_detected_spectrum`,
`model_coherent_spectra`, `montecarlo/spectrum/lines/_setup.py::_prepare_spectrum`,
`_kernels.py::_line_weight_core`, `_line_amp_sq_core`, and the batched
incoherent reduction. Detector scoring is owned by `detectors/spec.py::LegacyEDS`
and `detectors/response.py`; aperture and EDS widths by `montecarlo/detector.py`.

The implementation uses normalized velocity $\mathbf v/c=\beta\hat{\mathbf v}$
and normalized flight time $ct_L=L/\beta$. Its literal prefactor is

```text
_PREF_C1 = 4.0 * xp.pi**2 * HBARC_EV_ANG
return alpha_fs * omega_res / pref_c1 * (t_L * t_L) * T_abs
```

Because `omega_res` is $E_{\rm res}/H$ and the PXR amplitude is proportional
to the normalized velocity, this gives exactly
$\alpha E_{\rm res}L^2/(4\pi^2H^2)$ times the direction-dependent amplitude
square and escape transmission. The $\beta^2$ in the PXR amplitude cancels
the $1/\beta^2$ in flight time. `gamma` occurs in the CBS amplitude, not this
prefactor. No missing global speed, Lorentz, angular-frequency-to-eV,
incident-electron, aperture, or nA conversion factor was found.

There are identifiable approximations relative to the full SI expression:

- The smooth prefactor and couplings are frozen at the resonance energy.
  The prefactor alone differs from Eq. (2) off resonance by
  $E_{\rm res}/E$; at line center it matches exactly.
- Reflections are reduced as intensities. Relative to Eq. (5), the discarded
  term is $2\operatorname{Re}\sum_{g<h}A_gA_h^*$ within a segment.
  PXR–CBS interference within each reflection is retained. The supplementary
  modeling call uses the default collision-segment intensity reduction,
  despite the word “coherent” in its function name; it does not enable
  across-collision amplitude summation.
- The maintained source includes in-medium kinematics and segment-mean
  self-absorption. These are separately modeled extensions to the vacuum peak
  relation in SI Eq. (9), not an extra overall Lorentz factor.

These differences identify the calculation actually performed. Their contribution
to the reported energy trend has not been established by this symbolic comparison.

The detector owner multiplies window efficiency **before** convolution and scales
by 0.066 sr times `6.2415e9` electrons/s/nA. Rounding the latter differs from the
exact charge conversion by approximately 1.45 parts per million. It uses one
constant Gaussian width evaluated at the pre-response spectral maximum; an
energy-dependent response evaluates the width separately at each source energy.
The implemented EDS and aperture formulas match Eqs. (13), (14), (16) at that
representative energy. The derivative's magnitude is appropriate for a width.

The catalog contains only basal $(002)$ and $(004)$ families and scalar
`B_ang2=3.45`. Since these reciprocal vectors are parallel to the crystal
$c$ axis, this scalar correctly represents $B_{33}$ for the retained families;
$B_{11}$ cannot change these basal couplings. The scalar implementation would
need a directional tensor contraction to represent off-basal reflections with
unequal $B_{11}$ and $B_{33}$. The SI does not independently establish the
numerical $B_{33}$ value used here.

## Independent numerical spot check

Using only the source formulas, electron rest energy 510.99895 keV,
$H=1973.269804$ eV Å and approximate $c=6.66$ Å, the unscattered vacuum
$(002)$ peak and combined response width are:

| Beam energy (keV) | $\beta$ | $\gamma$ | $E_p$ (eV) | Total FWHM (eV) |
| --- | --- | --- | --- | --- |
| 17.5 | 0.25520355 | 1.03424665 | 808.621 | 66.030 |
| 20 | 0.27186591 | 1.03913902 | 855.268 | 69.376 |
| 22.5 | 0.28735018 | 1.04403140 | 898.024 | 72.667 |
| 25 | 0.30184151 | 1.04892378 | 937.531 | 75.905 |

These are source-formula checks, not predictions of the scattered, mixed,
absorbed detected spectrum or a new fit to the paper.

## Owner numerical evidence, independently assessed

The task owner ran `checks/zhai_hbn_energy_trend.py` on the remote worker,
using CPU float64 model revision `f3b44c1e`. Jobs 1022, 1023 and 1024 used
2000 incident electrons per condition for 921, 736.8 and 1105.2 nm,
respectively; job 1021 used 200 electrons at 921 nm. The verifier inspected
the returned JSON and diagnostic implementation, and recomputed the ratios
below. Compact provenance and results are archived in
[the numerical evidence record](../check-records/zhai_hbn_energy_trend.json).
The 921 nm seeds are 92100–92103. Thickness runs use their own canonical,
thickness-derived seeds; source variants at one thickness reuse identical
transport trajectories. This is sensitivity evidence, not an independent
Monte Carlo implementation or a digitized-paper regression anchor.

All peak values below have units photons/eV/s/nA, including the source column,
which has already been scaled by the aperture and incident-electron rate.
“Unit QE” removes only the assumed window efficiency.

| Beam (keV) | Source | EDS, unit QE | EDS + aperture, unit QE | Detected | Historical paper approximation | Model/paper |
| --- | --- | --- | --- | --- | --- | --- |
| 17.5 | 4.110620 | 1.826812 | 1.671768 | 1.023305 | 0.85 | 1.204 |
| 20 | 4.375793 | 1.884144 | 1.685681 | 1.067736 | 0.80 | 1.335 |
| 22.5 | 4.492632 | 1.893013 | 1.656758 | 1.077713 | 0.75 | 1.437 |
| 25 | 4.767883 | 1.894204 | 1.619701 | 1.075611 | 0.70 | 1.537 |

The paper column contains the inherited approximate visual readings, **not**
digitized source data with quantified uncertainties. The new baseline is
20.4–53.7% above those readings. Its endpoint peak grows 5.11%, while the
historical paper approximation falls 17.65%; the model/paper ratio grows
27.64%. These figures supersede the historical model values quoted in the
ledger when assessing this run.

The model contains competing energy-dependent effects. Its source peak grows
15.99% from 17.5 to 25 keV. EDS broadening reduces that growth to 3.69%.
Adding aperture broadening produces a 3.11% endpoint decrease at unit QE.
The assumed window QE changes the detected/unit-QE peak ratio from 0.61211 to
0.66408, bringing the final endpoint change to a 5.11% increase. These ratios
compare maxima of separately processed spectra; they are not efficiencies at
one fixed photon energy. This identifies modeled trend contributions without
establishing a defect in any of them or the actual experimental QE convention.

The controlled variants support narrower exclusions:

- Replacing the post-#338 paired coupling with the previous coupling changes
  each h-BN source spectrum by less than $2.3\times10^{-16}$ relative to its
  baseline maximum. That change does not explain this basal h-BN residual.
- Replacing node sampling with bin-mean line quadrature changes detected peaks
  by at most 0.0114%. This particular grid discretization does not explain a
  20–54% residual.
- Increasing the sample from 200 to 2000 electrons changes the detected peaks
  by up to 3.1%, using the 200-electron result as the denominator. This
  comparison does not establish a universal sampling-error bound, but the
  observed residual and its trend persist. The earlier “at most 2%” convergence
  statement must not be reused for these runs.
- Both signs of each basal family are included in isolation. With both family
  spectra convolved using the **same** baseline Gaussian width, the $(004)$
  contribution at the total detected peak is 0.5410%, 0.3973%, 0.2335%,
  0.1773%. The peak is overwhelmingly $(002)$, rather than a substantial
  energy-dependent $(004)$ mixture.
- The detected peak ratio for $B_{33}=3.45$ versus 0.6 Å² is 0.87721,
  0.87780, 0.87848, 0.87871. This agrees closely with the independently
  derived isolated $(002)$ factor near 0.879; its endpoint variation is about
  0.17%. This correction mostly changes normalization and does not account
  for the residual energy trend. Basal-only $B_{11}$ remains irrelevant.
- Removing escape opacity while retaining transport and real refractive
  kinematics raises the detected peaks. The baseline/no-opacity peak ratio is
  0.59637, 0.61969, 0.64420, 0.66630. Thus opacity materially changes modeled
  amplitude and trend; this counterfactual does not demonstrate that the
  maintained attenuation model is wrong or reproduce the paper trend.

The assumed thickness sensitivity interval gives:

| Beam (keV) | 736.8 nm detected | 921 nm detected | 1105.2 nm detected |
| --- | --- | --- | --- |
| 17.5 | 0.993653 | 1.023305 | 1.025585 |
| 20 | 1.015752 | 1.067736 | 1.083222 |
| 22.5 | 0.997984 | 1.077713 | 1.133040 |
| 25 | 0.995372 | 1.075611 | 1.151251 |

The approximately ±20% interval is an **assumed model sensitivity**, not the
specimen's measured uncertainty. Neither endpoint reproduces the inherited
paper amplitudes or their decreasing trend. This excludes those two tested
thickness changes as a complete explanation under the maintained model; it
does not establish the specimen's true thickness or exclude arbitrary
thickness/geometry changes.

## Adjudication

Units, vanishing-length/aperture limits, positivity, response ordering and the
beam-axis width convention pass. The source normalization and nominal response
width are independently rederived. The **end-to-end** claim remains
`discrepancy`: the new owner-run detected peak ratios are
1.204, 1.335, 1.437, 1.537 against historical approximate paper values at
increasing beam energy. A constant normalization factor cannot change those
ratios' relative energy trend. The controlled variants exclude the named
normalization, pairing, basal-parameter and quadrature explanations within
their tested scope. The approximate experimental reference, unspecified
experimental window-QE convention, and unvalidated source-model differences
remain limitations. No causal defect explaining the residual has been established.

Suggested ledger change: retain `discrepancy`, link this derivation, and replace
the suggestion of a missing global Eq. (2) speed/Lorentz conversion with the
explicit exclusion above. Keep the window-QE convention and any new numerical
comparison separately documented. No human sign-off is implied.
