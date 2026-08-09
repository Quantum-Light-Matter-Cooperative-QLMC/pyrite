# Validation: `external-brem-subtraction`

Independent verification of weighted external-background normalization and
subtraction.

- **Claim id:** `external-brem-subtraction`
- **Code:** `src/cxr_mc/validation/validation_background.py::fit_external_background`
- **Source:** weighted least squares through the origin. Zhai et al.,
  Supplementary Information S3, motivates numerical DTSA-II/PIXE
  bremsstrahlung subtraction but does not specify this fit.
- **Intended quantity:** one scale `a` fitted on background-only sidebands,
  followed by subtraction `r_i = y_i - a b_i`.

## 1. Independent derivation

Let `y_i` be measured detected intensity, `b_i` the external detected
background on the same energy grid, and `sigma_i` known independent standard
uncertainties. For sideband indices `S`, assume

```text
y_i = a b_i + epsilon_i
E[epsilon_i] = 0
Var(epsilon_i) = sigma_i^2
```

with no intercept, slope, energy shift, or extra broadening. Minimizing

```text
chi2(a) = sum_(i in S) (y_i - a b_i)^2 / sigma_i^2
```

gives

```text
d chi2 / da = -2 sum_i w_i b_i (y_i - a b_i) = 0
w_i         = 1 / sigma_i^2
a_hat       = sum_i w_i b_i y_i / sum_i w_i b_i^2 .
```

For known absolute uncertainties and an exact external shape, curvature of
`chi2` gives

```text
Var(a_hat) = 1 / sum_i w_i b_i^2
SE(a_hat)  = 1 / sqrt(sum_i w_i b_i^2) .
```

This is a conditional formal standard error. It does not include uncertainty
in `b_i`, sideband choice, interpolation, or background-shape mismatch, and it
must not be multiplied by `sqrt(reduced_chi2)` when supplied `sigma_i` are
treated as known absolute uncertainties. With `sigma=None`, choosing
`sigma_i=1` produces an OLS-through-origin fit and a formal uncertainty under
that unit-uncertainty convention; it does not estimate noise from residuals.

One parameter is fitted, so `reduced_chi2 = chi2 / (n - 1)`. The fitted
background and residual over the full grid are

```text
f_i = a_hat b_i
r_i = y_i - f_i .
```

### Units

For the versioned fixture, `y`, `b`, and `sigma` all have units
`Phs/eV/s/nA`. Thus `w` has inverse-square intensity units, numerator and
denominator of `a_hat` are dimensionless, and `a_hat` and its standard error
are dimensionless. `f` and `r` retain `Phs/eV/s/nA`. No solid-angle,
detector-efficiency, resolution, channel-width, live-time, or beam-current
factor belongs in this subtraction because both inputs are already detected
intensities.

### Limits and conventions

- If `y_i = a_0 b_i` on every selected point, `a_hat = a_0`, `r_i = 0` on
  those points, and `chi2 = 0`.
- With unit uncertainties, the expression reduces to ordinary least squares
  through the origin.
- Multiplying all supplied `sigma_i` by `c` leaves `a_hat` unchanged and
  multiplies `SE(a_hat)` by `c`.
- A physical background amplitude is non-negative. The implementation rejects
  a negative unconstrained optimum instead of returning the constrained
  boundary solution `max(0, a_hat)`. This is an explicit input/model rejection,
  covered by a test; callers must not interpret it as a boundary-fit result.

## 2. Source and fixture provenance

Zhai et al., *Nature Communications* **16**, 11218 (2025),
doi:`10.1038/s41467-025-66063-6`, states that Figure 3 experimental
bremsstrahlung is subtracted; SI S3 describes DTSA-II plus a numerical PIXE
method but does not publish an objective, sideband, or fitted parameterization.
Therefore the through-origin WLS procedure above is a cxr-mc analysis method,
not a reconstruction of an unpublished Zhai fit.

The cited DR-NTU dataset was checked independently through its Dataverse API:

- dataset V1, doi:`10.21979/N9/WZAMZ0`,
  `UNF:6:nJhYFFasbKoHid6aS6g/PA==`, CC BY-NC 4.0;
- file 287301, displayed as `Figure 3b Bremsstrahlung.tab`, original-format
  MD5 `930509a7e99c67a61936f94b7ad0510b`;
- file 287302, displayed as `Figure 3b Experiment.tab`, original-format MD5
  `68c414f7c14a8d323abdd994d6340152`.

After delimiter conversion only, every local bremsstrahlung value in columns
7--8 and every local experiment value in columns 13, 15, and 16 matched the
Dataverse tabular exports exactly. Headers identify the shared detected
intensity unit `Phs/eV/s/nA`; experiment column 16 is its y-error column.

## 3. Comparison with implementation

Implementation matches the independent expressions term by term:

| quantity | independent expression | implementation | result |
|---|---|---|---|
| weights | `1 / sigma_i^2` | `1 / square(uncertainty)` | match |
| scale | `sum(w b y) / sum(w b^2)` | same dot products | match |
| scale standard error | `1 / sqrt(sum(w b^2))` | `sqrt(1 / denominator)` | match |
| residual | `y - a b` | same | match |
| reduced chi-square | `sum(w r^2) / (n - 1)` | same | match |
| full-grid subtraction | `intensity - a*background` | same | match |

Selection also excludes non-finite values, non-positive uncertainties, and
non-positive external background samples. Requiring at least two selected
points makes `n - 1` positive. `load_external_brem` interpolates intensity
only, with zero outside fixture support; the deposited experiment and
background fixtures share their energy grid, so this validation condition
introduces no interpolation normalization.

An independent arithmetic evaluation of the deposited sidebands
`E < 900 eV` or `E > 1040 eV` gives:

```text
n              = 52
a_hat          = 1.05951249630711
SE(a_hat)      = 0.000878647031832134
chi2           = 5445.22929911931
reduced_chi2   = 106.769201943516
```

Large reduced chi-square shows supplied experimental error bars do not describe
all disagreement with the fixed external shape. It does not change formula
agreement, but the small `scale_std` must retain its documented conditional
meaning and must not be presented as total background-normalization
uncertainty.

## 4. Anchors and traceability

`tests/materials/test_validation_background.py` passes all five tests. It covers exact
scale recovery, zero sideband residual, formal scale uncertainty, fitted
degrees of freedom, negative/underdetermined rejection, use of the existing
external loader, a positive excluded coherent peak, and absolute no-rescale
comparison. Full primary-file checksums and all-column identity were verified
for this review but are documented provenance rather than network-dependent CI
tests.

The implementation docstring contains
`Validation: external-brem-subtraction`; the ledger row uses the same id and
resolves to `validation/validation_background.py::fit_external_background`. Source,
scope, limits, anchors, and warning against claiming exact Zhai processing
agree.

## 5. Verdict

- **Filters:** units PASS; exact-scale and zero-residual limits PASS;
  signs/conventions PASS.
- **Re-derivation:** `matches`.
- **Verdict:** `rederived`.
- **Suggested ledger change:** change status from `unverified` to `rederived`
  and link this write-up. Human applies the ledger edit; status is not
  `signed-off`.
