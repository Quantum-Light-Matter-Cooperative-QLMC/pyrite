# Cross-material comparison plots

## Purpose

Extend the Analysis app's Cross-material tab with two diagnostic scatter plots
that surface bright low-energy and line-to-background candidates while retaining
the existing quality-selected headline plot.

## Scope

The tab will show three independent one-point-per-material scatter plots:

1. Existing best quality-weighted coherent line selection.
2. Highest raw coherent peak spectral flux selection.
3. Highest integrated line flux relative to local incoherent bremsstrahlung.

All plots use dominant coherent-line energy on the x-axis, integrated line flux
on the logarithmic y-axis, and line-definition quality as point color. Each
label includes the selected beam energy. The two new selections exclude records
whose selected line energy is below 100 eV; the existing plot retains current
behaviour.

## Metrics and selection

`line_metrics` will add `line_brem_ratio`. It is the integrated coherent-line
flux divided by the integrated incoherent-bremsstrahlung flux over the same
peak-centred window already used for `line_flux`: plus or minus 1.5 FWHM (a
total width of three FWHM). If the local bremsstrahlung integral is zero or
negative, the metric is `NaN`.

`selection_score` will gain a `line_brem_ratio` mode, which sorts non-finite
ratios below finite candidates using its existing policy. The plotting helper
will accept a minimum dominant-line energy so selection can discard sub-100 eV
candidates without changing their underlying metric values.

## Implementation boundaries

- `src/cxr_mc/results/metrics.py`: calculate and document the local ratio.
- `src/cxr_mc/results/scoring.py`: expose the ratio as a selection mode.
- `src/cxr_mc/plots/spectra.py`: reuse the existing material-comparison renderer
  with its new selection and energy-floor parameter; update title wording to
  identify each criterion.
- `notebooks/analysis_app.py`: render the two new plots beneath the existing
  headline in the lazy Cross-material tab.

No transport physics, checkpoint format, or scan grid changes are required.

## Error handling

Materials without a checkpoint or without a candidate at/above the requested
energy floor are omitted. A zero local bremsstrahlung integral produces `NaN`,
which cannot win ratio selection. If no material supplies a candidate, the
existing empty-plot behaviour is retained.

## Tests and validation

Add focused tests that verify the local-ratio numerator and denominator use the
same FWHM window, zero local brem yields `NaN`, and the 100 eV floor changes
which record is selected by the material-comparison plot. Run those tests, the
plot-export guard, and `marimo check notebooks/analysis_app.py`.

## Deferred comparisons

Potential follow-ups are a quality-weighted local ratio, line FWHM versus flux
for the brightness-resolution tradeoff, and local coherent fraction. They are
not part of this change.
