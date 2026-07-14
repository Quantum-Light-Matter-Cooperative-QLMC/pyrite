# Final review-fix report

## Scope

Fixed the `line_brem_ratio` material-comparison edge case: a material whose
eligible records all have non-finite local line-to-brem ratios is now omitted
from that plot. Other selection modes retain their existing candidate behavior.
Added direct metric regression coverage for zero and negative local brem
integrals, and corrected the `line_metrics` brem-metric documentation.

## RED

Command:

```bash
uv run python scripts/dev.py test tests/test_material_comparison.py tests/test_results.py
```

Output: `1 failed, 8 passed in 1.23s`.

The new `test_material_comparison_omits_all_invalid_local_ratio_material`
failed because the plot annotated both `Invalid (30 keV)` and `Valid (60 keV)`.
This reproduced the review finding: `selection_score` maps the invalid ratio to
`-inf`, and `max` selected the only invalid candidate for that material.

The two direct `line_metrics` parameter cases (zero and negative brem) passed
in RED because the existing `brem_line_int > 0` guard already returns `NaN`;
they now freeze that contract independently.

## GREEN

Implementation filters `line_brem_ratio` candidates by finite raw
`line_brem_ratio` after applying the optional energy floor and before `max`.
This is deliberately mode-specific, preserving every other selection mode's
handling of non-finite scores.

Command:

```bash
uv run python scripts/dev.py test tests/test_material_comparison.py tests/test_results.py
```

Output: `9 passed in 1.17s`.

## Additional checks

```bash
git diff --check
```

Passed with no output.

```bash
uv run python scripts/dev.py lint
```

Failed on a pre-existing, unrelated import-order issue in
`notebooks/analysis_app.py:137` (`I001`): `cxr_mc.config.MATERIALS` is imported
after `cxr_mc.run._DEFAULT_CHECKPOINT_DIR`. This fix did not edit that import
block, so it was left outside scope.

## Self-review

- The filter runs after `min_line_eV`, so only eligible invalid candidates cause
  omission.
- It applies only to `select="line_brem_ratio"`; `peak`, `quality_peak`, and
  all other modes retain their prior behavior.
- The plot regression checks both the visible annotation and scatter-point count.
- The metric regression parametrizes exactly the two non-positive integral cases
  requested.
- The updated docstring distinguishes full-grid and local-window brem-derived
  metrics and no longer claims brem is limited to `total_flux` and `line_frac`.
