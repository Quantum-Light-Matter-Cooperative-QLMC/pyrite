# Maintain project branding

`scripts/pyrite_logo.py` owns the PyRITE wordmark and its derivatives. The
canonical generated assets live in `src/pyrite/_assets/branding/` and ship with
the Python package. Documentation builds and applications consume these files;
they never run the logo simulations.

## Regenerate the assets

From a checkout with reference tables installed and plotting dependencies:

```bash
PYRITE_MC_BACKEND=cpu MPLCONFIGDIR=/tmp/pyrite-matplotlib \
  uv run python scripts/pyrite_logo.py
```

The default sample uses 4,096 electron histories for the Y and 256 for the
spectral I. A 2,000-step per-electron capacity bounds the Y allocation;
incomplete histories raise rather than contribute truncated artwork. Increase
`--y-max-steps` only with an appropriate memory budget if changing the target.
The checked assets were generated within an 8 GiB address-space limit.

Transport and spectral arrays are cached under `pyrite-output/branding/cache/`.
Typography changes reuse these arrays. Use `--force` to recompute after changes
to the underlying models or data. The JSON report records simulation inputs,
resolved spectral seed, rendering choices, cache hashes, generator hash and
rendering checkout/version. A cache hash identifies reused inputs; the
rendering checkout is not a claim that cached arrays were freshly simulated.

## Use the appropriate derivative

| Asset | Use |
| --- | --- |
| `pyrite-wordmark.svg` | Scalable master for print or presentations; letters are outlined paths. |
| `pyrite-wordmark.png` | README, PyPI, documentation header, app headers and browser demo. |
| `pyrite-icon.png` | Square trajectory Y for avatars and app icons. |
| `pyrite-favicon.ico` | Browser tab icon with 16, 32, 48 and 256 pixel variants. |
| `apple-touch-icon.png` | Mobile home-screen icon. |
| `pyrite-social.png` | 1200 × 630 sharing preview; also suitable for a GitHub repository social preview. |
| `head.html` | Self-contained favicon metadata for packaged and exported Marimo apps. |
| `pyrite-wordmark.json` | Input and rendering provenance. |

The navy background is part of the artwork and works in light and dark UI
surfaces. Preserve the aspect ratio and provide `PyRITE` alternative text.
Sphinx copies the canonical assets through `html_static_path`; the demo export
copies its wordmark into `demo/public/`. App headers embed the PNG as a data URL
so exported HTML works without a running asset server. App head-file paths are
relative to each notebook source and work in installed wheels.

The README uses a repository-relative image. PyPI uses the raw GitHub `main`
asset URL because its Markdown renderer cannot load package-local files; that
image becomes available when the asset commit reaches `main`. Documentation
sharing metadata uses the configured canonical documentation base URL.
Repository social-preview and account-avatar settings are managed in GitHub's
settings; the matching assets above are ready for those uploads.

For the GitHub repository, upload `pyrite-social.png` in the Social preview
section of [repository settings](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/settings).
Use `pyrite-icon.png` when an account or social profile needs a square avatar,
and `pyrite-social.png` for wide sharing cards. Changing a repository preview
does not change the organization avatar. Documentation Open Graph and Twitter
card metadata use the wide preview with descriptive alternative text; those
changes become public when this documentation revision is deployed.

## Scientific meaning

The Y selects transmitted Monte Carlo histories, projects them with a magnified
lateral axis and colors them by energy. The incident source is a uniform
rectangular beam. The I uses a normalized, background-subtracted simulated
PXR/CBS peak. Glow, colors and typography are artistic. This artwork is not a
calibrated spectrum, an unbiased transport visualization or validation evidence.

Review all variants after regeneration, including a narrow documentation
header, both UI themes, the browser demo and an exported app. Run the normal
`pyrite-dev docs` gate and the affected app presentation checks.
