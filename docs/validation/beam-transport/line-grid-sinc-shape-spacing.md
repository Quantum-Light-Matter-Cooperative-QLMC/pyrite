# `line-grid-sinc-shape-spacing`

Ledger row: [`line-grid-sinc-shape-spacing`](../ledger-core-coherent-physics.md#line-grid-sinc-shape-spacing). Status: unverified. Validation: `line-grid-sinc-shape-spacing`. Issue #1230.

## Claim

The opt-in `sinc-shape` resolution places the measured `resonance-population` line axis on uniform bins of width

```{math}
:label: eq-line-grid-sinc-shape-step

h = \frac{3}{\pi}\sqrt{\varepsilon_\mathrm{shape}}\; W_q ,
```

where $W_q$ is the sinc-Nyquist feature width of the case (the first-zero width at the aliased-weight quantile of `sinc_feature_spacing`) and $\varepsilon_\mathrm{shape}$ is the shape tolerance (`DEFAULT_SHAPE_RTOL = 1e-2`, the #1143 shape margin). With `bin-mean` quadrature, an isolated line of first-zero width $W \ge W_q$ then has peak height (highest bin mean) and FWHM (linear interpolation between bin means) within $\varepsilon_\mathrm{shape}$ of the exact line. Integrated yield stays exact at any spacing, because bin means telescope ([`sinc-bin-integration`](../radiation-physics/sinc-bin-integration.md)). If the uniform axis would need more than `DEFAULT_SHAPE_MAX_POINTS` (1,000,000) coordinates, the policy falls back to `resonance-local` on the same measured band. The run record (`shape_axis`) states the route, the required count and the reason.

## Derivation

Take the finite-time lineshape ([`finite-time-lineshape`](../ledger-core-coherent-physics.md)) of one line, $f(u)=\operatorname{sinc}^2(u/W)$ in NumPy's convention, with $u=E-E_\mathrm{res}$ and $W=\pi/a_w$ the first-zero width. Near the peak,

$$
f(u) = 1-\frac{\pi^2}{3}\frac{u^2}{W^2}+O\!\left(\frac{u^4}{W^4}\right).
$$

The mean of $f$ over a bin $[\delta-h/2,\delta+h/2]$ is

$$
\bar f(\delta)=1-\frac{\pi^2}{3W^2}\left(\delta^2+\frac{h^2}{12}\right)+O\!\left(\frac{h^4}{W^4}\right).
$$

The highest bin has $|\delta|\le h/2$. Its relative height error is therefore at most

$$
\frac{\pi^2}{3}\left(\frac14+\frac1{12}\right)\frac{h^2}{W^2}=\frac{\pi^2}{9}\frac{h^2}{W^2}.
$$

The worst case is a resonance on a bin edge. Setting the bound equal to $\varepsilon_\mathrm{shape}$ gives {eq}`eq-line-grid-sinc-shape-step` for $W=W_q$. A wider line has a smaller error, since the bound falls as $W^{-2}$.

**FWHM.** The FWHM is read by linear interpolation of the bin means at half maximum. Its error comes from the curvature at $u\approx0.443W$ and from the height error, which moves the half-maximum level. No closed form is used. An exact single-line scan over 100 resonance offsets per step (antiderivative bin means) gives a worst relative FWHM error at or below the height bound for every $h/W\le1/4$:

| $h/W$ | worst height | $\pi^2h^2/9W^2$ | worst FWHM |
|---:|---:|---:|---:|
| 1/4 | 6.53% | 6.85% | 6.10% |
| 1/8 | 1.69% | 1.71% | 1.31% |
| 1/10 | 1.09% | 1.10% | 0.77% |
| $0.0955$ ($\varepsilon=10^{-2}$) | $\le$1.00% | 1.00% | $\le$0.77% |
| 1/16 | 0.43% | 0.43% | 0.33% |

**Assumptions.**

- Each peak is one line. Overlapping lines and lines wider than $W_q$ have less curvature relative to their height, so their error is smaller. #944 measured 0.65%/0.81% (height/FWHM) at $h=W_q/4$ on HOPG 300 keV, where the bound is 6.5%.
- Lines narrower than $W_q$ carry at most the aliased-weight share of $t_L^2$ weight (`aliased_weight_limit`, the strictest `rtol`). This budget is already accepted by `sinc-nyquist`. Their own shape error can exceed $\varepsilon_\mathrm{shape}$.
- Peaks are read from the bin means without interpolation, as the #944 qualification harness reads them.

**Limiting cases.** As $\varepsilon_\mathrm{shape}\to0$, $h\to0$: pointwise sampling of the lineshape. At $h=W$ the bound exceeds 100% and the axis no longer resolves the peak; there `sinc-nyquist` keeps only yield exact.

## Budget and fallback

On the Fourier reduction ([`cxr-line-fourier-reduction`](../ledger-core-coherent-physics.md#cxr-line-fourier-reduction)), cost is nearly flat in bin count. #944 measured 43k→865k bins within +60% on CUDA and ROCm. `DEFAULT_SHAPE_MAX_POINTS` is that largest measured axis, rounded. Above it (MeV-wide measured bands) the policy keeps `resonance-local` instead of over-allocating. That route's own budget and refusals still apply.

## Anchors

- `tests/energy-grid/test_line_grid_shape_spacing.py::test_shape_step_bounds_single_line_height_and_fwhm`: the single-line bound at $\varepsilon=10^{-2}$ and $10^{-3}$ (tight to 2% at the worst offset).
- `test_measured_shape_grid_is_uniform_at_the_shape_step` and `test_measured_shape_grid_falls_back_to_resonance_local_above_budget`: the runner's route.

Production qualification on fixed tracks (#1230 Plan) is outstanding. Fresh-context verification and human sign-off are pending.
