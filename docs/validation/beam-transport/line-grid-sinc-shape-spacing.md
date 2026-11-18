# `line-grid-sinc-shape-spacing`

Ledger row: [`line-grid-sinc-shape-spacing`](../ledger-core-coherent-physics.md#line-grid-sinc-shape-spacing). Status: anchored (isolated-line claim). Validation: `line-grid-sinc-shape-spacing`. Issues #1230, #1265.

## Claim

Under `bin-mean` quadrature, the `sinc-oversampled` resolution (the default on every bandwidth since #1265; `sinc-shape` is its deprecated alias) places the line axis on uniform bins of width

```{math}
:label: eq-line-grid-sinc-shape-step

h \le \frac{3}{\pi}\sqrt{\varepsilon_\mathrm{shape}}\; W_q ,
```

where $W_q$ is the sinc-Nyquist feature width of the case (the positive first-zero offset at the aliased-weight quantile of `sinc_feature_spacing`, capped at `max_spacing_eV`) and $\varepsilon_\mathrm{shape}$ is the shape tolerance (`DEFAULT_SHAPE_RTOL = 1e-2`, the #1143 shape margin). The maximum-spacing cap and coordinate-count rounding can refine the step further. With `bin-mean` quadrature, an isolated line of first-zero width $W \ge W_q$ then has highest-bin peak height within $\varepsilon_\mathrm{shape}$ of the exact line. Interpolated FWHM meets that tolerance in the isolated-line scans below; this is numerical evidence, not a general FWHM theorem. Bin means times bin widths telescope to exact mass over identical represented outer edges ([`sinc-bin-integration`](../radiation-physics/sinc-bin-integration.md)); finite-band truncation and reducer error remain separate. If the uniform axis would need more than its point budget (`DEFAULT_BIN_MEAN_MAX_POINTS`, 1,000,000), resolution refuses and names `sinc-nyquist`; it never reroutes.

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

This height bound is rigorous, rather than only a Taylor estimate: the triangular Fourier representation gives

$$
\operatorname{sinc}^2x=\int_{-1}^{1}(1-|t|)\cos(2\pi xt)\,dt,
\qquad 1-\operatorname{sinc}^2x\le\frac{\pi^2x^2}{3},
$$

using $1-\cos z\le z^2/2$. Averaging this inequality over the nearest bin gives the same bound. Setting it equal to $\varepsilon_\mathrm{shape}$ gives {eq}`eq-line-grid-sinc-shape-step` for $W=W_q$. A wider isolated line has a smaller height bound, since it falls as $W^{-2}$.

**FWHM.** The FWHM is read by linear interpolation of the bin means at half maximum. Its error comes from the curvature at $u\approx0.443W$ and from the height error, which moves the half-maximum level. No closed form is used. The original single-line scans used 100 resonance offsets at selected spacings. Independent verification at the default spacing used 2,001 offsets and antiderivative bin means: height deficit 0.992833%, FWHM error 0.792912%. These scans do not prove a bound at every spacing or offset.

| $h/W$ | worst height | $\pi^2h^2/9W^2$ | worst FWHM |
|---:|---:|---:|---:|
| 1/4 | 6.53% | 6.85% | 6.10% |
| 1/8 | 1.69% | 1.71% | 1.31% |
| 1/10 | 1.09% | 1.10% | 0.77% |
| $0.0955$ ($\varepsilon=10^{-2}$) | $\le$1.00% | 1.00% | 0.793% |
| 1/16 | 0.43% | 0.43% | 0.33% |

**Assumptions.**

- Each peak is one isolated line. Blended peaks can have shallow or additional half-maximum crossings; no overlap FWHM guarantee follows. Production qualification reports local L1/L∞ alongside FWHM.
- Lines narrower than $W_q$ carry at most the aliased-weight share of $t_L^2$ proxy weight (`aliased_weight_limit`, the strictest `rtol`). This is not a bound on photon yield or peak error. Their own shape error can exceed $\varepsilon_\mathrm{shape}$.
- Peaks are read from the bin means without interpolation, as the #944 qualification harness reads them.

**Limiting cases.** As $\varepsilon_\mathrm{shape}\to0$, $h\to0$: pointwise sampling of the lineshape. At $h=W$ the bound exceeds 100% and the axis no longer resolves the peak; there `sinc-nyquist` keeps only yield exact.

## Budget

On the Fourier reduction ([`cxr-line-fourier-reduction`](../ledger-core-coherent-physics.md#cxr-line-fourier-reduction)), cost is nearly flat in bin count. #944 measured 43k→865k bins within +60% on CUDA and ROCm. `DEFAULT_BIN_MEAN_MAX_POINTS` is that largest measured axis, rounded, and is the default budget of every `bin-mean` axis. Above it the axis refuses; `resonance-local` is an explicit choice for MeV-wide measured bands. Before #1265 the `sinc-shape` selector rerouted to `resonance-local` there and recorded `shape_axis`.

## Anchors

- `tests/energy-grid/test_line_grid_shape_spacing.py::test_shape_step_bounds_single_line_height_and_fwhm`: the single-line bound at $\varepsilon=10^{-2}$ and $10^{-3}$ (tight to 2% at the worst offset).
- `test_bin_mean_oversampled_payload_carries_the_shape_tolerance_on_any_bandwidth`, `test_bin_mean_oversampled_axis_takes_the_shape_step_and_refuses_over_budget`: the payload and step on measured and kinematic axes.
- `test_measured_bin_mean_grid_is_uniform_at_the_shape_step` and `test_measured_bin_mean_grid_refuses_above_budget_instead_of_rerouting`: the runner's route.
- `tests/energy-grid/test_line_grid_selector_table.py::test_deprecated_sinc_shape_is_the_bin_mean_oversampled_step`: the alias.

Independent read-only verification (2026-11-16, implementation `c9da59b95`) rederived the isolated-line height factor, confirmed units, the first-zero convention, the proxy quantile, and the same-band budgeted fallback. Seven owning tests passed. Its discrepancy was the default FWHM table entry and overbroad overlap/FWHM wording; the corrections above address those findings. Production qualification on fixed tracks (#1230 Plan) and human sign-off remain outstanding.

## Qualification limits

The bounded ROCm pilot (`16afce359`, #1230, 2026-11-16) captured one primary
per seed on three seeds per scene. On HOPG 300 keV, 1 mm, reported worst
height/FWHM errors were 0.9062%/0.8431%; on h-BN 5 MeV, 1 mm, they were
0.4281%/0.2588%. Their 100 eV mass-L1 errors were at most
$3.47\times10^{-6}$. These are small fixed-population checks, not full
production-scale qualification. The sparse HOPG 1 MeV, 1 µm scene still
exposes weak-peak FWHM failures; window-edge maxima must be reported as
unresolved rather than treated as measured peaks. The harness records
unresolved peak and FWHM window counts, alongside windowed L1/L∞ metrics.

Uniform scoring is not automatically faster. A 100-primary thin-demo pilot
was slower on both CPU and ROCm despite its bounded band. Host scheduling
improvements reduce the overhead, but the remaining small-population
crossover, CUDA qualification and production-flush memory headroom are
open acceptance work. The isolated-line anchor does not qualify these
performance or blended-peak claims.
