# Validation: `cross-reflection-coherence`

## Claim and where it enters

`mc_spectrum` accumulates the coherent segment sum as one complex field per `(reflection, mosaic orientation)` row and squares each row separately, so distinct reflections add as **intensities**. Writing the total far-field amplitude at one photon energy as a sum over the listed reciprocal-lattice vectors,

$$
F(\omega)=\sum_{\mathbf g}F_{\mathbf g}(\omega),
\qquad
F_{\mathbf g}=\sum_j c_{j,\mathbf g}\,
e^{i[\omega t_{{\rm abs},j}-(\omega\hat{\mathbf n}+\mathbf g)\cdot\mathbf r_j]},
$$

the exact intensity is

$$
|F|^2=\underbrace{\sum_{\mathbf g}|F_{\mathbf g}|^2}_{\text{implemented}}
+\underbrace{\sum_{\mathbf g\neq\mathbf g'}
F_{\mathbf g}F_{\mathbf g'}^{*}}_{\text{dropped}} .
$$

The implementation keeps the first sum. The `mc_spectrum` docstring asserts the second is negligible because "their resonances are spectrally separated"; this record measures what that assertion is worth. The same drop is inherited by `coherent-line-hkl-batch`, which is an evaluation-order claim over an already per-row squaring, and by the incoherent path, where dropping cross terms is the definition of the model rather than an approximation.

## Two independent suppression mechanisms

**Spectral separation.** Each reflection radiates into a `sinc²` line of first-zero half-width $\Delta E=2\pi\hbar c/(\text{denom}\cdot t_L)$ centred on its own $\omega_{\rm res}=\mathbf v\cdot\mathbf g/(1-\hat{\mathbf n}\cdot\mathbf v)$. Cauchy--Schwarz bounds the cross term pointwise by the per-reflection intensities alone,

$$
\Bigl|\sum_{\mathbf g\neq\mathbf g'}F_{\mathbf g}F_{\mathbf g'}^{*}\Bigr|
\le\sum_{\mathbf g\neq\mathbf g'}\sqrt{S_{\mathbf g}S_{\mathbf g'}},
\qquad S_{\mathbf g}=|F_{\mathbf g}|^2,
$$

so wherever one line is strong the others sit far out on their own tails and the geometric mean collapses. This bound needs no access to the fields, which makes it usable against the shipped per-reflection spectra.

**Reciprocal-lattice decorrelation.** The bound above is worst-case in phase. The cross term also carries a residual spatial phase that the diagonal does not: the $j=k$ contributions to $|F_{\mathbf g}|^2$ are exactly $|c_j|^2$, real and non-cancelling, whereas the same-segment part of $F_{\mathbf g}F_{\mathbf g'}^{*}$ retains $\exp[-i(\mathbf g-\mathbf g')\cdot\mathbf r_j]$. Segment midpoints are spread over a sample thousands of lattice spacings deep and are uncorrelated with the lattice period, so $\langle e^{-i\Delta\mathbf g\cdot\mathbf r}\rangle$ random-walks toward $1/\sqrt{n_{\rm seg}}$. The cross term is therefore zero-mean over the ensemble, not merely bounded — the two mechanisms are independent, and the measurement reports both.

## Measurement

`checks/cross_reflection_coherence.py` calls the production coherent kernel once per reflection and once for the full list, over three configurations spanning the catalog's mosaic and non-mosaic basal-plane materials. The per-reflection pieces re-sum to the shipped all-reflections call to $1$–$3\times10^{-7}$ of peak in every configuration, which is float32 reassociation and confirms the split is a faithful decomposition of the shipped result rather than a separate model.

| configuration | segments | worst bound, bins $>10^{-3}$ peak | bound on integrated yield | $\max_{\rm pairs}\lvert\langle e^{-i\Delta\mathbf g\cdot\mathbf r}\rangle\rvert$ |
|---|---|---|---|---|
| hopg 30 keV / 500 nm | 26926 | $9.3\times10^{-2}$ | $1.0\times10^{-2}$ | $8.4\times10^{-3}$ |
| h-BN 30 keV / 921 nm | 51283 | $1.9\times10^{-1}$ | $1.4\times10^{-2}$ | $8.3\times10^{-3}$ |
| hopg 100 keV / 10 µm | 4076 | $4.7\times10^{-1}$ | $1.6\times10^{-2}$ | $4.0\times10^{-2}$ |

Per reflection, the bound at that reflection's own line centre separates sharply by how much yield the reflection carries. The forward harmonics $(0,0,2)$ and $(0,0,4)$ carry $100\%$ and $4$–$6\%$ of peak and are bounded at $9.4\times10^{-4}$–$1.4\times10^{-2}$ there. The anti-parallel $(0,0,-2)$ and $(0,0,-4)$ resonate only for back-scattered segments, carry $1.5\times10^{-6}$–$1.8\times10^{-5}$ of peak, and their own maxima sit where the bound is $1.2$–$1.5$ — Cauchy--Schwarz going vacuous on a feature that contributes no yield, not a failure of the approximation on anything observable.

The same reading applies to the worst qualifying bins in the table: they are *inter-line valleys* sitting at $1$–$3\times10^{-3}$ of peak, so the fraction is large exactly where the denominator is small. Unrestricted over all bins the bound reaches $1$–$3$, which is only the statement that the bound is vacuous where every reflection is on a sinc tail near a zero. The integrated column is the one insensitive to that effect, and it is the $1$–$1.6\%$ figure quoted elsewhere.

## Assumptions and limits

- The bound is Cauchy--Schwarz, so it is worst-case in relative phase and ignores the decorrelation factor entirely; the two columns must be read together, not multiplied.
- Measured on the catalog's basal-plane families $\{(0,0,\pm2),(0,0,\pm4)\}$, whose harmonics are separated by roughly their own resonance energy. It says nothing about a hypothetical reflection set with near-degenerate resonances at the chosen observation angle, where the bound approaches unity at the peak and the approximation would fail.
- Single-orientation runs. Mosaic orientations are squared per row for the same reason and are not covered here.
- The decorrelation factor is a property of the transport's midpoint distribution, so it is a Monte Carlo quantity: it shrinks with electron count and does not certify any single realization.

## Status

`filtered`. The cheap filters pass — the decomposition is exact to float reassociation, both suppression mechanisms are present and measured, and the dropped term is bounded well below the model's other stated uncertainties over the catalog's production reflection sets. Promotion to `rederived` needs a fresh-context derivation of the cross term with its decorrelation factor carried explicitly, rather than the bound used here. No regression test pins these numbers; the script is a diagnostic.
