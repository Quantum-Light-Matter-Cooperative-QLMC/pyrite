# `grating-slit-image-width` — independent verification

**Code**: `src/pyrite/detectors/grating_image.py::{slit_image_width_mm, slit_image_pixels}` (helpers `_column_kernel`, `_aberration_sigma_mm`, `_kernel_cdf`, `_gauss_integrals`); focal-plane map `src/pyrite/detectors/grating.py::_focal_plane` / `FlatFieldMount`.
**Ledger claim**: a slit of width $w$ at entrance arm $r$ images at fixed $\lambda$ to $\Delta\beta = w\cos\alpha/(r\cos\beta)$, width $w' = w(\cos\alpha/\cos\beta)(r'/r)$ normal to the ray and $w'/\cos(\beta-\beta_H)$ on the focal plane; each column's point-slit flux is convolved with rect(width) $\otimes$ rect(pixel) $\otimes$ optional Gaussian of FWHM $\lvert ds/d\lambda\rvert\,\lambda/R$, conserving flux.
**Source**: differential of the grating equation (Samson & Ederer, *Vacuum Ultraviolet Spectroscopy*, Academic 2000); focal-plane map of `flat-field-focal-plane`; closed-form CDF of a rect $\otimes$ rect $\otimes$ Gaussian kernel.

All numbers below are synthetic (a 1200 l/mm grating at $\alpha = 87^\circ$, $r = 350$ mm, $r_H' = 250$ mm, $\beta_H = -1.45$ rad, 13.5 µm pixels). No vendor grating value is used.

## 1. Scope

- `slit_image_width_mm(E_eV, grating, slit_width_mm, *, entrance_arm_mm=None, distance_mm=None, beta_ref_rad=None)` returns the slit-image width on the focal plane [mm]. NaN for non-propagating energies.
- `slit_image_pixels(E_grid_eV, spec, grating, ccd, slit, ...)` returns `(pixel_centers_mm, flux_per_pixel)` in the integral units of `spec`.
- Conventions (module docstring): $\sin\alpha + \sin\beta = m\lambda/d$, angles from the grating normal; flat focal plane with unit normal at $\beta_H$, perpendicular distance $r_H'$; dispersion coordinate $s(\beta) = r_H'\tan(\beta-\beta_H)$. The explicit-distance variant is $s = L\tan(\beta-\beta_{\rm ref})$.
- Stated assumptions: point-to-point imaging at first order in the slit width, no defocus or aberration beyond the optional Gaussian; source-pixel flux uniform across its pitch; kernel width and $\sigma$ evaluated at each pixel column's focal-plane position.

## 2. Independent derivation (before reading the implementation body)

### 2.1 Angular image of the slit

At fixed $\lambda$ the grating equation gives

$$
\cos\alpha\,d\alpha + \cos\beta\,d\beta = 0
\quad\Rightarrow\quad
\lvert d\beta\rvert = \frac{\cos\alpha}{\cos\beta}\,\lvert d\alpha\rvert .
$$

A slit of width $w$ normal to the incident chief ray at distance $r$ subtends $\lvert d\alpha\rvert = w/r$ (small angle), so

$$
\Delta\beta = \frac{w\cos\alpha}{r\cos\beta}.
$$

Normal to the diffracted ray at path length $r'$ this is the anamorphic width $w' = r'\Delta\beta = w(\cos\alpha/\cos\beta)(r'/r)$. At grazing incidence $\cos\alpha \ll \cos\beta$, so $w' \ll w\,r'/r$ (demagnified image).

### 2.2 Projection onto the focal plane

A ray at $\beta$ meets the plane $\{\mathbf p : \mathbf p\cdot\hat{\mathbf n}_H = r_H'\}$ after $r'(\beta) = r_H'/\cos(\beta-\beta_H)$ and lies $s = r'\sin(\beta-\beta_H) = r_H'\tan(\beta-\beta_H)$ along the plane from the foot of the perpendicular. Then

$$
\frac{ds}{d\beta} = \frac{r_H'}{\cos^2(\beta-\beta_H)} = \frac{r'(\beta)}{\cos(\beta-\beta_H)},
$$

so the focal-plane width is

$$
\boxed{w_{\rm plane} = \Bigl\lvert \frac{ds}{d\beta}\Bigr\rvert\,\Delta\beta
= w\,\frac{\cos\alpha}{\cos\beta}\,\frac{r'}{r}\,\frac{1}{\cos(\beta-\beta_H)}}
$$

which reduces to $w'$ when $\beta = \beta_H$. For the explicit-distance map $s = L\tan(\beta-\beta_{\rm ref})$ the same steps give $w_{\rm plane} = \Delta\beta\,L/\cos^2(\beta-\beta_{\rm ref})$, and $L\,\Delta\beta$ at $\beta_{\rm ref} = \beta$.

### 2.3 Kernel CDF of rect $\otimes$ rect $\otimes$ Gaussian

Let $\Phi_\sigma(v) = \Phi(v/\sigma)$ be the $N(0,\sigma)$ CDF, $\varphi$ the standard normal density, and define

$$
A(v) = \int_{-\infty}^{v}\Phi_\sigma, \qquad B(v) = \int_{-\infty}^{v} A .
$$

Differentiating the candidates confirms

$$
A(v) = v\,\Phi(v/\sigma) + \sigma\,\varphi(v/\sigma), \qquad
B(v) = \tfrac12(v^2+\sigma^2)\,\Phi(v/\sigma) + \tfrac12 v\sigma\,\varphi(v/\sigma),
$$

using $\tfrac{d}{dv}\varphi(v/\sigma) = -(v/\sigma^2)\varphi(v/\sigma)$: $A' = \Phi + (v/\sigma)\varphi - (v/\sigma)\varphi = \Phi$ and $B' = v\Phi + \bigl[(v^2+\sigma^2)/(2\sigma) + \sigma/2 - v^2/(2\sigma)\bigr]\varphi = A$. Both vanish as $v\to-\infty$.

Convolving the Gaussian CDF with unit-area rects of widths $p$ (pixel) and $w$ (slit image) gives

$$
F(u) = \frac{1}{wp}\int_{-w/2}^{w/2}\!\!\int_{-p/2}^{p/2}\Phi_\sigma(u-a-b)\,db\,da
= \frac{B\bigl(u+\tfrac{w+p}{2}\bigr) - B\bigl(u+\tfrac{w-p}{2}\bigr) - B\bigl(u-\tfrac{w-p}{2}\bigr) + B\bigl(u-\tfrac{w+p}{2}\bigr)}{wp}.
$$

Limits:

- $w\to0$: the $w$-difference becomes a derivative, $F(u) = [A(u+p/2) - A(u-p/2)]/p$.
- $\sigma\to0$: $A\to\max(v,0)$, $B\to\tfrac12\max(v,0)^2$, so $F$ is the trapezoid CDF of rect $\otimes$ rect. With $w\to0$ as well it is the uniform pixel CDF, and the transfer to pixels on the same grid is exactly $\delta_{k0}$.
- Normalization: for large $u$, $B(v)\to v^2/2 + \sigma^2/2$. The four constants cancel ($1-1-1+1$) and the quadratics leave $a^2 - b^2 = wp$ with $a = (w+p)/2$, $b = (w-p)/2$, so $F(+\infty) = 1$.
- The kernel is even, so $F(u) = 1 - F(-u)$. Evaluating only at $-\lvert u\rvert$ keeps every $B$ small and avoids cancellation in the tails.

Pixel transfer from a source pixel at $c$ to the destination pixel $[e_j, e_{j+1}]$ is $F(e_{j+1}-c) - F(e_j-c)$. Summed over all pixels this telescopes to $F(+\infty) - F(-\infty) = 1$, so flux is conserved apart from what falls off the sensor ends. Truncating at $\lvert u\rvert \le (w+p)/2 + 8\sigma$ loses at most $\Phi(-8) \approx 6\times10^{-16}$ per side.

### 2.4 Aberration width from a resolving power

$R = \lambda/\Delta\lambda_{\rm FWHM}$, so $\Delta s_{\rm FWHM} = \lvert ds/d\beta\rvert\,\lvert d\beta/d\lambda\rvert\,\lambda/R$ with $d\beta/d\lambda = m/(d\cos\beta)$. Then $\sigma = \Delta s_{\rm FWHM}/(2\sqrt{2\ln2})$, where $2\sqrt{2\ln 2} = 2.35482$. Equivalently $\sigma = \lvert ds/dE\rvert\,E/(2.35482\,R)$.

## 3. Diff against the implementation

| Item | Independent | Code | Result |
|---|---|---|---|
| $\Delta\beta$ | $w\cos\alpha/(r\cos\beta)$ | `slit_width_mm * cos(alpha) / (entrance_arm_mm * cos(beta))` (`grating_image.py:163`, `:229`) | match |
| $ds/d\beta$, flat field | $r_H'/\cos^2(\beta-\beta_H)$ | `focal_distance_mm / cos(beta - focal_normal_rad)**2` (`grating.py:424`) | match |
| $ds/d\beta$, explicit | $L/\cos^2(\beta-\beta_{\rm ref})$ | `distance / cos(beta - reference)**2` (`grating.py:429`) | match |
| width | $\lvert ds/d\beta\cdot\Delta\beta\rvert$ | `np.abs(slope * dbeta)` (`:164`, `:230`) | match |
| $\sigma$ | $\lvert ds/d\beta\rvert\, m/(d\cos\beta)\,\lambda/R/2\sqrt{2\ln2}$ | `_aberration_sigma_mm` (`:171`–`:173`), $\lambda$ and $d$ both in Å | match |
| $A$, $B$ | §2.3 | `_gauss_integrals` (`:185`–`:193`) with `ndtr` and $e^{-z^2/2}/\sqrt{2\pi}$; step limits below $\sigma = 10^{-12}$ mm | match |
| $F(u)$ | §2.3, even-kernel trick | `_kernel_cdf` (`:206`–`:222`); `narrow` branch for $w \le 10^{-12}$ mm | match |
| window | $(w+p)/2 + 8\sigma$ | `reach` (`:312`); CDF evaluated at window-edge pixel boundaries | match |
| $w=0$, no aberration | point-slit image | early return of `bin_to_pixels` output (`:308`–`:309`) | exact |

The code is a term-by-term match. The kernel is applied to each lit source pixel, with width and $\sigma$ interpolated from the energy nodes to the pixel's focal-plane position (`_at_columns`). This is the documented "evaluated at the column" approximation of a wavelength-dependent kernel.

## 4. Numeric checks

Scripts are in the session scratchpad. Each comparison uses an independent reference, not an implementation helper.

- $A$, $B$ against nested `scipy.integrate.quad` of $\Phi_\sigma$ at $\sigma = 0.7$, $v\in[-5,5]$: max error $3.5\times10^{-14}$ and $1.9\times10^{-12}$.
- $F(u)$ at $\sigma = 0$ against the exact trapezoid CDF ($w = 0.003, 0.05, 0.3$ mm, $p = 0.0135$ mm): max error $2.8\times10^{-13}$.
- $F(u)$ with a Gaussian ($w,\sigma$ = $(0.05, 0.01)$ and $(0.003, 0.002)$ mm) against a 2001 $\times$ 2001 midpoint quadrature of $\Phi_\sigma$ over both rects: max error $2\times10^{-8}$, which is the quadrature error.
- Normalization: $F(-\text{reach}) \le 1.7\times10^{-17}$, $1-F(+\text{reach}) = 0$, $F(0) = 0.5$ to rounding.
- $w=0$, $\sigma=0$ pixel transfer on the same grid: exactly $[0,0,0,1,0,0]$.
- Width at $E = 120$ eV, $w = 0.04$ mm: the code gives $9.06068\times10^{-3}$ mm. The boxed formula agrees to $1.5\times10^{-9}$ relative; the residual comes from $\beta$ differing at $10^{-10}$ rad because of the $hc$ constant. A ray-level finite difference, $\lvert s(\alpha+h/2) - s(\alpha-h/2)\rvert$ with $h = w/r$, agrees to $5\times10^{-8}$ relative, the second-order term.
- The explicit map with $L = 200$ mm, $r = 400$ mm and $\beta_{\rm ref} = \beta + 0.05$ agrees to $1.6\times10^{-9}$ relative.
- $\sigma$ at $R = 500$ against a central difference of $s(E)$: $\lvert ds/dE\rvert E/(2.35482R)$ agrees to $3\times10^{-10}$ relative.
- Flux: for a line at 120 eV through $w = 0.04$ mm with no aberration, with $R = 500$, and with $w = 0$ at $R = 500$, the total equals the point-slit total to $\le 2\times10^{-16}$ relative.
- Full image against a brute-force reference: a narrow line (5 lit pixels) was convolved with the fixed-$E_0$ kernel by quadrature. The closed form with a fixed kernel matches the quadrature to $8\times10^{-8}$ of the peak. The production image differs from it by $7\times10^{-5}$ of the peak, all of which comes from the per-column variation of width and $\sigma$ (up to $10^{-3}$ relative across the lit pixels).
- Anchors: `tests/detectors/test_grating_image.py` passes, 12 tests.

## 5. Filters

- **Units**: $\Delta\beta$ is mm/mm, so dimensionless. Width and $\sigma$ are in mm via $ds/d\beta$ [mm/rad]. $\lambda/d$ uses Å/Å. $F$ is dimensionless, and $B/(wp)$ has units of mm$^2$/mm$^2$. **Pass.**
- **Limits**: $w=0$ returns the point-slit image exactly. $w\to0$ converges, and so does $\sigma\to0$. $\beta = \beta_H$ gives $w'$, and $\beta_{\rm ref} = \beta$ gives $L\,\Delta\beta$. $F(\pm\infty) = 0, 1$. **Pass.**
- **Signs/conventions**: the absolute value removes the sign of $\cos\beta$ and of $d\beta/d\alpha$. The even-kernel trick is valid because every factor is symmetric. The module's $\sin\beta = m\lambda/d - \sin\alpha$ convention is used throughout. **Pass.**
- **Notation**: the ledger row, docstrings and code agree, including the FWHM $\lvert ds/d\lambda\rvert\lambda/R$ and the projection $1/\cos(\beta-\beta_H)$. **Pass.**

## 6. Observations (not discrepancies)

1. **Small-width cancellation.** The `wide` branch is used for $w > 10^{-12}$ mm (`_ZERO_WIDTH_MM`, `grating_image.py:29`). Its second difference loses about $\epsilon\,p/w$ relative accuracy. Measured $\max\lvert F_w - F_0\rvert$ is $3\times10^{-12}$ at $w = 10^{-7}$ mm, $4\times10^{-10}$ at $10^{-9}$ mm and $6\times10^{-8}$ at $10^{-11}$ mm. Physical slits ($\ge 1$ µm) are unaffected. A switch threshold near $10^{-6}\,p$ would keep the error below $10^{-10}$.
2. **Explicit-map default reference.** With `distance_mm` given and `beta_ref_rad=None`, $\beta_{\rm ref}$ is the `nanmean` of the $\beta$ passed to the call (`grating.py:427`). For a scalar call to `slit_image_width_mm`, that means $\beta_{\rm ref} = \beta$ (normal incidence), whereas `slit_image_pixels` uses the grid mean, consistent with `disperse_spectrum`. The anchor `test_slit_image_width_explicit_distance_map` relies on the scalar behavior. Callers comparing a scalar width to an image under the explicit map should pass `beta_ref_rad`.
3. **First-order imaging.** $\Delta\beta$ is linear in $w/r$, and the slit is assumed normal to the incident chief ray. The finite-difference check shows the second-order correction is about $10^{-7}$ relative at $w/r \sim 10^{-4}$.
4. **Dependency.** The focal-plane map belongs to `flat-field-focal-plane`, which is still `unverified` in the ledger. §2.2 re-derives both $s(\beta)$ and $ds/d\beta$, so this claim's verdict does not rest on that row.

## 7. Verdict

**rederived.** The anamorphic width, the plane-tilt projection, the explicit-distance variant, the closed-form CDF and its limits, the normalization and windowing, the resolving-power $\sigma$ and the exact $w=0$ reduction all match an independent derivation. They also agree numerically against independent references.
