# `cr-transition-radiation-bound` re-derivation

Claim: `montecarlo/channeling/emission.py::{transition_radiation,transition_radiation_per_eV,transition_radiation_bound}`. Sources: Ginzburg & Frank (1946); Jackson, 3rd ed., §13.7 (spectral-angular, frequency and total TR); Ter-Mikaelian (1972).

## Independent derivation (before reading the bodies)

For X-rays the medium is $\epsilon=1-\xi^2$ with $\xi=\omega_p/\omega\ll1$. Write the far-field amplitude of a charge crossing a vacuum–medium boundary as the difference of the vacuum and medium "virtual photon" terms, $\beta\sin\theta\,[1/(1-\beta\cos\theta)-1/(1-\beta\sqrt\epsilon\cos\theta)]$. In the small-angle, ultrarelativistic limit, $1-\beta\cos\theta\simeq(\gamma^{-2}+\theta^2)/2$ and $1-\beta\sqrt\epsilon\cos\theta\simeq(\gamma^{-2}+\theta^2+\xi^2)/2$. Then

$$
\frac{d^2I}{d\omega\,d\Omega}=\frac{e^2}{\pi^2c}\,\theta^2\Bigl[\frac{1}{\gamma^{-2}+\theta^2}-\frac{1}{\gamma^{-2}+\theta^2+\xi^2}\Bigr]^2 .
$$

Dividing by $\hbar\omega$ for photon number and by $\hbar$ for $dE=\hbar\,d\omega$ gives

$$
\boxed{\frac{d^2N}{dE\,d\Omega}=\frac{\alpha}{\pi^2E}\,\theta^2\Bigl[\frac{1}{\gamma^{-2}+\theta^2}-\frac{1}{\gamma^{-2}+\theta^2+\xi^2}\Bigr]^2}
$$

The angle integral, with $u=\theta^2$, $a=\gamma^{-2}$ and $b=a+\xi^2$, is $\pi\int_0^\infty u[(a+u)^{-1}-(b+u)^{-1}]^2du=\pi[\tfrac{a+b}{b-a}\ln\tfrac ba-2]$. With $\nu=\omega/(\gamma\omega_p)$, $b/a=1+\nu^{-2}$ and $(a+b)/(b-a)=1+2\nu^2$, so

$$
\boxed{\frac{dN}{dE}=\frac{\alpha}{\pi E}\Bigl[(1+2\nu^2)\ln\Bigl(1+\frac1{\nu^2}\Bigr)-2\Bigr]},\qquad
\int_0^\infty F(\nu)\,d\nu=\frac\pi3\ \Rightarrow\ W=\frac{\alpha\gamma\hbar\omega_p}{3}.
$$

For $\nu\gg1$, $F\to1/(6\nu^4)$, so the spectrum vanishes as $(\gamma\omega_p/\omega)^4$; as $\omega_p\to0$ it vanishes identically.

**Two boundaries.** The entrance (vacuum→medium) and exit (medium→vacuum) amplitudes have opposite sign and relative phase $\varphi=t/Z_{\rm medium}$. The intensity is $\lvert1-e^{i\varphi}\rvert^2\le4$ times one boundary (with $\mathrm{e}^{-\mu t/2}$ on the entrance term, which only lowers it). Four times one boundary is therefore a strict upper bound for a slab in this model. For Si at 1 keV and $\gamma\approx7$ the formation length is $\sim0.02$ µm, much less than the 5 µm slab, and the attenuation length is $\sim3$ µm. The realistic factor is close to 1–2, so the bound is conservative by about 2–4.

**Include/exclude.** TR leaves the electron's transverse state unchanged, while a CR photon changes it. The final electron states are orthogonal, so TR and CR do not interfere and their intensities add. TR depends only on the boundary and $\omega_p$, not on the channeled state. Excluding it from the channeling channel, and keeping it as a recorded background bound owned by the separate TR work, is physically consistent.

**Drude $\omega_p$.** For Si with $n_e=14\times8/a^3$, $a=5.43102$ Å, $\hbar\omega_p=31.049$ eV (implementation: 31.050 eV). All 14 electrons are counted, which overestimates the response below the K edge (1.84 keV). This raises the TR at $\omega>\gamma\omega_p$ and is therefore conservative for a bound.

## Independent numerical checks

| check | result |
| --- | --- |
| $\int F(\nu)\,d\nu$ (adaptive `quad`) | 1.0471975512, equal to $\pi/3$ |
| $6\nu^4F(\nu)$ at $\nu=30$ | 0.99889, tending to 1 |
| angle integral of the spectral-angular form vs closed form, 3 and 5 MeV, $E=0.3$–3 keV and $10\gamma\hbar\omega_p$ | agree to $\le1.5\times10^{-5}$ |
| same at 17 MeV with breakpoints at $1/\gamma,10/\gamma$ | $1.811336\times10^{-9}$ vs $1.811336\times10^{-9}$ (3 keV) |
| `transition_radiation` vs own formula at $\theta=0.7/\gamma$ | $\le8\times10^{-9}$ relative (constant rounding) |
| `transition_radiation_per_eV` vs own closed form | equal to printed digits |
| emitted energy $\alpha\gamma\hbar\omega_p/3$ | 0.519 eV (3 MeV), 0.815 eV (5 MeV), 2.588 eV (17 MeV) |
| wide-angle amplitude form / small-angle form at $\theta=1/\gamma$, 1 keV | 0.946 (3 MeV), 0.978 (5 MeV), 0.999 (17 MeV) |

The last row quantifies the stated $O(\theta^2)$ validity caveat. At $\gamma\approx7$ the small-angle form overestimates by about 5%, which is covered by, and in the same conservative direction as, the factor-4 margin.

## Diff against the implementation

- `transition_radiation` and `transition_radiation_per_eV` are the boxed formulas with $\gamma=1+T/mc^2$ and `plasma_energy_eV`.
- `transition_radiation_bound` multiplies one boundary by 4 for both the band photons and the peak density at each forward line energy. It maximizes over $0\le\theta\le3/\gamma$ and compares with the CR spectrum maximum within one forward half width of the line. This matches the derivation. The CR peak density inherits the overestimated Lorentzian width of `cr-line-shape`, which can only lower the CR peak and so raise the TR/CR ratio, again conservative.
- The equation numbers "Jackson Eqs. 13.91–13.92" could not be checked offline. The formulas are the §13.7 results.

## Verdict

`rederived`. The include/exclude decision is recorded and physically consistent.

## Resolution (worker, after the verdict)

The formulas are unchanged. With the corrected `cr-line-shape` widths, the CR line peaks are higher and now include capture, so the recorded TR/CR peak ratios fall:

- 3 MeV: from 3.2–6.6% to 2.8–4.8%
- 5 MeV, strong lines: from 0.6–0.7% to 0.30–0.38%
- 10 MeV: from ≤ 2.9% to ≤ 1.5%
- 5 MeV, weak 0.63 keV line: from ≈ 110 to ≈ 31

`transition_radiation_bound` now compares only at bound→bound line energies. The exclude-with-bound decision stands.
