# `tr-interface-energy-scale`: independent re-derivation

Fresh-context verification for issue #102. The derivation in Sections 1–5 was written from the cited sources (PDG 2020 §34.7.3; Jackson 3rd ed. §13.7; the Ginzburg–Frank boundary problem) before reading the anchor test. The exact-solution numbers in Section 6 come from a standalone script that imports neither `pyrite` nor its `transition_radiation` helpers.

## 1. Claim, quantity and assumptions

- Claim: a relativistic electron crossing one ideal vacuum/material interface radiates a total transition-radiation (TR) energy

  $$
  W_{\rm TR}\simeq\frac{\alpha}{3}\gamma\hbar\omega_p,
  \qquad \gamma=1+\frac{T}{m_ec^2},
  $$

  with a photon roll-off scale of order $\gamma\hbar\omega_p$. This is PDG 2020, *Passage of particles through matter*, Eq. (34.45), for unit charge.
- Location: research claim, `docs/research/physics/transition-radiation-recommendations.md`, label `eq-research-tr-interface-energy`. There is no production implementation. The ledger anchor is `tests/montecarlo/spectrum/test_transition_radiation.py::test_interface_energy_scale_from_the_exact_solution`.
- Assumptions: a single planar interface at normal incidence; uniform straight motion with no recoil; a plasma (Drude) dielectric $\varepsilon(\omega)=1-\omega_p^2/\omega^2$ for the material; vacuum on the other side; forward hemisphere; all photon energies and angles; no absorption, no foil interference, no detector response.

## 2. Exact single-interface emission (Ginzburg–Frank)

Medium $\varepsilon$ fills $z<0$ and vacuum fills $z>0$. A charge $e$ moves along $+\hat{\mathbf z}$ with speed $v=\beta c$ and crosses $z=0$ at $t=0$. Fourier-transform in $(x,y,t)$ with transverse wave number $q$ along $\hat{\mathbf x}$. Only TM fields $(E_x,H_y,E_z)$ are radiated. The forced (particle) field in a medium $\varepsilon_j$ has $k_z=\omega/v$. In Gaussian units, with the gauge $\nabla\!\cdot\!\mathbf A-i\varepsilon\omega\phi/c=0$,

$$
A_z=\frac{4\pi v\rho_0}{cD_j},\quad
\phi=\frac{4\pi\rho_0}{\varepsilon_jD_j},\quad
D_j=q^2+\frac{\omega^2}{v^2}-\varepsilon_j\frac{\omega^2}{c^2},
$$

so that $H_y^{(f)}=-iqA_z$ and $E_x^{(f)}=-iq\phi$, with $\rho_0\propto1/v$. Free waves are added on each side: $H_1$ in vacuum with $k_z=+k_1$, $E_x=(ck_1/\omega)H_1$; and $H_2$ in the medium with $k_z=-k_2$, $E_x=-(ck_2/\omega\varepsilon)H_2$. Here $k_1=(\omega/c)\cos\theta$ and $k_2=(\omega/c)\sqrt{\varepsilon-\sin^2\theta}$, where ${\rm Im}\,k_2\ge0$ makes the wave decay into the medium. Continuity of $H_y$ and $E_x$ at $z=0$ gives two linear equations. Their solution for $H_1$, evaluated at $q=(\omega/c)\sin\theta$ and normalized by the perfect-conductor limit $\varepsilon\to-\infty$ (whose result $e^2\beta^2\sin^2\theta/[\pi^2c(1-\beta^2\cos^2\theta)^2]$ is the image-charge result), gives the Ginzburg–Frank spectrum for forward emission into vacuum:

$$
\frac{d^2W}{d\omega\,d\Omega}
=\frac{e^2}{\pi^2c}\,
\frac{\beta^2\sin^2\theta\cos^2\theta\,\lvert\varepsilon-1\rvert^2\,
\bigl\lvert1-\beta^2-\beta\sqrt{\varepsilon-\sin^2\theta}\bigr\rvert^2}
{(1-\beta^2\cos^2\theta)^2\,
\bigl\lvert1-\beta\sqrt{\varepsilon-\sin^2\theta}\bigr\rvert^2\,
\bigl\lvert\varepsilon\cos\theta+\sqrt{\varepsilon-\sin^2\theta}\bigr\rvert^2}.
$$

**Checks of this form:**

- *Boundary solve versus closed form.* I solved the 2×2 system numerically and compared the result with the closed form above. Over $\varepsilon\in\{-3,\,-0.2+0.1i,\,0.5,\,0.7,\,1-10^{-3},\,5,\,2.5+i\}$, $\theta\in\{0.05,0.3,1.1,1.5\}$ and $\beta\in\{0.3,0.8,0.999\}$, the amplitude ratio is the constant $4\pi\beta$ to $4\times10^{-12}$, with zero phase spread. With $\rho_0\propto1/v$ this ratio is a constant normalization. The opposite sign choice, $1-\beta^2+\beta\sqrt{\cdot}$ over $1+\beta\sqrt{\cdot}$, which describes backward emission at entry, fails: its ratio varies over a factor $10^5$.
- *Perfect-conductor limit.* The closed form evaluated at $\varepsilon=-10^{12}$ agrees with $\beta^2\sin^2\theta/(1-\beta^2\cos^2\theta)^2$ to $10^{-12}$.
- *X-ray limit (Jackson 13.91).* Put $\varepsilon=1-\xi$ with $\xi=\omega_p^2/\omega^2\ll1$, and take $\theta\ll1$ and $\gamma\gg1$. Then $1-\beta\sqrt{\cdot}\simeq\tfrac12(\gamma^{-2}+\xi+\theta^2)$, $1-\beta^2-\beta\sqrt{\cdot}\simeq-1$, $\varepsilon\cos\theta+\sqrt{\cdot}\simeq2$ and $1-\beta^2\cos^2\theta\simeq\gamma^{-2}+\theta^2$. The spectrum becomes

  $$
  \frac{d^2W}{d\omega\,d\Omega}\simeq\frac{e^2}{\pi^2c}
  \left(\frac{\theta^2}{\gamma^{-2}+\theta^2}-\frac{\theta^2}{\gamma^{-2}+\omega_p^2/\omega^2+\theta^2}\right)^2,
  $$

  which is Jackson Eq. (13.91) exactly.

## 3. Total energy in the relativistic limit

Integrating Jackson 13.91 over the forward cone, $d\Omega\simeq\theta\,d\theta\,d\phi$, gives Jackson Eq. (13.92), also called the Garibian spectrum:

$$
\frac{dW}{d\omega}=\frac{e^2}{\pi c}F(\nu),\qquad
F(\nu)=(1+2\nu^2)\ln\!\Bigl(1+\frac1{\nu^2}\Bigr)-2,\qquad
\nu=\frac{\omega}{\gamma\omega_p}.
$$

With $e^2/c=\alpha\hbar$ and $d\omega=\gamma\omega_p\,d\nu$,

$$
W=\frac{\alpha}{\pi}\gamma\hbar\omega_p\int_0^\infty F(\nu)\,d\nu .
$$

**The integral.** At large $\nu$, $F\simeq1/(6\nu^4)$; at small $\nu$, $F\simeq\ln\nu^{-2}-2$. Both ends are integrable, and $\nu F\to0$ at both ends. One integration by parts gives

$$
\int_0^\infty F\,d\nu=-\int_0^\infty\nu F'\,d\nu
=\int_0^\infty\Bigl[4-4\nu^2L-\frac{2}{1+\nu^2}\Bigr]d\nu,
\qquad L=\ln\!\Bigl(1+\frac1{\nu^2}\Bigr).
$$

Next use $\nu^2L=\frac{d}{d\nu}\bigl(\tfrac13\nu^3L\bigr)+\tfrac23\,\nu^2/(1+\nu^2)$ and $\tfrac13\nu^3L=\nu/3-1/(6\nu)+\dots$ at large $\nu$. These give $\int_0^\infty(1-\nu^2L)\,d\nu=\tfrac23\cdot\tfrac\pi2=\pi/3$. Therefore

$$
\int_0^\infty F(\nu)\,d\nu=\frac{4\pi}{3}-\pi=\frac\pi3,
\qquad
\boxed{W=\frac{\alpha}{3}\gamma\hbar\omega_p}
$$

for unit charge; a charge $z e$ multiplies this by $z^2$. Numerical quadrature gives $\int_0^\infty F\,d\nu=1.047197552$, and $\pi/3=1.047197551$. This reproduces PDG Eq. (34.45), with matching factor, power of $\gamma$ and units.

**Roll-off.** $dW/d\omega$ is logarithmic for $\nu\ll1$ and falls as $\nu^{-4}$ for $\nu\gg1$, so the spectral scale is $\gamma\hbar\omega_p$. It is a smooth roll-off, not a sharp cutoff, as the claim states.

## 4. Cheap filters

| Check | Result |
| --- | --- |
| Units | $\alpha$ is dimensionless and $\gamma$ is dimensionless, so $W$ has the units of $\hbar\omega_p$ (energy). $e^2/(\pi^2c)$ has units of energy·time, so $d^2W/d\omega\,d\Omega$ is energy per unit angular frequency per steradian. Pass. |
| Matched medium | The amplitude is proportional to $\varepsilon-1$. With $\varepsilon\to1$ (equivalently $\omega_p\to0$) the boundary emission vanishes identically, and so does the estimate. Pass. |
| Scaling | For Drude $\varepsilon$, the exact spectrum depends on $\omega$ only through $x=\omega/\omega_p$. The exact $W/(\alpha\gamma\hbar\omega_p/3)$ is therefore a function of $\gamma$ alone, and $W\propto\omega_p$ holds exactly, not only asymptotically. Pass. |
| Relativistic validity, $T\to0$ | The exact amplitude carries an overall factor $\beta$, so $W\propto\beta^2\to0$ as $T\to0$. The estimate would give $\alpha\hbar\omega_p/3\neq0$ at $\gamma=1$. The note correctly excludes $T\to0$ extrapolation. Pass. |
| Sign/branch | ${\rm Im}\sqrt{\varepsilon-\sin^2\theta}\ge0$ (decay into the medium). For $\varepsilon<1$ there is no Cherenkov pole, because $\beta\sqrt{\varepsilon-\sin^2\theta}<1$. The forward-exit sign was confirmed against the boundary solve. Pass. |
| Finite foil / detected yield | The claim is for one interface's total energy over all angles and frequencies. Two-surface interference and a detected band or acceptance are outside it, and the research note says so. Pass. |

## 5. Table arithmetic

Inputs: $\hbar\omega_p=31$ eV, $m_ec^2=0.51099895$ MeV and $\alpha=7.2973525643\times10^{-3}$.

| $T$ (MeV) | $\gamma$ | $\gamma\hbar\omega_p$ (keV) | $\alpha\gamma\hbar\omega_p/3$ (eV) | Note table |
| --- | --- | --- | --- | --- |
| 10 | 20.570 | 0.6377 | 1.551 | 20.6, 0.64, 1.6 |
| 50 | 98.848 | 3.064 | 7.454 | 98.8, 3.1, 7.5 |
| 100 | 196.695 | 6.098 | 14.832 | 196.7, 6.1, 14.8 |

All nine entries agree after rounding.

## 6. Exact finite-$\gamma$ check (own implementation)

I integrated the Ginzburg–Frank closed form of Section 2 with $\varepsilon=1-1/x^2$, $x=\omega/\omega_p$, over $\theta\in[0,\pi/2]$ and over $x$ on $[10^{-6},\,2000\gamma]$. The quadrature is adaptive (`scipy.integrate.quad`), with breakpoints at $\theta\gamma\in\{0.3,1,3,10\}$ and at $x\in\{0.5,1,3,\gamma/10,\gamma,10\gamma\}$ in $\ln x$, and relative tolerance $10^{-8}$. The truncated tails are negligible: the integrand is finite as $x\to0$ (perfect-conductor limit), and the tail falls as $x^{-4}$ at large $x$. As a spot check against Garibian at $\gamma=197$, $(\pi/\alpha\hbar)\,dW/d\omega$ is 1.26837 exact versus 1.2715 asymptotic at $x=10$, and 0.025284 versus 0.025287 at $x=\gamma$.

| $\gamma$ | $W/(\alpha\gamma\hbar\omega_p/3)$ | deficit $1-r$ | $2/\gamma$ | $\gamma(1-r)$ | fraction of $W$ below $3\hbar\omega_p$ |
| --- | --- | --- | --- | --- | --- |
| 20.6 | 0.91671 | 0.0833 | 0.0971 | 1.72 | 0.516 |
| 40 | 0.95552 | 0.0445 | 0.0500 | 1.78 | 0.350 |
| 98.8 | 0.98156 | 0.0184 | 0.0202 | 1.82 | 0.191 |
| 197 | 0.99068 | 0.0093 | 0.0102 | 1.84 | 0.115 |

For comparison, the asymptotic Garibian spectrum puts 0.373 of $W$ below $3\hbar\omega_p$ at $\gamma=40$.

Comparison with the ledger row:

- $0.955$ at $\gamma=40$ is reproduced (0.9555).
- $0.990$ at $\gamma=197$: I get $0.9907$, which rounds to $0.991$. The $7\times10^{-4}$ difference is consistent with the anchor test's fixed-grid trapezoid quadrature (400 log-spaced photon points and 600 angle points), which I read only after this computation. It does not affect the physics claim or the test's acceptance band $1-2.5/\gamma<r<1$, which my values satisfy at both test energies.
- "Deficit about $2/\gamma$": the deficit is real, positive and $O(1/\gamma)$, but its coefficient is about $1.8$ and drifts slowly upward with $\gamma$, not 2. Calling it "about $2/\gamma$" is an acceptable order-of-magnitude statement. "About $1.8/\gamma$" would be more accurate.
- "Truncating the band at $3\hbar\omega_p$ loses about 40% at $\gamma=40$": dropping all emission below $3\hbar\omega_p$ removes 35.0% of the exact $W$ (37.3% in the asymptotic form). If "truncating" instead means keeping only photons below $3\hbar\omega_p$, the loss is 65%. As worded, the statement is ambiguous and the number is high. A precise wording would be "emission below $3\hbar\omega_p$ carries about 35% of $W$ at $\gamma=40$".

## 7. Comparison with the research note and anchor (read after Sections 1–6)

Equation `eq-research-tr-interface-energy` uses the same factor $\alpha/3$, the same first power of $\gamma$, the same $\hbar\omega_p$ and $\gamma=1+T/m_ec^2$ as the derivation in Section 3. The note qualifies its scope correctly: it is a relativistic estimate, roll-off rather than cutoff, matched-medium vanishing, no $T\to0$ extrapolation, no foil interference, and not a photon count or a bound on a detected band.

The anchor test integrates the repository's general planar solver over the forward hemisphere and photon energies $[10^{-4},10^{3}]\gamma\hbar\omega_p$, at $T=20$ and 100 MeV. It asserts only $1-2.5/\gamma<r<1$, which is consistent with Section 6. It checks the solver against the estimate, not against an external reference, and the ledger row says so.

Nothing diverges in factor, sign, exponent or unit.

Findings for the task owner (not edits made here):

1. The ledger row's $\gamma=197$ value should read $0.991$ (or $0.9907$), not $0.990$.
2. Make the "loses ~40%" note precise: emission below $3\hbar\omega_p$ is 35% of $W$ at $\gamma=40$.
3. Optionally state the deficit as about $1.8/\gamma$.
4. `docs/validation/index.md` lists radiation-physics write-ups explicitly, so this page needs a toctree entry, and the ledger row needs a write-up link.

## Verdict

`rederived`. PDG Eq. (34.45) follows exactly from Jackson 13.91/13.92, which in turn follow from an independently solved Ginzburg–Frank boundary problem. The table arithmetic is correct. The exact finite-$\gamma$ integration reproduces the ledger's 0.955 and agrees within quadrature rounding at $\gamma=197$ (0.9907 versus 0.990). The two secondary numerical remarks in the row (the "~40%" truncation loss and the "~2/γ" coefficient) are approximate; they need wording or rounding fixes, not a physics change.
