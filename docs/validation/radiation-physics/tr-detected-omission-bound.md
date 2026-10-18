# `tr-detected-omission-bound`

## Claim and source

- Claim: exact classical transition radiation (TR) of a straight, uniform-velocity point charge through a planar stack of homogeneous isotropic layers. It is a boundary-value solve with tangential $\mathbf E$/$\mathbf H$ continuity, valid for any $\varepsilon$, speed, incidence, emission direction and layer count. A per-observable omission bound $b=\max_E T(E)/S(E)\le\text{target\_rse}/3$ applies to positive linear detector responses.
- Code: `montecarlo/spectrum/transition_radiation.py::planar_transition_radiation`, `ginzburg_frank_interface`, `medium_dielectric`, `omission_tolerance`, `transition_radiation_omission`; study `checks/transition_radiation_omission.py`.
- Sources: Ginzburg & Tsytovich 1990, Ch. 2 (Ginzburg–Frank boundary problem); Jackson 3rd ed. §13.7, Eq. 13.91; DeRaad, Tsai & Erber, Phys. Rev. D 18, 2152 (1978) (angle integral).
- Units: Gaussian, $\hbar=c=1$, energies in eV, $e^2=\alpha$; thicknesses in Å converted with $\hbar c$. Output in photons eV$^{-1}$ sr$^{-1}$ per crossing electron.

The derivation below was written from the ledger row and the docstrings only. The implementation bodies were read afterwards (see "Comparison with the implementation").

## Independent derivation

### Particular field of the moving charge

Take $\mathbf E(\mathbf r,t)=\int\frac{d^3k\,d\omega}{(2\pi)^4}\,\mathbf E(\mathbf k,\omega)\,e^{i(\mathbf k\cdot\mathbf r-\omega t)}$. For $\rho=e\,\delta(\mathbf r-\mathbf vt)$,

$$
\rho(\mathbf k,\omega)=2\pi e\,\delta(\omega-\mathbf k\cdot\mathbf v),\qquad \mathbf j=\rho\,\mathbf v .
$$

In an unbounded non-magnetic medium, $i\mathbf k\cdot\varepsilon\mathbf E=4\pi\rho$, $i\mathbf k\times\mathbf E=i\omega\mathbf H$ and $i\mathbf k\times\mathbf H=-i\omega\varepsilon\mathbf E+4\pi\mathbf j$. Eliminating $\mathbf H$ gives

$$
\mathbf k(\mathbf k\cdot\mathbf E)-k^2\mathbf E+\varepsilon\omega^2\mathbf E=-4\pi i\omega\mathbf j ,
\qquad \mathbf k\cdot\mathbf E=-\frac{4\pi i\rho}{\varepsilon},
$$

so that

$$
\mathbf E=\frac{4\pi i\,\rho\,(\varepsilon\omega\mathbf v-\mathbf k)}{\varepsilon(k^2-\varepsilon\omega^2)},
\qquad
\mathbf H=\frac{\mathbf k\times\mathbf E}{\omega}=\frac{4\pi i\,\rho\,(\mathbf k\times\mathbf v)}{k^2-\varepsilon\omega^2}.
$$

Go to the mixed representation $(\mathbf k_\perp,z,\omega)$ with $\int dk_z/(2\pi)$. Because $\delta(\omega-\mathbf k_\perp\cdot\mathbf v_\perp-k_zv_z)=\delta(k_z-q)/\lvert v_z\rvert$ with $q=(\omega-\mathbf k_\perp\cdot\mathbf v_\perp)/v_z$, the factor $2\pi e/(2\pi\lvert v_z\rvert)$ replaces $\rho$:

$$
\boxed{\phi=\frac{4\pi e}{\lvert v_z\rvert\,\varepsilon\,(k^2-\varepsilon\omega^2)},\quad
\mathbf E_p=i\phi(\varepsilon\omega\mathbf v-\mathbf k)\,e^{iqz},\quad
\mathbf H_p=i\varepsilon\phi\,(\mathbf k\times\mathbf v)\,e^{iqz},\quad \mathbf k=(\mathbf k_\perp,q).}
$$

This reproduces the ledgered $\phi$, $\mathbf E_p$, $\mathbf H_p$ and the factor $1/\lvert v_z\rvert$. Here $\mathbf E(\mathbf r,t)=\int\frac{d^2k_\perp\,d\omega}{(2\pi)^3}\,\mathbf E(\mathbf k_\perp,z,\omega)\,e^{i(\mathbf k_\perp\cdot\mathbf r_\perp-\omega t)}$.

### Free waves and boundary conditions

The homogeneous solutions are $\mathbf E\,e^{\pm i\lambda z}$ with $\lambda=\sqrt{\varepsilon\omega^2-k_\perp^2}$ and $\operatorname{Im}\lambda\ge0$, so the semi-infinite media keep only waves that decay or propagate away from the stack. Transversality gives $E_z=-(\mathbf k_\perp\cdot\mathbf E_\perp)/K_z$, and $\mathbf H=\mathbf K\times\mathbf E/\omega$ follows. Per interface there are four conditions: continuity of $E_x,E_y,H_x,H_y$. Each wave carries two tangential amplitudes. For $L$ layers there are $2(2L+2)=4(L+1)$ unknowns and $4(L+1)$ equations, so the system is square.

### Far field and photon normalization

In the vacuum above, the radiation part is $\mathbf E(\mathbf r,\omega)=\int\frac{d^2k_\perp}{(2\pi)^2}\,\mathbf a(\mathbf k_\perp)\,e^{i(\mathbf k_\perp\cdot\mathbf r_\perp+\lambda z)}$. The Weyl identity $e^{i\omega R}/R=\frac{i}{2\pi}\int d^2k_\perp\,e^{i(\mathbf k_\perp\cdot\mathbf r_\perp+\lambda\lvert z\rvert)}/\lambda$ fixes the stationary-phase factor:

$$
\int d^2k_\perp\,f(\mathbf k_\perp)\,e^{i(\mathbf k_\perp\cdot\mathbf r_\perp+\lambda z)}\simeq-2\pi i\,\omega\cos\theta\,f(\omega\mathbf n_\perp)\,\frac{e^{i\omega R}}{R},
$$

so $\lvert\mathbf E(\mathbf r,\omega)\rvert=\omega\cos\theta\,\lvert\mathbf a\rvert/(2\pi R)$. With $\mathbf S=\frac{1}{4\pi}\mathbf E\times\mathbf H$, $\lvert\mathbf H\rvert=\lvert\mathbf E\rvert$ and $\int dt\,\lvert\mathbf E\rvert^2=2\int_0^\infty\frac{d\omega}{2\pi}\lvert\mathbf E(\omega)\rvert^2$, the energy spectrum is

$$
\frac{d^2W}{d\omega\,d\Omega}=\frac{R^2\lvert\mathbf E(\mathbf r,\omega)\rvert^2}{4\pi^2}=\frac{\omega^2\cos^2\theta\,\lvert\mathbf a\rvert^2}{16\pi^4}.
$$

Dividing by $\hbar\omega$ with $\hbar=1$ gives

$$
\boxed{\frac{d^2N}{dE\,d\Omega}=\frac{\omega\cos^2\theta\,\lvert\mathbf a\rvert^2}{16\pi^4}.}
$$

Here $\lvert\mathbf a\rvert^2$ is the full vector norm, which is the sum over both polarizations, and $\mathbf a\propto e$, so $\lvert\mathbf a\rvert^2\propto e^2=\alpha$. This matches the ledgered far-field factor.

### Closed form at normal incidence

For $\mathbf v=v\hat{\mathbf z}$ and $\mathbf k_\perp=k_x\hat{\mathbf x}$, only p-polarization is radiated. The particular fields are $H_{p,y}=-i\varepsilon\phi v k_x$ and $E_{p,x}=-i\phi k_x$, with $qv=\omega$. Write $h_1$ for the medium's downward wave ($E_x=-\lambda_1h_1/(\omega\varepsilon)$) and $h_2$ for the vacuum's upward wave ($E_x=\lambda_2h_2/\omega$). Continuity at $z=0$ gives

$$
h_2=\frac{\omega\varepsilon\,[E_{p,x}(\varepsilon)-E_{p,x}(1)]-\lambda_1[H_{p,y}(1)-H_{p,y}(\varepsilon)]}{\varepsilon\lambda_2+\lambda_1},
\qquad \lvert\mathbf a\rvert=\lvert h_2\rvert .
$$

Substitute $k_x=\omega\sin\theta$, $\lambda_2=\omega\cos\theta$, $\lambda_1=\omega s$ with $s=\sqrt{\varepsilon-\sin^2\theta}$, and $q=\omega/v$. The result simplifies (sympy, then checked numerically at eight complex-$\varepsilon$ points for both senses, agreement $\le3\times10^{-15}$) to

$$
\frac{d^2N}{dE\,d\Omega}=\frac{\alpha\beta^2\sin^2\theta\cos^2\theta}{\pi^2E}
\left\lvert\frac{(\varepsilon-1)(1-\beta^2-\beta s)}{(1-\beta^2\cos^2\theta)(1-\beta s)(\varepsilon\cos\theta+s)}\right\rvert^2 .
$$

This is the Ginzburg–Frank form for a charge leaving $\varepsilon$ into vacuum. Taking $v<0$ (a charge entering the medium, with $\theta$ measured from $+\hat{\mathbf z}$, the reversed velocity) gives the same expression with $\beta\to-\beta$.

Limits:

- **Perfect conductor**, $\lvert\varepsilon\rvert\to\infty$: $s\simeq\sqrt\varepsilon$, so $(\varepsilon-1)(1-\beta^2-\beta s)/[(1-\beta s)(\varepsilon\cos\theta+s)]\to1/\cos\theta$. The result is $\alpha\beta^2\sin^2\theta/[\pi^2E(1-\beta^2\cos^2\theta)^2]$ for either sign of $\beta$. This is the charge-plus-image sudden-stop result.
- **Oblique conductor**: the image amplitude is $\frac{\alpha}{4\pi^2E}\left\lvert\mathbf n\times\left(\frac{\boldsymbol\beta}{1-\mathbf n\cdot\boldsymbol\beta}-\frac{\bar{\boldsymbol\beta}}{1-\mathbf n\cdot\bar{\boldsymbol\beta}}\right)\right\rvert^2$ with $\bar{\boldsymbol\beta}=(\beta_x,\beta_y,-\beta_z)$. At normal incidence it reduces to the line above.
- **Jackson Eq. 13.91**: put $\varepsilon=1-\xi^2$ with $\xi=\omega_p/\omega\ll1$, and take $\gamma\gg1$, $\theta\ll1$. Then $(\varepsilon-1)/(\ldots)$ expands to $\theta\left[\frac{1}{\gamma^{-2}+\theta^2+\xi^2}-\frac{1}{\gamma^{-2}+\theta^2}\right]$, giving $\frac{\alpha}{\pi^2E}\theta^2[\ldots]^2$.
- **Zero contrast** $\varepsilon=1$: the factor $(\varepsilon-1)$ vanishes.
- **Hemisphere integral, conductor**: $2\pi\int_0^1\frac{\beta^2(1-x^2)}{(1-\beta^2x^2)^2}dx\cdot\frac{\alpha}{\pi^2E}=\frac{\alpha}{\pi E}\left[\frac{1+\beta^2}{2\beta}\ln\frac{1+\beta}{1-\beta}-1\right]$. This is the ideal-conductor limit of the DeRaad–Tsai–Erber angle integral.

### Dielectric function

With $e^{-i\omega t}$, absorption means $\operatorname{Im}\varepsilon>0$ and $n=1-\delta+i\beta_n$. The forward-scattering susceptibility is $\chi=-\frac{r_e\lambda^2}{\pi}\sum_in_i f_i(0)$, with the complex factor $f=Z+f'-if''$ in this convention because $f''>0$ must give $\operatorname{Im}\chi>0$. Then

$$
\varepsilon=1-\frac{r_e\lambda^2}{\pi}\sum_in_i\,(Z_i+f'_i-if''_i).
$$

`chi_0` uses $e^{+i\omega t}$, $\chi_0=-\frac{r_e\lambda^2}{\pi V}\sum(Z+f'+if'')$, so $\varepsilon=1+\overline{\chi_0}$ with $n_i=N_i/V_{\rm cell}$. This uses the exact relation $k^2=\varepsilon\omega^2$, with no $n\simeq1-\delta$ linearization.

### Omission bound

Let $R$ be any linear map with a non-negative kernel, $O_X=\int K(x,E)\,X(E)\,dE$ with $K\ge0$. Then $0\le T\le cS$ on the support of $K$ gives $0\le O_T\le cO_S$, so the relative change of any observable with $O_S>0$ lies in $[0,c]$.

Consequences:

1. **Precondition at $S(E)=0$.** The premise must hold at every energy in the band. If $S(E)=0$ while $T(E)>0$, then $c=\infty$.
2. **Net line after sideband subtraction.** Let $\Delta L=R_W[T]-\kappa R_{sb}[T]$. If $S$ is the continuum only, then $R_W[T]\le cB$ and $\kappa R_{sb}[T]\le c\hat B\simeq cB$, so $\lvert\Delta L\rvert/L\le cB/L$. If $S$ includes the line itself, only $\lvert\Delta L\rvert/L\le c(1+B/L)$ follows. The ledgered $cB/L$ therefore relies on the continuum-only denominator, which the study uses.
3. **Coherent observables.** A coherent signal adds amplitudes, $\lvert A_s+A_T\rvert^2-\lvert A_s\rvert^2\le2\lvert A_s\rvert\lvert A_T\rvert+\lvert A_T\rvert^2$. The relative change is then of order $2\sqrt c$, not $c$, so it is correctly excluded.
4. **Non-linear detector effects** (pile-up, dead time) are not positive linear maps and are outside the bound.

Tolerance: a deterministic bias $b$ and a statistical error add in mean square, $\sqrt{\text{rse}^2+b^2}$. For $b\le\text{rse}/3$ this is at most $\text{rse}\sqrt{1+1/9}=\text{rse}\sqrt{10/9}=1.0541\,\text{rse}$.

Crossing count: a flat slab, or a half-space bounded by a plane, is convex, and field-free vacuum trajectories are straight. A primary therefore enters once and leaves through a face at most once; once outside, it cannot return. That makes at most 2 crossings per primary. Escaping secondaries add their mean number per primary, and the bound is on the expectation. The factor 3 therefore needs fewer than one escaping secondary per primary, each radiating no more than the maximum single crossing at speed $\le\beta_0$. For sub-relativistic electrons TR scales roughly as $\beta^2$ (monotone; checked below), so slow secondaries are covered. Two assumptions remain:

- **Entry/exit phase.** Entry and exit on the same face by the same electron add as amplitudes. The expectation equals the intensity sum only when their relative phase is random across electrons, that is, when the path between them is long compared with the formation length. The worst case for a coherent pair is $4\times$ one crossing, not $2\times$.
- **Sampled, not certified, sup.** "Maximized over directions" is a sampled maximum.

## Comparison with the implementation

| Item | Independent | Code | Result |
| --- | --- | --- | --- |
| $\phi$ | $4\pi e/(\lvert v_z\rvert\varepsilon(k^2-\varepsilon\omega^2))$ | `4.0 * np.pi * charge / (np.abs(v[:, 2]) * eps * denom)`, `charge = sqrt(ALPHA_FS)` | matches |
| $\mathbf E_p$, $\mathbf H_p$ | $i\phi(\varepsilon\omega\mathbf v-\mathbf k)$, $i\varepsilon\phi(\mathbf k\times\mathbf v)$ | identical | matches |
| branch | $\operatorname{Im}\lambda\ge0$ | `_outgoing_root` flips to $\operatorname{Im}\ge0$; on the real axis, $\operatorname{Re}\ge0$ | matches |
| free-wave fields | $E_z=-\mathbf k_\perp\cdot\mathbf E_\perp/K_z$, $\mathbf H=\mathbf K\times\mathbf E/\omega$ | identical | matches |
| wave reference | each wave referenced at the interface it leaves | `z_ref = z_if[j-1]` (up) / `z_if[j]` (down) | matches |
| far field | $\omega\cos^2\theta\lvert\mathbf a\rvert^2/(16\pi^4)$ | `omega * n_z**2 * power / (16 pi**4)` with full-vector `power` | matches |
| closed form | Ginzburg–Frank expression above | `ginzburg_frank_interface`, principal `np.sqrt` | matches |
| $\varepsilon$ | $1-\frac{r_e\lambda^2}{\pi}\sum n(Z+f'-if'')$ | identical (`henke_dispersion` is Chantler/FFAST via xraydb) | matches |
| tolerance | $\text{rse}/3$ | `rse / 3.0` | matches |
| $b$ | $\max_E T/S$ over the band | `max` over `keep = isfinite(source) & (source > 0)` within the band | differs at $S\le0$ (see findings) |

### Numeric checks (own code, no module helpers in the reference)

I wrote my own vector multilayer solver: my own unknown layout, the particular field derived above, and a dense solve. It uses $\alpha=1/137.035999084$; the module uses `ALPHA_FS = 1/137.035999`, which accounts for the constant $6.1\times10^{-10}$ relative offset below.

- **Normal incidence, both senses**: my solver, my closed form, `ginzburg_frank_interface` and `planar_transition_radiation` agree to 7 printed digits. The cases are $\varepsilon=1.5+0.2i$, $0.99+10^{-4}i$ and $-3+i$, with $\beta=0.3$–$0.995$.
- **Six random oblique multilayers** (1–3 complex layers on a complex substrate, $E=10$–$3000$ eV, both crossing senses, arbitrary azimuth): the module agrees with my solver to $6.1\times10^{-10}$ relative in every case. That is exactly the $\alpha$ literal difference, so there is no structural difference.
- **Cherenkov-regime layer** ($\varepsilon\beta^2>1$): also agrees to $6.1\times10^{-10}$.
- **Oblique perfect conductor** ($\varepsilon=-10^9+10^9i$) against the image-charge formula: agreement $\le3\times10^{-5}$ ($O(\lvert\varepsilon\rvert^{-1/2})$). The module and my solver agree with each other to 7 digits.
- **Jackson 13.91** at $\gamma=10^3$, $\xi,\theta\sim10^{-3}$: the closed-form-to-13.91 ratio is $1-5\times10^{-6}$.
- **Hemisphere integral** of `planar_transition_radiation` for a conductor, $\beta=0.3,0.8,0.99$, both senses: matches $\frac{\alpha}{\pi E}[\frac{1+\beta^2}{2\beta}\ln\frac{1+\beta}{1-\beta}-1]$ to $\le1.4\times10^{-5}$.
- **$\varepsilon$ convention**: `medium_dielectric("silicon", E)` equals $1+\overline{\chi_0}$ (`chi_0("si", E)`) to $3\times10^{-17}$ relative at 0.5–20 keV, with $\operatorname{Im}\varepsilon>0$ and an edge jump across the Si K edge.
- **Study sampling** (Si, 30 keV, bulk, front, 150°, $E=1.5$ keV): a fine grid gives a single-crossing maximum 3.7% above the study's coarse grid. The fine grid used $9\times9$ acceptance directions, a velocity polar step of 1° to 89.6°, an azimuth step of 5°, and $0.9\beta_0$ and $\beta_0$. The maximum over directions grows monotonically with speed from $0.05\beta_0$ to $\beta_0$, which supports covering secondaries by the $\le\beta_0$ maximum.

## Findings

1. **`transition_radiation_omission` drops energies where the premise fails.** The code is `keep = np.isfinite(source) & (source > 0.0)`. An energy in the band with $S(E)\le0$ (or NaN) and $T(E)>0$ is excluded from the maximum, so `bias_bound` is finite where the true $b$ is infinite. The study has the same gap: `_band_bounds` masks `np.isfinite(ratio)` before the band maximum, even though `main` sets `ratio = inf` there. `e_star` does count those points as failing. This is not expected to move the documented decisions, because a bremsstrahlung continuum on a grid above the continuum floor is positive throughout. It is a non-conservative edge case of the stated "$\max_E$". Suggested fix (owner, not verifier): treat $T>0$ with $S\le0$ as $b=\infty$, or raise.
2. **The $cB/L$ net-line statement needs a continuum-only $S$.** With the line in $S$, only $c(1+B/L)$ follows. The physics page and the study use a continuum-only $S$, so the stated decisions hold. The helper docstring frames continuum-only as merely "conservative", not as a requirement for the $cB/L$ statement.
3. **Factor 3 and direction maximization are an expectation-level, sampled bound, not a certified supremum.** It assumes random entry/exit phase and fewer than one escaping secondary per primary, and the study's grid maximum can sit a few percent below the fine-grid maximum. The `tens-of-keV` keV decision has about $100\times$ margin ($b\le1.5\times10^{-4}$ against $1.67\times10^{-2}$), so these do not affect it. They should not be read as a rigorous upper bound for scenes near the tolerance.
4. This write-up is not yet listed in the `docs/validation/index.md` toctree (task owner).

## Verdict

The physics of the claim reproduces term by term: the field normalization, the far-field factor, signs and branch, the Ginzburg–Frank closed form in both senses, the conductor, zero-contrast and Jackson 13.91 limits, the $\varepsilon$ convention, the positivity argument and the $\text{rse}/3$ rule. There is no divergent factor, sign, exponent, unit or convention.

Verdict: `rederived`. Findings 1–3 are recorded as implementation and scope caveats that do not change any documented decision. Only a human may mark the row `signed-off`.

## Screening solver amplitude API

`planar_transition_radiation(..., return_amplitude=True)` also exposes the
outgoing Cartesian field vector already used to compute the yield. Its source
docstring defines the normalization and interface reference plane. The default
intensity-only return is preserved. The CPU tests check yield reconstruction,
transversality, broadcasting, and matched-medium/zero-thickness limits; the
hardware test compares the complex vector on NumPy and CuPy.

This API exposure does not extend the existing `rederived` claim to bunch
translation phases, cross-component phase conventions, or production CTR.
Those require separate validation before coherent summation is advertised.
