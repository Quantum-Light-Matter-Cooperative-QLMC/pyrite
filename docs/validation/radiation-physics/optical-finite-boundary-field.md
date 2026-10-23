# `optical-finite-boundary-field`

## Claim and source

- Claim: vector equivalent surface currents with the finite-distance outgoing Green function are a candidate finite-radiator TR/DR model; DR can be emitted without an electron crossing the material.
- Code: `src/pyrite/montecarlo/spectrum/optical_finite_screen.py::{circular_pec_transition_field, observe_screen_green, hemisphere_energy_per_eV}`; transport handoff `src/pyrite/montecarlo/spectrum/optical_finite_transport.py::entrance_pec_screens_from_crossings`.
- Sources: Shkvarunets and Fiorito, arXiv:0712.2848 (PRSTAB 11, 012801), Eqs. (15), (17), (19)–(25); Ginzburg–Frank backward TR from a perfect conductor as the infinite-screen limit.
- Units: SI. Two-sided temporal Fourier transform $\tilde{\mathbf E}(\omega)=\int\mathbf E(t)\,e^{+i\omega t}\,dt$ in V·s/m; $k_0=\omega/c$; output J/eV.
- Frozen revision: commit `3a98f542` (fresh-context verifier, 2026-10-23).

The derivation was written from the ledger row and the docstrings. The implementation bodies were read afterwards.

## Independent derivation

### Incident field of an oblique uniformly moving charge

A charge $q$ with velocity $\mathbf v=v\,\hat{\mathbf d}$ passes $\mathbf r_0=(x_0,y_0,0)$ at $t_0$. For a point $\mathbf u=\mathbf r-\mathbf r_0$ in the plane $z=0$, write $\mathbf p=\mathbf u-\hat{\mathbf d}(\hat{\mathbf d}\cdot\mathbf u)$, $b=|\mathbf p|$, $a=b/\gamma$, and $s=t-t_0-\hat{\mathbf d}\cdot\mathbf u/v$. The exact field is

$$
\mathbf E(t)=\frac{q\gamma\,(\mathbf p-\hat{\mathbf d}\,v s)}{4\pi\varepsilon_0\,(\gamma^2v^2s^2+b^2)^{3/2}} .
$$

With $\tau=\gamma v s/b$, $x=\omega b/(\gamma v)=\omega a/v$, and the integrals

$$
\int_{-\infty}^{\infty}\frac{\cos x\tau}{(1+\tau^2)^{3/2}}\,d\tau=2xK_1(x),
\qquad
\int_{-\infty}^{\infty}\frac{\tau\sin x\tau}{(1+\tau^2)^{3/2}}\,d\tau=2xK_0(x)
$$

(the second by parts from $\int\cos x\tau\,(1+\tau^2)^{-1/2}d\tau=2K_0(x)$), the transform is

$$
\boxed{\tilde{\mathbf E}_{\rm inc}(\mathbf u,\omega)=\frac{q\,\omega}{2\pi\varepsilon_0\gamma^2v^2}\,
e^{i\omega(t_0+\hat{\mathbf d}\cdot\mathbf u/v)}
\Bigl[\frac{\mathbf p}{a}K_1\!\Bigl(\frac{\omega a}{v}\Bigr)-i\,\hat{\mathbf d}\,K_0\!\Bigl(\frac{\omega a}{v}\Bigr)\Bigr]} .
$$

Units: $q\omega/(\varepsilon_0 v^2)$ is C·s$^{-1}$/(F·m$^{-1}$·m$^2$s$^{-2}$) $=$ V·s/m. For normal incidence ($\hat{\mathbf d}=-\hat{\mathbf z}$, $\mathbf p=\mathbf u$, $\mathbf p/a=\gamma\hat{\boldsymbol\rho}$) the tangential part is the radial

$$
\tilde E_\rho=\frac{q\,\kappa}{2\pi\varepsilon_0 v}K_1(\kappa\rho),\qquad \kappa=\frac{\omega}{\gamma v},
$$

the stated $q\kappa K_1(\kappa r)/(2\pi\varepsilon_0|v_z|)$. The longitudinal ($K_0$) term is purely normal at normal incidence and acquires a tangential projection $\hat{\mathbf d}_t K_0$ only for oblique incidence; the phase $\hat{\mathbf d}\cdot\mathbf u/v$ is the transverse incidence-time sweep. Both require the full speed $v$, not $v_z$.

As a second, independent representation (used numerically below), the source $\rho=q\delta(\mathbf r-\mathbf v t)$ has $\tilde\rho(\mathbf k,\omega)=2\pi q\,\delta(\omega-\mathbf k\cdot\mathbf v)$, and with $e^{-i\omega t}$ time dependence $\tilde{\mathbf E}=i(\omega\mathbf v/c^2-\mathbf k)\tilde\rho/[\varepsilon_0(k^2-k_0^2)]$. Integrating over $k_z$ gives the plane angular spectrum

$$
\mathbf A_{\rm inc}(\mathbf k_t)=\int\tilde{\mathbf E}_{\rm inc}\,e^{-i\mathbf k_t\cdot\boldsymbol\rho}\,d^2\rho
=\frac{i\,q\,(\omega\mathbf v/c^2-\mathbf k)}{\varepsilon_0\,|v_z|\,(k^2-k_0^2)}\Big|_{k_z=(\omega-\mathbf k_t\cdot\mathbf v_t)/v_z} .
$$

### PEC boundary and equivalent sheet

On a perfect conductor the total tangential field vanishes, so the scattered field obeys $\tilde{\mathbf E}_{s,t}=-\tilde{\mathbf E}_{{\rm inc},t}$ on the metal (Shkvarunets–Fiorito Eq. 15). The scattered field is source-free in the half-space $z>0$ containing the incoming charge and the backward radiation, so it is fixed by its tangential trace on the whole plane (Smythe/Rayleigh–Sommerfeld vector form, equivalent to a doubled magnetic sheet $2\,\tilde{\mathbf E}_s\times\hat{\mathbf z}$ radiating in free space):

$$
\tilde{\mathbf E}(\mathbf r)=2\nabla_{\mathbf r}\times\int\bigl(\hat{\mathbf z}\times\tilde{\mathbf E}_s(\mathbf r')\bigr)G\,dA'
=2\int\nabla_{\mathbf r}G(\mathbf r-\mathbf r')\times\bigl(\hat{\mathbf z}\times\tilde{\mathbf E}_s(\mathbf r')\bigr)dA',
\qquad G(R)=\frac{e^{+ik_0R}}{4\pi R},
$$

$$
\nabla_{\mathbf r}G=\hat{\mathbf R}\Bigl(ik_0-\frac1R\Bigr)G ,
$$

which keeps the exact $1/R$ and $1/R^2$ terms. With $\int E\,e^{+i\omega t}dt$, a delay $\Delta t$ multiplies by $e^{+i\omega\Delta t}$, so $e^{+ik_0R}$ is the retarded (outgoing) choice. Factor and sign check: a uniform $\tilde{\mathbf E}_s=E_0\hat{\mathbf x}$ gives $\hat{\mathbf R}\times\hat{\mathbf y}$ whose $\hat{\mathbf x}$ part integrates to $-2\pi z\int_z^\infty G'(R)\,dR=2\pi zG(z)=e^{ik_0z}/2$, so $\tilde{\mathbf E}=E_0\hat{\mathbf x}e^{ik_0z}$: factor 2 and sign are exact.

The model sets $\tilde{\mathbf E}_{s,t}=0$ off the metal. This is exact for an unbounded screen and is the physical-optics (Kirchhoff-type) approximation for a bounded disc or an aperture; the true scattered trace off the metal is not zero. It is the stated source model, not an identity.

### Energy and the angular quadrature

Let $\mathbf A(\mathbf k_t)=\int\tilde{\mathbf E}_{s,t}e^{-i\mathbf k_t\cdot\boldsymbol\rho}d^2\rho$ (V·s·m). Each plane wave has $E_z=-\mathbf k_t\cdot\mathbf A/k_z$ and Poynting flux $\propto(k_z/k_0)|\mathbf E|^2/Z_0$. Parseval for the two-sided transform, $\int f g\,dt=\pi^{-1}\int_0^\infty\operatorname{Re}\tilde f\tilde g^*d\omega$, gives

$$
\frac{dU}{d\omega}=\frac{1}{4\pi^3Z_0}\int_{k_t<k_0}d^2k\,\frac{k_z}{k_0}\Bigl(|\mathbf A|^2+\frac{|\mathbf k_t\cdot\mathbf A|^2}{k_z^2}\Bigr).
$$

With $u=\cos\theta=k_z/k_0$ and $\mathbf k_t=k_0\sin\theta\,\hat{\mathbf n}$, $d^2k=k_t\,dk_t\,d\phi=k_0^2u\,du\,d\phi$ and

$$
\frac{dU}{d\omega}=\frac{k_0^2}{4\pi^3Z_0}\int_0^1\!du\int_0^{2\pi}\!d\phi\,\bigl(u^2|\mathbf A|^2+(1-u^2)|\hat{\mathbf n}\cdot\mathbf A|^2\bigr),
\qquad \frac{dU}{dE}=\frac{1}{\hbar}\frac{dU}{d\omega}.
$$

The $1/k_z$ factor cancels, so the $u$-integrand is a polynomial in $u$ times a smooth transform. The same result follows from the far field of the Green integral, $\tilde{\mathbf E}\to(ik_0e^{ik_0r}/2\pi r)\,\hat{\mathbf r}\times(\hat{\mathbf z}\times\mathbf A)$, with $d^2U/d\omega\,d\Omega=r^2|\tilde{\mathbf E}|^2/(\pi Z_0)$. Hence the angular density is

$$
\frac{d^2U}{d\omega\,d\Omega}=\frac{k_0^2}{4\pi^3Z_0}\bigl(\cos^2\theta|\mathbf A|^2+|\hat{\mathbf r}_t\cdot\mathbf A|^2\bigr).
$$

A plane-sampled $(\pi Z_0)^{-1}\int|\tilde{\mathbf E}|^2dA$ equals this only for paraxial fields or on a far sphere; it is not the exact propagating flux on a finite plane.

### Limits

- Zero screen: $\tilde{\mathbf E}_s\equiv0$, so the field and energy vanish.
- Aperture DR without crossing (annulus $\rho_i<\rho<\rho_o$, charge through the hole): $\tilde{\mathbf E}_s\ne0$ on the metal because $K_1(\kappa\rho)$ is nonzero, so emission occurs with no crossing; for $\kappa\rho_i\gg1$ the amplitude falls like $e^{-\kappa\rho_i}$.
- Large disc, normal incidence: $\int\hat{\boldsymbol\rho}K_1(\kappa\rho)e^{-i\mathbf k_t\cdot\boldsymbol\rho}d^2\rho=-2\pi i\hat{\mathbf n}k_t/[\kappa(k_t^2+\kappa^2)]$, so $|\mathbf A|^2=q^2k_t^2/[\varepsilon_0^2v^2(k_t^2+\kappa^2)^2]$ with $\mathbf A\parallel\hat{\mathbf n}$, and $k_t^2+\kappa^2=k_0^2(1-\beta^2\cos^2\theta)/\beta^2$. Using $Z_0\varepsilon_0^2c^2=\varepsilon_0c$,

$$
\frac{d^2U}{d\omega\,d\Omega}=\frac{q^2\beta^2\sin^2\theta}{4\pi^3\varepsilon_0c\,(1-\beta^2\cos^2\theta)^2},
$$

the Ginzburg–Frank conductor result. Its hemisphere integral is

$$
\frac{dU}{d\omega}=\frac{q^2}{4\pi^2\varepsilon_0c}\Bigl[\frac{1+\beta^2}{2\beta}\ln\frac{1+\beta}{1-\beta}-1\Bigr],
$$

which tends to $q^2\beta^2/(3\pi^2\varepsilon_0c)$ as $\beta\to0$.

## Cheap filters

| Check | Result |
|---|---|
| Units | $q\omega/(\varepsilon_0v^2)$ is V·s/m; $\mathbf A$ is V·s·m; $k_0^2\lvert\mathbf A\rvert^2/Z_0$ is J·s $=dU/d\omega$; dividing by $\hbar$ gives J/eV. Pass. |
| Limits | Zero screen gives zero; annulus gives nonzero DR without a crossing, decaying as $e^{-\kappa\rho_i}$; large disc gives Ginzburg–Frank, angularly and hemisphere-integrated. Pass. |
| Conventions | $\int E\,e^{+i\omega t}dt$ gives retarded $e^{+ik_0R}$ and arrival phase $e^{+i\omega t_0}$; the Smythe form has factor 2 and $+$ sign (uniform plane-wave test); the tangential $K_0$ term and the phase $\hat{\mathbf d}\cdot\mathbf u/v$ use the full speed $v$. Pass. |

## Comparison with the implementation

- `circular_pec_transition_field`: `perp = u - unit*(unit·u)`, with `perp_z = -unit[2]*projection` because $u_z=0$. `a = |perp|/gamma`, argument $\omega a/v$, phase `exp(1j*omega*(arrival_time_s + projection/speed))`, prefactor `-charge_c*omega/(2*pi*eps0*gamma**2*speed**2)` and bracket `perp/a*K1 - 1j*unit*K0` (x,y only), all on the metal mask $\rho_i\le\rho\le\rho_o$. This is exactly $-\tilde{\mathbf E}_{{\rm inc},t}$ of the boxed formula, with $\omega=E/\hbar$.
- `observe_screen_green`: `kernel = exp(1j*k0*rr)*(1j*k0*rr - 1)/(4*pi*rr**3)` is $G'(R)/R$, so $\nabla G=\mathbf R\,G'/R$. The code's components $(-2R_zE_x,\,-2R_zE_y,\,2(R_xE_x+R_yE_y))\cdot$`kernel` are $2\,\mathbf R\times(\hat{\mathbf z}\times\tilde{\mathbf E}_s)\,G'/R=2[\hat{\mathbf z}(\mathbf R\cdot\tilde{\mathbf E}_s)-\tilde{\mathbf E}_sR_z]G'/R$. Pixel-centre quadrature uses $dA=$`dx_m*dy_m`, with $R_z$ measured from `field.z_m`. Matches, including the $1/R^2$ term.
- `hemisphere_energy_per_eV`: Gauss–Legendre nodes $u=(t+1)/2$, weights $w/2$, uniform $\phi$ with $2\pi/n_\phi$. The direct transform is $e^{-i\mathbf k_t\cdot\boldsymbol\rho}$ times `dx*dy`, $F_z=-(\mathbf k_t\cdot\mathbf F)/k_z$, and the sum $\sum w\,k_zk_0u\,|\mathbf F|^2/(4\pi^3Z_0\hbar)$. This is $\frac{1}{4\pi^3Z_0\hbar}\int k_0^2u\,du\,d\phi\,(k_z/k_0)|\mathbf F|^2$, the derived expression with the stated Jacobian. Matches.
- `spectral_energy_per_eV` (same normalization, not an anchor of this claim): $\sum(k_z/k_0)|{\rm FFT}|^2\,dx\,dy/(n_xn_y\pi Z_0\hbar)$ is the discrete Parseval form of the same flux. `far_field_photons_per_eV_sr` uses $k_0^2\cos^2\theta(|\mathbf A|^2+|\mathbf k_t\cdot\mathbf A|^2/k_z^2)$, which equals $k_0^2(\cos^2\theta|\mathbf A|^2+|\hat{\mathbf r}_t\cdot\mathbf A|^2)$. Both match.
- `entrance_pec_screens_from_crossings`: $(x,y,z)\to(x,-y,-z)$ is a proper rotation (det $+1$) applied to the offset and to $\boldsymbol\beta$, giving $\beta_z<0$ for an entering $+z$ electron. It uses $\gamma=1+T/m_ec^2$ with $m_ec^2=510.99895$ keV, charge $-e$, and $t_0=(t-t_{\rm origin})[\text{Å}]\cdot10^{-10}/c$. The entry row is vacuum $\to$ layer 0 at interface 0, one per history $0..M-1$, at $z=0$. Matches the stated handoff.

### Independent numerical checks

Scripts used only `scipy.constants`, scipy special functions and quadrature, and the public entry points under test. No implementation helper served as a reference.

1. Oblique incident field: at $E=1.5$ eV, $\boldsymbol\beta=(0.25,-0.15,-0.8)$, an off-node intercept and $t_0=3$ fs, the direct time integral of the exact moving-charge field ($\tau=\sinh x$ substitution, dense trapezoid) agrees with `circular_pec_transition_field` $=-\tilde{\mathbf E}_{{\rm inc},t}$ to $\le3\times10^{-8}$ at 4 points. The boxed closed form agrees with the code to $8\times10^{-10}$, a CODATA constant difference.
2. Normal-incidence large disc, 30 keV ($\beta=0.3284$), 1 eV, $R=1.5$ µm ($\kappa R\approx22$): `hemisphere_energy_per_eV` divided by the closed-form Ginzburg–Frank hemisphere gives $0.9999947$ ($n=256$) and $0.9999997$ ($n=512$) with the charge centred. With the charge offset by a fraction of a pixel, the ratio is $1.0162/1.0041/1.0018$ at pitch $12.9/6.4/4.3$ nm, converging as pitch$^2$. This is sampling of the integrable $1/\rho$ source near the axis, not a normalization error.
3. Oblique 30° incidence, same $\beta$ and energy: the independent reference is the infinite-plane angular spectrum $\mathbf A_{\rm inc}(\mathbf k_t)$ from the $k$-space charge field, integrated with $d^2U/d\omega\,d\Omega$. It equals $0.7625$ of the normal GF hemisphere; the same integrator reproduces the normal GF closed form to $10^{-12}$. The code's 1.5 µm oblique disc gives $1.0053$ ($n=512$) and $1.0023$ ($n=768$) of this reference, converging as pitch$^2$.
4. Aperture DR (annulus $0.1$–$1.5$ µm, charge through the hole, no crossing): a 1-D Hankel reference $\int_{\rho_i}^{\rho_o}\rho K_1(\kappa\rho)J_1(k_t\rho)\,d\rho$ gives $0.4126$ of GF. The code gives $1.0017$ ($n=512$) and $1.0031$ ($n=768$) of it. The non-monotone 0.2–0.3% residual comes from the staircase hole edge at $\rho_i=1.5\kappa^{-1}$.
5. Green far field: for the 1.5 µm disc observed at $r=0.5$ mm, $r^2|\tilde{\mathbf E}|^2/(\pi Z_0)$ from `observe_screen_green` equals the Ginzburg–Frank angular density to 5 digits at $\theta=20^\circ,45^\circ,70^\circ$.
6. Green near field (1/$R^2$ terms): for a 0.4 µm disc, the reference is an independent Hankel angular-spectrum integral that includes evanescent waves, $E_\rho=(2\pi)^{-1}\int k\,a(k)\,iJ_1(k\rho)e^{ik_zz}dk$ and $E_z=-(2\pi)^{-1}\int k^2a(k)k_z^{-1}J_0(k\rho)e^{ik_zz}dk$. At $(\rho,z)=(0.2\,\mu{\rm m},\lambda/4)$ and $(0.6\,\mu{\rm m},\lambda/2)$, `observe_screen_green` agrees to $1$–$2\times10^{-5}$ in both $E_x$ and $E_z$, with zero phase difference.
7. A zero-radius screen gives an identically zero field and zero hemisphere energy.

## Adjudication

No divergent factor, sign, exponent, unit or convention. The oblique incident transform, the PEC sign, the factor-2 Smythe/magnetic-sheet Green curl with its exact $1/R$ and $1/R^2$ terms, the two-sided-transform energy normalization with $1/\hbar$, the $d^2k=k_0^2u\,du\,d\phi$ Jacobian, and the transport handoff all match the independent derivation. Verdict: **rederived**.

What this does and does not establish:

- It establishes the stated *model*: tangential source $-\tilde{\mathbf E}_{{\rm inc},t}$ on the metal and zero elsewhere. That source is exact for an unbounded PEC and is a physical-optics (Kirchhoff-type) approximation for a bounded disc or aperture, where the true scattered trace off the metal does not vanish. The ledger's "candidate" wording is appropriate; no published finite-disc or DR benchmark was reproduced. Schmidt et al. Fig. 7 remains the proposed anchor.
- Shkvarunets–Fiorito equation numbers were not checked against the paper text in this context. The physics was rederived from Maxwell's equations and the vector Smythe identity.

### Findings

1. Low, wording: the physics page and the `spectral_energy_per_eV` docstring state $dU/d\omega=(\pi Z_0)^{-1}\int|\tilde{\mathbf E}|^2dA$. The code (correctly) uses the propagating normal flux $(k_z/k_0)|\tilde{\mathbf E}|^2$ in $k$-space, including $E_z$. The plane form holds only paraxially or on a far sphere. Suggest "the normal Poynting flux, $(\pi Z_0)^{-1}\int|\tilde{\mathbf E}|^2dA$ in the paraxial limit".
2. Low, consistency: the ledger quotes hemisphere-vs-GF agreement to $10^{-4}$, the physics page to $10^{-5}$. Check 2 gives $3\times10^{-7}$ for a centred charge and up to $4\times10^{-3}$ for a sub-pixel-offset charge at 6 nm pitch. A future anchor should state the intercept offset and pitch.
3. Low, scope: state the physical-optics validity envelope explicitly (screen and hole dimensions $\gg\lambda$, edge distances $\gg\lambda$) beside the existing "induced-current approximation" phrase.

### Caveats

- The incident field is that of an infinite straight trajectory, the standard PEC-image construction; no finite-track, deflection or thickness effects.
- All checks are synthetic and numerical; no measured data.
