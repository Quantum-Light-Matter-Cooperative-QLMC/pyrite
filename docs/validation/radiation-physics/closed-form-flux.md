# Validation: `closed-form-flux`

## Claim and source

- Claim: Feranchuk–Spence (2000) Eq. (12) closed-form line flux, used as the single-segment analytic reference against which the Monte Carlo pipeline is anchored.
- Ledgered code: `src/pyrite/validation/anchor_figures.py::feranchuk_line_flux`.
- Source: Feranchuk, Ulyanenkov, Harada, Spence, *Phys. Rev. E* **62**, 4225 (2000), Eq. (12), with the Eq. (9) absorption-limited length and the Eq. (13)/(14) amplitudes.
- Signature: `feranchuk_line_flux(anchor: ZhaiAnchor, E0_keV: float, thickness_ang: float) -> float`.
- Docstring statement of the intended quantity: *"Feranchuk-Spence Eq. (12) closed-form line flux [photons / electron into dOmega], absorption-limited, at the dispersion-relation line energy."*

**Units of the returned quantity (confirmed).** The returned number is **photons per incident electron integrated over the line and into the fixed solid angle** $\Delta\Omega$ carried by `anchor.domega_sr` — a pure dimensionless count. It is **not** per eV and **not** per steradian: the energy integral over the line has already been done analytically (that is what makes it "closed form"), and `dOmega_sr` multiplies rather than divides. This differs from the Monte Carlo spectrum `mc_spectrum`, which returns $d^2N/(dE\,d\Omega)$ in photons / (eV sr electron); the anchor reconciles the two by trapezoid-integrating the MC spectrum over $E$ and evaluating the closed form at $\Delta\Omega = 1\ \mathrm{sr}$.

The amplitudes $A_{\rm PXR}$ and $A_{\rm CBS}$ are **out of scope** here — they are separately ledgered as `pxr-amplitude` and `cbs-amplitude`. This write-up verifies the Eq. (12) *wrapper*: the prefactor, the $\alpha$ and $\hbar c$ bookkeeping, the absorption/escape-length treatment, the solid-angle and energy-bin conventions, and the units.

## Independent derivation

*(Recorded before inspecting the body of `photons_per_electron`; the locked statement is reproduced verbatim in the comparison section below.)*

### Radiated photon number from a classical source current

Work in Gaussian units with $\hbar = c = 1$, so that $e^2 = \alpha$ and a wavenumber and an energy are the same quantity. For a localised current $\mathbf J(\mathbf r, t)$ the far-field spectral–angular energy is

$$
\frac{d^2W}{d\omega\,d\Omega}
=\frac{\omega^2}{4\pi^2}
\sum_{p}\left\lvert
\hat{\mathbf e}_p^{\,*}\cdot\widetilde{\mathbf J}(\mathbf k,\omega)
\right\rvert^2,
\qquad
\widetilde{\mathbf J}
=\int dt\!\int d^3r\;\mathbf J(\mathbf r,t)\,
e^{i(\omega t-\mathbf k\cdot\mathbf r)} .
$$

Dividing by $\omega$ per photon,

$$
\frac{d^2N}{d\omega\,d\Omega}
=\frac{\omega}{4\pi^2}
\sum_{p}\left\lvert
\hat{\mathbf e}_p^{\,*}\cdot\widetilde{\mathbf J}
\right\rvert^2 .
$$

### Straight-track amplitude and the resonance detuning

The PXR/CBS source is the electron's own field scattered by the periodic susceptibility harmonic $\mathbf g$. For one reflection and one polarization the paper factors the microscopic response into a dimensionless amplitude $A_p \equiv A_{{\rm PXR},p}+A_{{\rm CBS},p}$ (Eqs. (13)–(14)), leaving one explicit factor of the electron charge, i.e. one factor of $\alpha$ in the intensity. Along a straight trajectory $\mathbf r(t)=\mathbf v t$ inside the crystal the emission phase advances (see `coherent-emission`) as

$$
\Delta \;=\; \omega\bigl(1-\beta\,\hat{\mathbf n}\cdot\hat{\mathbf v}\bigr)
-\mathbf g\cdot\mathbf v
\;=\;\omega\bigl(1-\beta\cos\theta_{\rm obs}\bigr)-\mathbf g\cdot\mathbf v ,
$$

which vanishes at the Eq. (10) line energy $\omega_{\rm res}=\mathbf g\cdot\mathbf v/(1-\beta\cos\theta_{\rm obs})$. Here $\theta_{\rm obs}$ is the angle between the observation direction $\hat{\mathbf n}$ and the electron velocity.

Let the photon generated at track parameter $t$ suffer amplitude attenuation $e^{-\mu s(t)/2}$ on its way out, where $\mu = 1/L_{\rm abs}$ is the **intensity** attenuation coefficient and $s(t)$ the escape path. Then

$$
\hat{\mathbf e}_p^{\,*}\cdot\widetilde{\mathbf J}
=\sqrt{\alpha}\;A_p\int_0^{T}\! e^{-\mu s(t)/2}\,e^{i\Delta t}\,dt ,
\qquad T=\frac{L}{\beta},
$$

with $T$ the transit time of a track of geometric length $L$.

### Integrating across the line

The observable is the flux *in the line*, i.e. the integral over $\omega$ across the narrow resonance. Because $\Delta$ depends on $\omega$ only through $\partial\Delta/\partial\omega = 1-\beta\cos\theta_{\rm obs}$,

$$
d\omega=\frac{d\Delta}{1-\beta\cos\theta_{\rm obs}} ,
$$

and Parseval's theorem applied to the finite, attenuated track window gives

$$
\int_{-\infty}^{\infty}\!d\Delta
\left\lvert\int_0^{T}\! e^{-\mu s(t)/2}e^{i\Delta t}dt\right\rvert^2
=2\pi\int_0^{T}\! e^{-\mu s(t)}\,dt
=\frac{2\pi}{\beta}\int_0^{L}\! e^{-\mu s(\ell)}\,d\ell
\equiv\frac{2\pi}{\beta}\,L_{\rm eff}.
$$

Two things are worth stressing. First, the frequency-integrated line yield is **linear** in the interaction length, not quadratic: the $T^2$ of the peak height is traded against the $1/T$ line width. Second, the amplitude-level attenuation $e^{-\mu s/2}$ squares to the intensity attenuation $e^{-\mu s}$, so the length that appears is the ordinary Beer–Lambert intensity-weighted path integral. Both facts are what make the closed form a legitimate reference for an incoherent, intensity-additive Monte Carlo sum.

Collecting,

$$
\boxed{\;
\frac{dN}{d\Omega}\bigg|_{\rm line}
=\frac{\alpha}{2\pi}\;\omega_{\rm res}\;\frac{L_{\rm eff}}{\beta}\;
\frac{\lvert A_{\rm PXR}+A_{\rm CBS}\rvert^2}{1-\beta\cos\theta_{\rm obs}}\;}
$$

and the photons per electron into a finite acceptance $\Delta\Omega$ follow by multiplying by $\Delta\Omega$ (valid while the amplitude and $\theta_{\rm obs}$ are treated as constant across the acceptance).

The $1/(1-\beta\cos\theta_{\rm obs})$ factor is exactly the Jacobian of the $\delta$-function that Eq. (9)/(10) imposes; it is a genuine part of the frequency integral and cannot be dropped without an explicit $\beta\ll 1$ argument.

### The escape length $L_{\rm eff}$

$L_{\rm eff}=\int_0^L e^{-\mu s(\ell)}d\ell$ needs the geometric relation between the electron's path element $d\ell$ and the photon's escape path $s$. For the anchor geometry — beam along $+\hat{\mathbf z}$, normal incidence on a slab of thickness $t$ with the reflecting planes perpendicular to the beam, detector at $\theta_{\rm obs}$ — the electron path element is $d\ell = dz$ and the photon leaves through the entrance face along $\hat{\mathbf n}$ with $n_z=\cos\theta_{\rm obs}<0$, so

$$
s(z)=\frac{z}{\lvert\cos\theta_{\rm obs}\rvert},
\qquad
L_{\rm eff}
=\int_0^{t} e^{-\mu z/\lvert\cos\theta_{\rm obs}\rvert}\,dz
=\lvert\cos\theta_{\rm obs}\rvert\,L_{\rm abs}
\left[1-\exp\!\left(-\frac{t}{\lvert\cos\theta_{\rm obs}\rvert L_{\rm abs}}\right)\right].
$$

Only in the degenerate case $\lvert\cos\theta_{\rm obs}\rvert=1$ (photon escaping along the beam axis) does this reduce to the bare

$$
L_{\rm eff}^{(\lvert\cos\theta_{\rm obs}\rvert=1)}
=L_{\rm abs}\left[1-\exp\!\left(-\frac{t}{L_{\rm abs}}\right)\right].
$$

### $\hbar c$ bookkeeping

Restoring units, $\omega$ appears only in the dimensionless product $\omega L_{\rm eff}$, so the natural repository spelling is a photon wavenumber in $\mathrm{\AA}^{-1}$,

$$
\omega\;[\mathrm{\AA}^{-1}]=\frac{E\;[\mathrm{eV}]}{\hbar c},
\qquad \hbar c = 1973.269804\ \mathrm{eV\,\AA},
$$

paired with $L_{\rm eff}$ in $\mathrm{\AA}$. With $\alpha/2\pi$, $|A|^2$, $\beta$ and $1-\beta\cos\theta_{\rm obs}$ all dimensionless, the result is a dimensionless photon count per electron per steradian, and multiplying by $\Delta\Omega$ in sr gives photons per electron. There is no leftover $\hbar$, $c$, or $e$.

## Cheap filters

| Filter | Independent expectation | Result |
| --- | --- | --- |
| Dimensions | $\alpha$, $\lvert A\rvert^2$, $\beta$, $1-\beta\cos\theta_{\rm obs}$ dimensionless; $\omega L_{\rm eff}$ dimensionless with $\omega$ in $\mathrm{\AA}^{-1}$, $L_{\rm eff}$ in $\mathrm{\AA}$; $\times\Delta\Omega\,[\mathrm{sr}]$ gives photons/electron | pass |
| Thickness scaling, $t\ll L_{\rm abs}$ | $L_{\rm eff}\to t$, flux linear in thickness | pass |
| Thick target, $t\to\infty$ | $L_{\rm eff}$ saturates, flux independent of $t$ | pass in form, **wrong saturation value** — see below |
| Transparent limit, $\mu\to0$ | $L_{\rm eff}\to t$ for any $t$ | pass |
| Absorption-length scaling | flux $\propto L_{\rm abs}$ once $t\gg L_{\rm abs}$ | pass in form, off by $\lvert\cos\theta_{\rm obs}\rvert$ |
| Non-relativistic limit $\beta\to0$ | $\omega_{\rm res}/\beta \to \lvert\mathbf g\rvert$ finite, so the explicit $1/\beta$ does **not** diverge | pass |
| Forward/backward asymmetry | $1/(1-\beta\cos\theta_{\rm obs})$ enhances forward, suppresses backward emission | pass |
| Beam-energy behaviour | thin-film flux falls with $E_0$ (falling $\lvert A\rvert^2$, rising $E_{\rm line}$); bulk flux rises slowly as $L_{\rm abs}(E_{\rm line})$ grows | pass, monotone and smooth over 5–100 keV |
| Solid-angle convention | `dOmega_sr` multiplies (finite acceptance), not divides | pass |
| Energy-bin convention | no $dE$ anywhere; line already integrated | pass |

The escape-length row is the failure. Everything else survives.

## Comparison with the implementation

### What the code actually evaluates

The ledgered symbol is a thin wrapper. `anchor_figures.py::feranchuk_line_flux` resolves $\beta$, the Eq. (10) line energy, and an absorption length, then delegates the whole equation to `validation/feranchuk_spence.py::photons_per_electron`. The equation under test is therefore the last line of that function:

```text
L_eff = L_abs_ang * (1.0 - np.exp(-L_z_ang / L_abs_ang))

return ALPHA_FS / (2.0 * np.pi) * omega * (L_eff / beta) * A2 * dOmega_sr / (1.0 - beta * n_z)
```

with `omega = photon_E_eV / HBARC_EV_ANG`, `HBARC_EV_ANG = 1973.269804`, `ALPHA_FS = 1.0 / 137.035999`, `A2` the $\sigma+\pi$ sum of $\lvert A_{\rm PXR}+A_{\rm CBS}\rvert^2$, and, for `geometry="lif"`, `n_z = np.cos(theta_B_normal)` where that slot carries $\theta_{\rm obs}$.

The wrapper supplies

```text
L_abs = absorption_length_ang("C", E_line, anchor.n_atoms_per_ang3)
```

### Prefactor — matches exactly

Term by term against the boxed independent result:

| Independent term | Code | Verdict |
| --- | --- | --- |
| $\alpha/2\pi$ | `ALPHA_FS / (2.0 * np.pi)` | match |
| $\omega_{\rm res}$ in $\mathrm{\AA}^{-1}$ | `photon_E_eV / HBARC_EV_ANG` | match |
| $L_{\rm eff}/\beta$ (transit time) | `L_eff / beta` | match |
| $\lvert A_{\rm PXR}+A_{\rm CBS}\rvert^2$, $\sigma+\pi$ | `A2` | match (amplitudes out of scope) |
| $\Delta\Omega$ multiplying | `* dOmega_sr` | match |
| $1/(1-\beta\cos\theta_{\rm obs})$ | `/ (1.0 - beta * n_z)` | match |
| $\hbar c$, $\alpha$, $e$ bookkeeping | one $\alpha$, one $\hbar c$, no stray $e$ | match |

The exponent of $\alpha$ (one power, first-order single-photon emission), the factor $2\pi$ (from $4\pi^2$ in the source formula against $2\pi$ from Parseval), and the *linear* dependence on interaction length all agree.

The `n_z` branch table is also correct: `symmetric` uses $n_z=-\cos 2\theta_{B,\rm normal}$, which is $\cos(\pi-2\theta_{B,\rm normal})$ for a detector at $2\theta_{\rm Bragg}$ from the beam with $\theta_{\rm Bragg}=\pi/2-\theta_{B,\rm normal}$; `fixed` and `lif` both reduce to $\cos\theta_{\rm obs}$.

**Provenance caveat, not an error.** The `photons_per_electron` docstring states that the $1/(1-\beta\cos\theta_{\rm obs})$ Jacobian is *added* to the paper's Eq. (12), which the repository reads as omitting it (the sibling docstring in `feranchuk_spence.py` writes Eq. (12) as $\alpha/(2\pi)\,\omega\,(L_{\rm eff}/\beta)\,\lvert A\rvert^2$ with no Jacobian). The independent derivation above says the Jacobian **belongs there**, so the code is more correct than the cited equation. The ledger row's `Source: Feranchuk 2000 Eq.(12)` is therefore incomplete provenance: the implemented expression is Eq. (12) *plus* the exact $\delta$-function Jacobian.

### Numerical agreement, prefactor only

Independent reassembly of the boxed expression, with $\alpha$, $\hbar c$, $\beta(E_0)$, $\mathbf g$ and $L_{\rm eff}$ recomputed outside the function and only `A2` taken from the (separately ledgered) amplitude routine, at HOPG (002), $E_0 = 25\ \mathrm{keV}$, $\theta_{\rm obs}=119^\circ$, $\Delta\Omega=0.066\ \mathrm{sr}$:

| $t$ [$\mathrm{\AA}$] | independent | `photons_per_electron` | ratio |
| --- | --- | --- | --- |
| $290$ | $1.970222\times10^{-9}$ | $1.970222\times10^{-9}$ | $1.000000000000$ |
| $10^{7}$ | $1.350235\times10^{-7}$ | $1.350235\times10^{-7}$ | $1.000000000000$ |
| $10^{12}$ | $1.350235\times10^{-7}$ | $1.350235\times10^{-7}$ | $1.000000000000$ |

Supporting independent numbers, none of them taken from the implementation: $\beta(25\ \mathrm{keV}) = 0.301842$ from $\gamma = 1+E_0/m_ec^2$ with $m_ec^2 = 510998.95\ \mathrm{eV}$; $g_{002}=2\cdot 2\pi/c = 1.872503\ \mathrm{\AA}^{-1}$ for graphite $c=6.711\ \mathrm{\AA}$; $E_{\rm line}=972.918\ \mathrm{eV}$; graphite number density $0.113637\ \mathrm{\AA}^{-3}$, i.e. $2.266\ \mathrm{g\,cm^{-3}}$; $L_{\rm abs}=1.973\ \mathrm{\mu m}$ at $973\ \mathrm{eV}$, which reproduces the CXRO/Henke carbon attenuation length ($\mu/\rho \approx 2.2\times10^{3}\ \mathrm{cm^2\,g^{-1}}$ giving $1/\mu\approx 2.0\ \mathrm{\mu m}$) to a few percent.

The Parseval normalisation that fixes the $\alpha/2\pi$ and the linear-in-$L$ behaviour was also checked by direct quadrature, independent of any repository code: for $T=3.7$, $\mu\beta=0.31$,

$$
\int d\Delta\left\lvert\int_0^{T}e^{-\mu\beta t/2}e^{i\Delta t}dt\right\rvert^2
=13.8307
\quad\text{vs.}\quad
\frac{2\pi\left(1-e^{-\mu\beta T}\right)}{\mu\beta}=13.8314,
$$

a ratio of $0.99995$, limited by the quadrature grid.

### Escape length — discrepancy

The code computes

$$
L_{\rm eff}^{\rm code}
=L_{\rm abs}\left[1-\exp\!\left(-\frac{t}{L_{\rm abs}}\right)\right],
$$

i.e. the independent result with $\lvert\cos\theta_{\rm obs}\rvert$ set to $1$. `absorption_length_ang` is confirmed to return the **intensity** attenuation length $1/\mu$ (ledger `absorption-length`, status `anchored`), so the $\mu$ convention is right; the missing factor is purely geometric.

For the anchor geometry $\theta_{\rm obs}=119^\circ$, $\lvert\cos\theta_{\rm obs}\rvert = 0.4848$, so in the thick-target limit

$$
\frac{L_{\rm eff}^{\rm code}}{L_{\rm eff}^{\rm geom}}
\longrightarrow\frac{1}{\lvert\cos\theta_{\rm obs}\rvert}=2.063 .
$$

At the 1 mm bulk thickness the code returns $L_{\rm eff}=1.9729\times10^{4}\ \mathrm{\AA}$ where the escape-weighted value is $9.5648\times10^{3}\ \mathrm{\AA}$ — a factor $2.063$. At the 29 nm film thickness the two agree to $0.8\%$, because $t\ll L_{\rm abs}$ makes the geometry factor cancel.

The production Monte Carlo does **not** share this error. `montecarlo/spectrum/lines.py` computes the escape path as

```text
def _escape_length(z_mid, thickness, n_z):
    return z_mid / -n_z if n_z < 0 else (thickness - z_mid) / n_z
```

and folds `T_abs = exp(-(L_esc * mu))` into the per-segment weight, which integrates to precisely the $\lvert\cos\theta_{\rm obs}\rvert L_{\rm abs} [1-\exp(-t/(\lvert\cos\theta_{\rm obs}\rvert L_{\rm abs}))]$ of the independent derivation.

This was confirmed directly. Feeding `mc_spectrum` a contiguous stack of $290\ \mathrm{\AA}$ straight segments spanning a slab of thickness $t$ (HOPG (002), $25\ \mathrm{keV}$, $\theta_{\rm obs}=119^\circ$), integrating the MC spectrum over energy, dividing out the closed form evaluated with absorption disabled, and solving $x\,[1-\exp(-t/x)] = L_{\rm eff}^{\rm MC}$ for $x=\lvert n_z\rvert L_{\rm abs}$:

| $t/L_{\rm abs}$ | MC / closed form | implied $\lvert n_z\rvert$ |
| --- | --- | --- |
| $0.02$ | $0.9870$ | $0.5043$ |
| $0.25$ | $0.8801$ | $0.4847$ |
| $1.00$ | $0.6673$ | $0.4848$ |
| $3.00$ | $0.5075$ | $0.4848$ |
| $8.00$ | $0.4834$ | $0.4848$ |

The recovered $\lvert n_z\rvert = 0.4848$ is $\lvert\cos 119^\circ\rvert$ to four digits at every thickness where the fit is well conditioned. The closed form overstates the absorbing-regime flux by up to $2.06\times$, and the disagreement grows monotonically with $t/L_{\rm abs}$ — exactly the "narrow-regime agreement" failure mode the methodology warns about, since the one regime where the two agree ($t\ll L_{\rm abs}$) is the only regime the existing anchor exercises.

### Secondary findings

1. **Hardcoded absorber element.** `feranchuk_line_flux` calls `absorption_length_ang("C", ...)` unconditionally while taking the number density from `anchor.n_atoms_per_ang3`, which follows `anchor.crystal`. `ZhaiAnchor` is documented as overridable to "explore other crystals", so any non-carbon crystal silently gets carbon $f_2$ combined with that crystal's number density. Correct only for the HOPG default.
2. **Missing `Validation:` markers.** Neither `feranchuk_line_flux` nor `photons_per_electron` carries a `Validation: closed-form-flux` back reference, so the code↔ledger tie is one-directional.
3. **Ledger anchors the wrapper, not the equation.** The equation lives in `validation/feranchuk_spence.py::photons_per_electron`; `anchor_figures.py::feranchuk_line_flux` only chooses arguments.
4. **Downstream reach of the escape-length error.** `anchor_figures.py::figure_enhancement` recomputes the same bare $L_{\rm abs}[1-\exp(-t/L_{\rm abs})]$ for its bulk/film ceiling, giving $68.5$ where the escape-weighted ceiling is $33.5$. That is the separately ledgered `enhancement-bulk-film` claim, not this one, but the defect is shared.

## Is `single_segment_anchor` an independent check?

**Partly, and not for the part that fails.**

- **It does not exercise the ledgered symbol.** `single_segment_anchor` calls `photons_per_electron` directly with its own argument list; it never calls `feranchuk_line_flux`. The wrapper's line-energy choice, its hardcoded `"C"`, and its absorption length are all outside the anchor.
- **It disables the failing term by construction.** The anchor passes `L_abs_ang=1e12` with `L_seg_ang=290.0`, so $L_{\rm eff}\to L_{\rm seg}$ to ten significant figures and the escape-length geometry factor is multiplicatively $1$. The anchor is structurally incapable of detecting the $\lvert\cos\theta_{\rm obs}\rvert$ omission.
- **For the prefactor, the two legs are genuinely independent code.** `montecarlo/spectrum/lines.py` does not import from `validation/feranchuk_spence.py`; the finite-segment lineshape, its normalisation, and the $\alpha$/$\hbar c$ bookkeeping are written twice. What the two legs *do* share is the crystallographic layer — `materials/crystal.py` (`chi_g`, `U_g`, form factors, Debye–Waller) and the same attenuation tabulation — so the anchor cannot catch an error in $\lvert A\rvert^2$ or in $\mu$, only in the radiation kinematics wrapped around them. Those inputs are covered by the separate `pxr-amplitude`, `cbs-amplitude`, and `absorption-length` rows.
- **Reproduced anchor values** (`ZhaiAnchor` defaults, `L_seg = 290 Å`): ratio $0.99735$ at $17.5\ \mathrm{keV}$ and $0.99673$ at $25\ \mathrm{keV}$. Both are $\approx 1$, so the prefactor is cross-validated at the sub-percent level by a genuinely separate implementation.

Net: the anchor is a real, independent confirmation of the $\alpha/2\pi$, $\omega L/\beta$, $\Delta\Omega$ and Jacobian factors, and no confirmation at all of the absorption treatment that `feranchuk_line_flux` exists to add.

## Adjudication

The prefactor, the $\alpha$ and $\hbar c$ bookkeeping, the solid-angle and energy-bin conventions, the units of the return value, and the $1/(1-\beta\cos\theta_{\rm obs})$ Jacobian all reproduce the independent derivation exactly, symbolically and to twelve digits numerically, and are independently corroborated by the Monte Carlo at the sub-percent level.

The absorption/escape-length treatment does not. The first divergent factor is

$$
L_{\rm eff}^{\rm code}=L_{\rm abs}\left[1-e^{-t/L_{\rm abs}}\right]
\qquad\text{vs.}\qquad
L_{\rm eff}=\lvert\cos\theta_{\rm obs}\rvert L_{\rm abs}
\left[1-e^{-t/(\lvert\cos\theta_{\rm obs}\rvert L_{\rm abs})}\right],
$$

a missing $\lvert\cos\theta_{\rm obs}\rvert$ that reaches $2.063\times$ in the thick-target limit of the anchor geometry and is invisible to the existing anchor. Because this expression is the yardstick the Monte Carlo bulk results are compared against, the defect is load-bearing exactly as the ledger note about "a wrong prefactor here would silently certify a wrong pipeline" anticipates — with the twist that the prefactor is right and the escape length is not.

**Verdict: `discrepancy`.**

Not signed off. Human adjudication required; only a human may move this row past `rederived`.
