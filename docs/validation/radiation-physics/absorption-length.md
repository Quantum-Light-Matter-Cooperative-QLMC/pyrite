# Validation: absorption-length

## Independent derivation

Let $k_0=2\pi/\lambda$, and for the $\exp(-i\omega t)$ phasor convention write the forward wave in a passive medium as

$$
E(z,t)=E_0\exp[i k_0(n_r+i\beta)z-i\omega t].
$$

The field magnitude decays as $\exp(-k_0\beta z)$, so the intensity obeys

$$
\frac{I(z)}{I(0)}=\exp(-2k_0\beta z)
=\exp\!\left(-\frac{4\pi\beta}{\lambda}z\right).
$$

Comparing with $I(z)=I(0)\exp(-\mu z)$ gives

$$
\mu=2k_0\beta=\frac{4\pi\beta}{\lambda},
\qquad
L_{\rm abs}=\frac{1}{\mu}=\frac{\lambda}{4\pi\beta}.
$$

Using the supplied microscopic relation

$$
\beta=\frac{r_e\lambda^2 n_{\rm atom}f_2}{2\pi}
$$

gives, verbatim from the independent result,

$$
\boxed{\mu=\frac{4\pi\beta}{\lambda}
=2r_e\lambda n_{\rm atom}f_2},
\qquad
\boxed{L_{\rm abs}=\frac{\lambda}{4\pi\beta}
=\frac{1}{2r_e\lambda n_{\rm atom}f_2}}.
$$

For an independent additive mixture, replace $n_{\rm atom}f_2$ by $\sum_jn_jf_{2,j}$.

## Units, conventions, and limiting cases

With $r_e$ and $\lambda$ in Angstrom, number density in $\mathring{\mathrm A}^{-3}$, and dimensionless $f_2$, $\beta$ is dimensionless, $\mu$ is in $\mathring{\mathrm A}^{-1}$, and $L_{\rm abs}$ is in Angstrom. The factor of two between field-amplitude and intensity attenuation is explicit.

For a passive medium, $f_2\geq0$, so $\mu\geq0$. At zero distance the transmission is one. As either density or $f_2$ tends to zero, $\mu\to0$ and $L_{\rm abs}\to\infty$. At fixed $f_2$ and density, $\lambda\to0$ also gives $L_{\rm abs}\to\infty$; with tabulated energy-dependent $f_2$, that high-energy limit additionally requires $\lambda f_2(\lambda)\to0$.

"Attenuation length" here is the Beer--Lambert **intensity** e-folding length; the field-amplitude e-folding length would be twice as large.

## Implementation comparison

`src/pyrite/materials/crystal.py::absorption_length_ang` evaluates

```python
lam = HC_EV_ANG / photon_E_eV
beta_idx = R_E_ANG * lam**2 / (2.0 * np.pi) * number_density_per_ang3 * f2
k = 2.0 * np.pi / lam
mu = 2.0 * k * beta_idx
return 1.0 / mu
```

Term by term,

$$
\lambda=\frac{hc}{E},\qquad
\beta_{\rm idx}=\frac{r_e\lambda^2nf_2}{2\pi},\qquad
k=\frac{2\pi}{\lambda},
$$

so

$$
\mu=2k\beta_{\rm idx}
=2\left(\frac{2\pi}{\lambda}\right)
\left(\frac{r_e\lambda^2nf_2}{2\pi}\right)
=2r_e\lambda nf_2,
$$

and the return value is exactly the independently derived intensity length. The code has one factor `2` for converting amplitude loss to intensity loss, one `2*pi` in $k$, the compensating `2*pi` in $\beta$, and no $\hbar c$ conversion because it obtains wavelength directly from $hc/E$. Every factor and unit matches.

The function handles one elemental contribution. Compound attenuation is formed elsewhere by summing inverse elemental lengths, which is algebraically the independent mixture rule $\mu=2r_e\lambda\sum_j n_jf_{2,j}$.

The validated domain is positive photon energy inside the tabulated Chantler/FFAST range. `henke_dispersion` returns NaN outside that range, including the bremsstrahlung grid's zero-energy bin; downstream NaN handling is an implementation-domain policy, not part of the positive-energy Beer--Lambert derivation.

## Numerical spot check

The symbolic substitution above is exact and uses the implementation's stored units. No additional implementation-dependent material lookup is needed to distinguish the candidate factors: simplifying `2*k*beta_idx` yields `2*R_E_ANG*lam*number_density_per_ang3*f2` identically.

As limiting probes, `number_density_per_ang3 -> 0` or `f2 -> 0` makes `mu -> 0` and the returned length diverge, while doubling either density or $f_2$ halves the returned length, exactly as Beer--Lambert attenuation requires.

## Adjudication

**match**

The implementation exactly reproduces the Henke-$f_2$ Beer--Lambert intensity attenuation coefficient and e-folding length. Signs, the field-to-intensity factor of two, both $2\pi$ factors, the $hc/E$ wavelength conversion, units, positivity, mixture additivity, and limiting cases agree.
