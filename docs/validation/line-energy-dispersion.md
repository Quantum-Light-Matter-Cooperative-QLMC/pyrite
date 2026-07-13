# Validation: line-energy-dispersion

## Independent derivation

Take one spatial lattice harmonic to be

\[
H_{\mathbf g}(\mathbf r)=H_{\mathbf g}\exp(i\mathbf g\!\cdot\!\mathbf r),
\]

the charge trajectory to be \(\mathbf r(t)=\mathbf v t\), and the outgoing
positive-frequency plane wave to be

\[
E_{\rm out}(\mathbf r,t)\propto
\exp[i(\mathbf k\!\cdot\!\mathbf r-\omega t)],
\qquad \mathbf k=\omega\hat{\mathbf n}.
\]

Projection onto the outgoing mode uses its complex-conjugate phase. Along the
trajectory, the complete time-dependent phase is therefore

\[
\Phi(t)=\left[\mathbf g\cdot\mathbf v+
\omega-\mathbf k\cdot\mathbf v\right]t+\Phi_0.
\]

Stationarity gives

\[
\mathbf g\cdot\mathbf v+\omega-
\omega\hat{\mathbf n}\cdot\mathbf v=0,
\]

and hence, verbatim from the independent result,

\[
\boxed{\displaystyle
\omega=-\frac{\mathbf v\cdot\mathbf g}
{1-\hat{\mathbf n}\cdot\mathbf v}}
\]

subject to \(\omega>0\). For a subluminal charge this is equivalently the
harmonic-selection condition \(\mathbf v\cdot\mathbf g<0\). The minus sign is
fixed by the simultaneous choices \(\exp(+i\mathbf g\cdot\mathbf r)\) for the
lattice harmonic and \(\exp[i(\mathbf k\cdot\mathbf r-\omega t)]\) for the
outgoing wave. Relabeling the same harmonic by \(\mathbf g\mapsto-\mathbf g\),
or using \(\exp(-i\mathbf g\cdot\mathbf r)\), reverses the displayed numerator
sign.

For polar tilt, writing

\[
\mathbf g(\alpha)=\sigma G
(\hat{\mathbf s}_0\cos\alpha+\hat{\mathbf u}\sin\alpha)
\]

gives

\[
\mathbf v\cdot\mathbf g(\pm\alpha)
=\sigma vG\left[(\hat{\mathbf b}\cdot\hat{\mathbf s}_0)\cos\alpha
\mathbin{\pm}(\hat{\mathbf b}\cdot\hat{\mathbf u})\sin\alpha\right].
\]

Thus equal and opposite tilts have equal line energy when the untilted sample
normal is beam-aligned, because then
\(\hat{\mathbf b}\cdot\hat{\mathbf u}=0\). The independent derivation does not
claim that equality for a general non-collinear zero-tilt normal.

## Units, conventions, and limiting cases

The repository and derivation both use dimensionless \(\mathbf v\) (speed in
units of \(c\)) and \(\hat{\mathbf n}\), while \(\mathbf g\) and \(\omega\)
are in \(\mathring{\mathrm A}^{-1}\). Photon energy is
\(E=\hbar c\,\omega\), using `HBARC_EV_ANG` in eV Angstrom.

For \(|\mathbf v|<1\),
\(1-\hat{\mathbf n}\cdot\mathbf v\geq1-|\mathbf v|>0\). Therefore:

- \(v\to0\) gives \(\omega\to0\).
- \(\mathbf v\cdot\mathbf g=0\) gives only \(\omega=0\) when the denominator
  is nonzero.
- The denominator becomes small only in the ultrarelativistic forward limit.
- Reversing \(\mathbf g\) reverses the numerator and selects the reciprocal
  member capable of a positive-frequency line.

The sign comparison below is made under the repository's explicit
`structure_factor` convention
\(S(\mathbf g)=\sum_jF_j\exp(+i\mathbf g\cdot\mathbf R_j)\).

## Implementation comparison

`src/cxr_mc/montecarlo/spectrum.py::mc_spectrum` computes, for each segment,

```python
v_dot_g = v_all @ g_vec_d
denom = 1.0 - v_all @ n_hat_d
omega_res = v_dot_g / denom
E_res = HBARC_EV_ANG * omega_res
```

That is

\[
\omega_{\rm code}=+\frac{\mathbf v\cdot\mathbf g_{\rm code}}
{1-\mathbf v\cdot\hat{\mathbf n}},
\qquad E_{\rm code}=\hbar c\,\omega_{\rm code}.
\]

The denominator, \(\hbar c\) conversion, and dimensions match exactly. The
numerator sign does not match the independently derived expression for the
named \(+i\mathbf g\cdot\mathbf r\) harmonic.

The implementation does not document the mapping needed to remove this
difference. `reciprocal_g_vector(hkl, ...)` returns the vector with the sign of
the supplied `hkl`. For that same `hkl`, `structure_factor`, `chi_g`, and `U_g`
use phases `exp(+i * 2*pi * dot(hkl, R))`, while `mc_spectrum` passes the same
`g_vec` to the positive-numerator resonance calculation. There is no explicit
statement or operation establishing that `g_vec` in the resonance is the
negative reciprocal harmonic, or that the coupling is the conjugate
coefficient. Consequently the exact difference is a single unresolved minus
sign in the numerator; equivalence cannot be inferred merely because the
reflection list may also contain `-hkl`.

`checks/anchor_figures.py::line_energy_eV` independently repeats the production
choice as

\[
E=\hbar c\,\frac{\beta |g|}{1-\beta\cos\theta_{\rm obs}},
\]

so it is a same-convention regression anchor, not an implementation-neutral
resolution of the sign.

There is also a scope error in the repository's tilt explanation, although it
does not change the sign adjudication. In the default, beam-aligned,
\(\mathbf g\parallel\) sample-normal geometry at azimuth zero,
`tilted_geometry` returns
\(\mathbf v=\beta(-\sin\alpha,0,\cos\alpha)\) in the sample frame while the
default \(\mathbf g\) remains along \(+z\). Hence
\(\mathbf v\cdot\mathbf g=\beta |g|\cos\alpha\), which is even in
\(\alpha\), just like \(\mathbf v\cdot\hat{\mathbf n}\). The claim in
`docs/tilt-convention.md` and the `tilted_geometry` docstring that opposite
polar tilts differ in intensity *via* `v0.g` is therefore not valid in that
stated scope. Full simulated intensities can still differ through transported
trajectories, escape geometry, polarization/amplitudes, or a non-aligned
reciprocal vector; those effects do not make the default zero-scattering
`v0.g` odd in tilt.

## Numerical spot check

The required implementation-neutral check produced

```text
-0.48480962024633706 -0.484809620246337
-0.4848096202463371  -0.484809620246337
```

for polar tilts \(-17^\circ\) and \(+17^\circ\), respectively, at
\(\theta_{\rm obs}=119^\circ\). Both beam/detector dot products equal
\(\cos(119^\circ)\) to floating-point precision, confirming the lab-frame
Doppler denominator's opposite-tilt invariance.

## Adjudication

**discrepancy**

For the repository's explicitly named
\(\exp(+i\mathbf g\cdot\mathbf r)\) harmonic, the independent phase matching
has \(-\mathbf v\cdot\mathbf g\) while production has
\(+\mathbf v\cdot\mathbf g\). The repository does not explicitly map the
production vector to the opposite/conjugate Fourier harmonic, so the two
expressions cannot presently be certified as equivalent.
