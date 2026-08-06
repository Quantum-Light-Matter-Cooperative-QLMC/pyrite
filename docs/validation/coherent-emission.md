# Validation: `coherent-emission`

## Independent derivation (recorded before implementation inspection)

**Claim.** For one reciprocal reflection and polarization, radiation from
transport segments and electrons is added as a complex field. The intended
quantity is \(d^2N/(dE\,d\Omega)\) in
photons / (eV sr incident-electron), with the historical incoherent result
recovered when coherent summation is disabled.

**Sources and convention.** Feranchuk--Spence (2000), Eqs. (10), (12)--(14),
supplies the finite-interaction PXR+CBS amplitude and resonance; the repository
Zhai derivation writes
\(\epsilon(\mathbf r)=\epsilon_0+\sum_{\mathbf g}\chi_{\mathbf g}
\exp(+i\mathbf g\cdot\mathbf r)\) and its PXR delta function as
\(\delta[\omega-\mathbf k\cdot\mathbf v+\mathbf g\cdot\mathbf v]\).
The repository crystallographic coefficient instead is

\[
S(\mathbf g)=\sum_a F_a(\mathbf g)
             \exp(+i\mathbf g\cdot\mathbf R_a).
\]

This coefficient belongs to the **opposite spatial harmonic**. Directly
Fourier-transforming translated atomic densities gives

\[
\chi(\mathbf r)
 =\sum_{\mathbf q}\widetilde\chi_{\mathbf q}
   e^{+i\mathbf q\cdot\mathbf r},\qquad
\widetilde\chi_{\mathbf q}\propto
\sum_a F_a(\mathbf q)e^{-i\mathbf q\cdot\mathbf R_a}
=S(-\mathbf q).
\]

Therefore the value named \(\chi_{\mathbf g}\propto S(\mathbf g)\) in this
repository reconstructs in real space as

\[
\boxed{\chi_{\mathbf g}\exp(-i\mathbf g\cdot\mathbf r)}.
\]

This coefficient-to-harmonic mapping is the sign pivot. It is not equivalent
to inserting \(S(\mathbf g)\) beside \(e^{+i\mathbf g\cdot\mathbf r}\).

Take an outgoing positive-frequency wave
\(\exp[i(\mathbf k\cdot\mathbf r-\omega t)]\),
\(\mathbf k=\omega\hat{\mathbf n}\), with \(c=1\). Projection onto that
outgoing mode supplies its conjugate phase
\(\exp[i(\omega t-\mathbf k\cdot\mathbf r)]\). Multiplication by the
repository susceptibility harmonic gives the source phase

\[
\boxed{\Phi_{\mathbf g}(t,\mathbf r)
=\omega t-(\mathbf k+\mathbf g)\cdot\mathbf r}
=\omega(t-\hat{\mathbf n}\cdot\mathbf r)
-\mathbf g\cdot\mathbf r .
\]

For a straight trajectory
\(\mathbf r(t)=\mathbf r_c+\mathbf v(t-t_c)\), the time-dependent part is

\[
\Phi_{\mathbf g}(t)
=\Phi_{\mathbf g,c}
+[\omega(1-\hat{\mathbf n}\cdot\mathbf v)
-\mathbf g\cdot\mathbf v](t-t_c).
\]

Stationarity therefore gives

\[
\boxed{\omega_{\rm res}
=\frac{\mathbf g\cdot\mathbf v}
       {1-\hat{\mathbf n}\cdot\mathbf v}} .
\]

Thus the segment-centre phase

\[
\boxed{\exp\{i[\omega(t_{\rm abs,j}
-\hat{\mathbf n}\cdot\mathbf r_j)
-\mathbf g\cdot\mathbf r_j]\}}
\]

and the repository's positive-numerator resonance must occur together. At
\(\omega=\omega_{\rm res}\), phase advance along a straight trajectory is
exactly zero. Using \(+\mathbf g\cdot\mathbf r_j\) with that resonance would
leave a spurious \(2\mathbf g\cdot\mathbf v\) phase slope. Conversely, the
Zhai coefficient multiplying \(e^{+i\mathbf g_{\rm Z}\cdot\mathbf r}\) has
\(\mathbf g_{\rm Z}=-\mathbf g_{\rm repo}\), so its
\(\omega=-\mathbf g_{\rm Z}\cdot\mathbf v/(1-\hat{\mathbf n}\cdot\mathbf v)\)
is identical.

For a centered constant-amplitude segment of duration \(t_L\), let

\[
P={1\over2}(1-\hat{\mathbf n}\cdot\mathbf v)
(\omega-\omega_{\rm res}).
\]

The exact segment integral is

\[
\int_{-t_L/2}^{t_L/2}e^{i\,2P\tau}\,d\tau
=t_L\,\operatorname{sinc}\!\left({P t_L\over\pi}\right)
\equiv Q_j,
\]

where NumPy's normalized sinc is
\(\operatorname{sinc}(x)=\sin(\pi x)/(\pi x)\). Hence coherent summation must
use one unsquared real \(Q_j\); squaring the total field yields the historical
\(t_L^2\operatorname{sinc}^2(Pt_L/\pi)\) self-term. A windowed evaluation may
omit grid points outside a cutoff, but it must use the same \(P\), argument,
and unsquared \(Q_j\) on retained points as the full-grid path.

For polarization \(p\), the independently expected field is

\[
\mathcal E_{p,\mathbf g}(\omega)=
\sum_j
\sqrt{\frac{\alpha\omega}{4\pi^2\hbar c}\,T_j}\,
A_{p,j}Q_j
e^{i\Phi_{\mathbf g,j}},
\qquad
\frac{d^2N}{dE\,d\Omega}
=\sum_p|\mathcal E_{p,\mathbf g}|^2 .
\]

Different reciprocal vectors are assumed spectrally separated and mosaic
crystallites are mutually incoherent. Therefore the phased segment/electron
sum belongs **inside** each fixed-reflection, fixed-mosaic-orientation
evaluation; reflection intensities, mosaic quadrature intensities, and
polarization intensities add outside it.

## Cheap filters expected before code comparison

- **Units.** \(\omega,t,\mathbf g,\mathbf r\) use reciprocal Angstrom and
  Angstrom with \(c=1\), so every phase and sinc argument is dimensionless.
  \(E=\hbar c\,\omega\). \(Q\) has Angstrom units; its square supplies the
  historical finite-time factor.
- **Single segment.** Only its self-term remains, exactly equal to incoherent
  accumulation.
- **Coincident in-phase emitters.** \(N\) equal fields give \(N^2\) intensity
  before the existing per-electron normalization.
- **Decoherent phases.** Independent broad/random per-electron `t0_ang`
  eliminates inter-electron cross terms in expectation, leaving
  \(\sum_e|\sum_{j\in e}\mathcal E_{ej}|^2\); intra-electron segment cross
  terms survive. Fully segmentwise-independent phases would instead leave
  \(\sum_{e,j}|\mathcal E_{ej}|^2\).
- **Gaussian bunch.** For longitudinal offset
  \(t_0\sim N(0,\sigma_z^2)\),
  \(|\langle e^{i\omega t_0}\rangle|^2
  =e^{-\omega^2\sigma_z^2}\).
- **Mosaic placement.** Phase uses the same rotated \(\mathbf g\) as that
  orientation's resonance and amplitude; different orientations add
  incoherently with their quadrature weights.
- **Absorption.** Since historical intensity contains \(T_j\), each coherent
  field must contain \(\sqrt{T_j}\).

## Implementation comparison

The corrected implementation computes

```text
g_phase = r_mid @ g
phase = exp(1j * ((t_ang + t0_ang - n_hat.r_mid) * omega_grid - g_phase))
```

in both the full-grid and `sinc_cutoff` branches. This is exactly

\[
\exp\{i[\omega(t_{\rm abs}-\hat{\mathbf n}\cdot\mathbf r)
-\mathbf g\cdot\mathbf r]\}.
\]

Its resonance is
\(\omega_{\rm res}=\mathbf v\cdot\mathbf g/
(1-\mathbf v\cdot\hat{\mathbf n})\). Therefore a straight segment-to-segment
displacement \((\Delta t,\Delta\mathbf r)\) with
\(\Delta\mathbf r=\mathbf v\Delta t\) has

\[
\Delta\Phi
=\{\omega(1-\hat{\mathbf n}\cdot\mathbf v)
-\mathbf g\cdot\mathbf v\}\Delta t=0
\]

at resonance. Sign and factor match exactly; there is no extra factor of two.
The new parameterized regression verifies the resulting fourfold intensity
for two equal segments at resonance for both `sinc_cutoff=None` and
`sinc_cutoff=4.0`.

Both sinc paths use

```text
x = [dnm * t_L / (2 HBARC)] * (E_grid - E_res) / pi
SP = sinc(x) * phase
coefficient = sqrt[prefactor * T_abs] * t_L * A
```

so both retain the unsquared
\(Q=t_L\operatorname{sinc}(Pt_L/\pi)\). The cutoff path changes only the
evaluated energy slice. The field contains \(\sqrt{T_{\rm abs}}\), and its
self-term exactly reproduces the incoherent \(T_{\rm abs}\) factor.

Mosaic placement also matches: `_accumulate` receives the rotated
\(\mathbf g_m=R_m\mathbf g\); that same vector controls resonance, amplitude,
polarization, and `g_phase`. Each orientation's complex field is squared
before multiplication by its positive quadrature weight, so mosaic
orientations remain incoherent.

Single-segment and coincident-emitter limits match. With the function's final
division by `Ne`, two otherwise identical electrons produce twice the
per-electron spectrum, corresponding to the expected fourfold total intensity
before normalization.

The corrected **decoherent limit matches**. `t0_ang` is common to every
segment belonging to electron \(e\). Writing all non-bunch factors as
\(B_{ej}\), the field is

\[
\mathcal E=\sum_e e^{i\omega t_{0,e}}\sum_{j\in e}B_{ej}.
\]

For a broad or randomized bunch, averaging eliminates only terms with
\(e\ne e'\):

\[
\left\langle|\mathcal E|^2\right\rangle
\longrightarrow
\sum_e\left|\sum_{j\in e}B_{ej}\right|^2,
\]

not
\(\sum_{e,j}|B_{ej}|^2\). This is now exactly the docstring's stated limit:
cross terms between segments of the same electron survive because their
common \(t_{0,e}\) cancels. Recovery of the historical segmentwise incoherent
sum would additionally require one segment per electron, independent
per-segment phase randomization, or an averaging mechanism that destroys
intra-electron coherence. The Gaussian factor
\(\exp[-\omega^2\sigma_z^2]\) is correct specifically for inter-electron cross
terms.

Focused CPU result:

```text
tests/montecarlo/test_coherent_emission.py: 5 passed in 1.81s
```

The tests anchor the self-term, coincident-electron scaling, corrected
straight-trajectory phase for both sinc routes, and invalid component split.
They do not numerically anchor the qualified decoherent limit above; its
analytic characteristic-function limit matches the corrected claim.

## Adjudication

**rederived**

The coherent phase correction itself is rederived and matches exactly:
\(\exp\{i[\omega(t-\hat{\mathbf n}\cdot\mathbf r)
-\mathbf g\cdot\mathbf r]\}\), with no divergent sign or factor. The corrected
long-bunch/scrambled-`t0_ang` statement also matches:
\(\sum_e|\sum_{j\in e}B_{ej}|^2\). Bunch-length decoherence suppresses
inter-electron cross terms only; intra-electron segment coherence survives.

Suggested ledger action: advance `coherent-emission` to `rederived`; retain the
qualified per-electron decoherent limit. Human applies any ledger transition.
