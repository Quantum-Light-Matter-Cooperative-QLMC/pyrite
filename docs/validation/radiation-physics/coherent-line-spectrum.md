# Validation: `coherent-line-spectrum`

## Claim and scope

- **Claim:** `montecarlo/spectrum/lines.py::mc_spectrum` (the incoherent,
  ungrooved, unlayered path) returns the per-electron line spectrum
  $d^2N/(dE\,d\Omega)$ in photons / (eV sr incident-electron) as a sum of
  $\lvert A_{\rm PXR}+A_{\rm CBS}\rvert^2$ contributions over transport
  segments, with an exact incoherent average over mosaic crystallite
  orientations.
- **Source:** Feranchuk, Ulyanenkov, Harada, Spence, *Phys. Rev. E* **62**,
  4225 (2000), Eqs. (10) and (12); Zhai (2025) SI Eqs. (5)–(7).
- **This row covers:** the spectral prefactor multiplying
  $\lvert A\rvert^2$, how per-segment squared amplitudes are accumulated into
  $d^2N/(dE\,d\Omega)$, which degrees of freedom are added in intensity rather
  than in amplitude (segments, electrons, polarizations, reflections, mosaic
  crystallites), the per-electron normalization, and the mosaic orientation
  average.

**Given inputs, not re-verified here.** The constituent amplitudes and the
factors they carry are separately ledgered and are taken as given:
`pxr-amplitude` ($\chi_{\mathbf g}$), `cbs-amplitude` ($U_{\mathbf g}$ and the
braced relativistic PXR/CBS assembly, currently `filtered`),
`finite-time-lineshape` ($\lvert Q\rvert^2=t_L^2\operatorname{sinc}^2$),
`coherent-emission` (the opt-in phased branch), and
`line-absorption-tabulation` ($\mu(E)$ and the Beer–Lambert escape factor).
The open `line-energy-dispersion` harmonic-sign discrepancy sets the *centre*
$\omega_{\rm res}$ of every line; it is noted in passing and is not adjudicated
here, because it does not touch the squaring/summing structure that is this
row's subject.

The derivation in the next section was written from the source, the ledger
row, and the `mc_spectrum` signature and docstring, before the implementation
body was read.

## Independent derivation

### Photon-number spectral-angular density

Work in Gaussian units with $c=1$, so that times and lengths share units
(ångström) and $\hbar c=1973.269804$ eV·Å converts a wavenumber to an energy.
For a charge $-e$ on trajectory $\mathbf r(t)$ with
$\boldsymbol\beta=\dot{\mathbf r}$, the energy radiated per unit solid angle
per unit angular frequency in the far field is

$$
\frac{d^2W}{d\omega\,d\Omega}
=\frac{e^2\omega^2}{4\pi^2c}
\left\lvert\int_{-\infty}^{\infty}
\hat{\mathbf n}\times(\hat{\mathbf n}\times\boldsymbol\beta)\,
e^{i\omega(t-\hat{\mathbf n}\cdot\mathbf r(t)/c)}\,dt\right\rvert^2 .
$$

Resolving the transverse vector on an orthonormal pair
$\{\mathbf e_1,\mathbf e_2\}$ spanning the plane orthogonal to
$\hat{\mathbf n}$, and using
$\mathbf e_s\cdot[\hat{\mathbf n}\times(\hat{\mathbf n}\times\boldsymbol\beta)]
=-\mathbf e_s\cdot\boldsymbol\beta$, the two polarizations are orthogonal
final states and their intensities add:

$$
\frac{d^2W}{d\omega\,d\Omega}
=\frac{e^2\omega^2}{4\pi^2c}\sum_{s=1,2}
\left\lvert\int\mathbf e_s\cdot\boldsymbol\beta\,
e^{i\omega(t-\hat{\mathbf n}\cdot\mathbf r/c)}\,dt\right\rvert^2 .
$$

Photon number follows from $dN=dW/(\hbar\omega)$, and the per-energy
distribution from $dE=\hbar\,d\omega$:

$$
\frac{d^2N}{dE\,d\Omega}
=\frac{1}{\hbar^2\omega}\frac{d^2W}{d\omega\,d\Omega}
=\frac{e^2\omega}{4\pi^2\hbar^2c}\sum_s\lvert\mathcal I_s\rvert^2 .
$$

Setting $c=1$ (so $\mathcal I_s$ has units of length) and substituting
$e^2=\alpha\hbar c$ gives the prefactor this row must certify:

$$
\boxed{
\frac{d^2N}{dE\,d\Omega}
=\frac{\alpha\,\omega}{4\pi^2\hbar c}\sum_{s=1,2}
\lvert\mathcal I_s\rvert^2 } .
$$

### Factorization on one straight segment

In the kinematic (Born) treatment the free-space coupling
$\mathbf e_s\cdot\boldsymbol\beta$ is replaced by the crystal-scattering
amplitude at the reciprocal vector $\mathbf g$: the electron's virtual-photon
field diffracts off the susceptibility harmonic $\chi_{\mathbf g}$ (PXR) and
the electron scatters off the potential harmonic $U_{\mathbf g}$ (CBS). Both
mechanisms produce a real photon in $\hat{\mathbf n}$ and both are *linear in
the polarization vector* $\mathbf e_s$, so the result is a dimensionless
amplitude

$$
A_s=A^{(s)}_{\rm PXR}+A^{(s)}_{\rm CBS},
$$

whose exact tensor structure is the business of `pxr-amplitude` and
`cbs-amplitude`. Within one transport segment the velocity, the amplitude, and
the couplings are constant, so $A_s$ leaves the time integral and only the
residual phase remains:

$$
\mathcal I_s=A_s\int_{\rm seg}
e^{i[\omega t-(\omega\hat{\mathbf n}+\mathbf g)\cdot\mathbf r(t)]}\,dt .
$$

For $\mathbf r(t)=\mathbf r_j+\mathbf v\,(t-t_j)$ over a duration
$t_L=L_{\rm seg}/\beta$ centred on the segment midpoint, the phase advances
linearly at the rate

$$
2P=\omega(1-\mathbf v\cdot\hat{\mathbf n})-\mathbf v\cdot\mathbf g
=(1-\mathbf v\cdot\hat{\mathbf n})(\omega-\omega_{\rm res}),
\qquad
\omega_{\rm res}=\frac{\mathbf v\cdot\mathbf g}
{1-\mathbf v\cdot\hat{\mathbf n}},
$$

so $\mathcal I_s=A_s\,e^{i\phi_j}\,Q(P,t_L)$ with a constant midpoint phase
$\phi_j$ and the finite-time factor of the `finite-time-lineshape` row,

$$
\lvert Q(P,t_L)\rvert^2
=t_L^2\operatorname{sinc}^2\!\left(\frac{P\,t_L}{\pi}\right),
\qquad
\operatorname{sinc}(u)=\frac{\sin(\pi u)}{\pi u}.
$$

Writing the detuning in the photon-energy coordinate,
$E-E_{\rm res}=\hbar c(\omega-\omega_{\rm res})$, gives

$$
\frac{P\,t_L}{\pi}
=\frac{D\,t_L\,(E-E_{\rm res})}{2\pi\hbar c},
\qquad
D=1-\mathbf v\cdot\hat{\mathbf n}.
$$

Escape from the emission point is a Beer–Lambert intensity factor
$T_j=\exp(-\mu(E_{\rm res})L_{{\rm esc},j})$ applied to
$\lvert\mathcal I_s\rvert^2$ (row `line-absorption-tabulation`). One segment,
one reflection therefore contributes

$$
\frac{\alpha\,\omega}{4\pi^2\hbar c}
\left(\sum_s\lvert A_s\rvert^2\right)
t_L^2\operatorname{sinc}^2\!\left(
\frac{D\,t_L\,(E-E_{\rm res})}{2\pi\hbar c}\right)T_j .
$$

Because $A_s$ is linear in $\mathbf e_s$, the sum
$\sum_s\lvert A_s\rvert^2$ is the squared norm of the transverse projection of
one fixed complex vector and is therefore **independent of which orthonormal
basis of the plane $\perp\hat{\mathbf n}$ is used**. That is a testable
consequence, exercised numerically below.

### Incoherence structure

The incoherent path is defined by discarding every phase $\phi_j$ that
survives the segment integral, i.e. by the replacement

$$
\left\lvert\sum_j\mathcal I_{s,j}\right\rvert^2
\;\longrightarrow\;
\sum_j\lvert\mathcal I_{s,j}\rvert^2 .
$$

Each degree of freedom summed in intensity carries its own justification, of
differing strength:

- **Polarizations** — exact. $\mathbf e_1$ and $\mathbf e_2$ are orthogonal
  final photon states; their amplitudes never interfere in a
  polarization-insensitive measurement.
- **Electrons** — very good. Distinct electrons carry independent arrival
  times $t_{0e}$, so cross terms are weighted by
  $\langle e^{i\omega(t_{0e}-t_{0e'})}\rangle$, which vanishes for a bunch
  much longer than the emitted wavelength. The `coherent-emission` row is
  precisely the opposite (short-bunch, superradiant) limit.
- **Mosaic crystallites** — very good. Distinct grains are macroscopically
  separated with uncorrelated orientation, so their relative phases average
  away.
- **Reflections** — an approximation. Distinct $\mathbf g$ resonate at
  well-separated $\omega_{\rm res}$, so each line sits in the far
  $\operatorname{sinc}^2$ tail of the others and cross terms are small, but
  they are not identically zero.
- **Segments of one electron** — the weakest link, and a deliberate model
  choice. Consecutive transport segments belong to one continuous trajectory,
  so their relative phase is physical, not random; discarding it is exactly
  what the opt-in `coherent=True` branch reinstates. The consequence is
  quantified as a limiting-case filter below.

### Per-electron normalization and the mosaic average

A mosaic crystal is an incoherent ensemble of crystallites whose lattice is
tilted by a small random angle $\boldsymbol\delta=(\delta_x,\delta_y)$, drawn
from an isotropic 2-D Gaussian of per-axis width
$\sigma=\mathrm{FWHM}/(2\sqrt{2\ln 2})$. The exact average is the
intensity-weighted orientation integral

$$
\left\langle\frac{d^2N}{dE\,d\Omega}\right\rangle
=\int d^2\boldsymbol\delta\;
\frac{e^{-\lvert\boldsymbol\delta\rvert^2/2\sigma^2}}{2\pi\sigma^2}\;
\frac{d^2N}{dE\,d\Omega}\Big[\mathbf g\to R(\boldsymbol\delta)\,\mathbf g\Big],
$$

with the macroscopic slab geometry — $\hat{\mathbf n}$, the faces, the escape
path — held fixed, since only the lattice inside a grain is tilted. Under
$\delta=\sqrt2\,\sigma x$ each axis becomes a Gauss–Hermite integral,

$$
\int\frac{e^{-\delta^2/2\sigma^2}}{\sqrt{2\pi}\sigma}f(\delta)\,d\delta
=\frac{1}{\sqrt\pi}\int e^{-x^2}f(\sqrt2\sigma x)\,dx
\approx\sum_a\frac{w_a}{\sqrt\pi}\,f(\sqrt2\sigma x_a),
$$

so the 2-D product rule has nodes $R(\sqrt2\sigma x_a,\sqrt2\sigma x_b)$ and
weights $w_aw_b/\pi$, which sum to unity. (The quadrature itself is the
`mosaic-mc` row; what this row needs is that the average is taken *in
intensity* and that the weights are normalized.)

Assembling everything, and dividing by the number of incident electrons
$N_e$ to reach the documented per-electron unit:

$$
\boxed{
\frac{d^2N}{dE\,d\Omega}(E)
=\frac{1}{N_e}\sum_m w_m\sum_{\mathbf g}\sum_j
\frac{\alpha\,\omega_{{\rm res},j}}{4\pi^2\hbar c}
\Big(\sum_s\lvert A^{(s)}_{{\rm PXR},j}+A^{(s)}_{{\rm CBS},j}\rvert^2\Big)
t_{L,j}^2\operatorname{sinc}^2\!\left(
\frac{D_j t_{L,j}(E-E_{{\rm res},j})}{2\pi\hbar c}\right)
e^{-\mu(E_{{\rm res},j})L_{{\rm esc},j}} }
$$

with $\sum_m w_m=1$ and $\mathbf g\to R_m\mathbf g$ inside every
orientation-dependent factor.

One deliberate approximation enters here. The strictly correct prefactor
carries the *observed* frequency $\omega=E/\hbar c$; freezing it at
$\omega_{{\rm res},j}$ is the narrow-line approximation already applied to
$\chi_{\mathbf g}$, $U_{\mathbf g}$, and $\mu$, all evaluated at
$E_{\rm res}$. Its relative size across one line is
$\Delta E_{\rm FWHM}/E_{\rm res}$, and because $\operatorname{sinc}^2$ is even
about $E_{\rm res}$ the error is odd and cancels to first order in the
energy-integrated yield.

## Cheap filters

| filter | independent expectation | result |
| --- | --- | --- |
| units | $[\alpha\,\omega\,t_L^2/\hbar c]=\mathrm{\AA^{-1}\,\AA^2/(eV\,\AA)}=\mathrm{eV^{-1}}$ with $\lvert A\rvert^2$, $\operatorname{sinc}^2$, $T$ dimensionless, per sr, per electron | pass |
| $N_e$ scaling | replicating the ensemble scales the segment sum and $N_e$ together; the per-electron spectrum is invariant | pass |
| $N_{\rm seg}$ scaling | intensity addition makes the peak fall as $1/N$ under $N$-fold subdivision of one straight flight, while the energy-integrated yield $\propto\sum_j t_{L,j}$ is invariant | pass, with a caveat (below) |
| single-segment closed form | on resonance the peak is $\alpha\omega\lvert A\rvert^2t_L^2T/(4\pi^2\hbar c)$; the integrated line area is $\alpha\omega\lvert A\rvert^2 t_L T/(2\pi D)$ | pass |
| mosaic $\eta\to0$ | $R_m\to\mathbb 1$ and $\sum_m w_m=1$ recover the single-orientation spectrum | pass |
| signs / conventions | $\lvert A\rvert^2\ge0$ and real by construction; $T\in(0,1]$; the PXR/CBS relative sign is interior to $A$ and belongs to `cbs-amplitude` | pass |

**Subdivision caveat.** Because
$\int\lvert Q\rvert^2dE=2\pi t_L\hbar c/D$ is linear in $t_L$ while the peak
goes as $t_L^2$, the incoherent model's *integrated* line yield depends only
on the total path length, but its *peak height* depends on how that path is
cut into segments. This is a genuine property of the incoherent
approximation, not an implementation defect: it is the intra-electron
coherence that `coherent-emission` restores. It is defensible here only
because a segment is a physical elastic flight rather than a numerical step —
the `energy-step-convergence` row records that the fractional-energy-loss
rungs are no-ops, so segments are not artificially subdivided.

## Implementation comparison

The ungrooved, unlayered, incoherent run takes the batched branch of
`mc_spectrum`. Its accumulation reduces to (literal source, with the
per-hkl `_accumulate` fallback spelling the same thing inline):

```python
pref   = alpha_fs * omega_res / pref_c1 * (t_L * t_L) * T_abs   # _line_weight_core
weight = pref * A2 * WM
a_width = denom * t_L / (2.0 * HBARC_EV_ANG) * xp.ones_like(omega_res)
S = _sincsq_lineshape(a_width, E_grid, E_res)                   # sinc(a_width*(E-E_r)/pi)**2
spec += weight @ S
...
return _to_cpu(spec / Ne)
```

with `_PREF_C1 = 4.0 * xp.pi**2 * HBARC_EV_ANG`, `A2` the sum of
`_line_amp_sq_core` over the `(ES, EP)` polarization pair, and `WM` the
`(1, N_g)` row of mosaic weights from `_mosaic_quadrature`.

Term by term against the boxed independent result:

| independent term | implementation | agreement |
| --- | --- | --- |
| $\alpha/(4\pi^2\hbar c)$ | `ALPHA_FS / _PREF_C1` with `_PREF_C1 = 4 pi^2 HBARC_EV_ANG` | exact |
| $\omega_{{\rm res},j}$ | `omega_res` = `v_dot_g / denom` | exact, frozen-$\omega$ convention as derived |
| $\sum_s\lvert A_{\rm PXR}+A_{\rm CBS}\rvert^2$ | `A2` accumulated over `(ES, G_DOT_ES)` and `(EP, G_DOT_EP)` | exact; basis-independence confirmed numerically |
| $t_{L,j}^2$ | `t_L * t_L`, `t_L = L_ang / beta` | exact |
| $\operatorname{sinc}^2(D t_L(E-E_{\rm res})/2\pi\hbar c)$ | `a_width = denom * t_L / (2 HBARC_EV_ANG)`, `sinc(a_width*(E-E_r)/pi)**2` | exact |
| $e^{-\mu L_{\rm esc}}$ | `T_abs = exp(-(L_esc * mu))` inside `_line_weight_core` | exact |
| $\sum_j$ over segments | `weight @ S` matrix product over the segment axis | intensity sum, exact |
| $\sum_{\mathbf g}$ over reflections | rows of the `(n_seg, N_g)` grid, all reduced into one `spec` | intensity sum, exact |
| $\sum_m w_m$ over crystallites | `WM` multiplies `weight`; orientation rows also reduce into `spec` | intensity sum, weights sum to 1 |
| $1/N_e$ | `spec / Ne`, with `Ne = segments["Ne"]` or `electron_limit` and `line_electron = elec_id < Ne` masking surplus rows | exact |

Structural points confirmed on inspection:

- The mosaic rotation is applied to $\mathbf g$ and to the derived
  polarization pair only. $\hat{\mathbf n}$, the slab, and the escape geometry
  stay fixed, which is the correct picture of a tilted grain inside a fixed
  macroscopic crystal. $\chi_{\mathbf g}$ and $U_{\mathbf g}$ are tabulated per
  reflection and shared across orientations — correct, since they depend on
  $\lvert\mathbf g\rvert$ and the structure factor, both rotation invariant —
  while $\mu(E_{\rm res})$ and the escape factor *are* recomputed per
  orientation, because $E_{\rm res}$ shifts with the tilt.
- Segments are masked by `keep`, which drops lines whose $E_{\rm res}$ falls
  outside the output grid padded by 20% of its span, plus a hard
  $E_{\rm res}>10$ eV floor. This truncates the $\operatorname{sinc}^2$ tails
  of far-off-window lines rather than approximating them; it is a windowing
  choice, not a physics divergence.
- `components=True` returns $\lvert A_{\rm PXR}\rvert^2$ and
  $\lvert A_{\rm CBS}\rvert^2$ separately and therefore **drops the
  interference term** $2\,\mathrm{Re}(A_{\rm PXR}A_{\rm CBS}^{*})$. The split
  is a diagnostic, not a decomposition of the returned total.

## Numeric checks

All checks use a synthetic segment array fed directly to `mc_spectrum`, and a
reference spectrum coded from the boxed expression above with hard-coded
CODATA constants ($\hbar c=1973.269804$ eV·Å, $\alpha^{-1}=137.035999084$,
$m_ec^2=510998.95$ eV) and with an **arbitrary** orthonormal polarization
basis rather than the implementation's $\sigma/\pi$ construction. Only
$\chi_{\mathbf g}$ and $U_{\mathbf g}$ are taken from the repository, as the
given inputs of their own ledger rows. Case: HOPG $(002)$, 30 keV, one
$L=290$ Å segment along $+z$, $\theta_{\rm obs}=119^\circ$,
$B=0.6$ Å², giving $E_{\rm res}=1046.700402$ eV.

| check | result |
| --- | --- |
| absolute prefactor and lineshape, single segment, zero escape path | peak $2.1200470571\times10^{-9}$ (code) vs $2.1200470556\times10^{-9}$ (reference); ratio $1+6.69\times10^{-10}$ |
| same, with the reference switched to the repository's $\alpha^{-1}=137.035999$ | ratio $1+5.6\times10^{-11}$ — the $6.7\times10^{-10}$ offset is entirely the fine-structure constant literal, the $5.6\times10^{-11}$ residual is $\chi/U$ table interpolation |
| nonzero emission depth ($z=3000$ Å, $T_{\rm abs}=0.774$) | ratio $1+6.69\times10^{-10}$, unchanged |
| polarization-basis independence | implied by the two rows above: an arbitrary basis reproduces the $\sigma/\pi$ result to $10^{-10}$ |
| subdivision into $N$ collinear sub-segments, peak | peak$\times N$ = $2.119932$, $2.120020$, $2.120042$, $2.120047$, $2.120049\times10^{-9}$ for $N=1,2,4,8,16$ — exact $1/N$ falloff, i.e. intensity addition, not amplitude addition (which would leave the peak invariant) |
| subdivision, energy-integrated yield over a 10 eV–60 keV grid | $1.000000$, $0.999401$, $0.998181$, $0.995847$, $0.990724$ — invariant up to the $\operatorname{sinc}^2$ tail lost outside the window, which grows as the lines widen |
| single-segment closed form $\alpha\omega\lvert A\rvert^2t_LT/(2\pi D)$ | $2.5675986\times10^{-8}$ vs MC trapezoid $2.5660567\times10^{-8}$ (ratio $0.99940$); the independent reference integrated on the same grid gives $2.5660540\times10^{-8}$, so the $6\times10^{-4}$ deficit is quadrature/tail truncation shared by both, and code/reference $=1.0000011$ |
| $N_e$ scaling, $N_e=1,3,7$ identical electrons | per-electron spectra agree to $9.8\times10^{-17}$ of peak |
| `electron_limit=3` out of 6 electrons | agrees with the $N_e=1$ per-electron spectrum to $9.8\times10^{-17}$ of peak |
| reflection additivity, $\{(002)\}+\{(004)\}$ vs joint call | max difference $0.0$ (bitwise) |
| mosaic weights | $\sum_m w_m=1.000000000000000$ for $5\times5$ and $7\times7$ nodes |
| mosaic $\eta\to0$ ($10^{-7}$ deg, 5 nodes) | peak and integral both $1.000000$ of the perfect-crystal result |
| mosaic average vs an independently weighted average of per-orientation *intensities* | max difference $4.6\times10^{-8}$ (0.4 deg, 5 nodes) and $2.6\times10^{-7}$ (1.0 deg, 7 nodes) of peak |
| PXR/CBS interference size at the peak | total $2.120\times10^{-9}$, PXR $1.616\times10^{-9}$, CBS $3.450\times10^{-11}$, cross term $+4.691\times10^{-10}$ = $+22.1\%$ of the total |

The last row is direct evidence for the ledger's note that the interference is
non-separable: at this operating point more than a fifth of the line intensity
comes from the PXR–CBS cross term, so neither amplitude may be squared alone.

## Findings

1. **No divergence in this row's subject.** Prefactor, squaring, summation
   order, per-electron normalization, and mosaic weighting all reproduce the
   independent expression to the level of the fine-structure-constant literal
   ($6.7\times10^{-10}$).
2. **Frozen prefactor frequency.** The implementation evaluates the
   $\alpha\omega/(4\pi^2\hbar c)$ prefactor at $\omega_{\rm res}$, not at the
   grid frequency. This is consistent with the narrow-line freezing already
   applied to $\chi_{\mathbf g}$, $U_{\mathbf g}$, and $\mu$; the error is odd
   about the line centre and cancels to first order on integration. Documented,
   not a defect.
3. **Peak height is segmentation dependent.** The incoherent sum is exact in
   integrated yield and approximate in peak height. Absolute line-peak
   comparisons against a paper therefore inherit an implicit dependence on the
   transport's segment-length distribution. This is the known boundary between
   this row and `coherent-emission` and should stay visible in the ledger note.
4. **Upstream cap.** The Eq. (13)/(14) tensor structure of $A_{\rm PXR}$ and
   $A_{\rm CBS}$ is ledgered under `cbs-amplitude` at status `filtered`, not
   `rederived`. This row certifies how $\lvert A\rvert^2$ is used; it cannot
   raise the certainty of $A$ itself.
5. **Open sign question, untouched.** `line-energy-dispersion` remains a
   `discrepancy`: the independent $\exp(+i\mathbf g\cdot\mathbf r)$ derivation
   gives $-\mathbf v\cdot\mathbf g/(1-\mathbf v\cdot\hat{\mathbf n})$ where
   production uses $+$. That fixes where each line sits, not how it is squared
   or summed, so it does not block this row — but a signed-off
   `coherent-line-spectrum` still stands on an unresolved $\omega_{\rm res}$
   sign.
6. **`components=True` is not a decomposition.** `spec_pxr + spec_cbs` differs
   from `spec` by the interference term, which is $+22\%$ at the checked point.
   Any downstream use as an additive split is wrong; the docstring says so for
   the coherent path but not for the incoherent one.

## Adjudication

**match.**

The independently derived expression

$$
\frac{d^2N}{dE\,d\Omega}
=\frac{1}{N_e}\sum_m w_m\sum_{\mathbf g}\sum_j
\frac{\alpha\,\omega_{{\rm res},j}}{4\pi^2\hbar c}
\Big(\sum_s\lvert A^{(s)}_{\rm PXR}+A^{(s)}_{\rm CBS}\rvert^2\Big)
t_{L,j}^2\operatorname{sinc}^2\!\left(
\frac{D_j t_{L,j}(E-E_{{\rm res},j})}{2\pi\hbar c}\right)
e^{-\mu L_{{\rm esc},j}}
$$

is reproduced by `mc_spectrum` term for term, with no divergent factor, sign,
exponent, or unit. Numerically the absolute normalization agrees to
$6\times10^{-10}$ relative, whose entire budget is the repository's truncated
fine-structure constant. The claimed incoherence structure holds exactly in
the code: segments, electrons, polarizations, reflections, and mosaic
orientations all add in intensity, confirmed by the exact $1/N$ subdivision
falloff, the bitwise reflection additivity, and the invariance of the
per-electron spectrum under electron replication. The mosaic average is an
intensity average over a Gauss–Hermite quadrature whose weights sum to unity
and which collapses to the perfect crystal as $\eta\to0$.

Verdict: `rederived`. Not `anchored` — the ledgered anchor
`src/pyrite/apps/anchor_figures.py::single_segment_anchor` compares
`mc_spectrum` against another implementation helper
(`feranchuk_line_flux`, itself the `unverified` `closed-form-flux` row) and is
not a CI-green regression test on this row's normalization. Not `signed-off`;
only a human applies that.
