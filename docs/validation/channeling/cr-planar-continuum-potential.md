# `cr-planar-continuum-potential` — plane-averaged electron potential energy

Verifier: fresh context. Ledger: `docs/validation/ledger-channeling.md`. Code: `src/pyrite/materials/channeling/potential.py::{planar_profile,PlanarProfile.potential}`.

## Claim

$U(x)=\sum_{n\ne0}U_ne^{ingx}$ with

$$
U_n=-\frac{4\pi e^2}{Vg_n^2}\sum_j\bigl(Z_j-f_{0,j}(g_n)\bigr)e^{i2\pi n\,\mathbf h\cdot\mathbf R_j}e^{-B_js_n^2},
$$

$g_n=n g_{hkl}$, $s_n=g_n/4\pi$, $U_0=0$, $x$ measured from the cell origin along the plane normal. The true period follows from the gcd of non-extinct harmonics. Nuclear and electron densities use the same thermal smearing.

## Independent derivation (before reading the body)

The 3D crystal potential energy is $U(\mathbf r)=\sum_{\mathbf G}U_{\mathbf G}e^{i\mathbf G\cdot\mathbf r}$, with $U_{\mathbf G}$ as in `cr-mott-bethe-factor`. Averaging over the two coordinates parallel to a plane family with coprime $\mathbf h$ kills every $\mathbf G$ not parallel to $\mathbf g_{hkl}$. The survivors are $\mathbf G=n\,\mathbf g_{hkl}$, $n\in\mathbb Z$. Then $\mathbf G\cdot\mathbf R_j=2\pi n\,\mathbf h\cdot\mathbf R_j$ for fractional $\mathbf R_j$, and $\mathbf G\cdot\mathbf r=n g x$ with $x$ along $\hat{\mathbf g}$ from the origin. This gives the claimed series.

Reality: $U(x)$ is real, so $U_{-n}=U_n^*$. The code keeps $n\ge0$, so it should synthesize $U_0+2\,\mathrm{Re}\sum_{n>0}U_ne^{ingx}$. Thermal smearing multiplies the total charge of site $j$ by $e^{-B_js^2}$, applied to the nucleus and electron densities alike. Number densities are $n_{\rm nuc}(n)=V^{-1}\sum_je^{i\phi_j}e^{-B_js^2}$ and $\rho_e(n)=V^{-1}\sum_jf_{0,j}e^{i\phi_j}e^{-B_js^2}$. At $n=0$ these give $N_{\rm atoms}/V$ and $N_e/V$.

Diamond structure. Basis is fcc plus $(\tfrac14,\tfrac14,\tfrac14)$, with structure factor $S=F_{\rm fcc}\,(1+e^{i\pi(h+k+l)/2})$. $F_{\rm fcc}=4$ if $h,k,l$ are all odd or all even, otherwise 0.

- (110), harmonics $(n,n,0)$. $n$ odd is mixed parity, so $S=0$. $n=2$: $h+k+l=4$, so $1+e^{i2\pi}=2$, $S=8$. Allowed harmonics are $n$ even, so the gcd is 2 and the period is $d_{110}/2=d_{220}$. The planes are equally spaced; atoms sit at $\mathbf h\cdot\mathbf R\in\{0,\tfrac12\}$ mod 1.
- (100), harmonics $(n,0,0)$. $n=2$: $1+e^{i\pi\cdot2/2}\!=1+e^{i\pi}=0$, forbidden. $n=4$ gives $S=8$. The period is $d_{400}$.
- (111), harmonics $(n,n,n)$. $n=1$: $|S|=4\sqrt2$. $n=2$: $1+e^{i3\pi}=0$. $n=3$ is allowed. The gcd is 1 and the period is $d_{111}$.

Centrosymmetry. For (110) the two plane sets at $0$ and $\tfrac12$ are mirror images about $x=0$, so $U_n$ is real and $U(x)$ is even. The expectation is wells at $x=0$ and $x=d_{110}/2$.

Dimensions: $e^2=14.3996\ \text{eV\,\AA}$; $e^2/(Vg^2)$ has units eV·Å/(Å³·Å⁻²) = eV. ✓ Densities are Å⁻³. ✓

## Numerical independent reference: Si (110), 293.15 K

Script uses `xraydb.f0`, $a=5.431\ \text{\AA}$, $V=a^3$, $B=0.48166\ \text{\AA}^2$ (the repository PDOS value), eight explicit atoms, and all harmonics with $s\le6\ \text{\AA}^{-1}$ (that is $|n|\le46$). A first pass truncated at $|n|\le5$ and gave 17.7 eV, which is a truncation artefact. With the full sum:

| quantity | independent | implementation |
| --- | --- | --- |
| $U_{220}$ [eV] | $-4.31575$ | $-4.31589$ |
| $U_{440}$ [eV] | $-1.47340$ | $-1.47344$ |
| $U_{660}$ [eV] | $-0.70758$ | $-0.70759$ |
| odd harmonics | 0 | not present (period halved) |
| $\min U$ at $x=0$ [eV] | $-14.4153$ | $-14.4157$ |
| $\max U$ at $x=d_{110}/4$ [eV] | $6.6241$ | $6.6243$ |
| depth [eV] | 21.039 | 21.040 |

Imaginary parts of $U_n$ are at the $10^{-15}$ level. The depth of 21.0 eV agrees with Azadegan Fig. 1 (about 21 eV) and is within the ~22 eV literature value, the 5% difference being the thermal amplitude and form-factor parametrization. Wells are at the atomic planes ($x=0$ and $x=d/2$) as required. The profile evaluated at nine points over two periods agrees to $4\times10^{-4}$ eV.

## Comparison with the implementation

- Per-site sum: `phases = exp(i 2π n (hkl·R))`; `site_sum = Σ phases · exp(-B s²)`; potential accumulates `(Z - f0)/g² · site_sum`, times `-4π E2/V`. Matches term by term, including the $e^{-Bs^2}$ on the full bracket.
- Origin and phase: $\mathbf h\cdot\mathbf R_j$ uses the coprime $\mathbf h$ and the fractional basis; $x$ is therefore from the cell origin. ✓
- `nuclear = Σ site_sum / V`, `electrons = Σ site_sum f0 / V`, with $f_0(0)=Z$ at $n=0$. ✓ Mean densities are 8/V and 112/V for Si.
- Period: `live` harmonics have $|U_n|>10^{-9}\max|U|$, `step = gcd(live)`, `keep = n[::step]`, `period = 2π/(step g_prim)`. The phases are computed before the stride, so index $m$ maps to $n=m\cdot\text{step}$. ✓ Verified numerically: period 1.920113 Å and the coefficient list line up with the independent table above.
- `_synthesize`: $c_0^{\rm Re}+2\,\mathrm{Re}\sum_{m\ge1}c_me^{imgx}$, as derived. $U_0=0$ because `potential[0]` is never accumulated. ✓
- One caveat: the extinction test uses the potential rather than the structure factor. A harmonic that vanishes accidentally on $Z-f_0$ for a non-extinct site would be dropped. This does not occur for $Z-f_0>0$ and is not a defect for Si or diamond.

The repository test file passes (24 tests).

## Adjudication

Re-derivation matches in factor, sign, units, phase convention, thermal exponent, extinction and period logic, and numerical well depth. The 21.0 eV depth versus the ~22 eV literature figure is a documented parameter difference, not a formula difference. Verdict: `rederived`.
