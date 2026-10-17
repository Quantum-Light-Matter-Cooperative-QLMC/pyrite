# `cr-axial-2d-eigenproblem` — axial string potential and 2D transverse Bloch states

Verifier: fresh context (2026-10-16, worktree `issue-508-axial-tier` at `797a0159`). Ledger: `docs/validation/ledger-channeling.md`. Code: `src/pyrite/materials/channeling/axial.py::{axial_profile,AxialProfile.potential,AxialProfile.on_grid,axial_states}`.

## Claim

Averaging the crystal potential along a lattice axis $[uvw]$ keeps the zone reflections $\mathbf G\cdot[uvw]=0$:

$$
U(\mathbf r_\perp)=\sum_{\mathbf g\ne0}U_{\mathbf g}e^{i\mathbf g\cdot\mathbf r_\perp},\qquad
U_{\mathbf g}=-\frac{4\pi e^2}{Vg^2}\sum_j\bigl(Z_j-f_{0,j}(g)\bigr)e^{-i\mathbf G\cdot\mathbf R_j}e^{-\mathbf g\cdot\mathsf U_j\mathbf g/2},
$$

with $\mathsf U_j$ the site mean-square displacement tensor (isotropic default $B/8\pi^2$). $U$ is represented on the smallest rectangular cell whose reciprocal grid holds every non-extinct zone vector ($x$ along a short zone vector, $y=z\times x$). The $y$ average is the planar profile of the plane normal to $x$. Transverse states are plane-wave eigenstates of $-(\hbar^2/2\gamma m)\nabla_\perp^2+U$ at Bloch $\mathbf k$. Acceptance is peak ordering only.

## Independent derivation (before reading the body)

**Fourier coefficient.** Poisson's equation $\nabla^2\phi=-4\pi\rho$ with $\rho(\mathbf r)=e\sum_{\mathbf R}\sum_j[Z_j\delta(\mathbf r-\mathbf R-\mathbf R_j)-n_j(\mathbf r-\mathbf R-\mathbf R_j)]$ gives $\phi_{\mathbf G}=4\pi\rho_{\mathbf G}/G^2$, with $\rho_{\mathbf G}=V^{-1}\int_V\rho\,e^{-i\mathbf G\cdot\mathbf r}d^3r=(e/V)\sum_j(Z_j-f_{0,j})e^{-i\mathbf G\cdot\mathbf R_j}$. The electron potential energy is $U=-e\phi$, so

$$
U_{\mathbf G}=-\frac{4\pi e^2}{VG^2}\sum_j(Z_j-f_{0,j}(G))e^{-i\mathbf G\cdot\mathbf R_j},
$$

the Mott–Bethe form of `cr-mott-bethe-factor`. Units: $e^2/(VG^2)$ is eV·Å/(Å³·Å⁻²) = eV.

**Thermal average.** A Gaussian displacement $\mathbf u$ with covariance $\mathsf U_j$ gives $\langle e^{-i\mathbf G\cdot\mathbf u}\rangle=e^{-\mathbf G^{\mathsf T}\mathsf U_j\mathbf G/2}$ (Willis & Pryor). The isotropic case $\mathsf U_j=u^2\mathsf I$ with $u^2=B/8\pi^2$ and $s=G/4\pi$ gives $e^{-G^2B/16\pi^2}=e^{-Bs^2}$, the planar factor. A displacement along the axis enters only through $\mathbf G\cdot\hat{\mathbf z}=0$, so it drops out.

**Axial average.** Translate along the lattice vector $\mathbf t=a(u,v,w)$. Then $\mathbf G\cdot\mathbf t=2\pi(hu+kv+lw)$ and $T^{-1}\int_0^Te^{i\mathbf G\cdot\hat{\mathbf z}z}dz=\delta_{hu+kv+lw,0}$. Only zone vectors survive. Each one lies in the transverse plane, $\mathbf g=\mathbf G$.

**Sign and phase.** $Z_j-f_{0,j}>0$ for $g>0$, so every term is negative at the string, where $e^{i\mathbf g\cdot\mathbf R_j}e^{-i\mathbf G\cdot\mathbf R_j}=1$. The wells therefore sit on the strings. $U_{-\mathbf g}=U_{\mathbf g}^*$ makes $U$ real. $U_0=0$ by convention, as in the planar claim.

**Rectangular cells (diamond, conventional $a$; non-extinct reflections are all odd, or all even with $h+k+l\equiv0\pmod4$).**

| axis | surviving zone set | rectangular cell $L_x\times L_y$ | strings per cell | $\lambda$ per string | $\sum\lambda/A$ |
| --- | --- | --- | --- | --- | --- |
| $\langle001\rangle$ | $(h,k,0)$ with $h,k$ even and $h+k\equiv0\ (4)$: a square lattice of $\{220\}$ | $a/\sqrt8\times a/\sqrt8$ | 1 | $1/a$ | $8/a^3$ |
| $\langle110\rangle$ | $(h,\bar h,l)$, either both odd or $h$ even with $l\equiv0\ (4)$; in units $(\sqrt2,1)\,2\pi/a$: $(m,n)$ both odd, or $m$ even with $n\equiv0\ (4)$ | $a/\sqrt2\times a$ | 4 (two dumbbells, $a/4$ apart along $[001]$) | $\sqrt2/a$ | $8/a^3$ |
| $\langle111\rangle$ | all even with $h+k+l=0$: hexagonal $\{220\}$, centred-rectangular grid $(g_0/2,\sqrt3g_0/2)$ | $a/\sqrt2\times a/\sqrt6$ | 2 (both basis atoms lie on one string, since $(\tfrac14,\tfrac14,\tfrac14)\parallel[111]$) | $2/(\sqrt3a)$ | $8/a^3$ |

For $\langle110\rangle$ the shortest zone vector, $\{111\}$, is not orthogonal to a lattice vector that would give a smaller rectangle. The $[1\bar10]\times[001]$ frame has area $a^2/\sqrt2$, twice the projected fcc primitive area $a^2\sqrt2/4$. It is minimal.

**Planar limit.** Averaging over $y$ keeps $\mathbf g\parallel\hat{\mathbf x}$. Combined with the axial average, this is the average over the plane spanned by $\hat{\mathbf y},\hat{\mathbf z}$, the planar continuum potential of the family with normal $\hat{\mathbf x}$. For Si $\langle001\rangle$ with $x\parallel[110]$, $U_{(1,0)}$ must equal the planar $U_{220}=-4.3158$ eV (`cr-planar-continuum-potential` write-up).

**Eigenproblem.** With $\psi_{\mathbf k}=A^{-1/2}\sum_{\mathbf G}c_{\mathbf G}e^{i(\mathbf k+\mathbf G)\cdot\mathbf r}$, $\int_A|\psi|^2=\sum|c_{\mathbf G}|^2=1$, and

$$
H_{\mathbf G\mathbf G'}=\frac{\hbar^2|\mathbf k+\mathbf G|^2}{2\gamma m}\delta_{\mathbf G\mathbf G'}+U_{\mathbf G-\mathbf G'},
$$

which is Hermitian because $U_{-\mathbf g}=U_{\mathbf g}^*$. Here $\hbar^2/2m=3.80998$ eV·Å².

## Independent numerics

The scratch script uses `xraydb.f0` (not the implementation's Cromer–Mann table) and an explicit 8-atom basis. It enumerates the 3D reciprocal lattice, keeps the zone with $s\le6$ Å⁻¹, and sums $U(\mathbf r)$ directly in its own orthonormal frame. Only the catalog $a$ and $B$ (separate claims) are shared. Si: $a=5.4309$ Å, $B=0.48166$ Å². C: $a=3.5668$ Å, $B=0.14385$ Å². Temperature 293.15 K.

| axis | $U$ at string [eV] indep. / impl. | $\max U$ [eV] indep. / impl. | depth [eV] indep. / impl. | Molière string depth [eV] | Lindhard $\lvert U(\rho\approx\sqrt2u_1)\rvert$ [eV] |
| --- | --- | --- | --- | --- | --- |
| Si $\langle001\rangle$ | $-77.219$ / $-77.219$ | $8.186$ / $8.186$ | 85.40 / 85.40 | 89.8 | 86.5 |
| Si $\langle110\rangle$ | $-115.302$ / $-115.302$ | $11.973$ / $11.973$ | 127.27 / 127.27 | 135.6 | 122.4 |
| Si $\langle111\rangle$ | $-90.689$ / $-90.689$ | $7.696$ / $7.696$ | 98.38 / 98.38 | 103.5 | 99.9 |
| C $\langle110\rangle$ | $-117.777$ / $-117.777$ | $16.692$ / $16.692$ | 134.47 / 134.47 | 133.8 | 137.7 |

The independent minimum is at the string in every case, and $\max\lvert\mathrm{Im}\,U\rvert\le5\times10^{-14}$ eV. The Molière column replaces $(Z-f_0)/g^2$ by $Z\sum_i\alpha_i/(g^2+\kappa_i^2)$ with Thomas–Fermi $a_{\rm TF}$. The Lindhard column is $Ze^2\lambda\ln(3a_{\rm TF}^2/2u_1^2+1)$. Both are model estimates; they bracket the Hartree–Fock-like values within ±7%. The ledger depths (85, 127, 98, 134 eV) are physically plausible continuum-string depths.

Implementation cells match the table: Si $\langle001\rangle$ 1.9201 Å square with 1 string, $\lambda a=1$. Si $\langle110\rangle$ 3.8402 × 5.4309 Å with 4 strings, the dumbbell partner at $y=1.3577$ Å $=a/4$ and $\lambda a=\sqrt2$. Si $\langle111\rangle$ 3.8402 × 2.2172 Å with 2 strings, $\lambda a=2/\sqrt3$. Diamond $\langle110\rangle$ 2.5221 × 3.5668 Å with 4 strings. Every axis gives $\sum\lambda/A\cdot a^3=8.000000$. High-index probes ([135], [237], [149]) also find rectangular cells, with density sums of 8.

Si $\langle001\rangle$, $\gamma=1+10/0.51099895$, $M=8$ ($17^2$ plane waves), against an independent Hamiltonian built from the formula above:

| $\mathbf k$ | lowest six $E$ [eV], independent | implementation |
| --- | --- | --- |
| $(0,0)$ | $-43.2714,\ -17.9937,\ -17.9937,\ -6.1836,\ -1.7112,\ -1.6962$ | identical to 4 d.p. |
| $(0.3g_x,0.1g_y)$ | $-43.2702,\ -17.9932,\ -17.9914,\ -6.1811,\ -1.7058,\ -1.6971$ | identical to 4 d.p. |

$U_{(1,0)},U_{(1,1)},U_{(2,0)}$: independent $-4.315893,\ -2.563743,\ -1.473438$ eV; implementation $-4.315893,\ -2.563743,\ -1.473438$ eV. The degenerate 2p pair at $\mathbf k=0$ is reproduced. The $(1,0)$ coefficient equals the planar $U_{220}$.

## Comparison with the implementation

- `axial_profile`: `phases = exp(-2j π hkl·R)` and `debye = exp(-0.5 g·U_j·g)` with Cartesian `g_vec` and tensors; `potential = -4π E2/V Σ (Z - f0)·phases·debye / g²`. This matches the derivation term by term. The isotropic default is `B/(8π²)·I`.
- Zone filter `hkl @ uvw == 0`, cut at $g\le4\pi s_{\max}$. The extinction test is numeric (`_EXTINCTION_RTOL`), the same convention as the planar code.
- `_rectangular_cell` tries short zone directions as $\hat{\mathbf x}$, sets $\hat{\mathbf y}=\hat{\mathbf z}\times\hat{\mathbf x}$, and requires both projections of every live $\mathbf g$ on a 1D lattice. It keeps the smallest area. `_grid_step` searches divisors 1–6 of the smallest projection. This limit is a heuristic that could, in principle, reject a valid high-index axis as "oblique", but it did not for the axes probed.
- `_synthesize`: $\sum_{m,n}c_{mn}e^{i(g_xmx+g_yny)}$, real part. `_on_grid`: places $c_{mn}$ at FFT bin $(mc_x,nc_y)$ and drops $2\lvert mc_x\rvert\ge n_x$ (both $\pm$ Nyquist, so the result stays real). `ifft2·(n_xn_y)` carries the $e^{+i}$ sign, consistent with $e^{i\mathbf g\cdot\mathbf r}$.
- `axial_states`: $H_{ij}=U[m_i-m_j,n_i-n_j]+\hbar^2\lvert\mathbf k+\mathbf G_i\rvert^2/(2\gamma m)$, with `HBAR2_2M_EV_ANG2 = (ħc)²/(2 m_e c²)`. `eigh` gives orthonormal columns, and `wavefunction` divides by $\sqrt A$ for normalization per cell. This matches.
- `_strings`: the centring images of the projected lattice times the basis, folded and merged, with weight $A/(V N_{\rm centring})$. The string densities sum to $N_{\rm basis}/V$.

Tests: `uv run pyrite-dev test --slow tests/montecarlo/test_channeling_axial.py` gives 17 passed (155 s), including the five anchors listed in the row.

## Peak-order acceptance

Issue #508 acceptance reads "`cr-axial-2d-eigenproblem` reaches `anchored` (peak ordering only; experimental tier)", and the checklist item "Peak-order anchors: Watson axial Si lines; Gouanère" is open. The existing Watson/Gouanère tests (`tests/materials/test_channeling_profile.py`, `tests/montecarlo/test_channeling_emission.py::test_diamond_peaks_match_gouanere`) are planar. No axial peak-order anchor exists yet. The five listed anchors are internal consistency checks (planar limit, cell, symmetry, norm), not reference values, so the claim is not yet `anchored` in the acceptance sense.

## Adjudication

The re-derivation matches in factor, sign, phase convention ($e^{-i\mathbf G\cdot\mathbf R_j}$, wells on strings), anisotropic Debye–Waller exponent, zone selection, rectangular cell and string bookkeeping, planar limit, and Bloch Hamiltonian and normalization. Numerical agreement is at the meV level for the potential and to 4 d.p. for the eigenvalues. The depths are plausible against Molière/Lindhard continuum-string estimates. Verdict: `rederived`. The `anchored` status awaits the axial peak-order anchor.
