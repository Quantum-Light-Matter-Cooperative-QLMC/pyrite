# `cr-axial-helicity-emission` — fresh-context re-derivation

- **Claim:** `cr-axial-helicity-emission` (ledger: `docs/validation/ledger-channeling.md`)
- **Code:** `src/pyrite/montecarlo/channeling/axial_emission.py::{axial_line_catalog,AxialLine.polarization,axis_stokes,helicity_pattern,photon_angular_momentum,LevelDensity}` at `6e712cdc`
- **Verifier:** fresh context, 2026-10-16. The derivation below was worked out from the ledger row, the module/function docstrings and the dependency rows (`cr-dipole-rate`, `cr-boosted-dipole-angular`) before the function bodies were read. The docstrings already state the target formulas, so this derivation is independent of the code but not of the claimed formulas. The page itself was written after the bodies had been read.
- **Verdict:** `rederived`. No sign, factor or convention discrepancy. There are wording findings on the length/velocity agreement figure and on the statistics of the TDS figure.

## Conventions

- Quantum states evolve as $e^{-iEt/\hbar}$. Photon fields are written as the positive-frequency part $\propto e^{i(\mathbf k\cdot\mathbf r-\omega t)}$.
- Spherical unit vectors are $\hat{\mathbf e}_\pm=(\hat{\mathbf x}\pm i\hat{\mathbf y})/\sqrt2$. A plane wave along $+z$ with polarization $\hat{\mathbf e}_+$ has positive helicity (photon spin $+\hbar$ along $\mathbf k$). This is Jackson §7.2's "left circularly polarized" wave.
- The Stokes parameters follow Jackson (7.27): $\mathbf E=E_1\hat{\boldsymbol\epsilon}_1+E_2\hat{\boldsymbol\epsilon}_2$, with $Q=\lvert E_1\rvert^2-\lvert E_2\rvert^2$, $U=2\,\mathrm{Re}(E_1^*E_2)$ and $V=2\,\mathrm{Im}(E_1^*E_2)$. For $\hat{\mathbf e}_+$ ($E_2=iE_1$), $V=+2\lvert E_1\rvert^2>0$, so $V>0$ means positive helicity.

## 1. Selection rule and handedness

The dipole emission amplitude into polarization $\hat{\boldsymbol\epsilon}$ is $\propto\hat{\boldsymbol\epsilon}^*\cdot\mathbf d$, with $\mathbf d=\langle f\rvert\mathbf r_\perp\lvert i\rangle$. On axis this gives

$$
A_\pm=\hat{\mathbf e}_\pm^*\cdot\mathbf d=\frac{d_x\mp i d_y}{\sqrt2}=\frac{\langle f\rvert\rho e^{\mp i\varphi}\lvert i\rangle}{\sqrt2}.
$$

The operator $\rho e^{-i\varphi}$ lowers $\ell$ by one. So $A_+\neq0$ requires $\ell_f=\ell_i-1$, which gives the photon $J_z=\ell_i-\ell_f=+1$. The boost along $z$ commutes with $J_z$, so $J_z$ is the same in the lab.

The classical picture agrees. In a superposition, $\langle\mathbf r\rangle(t)\ni c_f^*c_i\,\mathbf d\,e^{-i\omega t}+{\rm c.c.}$ with $\omega=(E_i-E_f)/\hbar>0$. The field's positive-frequency part is $\propto\mathbf d_\perp$, so on axis $E_1=d_x$ and $E_2=d_y$, and

$$
V=2\,\mathrm{Im}(d_x^*d_y)=\lvert A_+\rvert^2-\lvert A_-\rvert^2 .
$$

The algebra is $\lvert a-ib\rvert^2-\lvert a+ib\rvert^2=4\,\mathrm{Im}(a^*b)$.

**Numerical check.** For a 2D oscillator transition $\ell=+1\to0$, a grid integral gives $\mathbf d=(0.5,\,0.5i)$. The dipole $\mathrm{Re}[\mathbf d e^{-i\omega t}]$ rotates counter-clockwise about $+z$. Jackson's $V/I=+1$, `axis_stokes` gives $V/I=+1$, and $\lvert\hat{\mathbf e}_+^*\cdot\mathbf d\rvert^2/\lvert\mathbf d\rvert^2=1$.

## 2. Polarization matrix

Take a final level $f$ with degenerate members and an initial level density $\rho=\sum\rho_{ij}\lvert i\rangle\langle j\rvert$. Trace the joint photon–electron state over $f$:

$$
T_{ab}=\sum_f\sum_{ij}A_a[f,i]\,\rho_{ij}\,A_b[f,j]^*=\sum_f\left(A_a\rho A_b^\dagger\right)_{ff}.
$$

$T$ is linear in $\rho$ and includes the coherences $\rho_{ij}$ within the degenerate initial level. Coherences between different levels radiate at other frequencies and belong to other lines, as in the cluster rule of `cr-dipole-rate`. The Cartesian form is $C_{ab}=\sum d_a d_b^*$. On axis $Q=C_{xx}-C_{yy}$, $U=2\,\mathrm{Re}\,C_{yx}$ and $V=2\,\mathrm{Im}\,C_{yx}$.

**Numerical check.** For random complex $d$ (three final states, two initial states) and random positive $\rho$, `axis_stokes(polarization)` equals the Jackson Stokes parameters to $3\times10^{-14}$. Linearity holds to $1\times10^{-14}$.

## 3. Velocity-form dipole

For $H=\hbar^2k^2/(2\gamma m)+U(\mathbf r_\perp)$ we have $[H,\mathbf r]=-i\hbar^2\mathbf k/(\gamma m)$. Then

$$
(E_f-E_i)\langle f\rvert\mathbf r\lvert i\rangle=-\frac{i\hbar^2}{\gamma m}\langle f\rvert\mathbf k\lvert i\rangle
\quad\Rightarrow\quad
\langle f\rvert\mathbf r\lvert i\rangle=-\frac{i\hbar^2}{\gamma m}\frac{\langle f\rvert\mathbf k\lvert i\rangle}{E_f-E_i},
$$

with $\hbar^2/(\gamma m)=2\,\hbar^2/(2m)/\gamma$.

For $k=0$ Bloch functions $\sum_{\mathbf G}c_{\mathbf G}e^{i\mathbf G\cdot\mathbf r}$, the matrix element is $\langle f\rvert\mathbf k\lvert i\rangle=\sum c^{f*}_{\mathbf G}\mathbf G\,c^i_{\mathbf G}$, with $\mathbf G=2\pi(m/L_x,n/L_y)$.

The velocity form is the well-defined interband position element for same-$k$ Bloch states, where $\mathbf r$ itself is not periodic (Blount). For flat (tight-binding) bands it equals the on-site orbital dipole.

**Code.** `dipole = -1j*(2*HBAR2_2M_EV_ANG2/gamma)*momentum/(-delta)` with `delta` $=E_i-E_f$. The sign, the factor and the $2\pi$ in `g_inv_ang` are all correct.

**Numerical check (W ⟨001⟩, 300 keV).** I compared the length form on the cut orbitals with the velocity form:

| grid | max element difference / max $\lvert d\rvert$ | $\lvert d\rvert^2$ velocity | $\lvert d\rvert^2$ length | cut-orbital overlap error |
| --- | --- | --- | --- | --- |
| $3\times3$ cells, $60^2$ | $1.9\times10^{-3}$ | 0.014295 | 0.014348 | $3\times10^{-4}$ |
| $3\times3$ cells, $120^2$ | $2.6\times10^{-3}$ | 0.014295 | 0.014369 | $1\times10^{-5}$ |

The forms agree to about 0.2–0.3 % per element (about 0.5 % in $\lvert d\rvert^2$). The ledger's "0.1%" is optimistic; the test tolerance of `rtol=5e-3` is the honest figure. The residual is the inter-cell tail that the cut omits.

## 4. Helicity patterns, aberration and invariance

In the rest frame, a $J=1$, $J_z=m$ source emits a helicity-$\Lambda$ photon toward $(\theta',\varphi')$ with amplitude $\propto D^{1*}_{m\Lambda}(\varphi',\theta',0)$ (Jacob–Wick). For unit $\lvert\mathbf d\rvert^2$ the density is

$$
\frac{dN_{m\Lambda}}{d\Omega'}=\frac{3}{8\pi}\lvert d^1_{m\Lambda}(\theta')\rvert^2,\qquad
\lvert d^1_{m\Lambda}\rvert=\tfrac12(1+m\Lambda\cos\theta').
$$

Summed over $\Lambda$ this is $\tfrac{3}{16\pi}(1+\cos^2\theta')$, which is normalized. A planar $x$-dipole has $T_{++}=T_{--}$, and its azimuth average $\tfrac{3}{8\pi}(1-\tfrac12\sin^2\theta')$ is the same function. The $m=\pm1$ cross term goes as $e^{\pm2i\varphi}$ and drops out of any azimuth-integrated quantity.

The lab mapping is

$$
\cos\theta'=\frac{\cos\theta-\beta}{1-\beta\cos\theta},\qquad
\frac{d\Omega'}{d\Omega}=\frac{1}{\gamma^2(1-\beta\cos\theta)^2}.
$$

Photon number is a count and is therefore invariant. Helicity of a massless particle is boost invariant, because the little group acts on it as a phase.

**Numerical check.** I built the pattern independently as $\tfrac{3}{8\pi}\lvert\hat{\mathbf e}_\Lambda(\hat{\mathbf n})^*\cdot\hat{\mathbf e}_m\rvert^2$, with $\hat{\mathbf e}_\Lambda=(\hat{\boldsymbol\theta}+i\Lambda\hat{\boldsymbol\varphi})/\sqrt2$, plus my own aberration step. It equals `helicity_pattern` to $2\times10^{-12}$ (relative) at $\gamma=1.0001$, $1.587$ and $20$, for all $m,\Lambda$ and 37 angles.

## 5. Twisted-photon split

Use the Knyazev & Serbo (2018) decomposition of a twisted photon with cone angle $\theta$, $J_z=m$ and helicity $\Lambda$:

$$
\hat{\mathbf e}_{\mathbf k\Lambda}=\sum_\sigma d^1_{\sigma\Lambda}(\theta)\,e^{-i\sigma\varphi_k}\boldsymbol\chi_\sigma .
$$

In this expansion the spin projection $\sigma$ carries orbital $m-\sigma$, with probability $\lvert d^1_{\sigma\Lambda}(\theta)\rvert^2$. That gives

$$
S_z=\Lambda\Big[\big(\tfrac{1+\cos\theta}{2}\big)^2-\big(\tfrac{1-\cos\theta}{2}\big)^2\Big]=\Lambda\cos\theta,\qquad L_z=m-\Lambda\cos\theta .
$$

This agrees with the Bliokh et al. (PRA 82, 063825, 2010) nonparaxial Bessel-beam result $\langle L_z\rangle=\ell+\sigma(1-\cos\theta_0)$ with $j_z=\ell+\sigma$. The transverse-plane integral makes it independent of the spin/orbital density definition.

**Numerical check.** Projecting $\hat{\mathbf e}_{\mathbf k\Lambda}$ onto $\boldsymbol\chi_\sigma$ gives $\sum_\sigma\sigma P_\sigma=\Lambda\cos\theta$ to $10^{-12}$ at four angles and both helicities.

**Lab angle.** Using the lab angle is correct. $J_z$ and $\Lambda$ are boost invariant but $\hat{\mathbf k}$ is not, so $S_z=\Lambda\cos\theta_{\rm lab}$ and $L_z$ is frame dependent. The split is defined per azimuth-integrated cone (annular acceptance). A point detector at fixed $\varphi$ does not select a $J_z$ eigenstate.

## 6. `photon_angular_momentum`

The lab cone $\theta\le\theta_{\max}$ maps to $\cos\theta'\ge(\cos\theta_{\max}-\beta)/(1-\beta\cos\theta_{\max})$, and $\theta_{\max}=\pi$ maps to $-1$. The azimuth integral turns $\tfrac{3}{8\pi}$ into $\tfrac34\,d\cos\theta'$. Only $T_{++}$ and $T_{--}$ enter, by item 4.

**Numerical check.** I ran an independent lab-angle adaptive quadrature (`scipy.integrate.quad` over $\theta$ with the oracle pattern). All five moments agree to $10^{-6}$ in every case:

| $\gamma$ | $\theta_{\max}$ | $(T_{++},T_{--})$ | fraction | $\langle\Lambda\rangle$ | $J_z$ | $S_z$ | $L_z$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5871 | $1/\gamma$ | (1, 0) | 0.46827 | 0.79511 | 1 | 0.75260 | 0.24740 |
| 1.5871 | $\pi$ | (1, 0) | 1 | 0 | 1 | 0.33071 | 0.66929 |
| 1.5871 | 0.3 | (0.7, 0.3) | 0.19872 | 0.39270 | 0.4 | 0.38488 | 0.01512 |
| 20 | $1/\gamma$ | (1, 0) | 0.49984 | 0.75024 | 1 | 0.75002 | 0.24998 |
| 20 | 0.2 | (0.2, 0.8) | 0.91715 | −0.05407 | −0.6 | −0.05530 | −0.54470 |
| 1 | $\pi/2$ | (1, 0) | 0.5 | 0.75 | 1 | 0.5 | 0.5 |

The Notes' "0.79 / 0.75 / 0.25" matches the 300 keV row.

## 7. Flat-band gate

For W ⟨001⟩ at 300 keV, a $5\times5$ scan over the irreducible zone reproduces the Γ/X/M spread used in the code: 0.0625 eV for the 2p pair at `max_order=10`. The 1s "bandwidth" is plane-wave truncation error:

| `max_order` | 10 | 12 | 14 | 16 |
| --- | --- | --- | --- | --- |
| 1s spread (eV) | 0.047 | 0.012 | 0.003 | 0.0007 |
| 2p spread (eV) | 0.0625 | 0.067 | 0.068 | 0.069 |

The 2p value is converged; $\lvert d\rvert^2$ is stable to $10^{-4}$.

A bandwidth $W$ means inter-string hopping over a path $\hbar\beta c/W$. For 2p that is 2.5 µm, while the TDS lifetime is about 10 nm. The tight-binding and cut-orbital picture is therefore fair here.

As a general criterion, the fixed 0.1 eV gate ($\approx1.5$ µm hop path at $\beta\approx0.78$) is reasonable for sub-µm crystals or decoherence-limited levels. It is not a dimensionless test. A more principled gate would be $W\ll\hbar v/\min(\ell_{\rm deph},t)$ together with $W\ll$ the level spacing. This is a note, not a discrepancy.

Two minor code assumptions:

- The gate samples only Γ, X and M (no Y point). That is fine for a square cell; a rectangular cell would need Y.
- The `break` assumes bandwidth grows monotonically with the level index.

## 8. Yield

$\mathrm{tr}\,T=\sum_f(d_x\rho d_x^\dagger+d_y\rho d_y^\dagger)_{ff}$. For a pure state this is $\lvert d_x\rvert^2+\lvert d_y\rvert^2$, the vector form of $\lvert x_{fi}\rvert^2$ in `cr-dipole-rate`. `yield_per_ang` passes $\sqrt{\mathrm{tr}\,T}$, which is then squared.

**Numerical check.** `dipole_rate_per_ang` equals $\tfrac43\alpha\gamma^2(\Delta E/\hbar c)^3\lvert d\rvert^2/\beta$ to $6\times10^{-10}$ (CODATA constant rounding).

## 9. Acceptance wording

Mirror symmetry of the string gives $\lvert A_+(\ell{=}{+}1)\rvert=\lvert A_-(\ell{=}{-}1)\rvert$. So for 2p → 1s the following are all exactly linear in the $\ell$ imbalance $p_+-p_-$:

- on-axis $V/I$;
- per-photon $J_z$;
- the cone moments $\langle\Lambda\rangle$, $S_z$ and $L_z$ (for example $0.795$, $0.753$ and $0.247$ times the imbalance at $\theta\le1/\gamma$, 300 keV).

Over the full sphere, $\langle\Lambda\rangle=0$ for any $\rho$. "Net circular polarization scales with the ℓ imbalance" is true on axis or in a forward aperture and false for $4\pi$ collection. $J_z$ and $L_z$ scale at any aperture.

The MCWF check ($V/I$ equal to the surviving imbalance per 2p electron) follows from linearity and the symmetry above, applied to the same $\rho$. It confirms the $L_z$ basis and the symmetry, not any TDS physics.

The acceptance is honestly met if the wording keeps the aperture qualifier.

## W TDS numbers in the Notes

I ran $5\times5$ cells on an $80^2$ grid with the TDS mechanism only and an LG $\ell=1$ entry beam of waist 0.35 Å. There were eleven independent 32-electron batches.

- **2p population.** It is 0.94 at entry, falls to 0.28–0.51 at 10 nm and reaches $0.013\pm0.003$ (mean ± s.e.) at 50 nm. This agrees with the Notes' 0.016.
- **$V/I$ at 50 nm.** The per-batch values are 0.89, 0.64, −0.24, 0.79, 0.83, 0.76, 0.91, 0.74, 0.57, 0.90 and −0.31. The population-weighted value is 0.76.

The Notes' "1.00 → 0.80" is plausible. At 50 nm, though, it rests on about 2–3 electrons' worth of 2p weight even with 192 electrons, so its statistical uncertainty is about ±0.1 and it should be quoted with an error bar.

## Tests

`OPENBLAS_NUM_THREADS=1 uv run pyrite-dev test --slow tests/montecarlo/test_channeling_axial_emission.py`: 8 passed. The slow MCWF test takes 1.5 s.
