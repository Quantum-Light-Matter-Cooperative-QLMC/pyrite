# `cr-dipole-far-field-amplitude` re-derivation

Claim: `montecarlo/channeling/dipole.py::dipole_far_field` and `apps/channeling.py::far_field_maps`. Sources: Jackson, *Classical Electrodynamics* 3rd ed., the Liénard–Wiechert acceleration field (cited as Eq. (14.14); see the citation note below) and the aberration of `cr-boosted-dipole-angular`; Knyazev & Serbo, Phys. Usp. 61, 449 (2018) for $J_z=\ell+\sigma$; `cr-axial-helicity-emission` for $A_\pm$ and $T_{ab}$.

## Independent derivation (from the ledger row and docstrings)

**Field.** The emitting electron moves along $\hat{\mathbf z}$ with $\beta$ and carries a small transverse oscillation $\mathbf r_\perp(t)=\operatorname{Re}[\mathbf d\,e^{-i\omega t}]$, $\mathbf d=(d_x,d_y,0)$ complex. The Liénard–Wiechert radiation (acceleration) field is

$$
\mathbf E=\frac{e}{c}\left[\frac{\hat{\mathbf n}\times\{(\hat{\mathbf n}-\boldsymbol\beta)\times\dot{\boldsymbol\beta}\}}{\kappa^3R}\right]_{\rm ret},\qquad \kappa=1-\hat{\mathbf n}\cdot\boldsymbol\beta .
$$

To first order in $\mathbf d$ the only non-zero term is the one with $\dot{\boldsymbol\beta}\propto\ddot{\mathbf r}_\perp\propto-\mathbf d$. The changes of $\kappa$ and of the retarded position caused by $\mathbf r_\perp$ multiply the zeroth-order radiation field, which is zero, and the velocity field does not radiate. Then $\boldsymbol\beta\to\beta\hat{\mathbf z}$ and $\kappa\to D=1-\beta\cos\theta$. The triple product expands as

$$
\hat{\mathbf n}\times\{(\hat{\mathbf n}-\beta\hat{\mathbf z})\times\mathbf d\}
=(\hat{\mathbf n}\cdot\mathbf d)(\hat{\mathbf n}-\beta\hat{\mathbf z})-\bigl(1-\beta\cos\theta\bigr)\mathbf d .
$$

The map $\mathbf d\mapsto\mathbf V(\mathbf d)\equiv(\hat{\mathbf n}\cdot\mathbf d)(\hat{\mathbf n}-\beta\hat{\mathbf z})-D\mathbf d$ is real-linear with real coefficients. So the complex amplitude of $\operatorname{Re}[\mathbf E\,e^{-i\omega t}]$ is $\mathbf V(\mathbf d)$ with the same $e^{-i\omega t}$ convention and no conjugation. The overall sign (from $\ddot{\mathbf r}=-\omega^2\mathbf r$) and the frequency factors are global, and the normalization fixes them. $\hat{\mathbf n}\cdot\mathbf V=0$ follows from $\hat{\mathbf n}\cdot(\hat{\mathbf n}-\beta\hat{\mathbf z})=D$.

**Normalization.** The angular power per unit retarded time is $\propto\lvert\mathbf V\rvert^2/D^5$. Dividing by the lab photon energy $\hbar\omega'/(\gamma D)$ gives

$$
\frac{dN}{d\Omega}\propto\frac{\lvert\mathbf V\rvert^2}{D^4}.
$$

Since $\mathbf d\perp\hat{\mathbf z}$ gives $(\hat{\mathbf n}-\beta\hat{\mathbf z})\cdot\mathbf d^*=\hat{\mathbf n}\cdot\mathbf d^*$ and $\lvert\hat{\mathbf n}-\beta\hat{\mathbf z}\rvert^2=1-2\beta\cos\theta+\beta^2$,

$$
\lvert\mathbf V\rvert^2=\lvert\hat{\mathbf n}\cdot\mathbf d\rvert^2\bigl(1-2\beta\cos\theta+\beta^2-2D\bigr)+D^2\lvert\mathbf d\rvert^2
=D^2\lvert\mathbf d\rvert^2-\frac{\lvert\hat{\mathbf n}\cdot\mathbf d\rvert^2}{\gamma^2}.
$$

Aberration gives $\hat{\mathbf n}'_\perp=\hat{\mathbf n}_\perp/(\gamma D)$, so $\lvert\hat{\mathbf n}\cdot\mathbf d\rvert^2=\gamma^2D^2\lvert\hat{\mathbf n}'\cdot\mathbf d\rvert^2$. With the scale $\sqrt{3/8\pi}/(\gamma D^2\lvert\mathbf d\rvert)$,

$$
\boxed{\sum_{s=\pm,z}\lvert E_s\rvert^2=\frac{3}{8\pi}\,\frac{\lvert\mathbf V\rvert^2}{\gamma^2D^4\lvert\mathbf d\rvert^2}
=\frac{3}{8\pi}\,\frac{1-\lvert\hat{\mathbf n}'\cdot\hat{\mathbf d}\rvert^2}{\gamma^2D^2}}
$$

This is the rest-frame pattern of a (possibly complex) dipole, $(3/8\pi)(1-\lvert\hat{\mathbf n}'\cdot\hat{\mathbf d}\rvert^2)$, times the Jacobian $d\Omega'/d\Omega=1/(\gamma^2D^2)$, so it integrates to 1 for every $\mathbf d$. For $\mathbf d=\hat{\mathbf x}$ it is $(3/8\pi)(1-n_x'^2)/(\gamma^2D^2)$, which is `boosted_dipole_pattern`. Because the basis change is unitary, the sum over $(E_+,E_-,E_z)$ equals $\lvert\mathbf V\rvert^2$. For real $\mathbf d$, $\mathbf V$ is real, so $E_-=E_+^*$.

**Circular components.** With $\mathbf e_\pm=(\hat{\mathbf x}\pm i\hat{\mathbf y})/\sqrt2$ and $E_\pm=\mathbf e_\pm^*\cdot\mathbf E$, $\operatorname{Re}[(\hat{\mathbf x}+i\hat{\mathbf y})e^{-i\omega t}]=\hat{\mathbf x}\cos\omega t+\hat{\mathbf y}\sin\omega t$ rotates counter-clockwise about $+\hat{\mathbf z}$, so $\mathbf d=\mathbf e_+$ is a $J_z=+1$ line. This agrees with `axial_emission.py`: $A_+=\mathbf e_+^*\cdot\mathbf d=(d_x-id_y)/\sqrt2=1$ and $A_-=0$. The relevant products are $\mathbf e_\pm^*\cdot\hat{\mathbf z}=0$, $\mathbf e_-^*\cdot\mathbf e_+=0$, $\hat{\mathbf n}\cdot\mathbf e_+=\sin\theta\,e^{i\phi}/\sqrt2$, $\mathbf e_+^*\cdot\hat{\mathbf n}=\sin\theta\,e^{-i\phi}/\sqrt2$ and $\mathbf e_-^*\cdot\hat{\mathbf n}=\sin\theta\,e^{i\phi}/\sqrt2$. For $\mathbf d=\mathbf e_+$ they give, exactly at every $\theta$,

$$
V_+=\frac{\sin^2\theta}{2}-D=-\frac{(\cos\theta-\beta)^2+\gamma^{-2}}{2},\qquad
V_-=\frac{\sin^2\theta}{2}\,e^{2i\phi},\qquad
V_z=\frac{\sin\theta(\cos\theta-\beta)}{\sqrt2}\,e^{i\phi}.
$$

$V_+$ is strictly negative, so its winding is 0 and its phase is constant at $\pi$. $V_-$ winds $+2$ and vanishes on the axis, and $V_z$ winds $+1$ and also vanishes on the axis. This is $\ell=J_z-\sigma$ for $\sigma=+1,-1,0$. A rotation by $\alpha$ about $\hat{\mathbf z}$ multiplies a $J_z$ dipole by $e^{-iJ_z\alpha}$ and $E_\sigma$ by $e^{i\sigma\alpha}$, so the winding $J_z-\sigma$ holds exactly in the fixed basis. The paraxial qualifier is needed only to read $E_\sigma$ as a helicity and $\ell$ as orbital angular momentum. $\mathbf d=\mathbf e_-$ mirrors this, with $E_+$ winding $-2$.

**Paraxial correspondence.** The exact helicity amplitude of `helicity_pattern(m=1, Λ=+1)` is $\propto D\,(1+\cos\theta')/2=(1-\beta)(1+\cos\theta)/2$. The fixed-basis $-V_+$ differs from it by

$$
-V_+-\tfrac12(1-\beta)(1+\cos\theta)=-\tfrac14(1-\beta)\theta^2+\tfrac18\theta^4+O(\theta^6),
$$

which is a relative error $\approx\theta^2(\gamma^2\theta^2-1)/4$ at fixed $\gamma\theta$. That is $O(1/\gamma^2)$ and is small only when $\gamma\gg1$. The $\Lambda=-1$ term behaves the same way. The total including $\lvert E_z\rvert^2$ equals $h_++h_-$ exactly (both equal the boxed pattern averaged over $\phi$, which is independent of $\phi$ for $\mathbf d=\mathbf e_\pm$).

**Incoherent mixture (`far_field_maps`).** In `cr-axial-helicity-emission` the line polarization is $T_{ab}=\sum A_aA_b^*$ in the $(+,-)$ basis. With $\mathbf d=u\mathbf A$, $u=\begin{pmatrix}1&1\\ i&-i\end{pmatrix}/\sqrt2$ (from $A_\pm=(d_x\mp id_y)/\sqrt2$), the Cartesian coherency is $C=uTu^\dagger=\sum\mathbf d\mathbf d^\dagger$. The field is linear in $\mathbf d$, $E_s=\mathbf L_s\cdot\mathbf d$, so the summed intensity is $\sum\lvert\mathbf L_s\cdot\mathbf d\rvert^2=\mathbf L_s C\mathbf L_s^\dagger$. This depends only on $C$, and $C=\sum_k\lambda_k\mathbf v_k\mathbf v_k^\dagger$ ($\lambda_k\ge0$, orthonormal $\mathbf v_k$) is one valid decomposition. Each mode emits a fraction $\lambda_k/\operatorname{tr}C$ of the photons, since the rate is $\propto\lvert\mathbf d\rvert^2$. With `dipole_far_field` normalized per photon, the density per line photon is

$$
\frac{dN_s}{d\Omega}=\sum_k\frac{\lambda_k}{\operatorname{tr}C}\,\lvert E_s(\mathbf v_k)\rvert^2,\qquad
\frac{dN_s}{d(\gamma\theta_x)\,d(\gamma\theta_y)}=\frac{1}{\gamma^2}\frac{dN_s}{d\Omega}.
$$

No single phase exists for a mixed state. The phase of the leading mode $\mathbf v_{\max}$ is exact when the purity $\lambda_{\max}/\operatorname{tr}C$ is 1. Its global offset is the arbitrary gauge of the eigenvector, not the phase of any physical $\mathbf d$; only phase differences and windings are meaningful.

## Independent numerical checks

Scratch oracle using SciPy quadrature and a direct boost of the rest-frame dipole field (no implementation helper except the function under test):

| Check | Result |
| --- | --- |
| Rest-frame $\mathbf E'\propto\omega'^2(\hat{\mathbf n}'\times\mathbf d)\times\hat{\mathbf n}'$, $\mathbf B'=\hat{\mathbf n}'\times\mathbf E'$, boosted ($\mathbf E_\perp=\gamma(\mathbf E'-\boldsymbol\beta\times\mathbf B')_\perp$); 2000 random complex $\mathbf d$, $\gamma\in[1.0001,400]$, $\theta$ over the full sphere and the $\gamma\theta\le3$ cone | `dipole_far_field` $=c\,\mathbf E_{\rm lab}$ with real $c$: residual $5.9\times10^{-11}$, $\max\lvert\operatorname{Im}c/\operatorname{Re}c\rvert=1.9\times10^{-11}$; $\lvert\hat{\mathbf n}\cdot\mathbf E\rvert/\lvert\mathbf E\rvert\le3.8\times10^{-14}$ |
| $\sum\lvert E_s\rvert^2$ vs rest pattern $\times$ Jacobian $1/(\gamma D)^2$, same 2000 points | max relative difference $9.3\times10^{-11}$ |
| Sphere integral, adaptive `dblquad` split at $\gamma\theta=1,5,30$; $\gamma=1,1.6,10.8,100$; $\mathbf d=\hat{\mathbf x}$, $(\hat{\mathbf x}+i\hat{\mathbf y})/\sqrt2$, $(0.3-0.7i,\,1.1+0.2i)$ | $\lvert\int-1\rvert\le4.4\times10^{-16}$ in all 12 cases |
| $\mathbf d=\hat{\mathbf x}$ vs `boosted_dipole_pattern`, 5000 random directions, $\gamma=1.6,10.8,10^3$ | max relative difference $1.4\times10^{-12}$; $E_--E_+^*=0$ exactly |
| $J_z=\pm1$ windings on $\gamma\theta=0.3,1,2.5$, $\gamma=1.6,10.8$ | $(E_+,E_-,E_z)=(0,2,1)$ for $J_z=+1$ and $(-2,0,-1)$ for $J_z=-1$; on axis only $E_{\pm}$ of the line's own sign is non-zero ($0.98431$ at $\gamma=1.6$, $7.44664$ at $\gamma=10.8$) |
| Closed forms $V_\pm,V_z$ above, 1000 random directions | relative error $\le3.6\times10^{-14}$ |
| Fixed basis vs `helicity_pattern`, $\gamma\theta\in[0,1]$ | $\gamma=10.8$: $1.1\times10^{-3}$ ($+$), $4.3\times10^{-3}$ ($-$); $\gamma=100$: $1.2\times10^{-5}$, $5.0\times10^{-5}$; $\gamma=1.6$: $5.7\times10^{-2}$, $0.26$. Sum including $E_z$ equals $h_++h_-$ to $1.3\times10^{-12}$ at every $\gamma$ |
| `far_field_maps` vs explicit $\sum_j\lvert\mathbf d_j\rvert^2\lvert E_s(\mathbf d_j)\rvert^2/(\gamma^2\operatorname{tr}C)$, 4 random complex $\mathbf d_j$, $\gamma=1.6,10.8,50$ | max difference $6.3\times10^{-16}$ of the peak; purity equals $\lambda_{\max}/\operatorname{tr}C$; phase equals that of `eigh`'s last eigenvector |
| Pure $C=\mathbf d\mathbf d^\dagger$ | purity 1; phase map equals that of $\mathbf d$ up to a constant offset (spread $1.9\times10^{-14}$, offset 2.68 rad from the eigenvector gauge) |

Anchor run: `uv run pyrite-dev test tests/montecarlo/test_channeling_axial_emission.py`, 10 passed and 1 slow test skipped.

## Diff against the implementation

`dipole_far_field` builds $\hat{\mathbf n}=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)$ from $\theta=\operatorname{hypot}(\theta_x,\theta_y)$ and $\phi=\operatorname{atan2}(\theta_y,\theta_x)$. It forms $(\hat{\mathbf n}\cdot\mathbf d)(\hat{\mathbf n}-\beta\hat{\mathbf z})-D\mathbf d$ with the cancellation-free $D$ of `cr-doppler-line-energy`, scales it by $\sqrt{3/8\pi}/(\gamma D^2\lVert\mathbf d\rVert)$, and returns $((E_x-iE_y)/\sqrt2,\,(E_x+iE_y)/\sqrt2,\,E_z)=(\mathbf e_+^*\cdot\mathbf E,\mathbf e_-^*\cdot\mathbf E,E_z)$. This is identical to the derivation, including the sign and the lack of conjugation. Note that $(\theta_x,\theta_y)$ are polar-coordinate components ($\theta_x=\theta\cos\phi$), not tangent projections; this is immaterial paraxially and as documented. `gamma` must be a scalar (`float(gamma)`).

`far_field_maps` diagonalizes $C$ with `eigh`, drops modes below $10^{-12}\operatorname{tr}C$, accumulates $(\lambda_k/\operatorname{tr}C)\lvert E_\pm(\mathbf v_k)\rvert^2/\gamma^2$, and takes the phase of `vectors[:, -1]` (the largest eigenvalue, since `eigh` sorts in ascending order). Every step matches the derivation. The `coherency` producer `apps/axial_channeling.py::AxialRadiation.coherency` uses the same $u$ as `axis_stokes`, so it supplies $C=uTu^\dagger=\sum\mathbf d\mathbf d^\dagger$ in the orientation that `eigh` expects. The planar app passes $\operatorname{diag}(1,0)$, which is $\mathbf d=\hat{\mathbf x}$.

## Findings and wording

1. **Citation.** As I recall Jackson 3rd ed. (I could not check the text in this context), Eq. (14.13) is the Liénard–Wiechert $\mathbf E$ (velocity plus acceleration terms) and Eq. (14.14) is $\mathbf B=[\hat{\mathbf n}\times\mathbf E]_{\rm ret}$. The acceleration field used here is the second term of (14.13), and the $\kappa^{-5}$ angular power is Eq. (14.38) in §14.3. §14.2 is the Larmor/Liénard total-power section, not aberration. The row inherits "§14.2 (aberration)" from `cr-boosted-dipole-angular`. Suggest "Eq. (14.13), acceleration term" and checking the section numbers against the book. None of this affects the physics.
2. **Paraxial tolerance.** The row states that $\lvert E_\pm\rvert^2$ agree with `helicity_pattern` to 2% for $\gamma\theta\le1$ without naming $\gamma$. The anchor tests only $\gamma=10.8$. The deviation is $O(\theta^2)=O(1/\gamma^2)$ at fixed $\gamma\theta$: 0.4% at $\gamma=10.8$, but 6% and 26% at $\gamma=1.6$. Suggest "at γ = 10.8 (deviation $O(\theta^2)$)".
3. **Winding scope.** In the fixed basis, $\ell=J_z-\sigma$ is exact at every $\theta$ by rotational covariance. "Paraxially" belongs only to the helicity/OAM interpretation. This is optional wording.
4. **Phase gauge of maps.** The row says the dipole field "carries the global phase of $\mathbf d$", which is true for `dipole_far_field`. The `far_field_maps` phase carries the arbitrary eigenvector gauge from `eigh` instead. Suggest adding "up to the eigenvector's arbitrary global phase" to the leading-mode sentence.

## Verdict

`rederived`.
