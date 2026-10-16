# `cr-dipole-rate` re-derivation

Claim: `montecarlo/channeling/emission.py::{rest_frame_dipole_rate_inv_s,dipole_rate_per_ang,transition_operator,first_order_transition,emission_lines}`. Sources: spontaneous electric-dipole emission (Sakurai §2.5; Bethe & Salpeter §59), Andersen, Bonderup, Lægsgaard & Sørensen, Phys. Scr. 28, 308 (1983), and the golden rule with the longitudinal balance of `cr-doppler-line-energy`.

## Independent derivation (before reading the bodies)

**Coupling and mode sum.** In Gaussian units the photon vector potential is $\mathbf A=\sum_{\mathbf k\epsilon}\sqrt{2\pi\hbar c^2/(\omega V)}\,(\epsilon\,a\,e^{i\mathbf k\cdot\mathbf r}+{\rm h.c.})$ and the first-order interaction is $-(e/c)\,\mathbf v\cdot\mathbf A$. The emission matrix element into $(\mathbf k,\epsilon)$ is $e\sqrt{2\pi\hbar/(\omega V)}\,c\,A_{fi}$ with the dimensionless

$$
A_{fi}=\Bigl\langle f\Bigm\vert\frac{\mathbf v\cdot\epsilon^*}{c}\,e^{-i\mathbf k\cdot\mathbf r}\Bigm\vert i\Bigr\rangle,\qquad
\mathbf v=\Bigl(\frac{p_x}{\gamma m},0,\beta c\Bigr).
$$

The $y$ and $z$ factors of $e^{-i\mathbf k\cdot\mathbf r}$ act on the longitudinal plane waves and only enforce momentum conservation. The transverse factor $e^{-iqx}$, $q=k_x$, moves the final Bloch state to $k-q$. Energy conservation for the whole electron, with $\delta E_\parallel=v\,\hbar k_z$, is $\Delta E_\perp=E_i(k)-E_f(k-q)=\hbar\omega(1-\beta\cos\theta)\equiv\hbar\omega D$. Fermi's golden rule with $V\,d^3k/(2\pi)^3=V\omega^2d\omega\,d\Omega/(2\pi c)^3$ and $\int d\omega\,\delta(\Delta E_\perp-\hbar\omega D)=1/(\hbar D)$ gives, with $e^2=\alpha\hbar c$,

$$
\frac{dW}{d\Omega}=\frac{2\pi}{\hbar}\,\frac{2\pi\hbar e^2c^2}{\omega}\,\frac{\omega^2}{(2\pi)^3c^3}\,\frac{\lvert A_{fi}\rvert^2}{\hbar D}=\frac{\alpha\,\omega\,\lvert A_{fi}\rvert^2}{2\pi D},
\qquad
\boxed{\frac{dN}{dz\,d\Omega}=\frac{\alpha\,(\omega/c)\sum_\epsilon\lvert A_{fi}\rvert^2}{2\pi\beta D}}
$$

per unit lab path ($dz=\beta c\,dt$).

**Long-wavelength limit.** From $[H,x]=-i\hbar p_x/(\gamma m)$, $\langle f\vert p_x\vert i\rangle/(\gamma m)=-i\Omega\,x_{fi}$ with $\Omega=\Delta E_\perp/\hbar=\omega D$. For $f\ne i$, $\langle f\vert e^{-iqx}\vert i\rangle\simeq-iq\,x_{fi}$ with $q=(\omega/c)n_x$. Hence

$$
A_{fi}\to-i\,x_{fi}\,\frac{\omega}{c}\bigl[D\,\epsilon_x^*+\beta n_x\,\epsilon_z^*\bigr].
$$

With $\mathbf w=D\hat{\mathbf x}+\beta n_x\hat{\mathbf z}$ and $\mathbf w\cdot\hat{\mathbf n}=n_x(D+\beta\cos\theta)=n_x$, the polarization sum is $\lvert\mathbf w\rvert^2-(\mathbf w\cdot\hat{\mathbf n})^2=D^2-n_x^2/\gamma^2$. Substituting $\omega=\Omega/D$,

$$
\frac{dN}{dz\,d\Omega}=\frac{\alpha}{2\pi\beta}\Bigl(\frac{\Omega}{c}\Bigr)^3\lvert x_{fi}\rvert^2\,\frac{D^2-n_x^2/\gamma^2}{D^4}.
$$

**Boosted dipole.** In the frame moving with $\beta c\hat{\mathbf z}$ the transverse Hamiltonian in proper time is $\gamma H$, so $\omega'=\gamma\Omega$; $x$ is transverse and unchanged. The textbook rate is $A'=\tfrac43\alpha\omega'^3\lvert x_{fi}\rvert^2/c^2$ (equal to $e^2\omega'^3\lvert x\rvert^2/(3\pi\epsilon_0\hbar c^3)$ in SI). Time dilation gives the lab rate $A'/\gamma$ and the path rate

$$
\boxed{\frac{dN}{dz}=\frac{A'}{\gamma\beta c}=\frac43\,\alpha\,\gamma^2\Bigl(\frac{\Delta E_\perp}{\hbar c}\Bigr)^3\frac{\lvert x_{fi}\rvert^2}{\beta}.}
$$

Multiplying by the boosted pattern of `cr-boosted-dipole-angular`, $(3/8\pi)(D^2-n_x^2/\gamma^2)/(\gamma^2D^4)$, reproduces the long-wavelength golden-rule result exactly. The generic operator and the boosted dipole are therefore the same physics at $q\to0$, two independent routes.

**Estimator.** The positive-frequency field is $\propto\sum a_{fi}\lvert f\rangle\langle i\rvert e^{-i\omega_{fi}t}$. Its time-averaged intensity keeps $a_{fi'}^*a_{fi}\langle\lvert i'\rangle\langle f\vert f\rangle\langle i\rvert\rangle$ only for a shared final state and $\omega_{fi}\simeq\omega_{fi'}$ within the widths. So $\langle d_c^\dagger d_c\rangle=\sum_f\sum_{ii'}a_{fi'}^*a_{fi}\rho_{ii'}$ per cluster, which is $P_i\,dN_{fi}/dz$ for nondegenerate levels.

**Harmonic limit and the Shatnev claim.** For an oscillator, $d=a_{01}\hat a$ and $d^\dagger d=\lvert a_{01}\rvert^2\hat n$ is diagonal. A coherent state radiates $\lvert\alpha\rvert^2\lvert a_{01}\rvert^2$, identical to its dephased mixture and to $\lvert\langle d\rangle\rvert^2$ (the classical field). With $x_{01}^2=\hbar/(2\gamma m\Omega)$ and $A^2=2\hbar\lvert\alpha\rvert^2/(\gamma m\Omega)$, $\lvert\alpha\rvert^2x_{01}^2=A^2/4$, and $\tfrac43\alpha\omega'^3(A^2/4)/c^2=\alpha\omega'^3A^2/(3c^2)$ is the Larmor photon rate $\langle P\rangle/\hbar\omega'$ with $\langle a^2\rangle=\omega'^4A^2/2$. There is no phase-locked gain for one emitter.

**Limits.** Units: $\alpha\,(\mathrm{eV}/\mathrm{eV\,Å})^3\,\mathrm{Å}^2=\mathrm{Å}^{-1}$ per path. At $\gamma=1$, $dN/dz\cdot\beta c$ is the atomic rate. Zero bound pairs give no terms.

## Independent numerical checks

Scripts are under `scratch/cr505verify/` (CODATA from `scipy.constants`; own finite-difference eigensolver).

| Check | Independent | Implementation | Difference |
| --- | --- | --- | --- |
| H $2p\to1s$, $\lvert z\rvert=2^7\sqrt2/3^5\,a_0$, 10.1988 eV | $6.25826\times10^{8}$ s⁻¹ | $6.25824\times10^{8}$ s⁻¹ | $-3\times10^{-6}$ |
| same with reduced-mass radius $a_0m/\mu$ | $6.2651\times10^{8}$ s⁻¹ | — | NIST $6.2649\times10^{8}$: $+3\times10^{-5}$ |
| $dN/dz$ at 17 and 80 MeV | own formula | `dipole_rate_per_ang` | $2.2\times10^{-9}$ (constants rounding) |
| Si(110), 17 MeV, $1\to0$, $\theta\gamma=0.3$, $\phi=0$ | $3.255894\times10^{-6}$ Å⁻¹sr⁻¹ | `first_order_transition` $3.255894\times10^{-6}$ | $4.6\times10^{-9}$ |
| same, $\theta\gamma=1.0$, $\phi=0.7$ | $5.761096\times10^{-7}$ | $5.761096\times10^{-7}$ | $4.6\times10^{-9}$ |
| same, $\theta\gamma=2.0$, $\phi=1.2$ | $2.035783\times10^{-7}$ | $2.035783\times10^{-7}$ | $4.6\times10^{-9}$ |
| golden rule / (dipole × pattern) | — | — | $1\times10^{-5}$ to $3.5\times10^{-4}$, $qx\simeq0.01$ |
| catalog strengths, block 0: $1\to0$, $3\to0$, $2\to1$, $3\to2$ | FD dipoles, 1200-point grid | `emission_lines` | $-1.3$ to $-3.7\times10^{-4}$ (plane-wave cut $H=8$ vs FD) |
| $\lvert\alpha\rvert^2=10.25$ coherent vs dephased, dense $\hat a^\dagger\hat a$ | 10.2500000000 | 10.2500000000 | 0 |

The golden-rule column is a real-space quadrature of $\langle f\vert(\mathbf v\cdot\epsilon^*/c)e^{-iqx}\vert i\rangle$ on a fourth-order finite-difference Bloch eigensolver. It uses the self-consistent final state at $k-q$ and shares no code with the plane-wave implementation. Parity forbiddance at $k=0$ (no even-$\Delta n$ lines in block 0) is reproduced.

## Diff against the implementation

- `rest_frame_dipole_rate_inv_s` evaluates $\tfrac43\alpha(\gamma\Delta E/\hbar c)^3\lvert x\rvert^2c$, and `dipole_rate_per_ang` divides it by $\gamma\beta c$. These match the boxed formula.
- `transition_operator` maps order $m$ to $m+(k-q-k_f)/g$, weights the initial plane wave by $\epsilon_x^*\hbar c(k+mg)/(\gamma mc^2)+\epsilon_z^*\beta$, and contracts with $c_f^*$. This is $\langle f\vert(\mathbf v\cdot\epsilon^*/c)e^{-iqx}\vert i\rangle$. Velocity-before-kick ordering differs from the symmetric one by $\hbar q\epsilon_x/(\gamma mc)\,\langle f\vert e^{-iqx}\vert i\rangle=O(q^2x)$, which is recoil order $O(\hbar\omega/E)$ and is neglected consistently.
- `first_order_transition` iterates $q=\omega n_x/c$ to self-consistency and returns $\alpha(\omega/c)\sum\lvert A\rvert^2/(2\pi\beta D)$, which matches the boxed rate.
- `emission_lines` uses $a_{fi}=\sqrt{dN/dz}\,x_{fi}/\lvert x_{fi}\rvert$ with commutator dipoles (`cr-dipole-matrix-element`). Its clusters share one final state, and `EmissionLines.rates` evaluates $\sum a_{fi}a_{fi'}^*\rho_{ii'}$, the derived estimator.

**Notes-text error (not physics).** The ledger Notes and the test comment give $(\sum_n\sqrt{P_nn})^2/\sum_nP_nn$ as "5–10 for $\lvert\alpha\rvert^2=10$". For a Poisson distribution with mean 10 it is **15.6** (8.1 for mean 3, 27.3 for mean 30). The test only asserts $>5$, so it passes. The corrected number lies inside the 12–31 range claimed by arXiv:2603.28827, which strengthens the identification of that "gain" with squaring amplitudes of distinguishable final states. The verdict on the paper is unchanged: $G=1$ for one emitter.

Clustering uses the line half widths of `emission_lines` as its default tolerance. Those widths are overestimated (see `cr-line-shape`), which loosens the tolerance but does not change any 1-D bound line, since those lines are nondegenerate.

## Verdict

`rederived`. Fix the "5–10" figure in the Notes and the test comment to 15.6.

## Resolution (worker, after the verdict)

The phase-locked ratio is corrected to 15.6 for a Poisson mean of 10 (8.1 at 3, 27.3 at 30) in the ledger Notes and the test comment. The test now asserts 15.6 ± 0.1.
