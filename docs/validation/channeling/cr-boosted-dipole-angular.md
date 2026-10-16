# `cr-boosted-dipole-angular` re-derivation

Claim: `montecarlo/channeling/emission.py::{boosted_dipole_pattern,dipole_stokes}`. Source: relativistic aberration and the solid-angle Jacobian (Jackson, 3rd ed., §11.4, §14.2).

## Independent derivation (before reading the bodies)

**Aberration.** The rest frame moves at $\beta c\hat{\mathbf z}$. A lab photon $(\omega,\hat{\mathbf n})$ has $\omega'=\gamma\omega(1-\beta\cos\theta)=\gamma\omega D$, $k'_x=k_x$ and $k'_y=k_y$, so

$$
n'_x=\frac{\sin\theta\cos\phi}{\gamma D},\qquad \cos\theta'=\frac{\cos\theta-\beta}{D},\qquad
\frac{d\Omega'}{d\Omega}=\frac{d\cos\theta'}{d\cos\theta}=\frac{1-\beta^2}{D^2}=\frac{1}{\gamma^2D^2}.
$$

**Pattern.** Photon number is invariant, so $dN=(dN/d\Omega')\,d\Omega'$. For a dipole along the plane normal $\hat{\mathbf x}$,

$$
\boxed{\frac{dN}{d\Omega}=\frac{3}{8\pi}\,\frac{1-n_x'^2}{\gamma^2D^2}}
$$

normalized to 1 by the change of variables (no extra Doppler power factor: that would apply to energy, not number). The lab total per lab time is $A'/\gamma$. At $\gamma=1$ it is $(3/8\pi)\sin^2\Theta$ about $\hat{\mathbf x}$. Forward, $D=1/(\gamma^2(1+\beta))$ and the value is $(3/8\pi)\gamma^2(1+\beta)^2$. The rest-frame null $\hat{\mathbf n}'=\hat{\mathbf x}$ ($\cos\theta'=0$, $\phi=0$) maps to $\cos\theta=\beta$.

**Polarization.** Let me check it by boosting the rest-frame radiation field, independently of a Liénard–Wiechert formula. Take $\mathbf E'\propto(\hat{\mathbf n}'\times\hat{\mathbf x})\times\hat{\mathbf n}'$ and $\mathbf B'=\hat{\mathbf n}'\times\mathbf E'$. Then $\mathbf E_\parallel=\mathbf E'_\parallel$ and $\mathbf E_\perp=\gamma(\mathbf E'-\boldsymbol\beta\times\mathbf B')_\perp$. Equivalently, from the golden-rule amplitude of `cr-dipole-rate`, $\mathbf E\propto\mathbf w_\perp$ with $\mathbf w=D\hat{\mathbf x}+\beta n_x\hat{\mathbf z}$:

$$
\mathbf E\propto D\hat{\mathbf x}+\beta n_x\hat{\mathbf z}-n_x\hat{\mathbf n}=-\bigl[n_x(\hat{\mathbf n}-\beta\hat{\mathbf z})-D\hat{\mathbf x}\bigr].
$$

The field is real, so the radiation is fully linearly polarized and $V=0$. With $\mathbf e_1\propto\hat{\mathbf x}-n_x\hat{\mathbf n}$ and $\mathbf e_2=\hat{\mathbf n}\times\mathbf e_1$, $Q=(E_1^2-E_2^2)/I$ and $U=2E_1E_2/I$. When $n_x=0$ (forward, or $\phi=90^\circ$), $\mathbf E\parallel\hat{\mathbf x}=\mathbf e_1$, giving $Q=1$ and $U=0$. The direction is independent of $x_{fi}$ and $\Delta E$, so all planar dipole lines share it.

## Independent numerical checks

| Check | Result |
| --- | --- |
| $\int(dN/d\Omega)\,d\Omega$, nested adaptive `quad`, $\gamma=1,1.5,7,34.27,400,10^4$ | $1.000000000000$ at all six |
| finite-difference $d\cos\theta'/d\cos\theta$ at $\gamma=34.27$, $\theta=0.02$ | 2174.649114 vs analytic 2174.649113 |
| pattern vs own aberration map, 50 random directions | max relative difference $2.6\times10^{-13}$ |
| Stokes from the boosted $(\mathbf E',\mathbf B')$, 200 random directions, $\gamma\in\{1.0001,2,34.27,157\}$ | max $\lvert\Delta(Q,U)\rvert=5.6\times10^{-12}$; $\mathbf E\cdot\hat{\mathbf n}=0$ checked |
| example $\theta=1/\gamma$, $\phi=45^\circ$, 17 MeV | $(Q,U)=(7.098\times10^{-4},-0.99999975)$, both routes |
| golden-rule amplitudes (`cr-dipole-rate`) at $\theta\gamma=1$, $\phi=0.7$ | $Q=-0.16886$, matching `dipole_stokes` |

## Diff against the implementation

`boosted_dipole_pattern` returns $(3/8\pi)(1-n_x'^2)/(\gamma D)^2$, with $D$ taken from the cancellation-free `doppler_denominator` (`cr-doppler-line-energy`). `dipole_stokes` forms $n_x(\hat{\mathbf n}-\beta\hat{\mathbf z})-D\hat{\mathbf x}$ and projects it on `_photon_frame`'s $(\mathbf e_1,\mathbf e_2)$. Its overall sign is opposite to my $\mathbf w_\perp$, which is irrelevant for $Q$ and $U$. Both are identical to the derivation. `_photon_frame` raises for $\hat{\mathbf n}=\pm\hat{\mathbf x}$, where the dipole pattern is null.

## Verdict

`rederived`.
