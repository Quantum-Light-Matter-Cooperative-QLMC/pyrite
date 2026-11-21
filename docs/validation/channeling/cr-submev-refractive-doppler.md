# `cr-submev-refractive-doppler` re-derivation

Claim: `checks/channeling_submev_optics.py::{medium_line,escape}`, recorded in
the {ref}`sub-MeV optics record <sec-submev-optics>`.
Research record for #585; production line energies still use
`cr-doppler-line-energy`.

Fresh context, 2026-11-15, commit `180691b42`. The derivations below were done
from the research page before the code was read.

## Conservation route

At fixed transverse level $\varepsilon$, $c^2p_z^2=E^2-m^2c^4-2E\varepsilon$
gives

$$
c\,dp_z=\frac{(E-\varepsilon)\,dE-E\,d\varepsilon}{cp_z}\approx\frac{dE-d\varepsilon}{\beta}.
$$

Emission sets $dE=-\hbar\omega$ and $d\varepsilon=-\Delta\varepsilon$. The
photon carries $n\hbar\omega\cos\theta/c$ along $z$. Conserving $p_z$:

$$
\frac{\hbar\omega-\Delta\varepsilon}{\beta}=n\hbar\omega\cos\theta
\quad\Longrightarrow\quad
\hbar\omega\,(1-\beta n\cos\theta)=\Delta\varepsilon .
$$

## Phase-matching route

This route is classical and makes no choice of photon momentum. A transverse
current oscillating at $\Omega=\Delta\varepsilon/\hbar$ moves along $z=vt$.
It radiates where its phase matches the field's:
$\omega-k_zv=\Omega$, with $k_z=n\omega\cos\theta/c$. That is the same
condition.

## Photon momentum

The Minkowski momentum $n\hbar\omega/c$ is the pseudo-momentum conserved in a
homogeneous medium, and it agrees with the phase-matching route. The Abraham
momentum $\hbar\omega/nc$ would give $1-\beta\cos\theta/n\approx1-\beta(1+\delta)$,
which has the wrong sign.

## Exactness and cap

Within first-order kinematics, the forward line $\Delta\varepsilon/[1-\beta(1-\delta)]$
is exact in $\delta$. With $x=\beta\delta/(1-\beta)$, the relative shift is
$-x/(1+x)$. The cap $x\le0.3$ therefore limits reliance on the tables. It is
not an expansion limit.

Limits:

- $\delta\to0$ gives the vacuum line $\Delta\varepsilon/(1-\beta)$.
- $n<1$ lowers the line.

## Hand checks

The values below use $\delta$ at the vacuum line. They agree with the record.

| Case | Shift (eV) |
| --- | --- |
| W⟨001⟩, 30 keV | −1.94 |
| W⟨001⟩, 100 keV | −3.81 |
| W⟨001⟩, 300 keV | −6.57 |
| W⟨001⟩, 1 MeV | −11.2 |

$\delta$ itself was checked by hand: $\delta=1.163\times10^{-2}$ for W at
333 eV.

Two rows need $\delta$ at the shifted line, which the fixed point uses:

| Case | Shifted line | $\delta$ at that line |
| --- | --- | --- |
| W, 500 keV (M₄ edge) | 1873 eV | $2.63\times10^{-4}$ |
| diamond(111), 1 MeV | 171.8 eV | $1.67\times10^{-2}$ |

For the diamond case, the self-consistent $x$ is 0.27.

## Escape

$\frac1t\int_0^te^{-s/L}ds=(1-e^{-t/L})/(t/L)$, with
$L=\lambda/4\pi\beta_\text{idx}$ the intensity length. That matches
`-expm1(-x)/x` in the code. For W at 30 keV:

| Slab | $T$ |
| --- | --- |
| 100 nm | 0.322 |
| 500 nm | 0.068 |

## Findings and adjudication

- The page's $\delta$ column held vacuum-line values, but the shifts use
  $\delta$ at the medium line. *Addressed*: the column is relabelled and the
  difference is stated.
- The page gave $x=0.15$; the self-consistent value is 0.27. *Addressed.*
- The page said 5–17×; the ratio to the finite-β shift is 10–17×.
  *Addressed.*
- The page said every omitted row lies below $3\hbar\omega_p$. That was false
  for diamond(111) at 500 and 750 keV and for Si⟨100⟩ at 1 MeV. *Addressed*:
  rows were added and refusals are named.
- The code gave converged medium lines below $3\hbar\omega_p$ that the page
  called refused. *Addressed*: `evaluate` now gives no medium line below
  $3\hbar\omega_p$.
- Not recorded: the absorptive broadening of about
  $\hbar\omega\,\beta\,\beta_\text{idx}/(1-\beta)$.

Verdict: `rederived` (physics and code). This is a static, homogeneous,
isotropic estimate from isolated-atom tables. It is not a dielectric theory
of emission below $3\hbar\omega_p$ or at edges. Human sign-off is tracked in
#277.
