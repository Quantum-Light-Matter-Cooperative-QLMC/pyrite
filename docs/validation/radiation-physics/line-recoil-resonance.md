# Exact vacuum recoil resonance

## Independent derivation recorded before implementation inspection

Claim: `line-recoil-resonance`; intended owner:
`src/pyrite/materials/recoil.py::recoil_resonance_eV`.
Source: [Shi et al., Nature Communications 15, 7803 (2024), Eq. (6)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11379926/).
The derivation below starts from exact conservation, not the approximate Eq. (7).

Assumptions: an infinitely massive crystal absorbs momentum but no energy;
initial and final electrons obey the positive relativistic mass shell; the photon
is a vacuum plane wave. PyRITE's reciprocal vector is the negative of the paper's.
Let $M=m_ec^2$, $E=T+M$, $P=\sqrt{T(T+2M)}$, and use energy-scaled momenta
$\mathbf P=P\hat{\mathbf u}$, $\mathbf Q=\hbar c\mathbf g$, with photon energy
$\epsilon$ and unit photon direction $\hat{\mathbf n}$. Thus $T,E,M,P,Q$
are in eV when $\hbar c$ is in eV angstrom and $g$ in inverse angstroms.
Momentum conservation and the final positive energy are

$$
 c\mathbf p_f=\mathbf a-\epsilon\hat{\mathbf n},\qquad
 \mathbf a=\mathbf P-\mathbf Q,\qquad
 E_f=\sqrt{M^2+\lVert\mathbf a-\epsilon\hat{\mathbf n}\rVert^2},\qquad
 E=E_f+\epsilon.
$$

Squaring only after retaining the necessary condition $E-\epsilon>0$ cancels
both photon-energy-square terms. Define

$$
 D=E-\mathbf a\cdot\hat{\mathbf n},\qquad
 N=\mathbf P\cdot\mathbf Q-\frac{Q^2}{2}
   =\frac{P^2-a^2}{2}.
$$

The unique candidate is

$$
 \boxed{\epsilon_*=\frac{N}{D}}.
$$

Physical-branch checks must enforce finite candidate, $0<\epsilon_*\le T$,
and the original positive mass shell. A stronger exact characterization is
$N>0$: this implies $a<P<E$, hence $D>0$. Moreover

$$
 E D-N=\frac{M^2+\lVert\mathbf a-E\hat{\mathbf n}\rVert^2}{2}>0.
$$

Consequently $E-\epsilon_*>0$, so the squared equation is on the positive
shell, and $E_f\ge M$ proves $\epsilon_*\le T$. Conversely any positive
physical root requires $a<P$ by monotonicity of the energy deficit (proved in
the formation write-up), hence $N>0$. Negative $D$ and negative $N$ can
produce a positive algebraic quotient on the negative-energy branch: reject it.
At $N=0$ only zero photon energy remains; it is outside the requested interval.
No positive branch exists at $\mathbf g=0$ or for $\mathbf P\cdot\mathbf Q\le0$.
The endpoint $\epsilon_*=T$ is possible only when $\mathbf p_f=0$.

## Classical and controlled low-recoil limits

Set $\beta=P/E$, $\mu=\hat{\mathbf u}\cdot\hat{\mathbf n}$,
$d_0=1-\beta\mu$, and $A=\beta\hat{\mathbf u}\cdot\mathbf Q$.
The exact ratio can also be written

$$
 \epsilon_* = \frac{A-Q^2/(2E)}{d_0+\mathbf Q\cdot\hat{\mathbf n}/E},\qquad
 \epsilon_{\rm cl}=\frac{A}{d_0}.
$$

For nonzero $A$, the expansion about the classical root is controlled by
both $|Q^2/(2EA)|\ll1$ and
$|\mathbf Q\cdot\hat{\mathbf n}/(Ed_0)|\ll1$:

$$
 \epsilon_* =\epsilon_{\rm cl}
 \left[1-\frac{Q^2}{2EA}
          -\frac{\mathbf Q\cdot\hat{\mathbf n}}{Ed_0}
          +O\!\left(\frac{Q^4}{E^2A^2},
                     \frac{(\mathbf Q\cdot\hat{\mathbf n})^2}{E^2d_0^2},
                     \frac{Q^2\mathbf Q\cdot\hat{\mathbf n}}{E^2Ad_0}\right)
 \right].
$$

Small $\epsilon/E$ alone does not bound transverse crystal momentum, a near-zero
longitudinal projection $A$, or the small forward denominator $d_0$.
For a Taylor expansion of the electron dispersion itself, with
$\mathbf R=\mathbf Q+\epsilon\hat{\mathbf n}$, require
$\lVert\mathbf R\rVert/E\ll1$ (and $\lVert\mathbf R\rVert/P\ll1$ when
claiming a small directional/kinetic perturbation; at low kinetic energy also
check $\epsilon/T\ll1$). The exact relativistic Hessian gives

$$
 E_f=E-\boldsymbol\beta\cdot\mathbf R
  +\frac{R^2-(\boldsymbol\beta\cdot\mathbf R)^2}{2E}
  +O(R^3/E^2).
$$

The longitudinal quadratic coefficient is $1/\gamma^2$ times the transverse
one. The isotropic quadratic term printed in Shi Eq. (7) is therefore not the
full relativistic second-order Taylor expansion; it becomes appropriate in
the nonrelativistic limit. This distinction is why Eq. (6) is the validation
starting point. The linear term reproduces the classical result above.

## Stable forward evaluation and cheap filters

For a forward photon, never obtain $E-P$ by subtracting nearly equal energies:

$$
 E-P=\frac{M^2}{E+P},\qquad
 D=\frac{M^2}{E+P}
   +\frac P2\lVert\hat{\mathbf u}-\hat{\mathbf n}\rVert^2
   +\mathbf Q\cdot\hat{\mathbf n}.
$$

This expression is exact for unit directions. Near the branch threshold,
$N=\mathbf Q\cdot(\mathbf P-\mathbf Q/2)$ still needs adequate arithmetic
precision; no subtraction-free formula can recover information lost in
input rounding at $N=0$. Units: $N$ is eV squared, $D$ eV, quotient eV.
Rotating all vectors together leaves the root unchanged. Reversing only
$\mathbf g$ changes the physical process and must not be treated as a symmetry.
Paired leading NumPy axes broadcast; final vector axes have length three.
Nonfinite/nonpositive $T$, nonfinite vectors, and nonunit directions are invalid
inputs, not absent physical branches.

## Implementation comparison

The independent derivation above was recorded before any implementation body was
read. The subsequent symbol-level comparison finds `matches`.

- `_coefficients` returns $N/E$ and $D/E$, preserving their ratio. Its mass term
  $(M/E)^2/(1+\beta)$ and squared direction difference implement the stable
  forward identity derived above, including when $\beta$ rounds to one.
- `recoil_resonance_eV` requires positive numerator, denominator and root,
  finiteness, and the kinetic-energy endpoint. Its endpoint allowance is eight
  machine epsilons times $T$ and only clips that roundoff-sized excess. It
  rejects negative-energy squared-equation branches.
- `_state` broadcasts the kinetic-energy leading axes with all three vector
  leading axes; validates finite positive energy and finite real vectors; and
  normalizes directions only inside its documented absolute norm tolerance of
  $10^{-12}$. Inputs outside that tolerance raise.

Units, physical limits and reciprocal-vector signs pass. The 29-test module
`tests/materials/test_recoil.py` passed in 0.61 s. Its unsquared 65-digit
mass-shell bisection anchors ordinary, forward, backward, and $10^{15}$ eV
cases; its remaining checks cover forbidden roots, the electron-at-rest
endpoint, the low-transfer limit and input validation. Additional verifier
checks used a rotated incident direction, a $(3,2)$ paired broadcast, and
65-digit positive-shell residuals: all root residuals were below
$2\times10^{-12}$ inverse angstroms. These checks use repository constants
but no implementation helper for the reference mass shell.

Verdict: **rederived**. Suggested owner-applied ledger status: **anchored**,
with `tests/materials/test_recoil.py` as the regression anchor. No human
sign-off is claimed. As usual, float64 input rounding limits resolution at the
allowed/forbidden boundary; arbitrary overflowing energy/momentum scales are
not independently certified by the finite numerical anchors.
