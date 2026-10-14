# Exact energy-deficit phase for a prescribed straight flight

## Independent derivation recorded before implementation inspection

Claim: `line-recoil-formation`; intended owners:
`src/pyrite/materials/recoil.py::recoil_mismatch_inv_ang` and
`src/pyrite/materials/recoil.py::recoil_mismatch_slope`.
Source: [Shi et al., Nature Communications 15, 7803 (2024), Eq. (6)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11379926/),
plus the finite-time interaction-picture integral derived here.
Use the quantities, vacuum assumptions, positive mass shell, and reciprocal-vector
sign convention defined in [the resonance derivation](line-recoil-resonance.md).
This claim covers a prescribed constant-momentum flight and a fixed final
vacuum plane-wave electron mode. It does not validate a radiation intensity.

Define the signed energy deficit and mismatch by

$$
 \Delta(\epsilon)=E_f(\epsilon)+\epsilon-E,\qquad
 q(\epsilon)=\frac{\Delta(\epsilon)}{\hbar c},\qquad
 E_f=\sqrt{M^2+\lVert\mathbf a-\epsilon\hat{\mathbf n}\rVert^2}.
$$

Thus the mismatch has inverse-angstrom units. Its exact derivative is

$$
 \boxed{\frac{dq}{d\epsilon}
 =\frac{1-(\mathbf a-\epsilon\hat{\mathbf n})\cdot\hat{\mathbf n}/E_f}
             {\hbar c}
 =\frac{1-\boldsymbol\beta_f\cdot\hat{\mathbf n}}{\hbar c}>0.}
$$

The mass makes the derivative strictly positive: there is at most one root.
At zero photon energy $\Delta(0)=\sqrt{M^2+a^2}-E$ is negative exactly when
$N>0$. At $\epsilon=T$, $\Delta(T)=E_f(T)-M\ge0$, proving existence and the
physical interval stated in the resonance derivation. A stable evaluation in
$0\le\epsilon\le T$ uses rationalization:

$$
 \Delta(\epsilon)=
 \frac{a^2-P^2+2\epsilon D}{E_f+E-\epsilon}
 =\frac{2(\epsilon D-N)}{E_f+E-\epsilon}.
$$

The denominator is positive in that interval. Computing the numerator from
energy-scale dot products avoids subtracting two nearly equal total electron
energies. For a final momentum nearly parallel to the photon, an alternative
stable slope numerator is

$$
 1-\frac{p_{f,\parallel}c}{E_f}
 =\frac{M^2+c^2p_{f,\perp}^2}
        {E_f(E_f+p_{f,\parallel}c)}.
$$

Use that rationalization for nonnegative parallel momentum; the direct form is
well conditioned for negative parallel momentum.

## Finite duration, Jacobian, and segmentation

Choose an interaction-picture phase $\exp(i\Delta t/\hbar)$; choosing its
complex conjugate consistently changes no single-mode intensity. For a constant
coupling over $t_0\le t\le t_0+\tau$ the temporal factor is exactly

$$
 I(t_0,\tau;\epsilon)
 =\int_{t_0}^{t_0+\tau}dt\,e^{i\Delta t/\hbar}
 =\tau e^{i\Delta(t_0+\tau/2)/\hbar}
    \operatorname{sinc}\!\left(\frac{\Delta\tau}{2\hbar}\right),
 \qquad \operatorname{sinc}(x)=\frac{\sin x}{x}.
$$

For prescribed speed $v=\beta c$ and length $L=v\tau$, the argument is
$qL/(2\beta)$ and the phase is $cq(t_0+\tau/2)$. A spatial integral introduces
the conversion $dt=ds/(\beta c)$. Replacing the argument by $qL/2$ omits a
factor $1/\beta$. If a classical mismatch was defined as
$(\epsilon-\mathbf v\cdot(\hbar\mathbf g+\epsilon\hat{\mathbf n}/c))/(\hbar v)$,
the new energy-deficit mismatch must be divided by $\beta$ before replacing
that spatial mismatch.

Near the root, $\Delta=\Delta'_*(\epsilon-\epsilon_*)+O((\epsilon-\epsilon_*)^2)$.
The main-lobe width therefore scales as
$\hbar/(\tau|\Delta'_*|)$; the first zeros lie approximately
$2\pi\hbar/(\tau|\Delta'_*|)$ from the root. The long-time distribution is

$$
 \frac{|I|^2}{\tau}\longrightarrow2\pi\hbar\delta(\Delta),\qquad
 \delta(\Delta)=\frac{\delta(\epsilon-\epsilon_*)}{|\Delta'_*|}
 =\frac{\delta(\epsilon-\epsilon_*)}{\hbar c|q'_*|}.
$$

This Jacobian is required when integrating over photon energy; a centre shift
alone does not supply it, nor does it supply a quantum matrix element.

Splitting one unchanged straight flight is an exact partition of an integral:

$$
 I(t_0,\tau_1+\tau_2)
 =I(t_0,\tau_1)+I(t_0+\tau_1,\tau_2).
$$

The complex starting phase of the second interval is essential. Summing squared
magnitudes, resetting the second starting phase, or redefining its final electron
mode violates split invariance. This identity assumes identical momentum,
reciprocal vector, photon mode and coupling in both subintervals.

## Scope and extension limits

The weak-recoil limit gives
$\Delta=\epsilon(1-\beta\mu)-\boldsymbol\beta\cdot\mathbf Q+O(R^2/E)$,
so the energy-deficit phase reduces to the classical phase with the appropriate
time-to-length conversion. At $q=0$ the temporal factor is $\tau$, not a
singularity. Units of $q'$ are inverse angstroms per eV.

A corrected line centre does not by itself justify changing the classical
formation envelope or arbitrary-trajectory coherent phase accumulation. A bent
or energy-losing trajectory does not have the single pair of stationary electron
modes assumed here. Different final recoil momenta generally denote orthogonal
unobserved electron states; amplitudes cannot all be summed coherently merely
because the photons share a direction. Inter-electron coherence also requires
an electron-state specification. These are separate derivations, not consequences
of the scalar root.

A first-order Snell correction changes internal optical wavevectors, boundary
matching and possibly reciprocal-mode coupling. The exact vacuum root assumes
photon momentum $\epsilon\hat{\mathbf n}/c$. Substituting an internal direction
while retaining its vacuum magnitude is not a derivation of an in-medium recoil
root. The vacuum result can serve as an explicitly limited diagnostic centre;
combining it with a classical refracted formation factor requires a separate
controlled approximation and must not be labelled an exact quantum spectrum.

## Implementation comparison

The independent derivation above was recorded before implementation inspection.
The subsequent comparison finds `matches` within the stated straight-flight
vacuum scope.

- `recoil_mismatch_inv_ang` uses the rationalized energy deficit. Its coefficient
  helper returns $N/E,D/E$; its prefactor $2E/(E_f+E-\epsilon)$ therefore
  recovers the exact $2(\epsilon D-N)/(E_f+E-\epsilon)$ numerator.
- `recoil_mismatch_slope` evaluates the independently derived positive derivative.
  Its forward expression is the rationalized mass-plus-transverse-momentum form;
  its backward expression uses the well-conditioned direct subtraction.
- The docstring expresses duration as $c\tau$ in angstroms. Its temporal
  factor is consequently $cI$, consistent with the integral above after the
  constant unit conversion. It does not silently substitute spatial length.
  NumPy's normalized sinc requires the displayed division by $2\pi$.
- Photon energy is restricted to $[0,T]$. Photon leading axes broadcast with
  the state axes. Nonfinite photon inputs and energies outside that domain
  raise rather than being interpreted as physical roots.

The 29-test module `tests/materials/test_recoil.py` passed in 0.61 s, including
mismatch zero, a finite-difference derivative, and seven-piece complex-amplitude
split invariance. Additional verifier checks covered rotated incident momenta
and a $(3,2)$ paired broadcast: independent 65-digit mass-shell derivatives
agreed within $2.6\times10^{-16}$ relative error. The largest computed mismatch
at the roots was $2.4\times10^{-16}$ inverse angstroms.

Verdict: **rederived**. Suggested owner-applied ledger status: **anchored**,
with `tests/materials/test_recoil.py` as the regression anchor. This status
certifies the scalar mismatch, slope and the prescribed constant-momentum
integral identity only. It does not certify quantum intensity normalization,
arbitrary-trajectory/inter-electron coherence, or an in-medium Snell extension.
Only a human may apply `signed-off`.
