# Vacuum line-recoil validity screen

Validation: `line-recoil-validity-screen`.

## Independent derivation

Fresh-context verification at `f617230c`. This section was written before
implementation-body inspection, from the ledger, signatures/docstrings and
[Shi et al. (2024), Eq. (6)](https://doi.org/10.1038/s41467-024-52050-w).
The source equation is energy and momentum conservation; its subsequent
small-recoil expansion is not used as an exact dispersion relation here.

Let $M=mc^2$, $T>0$ be kinetic energy, $E=M+T$, and
$\mathbf P=c\mathbf p=\sqrt{T(T+2M)}\hat{\mathbf u}=\beta E\hat{\mathbf u}$. Let $\epsilon$ be photon energy,
$\hat{\mathbf n}$ its vacuum direction, and
$\mathbf Q=\hbar c\mathbf g$ the repository reciprocal transfer, with
$\mathbf g=-\mathbf g_{\rm Shi}$. Vectors passed as directions are normalized;
$\mathbf g$ has units inverse angstrom and $\hbar c$ units eV angstrom.
The source conservation equations become

$$
\mathbf P_f=\mathbf P-\mathbf Q-\epsilon\hat{\mathbf n},\qquad
E-\epsilon=\sqrt{M^2+\lVert\mathbf P_f\rVert^2}.
$$

Squaring once, cancelling the photon-energy square, and collecting terms gives

$$
2\epsilon(E-\mathbf P\cdot\hat{\mathbf n}
 +\mathbf Q\cdot\hat{\mathbf n})
=2\mathbf P\cdot\mathbf Q-\lVert\mathbf Q\rVert^2.
$$

Define $A=\beta\hat{\mathbf u}\cdot\mathbf Q$,
$D=1-\beta\hat{\mathbf u}\cdot\hat{\mathbf n}$,
$B=\lVert\mathbf Q\rVert^2/(2E)$, and
$C=\mathbf Q\cdot\hat{\mathbf n}/E$. Then

$$
\epsilon_{\rm ex}=\frac{A-B}{D+C},\qquad
\epsilon_{\rm cl}=\frac{A}{D},\qquad
\boxed{s=\frac{\epsilon_{\rm ex}}{\epsilon_{\rm cl}}-1
=\frac{1-B/A}{1+C/D}-1}.
$$

$A,B$ are energies, $C,D,s$ dimensionless. The physical exact root also
requires a finite nonzero denominator, $0<\epsilon_{\rm ex}\le T$, and
satisfaction of the positive final-energy branch. A root of the squared
relation alone does not establish emission. For finite positive $T$, $D>0$;
thus a classical line exists exactly when $A>0$.
The screen returns NaN for no positive classical line and positive infinity
for a positive classical line whose exact root is forbidden. Infinity is a
screening sentinel, not a physical infinite photon energy.

For $\mathbf Q=\lambda\mathbf Q_0$ with fixed positive longitudinal
projection and fixed geometry, $s\to0$ as $\lambda\to0$. Expanding gives

$$
s=-\frac{\lVert\mathbf Q\rVert^2}
 {2E\beta\hat{\mathbf u}\cdot\mathbf Q}
 -\frac{\mathbf Q\cdot\hat{\mathbf n}}{ED}
 +O(\lambda^2).
$$

This expansion requires both displayed dimensionless transfer corrections
small; small $\lVert\mathbf Q\rVert/E$ alone is insufficient near forward
emission or a vanishing longitudinal projection. For
$\mathbf Q=q\hat{\mathbf u}$ and $\hat{\mathbf n}=\hat{\mathbf u}$,

$$
s\simeq-\frac{\epsilon_{\rm cl}}{\beta E}
\left[1+\frac{1-\beta}{2\beta}\right]
\longrightarrow-\frac{\epsilon_{\rm cl}}{E}
\quad(\beta\to1).
$$

The first contribution alone is $-q/(2\beta E)$; its magnitude, not its
signed value, is the often quoted longitudinal recoil scale. Transverse
transfer raises the squared-transfer term at fixed longitudinal projection,
so photon energy divided by electron energy is not a general upper bound.
Rotations preserve all dot products; negating only the reciprocal label
changes line existence, and must accompany the stated source convention.
Backward observation can change the sign of the denominator correction,
but the physical total shift is nonpositive. Convexity of the massive
electron energy gives

$$
E_f\ge E-\beta\hat{\mathbf u}\cdot
 (\mathbf Q+\epsilon\hat{\mathbf n}),\qquad
\epsilon D\le A.
$$

Thus every finite physical exact root satisfies
$\epsilon_{\rm ex}\le\epsilon_{\rm cl}$, despite the two expansion terms
possibly having opposite signs.

## Implementation comparison and evidence

The implementation matches the derived numerator, transfer correction and
ratio. Both exact and classical roots use the equivalent stable expression

$$
D=\frac{(M/E)^2}{1+\beta}
 +\frac{\beta}{2}\lVert\hat{\mathbf u}-\hat{\mathbf n}\rVert^2.
$$

This retains the forward mass term when floating-point $\beta$ rounds to one.
The exact-root helper rejects nonpositive numerator/denominator, nonfinite
roots and roots above kinetic energy, allowing only eight machine epsilons
of endpoint roundoff. Positivity of the physical denominator follows from
the following identity for a massive final electron:

$$
E-\mathbf P\cdot\hat{\mathbf n}+\mathbf Q\cdot\hat{\mathbf n}
=E_f-\mathbf P_f\cdot\hat{\mathbf n}>0.
$$

`recoil_shift_fraction` then implements the NaN and positive-infinity
sentinels above. Inputs are real finite arrays with broadcast leading axes;
nonpositive kinetic energy or nonunit directions raise `ValueError`.
Roundoff-sized norm errors are normalized. The positive longitudinal scale
is a magnitude; the signed expansion term is negative as derived above. The owner clarified that docstring wording
after the comparison, without changing arithmetic.

An independent 75-digit Decimal bisection used the unsquared final-electron
energy equation directly, rather than the implementation's algebraic root
or coefficient helpers. Constants were taken from the project constants
module; agreement therefore tests kinematics, not independent metrology.

| Kinetic energy (eV) | Reciprocal vector (inverse angstrom) | Photon direction | Independent shift | Absolute implementation difference |
| --- | --- | --- | --- | --- |
| 30000 | (0.4, 0.2, 1.8) | (0.6, 0, 0.8) | -0.01876894007557723 | 1.60e-16 |
| 1000000 | (0, 0, 1.875) | (0, 0, 1) | -0.04114894934166458 | 1.18e-16 |
| 50000000 | (0, 0, 3.75) | (0, 0, 1) | -0.7411356637197242 | below displayed precision |
| 1e15 | (0, 0, 2e-8) | (0, 0, 1) | -0.23211476337677953 | 2.78e-17 |

The electron direction in all four probes was (0, 0, 1).
The last is a numerical stability probe, not a claim about physical
applicability at that energy. Existing anchors independently bisect the
unsquared equation at 65 digits, include backward observation, test the
small-transfer expansion and forward limit, and demonstrate transverse
transfer exceeding the photon/total-electron-energy ratio. Explicit branch
probes returned positive infinity for reciprocal vector (10, 0, 0.001) and
NaN for (0, 0, -2), at 1 MeV with forward electron and photon directions.

## Count population and restrictions

`RECOIL_SHIFT_LIMIT` is 0.01. `_recoil_count` applies the strict comparison
$\lvert s\rvert>0.01$, so a forbidden exact root with positive classical
root is counted through its infinity sentinel, while a nonpositive
classical root is excluded through NaN. An isolated bookkeeping probe fed
exact shifts 0.01, -0.01, the next floating-point number above 0.01,
positive infinity and NaN: exactly two were counted. This checks comparison
semantics; it is separate from the physical reference-value probes.

The screen runs only when `PYRITE_LINE_DATA_COVERAGE` is set (#286).
The parent `line_data_coverage` loop selects CXR electrons with finite
positive escape-piece duration, each requested reflection, and each mosaic
orientation of positive quadrature weight. It next requires the production
vacuum seed to be finite and strictly positive in the active backend
precision. The recoil evaluator further requires finite positive kinetic
energy and a finite nonzero direction norm. It uses `E_repr_keV` when
present, otherwise `E_keV`, converts keV to eV, normalizes directions and
evaluates the recoil ratio in float64 in place on the active array backend
(#286). It runs the same formula code as the checked NumPy reference
(`_recoil_shift_fraction_unchecked`, bitwise-equal to
`recoil_shift_fraction` for NumPy inputs; regression in
`tests/materials/test_recoil.py`). On CUDA, compiler contraction can move
a ratio by a few ulp, so a pair within roundoff of the 1 % limit may be
counted differently from the host evaluation.

Consequently the count is an unweighted number of eligible
escape-piece/reflection/mosaic pairs that survive the production seed gate.
It depends on segmentation, selected electrons, reflections and mosaic
quadrature. It is neither a number of unique electrons nor a probability,
lost photon yield, or intensity-weighted fraction. The recoil count overlaps
the optical outcome counts; it is not an additional disjoint outcome.

The screen runs before optical-domain, Snell-root, line-floor and output-axis
filters; seeds outside optical support or a low explicit photon axis remain
eligible for recoil counting. However, a backend seed made nonfinite by
forward-denominator rounding is classified `invalid_vacuum_seed` and never
reaches `_recoil_count`. The later float64 evaluation does not recover that
population. In particular float32 speed can round to one at GeV energies.
Thus the record is not an exhaustive float64 survey of every geometrically
possible line.

The 1% limit is diagnostic policy, not a universal physical error bound or a
validated cutoff for production accuracy. Ordinary 30 keV HOPG (002) can
already have a longitudinal scale near 1%. No warning is emitted for this
recoil count. The record does not select a production recoil equation,
move line centers, modify spectra, or extend optical tables. Applicability
to Bloch electrons, finite slabs, radiation amplitudes, coherence and
interfaces is outside this vacuum free-electron derivation. Production
recoil remains deferred pending issue #527. Optical extension is separately
deferred until a forward-detector, bremsstrahlung-subtracted use case
requires it.

## Verification and adjudication

The following focused command passed: 60 tests, including reference roots,
shift limits, sentinels, broadcasting, coverage counts, supported spectra
bitwise unchanged without warning, both scoring routes and documentation
math-delimiter checks.

```text
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/materials/test_recoil.py tests/montecarlo/spectrum/test_line_data_coverage.py tests/dev/test_docs.py --durations=0
```

Verdict: **rederived**, restricted to the stated vacuum ratio and record-only
screen population. Units, limiting cases and sign/convention filters pass;
no divergent term was found. The named regression anchors passed locally.
The task owner may advance the ledger to `anchored` after the required
anchor/CI checks; human sign-off remains separate.

