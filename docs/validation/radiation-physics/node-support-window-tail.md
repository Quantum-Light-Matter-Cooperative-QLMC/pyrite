# Support-windowed node reduction with an analytic far tail

Validation: `node-support-window-tail`. Status: **rederived**
(issue #1191), applied by the task owner after the fresh-context recheck below.
No human sign-off implied.

## Quantity, source and conventions

Source: the exact linear-escape formation integral in
[coherent-formation-absorption](coherent-formation-absorption.md), with the
straight-flight split limit of `substep-radiation-invariance`. This record
checks the algebra and explicit approximation envelopes of a numerical
reduction. Relative spectral accuracy is qualified separately on saved scenes.

Use $s_j$ for the formation slope [eV$^{-1}$], $h_j=\Delta L_j/2$ [angstrom],
$d(E)=\delta\omega(E)$ [angstrom$^{-1}$], and $u_j,b_j$ for dimensionless start
and end attenuation amplitudes. The formation mismatch and field are

$$
v_j=s_j(E-E_{v,j})-h_jd(E),\qquad
F_j=\frac{b_je^{iv_j}-u_je^{-iv_j}}{2(iv_j-q_j)},\qquad
S(E)=\sum_j c_j e^{i\phi_j(E)}F_j(E).
$$

The coefficients $c_j$ contain the caller's field weight. The reduction sums
$|S|^2$ over polarizations and flights, then applies the caller's angular and
per-primary normalization. Endpoint moments $m_2,m_3,m_4$ have units intensity
$\times$ eV$^2$, eV$^3$, eV$^4$; their inverse powers return intensity.

## Independent derivation and the repaired discrepancies

The first fresh-context review of source `37a1c11ae` found two failures:
adjacent slope comparisons chained distinct endpoints into one cluster for a
1024-way straight split, and freezing $d(E)$ was incorrectly described as an
exact pole decomposition. These failed the limits filter. Both now have
regression tests; the revised fences below replace the unsupported claims.

### Affine mismatch and complex-pole sign

If $d(E)=\beta E+\gamma$, define

$$
A_j=s_j-h_j\beta,\qquad
E_{k,j}=\frac{s_jE_{v,j}+h_j\gamma}{A_j},\qquad
p_j=E_{k,j}-i\frac{q_j}{A_j}.
$$

Then $iv_j-q_j=iA_j(E-p_j)$ exactly. Endpoint residues are
$c_jb_j/(2iA_j)$ and $-c_ju_j/(2iA_j)$, with phases $\phi_j+v_j$ and
$\phi_j-v_j$. Both the residues and pole width use $A_j$, not $s_j$.
Nonpositive $A_j$ falls back to exact full-grid evaluation. Affine recognition
allows only float64 arithmetic roundoff, not a fitted curvature tolerance.

An independent probe with $s=1$, $h=10000$, $\beta=10^{-6}$, $E_v=990$
gives $A=0.99$, $E_k=1000$, and $m_2=0.5/0.99^2=0.5101520253035404$.
The revised implementation agrees to roundoff.

### Non-affine dispersion is a bounded denominator surrogate

For general tabulated $d$, freeze its denominator at $d(E_v)$:
$p=E_v+hd(E_v)/s-iq/s$. The omitted term is
$\Delta w=-ih[d(E)-d(E_v)]$; it is **not zero**. Let

$$
D_j=\max\bigl(|d_{\min}-d(E_{v,j})|,|d_{\max}-d(E_{v,j})|\bigr).
$$

The exact window includes a reach of at least $x_c|h_j|D_j/s_j$ from every
real pole centre. On every far node,

$$
\rho_j=\frac{|\Delta w_j|}{|w_{0,j}|}\le\frac1{x_c}<1,\qquad
\left|\frac1{w_j}-\frac1{w_{0,j}}\right|
\le\frac{\rho_j}{1-\rho_j}\frac1{|w_{0,j}|}.
$$

Sum these endpoint field-error envelopes **before** bounding intensity.
If $B_d$ is that sum and $S_0$ the frozen-denominator field, then
$\bigl||S|^2-|S_0|^2\bigr|\le2|S_0|B_d+B_d^2$.
This is an absolute endpoint-envelope bound. It does not guarantee a small
relative intensity error when endpoints interfere destructively. It is
separate from the weak-cluster budget below. For $x_c\le1$, grouped scoring
uses the exact full-grid reference.

### Phase clusters and junction cancellation

Let $d_i$ be the observer-arrival length (midpoint time times light speed
minus the observation-direction path projection), distinct from dispersion
$d(E)$. Every endpoint has the phase function

$$
\psi_i(E)=\kappa_iE+\lambda_i d(E)+\zeta_i,\qquad
\kappa_i=d_i/\hbar c\pm s_i,\quad
\lambda_i=-L_{\mathrm{esc},i}\mp h_i.
$$

For affine-corrected rows the equivalent coefficients use $A_i$ and the
transformed centre. Cluster membership compares each endpoint with **one
representative**, never its preceding neighbor. Its maximum band drift is
bounded by

$$
\eta_i=|\Delta\kappa_i|\max_{E\in\mathrm{band}}|E-E_n|
       +|\Delta\lambda_i|\max_{E\in\mathrm{band}}|d(E)-d(E_n)|.
$$

Only $\eta_i\le10^{-3}$ rad joins the representative; the constant phase at
$E_n$ stays in the complex residue. Since $|e^{it}-1|\le|t|$, the field-error
bound is the sum of $|g_i|\eta_i/(|x|-|e_i|)$, in addition to the denominator
error above. Equal secant slopes alone do not establish equal nonlinear
phase functions.

A shared junction cancels exactly only when its attenuation, phase function,
pole and residue $c/A$ match. A transparent unchirped straight flight meets
that condition. Chirped rows need not cancel; their residual endpoints remain
in the expansion and window fences. Splits into 1, 16 and 1024 pieces now give
identical windows and moments, including $m_2=0.5$ for $s_F=c_F=1$.

### Multipole field and intensity remainder

For $x=E-E_n$, $e_i=p_i-E_n$, and $g_i=\gamma_i e^{i\psi_i(E_n)}$, define
$A_m=\sum_i g_i e_i^m$ within a cluster. Expanding the rational field gives

$$
T(x)=\frac{A_0}{x}+\frac{A_1}{x^2}+\frac{A_2}{x^3},\qquad
|R(x)|\le\sum_i\frac{|g_i||e_i|^3}{|x|^3(|x|-|e_i|)}.
$$

The exact radius enforces $\max|e_i|/|x|\le0.1$. Averaging the intercluster
beats and retaining intensity through $x^{-4}$ gives

$$
K(x)=\frac{|A_0|^2}{x^2}+\frac{2\Re(A_0A_1^*)}{x^3}
 +\frac{|A_1|^2+2\Re(A_0A_2^*)}{x^4}.
$$

The truncation error includes **both** $2|T||R|+|R|^2$ and the discarded
intensity terms $2\Re(A_1A_2^*)/x^5+|A_2|^2/x^6$. The field remainder alone
is not a bound on the entire intensity error.

### Aggregate weak fields and all-interval beats

At the provisional smallest far distance $r$, the weak-cluster envelope uses
$|A_0|+|A_1|/r+|A_2|/r^2+r|R(r)|$, plus the phase-drift envelope, summed over
polarizations. Thus $A_0=0$ does not hide a nonzero $A_1$. Weak clusters may
be excluded from the beat-gap fence only when their **total** envelope is at
most $10^{-2}$ times the largest cluster's envelope; otherwise all of them
remain. This controls omitted surrogate interference by
$2B_{\mathrm{ret}}B_{\mathrm{weak}}+B_{\mathrm{weak}}^2$; it is not the
nonlinear-denominator error budget.

For retained clusters the beat derivative is
$\Delta\kappa+\Delta\lambda d'(E)$. Every interval of the tabulated dispersion
contributes to the minimum derivative bound. The implementation subtracts
$(\lambda_{\max}-\lambda_{\min})\max|d'-d_{\mathrm{sec}}|$ from the smallest
retained secant gap. This is a conservative lower bound for **any** retained
pair. A nonpositive bound uses full-grid exact windows; otherwise the beat
reach is at least $2x_c/g_{\min}$.

## Error scope and bookkeeping

The signed integrated oscillatory error of a resolved isolated sinc-squared
line is $O(x_c^{-2})$. Its mass-L1 error is **$O(x_c^{-1})$**, as is its missing
mass under plain truncation ($1/(\pi x_c)$). General grouped relative yield,
centroid, peak, FWHM, binned mass-L1 and line/background accuracy require the
saved-scene qualification; no universal relative tolerance follows from the
per-endpoint envelopes above. Node sampling must resolve the relevant beat;
coarse or aliasing grids cannot inherit the isolated continuous-integral bound.
The opt-in automatic policy retains exact full-node evaluation whenever the
fastest endpoint beat has fewer than four samples per cycle; explicit numeric
cutoffs remain available for controlled qualification. Unsupported routes stay
on their reference reduction.

The tail $K(0)=0$ is added everywhere by float64 FFT convolution, and exact
windows add exact minus $K$. Distant poles use geometric recentering (at most
$1/32$ of distance); far poles whose windows reach the band use direct
calculation. Nonuniform grids retain the full-node reference.

Per-row supports covering at least one quarter of the grid are evaluated
exactly over the full grid on both backends. CuPy uses the existing tuned
dense reducer for these broad rows, with at most four times their support
work; narrow rows use the window kernel. This selection also keeps the
backend approximation identical. Empty tail deposits skip the device FFT.

NumPy and float32 CuPy share the plan. CuPy fields use float32, while tail
corrections, atomic accumulation and FFT use float64. FP64 CuPy retains the
reference. CPU/GPU parity tests check implementation agreement, not independent
correctness of their shared plan.

## Evidence and qualification

Fast anchors: `tests/montecarlo/spectrum/test_node_support_window.py`, including
affine dispersion, 1024-way splitting, grid-wide exactness, random flights,
and far/direct poles. CUDA parity includes both kernels and the public fused
batch route (`test_node_support_window_cuda.py`; five tests passed on the RTX
5080 at `98944f5eb`, before the revised validation fences).

`checks/node_support_window_1191.py` alternates reference/window calls on
trusted #1142/#1143 transports, reporting first calls separately from three
warm repeats. It applies the existing yield/centroid $10^{-3}$ and peak/FWHM,
normalized 100 eV mass-L1, fixed-window line/background $10^{-2}$ margins.
Reports retain code identity, transport digest, population, backend, cap,
grid, memory and timing spread. Measurements before the revised fences do
not qualify their changed support or cost; the final revision needs new runs.


## Independent verdict (2026-11-14)

A fresh verifier returned **rederived** for the revised conditional
mathematical claim: units, limits, signs and conventions pass; independent
straight-flight and affine-dispersion probes match. The verifier reran 19
host regressions and 9 documentation checks (28 passed), inspected the rendered
article (80 math elements, no literal dollar delimiters), and confirmed the
cited earlier CUDA revision exists. The task owner applied the status.
Revised hardware parity, scene margins and cost remain separate acceptance
work; neither this verdict nor those measurements imply human sign-off.
