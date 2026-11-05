# Independent verification: `optical-stack-fast-solver`

Fresh-context verification for issue #962. Sections 1 to 5 were written before
the bodies of `transition_radiation_stack.py` and
`transition_radiation_stack_kernel.py` were opened; the only inputs were the
ledger row, the derivation docstrings, and the dense reference system
(`_free_system`, `_dense_amplitude`, `interface_response`), which defines what
"same boundary system" means.

## 1. Boundary system to be reproduced

Units are Gaussian with $\hbar=c=1$. In medium $j$ the free plane waves have
$\mathbf K=(\mathbf k_\perp,\kappa)$, $\kappa=\pm\lambda_j$,
$\lambda_j=\sqrt{\varepsilon_j\omega^2-k_\perp^2}$ with the root
$\operatorname{Im}\lambda_j\ge0$. `_free_system` parametrises each wave by its
tangential $\mathbf E_t$, with

$$
E_z=-\frac{\mathbf k_\perp\cdot\mathbf E_t}{\kappa},\qquad
\mathbf H=\frac{\mathbf K\times\mathbf E}{\omega}.
$$

Medium $0$ is the substrate and medium $n_{\rm if}$ the vacuum; interface $i$ at
$z_i$ joins media $i$ and $i+1$. The rows are (free field above) minus (free field
below) equal to a source $\mathbf s_i=(s_{E_x},s_{E_y},s_{H_x},s_{H_y})_i$. Up-going
amplitudes are referenced to the bottom of their medium, and down-going amplitudes
to the top. With $d_j=z_j-z_{j-1}$ and $p_j=e^{i\lambda_jd_j}$, the phases in the
continuity rows at interface $i$ are:

| wave | phase at $z_i$ |
|---|---|
| medium $i+1$, up | $1$ |
| medium $i+1$, down | $p_{i+1}$ |
| medium $i$, up | $p_i$ |
| medium $i$, down | $1$ |

The top wave is referenced to $z_{n_{\rm if}-1}$, and the substrate has no up wave.

## 2. Decoupling and admittances

Let $\hat{\mathbf k}=\mathbf k_\perp/k_\perp$ and
$\hat{\mathbf s}=\hat{\mathbf z}\times\hat{\mathbf k}$, so
$(\hat{\mathbf k},\hat{\mathbf s},\hat{\mathbf z})$ is right-handed. For
$\mathbf E_t=e_k\hat{\mathbf k}+e_s\hat{\mathbf s}$ and $E_z=-k_\perp e_k/\kappa$,

$$
\mathbf K\times\mathbf E
=-\kappa e_s\hat{\mathbf k}
+\left(\kappa e_k+\frac{k_\perp^2e_k}{\kappa}\right)\hat{\mathbf s}
+k_\perp e_s\hat{\mathbf z},
$$

and $\kappa^2+k_\perp^2=\varepsilon\omega^2$. Therefore

$$
H_k=-\frac{\kappa}{\omega}e_s,\qquad H_s=\frac{\varepsilon\omega}{\kappa}e_k.
$$

The pair $(E_s,H_k)$ involves only $e_s$ and the pair $(E_k,H_s)$ only $e_k$, so
the two polarizations decouple. With $\kappa=+\lambda$ for amplitude $a$ and
$\kappa=-\lambda$ for amplitude $b$,

$$
E=a+b,\qquad H=Y(a-b),\qquad
\boxed{Y_{\rm TE}=-\frac{\lambda}{\omega},\quad Y_{\rm TM}=\frac{\varepsilon\omega}{\lambda}},
$$

where TE is $(E,H)=(E_s,H_k)$ and TM is $(E,H)=(E_k,H_s)$. $Y$ is dimensionless.
At $k_\perp=0$ any $\hat{\mathbf k}$ gives the same physics, because
$Y_{\rm TE}=-Y_{\rm TM}=-\sqrt{\varepsilon}$ and the two equations are related by a
rotation by $90^\circ$.

## 3. Recursion from continuity

At the top of medium $i$ (just below interface $i$) write the up wave as the
reflection of the down wave plus a source term from everything lower:
$a=\rho_ib+A_i$, with $\rho_0=A_0=0$. Let $a',b'$ be the amplitudes at the bottom of
medium $i+1$ and $Y=Y_i$, $Y'=Y_{i+1}$. Continuity with jump $(s_E,s_H)$ reads

$$
a'+b'-a-b=s_E,\qquad Y'(a'-b')-Y(a-b)=s_H.
$$

Eliminate $a$ with $a=\rho_ib+A_i$:

$$
(1+\rho_i)b=a'+b'-A_i-s_E,\qquad
Y'(a'-b')+Y(1-\rho_i)b-YA_i=s_H.
$$

Multiply the second equation by $(1+\rho_i)$ and substitute the first:

$$
\bigl[(1+\rho_i)Y'+(1-\rho_i)Y\bigr]a'
=\bigl[(1+\rho_i)Y'-(1-\rho_i)Y\bigr]b'
+2YA_i+(1-\rho_i)Ys_E+(1+\rho_i)s_H.
$$

Hence $a'=\Gamma_ib'+u_i$ with

$$
D_i=(1+\rho_i)Y_{i+1}+(1-\rho_i)Y_i,\qquad
\Gamma_i=\frac{(1+\rho_i)Y_{i+1}-(1-\rho_i)Y_i}{D_i},
$$

$$
u_i=\tau_iA_i+\alpha_is_{E,i}+\beta_is_{H,i},\qquad
\tau_i=\frac{2Y_i}{D_i},\quad
\alpha_i=\frac{(1-\rho_i)Y_i}{D_i},\quad
\beta_i=\frac{1+\rho_i}{D_i}.
$$

The intermediate division by $1+\rho_i$ is not essential: at $\rho_i=-1$ the
continuity equations directly give $a'=-b'+A_i+s_E$, which is the same formula
with $D_i=2Y_i$.

Crossing medium $i+1$: the up wave at its top is $p_{i+1}a'$ and $b'=p_{i+1}b$ for
the down wave $b$ at its top, so

$$
\rho_{i+1}=p_{i+1}^2\Gamma_i,\qquad A_{i+1}=p_{i+1}u_i.
$$

The vacuum has $b'=0$, so the top amplitude is $u_{n_{\rm if}-1}$ at the same phase
reference as the dense column (the last interface).

## 4. Unrolled response and layout

Since $u_i$ is linear in $A_i$,

$$
a_{\rm top}=\sum_iC_i(\alpha_is_{E,i}+\beta_is_{H,i}),\qquad
C_{n_{\rm if}-1}=1,\quad C_i=C_{i+1}\tau_{i+1}p_{i+1}.
$$

The dense response is $R_{i,r,c}=(M^{-1})_{{\rm top}+c,\,4i+r}$, read off from the
transposed solve, with $r\in(E_x,E_y,H_x,H_y)$ the source component and
$c\in(x,y)$ the output component. Using the projections

$$
s_E^{\rm TE}=\mathbf s_E\cdot\hat{\mathbf s},\quad
s_H^{\rm TE}=\mathbf s_H\cdot\hat{\mathbf k},\quad
s_E^{\rm TM}=\mathbf s_E\cdot\hat{\mathbf k},\quad
s_H^{\rm TM}=\mathbf s_H\cdot\hat{\mathbf s},
$$

and $\mathbf a_{\rm top}=a^{\rm TM}\hat{\mathbf k}+a^{\rm TE}\hat{\mathbf s}$, the
expected tensor is, for $c\in(x,y)$ and Cartesian index $m\in(x,y)$,

$$
R_{i,E_m,c}=C^{\rm TM}_i\alpha^{\rm TM}_i\hat k_m\hat k_c
+C^{\rm TE}_i\alpha^{\rm TE}_i\hat s_m\hat s_c,
$$

$$
R_{i,H_m,c}=C^{\rm TM}_i\beta^{\rm TM}_i\hat s_m\hat k_c
+C^{\rm TE}_i\beta^{\rm TE}_i\hat k_m\hat s_c.
$$

## 5. Filters and predicted hazards

**Units.** $Y$, $\alpha$, $\beta$, $\tau$, $\rho$, $\Gamma$, $C$ and $p$ are all
dimensionless, and $\lambda_jd_j$ is (eV)(1/eV). The output has the units of the
source. Pass.

**Limits, from the formulas above.**

- Matched medium: $\rho\equiv0$, $D=2Y$, $\tau=1$, $C_i=\prod p$, and a jump gives
  $a'=(s_E+s_H/Y)/2$, which is the direct two-equation solution.
- Vanishing film ($p_{i+1}=1$): with $\rho_i=0$, one finds
  $\Gamma_{i+1}=(Y_{i+2}-Y_i)/(Y_{i+2}+Y_i)$ and
  $\tau_{i+1}\tau_i=2Y_i/(Y_i+Y_{i+2})$, independent of $Y_{i+1}$.
- Opaque layer ($p\to0$): $\rho\to0$, $A\to0$, $C\to0$ below it, so only sources
  above survive and the exit interface sees a bare substrate.

**Signs.** $\hat{\mathbf s}$ and the TE sign follow Section 2. $Y_{\rm TE}$ is
negative for a propagating wave, which only enters ratios.

**Predicted hazards, to be tested numerically.** Section 3 shows that $|p|\le1$
says nothing about $\rho$, $\Gamma$, $\tau$ or $1/D$. The Poynting identity

$$
\operatorname{Re}\bigl[(a+b)\,(Y(a-b))^*\bigr]
=\operatorname{Re}Y\,(|a|^2-|b|^2)+2\operatorname{Im}Y\operatorname{Im}(ba^*)
$$

bounds $|\Gamma|\le1$ only when $Y$ is real, that is, a propagating lossless layer.
In an evanescent layer $Y$ is imaginary and $\Gamma$ is unbounded. $D_i=0$ says
that the lower half-stack, with medium $i+1$ taken as semi-infinite, has a bound
mode. That is **not** a singularity of the real finite stack, whose top layer is
finite, so the dense matrix can stay regular while the recursion divides by zero.
This is a removable singularity of the recursion that the docstring's list
("$\lambda=0$ in a layer, and the Cherenkov pole") does not contain.

An exact example with representable numbers is $\omega=1$, $k_\perp=3/4$,
$\varepsilon_i=-9/2$ and $\varepsilon_{i+1}=1/2$, for which $\lambda_i=9i/4$,
$\lambda_{i+1}=i/4$, $Y^{\rm TM}_i=2i$, $Y^{\rm TM}_{i+1}=-2i$ and so $D_i=0$ with
$\rho_i=0$.

Other hazards to test: $\lambda\to0$ in a layer, where $Y^{\rm TE}\to0$ and
$Y^{\rm TM}\to\infty$ with cancellation in $D$; $\varepsilon=0$; $k_\perp\to0$ and
the normalisation of $\hat{\mathbf k}$; high-Q tunnel-coupled cavities, where the
true gain grows as $e^{\kappa L}$ in passive lossless media.

## 6. Comparison with the implementation

Bodies were read after Sections 1 to 5 were written. Frozen revision: `dfd72a3b`
(source identical to `e3a07bce`; the later commit adds only a check record).

### Symbolic diff against the NumPy path

Every term matches Sections 2 to 4.

| Derived | `transition_radiation_stack.py` | Result |
|---|---|---|
| $Y_{\rm TE}=-\lambda/\omega$, $Y_{\rm TM}=\varepsilon\omega/\lambda$; index 0 is TE, 1 is TM | `_modes`, line 69 | match |
| $\hat{\mathbf s}=(-\hat k_y,\hat k_x)$, and $\hat{\mathbf k}=\hat{\mathbf x}$ at $k_\perp=0$ | lines 65 to 67 | match |
| $D,\alpha,\beta,\tau$ with the old $\rho$ | lines 83 to 87 | match |
| $\rho_{i+1}=p_{i+1}^2\Gamma_i$ with $p_{i+1}$ of medium $i+1$ and $d=z_{i+1}-z_i$ | lines 90 to 91 | match |
| $p=1$ for the vacuum; $\rho$ unused afterwards | lines 88 to 89 | match |
| $u_i$, $A_{i+1}=p_{i+1}u_i$, top amplitude $u_{\rm last}$ | line 141 | match |
| TE and TM source projections, output recombination | lines 139, 140, 142 | match |
| $C_{i-1}=C_i\tau_ip_i$; `terms[i - 1][3]` is $p_i$ | line 122 | match |
| $R_{i,E_m,c}$ and $R_{i,H_m,c}$ of Section 4 | lines 110 to 113, 119, 120 | match |
| Source $-(\mathbf P_{i+1}-\mathbf P_i)e^{iqz_i}$, same as the dense right-hand side | lines 149 to 152 | match |

The dispatch at `transition_radiation.py:260` and `:516` happens after
`_checked_kinematics` in the amplitude path and passes the same $\omega$,
$\mathbf k_\perp$ and interface heights. The dense code path is a pure extraction;
the diff to `planar_transition_radiation` moves code without changing an expression.

### Line-by-line review of the C++ kernel

| Item | `transition_radiation_stack_kernel.py` | Result |
|---|---|---|
| $\varepsilon_{j,b}$ at `eps[j * pairs + b]`, C-contiguous `(n_media, B)` | lines 86, 89, 158, 267, 293 | correct |
| Per-pair offsets `b * n_if * 8`, `b * n_if * 2`; per-interface `i * 8` | lines 94, 194, 195, 215, 216 | correct |
| `stack_lambda` branch equals `_outgoing_root` | lines 60 to 64 | correct |
| `stack_step` reads `plus` and `minus` before writing $\rho$; `g[0..3]` do not alias $\rho$ | lines 67 to 75 | correct |
| $Y$ arguments and polarization order: slots 0, 1 TE; slots 2, 3 TM | lines 96 to 98, 171, 174 | correct |
| Phase of medium $i+1$ with $z_{i+1}-z_i$; unit for vacuum | lines 91, 92, 167, 168 | correct |
| Local copies of slots 0 to 3 taken before overwriting | lines 110, 111 | correct |
| Slot 4 of interface $i$ holds $p_{i+1}$; read as `g[-4]` during interface $i+1$, before interface $i$ is expanded | lines 101, 123 | correct |
| `g[-4]` guarded by `i > 0`, so never before the buffer | line 122 | correct |
| $C_{i-1}=C_i\tau_ip_i$ using `trans[2 * i]`, `trans[2 * i + 1]` | lines 99, 100, 124, 125 | correct |
| `g[2 * r + out]` is `[r][c]`; E block $\hat s_c\hat s_r$, $\hat k_c\hat k_r$ | line 114 | correct |
| `g[4 + 2 * r + out]`: TE $\hat s_c\hat k_r$, TM $\hat k_c\hat s_r$ | line 119 | correct |
| Own field: $\phi$, $\mathbf E$, $(\mathbf K\times\mathbf v)_{x,y}$ | lines 134 to 140 | correct |
| Source sign (below minus above) and $e^{iqz_i}$ with real $q$ | lines 161 to 166 | correct |
| TE, TM projections and output recombination | lines 172, 173, 176, 177, 183, 184 | correct |
| Launch arguments in signature order; thread offset; budget split | lines 206 to 213, 245, 248, 274, 300 | correct |

Slot reuse is safe. No kernel defect was found.

## 7. Numerical comparison

The reference is an mpmath solve at 80 to 1500 digits of a Cartesian 4-component
system built from Section 1 (`ref.py` in the verifier's scratch space). It shares no
code with PyRITE. All cases use stacks that the repository tests do not use. Errors
are $\max|\Delta R|/\max|R_{\rm ref}|$ per energy/direction pair.

### Randomized scan

48000 pairs over 6000 stacks, seed 962001. Layer classes: dielectrics, lossy metals,
real low $\varepsilon$ (evanescent), lossless negative, epsilon-near-zero (ENZ),
X-ray-like, vacuum; thicknesses to $10^7$ Å, including $d=0$; energies $0.05$ eV to
$20$ keV; angles to $89.98^\circ$.

| Quantity | Fast minus dense, median | p99 | max |
|---|---|---|---|
| Response | $4\times10^{-16}$ | $1.2\times10^{-12}$ | $1.4\times10^{-8}$ |
| Amplitude | $6\times10^{-16}$ | $2.8\times10^{-10}$ | see below |

No non-finite fast response occurred in this scan.

The amplitude maximum of 4.4 is noise on a true zero: the outliers are films of
$d=0$ between matched substrate and vacuum, where the reference amplitude is below
$10^{-296}$. Excluding those, the 14 largest differences in a sample of 1200 stacks all
have a fast error at or below the dense error against mpmath; they are near Cherenkov poles, where a jump near $10^7$
cancels to an amplitude near $1$. The largest was fast $3.6\times10^{-9}$ against
dense $2.6\times10^{-7}$.

### Limits and symmetry

| Check | Result |
|---|---|
| Rotation about $\hat{\mathbf z}$ by a random angle, no dense solve involved | $1.6\times10^{-15}$ |
| $\varepsilon=0$ layer (oblique), $\varepsilon=0$ substrate (oblique), $\varepsilon=10^{-300}$ (normal) | agrees with dense to $3\times10^{-16}$ |
| $n_z=10^{-7}$ grazing exit | agrees with dense to $3\times10^{-16}$ |
| $k_\perp=0$ exactly | 10% of the randomized scan; GPU equals NumPy to $2.4\times10^{-16}$ |
| Periodic metal films, $\varepsilon$ from $-50+3i$ to $-10^5+10^4i$, $80^\circ$ | fast $\le1.5\times10^{-15}$ |

### Defect 1: removable pole, not a singularity of the dense system

The exact case predicted in Section 5, $\omega=1$, $k_\perp=3/4$, substrate
$\varepsilon=-9/2$, first layer $\varepsilon=1/2$, gives $D_0^{\rm TM}=0$ exactly.

| Stack | Fast | Dense error against mpmath | True $\max\lvert R\rvert$ |
|---|---|---|---|
| 1 layer, 50 Å | NaN | $2.4\times10^{-16}$ | $0.96$ |
| 1 layer, 2000 Å | NaN | $1.2\times10^{-16}$ | $1.03$ |
| 3 layers | NaN | $2.4\times10^{-16}$ | $1.39$ |
| 16 layers, public API | NaN | $1.7\times10^{-15}$ | $2.10$ |

Through the public API at 16 layers, `planar_transition_radiation` returns yield NaN
against dense $2.259\times10^{-3}$. `interface_response` returns NaN; the same holds on
the RTX 3060 Ti through the kernel. NaN also propagates through an opaque absorber:
with the pole under $10^7$ Å of $\varepsilon=-30+2i$ the fast yield is NaN against
dense $5.858\times10^{-4}$, although $p=0$ physically decouples the pole. $D_i=0$ is
any bound mode of the lower half-stack, so TE guided modes should give the same
behaviour; only the TM case was exhibited, because it has exactly representable numbers.

This contradicts the docstring's "Both forms have the singularities of the dense
system" and the matching sentence in the ledger Notes. The dense system has no
singularity here.

The failure is loud, confined to the affected pair, and needs real permittivities
tuned to the last bit. Near the pole the recursion stays accurate. The likely reason
is that the same computed $D_i$ enters $\tau_i$ and $\rho_{i+1}$ and cancels in the next
step:

| Detuning of substrate $\varepsilon$ | $10^{-2}$ | $10^{-6}$ | $10^{-10}$ | $10^{-13}$ | $10^{-15}$ | $3\times10^{-16}$ | $0$ |
|---|---|---|---|---|---|---|---|
| Fast error | $2.3\times10^{-16}$ | $3.7\times10^{-16}$ | $4.3\times10^{-16}$ | $3.3\times10^{-16}$ | $6.9\times10^{-16}$ | $5.1\times10^{-16}$ | NaN |
| Dense error | $6.5\times10^{-16}$ | $4.1\times10^{-16}$ | $8.2\times10^{-16}$ | $7.8\times10^{-16}$ | $7.0\times10^{-16}$ | $5.3\times10^{-16}$ | $5.3\times10^{-16}$ |

### Defect 2: precision loss in $1+\rho$ under a falling admittance chain

When $Y_{i+1}\ll Y_i$ and $|p|\approx1$, $\rho\to-1$ and `plus = 1.0 + rho`
(`transition_radiation_stack.py:83`, kernel line 69) has absolute error near the unit
roundoff $u$. Then $\beta=(1+\rho)/D$ has absolute error near $u/Y_{i+1}$ against a
true size near $1/Y_i$. The relative error therefore grows as
$uY_{\max}/Y_{\min}$ along the chain. The dense LU does not form this difference.
Normal incidence, lossless, thin layers, no pole:

| Substrate / layer / layer / vacuum, $\varepsilon$ | $Y$ span | Fast error | Dense error |
|---|---|---|---|
| $1$, $10^{-4}$, $10^{-8}$ | $10^{4}$ | $2.2\times10^{-13}$ | $6.8\times10^{-16}$ |
| $1$, $10^{-8}$, $10^{-16}$ | $10^{8}$ | $1.0\times10^{-9}$ | $1.6\times10^{-13}$ |
| $1$, $10^{-12}$, $10^{-24}$ | $10^{12}$ | $3.8\times10^{-5}$ | $3.0\times10^{-11}$ |
| $1$, $10^{-16}$, $10^{-32}$ | $10^{16}$ | $0.37$ | $2.4\times10^{-9}$ |

A graded chain with $\varepsilon_{j+1}=10^{-6}\varepsilon_j$ down to about $10^{-290}$
(49 layers) gives fast $\max|R|=1.1\times10^{108}$ against the true $0.999$; dense has
error $10^{-2}$. So $|p|\le1$ does not bound the result, and the response can grow
without any physical cause.

This needs $Y$ spans that real materials rarely reach. At a span of $10^4$ the measured
loss is $2.2\times10^{-13}$. Periodic metal films (table above) show no fast deficit.
Periodic near-critical layers ($\varepsilon=\sin^2\theta\,(1+\delta)$ alternating with
$\varepsilon=2.25$) show none systematically: fast against dense is
$1.3\times10^{-14}$ against $5.6\times10^{-14}$ at $\delta=10^{-4}$,
$2.3\times10^{-12}$ against $5.7\times10^{-13}$ at $10^{-8}$, and
$1.1\times10^{-10}$ against $2.9\times10^{-10}$ at $10^{-12}$.

### $\lambda\to0$ in a layer

Both solvers lose digits together; this matches the shared singularity the docstring
lists. With $\varepsilon=\sin^2\theta\,(1+\delta)$:

| $\delta$ | $10^{-6}$ | $10^{-10}$ | $10^{-13}$ | $10^{-15}$ |
|---|---|---|---|---|
| Fast, 30 Å | $3.7\times10^{-14}$ | $7.1\times10^{-12}$ | $1.2\times10^{-10}$ | $1.9\times10^{-9}$ |
| Dense, 30 Å | $9.6\times10^{-14}$ | $1.1\times10^{-11}$ | $1.1\times10^{-10}$ | $9.6\times10^{-10}$ |

At exactly $\lambda=0$ the two differ in behaviour. The dense solve raises
`LinAlgError`, failing the whole batch, while the fast solve returns NaN for that pair
with no exception. The same split holds for an $\varepsilon=0$ layer at normal
incidence and two adjacent $\varepsilon=0$ layers. At $n_z=10^{-9}$, where the rounded
vacuum $\lambda$ vanishes, both return non-finite values without raising.

### High gain in passive media

For a lossless guided-mode slab tuned to resonance between evanescent cladding
($\varepsilon=4$ core, $\varepsilon=1/2$ barrier, $\sin\theta=0.9$), the true gain is
$e^{\kappa L}$ until double-precision thickness detuning caps it near $5\times10^{7}$:

| $\kappa L$ | $5$ | $10$ | $15$ | $18$ | $30$ | $50$ | $100$ |
|---|---|---|---|---|---|---|---|
| TE true $\max\lvert R\rvert$ | $2.1\times10^{2}$ | $3.1\times10^{4}$ | $4.6\times10^{6}$ | $5.5\times10^{7}$ | $3.5\times10^{2}$ | $1.41$ | $1.41$ |
| TE fast error | $3.8\times10^{-12}$ | $8.3\times10^{-8}$ | $1.8\times10^{-3}$ | $0.30$ | $0.33$ | $1.7\times10^{-7}$ | $3\times10^{-29}$ |
| TE dense error | $4.2\times10^{-13}$ | $9.3\times10^{-9}$ | $2.1\times10^{-4}$ | $0.047$ | $0.048$ | $2.4\times10^{-8}$ | $6\times10^{-17}$ |
| Dense condition number | $2\times10^{5}$ | $5\times10^{9}$ | $10^{14}$ | $2\times10^{16}$ | $2\times10^{16}$ | $3\times10^{16}$ | $2\times10^{17}$ |

TM behaves the same (fast $1.1$, dense $0.27$ at $\kappa L=30$). This regime is
ill-conditioned for any double-precision solver; the fast error runs a factor 2 to 9
above dense. Neither overflows. With 20 coupled cavities at $\kappa L\le30$ the two
are comparable (fast $2.3\times10^{-2}$, dense $8.3\times10^{-2}$). A 40-restart
hill-climb over lossless stacks reached only $|R|=2\times10^{2}$, with fast error
$3\times10^{-12}$. No growth beyond the conditioning was found outside Defect 2.

### Defect 3: subnormal $k_\perp^2$

$k=\sqrt{k_x^2+k_y^2}$ (`transition_radiation_stack.py:63,64`, kernel lines 53, 54)
loses bits when $k_\perp^2$ is subnormal, so $\hat{\mathbf k}$ is not a unit vector.
At $\mathbf n_\perp=(3,4)\times10^{-160}$ the response differs from dense by
$6.5\times10^{-6}$. At $10^{-170}$, where the square underflows to zero, it is correct
to $10^{-16}$, and at $10^{-30}$ too. This is outside any physical range, but it is a
wrong number returned silently. `hypot` would close it.

### Kernel parity

| Run | Worst difference from NumPy |
|---|---|
| Host C++ build, 48000 scan pairs, response | $4.2\times10^{-10}$ |
| RTX 3060 Ti, 20480 pairs to 120 layers, response | $3.0\times10^{-7}$ |
| RTX 3060 Ti, same pairs, amplitude | $4.8\times10^{-7}$ |
| RTX 3060 Ti, median response | $2.9\times10^{-16}$ |
| RTX 3060 Ti, 5000 layers, 4096 pairs, first 64 pairs of the amplitude | $1.8\times10^{-13}$ |

Finite and non-finite patterns matched on all 20480 GPU pairs. The worst GPU response
pair is a $1.2\times10^{4}$ eV pair with very thick layers, where NumPy and the GPU
each sit near $10^{-7}$ of mpmath (NumPy $2.2\times10^{-7}$, GPU $8.8\times10^{-8}$,
dense $2.2\times10^{-7}$). Its layers give $u\sum_j|\operatorname{Re}\lambda_jd_j|=1.5\times10^{-7}$,
so the difference is the phase rounding of large $\lambda d$. The worst GPU amplitude
pair (73 layers) was not adjudicated. The largest one adjudicated, at
$1.2\times10^{-7}$ with 24 layers, has an amplitude $6\times10^{-7}$ from jumps near
$16$, with NumPy $4.1\times10^{-8}$ and GPU $7.8\times10^{-8}$ from mpmath. The host
amplitude maximum of 6.1 is the true-zero case above (NumPy magnitude $10^{-17}$,
rounding noise). Splitting
launches left the response bit-identical. The 5000-layer run took $0.65$ s.

### Cost

NumPy `stack_response` took $38$, $38$, $37$ and $55$ µs per interface per 64 pairs at
$10^2$, $10^3$, $10^4$ and $10^5$ interfaces. Scaling is linear, with memory
pressure at the largest size. The check record
`docs/validation/check-records/optical-stack-cuda.json` (revision `e3a07bce`) is
consistent with this.

## 8. Adjudication

**Filters.** Units pass. The matched, vanishing-film and opaque limits pass.
TE/TM signs, phase references and the layout pass.

**Equations.** The independent derivation matches the NumPy and C++ code term by term.
No factor, sign, phase reference or index diverges.

**Statements that do not survive.**

1. "Both forms have the singularities of the dense system"
   (`transition_radiation_stack.py:37,38`, and the ledger Notes). The recursion also
   has $D_i=0$, a removable pole with no counterpart in the dense matrix. See Defect 1.
2. "only decaying factors enter" (`transition_radiation_stack.py:29`). True of $p$, but
   $\rho$, $\tau$ and $1/D$ are unbounded when $Y$ is not real. The narrower statement,
   that thick absorbing and evanescent layers do not cause overflow, holds: such
   layers give $p\to0$ and no overflow was seen.
3. Agreement with the dense solve is not uniform: Defects 1, 2 and 3 give the exceptions.

Physical cases in the tested range are not affected. These are numerical domain
limits, not physics errors.

**Not checked.** Born and Wolf and Ginzburg and Tsytovich were not opened. The
derivation stands on Maxwell continuity and the dense system, so no equation in this
claim rests on a transcription from them.

**Verdict:** `rederived`, for the algebra and kernel, with the three defects recorded
and the Notes sentence to be corrected. Human sign-off is tracked in #277.

## Addendum, 2026-11-04: propagated rounding estimate

This replaces an earlier addendum on a span guard, which is no longer in the code.
Reviewed at `c763ea3e`. The equations of Sections 1 to 4 are unchanged. The estimate and
the refusal are a numerical guard, not physics. "High precision" below means the dense
Cartesian system of Section 1 solved in mpmath, or, for large $N$ and the scan, the
recursion in mpmath, which agreed with that dense system to $2.5\times10^{-16}$ on 30
random stacks.

**Correction to Section 7.** The double-precision dense solve is not a safe reference for
falling chains at oblique incidence, and Defect 2 compared only at normal incidence. See
the dense-solve subsection below.

### Estimate against actual error

The estimate is $e$. The response uses $u\sum_i\lvert C_i/D_i\rvert/\max_i\lvert C_i\beta_i\rvert$.
The amplitude uses a forward reach over a forward sum of term magnitudes, $S$.

For the class it models, the $1+\rho$ rounding at normal or near-normal incidence, the
estimate is usually above the response error and is at most about 1.5 times below it near
and under the limit (table). Far above the limit it no longer tracks the error (the ENZ
grade, error $10^{108}$), but those pairs are refused.

| Case | Response $e$ | Response error | Result | Amplitude $e$ | Error over $S$ | Error over $\lvert a\rvert$ | Result |
|---|---|---|---|---|---|---|---|
| $1,10^{-8},10^{-16}$ | $5.4\times10^{-9}$ | $1.0\times10^{-9}$ | accepted | $1.5\times10^{-9}$ | $5.5\times10^{-10}$ | $9.5\times10^{-9}$ | accepted |
| $1,10^{-10},10^{-20}$ | $5.4\times10^{-7}$ | $4.1\times10^{-8}$ | refused | $1.5\times10^{-7}$ | $2.2\times10^{-8}$ | $3.7\times10^{-7}$ | refused |
| $1,10^{-16},10^{-32}$ | $0.25$ | $0.37$ | refused | $0.11$ | $0.20$ | $3.4$ | refused |
| $k_\perp=10^{-6}$ chain | $5.5\times10^{-12}$ | $5.7\times10^{-12}$ | accepted | $1.5\times10^{-12}$ | $3.2\times10^{-13}$ | $5.8\times10^{-12}$ | accepted |
| ENZ grade, 49 layers | $10^{27}$ | $10^{108}$ | refused | $3\times10^{82}$ | $10^{85}$ | $10^{85}$ | refused |
| Sawtooth span $3\times10^{7}$, 8 layers | $2.9\times10^{-9}$ | $6.9\times10^{-10}$ | accepted | $3.0\times10^{-10}$ | $8.7\times10^{-12}$ | $1.7\times10^{-10}$ | accepted |
| Same, 89 layers | $2.3\times10^{-8}$ | $3.9\times10^{-9}$ | refused | $3.2\times10^{-10}$ | $4.5\times10^{-11}$ | $2.2\times10^{-9}$ | accepted |
| Same, 8999 layers | $2.4\times10^{-6}$ | $2.5\times10^{-7}$ | refused | $3.3\times10^{-10}$ | $3.5\times10^{-11}$ | $1.3\times10^{-7}$ | accepted |

All ENZ grades at $k_\perp=0$ are refused. The oblique grade is accepted with error
$1.1\times10^{-16}$. On the amplitude, error over $S$ stays at or below $e$ for the
accepted chains in this table, and error over $\lvert a\rvert$ exceeds $e$ by up to about
400 in this table because terms cancel. A stack can therefore pass `planar_transition_radiation` and be
refused by `interface_response`.

### Accepted pairs above $10^{-8}$

The estimate models no other error source. These are accepted, with $e$ below
$10^{-15}$ unless noted.

| Class | Response error | Amplitude | Dense in double precision |
|---|---|---|---|
| Lossless tunnel cavity at resonance, $\kappa L=10$ to $30$ | $8.3\times10^{-8}$ to $1.1$ | same size | $9.3\times10^{-9}$ to $0.31$ |
| Same, TE $\kappa L=18$, core loss $10^{-12}$ | $4.3\times10^{-5}$ | same size | not run |
| Large $\lambda d$ phases, random scan | up to $6.7\times10^{-7}$ | up to $5.0\times10^{-7}$ over $S$ | same large error to the printed digits on the worst pairs compared |
| Thin oblique film, $d=10^{-8}$ | below $e$ | $3\times10^{-8}$ to $7\times10^{-8}$ over $\lvert a\rvert$; over $S$ near $10^{-16}$ | not run |
| Sawtooth at 899 and 8999 layers | refused | $1.2\times10^{-8}$ to $1.3\times10^{-7}$ over $\lvert a\rvert$ ($e$ near $10^{-9}$) | not run |
| Spike $\varepsilon=-10^{18}$ to $-10^{22}$, $d\le10^{-19}$, $k_\perp=0.7$ ($1-\rho$ rounding, unphysical) | $1.6\times10^{-8}$ to $5.0\times10^{-8}$ | up to $2.4\times10^{-7}$ over $\lvert a\rvert$ | not run |

The cavity case is the important one: the estimate normalizes the resonant gain away and
stays near $10^{-16}$ while the result is wrong at full scale. Double-precision dense is
also wrong there, by a factor 2 to 9 less, so this is conditioning and not a defect
peculiar to the recursion.

### Refusal rates

Random scan, seed 962001: $0$ of $48000$ pairs refused on response or amplitude. Largest
$e$ was $1.6\times10^{-9}$. $98$ pairs have response error above $10^{-8}$; $82$ of them
are within 10 times the phase rounding $u\sum_j\lvert\operatorname{Re}\lambda_jd_j\rvert$.
$761$ have amplitude error over $S$ above $10^{-8}$, of which $500$ are within 10 times
that phase rounding. The rest were not classified, except that the worst pairs compared
match dense to the printed digits.

| Set | Response refused or non-finite | Amplitude |
|---|---|---|
| Metal films to $-10^5+10^4i$, to $89.9999^\circ$ | $0$ of $2001$ each | $0$ |
| Grazing, $n_z$ down to $10^{-8}$ | $0$ | $0$ |
| Grazing, $n_z=3\times10^{-9}$ | non-finite: rounded vacuum $\lambda=0$ | non-finite |
| Conductors to $-10^{24}+i$ | $0$ | $0$ |
| Near-critical, $20001$ points, $\pm10^{-2}$ rad | $0$ or $1$ refused; $0$ or $1$ non-finite | $0$ refused; same non-finite |
| Same, $\pm10^{-9}$ rad | up to $36$ refused | $0$ refused |
| Same, $\pm10^{-12}$ rad | $101$ to $113$ refused at most angles; $11236$ at $\varepsilon=1-10^{-5}$; all $20001$ at $1-10^{-6}$ | $0$ refused; up to $351$ non-finite |
| Near-critical, $\operatorname{Im}\varepsilon=10^{-6}$ | $0$ | $0$ |

The false refusals of the span guard are gone. One pair still refuses a whole call, so a
lossless scan crossing a critical angle can lose the whole batch on the response path.

### Kernels

Slots 5 and 6 are written in the bottom-up pass and read before the top-down expansion
overwrites them. Slot 4 of interface $i-1$ is intact when `g[-4]` reads it. No slot-reuse
defect was found. On $15360$ random pairs to 120 layers, the estimate differs from NumPy
by at most $2.9\times10^{-8}$ relative on the GPU and $3.3\times10^{-8}$ on the host,
with $0$ accept or refuse mismatches. The exact pole and the falling chain raise on NumPy
and on CUDA, and `stack_amplitude` now raises at the pole. CuPy pool at 5000 layers and
4096 pairs is $656$ MB, so the materialization cost of the span guard is gone.

### Double-precision dense solve on oblique falling chains

Confirmed, with a formulation that does not use the recursion. The dense system in mpmath
at 100, 400 and 800 digits gives $0.830174890528442-0.922586820530028i$ for
`test_accepted_falling_chain_amplitude_matches_a_high_precision_recursion`. That agrees
with the test's expected value to $1.0\times10^{-15}$ and with the fast solve to
$2.0\times10^{-15}$. Double-precision dense is $20.7\%$ off.

| $\varepsilon$ pair, repeats | $n_x$ | Dense amplitude error | Dense response error | Fast amplitude error | Dense condition number |
|---|---|---|---|---|---|
| $10^{-4},10^{-8}$, 10 | $0.3$ | $1.4\times10^{-9}$ | $1.6\times10^{-12}$ | $1.8\times10^{-15}$ | $2\times10^{5}$ |
| $10^{-6},10^{-12}$, 10 | $0.3$ | $1.0\times10^{-7}$ | $9.5\times10^{-11}$ | $2.4\times10^{-15}$ | $2\times10^{7}$ |
| $10^{-8},10^{-16}$, 10 | $0.1$ | $1.4\%$ | $2.5\times10^{-7}$ | $9.5\times10^{-16}$ | $10^{11}$ |
| $10^{-8},10^{-16}$, 10 | $0.3$ | $20.7\%$ | $2.2\times10^{-8}$ | $2.0\times10^{-15}$ | $2\times10^{9}$ |
| $10^{-8},10^{-16}$, 10 | $0.6$ | $65\%$ | $3.6\times10^{-6}$ | $7.0\times10^{-15}$ | $4\times10^{11}$ |
| $10^{-10},10^{-20}$, 4 | $0.6$ | $196\%$ | $1.1\times10^{-7}$ | $2.4\times10^{-15}$ | $3\times10^{13}$ |
| $10^{-8},10^{-16}$, 20 | $0.9$ | $266\%$ | $9.7\times10^{-7}$ | $1.2\times10^{-14}$ | $10^{11}$ |

The measured range is wider than the 20 to 120% first reported. The mechanism was not
isolated; a response error near $10^{-7}$ multiplying own-field jumps that scale as
$1/\varepsilon$ would be consistent with it. So the dense solve has its own limit here,
and the recursion is the more accurate of the two on these stacks.

### Buried responses

The new statement holds only for lossless stacks, and "no pointwise accuracy" overstates
it. Relative error per response entry against high precision, by depth below the peak:

| Stack | Depth | Fast median | Fast max | Dense max |
|---|---|---|---|---|
| Lossless stop band, 200 layers | $10^{-12}$ to $1$ | at most $6.6\times10^{-16}$ | $3.5\times10^{-15}$ | $1.4\times10^{-14}$ |
| Same | $10^{-16}$ to $10^{-12}$ | $4.1\times10^{-15}$ | $5.5\times10^{-3}$ | $1.1\times10^{-3}$ |
| Same | $10^{-20}$ to $10^{-16}$ | $5.8\times10^{-15}$ | $7.9\times10^{-3}$ | $3.0\times10^{-3}$ |
| Same | $10^{-40}$ to $10^{-20}$ | $1.8\times10^{-3}$ | $0.25$ | $0.16$ |
| Absorbing metal, 40 layers | $10^{-300}$ to $10^{-80}$ | $2.6\times10^{-14}$ | $3.4\times10^{-14}$ | $3.4\times10^{-14}$ |

### Remaining defects

1. "Accuracy is relative to that largest term" and "smaller losses are returned"
   (`transition_radiation_stack.py:46,47`, physics page lines 219 to 221, ledger line 33)
   are false as stated. The estimate covers only $1+\rho$ rounding. Accepted pairs exceed
   it at full scale near lossless resonances, and by up to $6.7\times10^{-7}$ from phases.
2. The formula quoted in the same three places is the response estimate. The amplitude uses
   reach over the sum $S$, which is undocumented (`transition_radiation_stack.py:178,179,181`).
   The response's "largest term" is $\max\lvert C\beta\rvert$, which excludes $C\alpha$.
3. "Propagating it with the amplitude" (`transition_radiation_stack.py:43`) does not
   describe the response estimate, which is a top-down sum.
4. "No pointwise accuracy" (`transition_radiation_stack.py:47,48`, physics page lines 220
   to 221, ledger line 33) is too broad. See the table above.
5. "Agreement with the dense solve" (ledger line 31) is too broad, and "two limits the dense
   solve does not" (physics page line 213) omits the dense solve's own limit above.
6. $1-\rho$ rounding is neither estimated nor documented. It reached $5\times10^{-8}$ only
   in the unphysical spikes above.

**Adjudication.** `rederived` stands for the equations and kernels. The estimate does what it
was built for. Items 1 to 6 are documentation scope for the task owner.
