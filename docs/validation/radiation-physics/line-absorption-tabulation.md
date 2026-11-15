# Line absorption tabulation

Validation: `line-absorption-tabulation`.

## Independent derivation (2026-10-05)

This section was written before inspecting implementation bodies or the earlier
Chantler derivation. Inputs were the ledger claim, owner signatures/docstrings,
and the EPDL2025 MF=23 native lin-lin convention. The source is
[Cullen's EPDL2025 ENDF data](https://nuclear.llnl.gov/EPICS/ENDF2025/EPDL2025.ALL)
{cite:p}`epdl2025`; MF=23 alone does not imply a universal interpolation law:
the TAB1 interpolation declaration of the selected data supplies that law.

For element $i$, let $n_i$ be its atomic number density in
$\mathrm{\mathring A}^{-3}$ and $\sigma_i(E)$ its narrow-beam removal cross
section in barn. Photoelectric, coherent, incoherent, and both pair channels
contribute. On a native smooth interval each channel is linear in energy.
Their sum, on the union of all channel knots, is therefore also linear:

$$
\mu_i(E)=10^{-8} n_i\sigma_i(E)=a_i+b_iE,
\qquad \mu(E)=\sum_i\mu_i(E).
$$

Here $1\,\mathrm{barn}=10^{-8}\,\mathrm{\mathring A}^{2}$, so
$\mu_i$ has units $\mathrm{\mathring A}^{-1}$. Composition entries represent
number-density contributions, not an additional weighting to apply after this
conversion. Assume finite positive elemental totals, positive photon energies,
a homogeneous passive medium, and narrow-beam removal without scattered-photon
build-up or secondary return. Layered/grooved geometric integration is outside
this table claim.

For a table interval $[E_0,E_1]$ and $E_0\le E\le E_1$, define

$$
x=\ln E,\qquad h=\ln(E_1/E_0),\qquad
f=\frac{\ln(E/E_0)}{\ln(E_1/E_0)}.
$$

Tabulating the elemental logarithms gives the positive approximation

$$
\boxed{\widehat\mu(E)=\sum_i
\exp\big[(1-f)\ln\mu_i(E_0)+f\ln\mu_i(E_1)\big]}.
$$

The logarithm implicitly uses $1\,\mathrm{\mathring A}^{-1}$ as its reference
unit. That reference cancels on exponentiation. Interpolation of the logarithm
of the compound sum instead defines a different approximation. Elemental
interpolation followed by summation preserves elemental node values and
bounds compound relative error by the largest elemental relative error.

To derive the smooth-interval error, write $g_i(x)=\ln(a_i+b_ie^x)$ and
$p_i=b_iE/\mu_i(E)$. Then

$$
g_i''(x)=\frac{a_ib_iE}{(a_i+b_iE)^2}=p_i(1-p_i),
\qquad
\ln\frac{\widehat\mu_i(E)}{\mu_i(E)}
=\frac{g_i''(\xi)}{2}(x-x_0)(x_1-x),
$$

for some $\xi$ inside the interval. Consequently

$$
\left|\ln\frac{\widehat\mu_i}{\mu_i}\right|
\le\frac{h^2}{8}\sup|p_i(1-p_i)|,
\qquad
\left|\frac{\widehat\mu_i}{\mu_i}-1\right|
\le\exp\left(\frac{h^2}{8}\sup|p_i(1-p_i)|\right)-1.
$$

Thus the error is second order in relative energy spacing on a positive smooth
native segment, not exactly zero. The bound requires every native channel knot
in the shared mesh; it cannot be applied across a discontinuity. It is not a
global tolerance guarantee without a bound on curvature. A positive decreasing
segment with positive intercept has negative curvature in log coordinates and
is underestimated by the log-log chord. Constant coefficients are exact.

For path length $L$ in angstroms, the independent attenuation result is

$$
\tau=L\widehat\mu(E),\qquad T=\exp(-\tau),\qquad
\frac{\widehat T}{T}=\exp[-L(\widehat\mu-\mu)].
$$

Units and signs pass: optical depth is dimensionless; $0<T\le1$ for
nonnegative $L$; $L=0$ gives exactly $T=1$; increasing positive $L$ decreases
transmission; the infinite-path limit is zero. Coherent fields use
$\exp(-\tau/2)$ so their intensity uses the same attenuation. Small coefficient
error need not imply small relative transmission error at large optical depth.

At an exact table node the mathematical interpolation returns that node's
coefficient; finite arithmetic may introduce log/exp rounding. Outside the
table interval, endpoint clamping is a numerical policy, not EPDL
extrapolation, and must be stated explicitly.

At a right-continuous photoionization threshold $E_b$, select adjacent float32
values $E_-<E_b\le E_+$ and sample the native source separately at both.
No float32 query lies strictly between these values, so a float32 table cannot
smooth the jump at another representable query. Float64 queries can lie inside:
interpolation deliberately smooths the discontinuity over that one float32-ulp
interval, so exact native right-continuity is not recovered there. An exact
threshold node added separately would recover the right-hand value but would
change this policy. Evaluate the fraction as

$$
f=\frac{\ln(1+(E-E_0)/E_0)}{\ln(1+(E_1-E_0)/E_0)}
$$

using `log1p` to retain the separation of adjacent float32 nodes. This preserves
small differences in the fraction; it does not remove float32 rounding of
stored logarithmic coefficients.

## Implementation comparison and evidence

The implementation matches the boxed elemental interpolation and the stable
fraction. `_elemental_log_mu_table` samples EPDL totals per composition entry;
`_interp_elemental_mu` exponentiates each blended row before summing.
`_cxr_tabulation_grid` unions the edge-resolved base mesh `_line_mesh` (spacing
`max(1 eV, 1e-3 E)`, with the 1 eV mesh kept within 3 % of every basis/absorber
EPDL edge; #1201), basis/absorber Chantler energies, and absorber EPDL knots. `_epdl_mesh_nodes` adds adjacent float32 values around
each repeated-energy photoelectric edge. The repeated energy itself survives
`np.unique`, so exact float64 edge nodes retain the right-hand value, while the
interval immediately below that node smooths the jump. The discontinuous
interval has width at most one float32 ulp, not one float64 ulp.

`mc_cxr_spectrum` delegates setup and evaluation to `_setup.py`, `_batched.py`, and
`_per_hkl.py`. Setup builds coefficients on the host once and casts the mesh and
log rows to `REAL`. The batched gather and per-reflection route use a common
bracket and the log fraction. The streaming CUDA prologue computes the same
`log1p` fraction; its scalar elemental gather performs the same exponential
and sum with endpoint branches. Attenuation enters the incoherent segment-mean
transmission or coherent formation optical depths, using the same coefficient.
This is source-flow agreement, not hardware numerical parity. No GPU runtime
or spectrum-level exact-versus-tabulated A/B was executed.

A stale comment in `_setup.py` still describes exact reproduction of the
retired Chantler rule. The current attenuation table is EPDL and the smooth
submesh interpolation is approximate. Likewise `_interp_elemental_mu`'s
phrase “no mesh interval straddles ... a jump” needs the explicit one-ulp edge
exception stated in the ledger. These are documentation findings, not divergent
arithmetic in the declared approximation.

## Independent numerical evidence

The upstream EPDL2025 tape was downloaded directly and SHA-256 verified as
`59bbd8c559685dda0bf0de2762bc43126f599cd154d635940f17b6a59c1c43fd`.
An independent fixed-column reader extracted MF=23 MT=522, 502, 504, 517,
and 515 for C, Mo, S, and Se; every extracted TAB1 interpolation declaration
was law 2 (lin-lin). The reference interpolated native channel arrays with
right-hand edge selection, summed cross sections, and multiplied number
densities by the independent barn-to-square-angstrom conversion. It did not
call PyRITE's EPDL interpolation or attenuation helpers. Shared-grid geometric
midpoints were compared in the production bands below; the implementation
under test supplied only the table/grid and final gathered result.

| Material / band [eV] | Maximum smooth relative difference from native tape | Energy [eV] |
| --- | --- | --- |
| HOPG / 100–1500 | $4.8798246\times10^{-4}$ | 919.499864 |
| MoS2 / 350–3500 | $4.8600929\times10^{-4}$ | 2020.499938 |
| MoSe2 / 350–3500 | $4.7493922\times10^{-4}$ | 1848.499932 |

These differences include EPDL dataset knot thinning, whose separate stored
relative tolerance is $5\times10^{-4}$, and float32 storage of native cross
sections. Therefore the existing $2\times10^{-4}$ anchor tolerance against
`_mu_total_inv_ang` validates the additional line-table interpolation error
against the processed dataset; it is not a bound against the unthinned tape.
The “every EPDL knot” argument refers to every retained runtime knot. The
second-order derivation applies to those processed linear segments, while
thinning has its own error budget.

Edge-containing intervals were deliberately measured separately. At the C K
edge, the interval is $[287.9999694824219,288.0]$ eV, width
$3.0517578125\times10^{-5}$ eV. At its geometric midpoint
$287.99998474121054$ eV, native attenuation is
$5.594189026881321\times10^{-5}\,\mathrm{\mathring A}^{-1}$ and tabulated
attenuation is $2.5879007724789396\times10^{-4}\,\mathrm{\mathring A}^{-1}$:
relative error $3.6260517$ (362.6%). MoS2 and MoSe2 maximum edge-midpoint
relative errors are 43.45% and 39.70%; their maximum discontinuous-interval
width is $2.44140625\times10^{-4}$ eV. These large local errors are the declared
one-float32-ulp smoothing policy, not failures of the smooth-interval bound.
At $L=10000\,\mathrm{\mathring A}$ the carbon example gives approximately
$T=0.572$ natively and $\widehat T=0.0752$; coefficient errors can be amplified
in transmission even over a very narrow energy interval.

Focused checks, using the canonical runner and isolated environment:

- `pyrite-dev test tests/montecarlo/spectrum/test_line_absorption_tabulation.py`: 10 passed.
  Covers processed-EPDL midpoint tolerance, float32 host arithmetic, node
  identity, positivity, explicit Si absorbers, endpoint clamps, zero-path
  transmission, and neighboring table-ceiling behavior.
- `pyrite-dev test tests/dev/test_docs.py`: 6 passed.
- Sphinx build and positive rendered-math inspection are performed by the task
  owner after integration; host float32 checks do not establish CUDA parity.

## Validation state

Fresh-context re-derivation: **matches** the declared EPDL log-log submesh
approximation, elemental summation, attenuation units/sign, exact-node and
endpoint policies, and the explicit one-float32-ulp edge exception. Verdict:
**rederived**. The native-data comparison above independently quantifies
thinning plus interpolation and the narrow edge exception. Suggested ledger
update: advance to `rederived`, replace retired Chantler verification notes
with this evidence, distinguish processed-table and native-tape tolerances,
and retain GPU runtime and whole-spectrum comparison as untested. Only a
human may mark `signed-off`.

## Fresh-context re-verification of the #1201 mesh (2026-11-14)

Scope: the base-mesh edit of issue #1201 only. The elemental log-log rule,
composition sum, units and edge-pair policy derived above are unchanged and
were not re-derived. The error estimates below were written from the task
statement before reading `_tabulation_mesh.py`.

### Independent error estimate

For a function $y(E)$ with continuous second derivative on a mesh interval of
width $h$, the linear interpolant errs by at most

$$
\lvert y-\widehat y\rvert\le\frac{h^2}{8}\max\lvert y''\rvert ,
$$

attained at the midpoint for constant $y''$. Away from edges the
susceptibility has the form $\chi_{\mathbf g}(E)=-C\,E^{-2}F(E)$ with
$F=f_0(\mathbf g)+f'(E)+if''(E)$. If $F$ is locally constant,
$y''/y=6/E^2$, so

$$
\frac{\lvert\chi-\widehat\chi\rvert}{\lvert\chi\rvert}
\le\frac{3}{4}\left(\frac{h}{E}\right)^2
=7.5\times10^{-7}\quad\text{at } h/E=10^{-3}.
$$

The same holds for $\delta=1-\operatorname{Re}n\propto E^{-2}[Z+f'(E)]$.
A component carried only by $f''\propto E^{-p}$ behaves like $E^{-(2+p)}$.
For $p=2$, the prefactor is $(4\cdot5)/8=5/2$, giving $2.5\times10^{-6}$.
Slow variation of $f'$ adds $h^2F''/(8F)$, which is still second order in $h$.
For $\mu$, the elemental bound above becomes
$h_{\ln}^2\sup\lvert g''\rvert/8$ with $h_{\ln}=\ln(1+10^{-3})$.
A pure power law is exact.

The estimate fails in two cases. It fails across a slope discontinuity of a
tabulated source, which is why native Chantler and EPDL knots remain mesh
nodes. It also fails where $F$ has structure on a scale comparable to $h$.
xraydb's $f_1$ is an interpolating cubic spline over the Chantler nodes, so
its curvature near edges follows the dense Chantler edge sampling rather than
$E^{-2}$. Keeping the former 1 eV nodes within $\pm3\%$ of every edge
reproduces the old mesh's resolution where this structure is largest. For
edges above 1 keV, the window half-width is at least 30 eV. That exceeds the
few-eV differences between Chantler and EPDL edge energies, so EPDL edges are
an adequate window locator. Below 1 keV the whole mesh is the 1 eV mesh. The
window is a sufficient design only where spline structure is confined to
$\pm3\%$ of an edge. This is a data property, so it was tested numerically.
A weak reflection whose $F$ is a near-cancelling difference amplifies any
residual by $\lvert f\rvert/\lvert F\rvert$.

### Implementation comparison

`_line_mesh` matches the stated construction:
`np.arange(lo, min(hi, switch) + 1, 1)` up to `switch = max(lo, 1000)`,
then `switch * 1.001**k`; 1 eV nodes `lo + k` over
$[\max(lo,0.97E_e,\text{switch}),\min(\text{top},1.03E_e)]$ for each edge
$E_e$; truncation at `top`, the first node at or above `hi`.
`_cxr_tabulation_grid` takes the edges of basis and absorber elements and
unions them with Chantler, EPDL-knot and float32 edge-pair nodes as before.
Inside an edge window the mesh is a strict superset of the former uniform mesh
plus native nodes, because the geometric nodes remain as well.

### Numerical spot checks

Independent scratch probes (`scratch/verify_1201_mesh.py`,
`verify_window.py`, `verify_anchor_ext.py`, `verify_scaling.py` in the task
worktree) used `np.interp` and an independent per-element log-log blend
rather than the production gather helpers. Queries were the 1 eV-mesh
midpoints $E=1.5,\dots,359\,999.5$ eV. The excess was
$\lvert\text{err}_\text{new}\rvert-\lvert\text{err}_\text{1 eV}\rvert$,
divided by $\max\lvert y\rvert$ within $\pm2\%$, with no absorber other than
the basis.

| Crystal (reflections) | $n_\text{tab}$ old → new | max excess, $\chi$ re / im | $\delta$ | $\mu$ |
| --- | --- | --- | --- | --- |
| Si (111), (400) | 362 299 → 9 322 | $2.4\times10^{-5}$ / $1.15\times10^{-3}$ (1838.5 eV) | $9.7\times10^{-6}$ | $1.47\times10^{-6}$ |
| MoS2 (002), (110) | 363 096 → 11 694 | $9.6\times10^{-6}$ / $2.3\times10^{-6}$ | $4.5\times10^{-6}$ | $1.37\times10^{-6}$ |
| W (110), (200) | 362 224 → 15 837 | $2.7\times10^{-5}$ / $2.5\times10^{-6}$ | $1.29\times10^{-5}$ | $1.30\times10^{-6}$ |
| Ge (111), (220) | 362 288 → 10 062 | $8.9\times10^{-5}$ / $4.4\times10^{-5}$ | $6.6\times10^{-6}$ | $1.44\times10^{-6}$ |
| GaAs (111), (200) | 363 116 → 11 701 | $3.16\times10^{-4}$ (200, 2210.5 eV) / $5.4\times10^{-5}$ | $4.0\times10^{-6}$ | $1.33\times10^{-6}$ |
| Pb (111), (200) | 362 396 → 17 859 | $1.5\times10^{-5}$ / $2.1\times10^{-6}$ | $1.07\times10^{-5}$ | $1.30\times10^{-6}$ |

Without edge windows, the purely geometric arm reaches excesses of
$5\times10^{-3}$ to $1.0\times10^{-1}$ in $\chi$. The largest are GaAs (200)
and Ge (220) $\chi_\text{im}$ at the Ga and Ge K edges. These confirm that
the windows are necessary.

Scaling: above the Si K edge (5–300 keV, Si (111)), the maximum relative
errors at $h/E=1,2,4\times10^{-3}$ are $8.6\times10^{-7}$,
$3.4\times10^{-6}$ and $1.36\times10^{-5}$ in $\chi_\text{re}$. For $\delta$
they are $7.8\times10^{-7}$, $3.1\times10^{-6}$ and $1.23\times10^{-5}$.
This is exactly quadratic and within 15 % of the
$\tfrac34(h/E)^2$ prefactor. The $f''$-only Si (400) $\chi_\text{im}$ reaches
$2.5\times10^{-6}$ at 40 keV, as predicted by the $5/2$ prefactor.

Findings:

1. **In-window excess from the added geometric nodes.** Si (111)
   $\chi_\text{im}$ at 1838.5 eV, inside the Si K window, has interval
   $[1838.0325,1838.8157]$, where 1838.0325 is a geometric node. Its error is
   $1.16\times10^{-7}$, compared with $4.2\times10^{-8}$ on the old mesh. The
   absolute excess is $7.4\times10^{-8}$; the local $\lvert\chi_\text{im}\rvert$
   is $1.7\times10^{-6}$ near a sign change, and the scaled excess is
   $1.15\times10^{-3}$. Adding a node does not reduce the error of a function
   with an inflection inside the interval. Removing geometric nodes inside
   the windows makes every in-window excess exactly zero (Si (111),
   GaAs (200)). The claimed bound, $\chi_{\mathbf g}\le1.6\times10^{-4}$ with
   Si $5\times10^{-5}$, is therefore not reproduced with the stated
   1 eV-midpoint queries. The implementer's probe queried midpoints of the
   old full grid and used log-binned scales, which explains the difference.
2. **Off-edge spline structure in a weak reflection.** GaAs (200), where
   $F\propto f_\text{Ga}-f_\text{As}$, has a scaled off-edge error of
   $4.05\times10^{-4}$ at 2210.5 eV, 2.4 times the uniform-mesh value. Ga
   $f_1''$ has a localized $4.8\times10^{-4}\,\mathrm{eV}^{-2}$ peak at the
   Chantler node 2211.77 eV, far from any edge. Its error,
   $h^2\lvert f_1''\rvert/8\approx2.9\times10^{-4}$ electron, is amplified by
   the cancellation. The absolute $\chi$ error is about $1.4\times10^{-9}$.
   Ge (111) $\chi_\text{im}$ reaches $1.1\times10^{-4}$ scaled off-edge at
   1314.5 eV, between the L1 and L2 windows. The anchor's pure-relative
   `rtol=1e-4` off-edge test passes for Si, MoS2, W, Pb, MoSe2 and HOPG. Ge
   and GaAs would fail it, at $3.5\times10^{-2}$ and $2.4\times10^{-2}$, but
   only at zero crossings of $\chi_\text{im}(111)$ and $\chi_\text{re}(200)$.
   Pointwise relative error is undefined there, so this is not evidence of a
   defect.
3. **Prefix property.** For every `lo` in a random scan (300 values,
   250 `hi` each), `_line_mesh(lo, hi)` equals the leading part of
   `_line_mesh(lo, 3e5)`, except for $999<\text{hi}<1000$ with non-integral
   `lo`. In that range, the short mesh ends at the 1 eV node `lo + k` and
   lacks the geometric start node 1000.0. The docstring's “every node
   depends on `lo` but not on `hi`” is therefore false in that range. This
   does not affect `line-tabulation-nan-ceiling`, whose cut is a Chantler
   endpoint of at least 966 keV. It also does not affect seeding/kernel table
   agreement, which share `hi` below 1 keV.

$\mu$ behaves as derived. The maximum excess is $\le1.47\times10^{-6}$,
dominated by smooth log-log curvature. The windows do not affect it, because
EPDL knots are mesh nodes.

### Root-support compatibility

With `use_henke=True`, the setup ceiling is a native Chantler endpoint $C$.
That endpoint is a node of both meshes. Native, EPDL and edge-pair nodes are
filtered to $[lo,hi]$, while the base mesh is cut at the first node at or
above $hi$. The mesh at or below $C$ is therefore identical for $hi=C$ and for
any larger $hi$, and $C$ itself remains the NaN boundary node. The anchor
`test_setup_bound_preserves_root_support_and_native_mesh_prefix` exercises
this on the geometric part of the mesh (HOPG, 950–980 keV) and passes, along
with the final-update crossing test.

### Anchors

`pyrite-dev test tests/montecarlo/spectrum/test_line_absorption_tabulation.py`
gives 39 passed. The tests pin the spacing bound, the presence of the 1 eV
window, the prefix (only for `lo = 1`), the off-edge `rtol=1e-4` bound for
Si/MoS2/W over 1–40 keV, and the existing EPDL $\mu$ midpoint bound of
$2\times10^{-4}$, which now runs on the new mesh up to 3.5 keV. They pin the
claimed construction and the off-edge order. No anchor pins in-window excess
over the old mesh, a non-integral `lo`, or weak reflections.

### Adjudication

The mesh construction, the interpolation rules and the root-support prefix
match the claim. The second-order error law and its prefactor are confirmed.
The derivation finds no divergence in physics or convention. The numerical
evidence in the ledger Checks overstates the $\chi_{\mathbf g}$ bound. The
correct stated maximum is $1.15\times10^{-3}$ of the local scale,
$7.4\times10^{-8}$ absolute, at an Si K-edge sign change. GaAs (200) reaches
$3.2\times10^{-4}$ off-edge. The Checks field must state these values instead
of a $1.6\times10^{-4}$ bound. Verdict: **rederived**, conditional on that
correction. Only a human may mark `signed-off`.

### Owner resolution (code 43d0bae9f)

The owner fixed findings 1 and 3 in code rather than restating the bounds:

- Inside every edge window, `_line_mesh` now drops the geometric nodes, so
  there the mesh is exactly the former 1 eV mesh `lo + k` (plus the same
  native nodes). The in-window excess is therefore zero by construction.
- Every node is now fixed by `lo` and the edges alone. The 1 keV switch node
  is always a mesh node, and windows are generated whole before the cut at
  the first node at or above `hi`. The prefix property holds for any `lo`.

New anchors pin both: in-window equality with the 1 eV mesh, and the prefix
property over non-integral `lo` and random `hi`, including
$999<\text{hi}<1000$.

The verifier's probe, re-run on the fixed mesh with the same query set
(`scratch/verify_final.log` in the task worktree):

| Crystal | $n_\text{tab}$ old → new | max excess, $\chi$ re / im | $\delta$ | $\mu$ |
| --- | --- | --- | --- | --- |
| Si | 362 299 → 9 261 | $2.4\times10^{-5}$ / $2.8\times10^{-5}$ | $9.7\times10^{-6}$ | $1.47\times10^{-6}$ |
| MoS2 | 363 096 → 11 452 | $9.6\times10^{-6}$ / $2.3\times10^{-6}$ | $4.5\times10^{-6}$ | $1.37\times10^{-6}$ |
| W | 362 224 → 15 334 | $2.7\times10^{-5}$ / $2.5\times10^{-6}$ | $1.29\times10^{-5}$ | $1.30\times10^{-6}$ |
| Ge | 362 288 → 9 856 | $1.1\times10^{-5}$ / $4.4\times10^{-5}$ | $6.6\times10^{-6}$ | $1.44\times10^{-6}$ |
| GaAs | 363 116 → 11 330 | $3.16\times10^{-4}$ (200, 2210.5 eV) / $5.4\times10^{-5}$ | $4.0\times10^{-6}$ | $1.33\times10^{-6}$ |
| Pb | 362 396 → 17 354 | $1.5\times10^{-5}$ / $2.1\times10^{-6}$ | $1.07\times10^{-5}$ | $1.30\times10^{-6}$ |

Finding 2, the GaAs (200) off-edge Chantler spline curvature amplified by
cancellation, remains. It is a data property away from any edge, and the
absolute error is $1.4\times10^{-9}$. It is recorded in the ledger Checks
rather than given its own window.
