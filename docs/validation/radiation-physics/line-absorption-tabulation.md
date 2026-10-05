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
`_line_tabulation_grid` unions the 1 eV mesh, basis/absorber Chantler energies,
and absorber EPDL knots. `_epdl_mesh_nodes` adds adjacent float32 values around
each repeated-energy photoelectric edge. The repeated energy itself survives
`np.unique`, so exact float64 edge nodes retain the right-hand value, while the
interval immediately below that node smooths the jump. The discontinuous
interval has width at most one float32 ulp, not one float64 ulp.

`mc_spectrum` delegates setup and evaluation to `_setup.py`, `_batched.py`, and
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

- `pyrite-dev test tests/montecarlo/test_line_absorption_tabulation.py`: 10 passed.
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
