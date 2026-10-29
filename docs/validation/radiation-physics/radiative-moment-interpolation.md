# Radiative moment interpolation

Validation: `radiative-moment-interpolation`.

## Independent derivation

This verifier did not implement this claim. Before reading implementation
bodies, it read the ledger row, function/class/module docstrings, frozen
`checks/equal_precision/moment-policy-788.json`, and the existing independent
[BremsLib partition](bremslib-radiative-partition.md) and
[dielectric suppression](dielectric-suppression.md) derivations. This claim
starts from those ledgered moment definitions; it does not independently
revalidate the original BremsLib data or the bulk-medium approximation.

Let $T>0$ be kinetic energy, $k_c>0$ the photon cutoff,
$K=\min(k_c,T)$, and $\chi_a(T,k/T)$ the interpolated elemental scaled SDCS.
For fixed layer plasma energy $p$, define

$$
F(T,k)=\frac{k^2}{k^2+[(1+T/m_ec^2)p]^2},\qquad C_a=10^{-27}Z_a^2.
$$

For zero plasma energy use $F=1$, including the zero-photon endpoint in
first-moment integrals. The per-atom moments are

$$
S_{s,a}=C_a\int_0^K\chi_a F\,dk,\qquad
R_{h,a}=C_a\int_K^T\frac{\chi_a F}{k}\,dk,\qquad
S_{h,a}=C_a\int_K^T\chi_a F\,dk.
$$

The cross-section conversion gives $R_{h,a}$ in $\mathrm{cm^2}$ and both
first moments in $\mathrm{eV\,cm^2}$. For elemental number densities
$n_a$ in $\mathrm{cm^{-3}}$, layer moments are

$$
\mathbf M(T)=10^{-8}\sum_a n_a(S_{s,a},R_{h,a},S_{h,a}).
$$

Their units are respectively $\mathrm{eV/\mathring A}$,
$\mathrm{\mathring A^{-1}}$, and $\mathrm{eV/\mathring A}$.
Equivalently a density supplied in $\mathrm{\mathring A^{-3}}$ must first
be multiplied by $10^{24}$. At $T\le k_c$ both hard moments vanish;
the soft moment is the full first moment. At all covered energies
$S_s+S_h=S_{\mathrm{total}}$ for the exact reference.

For increasing nodes $T_i$, set

$$
w=\frac{\ln(T/T_i)}{\ln(T_{i+1}/T_i)},\qquad
\widehat{\mathbf M}(T)=(1-w)\mathbf M(T_i)+w\mathbf M(T_{i+1}).
$$

Within the closed node interval $0\le w\le1$: endpoints are reproduced,
nonnegative nodes imply nonnegative components, and zero moments require
no logarithm. Outside coverage there is no convexity guarantee and the
query must be refused. A cutoff node is needed when $k_c$ lies inside
coverage to preserve identically zero hard moments below the cutoff.
The interpolation is linear in moment values, not their logarithms.

The exact total first moment must be checked separately. Componentwise
first-moment errors each bounded by an absolute tolerance do not imply
their sum meets one absolute tolerance. The frozen qualification uses

$$
B_j(T)=a_j+r\lvert M_j(T)\rvert,\qquad
\lvert\widehat M_j-M_j\rvert\le B_j,
$$

with $r=2\times10^{-4}$ and
$(a_s,a_r,a_h)=(10^{-10},10^{-14},10^{-10})$ in the corresponding units.
Closure additionally requires

$$
\lvert\widehat S_s+\widehat S_h-S_{\mathrm{total}}\rvert
\le10^{-10}+r\lvert S_{\mathrm{total}}\rvert.
$$

Adaptive quarter/midpoint probes use one quarter of these budgets;
qualification uses a separate offset logarithmic grid. These are declared
numerical approximation budgets, not source-data uncertainty bounds.
Finite probes provide evidence on the tested grid, not an interval-wide
error theorem. Identity must include table contents, composition, plasma,
cutoff, coverage, projectile, model versions and numerical policy so reuse
cannot silently change the physical reference or approximation.

## Cheap filters before implementation inspection

Units pass: density and length conversion above are required. Limits pass:
endpoint reproduction, zero hard moments below threshold, and a separate
closure budget follow directly. Signs and conventions pass: convex weights
preserve nonnegativity and the kinetic-energy Lorentz factor is
$1+T/m_ec^2$. The approximation alone gives no bound on errors in transport
histories, samplers, or CPU/CUDA execution.

## Source-to-code comparison

`_exact_layer` multiplies packed density in $\mathrm{cm^{-3}}$ by
$10^{-8}$ and sums all three elemental moments. The packer converts
input $\mathrm{\mathring A^{-3}}$ density by $10^{24}$ and computes one
shared plasma energy for all constituents of each layer. These match the
derived quantity and units. `interpolate_moments` computes the convex
combination above in float64. The supported prepared-object evaluator
checks finite physical energies before logarithms and returns physical
endpoint values directly; this prevents logarithmic rounding from admitting
an adjacent out-of-range floating-point energy.

`_refine` uses the declared quarter/midpoint probes and one-quarter budgets.
`_validate` evaluates the fixed offset grid independently of those adaptive
probe positions. Its reference calls the same scalar exact evaluator used
to construct nodes, so this is independent sampling, not an independent
physical model. The tests additionally compare against
`build_radiative_partition`, a separate host moment implementation using the
same tabulated cross sections and physics. That is useful algorithmic
parity; it does not establish independent experimental agreement.

The first reviewed closure check incorrectly summed both absolute
first-moment budgets. The implementer corrected it during verification to
use one `policy.atol[0]`, matching the frozen closure budget. No factor of
two remains in the reviewed denominator.

Identity includes the interpolation-policy identifier, tolerances, maximum
nodes, held-out count, projectile label, cutoff, coverage, composition,
plasma, source keys/digests and BremsLib model marker. The digest additionally
hashes every actual packed quadrature input with shape/dtype and all
prepared grids/logs/moments. Custom data reusing source labels cannot alias
the same digest. Projectile is an explicit identity label: the caller must
supply already scaled positron tables; labeling electron tables as positron
does not perform or validate positron scaling.

The CUDA diagnostic evaluator implements the same convex formula and checks
physical endpoints before taking logarithms. Unsupported energies yield
explicit NaNs rather than a host exception; callers must reject them.
This source review is not hardware execution evidence. The uploaded object
retains the host digest and records its owning CUDA device.

## Numerical evidence and limits

Focused host anchors and documentation checks passed together: 17 tests,
8.64 s. The tests cover compound layers, both suppression settings, two
cutoffs, zero/tiny rates, endpoint coverage refusal, and identity changes.

An additional verifier-run calculation replaced the prepared elemental
scaled SDCS by a constant 10 mb. It did not use either partition helper to
calculate reference moments. For constant $\chi$, vacuum references are
proportional to $(k_c,\ln(T/k_c),T-k_c)$. With dielectric scale $a$, use

$$
G(k)=k-a\arctan(k/a),\qquad
(S_s,R_h,S_h)\propto
\left(G(k_c),\frac12\ln\frac{T^2+a^2}{k_c^2+a^2},G(T)-G(k_c)\right).
$$

The proportionality factor includes 10 mb, atomic number squared, cross-
section conversion and number density. Plasma energy was calculated from
SI constants as $\hbar\sqrt{n_e e^2/(\epsilon_0m_e)}$, with
$n_e=14\times0.05\times10^{30}\ \mathrm{m^{-3}}$. For Si density
$0.05\ \mathrm{\mathring A^{-3}}$, a 1 keV cutoff and kinetic energies
10, 13.721, 37.373, 113.171 and 200 keV, all ten vacuum/suppressed cases
passed. The largest component error was 0.114735 of its mixed budget;
physical endpoints matched the analytic moments at relative tolerance
$2\times10^{-13}$. This is a fresh analytic numerical check, not a retained
regression artifact or a released-table transport benchmark.

No CPU/CUDA runtime parity, production integration, ensemble equivalence,
transport speedup, or complete-request equal-precision speedup is certified
by these host component checks.

The first reviewed preparation path inherited a relative $10^{-12}$
table-endpoint admission tolerance from the existing packer. A verifier
probe confirmed that both adjacent floating-point energies outside the
source table were accepted. The implementer added strict physical table
bounds in this diagnostic preparation path and regression checks for both
adjacent outside endpoints. This resolves the stated no-clamping coverage
contract without changing the existing packer contract.

## Verdict

- **Claim:** `radiative-moment-interpolation` —
  `montecarlo/transport/radiative_moments.py::prepare_radiative_moments`,
  `::interpolate_moments`, and the diagnostic device interpolant — existing
  ledgered BremsLib/dielectric moment definitions and convex interpolation
  in log kinetic energy.
- **Filters:** units pass; limits pass; signs/conventions pass. Reference
  transcriptions: no new external tabulated measurements. Tolerances:
  predeclared numerical approximation budgets. External-code and hardware
  evidence: none claimed for this prototype by this verification.
- **Re-derivation:** matches — no remaining divergent factor, sign,
  exponent, unit, or interpolation convention after correction of the
  closure absolute budget and strict source-table coverage checks.
- **Verdict:** `rederived` for the interpolation equations and host
  component scope. Device source agrees symbolically; runtime parity remains
  a separate hardware qualification gate.
- **Write-up:**
  `docs/validation/radiation-physics/radiative-moment-interpolation.md`.
- **Suggested ledger change:** set `rederived`; `anchored` is supported for
  the host synthetic component scope by the passing named regression module.
  Keep hardware/released-table/ensemble qualification explicitly pending.
  Only a human may set `signed-off`.

Strict admission exposed a separate fixture representation detail: staging
stored the synthetic nominal 10 keV endpoint as
$10.000000000000004\ \mathrm{keV}$. The test fixture now explicitly sets
its two intended endpoints to 10 and 200 keV; the implementation retains
strict comparison against the actual staged table interval. Released-table
diagnostics must derive coverage from the intersection of actual staged
elemental intervals. The implementer reports the corrected eight-test
component module passes in 8.56 s. The analytic constant-SDCS calculation
above is also being retained as a regression anchor by the implementer.

Final focused rerun, explicitly including the four slow parametrizations:
`pyrite-dev test --slow tests/montecarlo/test_radiative_moments.py tests/dev/test_docs.py`:
19 passed in 7.17 s. This includes the retained
`test_constant_sdcs_against_analytic_moments` anchor for both suppression
settings. The author owns the complete docs build and rendered-math check.
