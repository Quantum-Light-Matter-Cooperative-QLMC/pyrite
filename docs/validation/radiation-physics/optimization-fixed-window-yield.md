# Fixed-window directional PXR/CBS yield

## Independent derivation before implementation inspection

Claim: `optimization-fixed-window-yield`, owned by
`src/pyrite/optimization/scoring.py::FixedWindowYield`.
This fresh verifier read the ledger row, class docstring, and
[energy-grid semantics](../../physics/radiation-physics/energy-grid-semantics.md),
particularly `eq-grid-physical-integral` and the density-units section, before
reading the implementation body. This section was written first.

The intended input is one scalar run record from the nested `run_sweep` store.
The signature is `FixedWindowYield(lower_eV: float, upper_eV: float,
name: str = 'window_yield')`, returning one named scalar from `__call__`.
The selected separate incoherent PXR/CBS density is already normalized per
incident electron. No sampled-electron count, beam current, detector response,
or solid-angle multiplication belongs in this reduction.

Let strictly increasing positive evaluation nodes be $E_i$, with finite signed
density samples $s_i$ in photons per eV per sr per incident electron. For
$E_i \le E \le E_{i+1}$, the stated interpolant is

$$
s(E)=s_i+\frac{s_{i+1}-s_i}{E_{i+1}-E_i}(E-E_i).
$$

Choose finite positive fixed limits $a<b$ inside the closed sampled range.
Form the ordered partition $x_0=a$, all strictly interior original nodes, and
$x_m=b$. Interpolate the endpoint values using the enclosing original segments;
retain each interior sample. The source physical-energy integral then gives

$$
\boxed{Y_{[a,b]}=\int_a^b s(E)\,\mathrm{d}E
=\sum_{j=0}^{m-1}\frac{s(x_j)+s(x_{j+1})}{2}(x_{j+1}-x_j).}
$$

This is exact for the specified piecewise-linear interpolant on uniform,
nonuniform, or logarithmically placed nodes. The widths remain physical eV;
there is no logarithmic-coordinate weighting. The source full-range local-node
weights are the regrouping of these trapezoids when limits coincide with the
outer nodes. Inserting the actual fixed limits preserves partial edge intervals.

## Cheap filters

- Units: pass. Multiplication by physical eV leaves photons per sr per incident
  electron. No extra normalization factor is allowed.
- Limits: pass. For $s_i=c$, $Y=c(b-a)$; for zero samples, $Y=0$. An affine
  density integrates to its analytic antiderivative independent of node spacing.
  As $b$ approaches $a$, the integral approaches zero for bounded density.
- Signs/conventions: pass. The functional is linear, hence reversing the density
  sign reverses the yield; signed Monte Carlo estimates must not be clipped.
  Positive node samples give a nonnegative integral. This is a directional
  source quantity, not a detector-accepted photon count.

Bin-mean densities do not supply node values for this interpolant and must be
refused. A two-beam joint-field correction does not represent the stated separate
PXR/CBS source component and must also be refused. Characteristic, continuum,
coherent, channeling, and transition arrays are outside this claim.

## Independent analytic reference points

For nodes $(100,200,500)$ eV and density $(1,3,9)$, the density is globally
$s(E)=0.02E-1$. Limits $(150,400)$ eV therefore give

$$
Y=[0.01E^2-E]_{150}^{400}=1125.
$$

For constant density $c=2$ over the same limits, the result is $500$; for its
negative it is $-500$. These references come directly from antiderivatives,
without using an implementation quadrature helper.

## Implementation comparison

The implementation matches the derived expression. `FixedWindowYield.__call__`
flattens the store with `records`, requires exactly one record, and selects only
`E_grid` and `spec`. Its finite-density and shape checks precede the helper.
`radiation_band_metrics` requires at least two finite strictly increasing nodes
and refuses bounds outside the sampled range. It inserts the two requested
limits and all strictly interior nodes, evaluates `np.interp`, and applies
`np.trapezoid` with those physical-energy coordinates. This is precisely the
boxed partition formula; there is no divergent factor, sign, or convention.

The scorer supplies only the selected line array to the helper and returns
`yields['line']`. Thus the helper's optional combined-component and physical
population metadata do not enter this scalar. The two-beam guard recognizes
both string and mapping forms. Any non-None `line_quadrature` is rejected;
canonical sweep construction represents node sampling by omission and writes
only `bin-mean` as a divergence key, so canonical node records are accepted.
An arbitrary hand-built record explicitly spelling `node` is outside this
canonical record encoding and is refused.

Independent numeric calls after implementation inspection returned 1125 for
the affine reference, 500 for constant 2, -500 for constant -2, and 0 for zero,
matching the independently written antiderivatives exactly.

## Evidence and scope filters

| Filter | Result | Evidence or scope |
| --- | --- | --- |
| Notation and normalization | pass | Ledger, source density units, class docstring and `units` agree: photons per sr per incident electron after integration. |
| Boundary units | pass | Input physical eV times photons per eV per sr per incident electron; output has no extra current, detector or electron-count factor. No loader or kernel is changed by this claim. |
| Reference transcription | not applicable | No external numerical table or digitized source values are used; reference densities are analytic. |
| Tolerance | pass | Existing constant/affine/zero/signed regression anchors use exact equality for exactly represented analytic values, without fitted tolerance. |
| External-code / hardware parity provenance | not applicable | This claim makes no external-code, CPU/GPU, or precision-parity assertion. |
| Component and refusal contracts | pass | Tests isolate auxiliary spectra and scaling fields, reject bin means and both joint-field encodings, and reject uncovered/nonfinite/malformed inputs. |

Verification: `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test
 tests/optimization/test_scoring.py tests/dev/test_docs.py` passed: 26 tests in
4.25 seconds. The write-up uses rendering math delimiters. The integrating task
owner will run the final documentation build and inspect the rendered page
once concurrent documentation edits finish; this verifier has not claimed a
rendered-page check.

Scope limits: this validates the integral of the stored piecewise-linear
interpolant only. It does not establish that nodes resolve narrow lines,
validate the upstream PXR/CBS model or normalization estimator, or demonstrate
energy/angular convergence or detector acceptance. Analytic node values are
independent anchors for the reduction, not a paper reproduction.

## Verdict

Units, limits and signs/conventions pass. Independent re-derivation **matches**
the implementation. Verdict: **rederived**. Suggested task-owner ledger change:
`unverified` to `rederived` for `optimization-fixed-window-yield`; keep the
existing scoring-test anchor. The owner may assess the additional anchored
status criterion after integration checks. Human sign-off remains separate.

## Integration checks

The task owner applied `anchored` after the independent `rederived` verdict
and passing analytic anchors. The integrated focused run passed 63 tests,
including optimization persistence, real four-history evaluator/CAS/executor
coverage, ledger checks and documentation checks. Full `pyrite-dev verify`
passed with writable UV tool/cache directories under `/tmp`. The generated
HTML page contains 21 `math notranslate` elements and no unrendered dollar
delimiters in its article. These checks establish the stated integration
claim. The separate physical search/finalist study completed as SLURM job 2511
at `c941146c`; its [scalar comparison record](../check-records/pxr-optimization.json)
and [workflow interpretation](../../reference/optimization-studies.md#measured-bounded-comparison)
record all nine independently re-evaluated finalists. That study does not extend
this integral claim to upstream radiation-model or detector validation.

