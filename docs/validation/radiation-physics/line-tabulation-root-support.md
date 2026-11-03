# Line-tabulation root-support bound

Independent verification of `line-tabulation-nan-ceiling` (issue #286).

## Independent derivation before implementation inspection

Source: the ledger's Chantler interior-domain contract for anomalous refraction,
and the production three-pass fixed-point root contract supplied to the verifier.
This is an evaluation-domain equivalence claim, not a new physical equation.
Energies and mesh endpoints are in eV; the refractive decrement is dimensionless.

Let $C$ be the minimum basis-element Chantler upper endpoint. With anomalous
refraction enabled, the decrement $\delta(E)$ is NaN for $E\ge C$. On the
original ordered mesh $x_j$, let $k$ be the first index with $x_k\ge C$.
Retaining $x_0,\ldots,x_k$ preserves every finite interpolation interval and the
first NaN boundary interval. The prefix must retain the original nodes exactly;
regenerating a shortened linspace would change interpolation. Native knots
below the boundary must also remain exactly unchanged.

For linear interpolation that propagates NaN endpoint values and clamps to
endpoint values outside the mesh, the shortened and original decrement
interpolants agree for every query: both have the same finite prefix and both
return NaN in the deleted suffix. Equality here means equal finite values and
equal NaN masks, not floating-point NaN equality. An exact finite node adjoining
a NaN node can itself produce NaN under implementations using a zero times NaN;
the same neighboring pair must therefore survive the cut.

A deterministic recurrence $E_{i+1}=F(E_i,\delta(E_i))$ with the same starting
energy then gives identical iterates, by induction, including all three passes.
A final finite update may cross $C$ without another decrement evaluation. Thus
root equivalence alone does not prove score equivalence: final-root coefficient
interpolation and the finite-score rejection guards require separate inspection.
Static CBS coupling and attenuation may remain finite above $C$, so their table
values are not claimed equal in the removed suffix. If final finite roots above
$C$ can emit using only those finite factors, this proposed cut fails score
equivalence unless a common refraction/validity factor rejects them.

If the initial lower mesh bound exceeds $C$, a viable shortened mesh must still
contain enough original nodes for every interpolation implementation (normally
at least two). All decrement queries are then NaN, but this does not excuse
invalid indexing. A float32 cast may collapse neighboring nodes or move an
endpoint across $C$; preserve the original cast prefix and verify the actual
backend interpolation behavior.

With anomalous refraction disabled, no decrement NaN suffix follows from $C$.
The proposed stronger cap therefore has no justification in that case; retain
the previous all-table bound. In the limit where the requested upper bound is
below $C$, no nodes may change.

Cheap filters: units pass; unchanged-below-bound and disabled-anomalous limits
pass conditionally; sign/convention pass (the upper endpoint is excluded).

## Implementation comparison

The inspected setup helper chooses the basis refraction endpoint only with
`use_henke=True`; otherwise it returns the previous maximum table onset.
`_cxr_tabulation_grid` uses the original lower endpoint and unit arange stride,
then unions native Chantler and EPDL nodes. The minimum basis endpoint is itself
a retained native Chantler node, so shortening the upper endpoint preserves the
finite prefix exactly. The arange may retain an additional NaN node after that
endpoint; retaining extra NaN nodes is harmless. If the lower endpoint already
exceeds the bound, the upper endpoint becomes at least one eV greater than the
lower endpoint, retaining a valid multi-node mesh of NaN decrements.

`henke_dispersion` implements the strict interior contract. Both eager and
float32 fused `_in_medium_kinematics` execute the same three passes with linear
gather arithmetic and endpoint clamping. The fused coherent-stream prologue
uses the same recurrence. Suffix NaNs therefore preserve recurrence outputs,
including the last sampled decrement and convergence/critical-angle guards.
Float32 casts preserve identical prefix bytes; duplicate cast energies are
existing behavior, rather than introduced by the cut. No new precision bound
or interpolation convention is asserted here.

The final-root caveat matters. A third update may cross the endpoint without
another decrement lookup. Such a root is not necessarily rejected by the root
solver. Its photon susceptibility is NaN in both meshes, while static CBS
potential and attenuation can stay finite and need not agree in the deleted
suffix. Incoherent scalar and batched routes use a finite-positive TOTAL weight
mask for every component, including the separately returned CBS component.
Consequently the changed finite CBS/attenuation suffix cannot create accepted
incoherent lines. There is no independently selectable CBS-only production
root route; `components=True` returns components under the common total mask.
Coherent and flight-grouped modes reject `components=True`.

Coherent scalar/batched/stream routes combine PXR and CBS into complex fields.
Their formation/prefactor masks do not reject NaN susceptibility coefficients:
a last-update crossing can propagate NaN fields in both versions. The stream
prologue's early amplitude guard checks the prefactor before forming those
coefficients; it is not a general finite-field guard. Thus unchanged finite
outputs and equal NaN support are the equivalence being established, not a
promise that coherent unsupported roots are cleanly dropped. This existing
boundary behavior needs a separate fix if clean rejection is required.

Single-slab/finite-footprint attenuation uses the shared table. Grooved and
layered scalar escape evaluates attenuation directly and retains the same
root and susceptibility support; unchanged accepted roots query unchanged
finite mesh intervals. The cap does not alter physical escape geometry.

## Evidence, limits, and adjudication

The focused HOPG prefix/root regression passes for anomalous refraction on/off
and carbon/silicon absorbers. It uses a nonintegral lower endpoint, native
Chantler boundary queries, and the actual three-pass eager root solver.
Together with `tests/dev/test_docs.py`, the invocation passed 13 tests.
No independent external table transcription or new tolerance is involved:
the assertion is exact finite-value and NaN-mask equivalence to the original
mesh, not agreement against an external physics oracle.

Recommended follow-up anchors: explicitly force a final-third-update crossing
and compare scalar/batched total and CBS component masks; compare coherent
coefficient NaN masks; exercise a lower mesh bound already above the endpoint;
exercise mixed-element crystal endpoints; run float32 fused and coherent-stream
parity on hardware. The present root test uses float64 host arrays and no
explicit escape gradients, so it does not experimentally cover those cases.

Verdict: **rederived**, with the coherent NaN-propagation and final-update
qualifications above. The mathematical setup bound matches the implementation
for equivalence of existing production behavior. Suggested ledger change:
replace the stale all-table/value-equivalence claim by the anomalous-refraction
prefix/root-and-score equivalence claim, retain the previous bound when
anomalous refraction is disabled, name `_cxr_setup_ceiling`, and cite this
write-up plus the focused root-support regression. Do not claim discarded
attenuation or static-CBS values are unchanged. Do not claim every finite
returned root above the endpoint is rejected in every route. Human sign-off
remains separate.

The added synthetic last-update regression explicitly returns a finite root
above the support boundary and verifies identical shortened/full recurrence
outputs. Its final delta query brackets two finite nodes, avoiding the existing
zero-times-NaN behavior at a finite node adjacent to a NaN node. This anchors
the final-update qualification independently of whether natural tables happen
to produce that crossing. The remaining component/coherent and hardware checks
listed above remain separate recommendations.

## Owner-collected hardware evidence

[Issue #286 hardware report](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/286#issuecomment-6613466647)
records job 3768 on an RTX 5080, code `67b50ee71`. The committed
`checks/line_setup_support_286.py` probe passes 48 exact full/short-mesh root
comparisons: HOPG and MoS2, anomalous refraction on/off, FP32/fused and FP64,
windows below/above support, bulk and explicit escape gradients, and an
explicit Si absorber. CPU anchors now also include above-support windows and
explicit gradients. These extend the original verifier's experimental coverage;
they do not constitute an independent new physical-material oracle.

The same job completes 20 production-policy spectra at 300 and 29,999 keV,
with two expected point-budget refusals and no GPU OOMs. High-energy population
and adaptive arms reserve 426.18 MiB in the device pool; the previous attempts
failed near the configured 11.97 GB pool limit. Successful reserved memory and
failed allocated memory are distinct metrics, so no exact reduction ratio is
inferred. Both saved high-energy refined reference arrays remain bitwise
identical to the old unbounded-setup run on identical tracks. Intrinsic integrals,
coordinate hashes and finite outputs were checked independently on collection.

With the independent derivation and regression anchors, the task owner records
`anchored`, never human sign-off. Component/coherent boundary-output probes
remain useful future coverage; the documented equivalence retains preexisting
coherent NaN behavior rather than claiming new finite-field rejection.

The integrated documentation build renders all expressions as math and passes
with the page linked from its canonical validation index.
