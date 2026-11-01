# Charged vacuum re-entry: independent verification

Validation: `photon-charged-reentry`.

## Scope, independence and source

The verifier did not implement this claim. This independent derivation was
written from the ledger, owner signatures/docstrings and source before reading
implementation bodies. Scope: the explicit CPU reference feedback stage;
component charged transport, scattering laws and photon sampling retain their
separate validation claims.

[PENELOPE-2024, NEA/MBDAV/R(2024)1](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf),
§1.4.4, printed pp.35–36, describes Markovian resumption at a sharp passive
interface using the next medium's interaction properties. Chapter 7 §7.1,
printed p.283, describes subtracting previously deposited secondary energy
when starting the secondary history. The positron reservoir convention is
the separately validated `photon-secondary-feedback` accounting convention. The PDF was downloaded and its text inspected on 2026-10-25;
the browser fetch returned 403, but direct download succeeded.

Assumptions: disjoint analytic material bodies; no forces, interactions or
annihilation in vacuum; Å positions, eV energies, unit directions; inherited
local condensed transport; visit clocks restart, so no time observable;
independently keyed visits rather than monolithic-flight bitwise equivalence.

## Independent derivation (before implementation inspection)

Let $M=m_ec^2$ and assign the pair's $2M$ reservoir to its positron, as in the
existing feedback accounting convention. A material visit with launch kinetic
energy $T$ and species indicator $p=1$ for a positron obeys

$$
T+2Mp=D+B+\sum_\gamma k_\gamma+\sum_{a\in X}(T_a+2Mp_a).
$$

$X$ contains every terminal escaped track, including shell and Auger electrons.
$B$ is the inherited signed binding remainder, not necessarily a nonnegative
local deposit. Rest energy belongs only to actual escaping positrons, never
to their electron daughters. The original source electron is not charged a
second time when a descendant is continued.

For an escaped position $\mathbf r$ and unit direction $\hat{\mathbf v}$,
intersect $\mathbf r+s\hat{\mathbf v}$ with each body's interior for $s\geq0$.
Discard zero-length intersections. Choose the smallest forward entry among
positive-length intervals. For a box with orthonormal axes $\mathbf e_i$,
centre $\mathbf c$ and half sizes $h_i$, intersect the three inequalities

$$
-h_i\leq(\mathbf r-\mathbf c)\cdot\mathbf e_i
+s\hat{\mathbf v}\cdot\mathbf e_i<h_i.
$$

At the first entry $s_*$, launch from

$$
\mathbf r'=\mathbf r+s_*\hat{\mathbf v},\qquad
T'=T,\quad p'=p,\quad\hat{\mathbf v}'=\hat{\mathbf v}.
$$

Only the material's local coordinate representation changes. A vacuum miss
books $T+2Mp$ as final escape, even for a zero-energy positron. A positive
material interval beginning at zero admits direct contact; the departed body
must not be spuriously selected again from roundoff at its exit surface.

For each continued exit, subtract its complete provisional escape reservoir
and replace it with the next visit's deposit, binding, emitted photons and
exits. Repeated substitution telescopes all intermediate kinetic and rest
reservoirs:

$$
T_0+2Mp_0=E_{\rm final\ escape}
+\sum_j\left(D_j+B_j+\sum_\gamma k_{\gamma,j}\right).
$$

Root owner, history and weight remain unchanged across the descendant tree;
each new visit receives a distinct reproducible stream. Pending tracks at a
visit cap must raise, since dropping them breaks this identity.

## Cheap filters

- Units: pass. Ray parameters, centres and half sizes are Å; direction and
  rotations are dimensionless. Every energy reservoir is eV.
- Limits: pass at the contract level. No exit gives an empty table; removing
  all subsequent material leaves one final escape; zero gap is direct contact;
  finite or arbitrarily long vacuum changes neither energy nor species;
  backscatter can return to an earlier body; a rigid rotation preserves chords.
- Signs/conventions: pass. Continuing an exit removes a positive escape term
  exactly once. Positron rest is retained on vacuum escape and is available
  for annihilation only after material admission. Electron daughters carry no
  assigned annihilation reservoir.

## Implementation comparison and evidence

Implementation inspection followed the derivation above. `escaped_charged_rows`
selects terminal rows by track, flight and substep, computes each segment's
endpoint, converts keV to eV and maps each terminal to the root launch owner.
Generation-zero species are inherited; shell daughters are electrons.
`SceneFeedbackStage.__call__` books only final misses as escape, accumulates
per-visit deposited/binding/radiated terms, maps emitted photons and terminal
exits back to root ownership and substitutes all intermediate exits. Its
returned initial positron reservoir is separate from those terms. Per-region
axes implement the dot-product transformation, and the factory slices each
material's explicit per-layer tables before building its contiguous local
stage. Vacuum layers do not get material stages. The cap raises rather than
discarding unresolved material visits.

### Discrepancy found before boundary-entry repair

The first divergent convention is the local launch position passed by
`_Region.local_rows`: a geometrically selected incoming side face can transform
to a position a few ulps outside the local footprint. The charged core's exact
footprint admission then rejects it, producing no terminal row. The scheduler
ignores the material stage's provisional escape total and relies on terminal
rows for continuation, so this rejected launch loses its complete energy.

A concrete witness uses a box centred at $(0,0,30)$ Å with half sizes
$(10,10,5)$ Å and row axes

$$
\mathbf e_x=(\cos\theta,0,-\sin\theta),\qquad
\mathbf e_y=(0,1,0),\qquad
\mathbf e_z=(\sin\theta,0,\cos\theta),\qquad
\theta=0.056733668341708544.
$$

Launch a 300000 eV electron from $\mathbf c-30\mathbf e_x$ along
$\mathbf e_x$, key 1, seed 5. Independently, entry distance is 20 Å and the
material chord is 20 Å. With the existing real Si test fixture, the pre-repair
stage returned zero in all four energy terms: the energy residual was
300000 eV. Ordinary 180-degree rotation tests did not expose this failure.
The negative local depth encountered at some other rotation angles alone did
not lose energy; the demonstrated failure specifically concerns transverse
footprint admission.

The appropriate repair is to represent the already selected analytic incoming
boundary in the receiving body's exact local bounds. An arbitrary inward
push distance would change the chord and is unnecessary. This independent
verifier did not change implementation code.

### Evidence and remaining scope

At the pre-repair snapshot `639d10599db8de30e41af1d54f47052d5560bcc1` plus the
uncommitted re-entry implementation, the focused re-entry module and
`tests/dev/test_docs.py` passed 23 tests in 3.59 s. The independent rotated
side-face probe through `uv run pyrite-dev test` failed as described above.
The analytic energy oracle is exact; $10^{-12}$ relative and $10^{-8}$ eV
absolute comparison tolerances accommodate binary64 accumulation, not source
uncertainty. No measured constants or new reference tables are transcribed
by this claim. Existing Si fixtures exercise accounting and table ownership;
they do not establish independent physical accuracy of the inherited tables.

Notation/normalization: pass; the ledger, owner docstrings and derivation all
use kinetic eV, Å, root launch ownership and one positron-owned rest reservoir.
Reference-data transcription: not applicable. External/PDF provenance: pass.
GPU parity, full-track external comparisons, thick-target/alloy/511 keV
anchoring and production admission are outside this claim.

The task owner's final `pyrite-dev docs` build passed with
`UV_TOOL_DIR=/tmp/pyrite-uv-tools`. Independent inspection of the resulting
HTML confirmed the final verdict, 26 `math notranslate` elements in the article
and zero surviving dollar delimiters. An earlier simultaneous build failed
during `_build` cleanup and was not used as successful-build evidence.

## Independent check of the author's repair

After receiving the discrepancy, the task author changed the selected-entry
transform to clip its local position to the exact receiving kernel bounds.
This represents an already admitted analytic boundary; it introduces no
inward displacement scale and does not change energy, direction or species.
The bounds include the kernel's Å-to-mm-to-Å footprint conversion, avoiding
a second representation mismatch. Interior positions remain unchanged.
`_check_visit_balance` additionally rejects a material history whose total
energy or retained-exit kinetic/rest partition differs from its launch.

The repaired real Si side-entry witness closes at 300000 eV: deposition is
approximately 0.20705021 eV and escape approximately 299999.79294979 eV,
consistent with the 20 Å material chord. Independent side-entry probes at
0.01, 0.056733668341708544 and 0.37 rad all close; repaired lower-depth
entry probes give nonnegative local depth. The focused re-entry module plus
the independent real-side probe passed 19 tests (one unrelated scratch probe
deselected) in 3.57 s. A separate rerun of both independent boundary probes
passed two tests in 2.52 s. `tests/dev/test_docs.py` passed eight tests after
this report's initial write. The owning module now includes the concrete
real-side witness and injected failures for dropped visit energy, absent
exit phase space and inconsistent escaped energy.

## Verdict

Final verdict: **rederived**, conditional on the separately ledgered charged
component laws. Units, limits and signs/conventions pass after the author's
boundary repair. No remaining divergent term was found in this claim.
Suggested ledger change: `unverified` to `rederived`, cite this independent
write-up and retain `tests/montecarlo/transport/test_photon_reentry.py` as the anchor;
`anchored` becomes appropriate when that regression evidence is green in CI.
Human sign-off remains separate.


## Production and CUDA extension: independent derivation

A fresh verifier (2026-10-25) received the production-extension scope and read
only the claim, prior source-based derivation and current signatures/docstrings
before writing this section. The verifier did not implement the extension.
The source remains PENELOPE-2024 §1.4.4 and Chapter 7 §7.1 cited above;
no new interaction law or measured constant is introduced.

For body $b$, the charged interaction resource must represent the same
composition and density as that body's photon material. Let $C_b$ denote the
resolved material identity and $R_b^-$, $R_b^+$ and $R_b^\gamma$ its electron,
positron and bremsstrahlung resources. Admission requires

$$
C_b^{\rm photon}=C_b^{\rm charged},\qquad
R_b^q=R^q(C_b),\quad q\in\{-,+,\gamma\}.
$$

A catalog label alone is insufficient if its resolved composition differs.
Vacuum admits no charged resource. Replay must reconstruct this association
from the recorded material labels and verify the same pinned content; source
transport remains owned by the original target material.

Navigation remains the forward positive ray-interval construction above.
Changing the array backend cannot change the interval inequalities, length
unit, or choice of nearest body. Binary64 clipping must preserve an already
selected incoming boundary without adding a finite inward displacement.
Backend-specific interaction kernels may consume different random streams;
this claim requires the same geometry and accounting conventions, rather
than individual-track equality.

Every descendant inherits the root primary identity and statistical weight
$w$. Therefore each root's weighted energy ledger is

$$
w(T_0+2Mp_0)=wE_{\rm final\ escape}
+w\sum_j\left(D_j+B_j+\sum_\gamma k_{\gamma,j}\right).
$$

The interface introduces no new weight, source normalization, rest reservoir
or emitted-photon multiplicity. A replayed history must retain the original
pins and ownership, rather than silently accepting a new catalog/table set.
Cheap contract filters pass: every term has weighted eV units, the vacuum
and no-filter limits recover the preceding claim, and the positive positron
reservoir telescopes exactly once through successive material visits.

Implementation comparison and the final extension verdict follow below.

### Extension comparison and filter results

Implementation-body inspection followed the derivation. The enabled production
path passes filter catalog keys separately from the original source Case.
`prepare_feedback_scene` rejects a box whose ordered element/number-density
composition differs from the catalog material, including density differences.
It resolves electron stopping and elastic arrays per material body, unions
BremsLib resources by element, and records those table identities together
with separately resolved positron stopping/elastic identities. The latter
match the resources selected by `_positron_overrides`; positron BremsLib
retains the inherited scaling law. Target-layer lists are followed by box
lists, then sliced to one body at stage construction. Vacuum entries retain
their place but receive no stage. Missing material ownership fails before
primary execution.

`prepare_source` forces the charged material core to CUDA for a CUDA
observation. `_next` uses the same forward half-open axis clipping expression
with NumPy or CuPy; inputs reach it as validated binary64 charged phase space,
and device distances/body choices return to the host scheduler. The local
entry clipping and final-exit substitution remain the preceding derivation's
operations. This inspection establishes algorithm and dtype intent only;
it does not establish hardware execution or numerical parity.

The scheduler maps each material result and escaped daughter to the original
launch owner. `_book_stage` applies the original weight and history to every
energy term and returned photon; the positron rest debit is checked against
exactly one reservoir per original positron. Visit streams retain descendant
keys and a visit index, rather than reusing a previous material flight. No
new weight or rest-energy factor was found.

Replay reconstructs boxes from the recorded geometry, material labels from
`charged_feedback` or `relaxation_feedback`, and the same enabled body-local
resources. It checks source Case content, source table markers, the complete
observation identity and its digest; a changed body-table pin is not accepted
as an equivalent replay. With no boxes the original stack-preparation branch
is retained.

| Additional filter | Result |
| --- | --- |
| Notation and normalization | Pass: eV, Å, root ownership and inherited weights agree. |
| Units at loader/kernel/output boundaries | Pass by inspection: material number density is atoms/Å³; geometry converts Å to the charged core's mm footprint and back. |
| Reference transcription | Not applicable: no new tabulated constants or measured reference values. |
| Tolerances | The analytic navigation probe uses $10^{-12}$ Å absolute allowance for binary64 coordinate transforms; accounting retains $2\times10^{-12}$ relative and $10^{-8}$ eV absolute inherited rounding allowances. |
| Existing parity evidence | Pass for the bounded fixtures: audited remote artifacts and source hashes are recorded below. |

### Extension verification evidence

A separate analytic probe constructed 1001 oriented boxes with angles from
0.0001 to 1.5 rad. Starting 30 Å from each centre along its local negative
$x$ axis, every ray independently has entry distance 20 Å. The CPU scheduler
selected the expected box at every angle; maximum absolute distance error was
$7.105427357601002\times10^{-15}$ Å. This probe used the project environment
and did not call interaction tables or implementation intersection helpers
as its oracle.

The first focused run of `test_photon_reentry.py`,
`test_photon_reentry_production.py` and `tests/dev/test_docs.py` returned
26 passes and four setup errors in 4.49 s. All four errors came from the
new production fixture specifying `atomic_electron_deflection="none"` in a
resolved Case, whose representation requires the field to be absent or
`"kawrakow"`. This was reported immediately to the task owner; the verifier
did not edit the tests or implementation. The 18 existing re-entry checks
and eight documentation checks passed. Two stale production-admission
statements in the factory/module docstrings were also reported for correction.

The task owner repaired the fixture by preserving the absent resolved Case
field, and corrected both stale production-admission docstrings. The verifier
inspected those edits. The owner then reported all four production tests
passing and all 18 CPU re-entry tests passing after backend factoring.
These outcomes are owner-supplied evidence, distinct from the independent
probe and original focused run above. The owner completed the final
documentation build. This verifier independently inspected the resulting
extension HTML: 39 `math notranslate` elements, zero dollar delimiters, and
the production-extension section present. The verifier performed no build.

### Extension verdict

**Rederived**, conditional on the separately ledgered charged interaction
laws. Units, limiting cases, signs, weight/rest conventions and body-resource
ownership match the independent extension derivation. No divergent physics
term remains after the fixture and traceability repairs. The audited hardware checks below supply bounded CUDA execution,
CPU/device parity and capture/replay evidence. Suggested ledger edit: extend
the existing claim's Code/Notes/Checks to identify the production
resource-preparation boundary and pinned body-material replay, with the
tested hardware/fixture scope stated explicitly. Do not mark `signed-off`.

### Audited hardware evidence in both precision settings

The verifier independently inspected the downloaded JSON and logs for remote
jobs 2528 and 2529 and matched all 14 recorded source-file SHA-256 values to
the local files. Both identify source commit
`70d37305b16ac193a5985496180188a9c45c5e9b`, a clean source snapshot, an NVIDIA
GeForce RTX 5080, CuPy 14.2.0, CUDA runtime 13020, Python 3.14.6 and NumPy
2.5.4. Each records 22 tests, zero failures, zero errors and zero skips.
Job 2528 used `PYRITE_FP64=1` and took 17.553 s; job 2529 used
`PYRITE_FP64=0` and took 17.777 s. These are two successful runs of the same
assertions at the two global precision settings; they do not constitute
an additional direct comparison of all output arrays across those settings.

The artifact files inspected were `/tmp/issue341-reentry-2528.json` and
`/tmp/issue341-reentry-2529.json`, with adjacent `.log` files. Their JSON
SHA-256 values, respectively, are
`731cfdcc4403427a3be9c0aefa4538a4f136f6cb7ca17315701de97ac432660e` and
`b01f22b18bad04a6637a397d075cc5a2bd6723ca042db2faf456910dcbeb5966`.
The durable combined record is
`checks/photon_cuda/reentry-rtx5080.json`; it retains run metadata, source
hashes and original log hashes for these jobs and the pair probe below.

The 13 extension cases cover eight public Si/W filter admission/provenance
combinations, explicit device navigation, forced CUDA charged-core selection,
one original primary pass, real electron and positron scene-stage comparisons
with CPU, forward/backward boundaries and empty batches, capture/replay,
checkpoint/CAS reconstruction, CPU-only reduced-payload reading and mode-off
bitwise stability. Nine existing production cases also pass. The charged
stage comparisons use $2\times10^{-10}$ relative tolerance, $10^{-6}$ eV
absolute tolerance for energy ledgers, and $10^{-5}$ absolute tolerance for
floating phase-space fields; integer lineage fields match exactly. Navigation
uses $10^{-13}$ relative and $10^{-8}$ Å absolute tolerance. These numerical
allowances bound implementation roundoff in the deterministic fixtures, not
experimental uncertainty or general model accuracy.

This evidence resolves the preceding pending hardware limitation for the
specific tested paths and fixtures in both precision settings. It does not
establish general tungsten full-track accuracy: the public tungsten checks
exercise admission and provenance without requiring a nonempty tungsten
charged cascade. The separate nonempty full coupled-queue filter pair/rest
transfer probe below is additional evidence beyond these 22 cases. The initial job 2527's four copper
pre-transport refusals are not successful copper physics validation; the
fixture was changed to supported tungsten because copper lacks the required
shell conduction data. No interaction law or tolerance was changed to obtain
the successful runs.

The extension verdict remains **rederived**, with bounded hardware regression
support now present. Suggested ledger notes should cite jobs 2528/2529 and
their retained artifacts, distinguish the two global precision settings from
cross-setting array equality, and preserve the inherited-model and
full-track validation limits. Human sign-off remains separate.

### Audited nonempty filter pair/rest-return probe

Job 2530 ran `test_photon_reentry_pair_cuda.py` at source commit
`cbcb7302e8d173c27bb1c251da315309d7805450`, separately with `PYRITE_FP64=1`
and `PYRITE_FP64=0`. Each precision run reports one passing test, no failures,
no errors and no skips (3.195 s and 1.643 s respectively). The verifier
inspected `/tmp/issue341-reentry-pair-2530.json` and its `.log`, matched all
six source hashes to local files, and checked that the durable
`filter_born_pair_queue` record in
`checks/photon_cuda/reentry-rtx5080.json` preserves every downloaded field
and the matching log hash. The downloaded JSON SHA-256 is
`f0732409bab54b252c9fbda44fa08587c1cb1512908db1b3c56bac7f945de994`.

The probe prepares the production scene/resources with a 20 µm Si filter,
then injects two unit-weight 1.03 MeV photons at its centre. Photon channel
rates are controlled throughout the probe: only nuclear pair conversion is
nonzero above threshold; all subthreshold returned photons are transparent.
Pair final states, charged CUDA transport, vacancies, annihilation and the
coupled return queue remain real. This controlled branch therefore tests
nonempty filter-born pair/rest transfer and accounting, rather than physical
photon attenuation probabilities or annihilation-photon absorption.

Assertions require both first-round positrons and their electron partners
to be launched, a nonempty charged feedback round, complete per-history
ledgers, zero outstanding positron-rest term within $10^{-8}$ eV, at least
four escaping photons above 400 keV in the combined two-history population,
and per-history total energy residual within $10^{-7}$ eV. The photon-count
assertion is aggregate; it does not independently require two photons from
each history. Both precision runs pass these unchanged assertions.

Final scoped verdict: **rederived**, with hardware regression support for
production-prepared body/table ownership, the tested CPU/CUDA scene-stage
continuation and energy partitions, pinned replay/persistence, mode-off
stability, and controlled nonempty filter-born pair/rest return in both
global precision settings. No pending hardware result remains for those
checks. This is neither an external full-track physics comparison nor a
claim of general material accuracy or cross-precision array identity.
Separately ledgered interaction laws and human sign-off retain their own
validation requirements.
