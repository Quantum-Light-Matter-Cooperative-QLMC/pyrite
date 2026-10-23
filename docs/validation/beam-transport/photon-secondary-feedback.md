# Charged secondary photon feedback

Validation ID: `photon-secondary-feedback`. Verdict: **rederived**, restricted
to bounded CPU stack bookkeeping and physical emission adapters. Snapshot:
`5d418b562bd85fa7132e02257ce1c93648012846`.
The verifier did not implement the claim. The independent expressions were
recorded before implementation inspection, using ledger rows,
signatures/docstrings and primary sources. Implementer research notes were
not used.

## Sources and assumptions

[PENELOPE-2024](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf),
§3.4 Eqs. (3.183–3.185), supplies free-electron two-photon annihilation
kinematics; §5.2 Eq. (5.53) supplies the rest-energy sign convention;
Chapter 7 §7.1 p.283 supplies secondary-stack energy replacement.
[ENDF-102](https://www-nds.iaea.org/public/documents/endf/endf102/endf102.pdf),
§§28.1–28.3, specifies radiative/nonradiative daughters and separately tabulated
transition energies `ETR` in eV. Component sampling laws remain the existing
`heitler-annihilation`, `positron-annihilation-at-rest` and
`eadl-sampled-relaxation` claims.

Assume sample-frame Å/eV, contiguous target stack, inherited charged soft-loss
deposition, neutral-atom independent EADL cascades and isotropic relaxation
directions. Target-electron binding and momentum in annihilation are neglected.
Positrons at or below the charged threshold deposit their kinetic energy and
annihilate at rest. Escaped charged tracks terminate; charged filter transport
and arbitrary region re-entry are outside this reference.

## Independent derivation

Write $M=m_ec^2$. Assign the pair's full $2M$ rest reservoir to its positron;
this is a bookkeeping convention, not an extra source of energy. Pair creation
already removed that energy from the parent photon. A charged launch must obey

$$
T+2M\,1_{e^+}=X+D+B_{\rm rem}+\sum_j k_j.
$$

For escape, $X$ includes both remaining positron kinetic energy and $2M$.
For annihilation at kinetic energy $T_a$, energy-momentum conservation gives

$$
k_1+k_2=T_a+2M,\qquad
k_1\hat{\mathbf n}_1+k_2\hat{\mathbf n}_2
=\sqrt{T_a(T_a+2M)}\,\hat{\mathbf v}.
$$

At rest, both energies are $M$ and the directions are opposite. Subthreshold
local deposition is $T$, rather than $T+2M$. When launching a positron from
the photon queue, its previously booked rest reservoir must therefore be
removed once and replaced by escaped rest or emitted photon energy.

For a frozen charged-impact vacancy, let $B_0$ be binding already reserved by
the original charged transport. A radiative transition leaves one daughter
hole; an Auger transition leaves two. Using the supplied transition energies,
define the residual accounting term

$$
B_* = B_0-\sum_{\gamma} E_{{\rm TR},\gamma}
-\sum_A T_A.
$$

If tabulated energies differ from binding-energy differences, this residual
contains terminal binding plus a signed defect. More explicitly, with initial
EADL binding $b_i$ and terminal-hole bindings $b_t$,

$$
B_* =\sum_t b_t+(B_0-b_i)
+\left(b_i-\sum_t b_t-\sum_{\gamma} E_{{\rm TR},\gamma}-\sum_A T_A\right).
$$

It must not be clamped to zero. The frozen source electron loses no additional
energy. For every transported Auger daughter substitute
$T_A=X_A+D_A+B_A+\sum_j k_{A,j}$; below-threshold daughters instead deposit
$T_A$ locally. Summing these substitutions yields

$$
B_0=X_A+D_A+B_{\rm rem}
+\sum_{\gamma} E_{{\rm TR},\gamma}+\sum_{A,j}k_{A,j}.
$$

All terms are eV; directions are dimensionless. Emission adapters copy parent
weights and history identity. At a photon cutoff they partition energy into
transported photons and a weighted local-energy account; they do not discard
the omitted photons' energy. Preexisting first-interaction absorption flags
must not suppress emitted photons before transport. Physical event batches
must remain separate from a weighted continuum representing the same shower.

## Source-to-code comparison

`annihilation_emissions` exports two unattenuated photons per event, converting
keV to eV. `_annihilation_return` excludes escaped positrons. `_launch_cascade`
applies positron transport and annihilation truncation only to generation zero;
later impact daughters use electron transport. `FeedbackChargedStage.__call__`
handles subthreshold positrons locally and aggregates Auger rounds by their
original launch owner. A relaxation-round cap raises instead of silently
discarding pending energy.

`_feedback_batch` adds $2M$ only to escaped positrons, otherwise exports the
annihilation photons. It subtracts the entire EADL emitted photon/Auger energy
from binding, deposits low-energy Auger electrons and transports the others.
`impact_relaxation_emissions` selects actual inner-shell hard collisions,
places them at collision endpoints, and creates no vacancy for outer proxy
channels. `_book_stage` subtracts exactly the launched positrons' prior rest
reservoir and books returned photons with parent weights.

`primary_relaxation_sources` starts from the recorded reserved binding,
subtracts release once, follows Auger descendants and leaves the original
charged result untouched. `emission_sources` preserves root history/weight,
uses disjoint caller-selected event ranges and retains subcutoff local energy.
`primary_annihilation_sources` reuses both recorded emission photons regardless
of old absorption flags. Scene feedback is enabled explicitly and carries
separate observation provenance; filters are refused in this mode.

No divergent term, sign, factor or unit was found. The original §1.4 secondary
stack citation needs the Chapter 7 correction: §1.4 explicitly omits secondary
production. The binding remainder is an accounting quantity that may be signed;
closure alone does not establish physical accuracy of the underlying data.

## Numerical evidence and verdict limits

An additional direct kinematic probe used $M=510998.95$ eV independently in the
expected expressions. At $T_a=0$, $0.3$ MeV and $1$ MeV, energy-sum errors were
$1.17\times10^{-10}$ eV or less and momentum-vector errors were
$1.38\times10^{-10}$ eV or less. At rest the emitted energies were
$510998.95$ eV each. These checks test the exported emission kinematics, not
the annihilation differential probability law.

The focused coupling/feedback/Scene-handoff modules passed 50 tests in 10.47 s;
the separately selected slow charged-stage test passed in 9.48 s. Coverage
includes real Si electron/positron closure, escaped rest, full-queue reservoir
replacement, Auger re-entry/cap refusal, source weight/history mapping and
primary-vacancy closure without source mutation. `tests/dev/test_docs.py`
passed eight tests. This proposed write-up must still receive rendered-math
inspection after the task owner installs it.

Verdict: **rederived** for the stated bookkeeping and adapters, conditional on
the separately ledgered component laws. No external full-track validation,
general charged geometry, GPU parity or production support follows. Human
sign-off remains separate.
