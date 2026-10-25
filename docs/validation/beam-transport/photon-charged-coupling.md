# Photon-born electron coupling

Validation ID: `photon-charged-coupling`. Verdict: **rederived**, restricted to
the bounded CPU stack reference. Snapshot: `5d418b562bd85fa7132e02257ce1c93648012846`.
The verifier did not implement this claim. The equations below were derived
from the ledger, signatures/docstrings and primary sources before reading
implementation bodies. Existing implementer research notes were not used.

## Sources and scope

[PENELOPE-2024](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf),
Chapter 7, especially §7.1 p.283, describes secondary-stack energy removal
when a secondary track starts. Section 1.4 explicitly omits secondary
production and is not the correct stack citation. Component transport laws
remain the separate `shell-secondary-transport`, `bremslib-radiative-partition`
and `photon-history-queue` claims.

The intended operation launches photon-born electrons in an existing charged
soft/hard cascade and returns its hard radiation to photon transport. Positions
are sample-frame Å; energies at this boundary are eV; charged transport uses
keV. Statistical weights are dimensionless and unchanged for descendants.
The charged stage assumes a contiguous stack starting at zero and matching
sample coordinates. Filter-box electrons deposit locally; positrons remain
pending in this electron-only stage. Charged clocks start at zero. Later
charged-generation stream addresses retain the existing batch dependence.

## Independent derivation

Let a launch have kinetic energy $T$, charged escape $X$, local deposition
$D$, reserved binding $B$ and emitted hard photons $k_j$. Energy conservation
requires

$$
T=X+D+B+\sum_j k_j.
$$

The deposited term includes inherited soft radiative loss and untransported
charged kinetic energy. This claim does not turn continuous soft loss into
new photon rows. With photon cutoff $E_c$, emitted energy partitions as

$$
\sum_j k_j=\sum_{k_j\leq E_c}k_j+\sum_{k_j>E_c}k_j.
$$

The first sum becomes local deposited radiation; the second becomes the next
photon round's incident energy. Launched kinetic energy and returned photon
energy are internal transfers. Substituting each stage balance into the
preceding photon balance cancels these internal transfers. For original
history $h$ this gives

$$
E_{0,h}^{(w)}=\sum_r P_{h,r}^{(w)}
+\sum_a w_a(X_a+D_a+B_a+L_a+S_a+V_a+U_a).
$$

Here $P$ contains photon-terminal terms, including unreleased pair rest
energy; $L$ is unlaunched local electron energy, $S$ is subcutoff returned
radiation, $V$ is pending positron kinetic energy and $U$ is capped unresolved
launch kinetic energy. These are mutually exclusive terminal accounts.
Weights multiply energy exactly once. Neither daughter count nor queue order
introduces a normalization factor. Scheduling may change random realizations;
the physical means are preserved provided the underlying streams and
transition laws remain valid.

Units pass: each term is weighted eV, and the explicit eV/keV conversion is
$10^3$. Signs pass: every terminal energy is added to the right-hand side;
binding remains reserved. Limits pass: no launch ends at round zero; a zero
round cap records launch kinetic energy as unresolved; a stage with no returned
photons ends after its charged stage; cutoff equality belongs to local energy.

## Comparison with the snapshot

`CascadeChargedStage.__call__` constructs explicit `LaunchState` positions,
directions and energies, converts eV to keV, and returns the per-history
escaped/deposited/binding/radiated split in eV. `transport_secondary_cascade`
and `_with_launch` replace beam sampling only for explicit generation-zero
launches; `launch=None` retains the original path.

`_hard_photons` exports every positive hard-radiative event at its row endpoint,
with its root launch owner and emission ordinal. `_book_stage` deposits
subcutoff photons and returns harder photons with their parent's weight,
history and root source kind. Daughter stream keys are derived from the
charged key and emission ordinal; `_root_keys` rejects duplicate source
addresses. `stage_seed` separates coupling rounds. `_split_charged` and the
round cap implement the terminal accounts above. `coupled_ledger_residual_eV`
subtracts the complete terminal sum from original incident energy.

No divergent factor, sign or unit was found. The original §1.4 stack citation
needs the Chapter 7 correction above; this is a provenance correction, not a
different energy equation.

## Evidence and limits

Focused coupling, feedback and Scene-handoff modules: 50 passed, one explicitly
slow test skipped, in 10.47 s. The separately selected slow
`test_cascade_stage_closes_per_row_and_returns_its_radiation` passed in 9.48 s
(8.24 s test call). Its eight Si launches span 0.3–1.5 MeV, with per-launch
closure and returned-radiation equality at relative tolerance $10^{-12}$.
The coupled 2 MeV Si test also passed. These tests support the source-to-code
comparison; their internal bookkeeping is not an external shower oracle.

This verdict certifies the stated energy substitution and handoff conventions.
It does not certify generalized charged geometry, external full-track spectra,
GPU parity or production support. Human sign-off remains separate.
