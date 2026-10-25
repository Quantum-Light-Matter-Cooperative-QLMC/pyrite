# `cr-dechanneling-collapse`: Kephart occupation lengths and cutoff sensitivity

Historical implementation evidence for issue #504, followed by the independent #522 boundary review below. The ledger retains its declared-calibration `anchored` scope; the exploratory replacement receives a separate verdict.

- **Harness:** `checks/kephart_occupation.py` at revisions `ebea431c` (remote sweep on `cluster-head`, SLURM CPU) and `42b93283` (local `H ≥ 40` runs). The solver code is the same in both.
- **Reference:** Kephart et al., Phys. Rev. B 40, 4249 (1989). Table I gives 1/e occupation lengths ±20%; Table II gives the inferred divergences of 2.2 mrad (17 MeV) and 1.0 mrad (54 MeV).
- **Assumed temperature:** 293.15 K, which the paper does not state.

## Method

1. Run `DensityMatrixSolver` with the TDS and electronic kernels from normal incidence with Gaussian divergence.
2. Weight the band populations as the measured line (17 MeV) or as the dipole-weighted window (54 MeV).
3. Integrate them into a yield `Y(Z)`.
4. Fit `A(1 − e^{−Z/L})` on 24 log-spaced thicknesses over 2.8–120 µm (17 MeV) or 2.8–500 µm (54 MeV), the range where the paper's exponential fit holds.

The collapse margin sets the subspace cutoff `barrier + margin`; a margin of ∞ leaves the plane-wave band edge as the only sink.

## Cutoff sensitivity

Occupation length `L` in µm, with the ratio to Kephart in parentheses. Settings: `H = 32` (17 MeV) or `H = 40` (54 MeV), `n_cells = 16`, `dz = 31.25 Å`.

| margin [eV] | Si(110) 17 MeV n=1 (20) | n=2 (17) | n=3 (17) | Si(100) 17 MeV (16) | Si(100) 54 MeV (24) | Si(110) 54 MeV (36) |
|---|---|---|---|---|---|---|
| 0 | 1.8 (0.09) | 2.0 (0.12) | 2.0 (0.12) | 1.6 (0.10) | — | — |
| 30 | 5.2 (0.26) | 5.4 (0.32) | 5.6 (0.33) | 5.4 (0.34) | — | — |
| 100 | 9.9 (0.50) | 10.1 (0.59) | 10.2 (0.60) | 10.1 (0.63) | 21.0 (0.88) | 20.8 (0.58) |
| 300 | 19.0 (0.95) | 18.8 (1.11) | 18.7 (1.10) | 18.9 (1.18) | 42.3 (1.76) | 42.3 (1.18) |
| ∞ | 37.1 (1.85) | 36.0 (2.12) | 35.2 (2.07) | 41.6 (2.60) | — | 64.4 (1.79), `H = 40`, not converged in `H` |

The relative RMS residual of the integrated-exponential fit is 0.1–1% for finite margins and 2–3% for ∞. With no near-barrier sink the population decay is visibly non-exponential. The paper reports the same for electrons beyond 1 mm and also fits a multiple-scattering `Z^{1/2}` form.

## Convergence (Si(110), 17 MeV, margin ∞, n=1)

- **Depth step** (`H = 24`): `dz` = 250 / 125 / 62.5 / 31.25 Å gives `L` = 24.5 / 27.1 / 28.4 / 29.1 µm. This converges at second order with a large constant; the balance defect at 31.25 Å is −9e-3.
- **Bloch grid:** `n_cells = 32` reproduces `n_cells = 16` to 0.1 µm (`H = 32`, `dz = 31.25 Å`).
- **Plane-wave band** (`dz = 62.5 Å`): `H` = 24 / 32 / 40 / 48 / 56 / 64 gives `L` = 28.4 / 36.2 / 40.9 / 43.6 / 44.9 / 45.6 µm. The increments shrink from 7.8 to 0.7 µm, so the no-sink limit is about 46–47 µm, about 2.3× Kephart.

Entry populations of bands 1–3 at 17 MeV (110) are 6.40 / 6.40 / 6.44%, against Kephart's Table II theory of 6.5% each and measured 5.5 / 5.6 / 6.3%.

## Detector observable (maintainer-chosen comparison basis)

Kephart's detector sits on the beam axis. `checks/kephart_occupation.py` weights each band signal by `W(z)`, the detector's acceptance of the electrons' `y`-angle spread:

- The boosted-dipole photon density is `1/(1 + γ²θ²)²` at photon angle `θ` perpendicular to the `x` oscillation.
- `θ` starts with the beam divergence. Its characteristic function then decays as `exp(−z Λ(s))`, with `Λ(s) = Γ(Δ = s/p)`, the validated decoherence rate of a planar-uniform density.
- The scattering rate is taken at the mean density (×1) and bracketed at ×4, the bound bands' TDS rate on the planes.
- At 17 MeV, `W` is 0.94 at 120 µm (×1). The incoherent `y`-angle RMS is about 5 mrad, against the Highland 21 mrad of random-direction electrons. That fits Kephart's observation that multiple scattering has no perceptible influence on electron CR spectra.

`L` in µm with the detector observable (×1 / ×4). Settings: `H = 56`, `n_cells = 16`, `dz = 62.5 Å`; 54 MeV pending.

| margin [eV] | Si(110) 17 MeV n=1 (20) | n=2 (17) | n=3 (17) | Si(100) 17 MeV (16) |
|---|---|---|---|---|
| 300 | 17.0 / 14.6 | 16.9 / 14.5 | 16.8 / 14.5 | 17.0 / 14.6 |
| ∞ | 35.5 / 26.7 | 34.3 / 25.8 | 33.4 / 25.2 | 34.1 / 25.7 |

Populations alone at the same settings give 18.7 / 18.6 / 18.5 / 18.6 µm (300 eV) and 44.9 / 43.2 / 41.9 / 43.2 µm (∞). With the detector observable and a 300 eV cutoff, every 17 MeV value is within ±20% at ×1. At ×4 all are within ±20% except Si(110) n=1 (0.73). Without a near-barrier cutoff, the detector observable still leaves `L` 1.3–2.1× too long.

### Bound-weighted `y` rate (parameter-free)

Channeled electrons sample the planes. The bound bands' expected loss rates, relative to the mean, give the weights directly (`bound_rate_weights` in the harness): TDS ×1.86 / 1.89 / 1.84 / 1.69 and electronic ×1.05 / 1.03 / 1.05 / 1.02, for Si(110) 17 MeV, Si(100) 17 MeV, Si(110) 54 MeV and Si(100) 54 MeV. With `Λ = f_TDS Λ_TDS + f_el Λ_el` and a 300 eV cutoff:

| Case | `L` [µm] | Kephart [µm] | ratio |
|---|---|---|---|
| Si(110) 17 MeV n=1 | 16.2 | 20 | 0.81 |
| Si(110) 17 MeV n=2 | 16.1 | 17 | 0.94 |
| Si(110) 17 MeV n=3 | 16.0 | 17 | 0.94 |
| Si(100) 17 MeV | 16.1 | 16 | 1.01 |
| Si(110) 54 MeV | 32.1 | 36 | 0.89 |
| Si(100) 54 MeV | 32.4 | 24 | **1.35** |

Settings are `H = 56` (17 MeV) or `H = 64` (54 MeV), `n_cells = 16`, `dz = 62.5 Å`. These were refitted from the stored band signals with the same formulas the harness now uses. At 54 MeV the 300 eV result is converged in `H` (`H = 64` and `80` both give 41.1 µm for populations alone); `dz` 62.5 → 31.25 Å moves `L` by about 3%.

Five of the six values are within ±20%. The outlier is Si(100) at 54 MeV, at 1.35. With the cutoff dominating, the model gives nearly equal `L` for both 54 MeV planes (32.1 and 32.4 µm), whereas Kephart measured 24 and 36 µm. Without a cutoff, the bound-weighted detector observable gives 1.6–2.75×.

## Interpretation

- **The occupation length is not set by the kernel alone.** It depends on where electrons stop counting as channeled. Planar rechanneling is a one-dimensional random walk in transverse angle, which keeps returning to the bound band. With a band-edge sink only, `L` converges in `H`, but to roughly twice the measured value.
- **A fixed 300 eV cutoff matches all four 17 MeV measurements within 20%,** at ratios 0.95–1.18. It overshoots at 54 MeV (ratios 1.18 and 1.76). A 100 eV cutoff matches Si(100) at 54 MeV (0.88) but not Si(110) (0.58). No single energy cutoff fits both energies. In angle, the 17 MeV match corresponds to about 5.9 mrad (about 3.8 critical angles).
- **The solver traces over `q_y`, but the experiment does not.** Kephart detected forward photons, so electrons scattered along the planes beyond about `1/γ` (29 mrad at 17 MeV, 9 mrad at 54 MeV) radiate outside the acceptance. That reduces the measured yield with depth in a way the transverse-state population alone does not.

## Open decisions

- **Decided.** On 2026-10-15 the maintainer chose the detector observable as the comparison basis, over a pure dechanneling criterion or a bare calibration factor.
- **Still open.** The detector observable does not by itself reproduce Kephart without a near-barrier dechanneling cutoff. A 300 eV cutoff matches at 17 MeV. Whether one cutoff can be declared as a calibration across both energies depends on the pending 54 MeV runs. Without the cutoff, 54 MeV at `H = 64` gives 2–3× Kephart, and those runs are converged in neither `dz` nor `H`. A declared cutoff would enter model identity and provenance.

## Independent boundary review, 2026-10-20 (#522)

### Derivation recorded before implementation inspection

Fresh verifier read the ledger, signatures/docstrings, proposal D4(i), and
ADR-0018 before inspecting implementation bodies. The declared collapse is a
projected open-system model, not a first-principles derivation of a 300 eV
escape energy. Let $P$ retain the evolving subspace, $Q$ contain represented
sink states, $A=\sum_a L_a^\dagger L_a$, and
$J(\rho)=\sum_a L_a\rho L_a^\dagger$. For $\rho=P\rho P$,

$$
\frac{d\rho}{dz}=-\frac{i}{\hbar v}[PHP,\rho]
-\frac12\{PAP,\rho\}+PJ(\rho)P.
$$

The represented free and omitted-band flux rates must be

$$
f=\operatorname{tr}[QJ(\rho)Q],\qquad
 o=\operatorname{tr}(A\rho)-\operatorname{tr}[(P+Q)J(\rho)(P+Q)].
$$

Thus $d\operatorname{tr}\rho/dz+f+o=0$. Units are inverse length;
positivity follows from positive jump maps and the interpretation of the
omitted contribution as a projection complement. With no scattering both
fluxes vanish; with no sink, represented jump gain cancels loss. An energy
basis change adds the nonnegative trace discarded by the new projector.
None of these identities fixes the physical boundary, makes an omitted-band
flux negligible, or establishes a detector-weighted occupation length.

For a classical continuum potential the transverse invariant is
$E_\perp=pv\psi^2/2+U(x)$. Barrier crossing and asymptotic angular escape
are different criteria when above-barrier particles can rechannel. A
boundary based on accumulated scattering variance, residence time, or a
near-barrier energy is therefore a model hypothesis requiring independent
observable and basis-convergence tests.

### Implementation comparison and bounded result

Inspection of `DensityMatrixSolver._propagators`, `jump`, `entry_from_plane`,
`_reproject`, and `propagate` matches the projected equations: the retained
loss is projected on both sides; jump gain is split into retained and free
diagonals; `loss_rate - band_trace` is recorded as overflow. The finite-step
flux quadrature is approximate, not an exactly conservative discretization.
The overflow includes missing transfer-kernel weight as well as outgoing
particle-band weight. It cannot by itself establish a momentum distribution
outside the particle band.

The exploratory harness uses

$$
E_{\rm cut}=U_{\min}+a^2(U_{\max}-U_{\min}),\qquad
m=(a^2-1)(U_{\max}-U_{\min}).
$$

This is dimensionally consistent, becomes the barrier at $a=1$, and loses
its energy scale when the potential vanishes. It defines an angle measured
from the potential minimum; it is not the production handoff angle measured
from $U_0=0$. The gap candidate's three-gap, width-fraction, and half-window
guards are declared numerical hypotheses. Neither follows as an irreversible
escape law from a Bloch Hamiltonian. [Sen and Lynn (2014)](https://arxiv.org/abs/1407.6053)
explicitly introduce a phenomenological dechanneling model and emphasize
rechanneling; their spectrum does not supply these threshold constants.

The four `a45-final-1577/run-{0,1,2,3}.json` records (revision `54b29d7c`,
$H=64$, 32 cells, $\Delta z=15.625$ Å) use $a=4.5$, margins
405.0213 eV for (110) and 224.4037 eV for (100). Directly recomputing their
ratios gives:

| Case | Bound-weighted ratio | Mean-weighted ratio | Bound-weighted ±20% |
|---|---:|---:|---|
| Si(110), 17 MeV, n=1 | 0.947807 | 1.005768 | pass |
| Si(110), 17 MeV, n=2 | 1.101662 | 1.168839 | pass |
| Si(110), 17 MeV, n=3 | 1.092553 | 1.158928 | pass |
| Si(100), 17 MeV | 0.884007 | 0.926075 | pass |
| Si(100), 54 MeV | **1.204927** | **1.276443** | **fail** |
| Si(110), 54 MeV | 1.072245 | 1.169043 | pass |

The first exact numerical failure is $28.9182399613/24=1.2049266651>1.2$;
rounding to 1.20 would falsely pass it. [Kephart's published abstract](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.40.4249)
independently confirms the 24 µm reference; fine band-resolved Table I data
and its tolerance remain the repository transcription because the full
article was not accessible during this review. The candidate's cumulative
overflow is 0.030686, 0.009950, 0.024348, and 0.083738 in run order;
balance defects are −0.001578, −0.000869, −0.002136, and −0.003935.
Small balance error does not repair the failed observable or certify a
basis-independent escape law. This four-record set has only one basis and
step configuration, so it cannot independently establish convergence.

**Filters:** units pass; projected-model limits pass; signs/conventions pass
only with the potential-zero and missing-kernel qualifications above.
**Re-derivation:** matches the declared projected accounting; differs from
any claim that the exploratory threshold is a derived physical escape law.
**Verdict:** `rederived` for projected accounting, `discrepancy` for promotion
of the $a=4.5$ candidate as a six-case replacement. The 300 eV declared
calibration and its previously documented Si(100) 54 MeV outlier remain
separate retained evidence; this review neither replaces that calibration
nor establishes its first-principles validity.

**Suggested ledger edit (owner applies):** retain the existing declared
300 eV anchored scope; add #522 notes giving the rejected candidate, exact
1.2049266651 failure, finite overflow, and unresolved convergence. Do not
extend `anchored` to the candidate or describe all six cases as passing.
No human sign-off is implied.
