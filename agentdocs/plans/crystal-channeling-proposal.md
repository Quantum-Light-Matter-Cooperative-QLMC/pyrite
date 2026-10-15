# Crystal channeling: implementation proposal (#56)

**Status:** Draft proposal for review (revised 2026-10-14 after the prior-art check, §12). Nothing here is implemented or ledgered.
**Issue:** [#56](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/56)
("quantum treatment, possibly MCWF state evolution + scattering/dechanneling,
below 100 MeV").
**Builds on:** [`channeling-radiation-physics.md`](../../docs/research/physics/channeling-radiation-physics.md)
(physics walkthrough: potential → eigenstates → occupation → evolution → emission)
and [`relativistic-electron-transport.md`](../../docs/research/physics/relativistic-electron-transport.md)
(architecture). This proposal keeps the walkthrough's physics, but revises two of
its decisions: depth evolution and how the module couples to transport.

## 1. Summary

Add an opt-in **quantum channeling zone** at the entrance of an aligned crystal.
While an electron is in the zone, its transverse state lives in the crystal's
continuum potential. Incoherent thermal-diffuse and electronic scattering enter
as Lindblad jump operators. For planar channeling, the state is evolved as a
**deterministic density matrix**. For axial, keV and two-electron work, it is
evolved by **Monte Carlo wave-function (MCWF) quantum jumps** over the same
operators (§3 D2, revised after the §12 prior-art check). Channeling radiation (CR) is
scored with an expected-value estimator. When the electron dechannels, it leaves
the zone as an ordinary classical launch state. From there, the existing native
transport cores take over. Native transport, PXR/CBS, bremsstrahlung,
characteristic emission and every existing checkpoint stay unchanged when
`channeling_model="off"`, the default.

Recommended scope for the first validated release:
**planar** channeling, electrons, 1–100 MeV,
**silicon and diamond**, CPU only. Axial channeling, the keV Bloch-wave regime,
CUDA and other materials are later tiers.

## 2. Context and constraints found in the repo

| Fact | Consequence |
| --- | --- |
| Transport is amorphous: single elastic scattering from ELSEPA/Mott random-atom tables (`transport/`). The crystal enters only through PXR/CBS emission. | Channeling is a new transport *regime*, not a parameter of the elastic model. It is an ADR-0017 **tier 3** change: a new native event and an additive spectral channel. |
| `materials.crystal.U_g` already computes $4\pi e^2\sum_j (Z_j - f_j)\,e^{i\mathbf g\cdot\mathbf R_j}e^{-W_j}/(g^2V)$. That is the Mott–Bethe electrostatic Fourier coefficient. | The continuum potential comes almost for free. Use `use_henke=False` and real $f_0$ only; anomalous terms must not enter the electrostatic potential. |
| Geant4/g4ppyy appears nowhere in `src/` or `pyproject.toml`. Native transport now reaches the 100 MeV ELSEPA wall. #439 covers 30–100 MeV and #440 covers >100 MeV. | Dechanneled electrons should go to **native** transport, not to Geant4 as the existing design doc proposes. That doc's Geant4 ownership map needs revising. |
| `transport.secondaries` already launches tracks from interior points with explicit launch state and keyed streams (`launch_kind` 2/3). | This gives a ready seam for the handoff after dechanneling. |
| `events.py` has a reserved event-code mechanism (`RESERVED_EVENTS`, `check_segment_event_contract`). | Add `EVENT_DECHANNEL` and possibly `EVENT_CHANNEL_EXIT`. |
| Catalog `B_ang2 = 0.6` placeholders exist for many crystals. HOPG has `mosaic_fwhm_deg = 0.8` (14 mrad). | Placeholder-B materials are ineligible. HOPG mosaicity is comparable to or larger than $\psi_c$ (27 mrad at 30 keV, 2–6 mrad at 1–10 MeV), so channeling is mostly washed out there. Start with Si and diamond. |
| #25 (user-defined crystal cuts) adds independent face/lattice orientation. | Channeling needs the channeling plane/axis direction relative to the beam, independent of the surface. Either depend on #25 or add a narrow plane selector (D7). |
| The validation rules call for a derivation, units, a limiting case, a `Validation:` id and a ledger row for every claim. Agents stop at `rederived`/`anchored`; only humans sign off (#277). | Every equation in §6 becomes a ledger row. |

Regime numbers for Si(110) ($d_p=1.92$ Å, $U_0\approx22$ eV):

| $T$ | $pv$ | $\psi_c$ | bound levels $n_b$ | forward CR $\hbar\omega$ |
| --- | --- | --- | --- | --- |
| 30 keV | 58 keV | 27 mrad | ~1–2 | ~10–30 eV (EUV; weak) |
| 1 MeV | 1.34 MeV | 5.7 mrad | ~2 | 0.1–0.35 keV |
| 10 MeV | 10.5 MeV | 2.0 mrad | ~7 | 4–17 keV |
| 100 MeV | 100 MeV | 0.66 mrad | ~20 | 0.1–1 MeV |

Above ~100 MeV, $n_b\gg1$ and quasiclassical methods take over (Baier–Katkov,
Kostyuk's ChaS); that regime belongs to #440's successor. Below ~1 MeV only
0–2 bound states survive, and the physics becomes many-beam Bloch-wave
diffraction (ECCI/ALCHEMI territory).

## 3. Decision points and options

Each decision lists options with pros and cons, followed by a recommendation.
D2 and D4 are the core decisions.

### D1. Energy regime of the first release

| Option | Pros | Cons |
| --- | --- | --- |
| **A. 1–100 MeV planar CR** | Cleanest physics: few bound states and a paraxial equation. Strong anchors: Watson/Pantell 1–3 MeV Si, Kephart 17/54 MeV occupation lengths, ELBE/Azadegan diamond 14–34 MeV, the Fermilab low-energy CR program. CR lands in the keV detector band. | Depends on #439 for physically supported transport after handoff above ~30 MeV. Not PyRITE's current headline regime (30 keV table-top). |
| **B. 10–300 keV Bloch-wave channeling** (orientation-dependent primary penetration, ALCHEMI-like characteristic modulation, ECCI contrast) | Directly relevant to the current 30 keV workflows. Changes characteristic and PXR/CBS yields by percent to tens of percent near zone axes. | Many-beam, with essentially no bound states. Dechanneling happens within tens of nm, and backscatter leaves the paraxial approximation. Anchors are imaging contrast rather than spectra. Effect on detector-integrated spectra is likely small. |
| **C. One engine, staged regimes** | The paraxial transverse Schrödinger equation with mass $\gamma m$ is the same equation in both regimes; multislice TEM uses it at keV. One validated engine serves both. | Requires a basis that survives the many-beam limit (D3), and more up-front design. |

**Recommend C, with A as the first validated tier.** Build the engine to be
regime-agnostic, but anchor and ship 1–100 MeV planar first. Keep B as a later
tier with its own validation issue.

### D2. State-evolution method (core)

Incoherent scattering (TDS from thermal vibrations, electronic excitation) acts
on the transverse state as an open quantum system. For jump operators
$L_k$, the ensemble obeys a Lindblad equation:

$$
\frac{d\rho}{dz}=-\frac{i}{\hbar v}[H_\perp,\rho]
+\sum_k\left(L_k\rho L_k^\dagger-\tfrac12\{L_k^\dagger L_k,\rho\}\right),
$$

where, for scattering with transverse momentum transfer $q$ off a density
$n(x)$, $L_q=\sqrt{n(x)\,d\sigma/dq}\;e^{iqx}$. The options differ in how this
equation is solved.

| Option | How | Pros | Cons |
| --- | --- | --- | --- |
| **A. Diagonal rate (Pauli master) equation.** This is the existing walkthrough's choice. | Evolve populations $P_n(z)$ with golden-rule rates $W_{nm}$. | Cheapest, deterministic, and validated against the literature (Kephart; Azadegan's code). Testable to solver tolerance. | Drops coherences, so thin crystals below the beat length $\lambda_{nm}$ (tens–hundreds of nm) are wrong; that is why the walkthrough flags them invalid. Ensemble-only, so there is no per-electron state for handoff, coherent emission or bunch coherence. Off-diagonal TDS contributions are lost. |
| **B. Deterministic density matrix (Lindblad/Redfield)** | Integrate $\rho$ ($N\times N$) along $z$. | Exact ensemble with coherences, deterministic, no MC noise. | $O(N^2)$ state and $O(N^3)$–$O(N^4)$ superoperator work. Fine for planar ($N\lesssim150$), heavy for axial ($N\sim10^3$). Still ensemble-only: dechanneling hands off a *flux*, so the classical handoff needs separate sampling and loses per-electron correlation (entry point ↔ exit state ↔ emitted photons). |
| **C. MCWF / quantum jumps.** This is the issue's suggestion. | Each electron carries $\psi$ ($N$ amplitudes). It evolves under $H_\text{eff}=H_\perp-\tfrac{i\hbar v}{2}\sum_k L_k^\dagger L_k$, and jumps $\psi\to L_k\psi/\lVert\cdot\rVert$ with probability $\delta z\,\langle L_k^\dagger L_k\rangle$. | The ensemble average reproduces the Lindblad result exactly. The method is **per-electron**, like PyRITE's whole architecture: keyed RNG streams, events become event rows, and handoff happens at a concrete jump. Memory is $O(N)$ per electron, so axial and GPU are feasible. Coherences are retained, so thin crystals are valid. Each jump has a physical $q$ and can deposit energy-loss bookkeeping. | Adds MC noise to quantities A/B give deterministically; controlled by an expected-value CR estimator (D5). A single trajectory is an unravelling, not a unique physical history: only ensemble observables are physical. This needs a documented rule that nothing scores per-trajectory coherence as physics. Requires sampling continuous-$q$ jump kernels. More code than A. |
| **D. Frozen-phonon multislice (real-space split-step)** | Propagate $\psi(x)$ (or $\psi(x,y)$) with FFT split-step through stochastically displaced atom potentials; add an absorptive potential for electronic loss. | Standard in TEM (abTEM, μSTEM, Dr.Probe). Covers keV and MeV and the full Bloch/continuum without basis truncation. Axial comes naturally. TDS needs no rate model. | $\sim10^4$ slices/µm with an FFT each, per electron: GPU-only in practice. Electronic dechanneling still needs absorptive/MCWF terms. Extracting CR needs projection onto bound states at every step. Frozen phonon is elastic, so energy-loss coupling is ad hoc. |
| **E. Classical/quasiclassical trajectories in the continuum potential** | Newton in $U(x)$ plus MSC kicks (ChaS-style), Baier–Katkov emission. | Simple, and fits the existing segment-based emission. | Invalid below ~100 MeV for electrons: it cannot produce discrete CR lines. Useful only as a $\gamma\to\infty$ limiting-case test and for the >100 MeV handoff. |

**Recommendation, revised after the prior-art check (§12).**

- **Planar production: B.** Use a deterministic density matrix, integrated
  split-operator style ("density-matrix multislice"). The coherent step runs
  in the eigenbasis; the dissipative step is an elementwise kernel on
  $\rho(x,x')$ in position representation. One noise-free solve covers the whole
  beam-divergence ensemble, because $\rho_0$ is the divergence-averaged mixed
  state. It is testable to solver tolerance, which makes it the easiest path to
  validate.
- **Axial, keV and two-electron: C (MCWF)**, sharing the same operator layer.
  At $M\sim64^2$ grid points, $\rho$ has ~$10^7$ entries, which is where MCWF's
  $O(M)$ per trajectory wins.
- **Cross-checks.** The MCWF ensemble mean must equal B (a cross-check that
  costs little because the operators are shared). A is a limiting case
  (dephasing-dominated). E is a $\gamma\to\infty$ limit test. D (frozen-phonon
  multislice) runs through **abTEM as an optional external oracle** (§12), not
  as production code.

### D3. Basis / representation

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Bloch eigenbasis** (plane-wave diagonalization, $N$ ≈ 64–150 bands × few $k$) | Free evolution is exact phases. Dipole matrix elements $x_{if}$ and bound/free classification are direct. Jump matrices $\langle m\lvert e^{iqx}\rvert n\rangle$ are precomputed on a $q$ grid. Cheap per step. | Must re-project as $\gamma(z)$ drifts (negligible at MeV, important at keV). Truncation is needed for high-lying continuum states, and dense continua (axial, keV) blow up $N$. |
| **B. Real-space grid** (split-step FFT) | Jumps are trivial multiplications. No truncation. Natural for 2D axial and keV. | Needs a bound-state projection to score CR. Smaller steps. More FLOPs per electron. |
| **C. Hybrid**: eigenbasis for bound + near-barrier states, plane waves above | Compact where it matters. | Extra complexity at the seam; risk of non-orthogonality bugs. |

**Recommend A for planar MeV**, with $H_\text{eff}$ propagators precomputed per
$(\gamma\text{-node}, \delta z)$. Revisit B for axial/keV tiers.

### D4. Coupling to existing transport (handoff)

| Option | How | Pros | Cons |
| --- | --- | --- | --- |
| **A. Entrance zone + one-way handoff** | The channeling module owns the electron from entry until dechanneling, then emits a classical launch state $(\mathbf r,\hat{\mathbf v},T,t)$. Native cores run it like a secondary launch. | Reuses all native cores unchanged (bit-for-bit when off). Clean ownership: native MSC is *off* inside the zone, because the jump operators *are* the multiple scattering there, so there is no double count. Per-electron correlation is retained. | Rechanneling of quasi-free electrons is omitted, which is second order at µm dechanneling lengths. Needs a defensible dechanneling criterion. Zone segments need a representation for PXR/CBS (D6). |
| **B. Fully embedded state in native cores** | Native Numba/CUDA cores gain a channeled flag plus $\psi$ per electron. | Allows rechanneling and a single loop. | Heavy changes to every core (CPU, CUDA, LUT, shell, radiative specializations). Large regression surface. Violates the "smallest owning change" rule. |
| **C. Analytic precompute + injected source** | Run deterministic Lindblad (D2-B) for the beam ensemble, then inject the dechanneled flux as a depth-distributed launch source for ordinary transport. | No per-electron quantum state. Simplest integration. | Loses per-electron correlations. The ensemble flux must be resampled into electrons. Divergence/footprint coupling needs care. |

**Recommend A (zone + one-way handoff).**

- *Planar (deterministic $\rho$):* the collapse rule below acts on the ensemble.
  After each jump term, the part of $L\rho L^\dagger$ projected onto
  $\mathcal F$ leaves as a recorded dechanneling flux
  $dP/dz\,dE_\perp$. Per-electron launch states (handoff depth, angle, time
  $z/v$, energy) are then **sampled from that flux** with keyed RNG streams.
  Nothing observable is lost by doing this:
  - spectra are expectation values;
  - planar entry is translation-invariant modulo $d_p$;
  - different electrons' CR emission is incoherent.
- *Axial/keV (MCWF):* the same rule applies per trajectory, as stated below.

The dechanneling criterion is itself a sub-decision:

- **(i) Collapse at incoherent jumps (recommended).** After each jump, project
  $\psi$ onto the bound (+ near-barrier) subspace $\mathcal B$ versus the free
  subspace $\mathcal F$. With probability $\lVert P_\mathcal F\psi\rVert^2$,
  hand off: sample $E_\perp$ from the free component and set the angle to
  $\sqrt{2E_\perp/pv}$, with the sign taken from the momentum. Otherwise,
  renormalize into $\mathcal B$. *Pro:* it is a physical measurement-like
  localization, and unbiased in the ensemble. *Con:* the subspace cutoff (barrier
  + margin) is a model parameter whose sensitivity must be reported.
- (ii) Threshold on bound weight $\lVert P_\mathcal B\psi\rVert^2<\epsilon$.
  *Pro:* deterministic. *Con:* biased, because it hands off partial channeled
  weight.
- (iii) Keep free states in the basis until a decoherence length.
  *Pro:* most complete. *Con:* large $N$; it is effectively D2-D.

Dechanneled electrons also leave at the exit face or at the energy cutoff. A
zone-length cap ($z_\text{max}$, e.g. 5× the dechanneling length) forces
handoff and records it with an audit counter.

### D5. Emission scoring

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Expected-value estimator**: each step adds $\langle\hat d_\omega^\dagger\hat d_\omega\rangle_{\rho(z)}\,\delta z$ per transition-frequency cluster $\omega$, where $\hat d_\omega=\sum_{E_i-E_f\approx\hbar\omega} x_{fi}\lvert f\rangle\langle i\rvert$ groups transitions degenerate within the linewidth, spread over the line shape | Every electron contributes, so variance is low (yields are only ~$10^{-3}$–$10^{-2}$ photons/e per line). Matches the coupled-BremsLib expected-value precedent. Exact for degenerate ladders: it reduces to $\sum\lvert c_i\rvert^2$ per line when transitions are nondegenerate, which they are for anharmonic electron wells. A harmonic-potential test (§12) checks it against the classical dipole limit. | Cross-line interference in the time-resolved field is not represented; time-integrated spectra do not need it. |
| B. Sample radiative jumps as photons | Exact MCWF unravelling, and gives recoil per photon. | Extremely noisy at these yields. |
| C. Coherent amplitude per trajectory (field sum) | Could feed `spec_coherent`/bunch coherence. | MCWF trajectories are unravellings, so per-trajectory field interference is not physical without the quantum regression theorem. Research item, not v1. |

**Recommend A.** Bound→bound lines in v1. Free→bound capture is a P3
diagnostic (see §11, decision 4); bound→free cannot radiate (see there).
**Free→free is excluded**: that is CBS, owned by the PXR/CBS kernel. A
regression test enforces this: zero bound states must give exactly zero
channeling emission. Absorption goes through the existing `_stack_tau`.
Line shape is occupation Lorentzian ⊗ Doppler/solid angle ⊗
$2\,\delta\gamma/\gamma$ beam spread.

Output surface:

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Diagnostic API first** (`channeling_spectrum(scene, …)` returning arrays, not persisted) | Fast to validate, no checkpoint migration. | Not visible in apps or sweeps. |
| **B. New additive channel `spec_channeling`** (ADR-0017 tier 3) | First-class: totals, plots, detectors. | Needs `_spectral_components`, checkpoint split/reconstruct, replay/recompute, directional stacking and an identity marker. |

**Recommend A, then B** once the physics is anchored.

### D6. PXR/CBS during the channeled zone

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Emit zone as straight mean-direction segments to the existing PXR/CBS kernel** | Preserves PXR from the first µm (important for thin targets). The kinematic PXR mainly depends on velocity. | Ignores channeled-state modulation of CBS/PXR (non-uniform sampling of the unit cell). Documented omission. |
| B. Exclude zone segments from PXR/CBS | Simple; no claim made. | Underestimates PXR for thin crystals. |
| C. Channeled-state-weighted structure factors | Physically richest (related to ALCHEMI). | Research-grade; new derivation and validation. |

**Recommend A**, with zone segments tagged so B/C can be evaluated later.
Bremsstrahlung inside the zone keeps using the amorphous estimator on those
segments (zone length ≪ total path; error recorded in the validity budget).

### D7. Configuration and orientation

- Selector: `Physics.channeling_model ∈ {"off", "planar"}` (later
  `"axial"`, `"bloch-keV"`; the evolution backend is an implementation detail, not the selector), plus `channeling_plane` (hkl) and the numerics
  `channeling_basis_size`, `channeling_dz_ang` and `channeling_zone_max_ang`.
  Absent key ⇒ `"off"` (historical omission rule). New `CHANNELING_MODEL` marker
  enters `dataset_identity`/`case_content_key` **only when not off**, so existing
  digests do not rekey.
- Orientation source:
  - **(a) Depend on #25's independent face/lattice orientation (recommended).**
    One orientation truth shared by transport, PXR/CBS and channeling.
  - (b) A narrow `channeling_plane` plus a tilt in the existing frame. This
    unblocks earlier, but risks a second orientation convention, which #25
    exists to prevent.
- Gate rules: refuse when $n_b<1$ for the requested plane and energy (outside
  the planar tier), when the material has a placeholder B, or when the energy is
  above 100 MeV. Warn on "planar" alignment within a few $\psi_c^\text{axial}$
  of a low-index axis. Warn when mosaic FWHM is ≫ $\psi_c$; then require the
  existing `_mosaic_quadrature` average.

### D8. Code placement

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Static physics in `materials/channeling/`** (potential, Debye $B(T)$, eigensolve, matrix elements, deterministic; `materials.crystal` dependency only) **+ dynamics in `montecarlo/channeling/`** (density-matrix solver, MCWF stepper, handoff, emission) | Respects the import-linter layering (materials below montecarlo). Static pieces are reusable by notebooks and validation without importing Monte Carlo. | Two packages. |
| B. Single `montecarlo/channeling/` | One place. | Notebooks and validation pull in montecarlo for pure-crystal quantities. |

**Recommend A.**

### D9. Backends

CPU NumPy/Numba first: planar $N\le150$ with precomputed $N\times N$ propagators
is cheap. Add a CUDA kernel (one thread or warp per electron, batched small
mat-vec) only after the CPU path is anchored, following the
`_jit_*` twin pattern and its parity tests. Heavy runs go through
`pyrite remote`.

## 4. Proposed architecture

```text
per (material, plane, T, γ-grid)                [cached, deterministic]
  materials/channeling/potential.py   U(x) from U_g (real f0, B(T)), n(x), ρe(x)
  materials/channeling/debye.py       B(T) Debye model anchored to catalog B
  materials/channeling/eigen.py       Bloch eigensolve -> E_n(k), u_n, x_if, parity
  montecarlo/channeling/operators.py  jump kernels: TDS (n(x)·dσ/dq), electronic (ρe·dσe/dq)
                                      -> <m|e^{iqx}|n> on q-grid, H_eff propagators

per (case: beam energy, divergence, tilt)          [deterministic, planar]
  montecarlo/channeling/entry.py      initial state: divergence-averaged ρ0 (or ψ0 / product state)
  montecarlo/channeling/density.py    ρ(z): eigenbasis coherent step + elementwise ρ(x,x')
                                      dissipator; collapse sink -> dechanneling flux dP/dz dE⊥;
                                      continuous energy loss, γ(z) re-projection
  montecarlo/channeling/emission.py   CR spectrum from ⟨d_ω† d_ω⟩ on ρ(z) (expected value)

per electron (inside transport driver, before native cores)
  montecarlo/channeling/handoff.py    sample handoff (z, angle, t, E) from flux, keyed streams ->
                                      launch state -> native cores (secondaries seam);
                                      zone mean segments (tagged) -> PXR/CBS + brem scorers

scalable backend (axial / keV / two-electron; same operators)
  montecarlo/channeling/mcwf.py       per-trajectory unravelling with the same collapse rule

validation only (optional dependency group)
  checks/abtem_channeling_oracle.py   frozen-phonon multislice populations P_n(z)
```

Data flow when on: `campaign` resolves `channeling_model` → `Case` → runner
builds/caches the static profile → deterministic ρ(z) solve per case → transport
driver samples per-electron handoffs → launch states go to existing cores → existing scorers run on all
segments → the CR accumulator returns a diagnostic spectrum (later
`spec_channeling`).

## 5. Phased plan (proposed issues)

Each phase is one GitHub issue, a sub-issue of #56 (triaged 2026-10-14). Quantum coherent emission follow-up: #501.

| Phase | Deliverable | Acceptance (summary) | Depends |
| --- | --- | --- | --- |
| **P0 (#502) ADR + doc reconciliation** | ADR-0018 "quantum channeling zone" recording D2/D4/D5 choices. Revise the research docs: drop the Geant4 handoff (decided), replace diagonal-only evolution with density-matrix (planar) + MCWF (axial/keV), and correct "bound→free" to free→bound capture. | Human-reviewed ADR. Research docs consistent. | — |
| **P1 (#503) Static profile** | Potential, $B(T)$, eigensolver, $x_{if}$, ideal-entry CR line energies (no scattering). | Si(110) depth ≈ 22 eV @ 293 K. Free-potential limit. Harmonic test limit. Emergent parity selection rule. $\sqrt\gamma$ level count. Line energies vs Azadegan's quantum planar CR package for diamond/Si (Zotero `azadeganMathematicaPackageCalculation2013a`) and the Watson/Pantell 1–3 MeV data. | #25 (orientation) |
| **P2 (#504) Open-system dynamics** | Jump kernels, deterministic density-matrix solver with collapse sink and flux output, MCWF stepper on the same operators (CPU), decoherence-length diagnostic $\Gamma(\Delta x)$, optional abTEM frozen-phonon oracle. | Zero scattering ⇒ norm and populations conserved (unitary). Trace + flux = 1. MCWF mean → density matrix within $3\sigma$. $P_n(z)$ vs abTEM frozen-phonon projection within a stated tolerance. Dephasing-dominated limit → rate equation. Occupation/dechanneling lengths vs Kephart 17/54 MeV Si. Conservation: bound + free + handed-off = 1. | P1 |
| **P3 (#505) Emission (diagnostic API)** | Expected-value CR spectrum and angular distribution, line shapes, stack absorption. | Boosted dipole integrates to $A'/\gamma$. $\gamma\to1$ isotropic limit. No bound states ⇒ exactly zero. Doppler denominator matches `line-energy-dispersion` for the same geometry. Watson/ELBE peak positions and widths. | P2 |
| **P4 (#506) Transport integration** | `Physics.channeling_model`, Case/identity marker, `EVENT_DECHANNEL`, handoff through the secondaries launch seam, tagged zone segments into PXR/CBS/brem. | Off ⇒ bit-for-bit vs parent (baseline fixtures). On ⇒ energy/population accounting closes. Replay/recompute/directional either supported or refused before work. | P3, #439 for >30 MeV support claims |
| **P5 (#507) Persisted channel** | `spec_channeling` through `_spectral_components`, checkpoints, apps. | ADR-0017 tier-3 checklist complete. | P4 |
| **P6 Extensions** (separate issues) | CUDA kernel, axial tier with vortex/OAM (#508) (2D, real-space basis), keV Bloch-wave tier, TR comparison at 1–3 MeV (#102/#103), channeled-state PXR/CBS modulation (D6-C). | Per-issue. | P4/P5 |

## 6. Ledger rows (proposed ids)

Reuse the walkthrough's `cr-*` ids for P1/P3 (`cr-mott-bethe-factor`,
`cr-planar-continuum-potential`, `cr-debye-b-of-t`,
`cr-transverse-eigenproblem`, `cr-initial-occupation`, `cr-doppler-line-energy`,
`cr-dipole-rate`, `cr-boosted-dipole-angular`, `cr-line-shape`,
`cr-yield-assembly`). Replace `cr-population-master-equation` /
`cr-dechanneling-rates` with:

- `cr-lindblad-jump-operators`: $L_q=\sqrt{n(x)\,d\sigma/dq}\,e^{iqx}$ (TDS)
  and the electronic analogue. Source: open-system derivation of the absorptive
  (Yoshioka) potential. Anchor: imaginary part reproduces the standard absorptive
  potential.
- `cr-density-matrix-split-operator`: coherent eigenbasis step plus
  position-space dissipator kernel $K(x,x')$. Anchor: zero-scattering
  unitarity, and agreement with the frozen-phonon (abTEM) oracle.
- `cr-mcwf-unravelling`: $H_\text{eff}$ and jump probability. Anchor: ensemble
  mean = density-matrix solver.
- `cr-dechanneling-collapse`: bound/free projection handoff and launch-state
  sampling. Anchor: Kephart occupation lengths, plus sensitivity to the subspace
  cutoff.
- `cr-zone-handoff-accounting`: energy, population and time continuity across
  the handoff.
- `cr-transition-radiation-bound` (kept from the walkthrough): include, or
  exclude with a bound.

Target status for agent work: `anchored`; human sign-off via #277.

## 7. Risks

- **Absolute dechanneling rates** are the least first-principles quantity. The
  walkthrough's escape hatch is a declared global calibration factor; any such
  factor must be visible in identity/provenance, never hidden.
- **Subspace cutoff sensitivity** in the collapse rule. Report spectra across
  cutoffs; if they are sensitive, enlarge the basis (D3-C).
- **Transition radiation at 1–3 MeV** is the same order as the CR lines in the
  soft band. Anchoring against soft-X-ray data without TR may disagree for
  reasons outside this module.
- **Mosaic crystals** (HOPG, many vdW catalog entries) mostly wash channeling
  out. Expect little effect for current catalog favorites; that is a result
  worth stating, not a bug.
- **#439 dependency**: post-handoff transport above 30 MeV lacks radiative loss
  until #439 lands, so support claims there must wait.

## 8. Atomic and material data requirements

Inventory of the existing catalog (2026-10-14):

- Waasmaier–Kirfel $f_0$ through xraydb (`atomic.cromer_mann_f0`; WK fits valid
  to $s=\sin\theta/\lambda\le 6$ Å⁻¹) and Chantler $f',f''$.
- CIF bases.
- **One isotropic `B_ang2` per crystal**: no per-site values, no anisotropy, no
  temperature dependence. 25 of 37 crystals carry the placeholder `0.6`.
- ELSEPA/Mott elastic, SBETHE stopping, PENELOPE shell GOS plus
  `conduction_band.toml` plasmon parameters, and EPDL/Chantler photon
  attenuation.

| # | Need | Used by | Have? | Proposed source | Priority |
| --- | --- | --- | --- | --- | --- |
| 1 | Electron scattering factor $f^e(s)$ accurate to large $s$ (≥ 6 Å⁻¹). Electron wells sit *on the nuclei*, so deep states probe the near-nucleus potential. | potential, TDS kernel | partial: Mott–Bethe from WK $f_0$; the $s\to0$ limit needs analytic handling | Primary: Mott–Bethe on WK. Cross-check: Peng 1996 / Lobato–Van Dyck 2014 / Kirkland parametrizations, which are direct $f^e$ fits valid to $s\sim6$ Å⁻¹. | P1, required |
| 2 | **Per-site** thermal displacement $\langle u^2\rangle$ (or $B_j$) | potential smearing, plane density $n(x)$, TDS rates | no: one value per crystal | Measured/DFT-phonon per-site values: Sears–Shelley; Peng et al. 1996 *Acta Cryst A* DW tables; Gao–Peng 1999 | P1, required for compounds |
| 3 | **Temperature dependence** $B(T)$ | potential depth (several % over 100–400 K), rates | no | Debye temperature $\Theta_D$ per element/site, plus zero-point term; anchored to the RT value. Better: phonon-DOS-derived $B(T)$ where published (Si, C). | P1, required |
| 4 | **Anisotropic** $U_{ij}$ | layered/vdW crystals (HOPG, MoS₂, hBN): in-plane ≠ out-of-plane, so planar (basal) and axial profiles differ | no | CIF `_atom_site_aniso_U_*` when present; literature otherwise | P6 (vdW tier); not needed for Si/diamond (cubic, isotropic) |
| 5 | TDS cross section $d\sigma_\text{TDS}/dq$ | jump operators (dominant dechanneling) | derivable | Einstein model, $\lvert f^e(q)\rvert^2(1-e^{-2W(q)})$, from #1–#3. Correlated phonons (dispersion) are a refinement and need phonon data. | P2 |
| 6 | Local electron density $\rho_e(\mathbf r)$, plus core/valence split | electronic jumps, local stopping | derivable (Poisson on $V_\mathbf g$); valence count in `conduction_band.toml` | IAM superposition first. Validate against the DFT/measured charge density for Si and diamond. | P2 |
| 7 | Transverse-momentum-resolved electronic loss, i.e. $d\sigma_e/dq_\perp$ for valence (plasmon) and core shells | electronic jump kernel | partial: PENELOPE GOS and plasmon $W_{cb}$ give amorphous, $q$-integrated rates | Derive $q_\perp$ dependence from the GOS/Lindhard dielectric model; local-density approximation weighted by $\rho_e(x)$. | P2 |
| 8 | Low-order structure factors measured beyond the independent-atom model (bonding charge) | well shape of shallow/near-barrier states, few % | no | Zuo et al. CBED-refined Si/diamond $V_\mathbf g$; Spackman/Saka | P1 anchor (validation-only) |
| 9 | Mean inner potential $V_0$ | surface refraction at entry | derivable ($\propto\sum f^e(0)/\Omega$) | Measured holography/RHEED values (Si ≈ 12–13 V) as an anchor | P1 |
| 10 | Near-edge photon absorption | low-energy channeling radiation: at 1 MeV, diamond lines fall near the C K-edge (284 eV); Si lines near Si L₂,₃ (99 eV) | coarse: Chantler/EPDL grids miss NEXAFS structure | Measured NEXAFS/Henke fine grids for Si and C, or a declared refusal band | P3 |
| 11 | Dielectric function / plasma frequency | transition-radiation bound (`cr-transition-radiation-bound`) | yes: Chantler $n(\omega)$, plasmon energies | — | P3 |
| 12 | Crystal quality: mosaic spread, dislocation density, miscut | dechanneling, alignment fraction | mosaic only (HOPG) | Per-sample metadata. Si/diamond assume perfect crystal, with a stated limit. | P4 |
| 13 | Thermal expansion | lattice spacing vs $T$ | no | Negligible (~0.1% over 100 K); document and skip | — |
| 14 | Experimental reference data, pinned with provenance under `pyrite.validation` | anchors | no | Watson/Pantell 1–3 MeV Si planar and axial CR; Kephart 17/54 MeV occupation lengths; ELBE/Azadegan diamond 14–34 MeV; Fermilab low-energy CR; Gouanère axial | P1–P3 |

Schema consequence: the crystal record grows from `B_ang2` to per-site thermal
parameters plus a reference temperature and $\Theta_D$ (or a $B(T)$ table).
Existing consumers keep reading the scalar, so PXR/CBS and its identity stay
untouched. Channeling consumes the new fields and refuses crystals that lack
them. Any change to catalog values goes through `regen-golden`.

## 9. Axial vs planar

| Aspect | Planar | Axial |
| --- | --- | --- |
| Potential | 1D: Fourier sum over harmonics $n\mathbf g$ normal to one plane (~30–60 coefficients) | 2D: every $\mathbf g\perp$ axis (~10³ coefficients). String potentials; mixed-composition strings in compounds. |
| Well depth (electrons, Si) | ~20–25 eV | ~10² eV (order of magnitude; to be computed) |
| Critical angle | $\psi_c^{p}$ | ~2–3× larger |
| Bound states at 10 MeV | ~7 | tens to hundreds. Approximate $(n,\ell)$ labels near the string, mixed by the lattice. |
| Photon energies (same $\gamma$) | baseline | ~2–5× higher; more lines, which overlap |
| Yield | lower | higher per electron, but spread over many lines |
| Polarization | linear, along the plane normal | $\Delta\ell=\pm1$ gives circular components; azimuthal pattern differs |
| Dechanneling | ~µm at 10 MeV | low states sit on the strings, so they dechannel fast; doughnut scattering redistributes quasi-free electrons azimuthally, and rechanneling matters |
| Basis | Bloch eigenbasis, $N$≈64–150; precomputed jump matrices on a 1D $q$ grid | eigenbasis $N\sim10^3$ gives $N^2\times n_q^2$ jump tables, which is infeasible. Needs a **real-space split-step** representation (64² FFT), where jumps are pointwise multiplications. |
| Entry | 1D Bloch projection; divergence in one plane matters | 2D $(k_x,k_y)$ projection; both divergence planes matter |
| Alignment gate | must avoid low-index axes within a few $\psi_c^{a}$, otherwise the run is silently axial | must sit within $\psi_c^{a}$ of the axis; planar channeling in the planes crossing the axis contaminates small-tilt runs |
| Overlap with CBS | moderate | strong: axial alignment also boosts coherent bremsstrahlung, so the free→free exclusion test is more critical |
| Data sensitivity | isotropic B suffices for cubic crystals | anisotropic/per-site $u$ matters more (string smearing is in-plane) |
| Validation | quantitative: line energies, widths, occupation lengths | mostly peak order and approximate positions (Watson, Gouanère); fewer published absolute yields |
| Compute | cheap; CPU fine | ~10–100× per electron; CUDA realistically needed for production statistics |
| Relation to keV regime | — | keV zone-axis channeling is the same 2D many-beam problem, so the axial engine is reused by the keV Bloch-wave tier |

**Recommendation.** Ship planar first; it is the validatable tier. Build the
operator layer (jump kernels, $H_\text{eff}$) behind a representation interface
from the start, so that axial plugs in a real-space backend without touching
the MCWF stepper or the handoff. Release axial as `experimental`, with
peak-order anchors only, and pair it with the CUDA work. Keep planar ↔ axial
consistency as tests:

- A 2D potential from strings averaged along one in-plane direction must
  reproduce the planar $U(x)$.
- A beam tilted far from the axis but aligned to a plane must converge to the
  planar result.

### OAM (vortex) electron beams and channeling

**Question:** would an electron beam carrying orbital angular momentum (OAM)
$\ell\hbar$ about its axis couple an OAM mode into the channels?

**Axial: yes, conditionally.**

- *Symmetry.* Near one atomic string the transverse potential is nearly
  cylindrical, so bound states classify as $(n,\ell)$: 1s, 2p±, 3d±…
  The string lattice's $m$-fold point symmetry reduces conservation to
  $\ell \bmod m$. For the projected rotation symmetry of the chosen axis $m$ is
  4 for ⟨100⟩, 3 or 6 for ⟨111⟩, and 2 for ⟨110⟩; P1-axial computes the
  exact values.
- *Coupling.* A vortex centered on a string projects onto states with
  $\ell'\equiv\ell \pmod m$. Measured and simulated case: an atom-sized
  $\ell=\pm1$ probe (aberration-corrected STEM, keV) couples to on-column 2p
  bound states and keeps its helical phase for >50 nm
  ([Xin & Zheng 2012](https://arxiv.org/pdf/1204.0021)).
- *Centering requirement.* OAM about the *beam* axis is not OAM about each
  *string*. A vortex wider than the string spacing (a few Å), or off-center,
  looks locally like tilted plane waves and spreads over many $\ell$. Its
  expectation $\langle L_z\rangle$ then changes with impact parameter and depth
  ([Lubk et al.](https://arxiv.org/pdf/1111.6050)). Averaging over uncontrolled
  impact parameters washes out the selectivity. So the effect needs an
  Å-positioned, atom-sized vortex probe. That exists only at TEM energies
  (keV), not at MeV.
- *Decoherence.* Scattering jumps $e^{i\mathbf q\cdot\mathbf r}$ with random
  $\mathbf q$ direction scramble $\ell$. The $\ell$-imbalance decays over a
  mixing length that the operator layer computes directly (an analogue of
  #501's decoherence budget).
- *Emission signature.* Axial CR transitions obey $\Delta\ell=\pm1$ and emit
  circular polarization about the axis; off-axis photons are twisted, i.e.
  carry photon OAM ([Korotchenko & Kunashenko 2024](https://arxiv.org/pdf/2403.01930)).
  An unpolarized, ℓ-balanced beam gives **zero net helicity**. A vortex beam that
  preferentially populates $+\ell$ states biases $\ell\to\ell-1$ against
  $\ell\to\ell+1$ decay. That produces **net circular polarization / net photon
  OAM in specific lines**, roughly proportional to the surviving
  $\ell$-imbalance. This is a clean, measurable transfer of angular momentum
  from electron to photon. It is also a concrete instance of item 1 in the
  "channeling-specific coherent effects" list below: the crystal's
  position-dependent potential is what lets wavefunction shape matter
  (consistent with the guard-rail).

**Planar: essentially no.** A planar potential has no rotational symmetry, so
$L_z$ is not conserved. A vortex beam enters as a ring of transverse momenta
with azimuthal phases. Only the $k_x$ distribution sets channel populations,
and the phases make $\pm k_x$ coherences that dephase within the decoherence
length. Expect at most a small change in populations, similar to a modified
divergence profile. There is no OAM mode in the channel and no
helicity-resolved CR.

**Implementation consequences (axial tier).** No new atomic data is needed. In
the operator layer:

- the initial state accepts a Laguerre–Gauss/Bessel vortex with explicit
  impact parameter and probe size (the general-initial-state interface);
- an observable projects $\rho$ onto cylindrical harmonics about a chosen
  string, giving $P_\ell(z)$ and $\langle L_z\rangle(z)$;
- emission records Stokes parameters and photon-OAM content per line, rather
  than linear polarization only;
- tests: an ℓ-balanced input gives zero net helicity (symmetry); a centered
  $\ell=\pm1$ probe gives dominant 2p population (Xin & Zheng anchor); an
  off-center probe loses selectivity (Lubk anchor).

Regime: keV axial channeling (60–300 keV TEM vortex probes), which is the
same 2D engine as the keV tier. Physics scope belongs to the axial issue; the
quantum-correlation side (vortex as a shaped state) cross-links to #501.

## 10. Reuse for quantum coherent CXR emission (future)

**Related long-term goal:** super-/subradiant CXR from free electrons whose
quantum state is shaped or entangled in transverse momentum. Sources in Zotero:

- Karnieli et al., PRL 127, 060403 (2021), `karnieliSuperradianceSubradianceDue2021`:
  path-entangled Bell-state electron pairs give super- or subradiance that no
  classical mixed state can reproduce.
- Wong et al., Nat. Commun. 12, 1700 (2021), `wongControlQuantumElectrodynamical2021`:
  shaped wavepackets interfere QED pathways and reshape bremsstrahlung
  directionality and spectra.

This is distinct from `docs/research/physics/superradiant-pxr.md`, which treats
*classical* microbunching (Feranchuk 2022). Reuse of the channeling stack:

| Channeling component | Reuse for quantum coherent emission |
| --- | --- |
| Transverse Hamiltonian $H_\perp$ (mass $\gamma m$, crystal potential) and its propagator | Propagates an arbitrary shaped/entangled transverse wavefunction through the crystal. Identical physics. |
| Matrix elements $\langle f\lvert e^{i\mathbf q\cdot\mathbf r}\rvert i\rangle$ (jump kernels) | Same object as the first-order QED emission amplitude $\langle f\lvert e^{-i(\mathbf k+\mathbf g)\cdot\mathbf r}\rvert i\rangle$ for quantum PXR/CBS. |
| Lindblad jump operators (TDS, electronic) | **Decoherence budget.** A superposition separated by $\Delta x$ decoheres at $\Gamma(\Delta x)=\int dq\,w(q)\,[1-\cos(q\Delta x)]$. Small-$q$ plasmon losses barely decohere small separations; TDS (large $q$) does. This is the first question any quantum-enhancement claim must answer: does the coherence survive the formation length? |
| MCWF unravelling + expected-value estimator | Two-electron emission rates are *linear in the two-body density matrix*, so an MCWF unravelling on the product space, scored as $\langle\psi\lvert\hat A\rvert\psi\rangle$ per trajectory, is unbiased. |
| Physical-population normalization (`superradiant-pxr.md`) | Its pair term $N(N-1)\lvert\langle\mathcal E\rangle\rvert^2$ generalizes to $\sum_{a\ne b}\langle\hat{\mathcal E}_a^\dagger\hat{\mathcal E}_b\rangle$ from the reduced two-body state. Classical i.i.d. is the separable limit and must be recovered as a regression test. |
| Recoil to first order in $\hbar\omega/E$ | Karnieli's effect *relies* on electron recoil distinguishing final states, so it cannot be dropped there. |

### Design adjustments so that reuse is real, not aspirational

1. **D3 (basis):** don't truncate to bound + near-barrier states as a
   structural assumption. Quantum PXR/CBS needs free and continuum states, so
   the hybrid or real-space representation should be reachable behind the same
   operator interface. Planar CR can still *choose* a truncated basis.
2. **D5 (emission):** implement emission as a generic first-order transition
   operator, $\hat A_{\mathbf k,\epsilon}\propto\hat{\mathbf j}\cdot\epsilon^*e^{-i\mathbf k\cdot\mathbf r}$
   evaluated in the chosen basis. The CR dipole formula is then its long-wavelength,
   bound→bound specialization, which serves as a limiting-case test.
   Hard-coding the dipole rate would block reuse.
3. **D5-C remains forbidden as a shortcut.** Squaring per-trajectory MCWF field
   amplitudes across trajectories is still unphysical. Quantum coherence enters
   only through density-matrix-linear observables.
4. **Entry module:** accept an arbitrary initial transverse state (plane wave,
   Gaussian wavepacket, superposition, or a two-particle state on a product
   grid), not only "plane wave at angle $\psi$". The tilted plane wave is the
   default.
5. **Lindblad path is first-class, not just an oracle.** Two-body reduced density
   matrices at small $N$ ($N^2\le10^4$ planar) are exactly where entanglement
   observables are cheapest and least noisy.
6. **Ship a decoherence-length diagnostic** with P2: $\Gamma(\Delta x, T, E)$ per
   material and plane. It is cheap, independently useful, and gates the
   feasibility of quantum enhancement before any emission code is written.
7. **Representation interface handles tensor products**: operators act on a
   named particle axis, so a two-electron state reuses single-particle kernels.

Also relevant: Reese, Spence and Yamamoto, Phil. Mag. A (1984),
`reeseCoherentBremsstrahlungKilovolt1984`, kilovolt CB in zone-axis orientation.
It is a candidate anchor for the keV/axial tier, where the classical
straight-line CBS kernel and the quantum Bloch-state treatment should agree in
the weak-potential limit.

### Channeling-specific coherent and superradiant effects worth tracking

Ranked by plausibility for PyRITE's regimes; full sources in §12.

1. **The crystal is the wavefunction shaper (most relevant, near term).**
   Wong et al. (Light Sci. Appl. 2024) predict >10³ bremsstrahlung enhancement
   when a 20–300 keV electron wavefunction is shaped *externally* to overlap
   atoms periodically. They hold the shaped wavefunction fixed and neglect
   channeling and dynamical diffraction. Bound channeled states do this shaping
   *internally*: for electrons they concentrate $\lvert\psi(x)\rvert^2$ on the
   atomic planes. The density-matrix solver gives $\rho(x,x;z)$ directly, so
   channeling-modulated incoherent emission costs little to add as a diagnostic:
   bremsstrahlung and characteristic X-rays weighted by $\int\rho(x,x)\,n(x)\,dx$
   relative to the uniform value. This is the ALCHEMI-type orientation effect.
   It also quantifies how fast propagation in a real bulk crystal reshapes or
   destroys an externally shaped wavefunction. That is a direct test of whether
   the Wong et al. enhancement survives beyond a monolayer.
2. **Transverse recoil–photon entanglement** (Shi et al., Nat. Commun. 2024).
   Crystal-mediated free-electron radiation with transverse scattering creates
   electron–photon entanglement and strong quantum spectral features. Channeling
   radiation is the bound-state version: each line *is* a resolved transverse
   recoil. The free-state version (quantum CBS/PXR) belongs to the §10 research
   issue. The operator layer (matrix elements of $e^{i\mathbf q\cdot\mathbf r}$
   between transverse states) is the shared piece.
3. **Phase-locked multi-level emission from coherent entry states**
   (Shatnev, arXiv:2603.28827, 2026). This is a single-author preprint claiming
   gain G≈12–31 for GeV positrons in harmonic diamond(110) wells, from
   interference of equidistant n→n−j transitions after Glauber-coherent entry.
   *Caution:* for one emitter, the photon-number rate of a harmonic coherent
   state is $\langle a^\dagger a\rangle=\lvert\alpha\rvert^2$, the same as the
   incoherent level sum. Any real gain must come from spectral or angular
   redistribution, or from multi-particle phase locking. The effect also
   requires equidistant levels: electron wells are strongly anharmonic, so it is
   absent for PyRITE's electron case. Action: the D5 estimator's harmonic-limit
   test adjudicates this claim at no extra cost. Do not adopt the claim.
4. **Dicke superradiance of CR from prebunched beams.** $N^2$ scaling needs
   density modulation at the CR wavelength. At 1–10 MeV that is 0.1–10 nm, out
   of reach. At keV beam energies, CR falls in the EUV (10–30 eV, roughly
   40–120 nm), where ponderomotive/IFEL bunching is closer to feasible. This is
   speculative, but it connects to #25's 13.5 nm interest. It reuses
   `superradiant-pxr.md`'s physical-population normalization unchanged (CR
   emission phase is set by entry time). A precedent is Gevorgyan & Gevorgian
   (arXiv:2412.10205): XFEL-microbunched positrons in crystalline undulators,
   with coherent "oscilex" gain 10³–10⁴.
5. **Entangled-pair super/subradiance in CR lines** (Karnieli et al. 2021,
   applied to channeling). Both electrons must be coincident within the line
   coherence time (lifetime ~ $L/c$ ≈ fs at µm occupation lengths), and the
   final states must be indistinguishable. Requires product-space evolution:
   planar $N^2\le10^4$ fits the density-matrix backend. A research item only.
6. **Twisted (OAM) photons from axial channeling** (arXiv:2403.01930).
   $\Delta\ell=\pm1$ transitions carry orbital angular momentum. Record OAM and
   helicity per component in the axial tier's polarization bookkeeping, rather
   than linear polarization only.

Physics guard-rail for all of the above (Pan & Gover, arXiv:2401.05978;
Karnieli et al., Sci. Adv. 2021): a **single free electron's spontaneous
emission rate is independent of its wavefunction shape**. Shape dependence
needs at least one of:

- a position-dependent interaction (a crystal potential or atoms: items 1–2);
- recoil-resolved or postselected final states;
- multi-particle correlations (items 4–5).

PyRITE should never report a "quantum enhancement" whose mechanism is not one
of these.

**Out of scope** for #56: two-electron states, shaped-beam sources and quantum
PXR/CBS. These belong in a separate research issue. #56 only commits to the
interface choices above, so that the quantum issue does not need to rewrite the
engine.

## 11. Maintainer decisions (2026-10-14)

1. **Regime:** take the most direct, least-duplicated and most validatable route.
   That is **D1-A: 1–100 MeV planar, Si + diamond, first**. It has quantitative
   anchors (line energies, widths, occupation lengths), few states, CPU only,
   and no dependence on the many-beam basis. The keV tier reuses the axial
   real-space engine later, so it is not repeated work.
2. **Geant4 route abandoned.** Dechanneled electrons go to native transport.
   P0 removes the Geant4 ownership map from both research docs.
3. **Orientation depends on #25** (D7-a). P1 is blocked by #25's independent
   face/lattice orientation.
4. **Emission between free and bound states: (a) chosen** (bound→bound lines in v1, plus a P3 capture bound).
   - *Correction to the walkthrough:* bound→free cannot *emit*. Photon emission
     lowers $E_\perp$, and free states lie above the barrier, above every bound
     level. The emitting process is **free→bound radiative capture**. It gives a
     continuum with a sharp low-energy edge at
     $\hbar\omega_\text{edge}\simeq2\gamma^2(U_\text{barrier}-E_n)$ (forward),
     falling off above it.
   - Options:
     - **(a) Bound→bound only, plus a P3 capture bound.** Compute an upper
       bound on free→bound yield from the entry state and the near-barrier
       margin. Promote to (b) only if it exceeds ~a few % of line yield in an
       anchor configuration. *Pros:* simplest; the anchors are line
       positions/widths; no CBS boundary ambiguity. *Cons:* may miss a shoulder
       near the barrier edge for divergent beams.
     - **(b) Score transitions with a bound final state and any initial state
       in the basis (bound + near-barrier margin); exclude free→free.**
       *Pros:* more complete. Free→bound does not double-count first-Born CBS,
       whose final states are plane waves, never bound. *Cons:* the yield
       depends on the collapse margin, because under D4(i) free amplitude only
       lives until the first jump. Margin sensitivity must be reported. Larger
       basis.
     - (c) Full continuum in basis. *Pros:* complete. *Cons:* this is §10's
       quantum-CBS work; out of scope.
   - **Recommend (a)** under the "easiest to validate" criterion. The basis and
     operator interface (§10 adjustments 1–2) keep (b) a scoring flag rather
     than a rewrite.
5. **Calibration factor accepted**, provided it is declared in profile
   metadata, enters case identity, and is reported in provenance.
6. **Agreed:** the §10 quantum coherent emission work becomes a separate
   research issue, blocked by P2 (decoherence diagnostic) and P3 (emission operator).

## 12. Prior-art check: what existing codes do (2026-10-14)

### Electron-microscopy wave-propagation codes (keV–MeV, elastic + TDS)

| Code / method | Representation | TDS / incoherent scattering | Electronic (plasmon/core) | Fit for channeling |
| --- | --- | --- | --- | --- |
| [abTEM](https://abtem.readthedocs.io/) (also Prismatic, MULTEM, Dr. Probe) | Real-space multislice (split-step FFT); PRISM S-matrix reuse for many probe positions | **Frozen phonon**: average intensities over randomly displaced configurations. Justified by the QEP (Born–Oppenheimer) model of Forbes/Allen. | Core loss via Dwyer transition potentials. Plasmons via Mendis Monte Carlo "phase-scramble" ([PR #295](https://github.com/abTEM/abTEM/pull/295), superseded by #385, not merged as of 2026-10). | Ground-truth physics for elastic + TDS. Costly: ~1–2 Å slices through µm thickness, a 2D/3D supercell (frozen phonon breaks planar projection symmetry), $10^2$–$10^3$ configurations. Tracks no energy loss. **Use as oracle.** |
| μSTEM and others | Multislice + **absorptive (imaginary) potential** (Hall–Hirsch / Weickenmeier–Kohl) | Depletion only: TDS flux is removed, not redistributed. | Absorptive | Cheap, but wrong for channeling: intra-band redistribution and rechanneling need the scattered flux kept. A 2026 Hájek/Rusz comparison ([arXiv:2608.24679](https://arxiv.org/abs/2608.24679)) bounds where absorptive models suffice. Useful only as a total-depletion cross-check. |
| Bloch-wave codes; Mendis Bloch + Monte Carlo | Diagonalize once per orientation; Mendis's reformulation is $O(N^2)$/slice | Monte Carlo sampled TDS/plasmon events on Bloch waves | Monte Carlo | Closest to this proposal (eigenbasis + stochastic events). Tens of thousands of Monte Carlo iterations reported for convergence. |
| Bohmian quantum-trajectory Monte Carlo (STEM) | Multislice wave + Bohmian trajectories, inelastic MC along them | — | MC | Research method; no advantage here. |

### Channeling-radiation codes

| Code | Approach | Gap |
| --- | --- | --- |
| Azadegan (Mathematica, CPC 2013; ELBE) | Quantum eigenstates + phenomenological Fokker–Planck / rate-equation dechanneling | Diagonal populations only. The authors state that a proper low-energy quantum dechanneling treatment "seems to be lacking". |
| Wistisen & Di Piazza, PRD 99, 116010 (2019) | Full Dirac-equation emission with recoil and spin | No incoherent scattering. A reference for line positions and recoil at high γ. |
| MBN Explorer, Kostyuk ChaS, Geant4 `G4ChannelingFastSimModel` | Classical/quasiclassical trajectories | Invalid below ~100 MeV for electrons. A limit test only. |

### Verdict

- **Representativeness.** Einstein-model Lindblad jump operators are the Markov
  limit of the same QEP/frozen-phonon physics the TEM field treats as standard.
  Each atom is visited once, and the transverse Einstein displacements are
  independent, so the Markov approximation is benign. Correlated-phonon
  (dispersion) effects mostly alter diffuse/Kikuchi detail; they are recorded as
  a refinement. Absorptive-only models are *less* representative and are
  rejected. Electronic small-$q$ (plasmon) scattering decoheres only large
  separations: $\Gamma(\Delta x)\propto\int w(q)\,[1-\cos q\Delta x]$. That
  matches Mendis's finding that the Bragg/channeling structure survives plasmon
  loss.
- **Efficiency (this changed the recommendation).**
  - Planar, per case: a deterministic $\rho$ with $N\approx100$ costs about
    $10^9$–$10^{10}$ flops for a full µm solve. The coherent step is
    $U\rho U^\dagger$; the dissipator is elementwise on $\rho(x,x')$ at $M\sim256$.
    One noise-free solve also covers the beam-divergence ensemble.
  - MCWF needs ~$10^5$ trajectories to match that noise level.
  - Frozen-phonon multislice would need $10^4$ slices × $10^2$ configurations ×
    a 2D supercell FFT.
  - Hence density matrix for planar, MCWF for axial/keV (where $\rho$ is
    ~$10^7$ entries), and frozen phonon only as an oracle.
  - The PRISM insight (reuse a precomputed scattering operator across many
    input states) is already captured: the coherent propagator per
    ($\gamma$-node, $\delta z$) is precomputed once and shared across cases.
- **Oracle plan.** Add `abtem` to the optional `oracle` dependency group, next
  to Dans_Diffraction and xraylib. Run `checks/abtem_channeling_oracle.py` for
  Si(110) at a few MeV (paraxial, relativistic wavelength) with frozen phonons,
  and project depth-resolved waves onto the P1 bound states to get $P_n(z)$.
  abTEM's own potential parametrizations (Lobato/Kirkland) differ from the
  Mott–Bethe/WK route, which gives an independent check of data item #1.
  - Caveat: abTEM targets 30–300 keV TEM. At MeV, check its relativistic
    wavelength/interaction constant and its slice-thickness convergence before
    trusting it.
  - Caveat: frozen phonon has no energy loss. Compare at fixed energy over
    thicknesses ≪ the energy-loss scale.

### Sources

- abTEM docs: <https://abtem.readthedocs.io/en/main/reference/api/api.html>,
  core loss <https://abtem.readthedocs.io/en/main/user_guide/tutorials/core_loss.html>,
  frozen phonons <https://abtem.readthedocs.io/en/main/reference/api/_autosummary/abtem.inelastic.phonons.FrozenPhonons.html>;
  plasmon/phonon phase-scramble PR <https://github.com/abTEM/abTEM/pull/295>.
- QEP model: Forbes et al., <https://findanexpert.unimelb.edu.au/scholarlywork/324015-quantum-mechanical-model-for-phonon-excitation-in-electron-diffraction-and-imaging-using-a-born-oppenheimer-approximation>;
  frequency-resolved frozen phonon <https://arxiv.org/pdf/2104.03197>, <https://arxiv.org/pdf/2304.13510>.
- Absorptive vs frozen phonon (Hájek, Rusz et al. 2026): <https://arxiv.org/abs/2608.24679>.
- Mendis Bloch-wave + MC: <https://durham-repository.worktribe.com/output/3363920/a-physical-optics-formulation-of-bloch-waves-and-its-application-to-4d-stem-3d-ed-and-inelastic-scattering-simulations>,
  plasmon multislice <https://www.sciencedirect.com/science/article/abs/pii/S0304399119301627>,
  3D ED inelastic <https://journals.iucr.org/a/issues/2024/02/00/tw5007/index.html>.
- Quantum-trajectory MC (STEM): <https://pubs.rsc.org/en/Content/ArticleLanding/2015/CP/C5CP02300A>.
- Channeling: Wistisen & Di Piazza <https://arxiv.org/abs/1904.02997>;
  ELBE/Azadegan context <https://www.researchgate.net/profile/Wolfgang-Wagner-3>;
  MBN Explorer all-atom MD <https://arxiv.org/pdf/2305.09522>;
  twisted photons in axial channeling <https://arxiv.org/pdf/2403.01930>.
- Coherent/quantum emission: Wong et al., Light Sci. Appl. 2024
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC10808554/>; Shi et al., Nat. Commun.
  2024 <https://www.nature.com/articles/s41467-024-52050-w>; Pan & Gover
  <https://arxiv.org/abs/2401.05978>; Karnieli et al., Sci. Adv. 2021
  <https://arxiv.org/abs/2011.00623>; Karnieli et al., PRL 127, 060403 (2021);
  Shatnev 2026 <https://arxiv.org/abs/2603.28827>; Gevorgyan & Gevorgian
  <https://arxiv.org/abs/2412.10205>; Gover et al., RMP 91, 035003 (2019)
  <https://doi.org/10.1103/revmodphys.91.035003>.
