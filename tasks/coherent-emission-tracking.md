# `feature/coherent-emission-tracking`

## Problem

`mc_spectrum` sums the segment line spectrum **incoherently**: per segment it
forms `|A_PXR + A_CBS|²` and accumulates `Σ_j |A_j|²·sinc²·T_abs` (spectrum.py
`_accumulate`, step 5–7). That is correct when emitters are mutually incoherent
— distinct reflections are spectrally separated, and single-electron segments
are treated as independent. It **cannot** represent superradiant / coherent
enhancement, where the observed intensity is `|Σ_j A_j e^{iφ_j}|²` and the
cross terms survive: emission at wavelengths ≳ the bunch length (or the
intra-electron coherence length) adds in phase and scales super-linearly.

On-Hold backlog item: *Superradiant PXR/CBS need bunch-length knowledge,
coherent emission across segments* (also a prerequisite for channeling-radiation
long-term features #1/#3). This branch delivers the **optional** coherent-sum
path, gated by a boolean so the default (incoherent) result stays bit-for-bit.

## What already exists (inert substrate — do not re-add)

The phase inputs are already carried on the segment dict and are documented as
"inert; nothing reads it yet" (transport.py ~844):

- `r_mid` (M,3) — segment midpoint position [Å]  → spatial phase `k·r`
- `t_ang` (M,) — segment-start age `Σ L/β` [Å, c=1]  → emission-time phase `ω·t`
- `t0_ang` (M,) — per-electron longitudinal **bunch offset** [Å, c=1], broadcast
  onto every segment. `t_abs = t_ang + t0_ang` is the absolute emission time.
- `elec_id` (M,) — emitting electron index in `[0, Ne)` → group segments per
  electron for the intra-electron coherent sum.
- Beam (`sweep.BeamSpec`) already has `bunch_length_fs` / `long_shape` /
  `long_offsets_fs`; docstring: *"Incoherent today … the sampled offsets are
  the input the future coherent form factor consumes."* This branch is that
  consumer. All-zero `t0_ang` when no bunch sampled → coherent path collapses to
  a pure geometric (position) phase, still a legitimate limit.

So the transport side is done. This branch is almost entirely in `spectrum.py`
plus config/profile plumbing and validation.

## Physics

Far-field radiated field into direction `n̂` at frequency ω is the phased sum
over emitting segments:

```
E_tot(ω, n̂) = Σ_j A_j(ω, n̂) · exp[ i ( ω t_abs,j − k_out · r_j ) ]
dN/dE dΩ    ∝ |E_tot|²          (replaces Σ_j |A_j|²)
```

- `k_out = ω n̂` (observation wavevector); for the diffracted PXR channel the
  relevant lattice phase `g·r` is a constant per reflection and factors out of
  the cross terms within one g, so the retarded phase reduces to
  `φ_j = ω t_abs,j − ω n̂·r_j` per segment. **Derive this cleanly** — confirm
  whether the correct phase uses `k_out·r` or `(k_out+g)·r` and that the g-term
  is common-mode across segments of one reflection (it likely is; verify).
- **Two coherence scales, both from the same sum:**
  1. *Intra-electron* — segments of one electron are correlated in space/time
     (the trajectory is continuous). Cross terms give the single-particle
     coherent lineshape.
  2. *Inter-electron (superradiance)* — `t0_ang` spread over the bunch. Summing
     `e^{iω t0}` across electrons yields the **bunch form factor**
     `F(ω) = |⟨e^{iω t0}⟩|²`; for a Gaussian bunch of RMS length σ_t,
     `F = exp[−(ω σ_t)²]` (in c=1 units, `ω` in 1/Å, `σ_t = σ_z`). Coherent
     enhancement ≈ `1 + (N_e − 1)·F(ω)`. **This is the load-bearing result** —
     verify the form factor against the analytic Gaussian limit.

### Limiting cases (each becomes a test)
- **Flag off** → existing incoherent path, bit-for-bit (regression anchor).
- **`F(ω) → 0`** (bunch much longer than λ, or randomized phases) → coherent
  intensity → `Σ|A_j|²`, i.e. recovers the incoherent result up to the
  self-term. This is the correctness bridge between the two paths.
- **`F(ω) → 1`** (bunch ≪ λ, all in phase) → `|Σ A_j|²` = N²-scaling limit.
- Single segment, single electron → identical to incoherent (only self-term).

## Design decisions

1. **Where the boolean lives.** Add `coherent_emission: bool = False` to
   `profiles.SweepProfile`. It is a *policy*, not a grid reduction, so it rides
   the profile like `n_electrons`. Thread it through
   `SweepProfile.apply_settings` → `results.store.Settings` (new field, inert
   default) → `montecarlo/runner.py` → `mc_spectrum(coherent=...)`.
   - New `Settings.coherent_emission: bool = False` field.
   - New `mc_spectrum` kwarg `coherent: bool = False` (default = today,
     bit-for-bit). Same for `mc_spectrum_solid_angle` passthrough.
2. **Dataset identity / hashing.** Coherent is run-affecting, so it MUST join
   `profiles.dataset_identity` — but only when `True`, following the existing
   compatibility rule (like `catalog_profile != "standard"`, elliptical y,
   bunch fields): emit the key only on divergence from the inert default so
   every existing `parameter_sha256` (and checkpoint stem) stays bit-for-bit.
   Coherent + incoherent runs of the same material must never collide into one
   checkpoint. **Also**: coherent depends on `bunch_length_fs` etc., which
   already conditionally hash — confirm they always hash when coherent is on.
3. **Coherent implies a bunch.** Coherent with `bunch_length_fs=None` gives the
   pure-geometry (position-phase) result with all `t0_ang=0`. Decide: allow it
   (documented degenerate) or raise. Recommend **allow** but `log`/warn — the
   position phase alone is still physics, and channeling features will want it.
4. **Component split.** `components=True` returns PXR/CBS separately. Under
   coherence the split is ambiguous (cross terms `A_PXR·A_CBS*`). Decide: return
   `|Σ(A_PXR+A_CBS)|²` as total and keep the component channels as the coherent
   sums `|ΣA_PXR|²`, `|ΣA_CBS|²` (they no longer add to the total — document
   loudly), or disallow `components` with `coherent`. Recommend disallow in v1
   (raise ValueError), revisit if needed.
5. **Performance.** The incoherent path is `w @ S` (segment-weight × sinc
   matrix). The coherent path needs the **complex** amplitude on the grid:
   `Σ_j A_j e^{iφ_j} q_j(E)` where `q_j` is the (complex) finite-time factor
   `Q(P,t_L)` — note the current code squares `|Q|²=sinc²`; coherence needs the
   un-squared complex `Q`. Accumulate a complex `E_grid`-length vector, square
   at the end. Keep the same chunking / `sinc_cutoff` block structure. Expect
   ~2× cost (complex) — acceptable behind the flag.

## Implementation path / checklist

- [ ] **Derivation doc** `docs/validation/coherent-emission.md`: phase
      convention, bunch form factor, both limiting cases, N-scaling. Fresh
      derivation from Zhai SI / Feranchuk, not paraphrase.
- [ ] **`spectrum.py`**: add `coherent` kwarg; in `_accumulate`, when set, build
      the complex `Q(P,t_L)` finite-time factor (un-squared) and the per-segment
      phase `exp[i(ω t_abs − ω n̂·r)]`, accumulate a complex spectrum, `|·|²` at
      the end. Reuse `t_L`, `a_width`, the keep mask, chunk loop, `sinc_cutoff`
      block. Group by `elec_id`+`t0_ang` for the bunch form factor.
      `Validation: coherent-emission` marker on the new branch.
- [ ] **`profiles.py`**: `SweepProfile.coherent_emission` field +
      `apply_settings` wiring; hash key in `dataset_identity` (divergence-only).
- [ ] **`results/store.py`**: `Settings.coherent_emission` field.
- [ ] **`montecarlo/runner.py`**: pass `coherent=settings.coherent_emission`
      into `mc_spectrum` / `mc_spectrum_solid_angle`.
- [ ] **CLI**: expose the toggle (profile flag and/or `--coherent`). Per
      `AGENTS.md`, **invoke the `cli-ui-ux` skill** for design + tests before
      touching CLI surface.
- [ ] **Ledger row**: add `coherent-emission` to
      `docs/physics-validation-ledger.md` (Core coherent physics table); status
      `unverified` → drive toward `rederived`/`anchored`. Independent
      verification in fresh context (physics-validator), human signs off.
- [ ] **Tests** (`tests/test_coherent_emission.py`):
      - flag off ⇒ bit-for-bit vs current `mc_spectrum` (add to chunk-invariance
        style anchor).
      - decoherent limit (long bunch / scrambled `t0`) ⇒ → incoherent `Σ|A|²`.
      - Gaussian bunch form factor vs closed form `exp[−(ω σ_z)²]`.
      - N-scaling: `F→1` gives `|ΣA|²`, monotonic in `bunch_length_fs`.
      - `components` + `coherent` raises (v1).
      - identity: coherent vs incoherent produce distinct `parameter_sha256`;
        incoherent digest unchanged vs `main`.
- [ ] **`verify`** the affected flow end-to-end (a real small run with the flag),
      not just unit tests.

## Open questions / risks

- Exact phase convention (`k·r` vs `(k+g)·r`, sign) — ties into the open
  `line-energy-dispersion` ledger *discrepancy* (the `exp(±i g·r)` sign is
  already unresolved there). Resolve before anchoring; may block sign-off.
- Whether intra-electron segment coherence is physically significant at these
  energies or whether the bunch form factor dominates — the derivation should
  scope this; v1 may implement the inter-electron form factor first and treat
  intra-electron cross terms as a documented follow-on.
- GPU memory: complex `E_grid` accumulation + per-segment phase doubles the
  working set; check against the existing GPU-OOM follow-up before large sweeps.
- Solid-angle path (`mc_spectrum_solid_angle`) sums per-direction spectra —
  coherence is per-direction (phase depends on `n̂`), so the per-`n̂` `|Σ|²` is
  correct and directions still add incoherently. Confirm.

## Delegation

Single-context task; no subagent fan-out needed. Physics verification is the one
mandatory hand-off: `physics-validator` in fresh context per the
`Validation: coherent-emission` contract, human sign-off only.
