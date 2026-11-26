# 0024 — Texture versus CXR reflection domain

- **Status:** Accepted
- **Date:** 2026-11-26
- **Context:** [#1483](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1483), #1480 (research and opt-in model), #1301 (basal fiber domain)

## Context

The catalog's `texture = "fiber"` (HOPG, pyrolytic h-BN) did two jobs. It
described the material: aligned `c` axes and random in-plane grain azimuths.
It also chose the CXR approximation: only reciprocal vectors parallel to the
fiber axis, `(00l)`, were scored. #1480 showed that the second job is not a
consequence of the first. A random roll averages each non-basal reflection's
intensity; it does not remove it. Basal-only captures 2–75 % of the
roll-averaged line yield in the measured scenes. #1480 added an opt-in full
domain model (`FiberCXRModel`) with its own numerics, `FiberAverage`, but the
campaign path refused it.

## Decision

1. **Texture stays a catalog fact.** `texture` (`single-crystal` or `fiber`,
   with `surface_hkl` as the fiber axis) describes the material and is not
   changed. A future `powder` texture belongs here too.
2. **The reflection domain is a numerics choice.**
   `Numerics.cxr_reflections.fiber` selects it: `None` (default) is the
   basal-only approximation; a `FiberAverage` (required reciprocal cutoff,
   roll count, batch size) is the full fiber domain. It sits beside the mosaic
   controls because it is a quadrature over crystallite orientation. It
   excludes `n_families`, `max_reflections` and `line_yield_tolerance`: the
   fiber domain owns the line reflections. A profile spells it as a
   `cxr_fiber` table of the same fields
   (`[profiles.NAME.cxr_fiber] cutoff_invang = 8.0`), which lowers to
   `Sweep.cxr_fiber` and joins the dataset identity only when set.
3. **Identity.** A lowered case carries the resolved
   `FiberCXRModel.case_record()` as `Case.cxr_fiber`: model marker
   `FIBER_CXR_MODEL`, crystal, cutoff, roll count, tensor temperature and
   thermal source. It enters the case digest and content key. The batch size
   travels as `Case.cxr_fiber_batch_size`, which is execution-only (content-key
   denylist, like the spectrum chunks) and is omitted at its default. A basal
   case gains no key, so every existing identity, golden and checkpoint is
   unchanged.
4. **One scoring seam per channel.** The runner routes a fiber case through
   the same entry points every case uses: `_cxr_for_segments` for lines (roll
   average over the basal-route kernel) and `layer_emission_tables` for the
   matched BremsLib subtraction. Streaming, audits and line-grid derivation
   then cannot bypass the domain silently.
5. **Refuse outside the qualified envelope.** HOPG only, incoherent, single
   flat slab, BremsLib, deterministic mosaic quadrature, fixed electron
   counts, explicit line grid (`EnergyBins.cxr`), bin-mean lines. Stored and
   automatic line grids, measured selection, truncation audits, streaming,
   line roulette, cone splitting, channeling, transition radiation, photon
   transport and temporal profiles refuse a fiber case, at lowering or in
   `Case` validation, never by falling back to basal-only.
6. **Diffracted bremsstrahlung and Kossel terms keep the catalog basal
   families** (`n_families = 4`), as explicit-count basal cases already do.
   Their non-basal extension is separate physics.

## Consequences

- No default, golden or basal identity changes; the option is opt-in.
- Fiber spectra equal the paired scorer
  (`fiber_case.score_fiber_case`) on the same transport to 1e-12.
- Automatic line grids, truncation audits and streamed scoring need
  roll-population-aware seeding and are follow-up work. Until then a fiber
  run needs an explicit grid that covers its non-basal lines; at 300 keV and
  above the strongest line can be odd-`l` `(101)` at half the `(002)` energy.
- Cutoff convergence is the user's responsibility: #1480 measured +22 % /
  +43 % yield going from 6 to 8 / 10 Å⁻¹ at 30 keV, so no cutoff default
  exists.
