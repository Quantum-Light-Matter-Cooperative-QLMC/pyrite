# Validation: `bremslib-radiative-event-spectrum`

## Scope and source

Validation: `bremslib-radiative-event-spectrum`. This fresh-context verification used the [BremsLib v2.0.8 description](https://web.vu.lt/ff/a.poskus/files/2025/02/BremsLib_v2.0.pdf), the [BremsLib angular-model derivation](bremslib-angular-model.md), and the [energy-grid convention](../../physics/radiation-physics/energy-grid-semantics.md) before inspecting the scorer. BremsLib's scaled SDCS and DDCS have units mb and mb/sr. The intended output is photons/(eV sr incident electron) for a planar-slab observation direction, with hard photons already sampled by coupled transport.

## Independent derivation

For one hard event of energy $k$ emitted by an electron of pre-event energy $T$, let the physical differential cross sections from the same BremsLib table be $d\sigma/dk$ and $d^2\sigma/(dk\,d\Omega)$. Their ratio is the conditional directional density

$$
p(\Omega\mid T,k)
=\frac{d^2\sigma/(dk\,d\Omega)}{d\sigma/dk},
\qquad
\int_{4\pi}p(\Omega\mid T,k)\,d\Omega=1.
$$

The shared factors $10^{-27}Z^2/k$ cancel, leaving the normalized scaled DDCS divided by its SDCS. The ratio has units sr$^{-1}$. It is evaluated at $\cos\theta=\hat{\mathbf v}\cdot\hat{\mathbf n}$, with the incoming electron direction $\hat{\mathbf v}$ and observation direction $\hat{\mathbf n}$. For a planar slab and Beer–Lambert optical depth $\mu(k)L_{\rm escape}$, an event in photon bin $i$ of width $\Delta k_i$ contributes

$$
\frac{p(\Omega\mid T,k)\,\exp[-\mu(k)L_{\rm escape}]}{N_e\,\Delta k_i}
$$

to that bin's mean density. The event itself was sampled from the hard rate, so multiplying this contribution by a second cross section, atomic density, or flight length would double count the event frequency. When $\mu=0$, the expression reduces to $p/(N_e\Delta k_i)$.

Evaluation nodes $E_i$ imply midpoint bin edges $\epsilon_i=(E_{i-1}+E_i)/2$ internally, with one-sided outer edges. A cutoff $k_c=\epsilon_j>0$ separates soft bins $i<j$ from hard bins $i\ge j$. A hard photon at $k=k_c$ belongs to bin $j$ under the usual left-closed histogram convention; a terminal electron row still contains that photon. Soft track-length scoring uses the same BremsLib cross section at nodes below $k_c$, while the hard array stores event bin means above it. Thus the two arrays have disjoint bin support. Their sum is an approximate hybrid of node-sampled soft density and bin-mean hard density, as the energy-grid convention distinguishes.

## Source-to-code comparison

`brem_events.py::mc_hard_brem_event_spectrum` obtains the pre-event energy from `E_end_keV`, the emission point from the row endpoint, and the observation angle from the electron/observer dot product. It evaluates the SDCS and DDCS at that same $(T,k)$, divides them once, multiplies by escape transmission, histograms the event, and divides by incident-electron count and that bin's width. It accepts a terminal `CUTOFF` row carrying a photon. There is no extra rate, number-density, or path-length factor. `_check_transport_partition` compares the cutoff and table identity with coupled-transport metadata when present; missing or out-of-range tables fail during hard-event scoring.

`mc_soft_brem_spectrum` evaluates BremsLib track-length density and zeros bins on and above the split. `events.py::check_segment_event_contract` checks that nonterminal hard-event energy equals the subsequent electron energy jump and that only hard-radiative or terminal-cutoff rows carry a photon. The synthetic-table test anchors the exact DDCS/SDCS ratio, absence of an extra rate factor, disjoint support, and terminal cutoff scoring. The focused `test_hard_radiative.py` and `test_brem_events.py` run passed together (14 tests, 2026-09-25).

The verifier found that `brem_events.py::_cutoff_edge` accepted an edge within relative tolerance $10^{-12}$ of $k_c$, although the stated contract requires exact equality. The implementation was then changed to require exact equality, with a regression for the next representable value above an edge. The scorer does not model layered or finite-footprint escape or downstream photon transport; no independent full-track comparison of the directional scorer is available yet.

## Full-track cross-check (issue #182)

The Geant4 comparison (`checks/full_track_bremslib/README.md`) scores photons
at creation in W and Si at 300 and 800 keV. The sampled hard-photon energy
yield agrees with Geant4 on matched electron paths within counting error.
For photons of at least 10 keV, the sampled photon polar angle about the
parent electron agrees in mean cosine within 0.039 (at most 2.18 standard
errors). That angle is drawn from the same conditional DDCS this scorer
weights, so the result supports the shared angular law. It does not test the
point-detector weighting, escape factor or detected yield. Independent
full-track validation of those observables remains open.

## Verdict

- **Claim:** `bremslib-radiative-event-spectrum` — `montecarlo/spectrum/brem_events.py::mc_soft_brem_spectrum` and `::mc_hard_brem_event_spectrum`; `montecarlo/transport/events.py::check_segment_event_contract` — BremsLib v2.0.8 SDCS/DDCS relation and node-bin midpoint edges.
- **Filters:** units pass; limits pass; signs and conventions pass for the event weight and bin support; exact cutoff-edge equality passes after the regression fix.
- **Re-derivation:** matches for the scored conditional density and cutoff-bin boundary.
- **Verdict:** `rederived`.
- **Write-up:** `docs/validation/radiation-physics/bremslib-radiative-event-spectrum.md`.
- **Ledger change:** status `rederived`, with the cutoff-edge correction recorded in Notes; human sign-off remains pending.
