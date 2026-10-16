# `cr-zone-handoff-accounting`: the channeling-zone handoff into native transport

Derivation and implementation-context evidence for issue #506. Fresh-context verdict `rederived`: the declared population partition and continuous launch state match the independent derivation below. Existing implementation-context evidence and validity limits are retained.

- **Code:** `montecarlo/channeling/handoff.py` (ensemble solve, keyed sampling, directions) and `montecarlo/runner/channeling.py::transport_channeling_case` (zone rows, launch, join).
- **Anchors:** `tests/montecarlo/test_channeling_handoff.py`; `checks/channeling_handoff.py`.
- **Design:** [ADR-0018](../../adr/0018-quantum-channeling-zone.md), one-way handoff (D4-A, collapse at incoherent jumps D4(i)); zone segments scored as straight mean segments (D6-A).

## Population

One `DensityMatrixSolver.propagate` call per case integrates the divergence-averaged entry state over the mean path $L=T/v_z$ (slab thickness over the mean beam's normal component), with constant continuous stopping $S(E_0)$. Its books give, at every recorded depth, the cumulative free, reprojection and overflow flux, plus the entry flux $P_{\rm entry}$ and the final subspace trace $\operatorname{tr}\rho_S(L)$. They satisfy

$$
P_{\rm entry}+P_{\rm free}(L)+P_{\rm reproj}(L)+P_{\rm overflow}(L)+\operatorname{tr}\rho_S(L)=1+\delta.
$$

where $\delta=O(\Delta z^2)$ is the Strang balance defect. The handoff renormalizes by $1+\delta$ and reports $\delta$. Each electron then gets exactly one fate:

| Fate | Probability | Zone row | Native transport |
|---|---|---|---|
| entry | $P_{\rm entry}$ | none | from its own entry state and direction |
| free | $P_{\rm free}+P_{\rm reproj}$ | entry → depth $z$, ends `DECHANNEL` | from the handoff state |
| overflow | $P_{\rm overflow}$ | entry → depth $z$, ends `DECHANNEL` | from the handoff state, band-edge angle |
| bound | $\operatorname{tr}\rho_S(L)$ | entry → exit face, ends `EXIT_BOTTOM`/`EXIT_SIDE` | none |

A free or overflow electron whose sampled depth lies beyond its own exit length becomes bound at the face. A zone row whose continuous loss reaches the electron's cutoff ends `CUTOFF`. Bound + handed-off + exited = 1 holds per electron by construction, and in frequency within binomial noise of the ensemble probabilities.

## Handoff state

Four uniforms per electron come from SplitMix64 of `(seed ⊕ salt, electron, draw)`. They are independent of batch and order, and disjoint from the transport, secondary, pair and MCWF salts. They fix:

1. the fate, by the cumulative probabilities;
2. the depth $z$, by linear inversion of that fate's own cumulative flux $F(z)$;
3. for free handoffs, $E_\perp$: a bin of the depth-integrated `free_flux_by_energy`, then uniform within it;
4. the sign $s=\pm1$, with equal weight.

The handoff angle to the planes is the Lindhard relation $\psi=\sqrt{2E_\perp/(pv)}$, with $E_\perp$ measured from the cell-mean potential $U_0=0$, i.e. the cell-averaged transverse kinetic energy. The new direction keeps the electron's own in-plane component $\hat{\mathbf t}=(\hat{\mathbf v}_0-(\hat{\mathbf v}_0\cdot\hat{\mathbf n})\hat{\mathbf n})/\|\cdots\|$:

$$
\hat{\mathbf v}_h=\cos\psi\,\hat{\mathbf t}+s\sin\psi\,\hat{\mathbf n}.
$$

Overflow is jump weight beyond the plane-wave band $\lvert h\rvert\leq H$, whose angle the solver does not resolve. It leaves at the band-edge angle $\psi_{\rm edge}=(H+1/2)g\hbar/p$, a lower bound. Continuity across the `DECHANNEL` row:

$$
\mathbf r_h=\mathbf r_0+z\hat{\mathbf v}_0,\quad E_h=E_0-S(E_0)z,\quad t_h=t_0+z/\beta((E_0+E_h)/2).
$$

which is the clock rule the native cores use for a midpoint row. The launch goes through `transport.secondaries.LaunchState` on the per-electron core. It carries the primary's own stream key and the primaries' table energy range, so an entry-fate electron runs exactly the amorphous trajectory.

## Assumptions

Maintainer decisions (2026-10-16):

- Overflow is handed off at the band-edge angle (a lower bound); a run warns when overflow exceeds 10% of electrons, telling the user to raise `channeling_basis_size`.
- The plane-angle sign is symmetric. This is exact for an aligned, translation-averaged entry; the solver's binned flux carries no momentum sign.
- Depth and $E_\perp$ are sampled separably (measured below).
- $S$ is constant at $S(E_0)$, the case's SBETHE collision stopping, in both the solve and the zone row.
- Zone caps shorter than the path are refused; there is no forced handoff of still-channeled electrons.
- $\psi=\sqrt{2E_\perp/(pv)}$ toward the plane normal, with $E_\perp$ from $U_0=0$, keeping the in-plane component.

Further assumptions:

- the zone row is the straight mean segment along the entry direction, with native multiple scattering off inside it, and it is scored by the amorphous PXR/CBS and bremsstrahlung estimators (D6-A);
- rechanneling is omitted;
- the ensemble is solved at the mean beam energy and mean plane incidence. Its divergence is the RMS spread of the sampled entry angles normal to the planes, snapped to the nearest grid momentum when the Bloch grid cannot resolve it;
- there is no radiative loss after handoff above 30 MeV (#439; runs warn).

## Separability defect

`checks/channeling_handoff.py` (Si(110), 293.15 K, 5 MeV, aligned without divergence, $H=16$, 16 Bloch cells, $\Delta z=250\,\mathrm{\AA}$, 20 µm in four 5 µm chunks, constant energy):

| Chunk | 0–5 µm | 5–10 µm | 10–15 µm | 15–20 µm |
|---|---|---|---|---|
| TV distance of the $E_\perp$ marginal from the whole-depth marginal | 0.023 | 0.011 | 0.016 | 0.016 |

Free share 0.685, overflow 0.167, trace 0.148.

## Physics flag: the margin sets the handoff angles

The declared calibration `DECHANNELING_MARGIN_EV = 300` (`cr-dechanneling-collapse`) places the sink 300 eV above the barrier, so free flux lands just above `barrier + 300 eV`. At 5 MeV Si(110):

- $\psi_c=\sqrt{2U_{\rm depth}/(pv)}$ = 2.78 mrad;
- the cutoff angle is 10.7 mrad = 3.8 $\psi_c$;
- the mean free handoff angle is 13.0 mrad = 4.7 $\psi_c$.

Physical dechanneling sets in near the barrier, at about $\psi_c$. The calibration, chosen to reproduce Kephart occupation lengths, therefore also fixes the post-handoff angular distribution. Native multiple scattering then broadens it. Any observable sensitive to the dechanneled angular distribution should be checked against the margin.

## Checks

- Units: eV/eV under the root for $\psi$; $(H+1/2)g\hbar c/(pc)$ is dimensionless; $SL$ is in keV.
- Limiting cases:
  - no flux gives only entry and bound fates;
  - a beam far from the planes (all entry weight above the cutoff) reproduces the amorphous rows bit for bit;
  - a still-channeled electron is transmitted along its entry direction.
- Continuity: the segment contract passes on joined rows, and the zone loss equals $S(E_0)L$.
- Sign and frame: $\arcsin(\hat{\mathbf v}_h\cdot\hat{\mathbf n})=s\psi$ to 1e-14 with the in-plane part parallel. The plane normal and the transport direction share the sample (slab) frame (`cr-plane-incidence`).


## Independent derivation (fresh verifier, 2026-10-16)

This section was written before inspecting the implementation bodies. Inputs
were the claim ledger row, owner signatures/docstrings, ADR-0018's one-way
handoff decision and the flux definition in `cr-dechanneling-collapse`.
The existing implementation-context evidence above remains part of the record.
This verifies the declared reduced model, including its calibrated sink and
separable sampling, rather than independently establishing physical exit angles.

Let $B(z)$ be the retained subspace trace, $A$ entry rejection, and
$F(z),R(z),O(z)$ the cumulative free, reprojection and overflow losses. Removal
is one-way; each cumulative loss is nonnegative and nondecreasing. Integrating
probability conservation gives

$$
A+B(L)+F(L)+R(L)+O(L)=1+\delta \equiv N.
$$

Here $\delta$ is the numerical balance defect, zero for the continuous
conservative equation. Drawing one category with weights

$$
(p_A,p_F,p_O,p_B)=\frac{1}{N}
\bigl(A,F(L)+R(L),O(L),B(L)\bigr)
$$

partitions unity exactly for $N>0$. Conditional on free loss, the depth CDF
is $(F(z)+R(z))/(F(L)+R(L))$; conditional on overflow it is $O(z)/O(L)$.
Linear inversion of tabulated cumulative values is the piecewise-constant
flux approximation. Zero-loss channels must never be sampled. If $W_j$ is
the depth-integrated free/reprojection mass in energy bin $j$, select it with
probability $W_j/\sum_k W_k$ and draw uniformly across that bin. This is the
stated product-of-marginals approximation; it does not reproduce a general
joint depth/energy distribution.

For an individual face path $\ell_i$, replace a sampled loss deeper than
$\ell_i$ by termination on that face. Thus the categorical population before
geometrical censoring matches the mean-path ensemble, while actual handed-off
and transmitted populations can differ from its terminal weights. Energy
cutoff is another termination, not a second handoff.

With kinetic energy $K$, $pc=\sqrt{K(K+2mc^2)}$ and
$\beta=pc/(K+mc^2)$, so $pv=(pc)\beta$. The paraxial transverse kinetic energy
relative to the declared zero-mean potential is

$$
E_\perp=\frac{p_\perp^2}{2\gamma m}
       =\frac{pv}{2}\psi^2,
\qquad \psi=\sqrt{\frac{2E_\perp}{pv}}.
$$

The small-angle approximation is inherited from the continuum model.
For unit plane normal $\hat{\mathbf n}$ define

$$
\hat{\mathbf t}_i=
\frac{\hat{\mathbf v}_i-(\hat{\mathbf v}_i\cdot\hat{\mathbf n})\hat{\mathbf n}}
{\left\|\hat{\mathbf v}_i-(\hat{\mathbf v}_i\cdot\hat{\mathbf n})\hat{\mathbf n}\right\|},
\qquad
\hat{\mathbf v}_{h,i}=\cos\psi\,\hat{\mathbf t}_i+
 s\sin\psi\,\hat{\mathbf n}.
$$

Orthogonality gives unit norm and plane angle $s\psi$ for
$0\leq\psi\leq\pi/2$. The in-plane *direction*, rather than its original
magnitude, is preserved. The equal sign weights are a declared symmetry
closure because energy bins contain no signed transverse momentum. Momentum
outside the represented band has unresolved angle; the declared overflow
closure is $\psi_{\rm edge}=(H+1/2)g\hbar c/(pc)$.

For the straight row parameter $q$ (path length in angstrom), constant
stopping $S_0$ in keV/angstrom yields

$$
\mathbf r_h=\mathbf r_0+q\hat{\mathbf v}_0,
\quad K_h=K_0-S_0q,
\quad \tau_h=\tau_0+\frac{q}{\beta((K_0+K_h)/2)}.
$$

The clock $\tau=ct$ is a length, not seconds. The midpoint clock rule is a
quadrature convention; it is not the exact integral of changing inverse speed.
Launching native transport with these identical endpoint values ensures
position, energy and clock continuity without requiring an equal direction
across the dechanneling event.

Cheap filters pass: all probabilities and angles are dimensionless;
$S_0q$ has energy units and $q/\beta$ has length units. With no losses only
entry and bound categories remain. With entry rejection one, native transport
starts at the original state. Zero stopping preserves energy, zero depth
preserves position and clock, and symmetric signs give zero mean normal
component in the declared closure. A still-bound primary ends on its face.


## Comparison and verifier report

Inspection after the derivation found the same normalized weights in
`ZoneEnsemble.probabilities`, channel-specific cumulative inversion in
`sample_handoffs`, and the derived orthogonal direction construction in
`handoff_directions`. The energy bin's residual CDF fraction supplies a uniform
position within that bin, so a fifth random draw is unnecessary.
`handoff_uniforms` uses only seed, electron index and draw index before keeping
53 bits; no batch size or iteration order enters its definition.
`transport_channeling_case` passes the derived endpoint directly to
`LaunchState`; `_zone_rows` computes the same energy and midpoint clock.
Face/cutoff censoring and remapping into terminal counts preserve one fate
per primary. These comparisons reveal no missing factor or discontinuity.

One additional convention is explicit in the code: both $pv$ and the overflow
band-edge momentum are evaluated at the mean entrance energy, rather than the
individual handoff energy. For the same sampled $E_\perp$, the angle ratio
relative to a local-energy reconstruction is

$$
\frac{\psi_{\rm implemented}}{\psi_{\rm local}}
=\sqrt{\frac{(pv)_h}{(pv)_0}}.
$$

Thus positive stopping makes the implemented angle smaller than the local
reconstruction. This is an approximation of the shared-ensemble direction
closure; position, energy and clock continuity remain exact under the stated
row rule. The declared calibration, symmetric sign, separability and unresolved
overflow angle remain validity limits, rather than independently validated
physical scattering predictions.

Numeric anchors: the synthetic case uses independently specified weights
$(0.2,0.3,0.1,0.4)$, uniform depth over $10^4$ angstrom and energy-bin shares
$(0.75,0.25)$ over $[300,400]$ and $[400,600]$ eV. Its 40,000 draws pass the
four-standard-deviation frequency, depth and energy-bin gates, and the analytic
angle reference uses $pv=5.4\times10^6$ eV. The 200-primary transported case
passes the segment continuity contract and exact population totals; independent
plane-angle projections agree at $10^{-14}$ absolute tolerance. Entry rejection
reproduces the amorphous row arrays exactly. These references are suitable for
`anchored` status for the declared accounting/launch model, not for a measured
physical angular distribution.

Verification command:

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/montecarlo/test_channeling_handoff.py tests/dev/test_docs.py --durations=0
```

Result: 17 passed in 12.12 s; longest fixture setup 4.23 s, longest test call
3.38 s. No new tests were added by the verifier.

- **Claim:** `cr-zone-handoff-accounting` — the five ledgered handoff/runner
  owners — ADR-0018 D4-A/D4(i), Lindhard transverse kinetic energy and the
  `cr-dechanneling-collapse` one-way ensemble flux.
- **Filters:** units pass; limits pass; signs/conventions pass within the
  declared small-angle, mean-ensemble closure.
- **Re-derivation:** matches — normalized population weights, conditional depth
  CDF, separable energy-bin draw, plane-angle reconstruction and continuous
  native endpoint.
- **Verdict:** `rederived`.
- **Write-up:** `docs/validation/channeling/cr-zone-handoff-accounting.md`.
- **Suggested ledger change:** owner may advance `filtered` to `anchored` using
  the existing green reference tests; clarify that the clock is a length, fate
  probabilities precede geometrical/cutoff censoring, and angle reconstruction
  uses entrance-energy momentum. Human sign-off remains separate.
