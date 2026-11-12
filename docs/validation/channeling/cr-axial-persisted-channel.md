# `cr-axial-persisted-channel`: the persisted axial fibre source

Fresh-context verification of the #994 admitted envelope (`si-fibre/A`,
`si-fibre/B`) at commit `f0cbdeb58` on branch `issue-994-axial-admission`.
The verifier did not write the implementation. It read the ledger row, the
[source contract](../../research/physics/axial-run-integration.md#expected-value-spectral-source),
the #993 calibration note and #994's decisions before the code bodies.

The claim is an integration and persistence claim. It does not derive a new
emission formula; the per-primary estimator $\hat s_i$ is owned by
[`cr-axial-fiber-regression`](cr-axial-fiber-regression.md), the fates by
`cr-axial-zone-handoff-physical`, $\varepsilon$ by
[`cr-axial-depth-calibration`](cr-axial-depth-calibration.md) and the zone-row
rule by `cr-axial-zone-row-native-lines`. This page re-derives what the runner
must do with those pieces: normalization, joint sampling, frames, exactly-once
persistence and refusals.

## Independent expectations

### Normalization

Let $N_e$ be the line-scoring incident count (`case["Ne"]`, not the
bremsstrahlung or transport count). For an observable $O$ per incident
electron, the unbiased estimator is the incident mean. The native line kernel
returns

$$
S_{\rm nat}(E)=\frac{1}{N_e}\sum_{r\notin\mathcal Z}
\frac{d^2N_r}{dE\,d\Omega},
$$

over native rows $r$ of the line electrons ($\mathcal Z$ = zone rows). The fibre
source has to have the same denominator and units:

$$
S_{\rm ch}(E)=\frac{1}{N_e}\sum_{i=0}^{N_e-1}\hat s_i(E),\qquad
[\hat s_i]=\frac{\text{photons}}{\text{eV sr}},
$$

where $\hat s_i$ includes the entrance weight $W$ and is signed (the regression
estimator is unbiased only before clipping). Then
$S_{\rm nat}+S_{\rm ch}$ estimates the line-band emission per incident
electron, with the zone emission counted once. Brem-only electrons
($i\ge N_e$) must not enter either sum.

### One history, two outputs

Primary $i$'s fibre history is drawn from `fiber_stream(seed, i)`. That history
gives $\hat s_i$ and the terminal event (depth $z_i$, kind, in-cell window index,
transverse momentum). The fate must be read from the same event, not drawn
again. Native continuation uses a counter-addressed stream of a different
generator family. For it to be correct, primary $i$'s native key must not be
used by any other native trajectory in the run. Since line electron $i$ never
starts native transport at the face, reusing its own primary key $k_i$ at the
launch point is admissible. Brem-only electrons keep $k_j$, $j\ge N_e$.

Exception expected from the instrument: the surface entry split. The incident
enters the retained subspace with probability $w=\lVert P(\kappa_0)\psi_0\rVert^2$.
If this split is sampled as a Bernoulli draw, while $\hat s_i$ already carries
$W=w$ as a weight, then the per-electron joint law is broken on the
non-entering branch. The *means* still agree:
$\mathbb E[\hat s_i]=w\,\mathbb E[\hat s\mid\text{entered}]$ and the fate
marginal has the entered fraction $w$. For an incoherent expected-value
spectrum, only the means enter.

### Frames

Let $R$ map crystal to slab (lab) coordinates. Let the profile axes
$(\hat{\mathbf x}_p,\hat{\mathbf y}_p,\hat{\mathbf a})$ be the rows of $A$.
The zone frame $F=RA^{\mathsf T}$ has columns equal to the slab-frame profile
axes. The fibre depth is measured along $\hat{\mathbf a}$, so the zone requires
$F\hat{\mathbf z}=\hat{\mathbf z}_{\rm slab}$. For the slab-frame beam
$\hat{\mathbf b}$,

$$
(b_x,b_y,b_z)=F^{\mathsf T}\hat{\mathbf b},\quad
\theta=\operatorname{atan2}\!\left(\sqrt{b_x^2+b_y^2},\,b_z\right),\quad
\phi=\operatorname{atan2}(b_y,b_x),
$$

and the entrance carrier is $\mathbf k_\perp=p\,(b_x,b_y)$, folded as
$\mathbf k_\perp=\kappa_0+\mathbf m_0\mathbf g$ with $\kappa_0$ in the first zone.
An outgoing state with transverse momentum $\kappa+\mathbf m\mathbf g$ at
depth $z$ and in-cell position $\mathbf x$ launches with

$$
\hat{\mathbf v}_{\rm lab}=F
\begin{pmatrix}k_x/p\\ k_y/p\\ \sqrt{1-k_\perp^2/p^2}\end{pmatrix},\qquad
\mathbf r_{\rm lab}=F
\begin{pmatrix}\lfloor\boldsymbol\rho/L\rfloor L+\mathbf x\\ z\end{pmatrix},
$$

with $\boldsymbol\rho=(F^{\mathsf T}\mathbf r_0)_{xy}$ the impact. The time is
$t=t_{\rm surf}+\Delta t(z)$. Under CSDA,
$\Delta t=\int_0^z dz'/\beta=z(E_0+E_1+2mc^2)/(p_0c+p_1c)$, and the bunch offset
$t_0$ is unchanged. Limiting check: the unscattered incident state
$\mathbf m=\mathbf m_0$, $\kappa=\kappa_0$ must map back to $\hat{\mathbf b}$
exactly. Retained electrons reach $z=T$ and leave through the back face
without native transport. Unresolved electrons end at their terminal depth.

### Persistence and identity

$S_{\rm ch}$ occupies only the `spec_channeling` slot. `spec` is the
native-row sum alone. Totals add each line component once. The resolved case
record (lookup result: key, digest, numerics, $\varepsilon$, free→free owner,
bound and records) is part of the case content, so its content key differs from
an otherwise identical channeling-off case. The `axial-fibre-v3` marker keeps it
apart from the planar marker and from the v1 recipe.

## Comparison with the implementation

| Item | Expected | Code | Result |
|---|---|---|---|
| Source normalization | $\frac{1}{N_e}\sum_{i<N_e}\hat s_i$ with $W$ | `fibre_source_and_fates` calls `fiber_on_axis_source(..., incident_ids=range(count))`, `count` = line electrons (`axial.py:94,108-118`); the regression mean is `totals/count` with signed spectra | match |
| Native normalization | same $N_e$ | `_cxr_for_segments(..., Ne=Ne_lines)` with `Ne_lines = case["Ne"]` (`case_driver.py:434`); zone rows removed by `native_line_rows` (`axial.py:166-200`) | match |
| Units | photons/(eV sr incident electron) both | `case_axial_spectrum` returns the stored spectrum after a grid-equality and finiteness check (`axial.py:334-352`) | match |
| Absorption | exit path $T-z$ along the axis | `exit_path_ang = depths[-1] - depths` (`axial.py:96-100`); for A, the beam-direction path differs by $1-\cos(1.5\,\text{mrad})\approx10^{-6}$ | match |
| One history | fate from the source's events | `sample_physical_handoff(..., events=source.events)` (`axial.py:131-140`); `_supplied_events` checks identity order | match |
| Entry split | see above | `handoff_stream` Bernoulli at $w$ (`axial_outgoing_launch.py:391-404`) | mean-consistent; $1-w=9.8\times10^{-6}$ (A), $4.4\times10^{-6}$ (B) |
| Stream keys | $k_i$ for launch $i$, $k_j$ for brem-only | `keys = stream_keys(seed, Ne)`; launches carry `incident.stream_key = keys[i]`, checked unchanged (`axial_native.py:64`); brem-only use `keys[extra]` (`axial.py:255-268`); fibre and handoff streams are Philox `SeedSequence` domains | disjoint |
| Frame | $F=RA^{\mathsf T}$, axis = slab normal | `axial_zone_frame` (`axial_case.py:144-159`), `impact = r0 @ frame[:, :2]` (`axial.py:119`) | match |
| Launch kinematics | as above | `_Mapper.launch` (`axial_outgoing_launch.py:213-237`) | match |
| Retained | exit at $T$, no transport | `_classify` requires depth $=T$ and no launch; `n_transmitted += exit_ids` (`axial_native.py:97-102`, `axial.py:310`) | match |
| Unresolved | stop, counted | no launch, zone row to terminal depth, `n_fibre_unresolved` in the transport diagnostics | match (see findings) |
| Exactly once | `spec_channeling` only | `optional_line_components` (`emission.py:349-357`); CAS payload keys (`persistence.py:93-110`); diagnostics `out["channeling"]` not persisted (`case_driver.py:140-141`) | match |

### Numeric probes (this verification)

For case A lowered from public inputs (Si(001) slab, lab tilt 0.088° at lab
azimuth 225°):

- The zone frame axis is $(6\times10^{-17},6\times10^{-17},1)$. The profile x
  axis is $(1,1,0)/\sqrt2$ in the slab frame. Resolved
  $(\theta,\phi)=(0.088°,\,-6\times10^{-15}°)$ matches the lookup
  `tilt_azimuth_deg = 0`.
- $\mathbf k_\perp=(3.09651,0)$ Å⁻¹, $g=3.27230$ Å⁻¹ and $\mathbf m_0=(1,0)$
  give carrier $-0.175789$ Å⁻¹, equal to `axial_fiber_inputs` to
  $2\times10^{-13}$.
- Mapping the unscattered state back through `on_shell_direction` and $F$
  reproduces the lab beam to $6\times10^{-17}$.
- Content keys: B `69d8adca…`, the same case without the channeling record
  `200ea326…`, A `54e1a2f4…`. All are distinct.
- Entrance weights are $w_A=0.9999902$ and $w_B=0.9999956$.

### Calibration arms

The #993 records (`depth-calibration-993-{A,B}.json`) contain the arms
`control`, `return_flux`, `unresolved` (box $B=28$), `matching_surface`,
`grid_supercell`, `source_model` and `all_on`. The control numerics (window
$M=9$, box 20, step 15 Å, harmonics 4, grid 32, `boundary="max"`,
absorption on, `tilt_azimuth_deg = 0`) are those of the lookup entries. The
unresolved term is the largest component of A's $\varepsilon$
($C_uF_u$, $F_u^{\rm ctl}=0.87\%$).

## Anchors run

All with `--slow`, under an 8 GiB cap:

- `tests/montecarlo/channeling/test_axial_fibre_runner.py`,
  `tests/campaign/test_axial_fibre_milestone3.py`,
  `tests/detectors/test_point_detector.py`: 42 passed (36 s), including the
  six named nodes and the two slow zone-row nodes.
- Supporting: `tests/montecarlo/quantum/test_channeling_axial_fiber_case.py`,
  `test_channeling_axial_envelope_lookup.py`,
  `tests/montecarlo/channeling/test_channeling_producer.py`,
  `tests/dev/test_validation_ledger.py`.

## Findings

1. **Fate counts and the source SEM are not persisted.** `n_fibre_*`,
   `fate_fractions`, `entrance_weight` and `source.sem` live only in the
   transport diagnostics. `run_case` returns them under `out["channeling"]`,
   which `store_result` and the checkpoint payload drop. The claim promises
   persistence only for the source, $\varepsilon$, the free→free owner and
   bound, and the evidence IDs, and it says only that unresolved electrons are
   "counted". The text is therefore not false, but a stored result cannot show
   its own unresolved fraction or statistical error. `axial_native.py:50`
   ("Counts to persist with the calibrated depth/free-free budgets") and the
   module docstring (`axial_native.py:11-13`, "must be reported/bounded by the
   later persistence adapter") promise more than is delivered. Suggest either
   persisting the counts and SEM with the record, or rewording those
   docstrings and the claim's "counted" to "counted in run diagnostics (not
   persisted)".
2. **$\varepsilon$ covers the unresolved term of the photon source only.** The
   native emission (`spec`, characteristic, bremsstrahlung) that an
   unresolved electron would produce after its terminal depth is omitted and
   not bounded by $\varepsilon$ or any other declared quantity. At A's control
   rate ($F_u\approx0.9\%$ of line electrons, terminating inside 0.3 µm) this
   is small. The claim does not assert otherwise, but it should state this
   explicitly in Assumptions.
3. **"No second draw" is exact only on the entered branch.** The surface entry
   split is a separate `handoff_stream` Bernoulli draw. A non-entering electron
   keeps its weighted $\hat s_i$ and launches at depth 0. The expected values of
   `spec_channeling` and of the native continuation are unaffected; only the
   per-electron correlation differs, at probability $\le10^{-5}$ for A and B.
   Suggest wording: "one fibre history per line electron gives its photon
   estimate and, if it enters (probability $w\ge0.99999$), its fate".
4. **Lateral position seam (cosmetic).** A tilted zone row ends at
   $\mathbf r_0+\hat{\mathbf b}L$, but the launch is placed at the impact cell
   plus the drawn in-cell position. For A at full depth these differ by up to
   about $T\tan\theta+L_{\rm cell}\approx6$ Å laterally. In a laterally
   unbounded slab no scored quantity depends on it.
   `AxialLaunch.lateral_bound_ang` already declares the drift.
5. **Anchor scope.** `RUNNABLE_AXIAL_CASES` is empty. Every production anchor
   monkeypatches one key runnable: B for the six named nodes, A for the zone-row
   nodes. The anchors therefore pin the code path, not a shipped
   configuration. Bit-identity to the direct estimator, round trip and the
   public `simulate` path are asserted for B only. For A, the anchors assert
   source equality with the transport record and bound round trip, but not the
   direct-estimator identity. The count test compares the record counts with
   `direct.events` kinds, which equals the fates only when no surface split
   fires (true at 4 primaries with $1-w\sim10^{-5}$, but not guaranteed by
   construction).

No item changes a factor, sign, unit or normalization of `spec_channeling`.

## Verdict

- **Claim**: `cr-axial-persisted-channel` —
  `src/pyrite/montecarlo/runner/axial.py::{transport_axial_case, fibre_source_and_fates, case_axial_spectrum}`,
  `axial_native.py`, `channeling/axial_fiber_case.py`,
  `channeling/axial_case.py` — standard-run source contract plus the
  `cr-axial-fiber-regression` incident mean.
- **Filters**: units pass; limits pass (direct-estimator bit identity;
  unscattered launch maps back to the beam; channeling-off/planar keys
  distinct); signs/conventions pass (profile azimuth, carrier fold, lab
  mapping).
- **Re-derivation**: `matches`. The normalization, joint draw (mean-exact),
  frames, exactly-once storage and refusals agree. The documentation findings
  1–3 do not change the persisted spectrum.
- **Verdict**: `rederived`.
- **Suggested ledger change**: advance to `rederived` (task owner). Add to
  Assumptions: "the surface entry split is a separate Bernoulli draw at
  $w\ge0.99999$; mean-exact"; "native emission after an unresolved
  electron's terminal depth is omitted and is not covered by ε"; "fate counts
  and source SEM are run diagnostics, not persisted". Alternatively, persist
  them and keep the wording. `anchored` is justified by the listed anchors
  for the code path, with the caveat that they run under a test-local
  runnable key. Consider adding the direct-estimator identity for A as well.
