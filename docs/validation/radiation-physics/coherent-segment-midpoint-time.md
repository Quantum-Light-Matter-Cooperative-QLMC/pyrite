# Validation: `coherent-segment-midpoint-time`

## Claim and source equation

`mc_spectrum` stores a material flight by midpoint position `r_mid`, length `L_ang`, constant segment speed `beta`, and segment-start transport age `t_ang`. Its coherent field uses the centered finite-time integral from Feranchuk--Spence (2000), Eqs. (8), (10), (12)--(14), already derived under `coherent-emission`:

$$
\int_{-T/2}^{T/2} e^{i 2P\tau}\,d\tau
=T\,\operatorname{sinc}(PT/\pi),
\qquad T=L/\beta.
$$

The phase multiplying that real centered integral must be evaluated at the same center event as `r_mid`. Therefore

$$
\boxed{t_{\rm mid}=t_{\rm ang}+{L_{\rm ang}\over2\beta}},
\qquad
\Phi_j=\omega(t_{\rm mid}+t_{0,j}-\hat{\mathbf n}\cdot\mathbf r_{\rm mid})
-\mathbf g\cdot\mathbf r_{\rm mid}.
$$

All time-like lengths are in Angstrom with `c=1`; `omega`, `g`, and photon wavenumber are in inverse Angstrom, so the phase is dimensionless. `t0_ang` remains the per-electron absolute bunch offset. `t_ang` remains start age for trajectory ordering and compatibility.

## Assumptions

- Velocity, energy, radiation amplitude, and stopping coefficients are constant over one stored segment, matching the existing finite-time line model.
- The segment midpoint is `r_start + 0.5 L v_hat` and the transport clock advances by `L/beta`.
- The correction changes only coherent phase. Incoherent intensity, transport schema, RNG streams, and segment ordering are unchanged.
- Distinct reflections and mosaic orientations continue to add incoherently as specified by `coherent-emission`.

## Subdivision limiting case

For one straight flight over `[0,T]` with constant amplitude, splitting at `T/2` gives

$$
\int_0^T e^{i a t}\,dt
=\int_0^{T/2}e^{i a t}\,dt+\int_{T/2}^{T}e^{i a t}\,dt.
$$

Writing each integral as a real sinc times its center phase makes this identity exact: the full segment uses center `T/2`; the halves use centers `T/4` and `3T/4`. Using each half's start age beside its midpoint position instead shifts its field by `exp[-i omega T/4]`, while the unsplit field is shifted by `exp[-i omega T/2]`. Their intensities in isolation can hide that global phase, but interference with a fixed reference emitter changes. The regression therefore checks both the complex field identity and the public coherent spectrum with a fixed reference.

Other limits: `L -> 0` makes the midpoint correction vanish; a single segment's self-term remains identical to the incoherent spectrum because a unit-modulus phase cancels after squaring.

## Implementation and evidence

`montecarlo/spectrum/lines.py::mc_spectrum` derives one shared `seg_t_mid = seg_t + 0.5 * seg_L / beta_all` before constructing `d_all`. Both the batched coherent path (`sinc_cutoff=None`) and the per-reflection compatibility path consume that `d_all`.

`tests/montecarlo/test_coherent_emission.py::test_straight_flight_is_invariant_to_two_half_segments` uses a deterministic 30 keV HOPG flight. Its independently evaluated centered complex field is invariant at `rtol=1e-11`, `atol=1e-12`; the spectrum uses the backend-scaled `BATCH_RTOL`. Before the implementation correction, both spectrum routes fail across every tested energy bin by up to about 96% relative because the split flight interferes incorrectly with the fixed reference emitter.

## Fresh-context adjudication

An independent fresh-context review rederived the correction from the cited finite-time phase. With $\mathbf K=\omega\hat{\mathbf n}+\mathbf g$, $T=L/\beta$, $t_m=t_s+T/2$, and $\mathbf r_m=\mathbf r_s+\mathbf vT/2$, centering the integration variable at the flight midpoint yields $\exp[i(\omega t_m-\mathbf K\cdot\mathbf r_m)]\, T\operatorname{sinc}(\Delta T/(2\pi))$, where $\Delta=\omega-\mathbf v\cdot\mathbf K=2P$. This is identical to $\sin(PT)/P$ and requires $t_m=t_{\rm ang}+L_{\rm ang}/(2\beta)$.

Units, sign, and the repository's $c=1$ convention agree. Two half-flight centers at $T/4$ and $3T/4$ reproduce the full integral exactly; the deterministic subdivision anchor is green for both coherent reduction routes. The $L\to0$ and single-segment self-term limits also pass.

Status: **rederived**. The independent derivation found no factor, sign, unit, or convention discrepancy. Human `signed-off` remains pending.

## Addendum 2026-08-19: subdivision invariance is now first-order

The one-flight-versus-two-halves identity checked in this record is exact only for the vacuum phase, whose linear variation along a segment is precisely what the sinc finite-time factor sums. Since the in-medium dispersion became unconditional (`xray_dispersion` removed), each segment also carries `-delta(E) omega(E) L_esc,j`, whose within-segment variation the sinc does not carry, so splitting a flight moves the coherent result at first order in `delta omega dL_esc`.

Measured on the anchor's geometry (hopg 002, 30 keV, near-grazing exit, so `L_esc` is ~100x the depth step): 7.8e-6 of peak at a 40 Ang flight, falling to 4.0e-6 / 2.9e-6 / 1.1e-6 at 20 / 10 / 5 Ang. That is a discretization artifact that shrinks with the segment length, not a modelling error, and the anchor now gates both the bound and the shrinkage. The independent centered-segment integral in the same test carries no in-medium leg and still matches to 1e-11 rel, so the midpoint pairing this record derives is still pinned exactly.

Re-validation of this row against the unconditional model is warranted.
