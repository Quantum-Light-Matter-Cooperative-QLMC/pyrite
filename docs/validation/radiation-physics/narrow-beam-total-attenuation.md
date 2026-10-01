# narrow-beam-total-attenuation

Independent re-derivation of the narrow-beam total linear attenuation coefficient as sourced from EPDL2025 (issue #274). Written from the ledger row in `docs/validation/ledger-crystallography-atomic-data.md`, the derivation docstrings, the signatures, and the cited sources. The implementation bodies were read only after the derivation below was fixed.

Signatures under review:

- `pyrite.materials.photon_cross_sections.photon_cross_sections_ang2(element, E_eV) -> dict[str, ndarray]`, per-atom, in $\mathring{\mathrm A}^2$
- `pyrite.materials.photon_cross_sections.total_photon_cross_section_ang2(element, E_eV) -> ndarray`, per-atom, in $\mathring{\mathrm A}^2$
- `pyrite.materials.attenuation._mu_total_inv_ang(comp, E_eV)`, in $\mathring{\mathrm A}^{-1}$, with `comp` a list of (element, number density in atoms/$\mathring{\mathrm A}^3$)
- `pyrite.materials.attenuation._finite_mu_or_raise(mu, energy_eV, context)`, the continuum out-of-domain policy

On the reference side every number comes from the upstream EPDL2025 tape parsed directly (an independent TAB1 reader, cross-checked against `endf_parserpy`), from the NIST XCOM fixtures in `tests/data/xcom/` converted with atomic masses and Avogadro's constant taken outside PyRITE, and from CODATA through `scipy.constants`. No PyRITE helper appears on the reference side.

## Independent derivation

### Narrow-beam exponent

A collimated monoenergetic beam crosses a homogeneous mixture of species $i$ with number densities $n_i$. In good geometry the detector accepts only the unscattered forward ray, so every interaction of any kind removes the photon. Over $\mathrm dx$ the removal probability is $\sum_i n_i\sigma_{{\rm tot},i}\,\mathrm dx$, which integrates to the Bouguer–Beer law

$$
I(x)=I_0\,e^{-\mu x},\qquad
\mu(E)=\sum_i n_i\,\sigma_{{\rm tot},i}(E).
$$

Under the free-atom (independent-atom) approximation, the per-atom total is the sum of the mutually exclusive interaction channels:

$$
\sigma_{\rm tot}
=\sigma_{\rm photo}+\sigma_{\rm coh}+\sigma_{\rm incoh}
+\sigma_{\rm pair,nuc}+\sigma_{\rm pair,el}.
$$

EPDL tabulates exactly these channels in ENDF-6 File 23: MT 522 (total photoionization, the sum over subshells MT 534+), MT 502 (coherent), MT 504 (incoherent, bound with $S(x,Z)$), MT 517 (pair production in the nuclear field), and MT 515 (triplet production in the electron field). MT 501 on the tape is their stated sum. Photonuclear absorption is not part of EPDL. It is a sub-percent giant-resonance term near 10–30 MeV and is therefore a known, documented omission, not a factor error.

Assumptions: good geometry (no build-up); free-atom additivity (no solid-state, XAFS, or Bragg coherent effects); no photonuclear channel.

### Units

ENDF File 23 gives $\sigma$ in barns per atom and $E$ in eV. Since $1\,{\rm b}=10^{-24}\,{\rm cm}^2=10^{-28}\,{\rm m}^2$ and $1\,\mathring{\mathrm A}=10^{-10}\,{\rm m}$,

$$
\boxed{1\,{\rm b}=10^{-8}\,\mathring{\mathrm A}^2}.
$$

With $n_i$ in atoms/$\mathring{\mathrm A}^3$, $\mu=\sum_i n_i\sigma_i$ comes out in $\mathring{\mathrm A}^{-1}$. A conversion from a mass-attenuation fixture uses

$$
\sigma\,[{\rm b}]=\frac{(\mu/\rho)\,[{\rm cm^2\,g^{-1}}]\;A\,[{\rm g\,mol^{-1}}]}{N_A}\times10^{24}.
$$

### Interpolation law and edges

Each MF 23 section is an ENDF TAB1 record whose interpolation code is declared per range. EPDL declares INT = 2 (linear in $\sigma$, linear in $E$). Between knots $(E_k,\sigma_k)$ and $(E_{k+1},\sigma_{k+1})$,

$$
\sigma(E)=\sigma_k+\frac{\sigma_{k+1}-\sigma_k}{E_{k+1}-E_k}\,(E-E_k).
$$

A photoionization edge appears as a repeated energy $E_k=E_{k+1}$ with $\sigma_k<\sigma_{k+1}$. The physical channel opens *at* the binding energy, so the value at $E=E_k$ should be the post-edge value $\sigma_{k+1}$, i.e. $\sigma$ is right-continuous. Any thinning of the knot list that keeps every point within relative $\varepsilon$ of the lin-lin interpolant of the retained knots preserves this law, as long as both repeated edge knots are retained.

### Out-of-domain convention

The tape covers $1\,{\rm eV}\le E\le 10^{11}\,{\rm eV}$. Outside that band no value is defined, so NaN is the faithful return. Reading NaN as $\mu=0$ would mean unit transmission through any thickness, which is unphysical above $100$ GeV and an unsupported claim below $1$ eV. A caller that integrates escape factors must therefore either reject NaN or explicitly decide a sub-band convention.

### Limiting cases

1. $n_i\to0$: $\mu\to0$ linearly, with no offset.
2. Incoherent, high energy: when $E\gg$ the K binding energy, $S(x,Z)\to Z$ over almost all momentum transfers, so $\sigma_{\rm incoh}/Z\to\sigma_{\rm KN}$. With $k=E/m_ec^2$,

   $$
   \sigma_{\rm KN}=2\pi r_e^2\left\{\frac{1+k}{k^2}\left[\frac{2(1+k)}{1+2k}-\frac{\ln(1+2k)}{k}\right]+\frac{\ln(1+2k)}{2k}-\frac{1+3k}{(1+2k)^2}\right\}.
   $$

   CODATA 2018 ($r_e=2.8179403205\times10^{-15}$ m, $m_ec^2=510998.95069$ eV) gives $\sigma_{\rm KN}(500\,{\rm keV})=0.289166\,{\rm b}=2.89166\times10^{-9}\,\mathring{\mathrm A}^2$. For low-Z, the bound correction at 500 keV is expected below 1%.
3. Pair thresholds: $\sigma_{\rm pair,nuc}=0$ for $E<2m_ec^2=1.021998$ MeV, and $\sigma_{\rm pair,el}=0$ for $E<4m_ec^2=2.043996$ MeV (free electron at rest).
4. Additivity: $\sum_{\rm channels}\sigma_c$ equals the tape's MT 501 at every node, and $\mu$ for a compound is linear in each $n_i$.

### Expected quantity

$$
\mu(E)\,[\mathring{\mathrm A}^{-1}]=\sum_i n_i\,[\mathring{\mathrm A}^{-3}]\;\times\;10^{-8}\sum_{c\in\{522,502,504,517,515\}}\sigma^{\rm EPDL}_{c,Z_i}(E)\,[{\rm b}],
$$

with each $\sigma_c$ evaluated lin-lin on the upstream knots and right-continuous at edges. Each channel below its first knot is zero, because each channel starts at a threshold. The whole expression is NaN outside $[1\,{\rm eV},10^{11}\,{\rm eV}]$.

## Comparison with the implementation

The implementation was read after the derivation above was fixed.

### Symbolic and dimensional comparison

| Element | Derivation | Implementation | Agreement |
| --- | --- | --- | --- |
| Channel set | MT 522, 502, 504, 517, 515 | `PHOTON_CHANNELS` with the same five MT numbers | exact |
| Total | $\sigma_{\rm tot}=\sum_c\sigma_c$ | `total_photon_cross_section_ang2` sums the five arrays | exact |
| Unit factor | $1\,{\rm b}=10^{-8}\,\mathring{\mathrm A}^2$ | `_BARN_TO_ANG2 = 1.0e-8`, applied once in `_table` | exact |
| Linear coefficient | $\mu=\sum_i n_i\sigma_{{\rm tot},i}$, in $\mathring{\mathrm A}^{-1}$ | `_mu_total_inv_ang`: `mu + n_i * total_photon_cross_section_ang2(el, E)` | exact |
| Interpolation | lin-lin, INT = 2 | `_interpolate`: $s_0+(x-e_0)(s_1-s_0)/(e_1-e_0)$ | exact |
| Edge convention | right-continuous at a repeated energy | `searchsorted(..., side="right") - 1` selects the interval that starts at the post-edge knot | exact |
| Below first knot | $0$ (threshold) | `out = zeros`, filled only where `energy >= knots_e[0]` | exact |
| Out of band | NaN outside $[1,10^{11}]$ eV | `in_band` mask, NaN elsewhere | exact |
| Continuum policy | reject NaN where a value is claimed | `_finite_mu_or_raise` raises on non-finite $\mu$ at $E\ge1$ eV; maps sub-eV NaN to $0$ | consistent (see below) |

The literal evaluation core is:

```text
i = np.clip(np.searchsorted(knots_e, x, side="right") - 1, 0, knots_e.size - 2)
out[inside] = s0 + (x - e0) / (e1 - e0) * (s1 - s0)
```

The `clip` to `knots_e.size - 2` would extrapolate linearly if a channel ended below $10^{11}$ eV. The upstream tape ends every MF 23 section at exactly $10^{11}$ eV, so the clip only maps $E=10^{11}$ eV onto the last interval's endpoint value. No extrapolation occurs.

### Upstream tape, independently parsed

The tape at SHA-256 `59bbd8c5…c43fd` was parsed by an independent fixed-column TAB1 reader written for this verification. For Z = 1, 6, 14, 29, 74, 82 that reader agrees bit-for-bit with `endf_parserpy` 0.17.0 on MT 501, 522, 502, 504, 517, and 515. The tape findings:

- Every MF 23 interpolation range declares INT = 2 (lin-lin). Every section ends at $10^{11}$ eV. The channels start at 1 eV (MT 502/504), at the first ionization energy (MT 522), at $1.022$ MeV (MT 517), and at $2.044$ MeV (MT 515).
- Upstream additivity: $\sum_c\sigma_c$, evaluated lin-lin on each channel's own knots, reproduces MT 501 at every MT 501 node (right limits at edges) for all Z = 1–100 to $5.7\times10^{-6}$ relative. That residual is the tape's six-significant-figure formatting.

Packaged loader against the upstream tape, all Z = 1–100:

| Check | Max relative deviation | Bound |
| --- | --- | --- |
| Total vs MT 501 at every upstream node | $4.99989\times10^{-4}$ (Ra, 326 eV) | $5\times10^{-4}$ |
| Total vs MT 501 at every interval midpoint | $4.99934\times10^{-4}$ | $5\times10^{-4}$ |
| MT 522 at its own nodes | $5.00011\times10^{-4}$ | $5\times10^{-4}$ + float32 rounding |
| MT 502, 504, 517, 515 at their own nodes | $\le 4.9997\times10^{-4}$ | $5\times10^{-4}$ |
| 1519 MT 522 edges: value at $E_{\rm edge}$ and at $E_{\rm edge}(1-10^{-12})$ | all within $6\times10^{-4}$ of the post-edge and pre-edge knot | right-continuous |
| Upstream zero-valued knots and points just below each threshold | exactly $0$ | exact |

The midpoint bound follows from the node bound. Within one upstream interval both the upstream and the thinned interpolants are linear, so the midpoint error is the mean of the two node errors, and the upstream midpoint value is the mean of the node values.

Pair thresholds in the loader: Pb at $1.0219$ MeV gives $\sigma_{\rm pair,nuc}=0$, and at $1.0221$ MeV it is nonzero. At $2.0439$ MeV, $\sigma_{\rm pair,el}=0$, and at $2.0441$ MeV it is nonzero. EPDL places the thresholds at $2\times0.511$ and $4\times0.511$ MeV, which is $2$–$4$ eV above the CODATA $2m_ec^2=1.021998$ MeV and $4m_ec^2=2.043996$ MeV. That is a property of the evaluation, not of the code, and it is immaterial.

### Generator `scripts/release_epdl_table.py`

The docstring claims match the code:

- It reads the SHA-pinned tape and refuses any INT other than 2.
- It splits each section into runs at repeated energies and around zero values, and it keeps both edge knots and every zero knot verbatim.
- Within a run, greedy galloping and bisection keep a chord only after `_chord_ok` confirms that every skipped upstream node lies within the relative tolerance. Every retained chord is therefore verified, even though the feasibility of a chord is not monotone in its length.
- Energies stay float64 and cross sections are stored as float32. The extra relative rounding, about $6\times10^{-8}$, shows up as the $5.00011\times10^{-4}$ MT 522 maximum above.
- The `.npz` is written with fixed member order and timestamps.

Rerunning the generator on the pinned tape into a scratch path reproduced 500 tables and 213,332 knots, down from 958,237 upstream. The regenerated file has SHA-256 `fcc2f00c…fcfc9`. That equals both the packaged file and `EPDL_TABLE_SHA256`, so the table is deterministic and reproducible. The stored `tolerance` member is a shape-(1,) array, because `np.ascontiguousarray` promotes the 0-d input. This is cosmetic: the test reads it with `.item()`.

### NIST XCOM

The XCOM fixtures (cm²/g) were converted to barns per atom as $\sigma=(\mu/\rho)A/N_A\times10^{24}$. The conversion used IUPAC standard atomic weights (C 12.011, N 14.007, Al 26.9815385, Si 28.085, Se 78.971, W 183.84, Pb 207.2) and $N_A$ from `scipy.constants`. Points within 2% in energy of any XCOM or EPDL (tape) edge were excluded, and the comparison was taken between 1 keV and 100 GeV.

| Element | Total, max rel. | Incoherent 10 keV–50 MeV | Pair (nuc.) > 1.5 MeV | Pair (el.) > 1.5 MeV |
| --- | --- | --- | --- | --- |
| C | 0.30% | 0.74% | 0.04% | 0.04% |
| N | 0.34% | 0.74% | 0.06% | 0.04% |
| Al | 0.32% | 0.76% | 0.04% | 0.04% |
| Si | 0.31% | 0.78% | 0.03% | 0.03% |
| Se | 0.48% | 0.78% | 0.07% | 0.06% |
| W | 2.83% (2 keV) | 1.16% | 0.07% | 0.04% |
| Pb | 2.57% (80 keV) | 1.27% | 0.07% | 0.06% |

These results agree with the ledger's checks. One exception is that W reaches 2.83% with this verifier's linear 2% edge window and IUPAC masses, against the stated "≤ 2.8%". The anchor test uses a 3% tolerance, so this is a rounding difference in the ledger text, not a failure.

### Limits

- Klein–Nishina: with CODATA $r_e$ and $m_ec^2$, $\sigma_{\rm incoh}/(Z\sigma_{\rm KN})$ at 500 keV is 0.99950 (H), 0.99920 (C), 0.99710 (Si), 0.99323 (Cu), 0.98009 (W), and 0.97767 (Pb). The ratio approaches 1 from below as Z falls, as the binding correction requires. A barn-to-$\mathring{\mathrm A}^2$ slip would put this ratio off by powers of ten.
- $n\to0$: `_mu_total_inv_ang([("Si", n)], E)` is exactly $0$ at $n=0$ and exactly linear in $n$. A two-species mixture equals the sum of the single-species terms with zero difference.
- Unit chain, end to end: Si at $\rho=2.329$ g/cm³ and 8 keV gives $\mu/\rho=64.75$ cm²/g, which is consistent with the XCOM total to within the 0.31% bound above.
- Out of band: `_mu_total_inv_ang` returns NaN at $0.999$ eV and at $1.0001\times10^{11}$ eV, and finite values at $1$ eV and $10^{11}$ eV. `_finite_mu_or_raise` raises for NaN at $E\ge1$ eV and maps sub-eV NaN to $0$. The sub-eV zero is a declared convention for unfloored grids below every modelled band, not a physics claim.

## Adjudication

The implementation matches the independent derivation symbolically, dimensionally, and numerically. No factor, unit, sign, or convention diverges:

- The channel set and sum are those of the tape's MT 501.
- The barn-to-$\mathring{\mathrm A}^2$ factor is $10^{-8}$.
- Interpolation is lin-lin and right-continuous at edges, as the tape declares.
- The out-of-band domain is NaN.
- The packaged table reproduces the upstream tape to its stated $5\times10^{-4}$ tolerance at every node and midpoint for all 100 elements, and it regenerates bit-for-bit from the pinned tape.

Outstanding items are model scope, not discrepancies, and are carried in the ledger notes:

- Good geometry is assumed and not enforced; there is no build-up factor.
- Bragg coherent removal at a reflection condition is not represented by the free-atom $\sigma_{\rm coh}$.
- Photonuclear absorption is omitted.
- $\mu_{\rm photo}\ne2k\beta$ by design, because the refractive index uses a different compilation.

## History

- 2026-09-20: an independent re-derivation of the earlier model was completed. That model took photoabsorption from Chantler/FFAST $f_2$ and coherent plus incoherent scattering from Elam (`xraydb.mu_elam`). It was superseded when issue #274 replaced the source with EPDL2025; git history holds that write-up.
- 2026-10-01: fresh-context re-derivation of the EPDL2025 model (this document).
