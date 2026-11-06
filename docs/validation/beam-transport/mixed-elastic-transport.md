# Mixed elastic transport with a random hinge

Design and derivation of the opt-in `transport_mode="mixed"` scheme (#894).
Detailed single scattering stays the default and the reference. Claims:
`mixed-elastic-split`, `mixed-elastic-soft-moments`, `mixed-elastic-hinge`.

Source: [PENELOPE-2024, NEA/MBDAV/R(2024)1](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf),
§§4.1 and 4.3–4.4; Fernández-Varea et al. (1993) and Baró et al. (1995) for the
scheme's origin.

## Scheme

Elastic events with polar deviation below a cutoff are not simulated one by
one. They are aggregated into one artificial deflection per step, at a hinge
placed uniformly along the step. Events above the cutoff stay explicit.

| Quantity | Definition | Source |
|---|---|---|
| Hard mean free path | $\lambda_h=\max(\lambda_{\rm el},\,C_1\lambda_1)$ per element | Eq. 4.10 |
| Soft inverse transport paths | $\lambda_{s,1}^{-1}=2T_1/\lambda_{\rm el}$, $\lambda_{s,2}^{-1}=6(T_1-T_2)/\lambda_{\rm el}$ | Eq. 4.38 |
| Hinge moments over step $s$ | $\langle\mu\rangle=\tfrac12(1-e^{-s/\lambda_{s,1}})$, $\langle\mu^2\rangle=\langle\mu\rangle-\tfrac16(1-e^{-s/\lambda_{s,2}})$ | Eqs. 4.28–4.29 |
| Hinge law | $F_a=aU_{0,\mu_0}+(1-a)U_{\mu_0,1}$, $\mu_0=\dfrac{2\langle\mu\rangle-3\langle\mu^2\rangle}{1-2\langle\mu\rangle}$, $a=1-2\langle\mu\rangle+\mu_0$ | Eqs. 4.30–4.32 |
| Hinge position | $\tau=\xi s$; moments evaluated at the energy reached at the hinge | §§4.1.2, 4.3.3 |

With $\mu=(1-\cos\theta)/2$, $T_n=\int_0^{\mu_c}\mu^n p(\mu)\,d\mu$.

## Decisions

**Soft law.** PENELOPE's two-box distribution, not Goudsmit–Saunderson. It
matches both moments, samples with one uniform, and needs no extra table. Its
uniform tail reaches $\mu=1$ with small probability, which PENELOPE accepts once
a track has about ten or more hinges.

**Cutoff per element and per panel.** A layer's rate is a sum over elements, so
each element gets its own rule, $\Sigma_{h,i}=\min(\Sigma_i,\Sigma_{1,i}/C_1)$.
An element with a soft part has $\Sigma_{s,1,i}\le\Sigma_{1,i}=C_1\Sigma_{h,i}$
and a detailed element has no soft part, so the soft deflection per hard step,
$\Sigma_{s,1}/\Sigma_h$, stays at most $C_1$ in compounds too. The hard fraction depends only on the angular density,
$f_{h,i}=\min(1,2\langle\mu\rangle_i/C_1)$, and carries atomic-electron rate
scaling through $\Sigma_i$ unchanged.

The quantile cutoff $\xi_c=1-f_h$ is frozen across each energy panel, from the
lower node, whose $\langle\mu\rangle$ is the larger. The detailed sampler draws
$\mu(\xi,E)=(1-f)\mu_k(\xi)+f\mu_{k+1}(\xi)$, so for a fixed $\xi_c$

$$
T_1(E)=(1-f)Q_k+fQ_{k+1},\qquad
T_2(E)=(1-f)^2Q^{(2)}_k+2f(1-f)Q^{(\times)}_k+f^2Q^{(2)}_{k+1},
$$

with $Q$ the quantile integrals over $(0,\xi_c)$ on the panel's two rows. Those
are exact at every energy, not only at nodes, so soft plus hard reproduces the
detailed law's moments everywhere. Interpolating node-level moments instead
would lose that closure, on panels up to $\ln 1.33$ wide. The price is a
jump in $\Sigma_h$ at a node. It changes speed, not physics, because the hard
optical depth is consumed row by row.

**Energy control.** Three limits bound a step: $s\le 4\lambda_T^{(h)}$ (§4.4),
$s\le C_2E/S$, and $s\le$ one tenth of the layer thickness or an explicit
`mixed_max_step_um`. The tenth follows §4.4.1's recipe for thin regions and
keeps enough hinges per track. PENELOPE's $C_2$ raises the hard elastic rate;
here it truncates the step, with a delta row, and costs fewer rows for the same
limit on mean fractional loss. This is why the hard fraction has no stopping
dependence. It also keeps the elastic table independent of the collision model.
`max_dE_frac` substeps split rows without touching the step.

**Competing hazards.** The step length is the distance to the next hard event
of any kind at the step's start energy, then capped. Hinge and delta rows are
fictitious for hard processes, so every optical depth survives them. A hard
event that overtakes a pending hinge because rates rose along the step drops
that hinge, an effect of the order of the energy change across a step.

**Interfaces.** A row stopped at a boundary discards the pending hinge and
starts a new step (§4.4, (vi)). Eq. 4.23 shows this gives the correct mean
deflection at the interface. Row positions are never adjusted, so the core's
geometry code is unchanged.

**Row contract.** A hinge row ends in `EVENT_SOFT_HINGE`; a capped step in
`EVENT_DELTA`. Both start a new flight, so straggling keys stay unique.

## Limiting cases

- $C_1=0$: nothing is soft, no hinge or delta draws happen, and transport is bit
  for bit detailed. This is tested on the full row set.
- Energies where $\lambda_{\rm el}\ge C_1\lambda_1$ are detailed in that panel.
- Soft plus hard first moment closes on the detailed law between nodes.
- $F_a$ reproduces both Lewis moments for every tested step, including tails
  with probability near $10^{-6}$.
- Row continuity: position, energy and clock are continuous across hinges;
  direction changes only at hinges and hard events.

## Eligibility

The scheme replaces the small-angle trajectory structure, so observables that
read it cannot use it.

| Observable | Mixed | Reason |
|---|---|---|
| Terminal fractions, exit energy and angle, deposition | Eligible, pending the equivalence study | Moment-preserving by construction |
| Standard bremsstrahlung | Consistency check in #990 | Depends on row energy and direction, weakly on small deviations |
| Incoherent CXR lines | Under study in #990 | Reads each row's direction and duration |
| Coherent emission, temporal profiles | Refused | Depend on flight-level phase and small-angle structure |
| Channeling | Refused | Needs real transverse dynamics |
| Grooved geometry, cone splitting, LUT, Mott | Refused | No specialization |

## Pilot (not the equivalence study)

CPU per-electron core, ELSEPA, continuous stopping, $E_{\rm cut}=20$ keV,
$C_1=C_2=0.05$, seeds 1000 onward with 300 electrons per seed, 2026-11-05.
Intervals are one standard error. The #821-anchored study with
empirical-Bernstein intervals is still to be run.

| Case | Mode | $N$ | Rows/history | Backscatter | Transmitted $\bar E$ (keV) | Transmitted $\langle\cos\rangle$ |
|---|---|---|---|---|---|---|
| W 1 MeV, 100 µm | detailed | 6,000 | 15,334 | 0.4558 ± 0.0064 | 570.84 ± 4.08 | 0.7228 ± 0.0056 |
| | mixed | 24,000 | 1,222 | 0.4505 ± 0.0032 | 570.97 ± 1.99 | 0.7200 ± 0.0027 |
| Si 1 MeV, 100 µm | detailed | 12,000 | 529 | 0.0026 ± 0.0005 | 962.60 ± 0.06 | 0.9344 ± 0.0008 |
| | mixed | 48,000 | 24 | 0.0029 ± 0.0002 | 962.63 ± 0.03 | 0.9356 ± 0.0004 |
| W 300 keV, 10 µm | detailed | 6,000 | 1,938 | 0.4283 ± 0.0064 | 244.91 ± 0.62 | 0.7220 ± 0.0037 |
| | mixed | 24,000 | 243 | 0.4251 ± 0.0032 | 246.00 ± 0.30 | 0.7183 ± 0.0019 |

All differences are within 1.6 standard errors. Row reductions are 12.6×, 22×
and 8×.

## Thin targets

The layer step limit bounds every step by a tenth of the layer thickness. A
target thin against the hard mean free path therefore takes more rows under
mixed transport than under detailed scattering: Si at 100 keV and 1 um took
about 27 rows per history against 15 (40 histories, local, no job ID). Mixed
transport pays off only when detailed scattering would take many more than the
roughly twenty rows that the limit forces per layer.
