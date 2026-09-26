# Validation: `bremslib-radiative-partition`

## Scope and source

Validation: `bremslib-radiative-partition`. This fresh-context verification started from the [BremsLib v2.0.8 description](https://web.vu.lt/ff/a.poskus/files/2025/02/BremsLib_v2.0.pdf), the ledgered quantity and assumptions, and the [BremsLib angular-model derivation](bremslib-angular-model.md) before inspecting the implementation. BremsLib supplies scaled single-differential cross sections (SDCS) in mb and scaled double-differential cross sections (DDCS) in mb/sr. It does not supply a joint photon/electron angular distribution. The question here is whether one interpolated SDCS can be divided at a positive photon threshold without changing its first moment, and whether sampled photons obey the corresponding conditional angular law and momentum convention.

## Independent derivation

For incident electron kinetic energy $T$, photon energy $k$, atomic number $Z$, and reduced energy $x=k/T$, let $\chi(T,x)=(k/Z^2)d\sigma/dk$ denote the BremsLib SDCS in mb. With $C=10^{-27}Z^2\,{\rm cm}^2/{\rm mb}$,

$$
\frac{d\sigma}{dk}=C\frac{\chi(T,k/T)}{k},\qquad
\sigma_h(T;k_c)=C\int_{k_c}^{T}\frac{\chi(T,k/T)}{k}\,dk,
$$

$$
S_s(T;k_c)=C\int_0^{k_c}\chi(T,k/T)\,dk,\qquad
S_h(T;k_c)=C\int_{k_c}^{T}\chi(T,k/T)\,dk.
$$

Here $0<k_c\le T$. The hard rate $\sigma_h$ has units cm$^2$ per atom. The first moments $S_s$ and $S_h$ have units eV cm$^2$ per atom; multiplying them by number density in cm$^{-3}$ gives eV/cm, while multiplying $\sigma_h$ by that density gives cm$^{-1}$. The identity $S_s+S_h=C\int_0^T\chi\,dk$ is exact for one fixed interpolated $\chi$, regardless of $k_c$.

On a cell $[k_0,k_1]$ with $k_0>0$ and linear $\chi(k)=a+bk$, direct integration gives

$$
\int_{k_0}^{k_1}\frac{\chi(k)}{k}\,dk
=a\ln\frac{k_1}{k_0}+b(k_1-k_0),\qquad
\int_{k_0}^{k_1}\chi(k)\,dk
=\frac{\chi(k_0)+\chi(k_1)}{2}(k_1-k_0).
$$

Inserting $k_c$ as a cell boundary makes each cell wholly soft or hard. The first cell may start at zero; only its finite first moment is integrated. At $k_c=T$, $\sigma_h=S_h=0$ and $S_s$ equals the full first moment. If $\chi(T,0)>0$, the zeroth moment diverges logarithmically as $k_c\to0^+$, so no finite zero-cutoff hard rate is claimed.

The BremsLib scaled DDCS $D(T,x,\theta)$ has units mb/sr. After normalizing its angular integral to the parent SDCS, the conditional solid-angle density is $p(\Omega\mid T,k)=D/\chi$, with polar density

$$
p(\theta\mid T,k)=\frac{2\pi\sin\theta\,D(T,k/T,\theta)}{\chi(T,k/T)},
\qquad 0\le\theta\le\pi.
$$

Azimuth is uniform on $[0,2\pi)$. Sampling the hard photon energy from the normalized hard SDCS and its angle from this conditional DDCS therefore uses the same parent evaluation.

For an unchanged electron direction $\hat{\mathbf v}$, a photon direction $\hat{\mathbf n}$, and electron rest energy $m_ec^2$, momenta are written in eV/$c$ as $p(T)c=\sqrt{T(T+2m_ec^2)}$. The residual target momentum required by vector conservation is

$$
\mathbf p_{\rm target}
=\bigl[p(T)-p(T-k)\bigr]\hat{\mathbf v}
-\frac{k}{c}\hat{\mathbf n}.
$$

The target is treated as infinitely heavy: its recoil energy is neglected. This is a momentum-accounting convention, not an electron-deflection prediction from BremsLib.

## Source-to-code comparison

`hard_radiative.py::_cell_moments` evaluates the logarithmic zeroth moment and trapezoidal first moment of each linear cell. `build_radiative_partition` includes both incident-grid tip locations and the cutoff in its breakpoint union, integrates the zero-start cell only for its first moment, and partitions all positive cells at the cutoff. Its `sample_photon_energy` inverts the same cell masses. `_jit_radiative.py::radiative_moments_scalar` and `::sample_hard_photon_energy_scalar` use the same formulas in both exact CPU schedulers, and `_jit_radiative_device.py::_rad_moment` and `::_rad_sample_photon_eV` transcribe them for the CUDA kernel. The host/scalar tests compare moments and fixed-quantile samples at grid nodes, between nodes, and at cutoff endpoints.

`hard_radiative_photon_at_energy` integrates the piecewise-linear DDCS against $2\pi\sin\theta$, draws uniform azimuth, keeps the outgoing electron direction, and assigns the vector residual above to the target. `complete_hard_radiative_events` uses a separate random stream for the post-transport direction. The test checks photon plus outgoing-electron energy, unit directions, and the full three-vector momentum sum.

The focused `test_hard_radiative.py` and `test_brem_events.py` run passed together (14 tests, 2026-09-25). These synthetic-table and internal-parity anchors do not compare a full transport history with an independent transport code; the Geant4 comparison below does. `test_hard_radiative_cuda.py` passes 4/4 on an NVIDIA GeForce RTX 5080 (CuPy 14.2.0, 2026-09-25), including first-row parity with the per-electron CPU core at `rtol=1e-12`.

## Full-track cross-check against Geant4 (issue #182)

The pinned Geant4 11.4.2 TestEm5 comparison is in
`checks/full_track_bremslib/README.md`. It covers 300 and 800 keV primaries in
10 µm W and 100 µm Si, with a 10 keV electron stop and PenBrem photons
above 990 eV.

*Terminal fractions (10,000 primaries).* Source photon yields above 10 keV
agree within 1.91 standard errors in all seven runs, for both elastic
options. Electron transmission and backscatter depend on the elastic model:
Mott differs from Geant4 by up to 14 standard errors for W 300 keV and Si
800 keV. This is an elastic-transport discrepancy outside this claim; the Si
Mott excess is tracked in #183.

*Isolated radiative loss (100,000 primaries).* Each code's primary
track-length spectrum is scored in 1 keV energy bins. Evaluating this
partition's hard moments on Geant4's spectrum reproduces Geant4's realized
photon energy and count above 1 and 10 keV within 1.94 standard errors in
every case. The largest relative difference is −6.7 % for Si, where the
1σ resolution is about 5–7 %. On PyRITE's own spectrum, realized hard
counts and energies at `kc` = 1, 5 and 10 keV match the partition within 1.48
standard errors (W within 2.4 %). Hard energy plus the re-evaluated soft
first moment is independent of `kc` within 4.1 % and 0.66 standard errors.
This isolates the radiative moments from the elastic mismatch. Direct
full-track W 300 keV photon energy is 10 % above Geant4 because PyRITE's
Mott primary path is 11 % longer.

*Photon angle and recoil.* The mean cosine of the photon angle to the
parent electron, for photons of at least 10 keV, agrees within 0.039 (at most
2.18 standard errors) with PENELOPE's independent angular shape functions.
Target recoil computed in this claim's convention from both codes' joint
`(T, k, θ)` samples agrees within 6 % (at most 1.01 standard errors).
Geant4's PenBrem uses a different convention. The electron leaves along
`p_in u − k n`, a mean 2.6–4.6° deflection, and the target takes a collinear
remainder 14–32 % smaller. Both neglect recoil energy and debit exactly `k`.
The unchanged-direction convention is therefore a documented modelling choice,
not a defect. Neither code derives the electron deflection from BremsLib.

All 76 pre-registered checks pass (tolerances: 5 % model, 1 % implementation,
0.5 % cutoff estimator, 0.03 mean cosine, each plus 3σ). This does not cover
electron–electron bremsstrahlung, photon transport or detected yield (#171),
Mott elastic transport (#183), or energies and materials outside the four
cases.

## Verdict

- **Claim:** `bremslib-radiative-partition` — `montecarlo/transport/hard_radiative.py::build_radiative_partition` and related sampling functions; `montecarlo/transport/_jit_radiative.py::radiative_moments_scalar` and `::sample_hard_photon_energy_scalar` — BremsLib v2.0.8 scaled SDCS/DDCS definitions.
- **Filters:** units pass; limits pass; signs and conventions pass.
- **Re-derivation:** matches — no divergent factor, moment, angular measure, or momentum sign found.
- **Verdict:** `rederived`.
- **Write-up:** `docs/validation/radiation-physics/bremslib-radiative-partition.md`.
- **Ledger change:** status `rederived`, linked from the ledger; human sign-off remains pending.
