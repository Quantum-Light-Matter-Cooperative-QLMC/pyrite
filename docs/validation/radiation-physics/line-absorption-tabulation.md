# Validation: line-absorption-tabulation

## Source behavior and derivation

PyRITE pins xraydb 4.5.8 as the Chantler/FFAST source. For every column except
`f1`, `xraydb.xraydb.XrayDB._from_chantler` evaluates

```{math}
:label: eq-line-absorption-xraydb-interpolant

y(E)=\exp\!\left[(1-t)\log y_0+t\log y_1\right],
\qquad
t=\frac{\log E-\log E_0}{\log E_1-\log E_0}.
```

For element $i$, the Beer--Lambert coefficient used by PyRITE is

```{math}
:label: eq-line-elemental-mu

\mu_i(E)=2r_e\lambda(E)n_i f_{2,i}(E)
=2r_e hc\,n_i\frac{f_{2,i}(E)}{E}.
```

Because both $\log f_{2,i}$ and $\log E$ are affine in $t$,
$\log\mu_i$ is affine on the same native interval:

```{math}
:label: eq-line-elemental-log-interpolation

\log\mu_i(E)=(1-t)\log\mu_i(E_0)+t\log\mu_i(E_1).
```

The compound coefficient must then be formed after interpolation,

```{math}
:label: eq-line-compound-mu

\boxed{\mu(E)=\sum_i
\exp\!\left[(1-t)\log\mu_i(E_0)+t\log\mu_i(E_1)\right]}.
```

Interpolating either $\sum_i\mu_i$ or
$\log(\sum_i\mu_i)$ is not equivalent to {eq}`eq-line-compound-mu`, because
different elements have different slopes between their native nodes.

The implementation evaluates the fraction without subtracting close logarithms:

```{math}
:label: eq-line-stable-log-fraction

t=\frac{\log1p[(E-E_0)/E_0]}
        {\log1p[(E_1-E_0)/E_0]}.
```

The shared table grid is the 1 eV mesh unioned with native Chantler energies
for both the crystal basis and the explicit absorber composition. Adding a node
from another element only subdivides an interval on which
{eq}`eq-line-elemental-log-interpolation` is already exact.

## Units, assumptions, and limits

- $E,E_0,E_1$ are positive energies in eV, so $t$ is dimensionless.
- $r_e$, $\lambda$, and $n_i$ carry Å, Å, and $\AA^{-3}$, respectively;
  therefore each $\mu_i$ and their sum carry $\AA^{-1}$.
- The material is homogeneous along the selected single-slab escape path and
  attenuation is passive. Layered and grooved paths continue to evaluate their
  piecewise coefficients exactly per resonance energy.
- At a native or inserted table node, the interpolated coefficient is the
  tabulated coefficient. Queries beyond the shared table retain endpoint
  clamping. As $L_{\rm esc}\to0$,
  $\exp[-L_{\rm esc}\mu(E)]\to1$. As an elemental number density or
  $f_2$ tends to zero from above, that element's contribution tends to
  zero.
- Every positive elemental contribution remains positive; their compound sum
  is positive. The correction changes only numerical evaluation of
  $\mu(E)$, not the sign in $\tau=L_{\rm esc}\mu$ or $T=\exp(-\tau)$.

## Implementation comparison

The old line table stored one compound $\mu(E)$ row and blended it linearly
in energy. At the HOPG C K edge this overestimated $\mu$ by 27.2% at
283.7351 eV;
for a 10000 Å escape path the transmission changed from 0.06593 (exact) to
0.03148 (old table), a 52.3% relative error.

`montecarlo/spectrum/lines.py` now stores one `log(mu_i)` row per composition
entry and applies {eq}`eq-line-compound-mu` in the per-reflection, batched
incoherent, and batched coherent routes. The fused float32 gather and
`coherent_stream_jit_kernel.py` implement the same operation on device without
a resonance-energy transfer to the host. Single-slab and finite-footprint
paths consume the result. `_stack_tau` and grooved escape still call the exact
per-point xraydb-backed coefficient.

The CPU regression samples geometric midpoints of every adjacent shared-grid
pair in HOPG, MoS2, and MoSe2 production windows. Maximum relative errors
against direct xraydb are `1.25e-12`, `1.94e-13`, and `1.27e-13` in float64;
the corresponding float32 maxima are `9.01e-5`, `5.26e-5`, and `2.19e-5`.
The float64 HOPG residual is FITPACK evaluation rounding at an inserted node.
The regression also checks native Si absorber nodes absent from the HOPG basis,
endpoint clamps, positivity, and zero-path transmission.

## Performance evidence

An identical whole-route CPU benchmark used NumPy float64, MoS2, 4000
deterministic segments, two reflections, and 315 energy bins. After two warm-up
runs, seven measured runs gave `27.202--27.865 ms` (median `27.597 ms`) on the
old method and `28.616--29.044 ms` (median `28.708 ms`) on the corrected method:
`+4.03%`. Tracemalloc peak memory changed from 43,108,750 to 43,165,968 bytes
(`+0.13%`). No RNG is used by this workload.

An interpolation-only MoS2 microbenchmark used 400,000 seeded (`1729`) queries,
3423 grid points, two elements, three warm-ups, and 15 repeats. Its median rose
from `1.474 ms` to `23.550 ms` (`15.98x`) because log/exp/sum now dominate the
isolated operation; the elemental table doubled from 27,384 to 54,768 bytes.
The representative whole route above bounds the observed CPU impact. Both
implementations remain fully on-device in CUDA routes, with no per-segment
host transfer. CUDA timing and numerical checks were unavailable locally
(`nvidia-smi` absent), and no remote job was authorized.

## Validation state

This document began as an implementation-context derivation. A fresh-context
independent verifier subsequently rederived the pinned-source behavior, checked
the source-to-code mapping, units, signs, and limiting cases, and found no
physics divergence; the ledger is therefore `rederived`. CUDA runtime/numerical/
performance evidence and a spectrum-level exact-vs-tabulated A/B remain open.
Only a human may mark the claim `signed-off`.
