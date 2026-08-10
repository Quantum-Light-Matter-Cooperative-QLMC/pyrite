# Validation: self-absorption

## Claim

- **ID:** `self-absorption`
- **Symbol:** `montecarlo/spectrum/lines.py::mc_spectrum`
- **Source:** Beer--Lambert law
- **Intended quantity:** photon survival probability from an emission point through a piecewise-homogeneous absorber stack.

## Independent derivation

Let a photon start at `r0` and propagate along unit vector `n_hat`.  In a
passive medium with linear attenuation coefficient `mu(E, r) >= 0`, intensity
obeys

```text
dI/ds = -mu(E, r0 + s n_hat) I,
T(E) = I(s_exit)/I(0) = exp[-tau(E)],
tau(E) = integral_0^s_exit mu(E, r0 + s n_hat) ds.
```

For planar layers `i`, constant `mu_i(E)`, and z-directed thickness traversed
`Delta z_i`, `ds = |dz|/|n_z|`; therefore

```text
tau(E) = sum_i mu_i(E) Delta z_i / |n_z|.
```

Equivalently, for a finite prism or lateral ray, compute each nonnegative ray
length `ell_i` inside layer `i` up to the first exit face and use
`tau = sum_i mu_i ell_i`.  This form remains finite when `n_z = 0`.

Units: `[mu_i] = Angstrom^-1`, `[ell_i] = Angstrom`, so `tau` and `T` are
dimensionless.  Required limits: zero path or zero attenuation gives `T=1`;
increasing any nonnegative path or coefficient cannot increase `T`; one layer
reduces to `exp(-mu ell)`; contiguous layer subdivision with equal `mu` leaves
`T` unchanged.  The absolute value of `n_z` is required for identical positive
optical depth at front and back exit.

## Implementation comparison

Production computes `tau = sum_i mu_i(E) * ell_i` in
`materials.attenuation::_stack_tau`.  For the infinite planar stack it obtains
`ell_i = Delta z_i / |n_z|`; for a finite prism it intersects the forward ray
with every layer and caps the interval at the first-face exit distance.
`mc_spectrum` applies exactly `T_abs = exp(-tau)` at each segment's
orientation-dependent resonance energy.  This matches the independent result
term-for-term, including the absolute-value convention and lateral-ray branch.

Independent numeric point: two layers, `z_mid = [100,300] A`, `n_z=0.6`,
`E=1500 eV` matched the manually assembled
`mu_film(500-z)/n_z + mu_sub*400/n_z` with maximum absolute difference `0.0`.
Focused multilayer, mosaic, and Eagle tests passed together (`36 passed`).

Follow-up applied after independent verification: both `mc_spectrum` and
`_stack_tau` now contain `Validation: self-absorption`.

## Verdict

- **Claim**: `self-absorption` — `montecarlo/spectrum/lines.py::mc_spectrum` — Beer--Lambert piecewise path integral
- **Filters**: units `pass`; limits `pass`; signs/conventions `pass`
- **Re-derivation**: `matches` — no divergent term or convention
- **Verdict**: `rederived`
- **Write-up**: `docs/validation/self-absorption.md`
- **Suggested ledger change**: applied — status `rederived`, write-up linked, exact back-references added
