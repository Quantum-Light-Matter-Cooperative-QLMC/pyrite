# Coherent PXR and CBS radiation

The line-spectrum kernel combines parametric X-ray radiation (PXR) and
coherent bremsstrahlung (CBS) amplitudes for reciprocal-lattice reflections,
then integrates each transported segment over its finite flight time.

## Crystal couplings

PXR uses the X-ray susceptibility Fourier component

```{math}
\chi_{\mathbf g}=-\frac{r_e\lambda^2}{\pi V_{cell}}S_{\mathbf g},
```

where $S_{\mathbf g}$ is the Debye--Waller-weighted unit-cell structure
factor. CBS uses the screened crystal-potential component

```{math}
U_{\mathbf g}=\frac{4\pi e^2}{V_{cell}g^2}
\sum_j[Z_j-f_j(\mathbf g)]e^{i\mathbf g\cdot\mathbf r_j}e^{-W_j},
```

subject to the repository's amplitude sign convention. The PXR and CBS terms
are combined as a complex amplitude; interference is therefore retained within
one segment/reflection contribution.

Finite segment duration produces a sinc-like line shape around the kinematic
dispersion condition. Escape transmission applies at the segment midpoint.
Reflections are resolved from pinned catalog families and both reciprocal
directions.

## Incoherent and experimental coherent modes

The default `incoherent` policy sums the intensity from segments and electrons.
`coherent` instead phase-sums segment fields using emission midpoint position
and time plus electron bunch timing; `both` stores the two results side by side
from one transport. The cross-electron coherent path is experimental and
currently unverified. See [Coherent emission](coherent-emission.md) for its
phase convention and validation boundary.

When transport has split flights into numerical substeps, the physical emitter
is the flight, not the row. The `incoherent` policy then sums the rows of one
`(electron_id, flight_id)` as a complex field before squaring, so only whole
flights add incoherently; each row is evaluated at the flight's representative
energy `E_repr_keV` rather than its start energy, which makes its one-point path
integral a midpoint rule. Without the grouping, splitting a flight into $N$
substeps would give $N$ rows carrying $(t_L/N)^2$ in place of one carrying
$t_L^2$ and the line peak would fall roughly as $1/N$ — tightening a numerical
tolerance would dismantle the line. At frozen energy and clock the grouped sum
recovers the unsplit row exactly by the Dirichlet-kernel identity, so all
residual under refinement is the physical variation of $E$ and $\beta$ along the
flight. The grouped reduction is host-only and non-batched; substepped rows on a
device backend, with `components=True`, or with refractive dispersion across
`layers` raise rather than silently degrading to a row-incoherent sum.

## Assumptions and limits

- kinematic/Born treatment; no dynamical diffraction or photon multiple
  scattering;
- straight, constant-velocity motion within each transport segment;
- tabulated/derived crystal structure, form factors, Debye--Waller factors, and
  selected orientation determine the coupling;
- Beer--Lambert attenuation handles escape, without feedback on emission;
- detector response is downstream and must not be folded into source physics.

Validation: `pxr-amplitude`, `cbs-amplitude`, `finite-time-lineshape`,
`line-energy-dispersion`, `self-absorption`, and — for the representative energy
and flight grouping — `substep-radiation-invariance`. Follow those rows in the
[validation ledger](../../validation/physics-validation-ledger.md) for source
derivations and current status. Implementation owners:
`pyrite.materials.crystal.chi_g`, `U_g`, and
`pyrite.montecarlo.spectrum.lines.mc_spectrum`.
