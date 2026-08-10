# Electron transport

PyRITE transports independent incident electrons through crystalline or
layered matter as piecewise-linear segments. Each segment records position,
direction, kinetic energy, path length, material, and elapsed flight time for
the radiation kernels.

## Model

At each material step the transport samples an elastic free path, advances the
electron, applies continuous energy loss, and samples an elastic deflection.
The implementation uses tabulated NIST SRD 64 Mott/Browning transport data and
a Joy--Luo stopping-power model. Compound rates are assembled from element
number densities. In a stack, the active layer determines rates and stopping;
flights are truncated at interfaces before continuing with the next medium.

Vacuum legs do not scatter, stop, or radiate, but their distance advances the
transport clock. Transport ends when the electron exits permanently, falls
below the model cutoff, or exhausts the bounded step budget.

## Initial conditions and geometry

The selected beam supplies kinetic-energy spread, entrance position,
direction, and optional bunch timing. These inputs are sampled on independent
random streams so enabling an otherwise inert distribution does not perturb
transport draws. Coordinate and phase-space conventions are documented in
[Beam phase space](beam-phase-space.md).

Finite transverse dimensions, surface tilt, multilayer interfaces, and optional
blazed-groove facets constrain crossings. The flat, single-material slab is the
compatibility limit. See [Multilayer materials](../materials/multilayer-materials.md)
and [Tilt convention](../geometry/tilt-convention.md).

## Outputs and coupling

Segments are Monte Carlo histories, not detector events. The line and
bremsstrahlung kernels consume them separately. Default spectra add segment and
electron intensities incoherently; experimental coherent emission additionally
uses segment midpoint times and bunch offsets.

## Assumptions and limits

- independent electrons; no space charge or collective beamline evolution;
- condensed-history continuous stopping between elastic scatters;
- transport data limited to catalog-supported elements and model energy range;
- no radiation reaction or energy removal from emitted photons;
- material interfaces are sharp and static;
- Monte Carlo convergence must be checked against electron count and seed.

The canonical ledger claim is `electron-transport`; phase-space injection and
multilayer behavior have separate rows. Consult the [validation
ledger](../../validation/physics-validation-ledger.md) before scientific use.
Implementation owner: `cxr_mc.montecarlo.transport.simulate_trajectories`.
