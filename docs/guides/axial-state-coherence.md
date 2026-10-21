# Axial finite-grid eigenstate populations and spatial ensemble coherence

These **experimental measurement-only diagnostics** are independent of the
restricted radiation-active string-local levels in `AxialLineCatalog`. Select
`AxialDiagnosticConfig(spectral_states=16, spectral_representation="energy_bins",
record_coherence=True)` to record them; the axial marimo form also exposes
sampling options. To use **fixed-sector eigenstate/degenerate-level projectors**,
select `spectral_representation="states"` and disable electronic kicks (which
change the trajectory's carrier sector). Neither option changes the propagator
or the sequence of random draws.

The orthonormal projectors use eigenvectors of the *same discrete FFT kinetic
operator* and periodic supercell potential as the split-step solver:

`H(q) = (HBAR2_2M_EV_ANG2/gamma) * |k_grid + q|² + U_grid`.

Each MCWF trajectory is normalized in grid-area measure before observation.
State probabilities are `area * |v_n^H phi|²`, averaged per incident electron.
Nearby degenerate eigenvalues are grouped as a single subspace projector, not
arbitrarily assigned individual populations. Energy-bin mode recomputes the
local eigenstates for the **exact instantaneous carrier** following a kick;
energies are grouped into fixed eV intervals, which remain comparable across
sectors. LRU caches hold at most 64 sectors and are additionally bounded by
64 MiB host / 32 MiB device eigenvector budgets. Large grids
use matrix-free `eigsh` on at most 64 low states; full eigensystems require at
most 256 grid points. Truncation and energy-cutoff omissions are shown against
survival. A finite-supercell state with energy above the maximum grid potential
is **not** a physical escape/dechanneling event. Omitted weight may contain
both bound and above-barrier states.

With `record_coherence=True`, a reference pixel `r0` is selected from the
maximum weighted entrance density (or specify `coherence_reference_index`).
At each recorded depth the online observable accumulates

`rho(r,r0) = <phi_j(r) phi_j*(r0) exp(i q_j · (r-r0))>_incident`

plus its squared-magnitude second moment for a conservative MC sampling error
mask. The complex correlation, reference density, and local density provide
`g1 = rho / sqrt(rho(r,r) rho(r0,r0))`. Plots show `arg(g1)` and `|g1|`;
low reference/local densities and statistically unresolved correlations are
masked. A mixed MCWF ensemble has **no single wavefunction phase**: these maps
are first-order mutual-coherence observables, not averaged trajectory phases.
Purity and survival are displayed separately. The selected reference is fixed
throughout a solve; a vortex center is not chosen if its entry density vanishes.

Recorded arrays scale linearly with grid points and depth count, *not* with
full density-matrix size. The combined 128 MiB recording guard also includes
these arrays; CUDA callbacks keep their grid reductions on device, but new
carrier sectors may require an expensive CPU eigenproblem and basis upload.
Fixed-sector mode avoids that cost and supports CPU/CUDA projection parity.

The periodic supercell is physical numerical geometry, **not merely a cropped
view**. Increase its extent and the sample count together when checking
convergence; outer-boundary wrapping is not physical escape. These diagnostics
do not validate the outstanding axial handoff, photon source or escape model.
