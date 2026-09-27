# `line-grid-resonance-local-spacing`

Ledger row: [`line-grid-resonance-local-spacing`](../ledger-core-coherent-physics.md#line-grid-resonance-local-spacing). Status: unverified. Validation: `line-grid-resonance-local-spacing`.

## Resolution rule

The finite-flight line has first-zero width $w_i=\pi/a_i$. The opt-in `resonance-local` policy keeps a coarse backbone and places finer windows where measured lines resonate. For a line narrower than the backbone, a halo of half-width $D_i=w_i/(\pi^2\eta)$ follows from the one-sided sinc-squared tail bound: at most $\eta$ of that line's mass lies beyond each halo edge. Within the halo, the chosen power-of-two spacing is at most $w_i$ unless the global sinc spacing floor is coarser. One-hundred-eV bins take the finest overlapping requirement; adjacent bins at the same level merge into one window.

The floor is the existing weighted global sinc rule. It can permit narrow lines to alias within its own budget. The halo bound limits omitted mass outside a window; it does not bound nonuniform trapezoid error inside the window. That error needs an identical-trajectory comparison with a uniform reference.

For one isolated line, the unbinned window spans $E_i\pm D_i$, clipped to the selected bandwidth, at spacing no larger than $w_i$ when the global floor permits it. The policy changes only the opt-in case identity; `high_energy` does not yet select it.

Production Ne=20,000 comparisons, float32/float64 agreement, detected counts, and peak memory remain outstanding. Human sign-off pending.
