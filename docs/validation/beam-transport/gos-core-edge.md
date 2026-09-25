# SBETHE Si core-edge GOS diagnostic

Validation: `gos-core-edge`. Status: filtered. The host-only diagnostic has
not passed independent core-loss spectra or ionization-rate validation.

## Source and equation

SBETHE's `OOS.dat` represents the optical oscillator-strength density on a
transferred-energy grid. Repeated energies encode discontinuities at shell
edges. The first positive Si edge jump in the shipped table is at 102.2154 eV;
the preceding positive-width intervals integrate to 3.769 electron strengths,
while the intervals at and above it integrate to 10.237. Their sum is 14.006,
the table's total for Si within its numerical accuracy. The
[NIST X-ray transition tables](https://physics.nist.gov/cgi-bin/XrayTrans/search.pl?lower=0&sorttype=energy&units=eV&upper=1e6)
place Si L₂/L₃ edges near 99–101 eV, supporting the interpretation of the
first SBETHE jump as the onset of semi-core excitation.

For interval strengths $f_i$ and centroids $\bar W_i$ from
[`gos-optical-quadrature`](gos-optical-quadrature.md), the diagnostic keeps
only intervals with $\bar W_i\geq W_{\mathrm{edge}}$. The existing GOS
distant and close expressions then use those strengths without the multiplier
that matches the *whole-material* corrected stopping cross section:

$$
g_{\mathrm{core}}(W;E)=\sum_{\bar W_i\geq W_{\mathrm{edge}}}
g_i(W;E),\qquad
M_{n,\mathrm{core}}(E)=\int_0^E W^n g_{\mathrm{core}}(W;E)\,dW.
$$

The close-continuum quadrature retains the full model's transfer grid, so
isolating core oscillators does not move interpolation nodes. The raw moments
and cross section remain fixed as the hard cutoff $W_c$ changes. Each core
event transfers at least the selected edge energy; for $W_c\geq E$ the hard
rate is zero. Soft and hard first moments sum to the *raw core* first moment,
not to the corrected SBETHE whole-material stopping.
The hard-event sampler suppresses a secondary state for this diagnostic:
without the binding energy and subshell, transferred energy cannot be used
as free-secondary kinetic energy.

## Scope and checks

The shipped Si table recovers 3.769 plus 10.237 electron strengths across the
edge. At 3 keV the core-only candidate has positive zeroth, first, and second
moments; its hard-event samples stay above 102.2154 eV. Four cutoff choices
keep its raw moments fixed and partition its first moment without a gap.
No core event emits a free-secondary proxy. An arbitrary non-edge cutoff is
rejected.

[Vos and Grande (2019)](https://people.physics.anu.edu.au/~vos107/my_pub/eps_extension.pdf)
show that Si L-shell excitation begins near 100 eV but that atomic GOS
approximations can misrepresent the shallow-core loss shape and onset. The
SBETHE split is therefore an input and bookkeeping diagnostic. Its L-shell
differential spectrum, K-shell contribution, absolute rate, overlap with the
valence dielectric response, and corrected-stopping closure still require
validation before a combined transport spectrum can be enabled.
