# SBETHE corrected collision stopping in transport

Validation: `sbethe-corrected-stopping`.

## Independent derivation from Salvat's source

Salvat's `sbethe.f` header (lines 5–37) defines the electronic stopping
quantity and limits the corrected Bethe formula to energies above `ECUT`.
For an electron projectile, `BETHE` (lines 2319–2397) sets the heavy-projectile
Lindhard–Sørensen and Barkas terms to zero. Its cross section is

$$
\sigma_{\mathrm{stop}}(E)=2\,\mathrm{CONS}(E)\,Z_{\mathrm{mol}}
\left[L_0(E)+\frac{f(\gamma)}{2}
-C_{\mathrm{shell}}(E)-\frac{\delta_F(E)}{2}\right].
$$

Here `CONS` is the kinematic cross-section prefactor, `ZT` is the source's
electron count $Z_{\mathrm{mol}}$ per molecule, and `CSHC` is the interpolated
DHFS shell correction. `DENSIT` (lines 2135–2210) constructs the Fano
density correction from the material oscillator-strength distribution;
`BETHE` interpolates it as `CDEN` $=\delta_F/2$. Each bracketed term is
dimensionless; the cross section has units eV cm$^2$ per molecule.

The shell-corrected `stp.dat` column (lines 440–478) is

$$
S_{\mathrm{eV/\mathring A}}(E)
=\sigma_{\mathrm{stop}}(E)n_{\mathrm{mol}}
\left(10^{-8}\,\frac{\mathrm{cm}}{\mathring A}\right),
\qquad n_{\mathrm{mol}}=\mathrm{VMOL}
\quad[\mathrm{cm}^{-3}].
$$

Thus $[\sigma n_{\mathrm{mol}}]=\mathrm{eV/cm}$ and the factor $10^{-8}$
converts distance from cm to Å. The separate no-shell column adds back
$2\,\mathrm{CONS}\,Z_{\mathrm{mol}}\,n_{\mathrm{mol}}C_{\mathrm{shell}}$
before that distance conversion. For the transport convention,

$$
S_{\mathrm{keV/\mathring A}}=10^{-3}S_{\mathrm{eV/\mathring A}},
\qquad \frac{dE}{ds}=-S_{\mathrm{keV/\mathring A}}.
$$

Given strictly positive, increasing energies $E_i$ and stopping magnitudes
$S_i$, the specified interpolation for $E_i\leq E\leq E_{i+1}$ is

$$
f=\frac{\ln E-\ln E_i}{\ln E_{i+1}-\ln E_i},\qquad
S(E)=\exp\!\left[(1-f)\ln S_i+f\ln S_{i+1}\right]
=S_i^{1-f}S_{i+1}^{f}.
$$

At $E_i$ and $E_{i+1}$ this returns the native magnitudes. At the
geometric energy midpoint it returns $\sqrt{S_iS_{i+1}}$; if the two
magnitudes agree it is constant across the interval. It stays positive and
continuous but need not have a continuous derivative at a node. No
extrapolation follows from this definition. Transport energies and cutoffs
must remain in the table domain; an endpoint clamp is justified only for
roundoff after that domain check. The signed rate is non-positive.

Below the material-dependent `ECUT` (about 1 keV for electrons), Salvat uses
the empirical extrapolation in lines 525–584. Therefore a 1 keV table node
need not be an evaluation of the corrected Bethe equation. The validation
target is the material-level output table over its declared domain.

## Implementation comparison

The parser's `parse_stopping` takes column two of `stp.dat` as the
shell-corrected eV/Å magnitude and retains the no-shell column separately.
`prepare_sbethe_stopping_table` converts both energy and stopping magnitude
from eV to keV with $10^{-3}$, checks positive finite ordered nodes, and
stores their logarithms. Both `_dEds_sbethe_scalar` and CUDA `_dEds_sbethe`
evaluate the derived linear interpolation in log space, then negate the
exponential. Their different binary-search implementations select the same
adjacent nodes. The host-facing `sbethe_stopping_keV_per_ang` rejects energies
outside the table, allowing a one-ULP outward tolerance at the endpoints.

`simulate_trajectories` checks every layer's minimum cutoff and maximum
initial electron energy against its table before launching CPU or CUDA work.
The CPU lockstep/per-electron core, CPU LUT, and exact CUDA core all select the
SBETHE rate when a table is supplied. Midpoint and cutoff evaluations also
select it. The LUT samples the same table but its interpolation introduces a
separate bounded numerical approximation. Urban sampling scales its
elemental reference mean by the ratio of the SBETHE and reference rates.

Production `_case_stopping_table_records` resolves each transport layer's
table from that layer's composition. `STOPPING_MODEL` is
`sbethe-corrected-v1`; the table manifest digest is included in dataset and
public result identity, so a changed table produces a changed key. Tests pin
the catalog tables' 1 keV and 1 GeV endpoints, positivity, shell-column
difference, node and geometric-midpoint interpolation, domain rejection,
cutoff distance, LUT rate, and CPU core paths. The CUDA parity test is gated
on hardware; inspecting the source proves formula and route selection, not
execution on a GPU in this verification.

As an independent numerical check, let $E_0=1\,\mathrm{keV}$,
$S_0=8\,\mathrm{eV/\mathring A}$, $E_1=4\,\mathrm{keV}$, and
$S_1=4\,\mathrm{eV/\mathring A}$. At $E=2\,\mathrm{keV}$ the prediction is

$$
\frac{dE}{ds}=-10^{-3}\sqrt{8\cdot4}
=-0.00565685424949\,\mathrm{keV/\mathring A}.
$$

The scalar interpolation anchor uses that same midpoint. No divergent
factor, sign, exponent, or column was found. The independent derivation and
implementation comparison support `rederived`. On an NVIDIA GeForce RTX 5080
with driver 610.47, the SBETHE/Urban first-row CPU comparison and the exact
and LUT CUDA cutoff anchors passed. The latter use a constant
$0.1\,\mathrm{keV/\mathring A}$ stopping table and independently predict a
$50\,\mathring A$ range for a $5\,\mathrm{keV}$ loss. Human sign-off remains
separate.
