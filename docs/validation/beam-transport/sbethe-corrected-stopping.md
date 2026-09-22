# SBETHE corrected collision stopping in transport

Validation: `sbethe-corrected-stopping`.

## Independent derivation from Salvat's source

The vendored Salvat `sbethe.f` header (lines 1–43) identifies its target as
charged-particle electronic stopping, with DHFS shell and Fano density-effect
corrections. `BETHE` (lines 2319–2397) constructs the electron corrected
stopping cross section as

$$
\sigma_{\rm stop}(E)=2C(E)Z_{\rm mol}
\left[L_0(E)+\frac{f(\gamma)}{2}-C_{\rm shell}(E)
-\frac{\delta_F(E)}{2}\right].
$$

For electrons the Lindhard–Sørensen and Barkas terms are zero in that
routine. Here $C(E)$ carries the source's cross-section prefactor;
$Z_{\rm mol}$ is the number of electrons per molecule. `DENSIT` (lines
2135–2210) evaluates the density effect from the material oscillator-strength
distribution. `stp.dat` (lines 440–478) multiplies the cross section by
`VMOL`, the molecular number density in cm$^{-3}$, and by $10^{-8}$ cm/Å.
Its shell-corrected stopping column is thus positive eV/Å. Its separate
no-shell column restores the omitted shell term. Conversion to the transport
unit gives

$$
S_{\rm keV/Å}=10^{-3}S_{\rm eV/Å},\qquad
\frac{dE}{ds}=-S_{\rm keV/Å}.
$$

For positive ordered table nodes $(E_i,S_i)$, interpolation linear in both
logarithms has $f=(\ln E-\ln E_i)/(\ln E_{i+1}-\ln E_i)$ and

$$
S(E)=\exp\!\left[(1-f)\ln S_i+f\ln S_{i+1}\right]
=S_i^{1-f}S_{i+1}^{f}.
$$

It returns each node exactly, preserves positivity, and returns the geometric
mean at a logarithmic energy midpoint. It is continuous at nodes, with no
claim of slope continuity. Host transport must reject energy or cutoff outside
the table domain; a device endpoint clamp is valid only after that check.

The source header says the corrected Bethe expression is valid above
material-specific `ECUT` (approximately 1 keV for electrons). Below `ECUT`,
Salvat uses empirical extrapolation, so a tabulated 1 keV point is not
necessarily itself a corrected-Bethe evaluation. The claim is about a
material-level output table, not applying the formula below its validity
range.

## Implementation comparison

Pending independent comparison with the table parser, CPU and CUDA transport
paths, domain checks, identity binding, and tests.
