# Continuous weighted photon source

Validation: `photon-continuous-source`.

## Independence and scope

Fresh-context derivation on 2026-10-22, written before inspecting implementation
bodies. Inputs were the ledger claim, owner signatures/docstrings,
[ADR-0020](../../adr/0020-photon-transport-source-and-scoring.md), and the
[BremsLib angular convention](bremslib-angular-model.md). The claim concerns
integration of a prescribed straight-segment source, not physical emission
events, primary recoil, full-track accuracy, or the photon interaction queue.

## Independent derivation

Let segment $i$ have length $L_i$ in Å, fixed representative kinetic energy
$T_i$ in keV, and unit tangent $\hat{\mathbf v}_i$. Its supported photon-energy
width is $\Delta k_i=(\min(k_{\rm ceiling},1000T_i)-k_{\rm cutoff})_+$ in eV.
For arclength $s\in[0,L_i]$, the source measure is

$$
d\nu_i=j_i(k,\Omega)\,ds\,dk\,d\Omega.
$$

The intended source is constant in position within each representative segment.
For any linear response $R$, including an indicator of a source phase-space
region, its mean is $\sum_i\int j_i R\,ds\,dk\,d\Omega$.
Choose $A=\sum_i L_i\Delta k_i$ and $p_i=L_i\Delta k_i/A$ over active segments.
Draw segment $i$, uniform arclength, uniform energy in its own interval, and an
independent isotropic direction. The density with respect to the joint measure
is

$$
q(i,s,k,\Omega)=p_i\frac{1}{L_i}\frac{1}{\Delta k_i}\frac{1}{4\pi}
=\frac{1}{4\pi A}.
$$

The importance-sampling identity then requires each of $M$ trials to carry

$$
\boxed{w=\frac{j_i(k,\Omega)}{Mq(i,s,k,\Omega)}
=\frac{4\pi A}{M}j_i(k,\Omega).}
$$

Evaluation must retain joint energy/direction dependence. Sampling an angular
marginal followed by an unrelated energy marginal would generally fail for a
nonseparable DDCS. Uniform position has density per arclength, not per volume;
there is no transverse-volume Jacobian. No primary denominator appears here.

BremsLib's physical DDCS is $d^2\sigma_a/(dk\,d\Omega)$ in
cm$^2$/(eV sr), evaluated at
$\theta=\arccos(\hat{\mathbf v}_i\cdot\hat{\mathbf n})$.
Because $1\,{\rm cm}^2=10^{16}\,{\rm Å}^2$ and the composition density
$n_a$ is atoms/Å$^3$,

$$
j_i(k,\Omega)=\sum_a n_a10^{16}
\frac{d^2\sigma_a}{dk\,d\Omega}
\quad[\mathrm{photons}/(\mathrm{Å}\,\mathrm{eV}\,\mathrm{sr})].
$$

Thus $A$ has units Å eV and $w$ is a photon count. The BremsLib angular shape
already integrates to its parent SDCS; no additional physical $4\pi$ belongs
in $j_i$. The $4\pi$ in the weight comes only from the isotropic proposal.

## Cheap filters

Weights are nonnegative for a nonnegative source. Constant $j$ makes every
trial have weight $4\pi Aj/M$, so their sum is exactly $4\pi Aj$.
Zero length or zero energy width supplies zero measure. Zero-emissivity trials
may be discarded only while retaining the original $M$ in all surviving
weights. Full-sphere sampling preserves photons initially outside the aperture.
No attenuation factor or extra primary energy debit is part of this measure.

## Source-to-code comparison

After writing the derivation above, inspection of `sample_continuous_photons`
and `_segments` found exactly $L_i\Delta k_i$ proposal masses, uniform
arclength offsets about each midpoint, uniform supported energies, and
uniform cosine and azimuth over the full sphere. `weight` is exactly
`rate * (4*pi*total/n_samples)`. The denominator remains the original trial
count when zero-weight rows are omitted. Primary identifiers are retained;
there is no division by their population in the source.

`sample_bremslib_continuum` evaluates the dot product of each sampled direction
with its own segment tangent and passes its own sampled energy to
`evaluate_bremslib_rows`. Explicit segment material labels select the atomic
densities. `_CM2_TO_ANG2` is $10^{16}$, with no extra isotropic factor or
attenuation. The existing validated BremsLib DDCS convention is inherited;
the library model itself was not revalidated here. Unsupported active incident
energies fail closed. The source returns a separate weighted batch and does
not mutate or debit the charged track.

An independent constant-source probe used lengths 2 and 3 Å, respective energy
widths 2 and 3 eV, emissivity 0.25 photons/(Å eV sr), and 101 proposals.
Both the analytic integral and returned total were 40.840704496667314 photons.

## Verdict

- **Filters:** units pass; limits pass; signs/conventions pass.
- **Re-derivation:** matches; no divergent factor, joint-measure Jacobian,
  angular convention, unit conversion, or primary denominator found.
- **Verdict:** `rederived`.
- **Suggested ledger change:** set `photon-continuous-source` to `rederived`,
  replace the pending note with this fresh-context comparison and link this page.
- **Human sign-off:** pending.
