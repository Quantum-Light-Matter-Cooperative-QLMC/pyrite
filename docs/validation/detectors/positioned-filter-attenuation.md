# Positioned-filter attenuation verification packet

**Validation ID:** `positioned-filter-attenuation`

**Status:** second independent fresh-context re-derivation completed 2026-09-13 (verdict `rederived`); human ledger transition still pending. The earlier 2026-08-14 re-derivation and the original author packet are retained below as [Part IV](#part-iv--retained-author-packet-and-2026-08-14-verifier-record).

> [!important]
> **2026-09-20 — finding III.3(2) is resolved in code.** $\mu_j$ is no longer
> photoabsorption-only: `_mu_total_inv_ang` now adds the Elam coherent +
> incoherent term, so $T_p$ uses the narrow-beam **total** attenuation
> coefficient. The new claim is
> [`narrow-beam-total-attenuation`](../ledger-crystallography-atomic-data.md#narrow-beam-total-attenuation);
> the fix reaches crystal-source self-absorption by the same helper
> ([#104](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/104)).
> Parts I-IV below are the verifiers' records as written and are **not**
> retro-edited: where they say $\mu$ omits scattering, read that as the state
> of the code at the time of that pass. The quantified missing fractions they
> report are now inside $\mu$. What survives unresolved is the *geometry*
> condition -- good geometry is assumed, no build-up factor is applied -- and
> findings III.3(1) and (3)-(7).

**Code:** `src/pyrite/materials/attenuation.py::linear_attenuation_inv_mm`, `src/pyrite/instrument/geometry.py::ray_box_path_lengths`, `src/pyrite/instrument/attenuation.py::primary_transmission`.

**Source:** Bouguer–Beer exponential attenuation; Henke et al. 1993, *Atomic Data and Nuclear Data Tables* **54**, 181–342, DOI [10.1006/adnd.1993.1013](https://doi.org/10.1006/adnd.1993.1013) (elemental $f_2$, reached through the ledgered `absorption-length` claim).

## Part I — Independent re-derivation (verifier)

*Written from the ledger row, the cited sources, and the public signatures only, before any implementation body or the author packet below was read. The one exception is noted in [Leakage disclosure](#leakage-disclosure).*

### I.1 What must be produced

| symbol | meaning | unit |
| --- | --- | --- |
| $\mu_j(E)$ | linear attenuation coefficient of plate $j$'s medium | $\mathrm{mm}^{-1}$ |
| $\ell_{pj}$ | length of the source-to-pixel-$p$ ray lying inside plate $j$ | $\mathrm{mm}$ |
| $\tau_p(E)$ | optical depth $\sum_j \mu_j \ell_{pj}$ | dimensionless |
| $T_p(E)$ | primary transmission $\exp(-\tau_p)$ | dimensionless, $\in(0,1]$ |
| $I_{q(p)}(E)$ | source differential emission toward $p$ | photons $\mathrm{sr}^{-1}$ per energy bin |
| $\Delta\Omega_p$ | solid angle of pixel $p$ seen from the source | $\mathrm{sr}$ |
| $F_p(E)$ | primary photons arriving at pixel $p$ | photons per energy bin |

Signatures under review:

- `linear_attenuation_inv_mm(material: str | MediumSpec, energy_eV) -> ndarray`, returning $\mu$ in $\mathrm{mm}^{-1}$ shaped like the eV grid. A `MediumSpec` carries `composition`: `(element, number_density)` pairs in **atoms per cubic angstrom**.
- `ray_box_path_lengths(rays: PixelRays, plate: FilterPlate) -> ndarray`, returning $\ell_{pj}$ in mm on the `(ny, nx)` pixel grid. `FilterPlate` is a box: `thickness_mm` along `pose.normal`, `size_mm = (w_x, h_y)` in the plane, centred at `pose.center_mm` with orthonormal local axes $(\hat{\mathbf x},\hat{\mathbf y},\hat{\mathbf n})$. `PixelRays` carries `centers_mm`, unit `directions_lab`, `distance_mm`, and `solid_angle_sr`.
- `primary_transmission(path_length_mm[..., n_f], mu_by_filter_inv_mm[n_f, n_E]) -> ndarray[..., n_E]`.

### I.2 Bouguer–Beer law

Let a primary (unscattered, unabsorbed) photon of energy $E$ travel along a ray parameterised by arc length $s$. In a passive medium the probability per unit path of removal from the primary beam is, by definition, the linear attenuation coefficient $\mu(E,s)$. The survival probability $P$ obeys

$$
\frac{\mathrm{d}P}{\mathrm{d}s} = -\mu(E,s)\,P
\quad\Longrightarrow\quad
P(s) = \exp\!\left[-\int_0^s \mu(E,s')\,\mathrm{d}s'\right].
$$

For a scene made of $N_f$ homogeneous plates with disjoint interiors, the medium is piecewise constant along the ray and the integral collapses to a sum over the path length $\ell_{pj}$ spent inside plate $j$:

$$
\boxed{\;T_p(E)=\exp\!\left[-\sum_{j=1}^{N_f}\mu_j(E)\,\ell_{pj}\right]
=\prod_{j=1}^{N_f} e^{-\mu_j(E)\ell_{pj}}\;}
$$

This reproduces the claimed form. Three structural consequences follow for free and are the cheap filters:

- **Plate-order invariance.** The exponent is a plain sum, so any permutation of $j$ gives the same $T_p$. Physically the plates commute because attenuation is a scalar multiplicative filter, not an operator with ordering.
- **Multiplicative identity.** $N_f = 0$ gives an empty sum, $\tau_p = 0$, $T_p = 1$ exactly — not merely numerically close to $1$.
- **Bounds.** $\mu_j \ge 0$ and $\ell_{pj} \ge 0$ give $\tau_p \ge 0$, hence $T_p \in (0,1]$, with $T_p = 1$ iff the ray misses every plate.

The integral form also shows why the vacuum gaps between plates contribute nothing ($\mu = 0$ there) and why only the *in-plate* chord matters, never the source-to-pixel distance.

### I.3 Linear attenuation coefficient from Henke $f_2$

Write the complex X-ray refractive index of a medium as

$$
n(E) = 1 - \delta - i\beta .
$$

A plane wave propagating a distance $z$ has amplitude $\propto \exp(i k n z)$ with $k = 2\pi/\lambda$ the **vacuum** wavenumber, so the amplitude decays as $\exp(-k\beta z)$ and the intensity — which is what a transmission factor tracks — as $\exp(-2k\beta z)$. Hence

$$
\mu = 2k\beta = \frac{4\pi\beta}{\lambda}.
$$

The factor $2$ (amplitude versus intensity) is the first place a derivation can go wrong; it is the same factor that separates the $1/e$ *amplitude* length from the $1/e$ *intensity* (attenuation) length.

Henke et al. (1993) parameterise the forward atomic scattering factor as $f(E) = f_1(E) + i f_2(E)$, with the dispersion relation for a medium of atomic number density $n_a$

$$
\delta + i\beta = \frac{r_e \lambda^2}{2\pi}\,n_a\,\bigl(f_1 + i f_2\bigr),
\qquad
\beta = \frac{r_e\lambda^2}{2\pi}\,n_a f_2 .
$$

Substituting,

$$
\boxed{\;\mu(E) = \frac{4\pi}{\lambda}\cdot\frac{r_e\lambda^2}{2\pi}n_a f_2
= 2\,r_e\,\lambda\,n_a\,f_2\;}
$$

Equivalently $\mu = n_a \sigma_{\rm abs}$ with the optical-theorem photoabsorption cross section $\sigma_{\rm abs} = 2 r_e \lambda f_2$ — the identity Henke uses to build $f_2$ from measured cross sections in the first place. The Henke "attenuation length" is $1/\mu$, an **intensity** $1/e$ length; reading it as an amplitude length would introduce a spurious factor $2$.

#### Unit chain to $\mathrm{mm}^{-1}$

Working natively in angstroms, with $[\,\cdot\,]$ denoting the numeric value in the stated unit:

$$
\lambda[\text{Å}] = \frac{hc[\text{eV\,Å}]}{E[\mathrm{eV}]}
= \frac{12398.419843320026}{E[\mathrm{eV}]},
\qquad
r_e[\text{Å}] = 2.8179403262\times10^{-5},
$$

$$
\mu[\text{Å}^{-1}] = 2\,r_e[\text{Å}]\;\lambda[\text{Å}]\;n_a[\text{Å}^{-3}]\,f_2 ,
$$

since $[\text{Å}]\cdot[\text{Å}]\cdot[\text{Å}^{-3}] = [\text{Å}^{-1}]$ and $f_2$ is dimensionless. Because $1\,\text{Å} = 10^{-7}\,\mathrm{mm}$,

$$
\mu[\mathrm{mm}^{-1}] = 10^{7}\,\mu[\text{Å}^{-1}].
$$

So a `MediumSpec` composition quoted in atoms $\text{Å}^{-3}$ requires exactly one $10^{7}$ conversion between the natural absorption-length unit (Å) and the returned $\mathrm{mm}^{-1}$; equivalently $L_{\rm abs}[\text{Å}]$ converts as $\mu[\mathrm{mm}^{-1}] = 10^{7}/L_{\rm abs}[\text{Å}]$.

#### Compound and mixture additivity

For a medium whose element $i$ has *partial* atomic number density $n_i$ (atoms of $i$ per unit volume **of the compound**, not of the pure element), the removal rates add because distinct atoms are independent scatterers in the incoherent-sum sense that underlies Bragg's additivity rule:

$$
\mu(E) = \sum_i n_i\,\sigma_{{\rm abs},i}(E)
= 2 r_e \lambda \sum_i n_i f_{2,i}(E)
= \sum_i \frac{1}{L_{{\rm abs},i}(E)} ,
$$

where $L_{{\rm abs},i} \equiv 1/(n_i\sigma_{{\rm abs},i})$ is the absorption length of a *fictitious medium containing only element $i$ at its partial density $n_i$*. The inverse-length sum is therefore correct **only** if each elemental absorption length is evaluated at the partial density inside the compound. Using a pure-element bulk density instead would be wrong, and is the single most likely implementation error in this step.

The mass form is the same statement with $n_i = \rho w_i N_A / A_i$:

$$
\frac{\mu}{\rho} = \sum_i w_i \left(\frac{\mu}{\rho}\right)_i ,
\qquad \sum_i w_i = 1 ,
$$

i.e. mass attenuation coefficients are **mass-fraction** weighted while linear attenuation coefficients are **number-density** weighted. Mixing the two weightings is the classic Bragg-rule error. The `MediumSpec` number-density representation sidesteps it: density enters once, through $n_i$, and no separate bulk $\rho$ multiplies the sum.

Limiting case: if every $f_{2,i}\to 0$ (far above all edges, or a vacuum medium $n_i \to 0$), then $\mu \to 0$ and $T_p \to 1$.

### I.4 Slab-crossing geometry: ray versus oriented box

Put the point source at the lab origin (the target reference emission point). Pixel centre $\mathbf p$ defines the ray

$$
\mathbf r(s) = s\,\hat{\mathbf d},
\qquad \hat{\mathbf d} = \frac{\mathbf p}{\lVert\mathbf p\rVert},
\qquad s \in [0, D_p], \quad D_p = \lVert\mathbf p\rVert .
$$

The plate is an axis-aligned box in its own frame. With centre $\mathbf c$, orthonormal basis $\hat{\mathbf e}_1=\hat{\mathbf x}$, $\hat{\mathbf e}_2=\hat{\mathbf y}$, $\hat{\mathbf e}_3=\hat{\mathbf n}$ and half-extents $h_1 = w_x/2$, $h_2 = h_y/2$, $h_3 = t/2$, define local coordinates

$$
u_a(s) = \bigl(\mathbf r(s) - \mathbf c\bigr)\cdot\hat{\mathbf e}_a
= o_a + s\,d_a ,
\qquad
o_a = -\,\mathbf c\cdot\hat{\mathbf e}_a,
\quad d_a = \hat{\mathbf d}\cdot\hat{\mathbf e}_a .
$$

Because the basis is orthonormal, $s$ remains true arc length in local coordinates — no Jacobian correction is needed, which is why the returned interval length is directly a path length in mm.

The box is the intersection of three slabs $\lvert u_a\rvert \le h_a$. Each slab contributes an interval in $s$:

$$
s_a^{\pm} =
\begin{cases}
\dfrac{\pm h_a - o_a}{d_a} \ \text{sorted so } s_a^-\le s_a^+, & \lvert d_a\rvert > \epsilon,\\[8pt]
(-\infty, +\infty) \ \text{if } \lvert o_a\rvert \le h_a, \ \text{else } \varnothing,
& \lvert d_a\rvert \le \epsilon .
\end{cases}
\quad
$$

The degenerate branch is essential: for a ray exactly parallel to a slab, $d_a=0$ makes the quotient ill-defined, and the correct answer is a pure inside/outside test on the constant $o_a$. Intersecting the three slab intervals **and** the finite physical segment $[0, D_p]$,

$$
s_{\rm in} = \max\Bigl(0,\ \max_a s_a^-\Bigr),
\qquad
s_{\rm out} = \min\Bigl(D_p,\ \min_a s_a^+\Bigr),
$$

$$
\boxed{\;\ell_{pj} = \max\bigl(0,\ s_{\rm out} - s_{\rm in}\bigr)\;}
$$

with $\ell_{pj} \equiv 0$ whenever any parallel-axis test failed.

Limiting and edge cases this must reproduce:

| case | expected $\ell$ | mechanism |
| --- | --- | --- |
| normal incidence, $\hat{\mathbf d}\parallel\hat{\mathbf n}$, laterally inside | $t$ | thickness slab is binding |
| oblique incidence at angle $\theta$ to $\hat{\mathbf n}$, full crossing | $t/\cos\theta$ | $\lvert d_3\rvert=\cos\theta$, so $s_3^+-s_3^-=t/\cos\theta$ |
| ray outside the lateral footprint | $0$ | an $x$/$y$ slab interval is disjoint from the $z$ one, $s_{\rm out}<s_{\rm in}$ |
| ray enters the front face, escapes a side | $<t/\cos\theta$ | lateral $s_a^+$ becomes the binding $\min$ |
| ray exactly parallel to the plate face, laterally inside | clipped chord | parallel branch keeps the axis unconstrained |
| ray parallel and laterally outside | $0$ | parallel branch flags a miss |
| pixel inside the plate | $s_{\rm out}=D_p$ | segment clip |
| plate entirely beyond the pixel | $0$ | $s_{\rm in}>D_p$, caught by $\max(0,\cdot)$ |
| plate entirely behind the source | $0$ | $s_{\rm out}<0$, caught by $\max(0,\cdot)$ |
| grazing exactly on an edge/corner | $0$ or a degenerate chord | measure-zero; must be tolerance-controlled, never NaN |

The $t/\cos\theta$ obliquity is *derived*, not applied as a separate correction: it is already inside the slab interval width. A code path that multiplies the slab result by an extra $1/\cos\theta$ would double-count.

Additivity across plates requires disjoint plate interiors. If two plates overlapped, the overlap chord would be counted in both $\ell_{pj}$ and the optical depth would double-count that volume; scene validation, not this function, must forbid overlap.

### I.5 Solid angle and the flux definition

$I_{q}(E)$ is the source's differential emission **per steradian** in direction $q$, per energy bin. The primary photon count reaching pixel $p$ is the integral of that differential emission over the cone subtended by the pixel, weighted by transmission:

$$
F_p(E) = \int_{\Omega_p} I_{\hat{\mathbf d}}(E)\,T(\hat{\mathbf d},E)\,\mathrm{d}\Omega
\;\simeq\; I_{q(p)}(E)\,\Delta\Omega_p\,T_p(E),
$$

the approximation being the centre-ray (midpoint) rule: $I$ and $T$ are sampled at the pixel centre and treated as constant across the pixel's cone. Units check: $[\mathrm{sr}^{-1}]\cdot[\mathrm{sr}]\cdot[1] = $ photons per energy bin, so the $\Delta\Omega_p$ factor is exactly what makes $F_p$ a count rather than a brightness. For a planar pixel of area $A_p$ at distance $D_p$ with face normal $\hat{\mathbf n}_{\rm det}$,

$$
\Delta\Omega_p = \frac{A_p\,\lvert\hat{\mathbf d}\cdot\hat{\mathbf n}_{\rm det}\rvert}{D_p^{2}}
= \frac{A_p\cos\theta_p}{D_p^{2}}
\qquad (\text{small-pixel limit, } \sqrt{A_p}\ll D_p),
$$

the $\cos\theta_p$ obliquity being the projection of the pixel onto the sphere. Omitting it over-counts off-axis pixels on a tilted or large detector. The exact rectangular-pyramid solid angle replaces this with the Gauss/van Oosterom form; the point-sample version errs at $O(A_p/D_p^2)$.

**Single ownership.** $\Delta\Omega_p$ must appear exactly once in the chain. If a detector response object also carries a solid-angle factor (as an $\Omega \times \mathrm{QE}$ operator would), applying both squares the solid angle and the flux acquires an unphysical $D_p^{-4}$ scaling. This is a consistency requirement on the assembly, not on the three anchored functions individually, and it is the one part of the ledger claim that the three code anchors do not by themselves establish.

### I.6 Physical scope of the model

*Superseded in part as of 2026-09-20; see the note at the top of this file. The first bullet below describes the code as it stood at this pass.*

$\mu$ built from Henke $f_2$ is the **photoabsorption** coefficient. The correct narrow-beam attenuation coefficient for a primary-beam transmission factor is the *total* one, $\mu_{\rm tot} = \mu_{\rm photo} + \mu_{\rm coh} + \mu_{\rm incoh}$, because coherently and incoherently scattered photons are also removed from the primary. The model therefore:

- **under**-estimates $\tau$ (over-estimates $T$) by the scattering fraction — negligible where photoabsorption dominates, growing with energy and falling with $Z$ (quantified in [I.7](#i7-numeric-cross-check));
- carries **no build-up factor**: photons scattered *into* the pixel are not added back, which pushes the other way. The result is a genuine narrow-beam ("good geometry") transmission, appropriate for a small pixel far from a thin plate, and increasingly wrong for a thick plate subtending a large solid angle at the pixel;
- contains **no fluorescence** re-emission from the filter, which for a filter used just above its own K edge can be a significant soft component;
- contains **no coherent/diffractive** redirection, so a crystalline filter near a Bragg condition is out of scope;
- treats each plate as **homogeneous**, ignoring density gradients, roughness, and pinholes.

Additionally the centre-ray sampling means neither the finite pixel area nor the finite source spot is integrated: a pixel straddling a plate edge gets a hard 0/1 transition rather than the true penumbral ramp.

### I.7 Numeric cross-check

Computed from the independent expression $\mu = 2 r_e \lambda n_a f_2$ with $r_e = 2.8179403262\times10^{-5}\,\text{Å}$, $hc = 12398.419843320026\,\mathrm{eV\,Å}$, and Chantler $f_2$ obtained directly from `xraydb.f2_chantler`, versus the **independently tabulated** Elam mass attenuation coefficients `xraydb.mu_elam` (a different compilation, not derived from $f_2$ in the same code path). No PyRITE helper is used on either side.

Aluminium, $\rho = 2.6989\ \mathrm{g\,cm^{-3}}$, $n_a = 0.060238\ \text{Å}^{-3}$, values in $\mathrm{cm^2\,g^{-1}}$:

| $E$ (eV) | $f_2$ | $\mu/\rho$ from $2r_e\lambda n f_2$ | Elam photo | Elam total | ratio to photo | ratio to total |
| --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.70499 | 1099.50 | 1182.94 | 1185.21 | 0.930 | 0.928 |
| 3000 | 1.49736 | 778.43 | 786.54 | 788.11 | 0.990 | 0.988 |
| 5000 | 0.59327 | 185.05 | 192.21 | 193.40 | 0.963 | 0.957 |
| 8000 | 0.24780 | 48.31 | 49.51 | 50.32 | 0.976 | 0.960 |
| 12000 | 0.10917 | 14.19 | 14.79 | 15.34 | 0.959 | 0.925 |
| 20000 | 0.03824 | 2.98 | 3.10 | 3.44 | 0.962 | 0.866 |

Agreement with the *photoabsorption* column is $1$–$7\%$ across a 20:1 energy span — the level of Chantler-versus-Elam tabulation disagreement, not a formula error. Any missing factor of $2$, $4\pi$, $2\pi$, or $10^{7}$ would show as a factor $\gtrsim 2$ offset, flat in energy; none is present. For orientation, the NIST XCOM/FFAST aluminium mass attenuation coefficient at $10\ \mathrm{keV}$ is $\approx 26\ \mathrm{cm^2\,g^{-1}}$, bracketed by the 8 and 12 keV rows above.

Compound additivity, Kapton $\mathrm{C_{22}H_{10}N_2O_5}$, $\rho = 1.42\ \mathrm{g\,cm^{-3}}$, $M = 382.33\ \mathrm{g\,mol^{-1}}$, $n_{\rm formula} = 2.2367\times10^{-3}\ \text{Å}^{-3}$, so $n_{\mathrm C} = 22 n_{\rm formula}$ etc.:

| $E$ (eV) | $\sum_i 2r_e\lambda n_i f_{2,i}$ ($\mathrm{mm^{-1}}$) | mass-weighted Elam photo | ratio | Elam total | ratio |
| --- | --- | --- | --- | --- | --- |
| 2000 | 51.4536 | 55.1720 | 0.933 | 55.3085 | 0.930 |
| 8000 | 0.78039 | 0.82264 | 0.949 | 0.87477 | 0.892 |

The number-density sum and the mass-fraction-weighted sum agree to $5$–$7\%$ (again tabulation-level), confirming that the two additivity statements in I.3 are the same physics.

Scattering fraction of the true narrow-beam $\mu_{\rm tot}$ that this model omits, from Elam:

| medium | $E$ (eV) | $(\mu_{\rm coh}+\mu_{\rm incoh})/\mu_{\rm tot}$ |
| --- | --- | --- |
| Al | 8000 | 1.6 % |
| Al | 20000 | 9.9 % |
| C | 8000 | 7.3 % |
| C | 20000 | 50.8 % |

So the no-scatter assumption is quantitatively safe for mid-$Z$ filters below $\sim10\ \mathrm{keV}$ and becomes a first-order error for low-$Z$ filters above $\sim15\ \mathrm{keV}$. This is a **scope limit to record**, not an arithmetic discrepancy.

### Leakage disclosure

While extracting signatures and docstrings for [I.1](#i1-what-must-be-produced), the whole of `primary_transmission` (43 lines, mostly validation) came into view in the same `sed` window as its docstring. Its body is one `np.exp(-np.tensordot(...))`; the derivation in [I.2](#i2-bouguerbeer-law) is the textbook Bouguer–Beer solution and was not shaped by it. Nothing else in the three anchors, and no part of [Part IV](#part-iv--retained-author-packet-and-2026-08-14-verifier-record), was read before Part I was written to disk.

## Part II — Diff against the implementation (verifier)

Read after Part I was committed to disk.

### II.1 `linear_attenuation_inv_mm` and `absorption_length_ang`

`crystal.py::absorption_length_ang` literally computes

```text
_, f2 = henke_dispersion(element, photon_E_eV)
lam = HC_EV_ANG / photon_E_eV
beta_idx = R_E_ANG * lam**2 / (2.0 * np.pi) * number_density_per_ang3 * f2
k = 2.0 * np.pi / lam
mu = 2.0 * k * beta_idx  # 1/Angstrom
return 1.0 / mu
```

`attenuation.py::_mu_total_inv_ang` then does `mu = mu + 1.0 / absorption_length_ang(el, E_cpu, n_i)` over the composition pairs, and `linear_attenuation_inv_mm` returns `_mu_total_inv_ang(composition, energy) * 1.0e7`.

| derived quantity | Part I | implementation | agreement |
| --- | --- | --- | --- |
| $\beta$ | $r_e\lambda^2 n_a f_2/(2\pi)$ | `R_E_ANG * lam**2 / (2*pi) * n * f2` | identical |
| amplitude$\to$intensity factor | $\mu = 2k\beta$ | `mu = 2.0 * k * beta_idx` | identical, factor $2$ present |
| $k$ | $2\pi/\lambda$ | `2.0 * np.pi / lam` | identical |
| closed form | $\mu = 2 r_e\lambda n_a f_2$ | same after cancellation | identical |
| mixture rule | $\sum_i 1/L_{{\rm abs},i}$, each at **partial** $n_i$ | `1/absorption_length_ang(el, E, n_i)` with `n_i` the `MediumSpec` partial density | identical — the pure-element-density trap of I.3 is avoided |
| unit conversion | $\mu[\mathrm{mm^{-1}}] = 10^{7}\mu[\text{Å}^{-1}]$ | `* 1.0e7` | identical, applied once |

No factor, sign, exponent, or unit diverges. The only numeric divergence is in the truncated physical constants: `HC_EV_ANG = 12398.4198` against CODATA $12398.419843320026$ ($3.5\times10^{-9}$ relative) and `R_E_ANG = 2.8179403e-5` against $2.8179403262\times10^{-5}$ ($9.3\times10^{-9}$ relative), summing to the $1.28\times10^{-8}$ offset seen in [II.4](#ii4-numeric-diff). Irrelevant next to the few-percent $f_2$ tabulation spread.

### II.2 `ray_box_path_lengths`

Term-for-term against [I.4](#i4-slab-crossing-geometry-ray-versus-oriented-box):

| derived symbol | implementation | agreement |
| --- | --- | --- |
| $o_a = -\mathbf c\cdot\hat{\mathbf e}_a$ | `origin_local = -center @ basis` | identical; source implicitly at the lab origin |
| $d_a = \hat{\mathbf d}\cdot\hat{\mathbf e}_a$ | `direction_local = directions @ basis` | identical |
| $h_a = (w_x/2, h_y/2, t/2)$ | `half_extent = [size_mm[0]/2, size_mm[1]/2, thickness_mm/2]` | identical; thickness is the third (normal) axis |
| segment clip $[0, D_p]$ | `enter = zeros_like(distance)`, `exit = distance.copy()` | identical — initialised to the segment, then narrowed |
| $s_a^{\mp} = (\mp h_a - o_a)/d_a$ sorted | `first`/`second` then `np.minimum`/`np.maximum` | identical, and the sort makes the result sign-of-$d_a$ agnostic |
| parallel branch | `parallel = abs(component) <= 64*eps`; `lower=-inf`, `upper=+inf`; an or-accumulated `missed` flag on the outside test | identical to the $\varnothing$ / $(-\infty,\infty)$ dichotomy |
| $s_{\rm in}=\max$, $s_{\rm out}=\min$ | `enter = maximum(enter, lower)`, `exit = minimum(exit, upper)` | identical |
| $\ell = \max(0, s_{\rm out}-s_{\rm in})$ | `length = np.maximum(exit - enter, 0.0)` | identical |

One implementation detail has no counterpart in the derivation:

```text
length[missed | (length <= length_tolerance)] = 0.0
```

with `length_tolerance = 64*eps*max(1, max(half_extent))`. This is a grazing-edge denoiser at the $\sim10^{-14}\,\mathrm{mm}$ level, exactly the measure-zero corner case flagged in [I.4](#i4-slab-crossing-geometry-ray-versus-oriented-box). It is a numerical hygiene clamp, not a physics term.

Critically, there is **no** separate $1/\cos\theta$ obliquity multiplier and no projected-footprint mask anywhere in the routine — obliquity emerges from the slab interval as derived, so the double-count risk of [I.4](#i4-slab-crossing-geometry-ray-versus-oriented-box) does not materialise.

### II.3 `primary_transmission`, $\Delta\Omega_p$, and the flux assembly

`primary_transmission` is `np.exp(-np.tensordot(paths, coefficient, axes=([-1], [0])))` on `paths[..., n_f]` and `coefficient[n_f, n_E]`, i.e. exactly $\exp(-\sum_j \ell_{pj}\mu_j(E))$, with an explicit `return np.ones(...)` for the $n_f = 0$ case — the exact multiplicative identity of [I.2](#i2-bouguerbeer-law), not a numerically-close one. Non-negativity of both inputs is validated, so $T_p\in(0,1]$ is enforced rather than assumed.

`geometry.py::planar_detector_rays` builds

```text
obliquity = directions @ normal
solid_angle = pixel_area_mm2 * obliquity / distance**2
```

which is the small-pixel $\Delta\Omega_p = A_p\cos\theta_p/D_p^{2}$ of [I.5](#i5-solid-angle-and-the-flux-definition), including the $\cos\theta_p$ projection. `obliquity <= 0` raises. For an unpixelated detector, the same function instead integrates its finite rectangular face exactly; the scalar Eagle path reuses the centred-rectangle specialization.

`results/model.py::SpatialResult._materialize` assembles

```text
factor.intrinsic_by_tile[tile]
* self.ray_map.solid_angle_sr[rows, columns, None]
* transmission
```

with `SpectralFactors.intrinsic_by_tile` documented as "photons per incident electron per eV per sr. Pixel solid angle, filter transmission, and detector response have not been applied." That is $F_p = I_{q(p)}\,\Delta\Omega_p\,T_p$ term for term, and the units close: $\mathrm{sr^{-1}}\times\mathrm{sr}\times 1$.

**Single ownership of $\Delta\Omega_p$ confirmed.** `spectra(measured=True)` calls `self.detector.score(..., scale=1.0)`, and `scale` is the *only* hook through which any response in `detectors/spec.py` (`Timepix`, `EagleXO`, `LegacyEDS`) applies a flux normalisation — each does `intrinsic_density * scale` and then applies QE/blur only. The $\Omega\times\mathrm{QE}$ reading of the Eagle module belongs to the legacy scalar path (`case['domega_sr'] -> r['scale']`), which never composes with `ray_map.solid_angle_sr`. The $D_p^{-4}$ failure mode of [I.5](#i5-solid-angle-and-the-flux-definition) does not occur.

Filter ordering is consistent by construction: `instrument/attenuation.py::attenuation_matrix` and `geometry.py::filter_path_lengths` both iterate `scene.filters` in declaration order, so row $j$ of `mu_by_filter_inv_mm` pairs with column $j$ of `path_length_mm`. Order invariance makes a mispairing harmless for $T_p$ only if the permutation is joint; it is.

### II.4 Numeric diff

Independent side: the closed form $\mu = 2r_e\lambda n_a f_2$ evaluated in a standalone script from CODATA constants and `xraydb.f2_chantler`, plus the independent Elam tabulation via `xraydb.mu_elam` (a different compilation from the Chantler/FFAST table the code interpolates). No PyRITE helper on the reference side.

Aluminium, $n_a = 0.060238\ \text{Å}^{-3}$, $\mu$ in $\mathrm{mm^{-1}}$:

| $E$ (eV) | `linear_attenuation_inv_mm` | independent closed form | rel. diff | Elam photo | code/Elam |
| --- | --- | --- | --- | --- | --- |
| 1000 | 296.743756 | 296.743759 | $1.3\times10^{-8}$ | 319.265 | 0.930 |
| 3000 | 210.089936 | 210.089939 | $1.3\times10^{-8}$ | 212.280 | 0.990 |
| 5000 | 49.944145 | 49.944145 | $1.3\times10^{-8}$ | 51.877 | 0.963 |
| 8000 | 13.038243 | 13.038244 | $1.3\times10^{-8}$ | 13.361 | 0.976 |
| 12000 | 3.829276 | 3.829276 | $1.3\times10^{-8}$ | 3.992 | 0.959 |
| 20000 | 0.804761 | 0.804761 | $1.3\times10^{-8}$ | 0.837 | 0.962 |

Compound `MediumSpec`, Kapton $\mathrm{C_{22}H_{10}N_2O_5}$ at $\rho = 1.42\ \mathrm{g\,cm^{-3}}$ entered as four partial number densities:

| $E$ (eV) | `linear_attenuation_inv_mm` | independent $\sum_i 2r_e\lambda n_i f_{2,i}$ | rel. diff |
| --- | --- | --- | --- |
| 2000 | 51.453642 | 51.453642 | $1.3\times10^{-8}$ |
| 8000 | 0.780392 | 0.780392 | $1.3\times10^{-8}$ |

Geometry, `ray_box_path_lengths` against an $8\times10^{6}$-sample independent arc-length quadrature of the indicator function $\mathbf 1\bigl[\lvert u_a(s)\rvert\le h_a\ \forall a\bigr]$ over $s\in[0,D_p]$ (no shared code with the slab routine), lengths in mm:

| case | code | quadrature | closed form |
| --- | --- | --- | --- |
| normal incidence, $t=2$ | 2.000000 | 2.000024 | $t = 2$ |
| oblique $30^\circ$, $t=2$ | 2.309401 | 2.309429 | $t/\cos\theta = 2.309401$ |
| plate tilted $25^\circ$, off-axis pixel, $t=3$ | 3.182362 | 3.182354 | $t/\cos\theta = 3.182362$ |
| lateral miss | 0.000000 | 0.000000 | $0$ |
| genuine side escape ($t=20$, exits an $x$ face) | 10.007997 | 10.008008 | $10/\cos\theta = 10.007997$, **not** $t/\cos\theta$ |
| pixel inside the plate | 5.000000 | 5.000000 | $D_p - s_{\rm in} = 5$ |
| plate entirely beyond the pixel | 0.000000 | 0.000000 | $0$ |
| ray parallel to a face, laterally outside | 0.000000 | 0.000000 | $0$ |
| grazing a corner | 10.015987 | 10.015998 | finite, no NaN |

The quadrature residual is the $O(D_p/N)$ grid bias of the reference, one-sided as expected. Every derived limiting case in [I.4](#i4-slab-crossing-geometry-ray-versus-oriented-box) is reproduced, including the side-escape case that distinguishes a true slab intersection from a $t/\cos\theta$ shortcut.

Structural checks on `primary_transmission`: permuting the filter axis of both arguments jointly changes $T$ by $5.6\times10^{-17}$ (round-off only); a zero-filter call returns exact $1.0$; and the returned array equals $\exp(-\text{paths}\cdot\mu)$ computed independently to machine precision.

The 36 anchor tests named in the ledger (`tests/materials/test_attenuation.py`, `tests/instrument/test_attenuation.py`, `tests/instrument/test_geometry.py`) pass.

## Part III — Adjudication (verifier)

### III.1 Cheap filters

| filter | result | evidence |
| --- | --- | --- |
| dimensional consistency | **pass** | $[\mathrm{mm^{-1}}]\cdot[\mathrm{mm}]$ dimensionless; $\mathrm{sr^{-1}}\cdot\mathrm{sr}$ closes $F_p$; single $10^{7}$ Å→mm conversion |
| limit $N_f\to0$ | **pass** | exact `np.ones`, not $\exp(-0)$ |
| limit $\mu\to0$ or $\ell\to0$ | **pass** | $T\to1$ |
| limit $\mu\ell\to\infty$ | **pass** | $T\to0^+$, monotone |
| normal incidence | **pass** | $\ell = t$ to machine precision |
| oblique incidence | **pass** | $\ell = t/\cos\theta$ derived, not bolted on |
| miss / parallel-outside / behind-pixel | **pass** | exact $0$, no NaN |
| sign / positivity | **pass** | $\mu\ge0$, $\ell\ge0$ validated; $T\in(0,1]$ |
| plate-order symmetry | **pass** | $5.6\times10^{-17}$ |
| single ownership of $\Delta\Omega_p$ | **pass** | `scale=1.0` on the spatial path |

### III.2 Verdict

The implementation matches the independent derivation of Part I with **no divergent factor, sign, exponent, unit, or convention**. The largest numeric disagreement traceable to the code is $1.3\times10^{-8}$, entirely explained by two truncated physical constants.

Verdict: **`rederived`**.

This independent pass reaches the same verdict as the 2026-08-14 pass retained in Part IV, by a different route (closed-form $\mu$ plus an external Elam tabulation rather than a source-level $f_2$ recomputation; quadrature rather than closed forms for the geometry). The findings below are additions, not contradictions.

### III.3 Findings and scope limits

None of these is an arithmetic discrepancy in the three anchored expressions; all are scope or provenance items a human should weigh before sign-off.

1. **Cited source versus data actually used.** The ledger `Source` field names Henke et al. 1993 with its DOI, but `materials/atomic.py::henke_dispersion` states in its own Notes that it returns "Chantler/FFAST via xraydb; name kept for API compatibility with the old Henke/CXRO implementation." The *convention* is Henke's ($f_2$ in electron units, $\sigma_{\rm abs} = 2r_e\lambda f_2$) and the formula is unaffected, but the tabulation is Chantler. The ledger row should say so; the `absorption-length` row has the same issue. Chantler-versus-Elam spread is the 1–7 % seen in [II.4](#ii4-numeric-diff), so this is not a rounding-level distinction.
2. **$\mu$ is photoabsorption-only, and the primary beam also loses scattered photons.** Part IV lists scattering among excluded processes but does not quantify the resulting bias in $\mu$ itself. From the Elam decomposition, $(\mu_{\rm coh}+\mu_{\rm incoh})/\mu_{\rm tot}$ is 1.6 % for Al at 8 keV, 9.9 % for Al at 20 keV, 7.3 % for C at 8 keV, and 50.8 % for C at 20 keV. $T_p$ is therefore systematically **over**-estimated, mildly for mid-$Z$ filters below $\sim10\ \mathrm{keV}$ and at first order for low-$Z$ filters above $\sim15\ \mathrm{keV}$. The absence of a build-up factor pushes the other way but does not cancel it. This belongs in the ledger `Notes`. **Resolved 2026-09-20** by [#104](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/104): the scattering term is now summed into $\mu$ under the [`narrow-beam-total-attenuation`](../ledger-crystallography-atomic-data.md#narrow-beam-total-attenuation) claim. The build-up factor is still absent, so the good-geometry condition this finding implies remains an open scope limit rather than a bias.
3. **No plate-overlap validation.** `instrument/model.py::validate_downstream_scene` checks name uniqueness, downstream normal orientation, and that each plate volume lies between source and detector plane — but never that plate interiors are disjoint. Two overlapping plates double-count the overlap chord in $\sum_j\mu_j\ell_{pj}$, which is unphysical unless the plates are deliberately nested. The additivity of [I.2](#i2-bouguerbeer-law) assumes disjointness. Worth a scene-validation check or an explicit documented caveat; it is a scene-construction gap, not an error in the three anchors.
4. **Finite-face solid angle.** `planar_detector_rays` integrates an unpixelated `PlanarDetector` face exactly, reducing to $\Omega = 4\arctan\!\bigl[(w/2)(h/2)\big/\bigl(d\sqrt{(w/2)^2+(h/2)^2+d^2}\bigr)\bigr]$ for a centred face. Pixel grids retain the centre-ray $A_p\cos\theta_p/D_p^{2}$ approximation; its fine-grid convergence is an explicit regression anchor.
5. **Phasor-convention wording.** Part IV writes the passive index as $\tilde n = 1-\delta+i\beta$ under $\exp(-i\omega t)$, whereas `crystal.py::optical_constants` documents $n = 1-\delta-i\beta$. Both are standard under their respective time conventions and both give $\mu = 2k\beta$ with $\beta>0$; $\mu$ is unaffected. Flagged only so the two documents do not read as contradicting each other.
6. **`PixelRays` does not normalise `directions_lab`.** `planar_detector_rays` supplies unit vectors, but `PixelRays.__post_init__` validates only shape. A hand-constructed `PixelRays` with non-unit directions would make $s$ not an arc length and silently rescale every $\ell_{pj}$. Robustness note for the implementation owner, outside this claim's expressions.
7. **Centre-ray sampling.** Neither the finite pixel area nor a finite source spot is integrated, so a pixel straddling a plate edge sees a hard $0/1$ transition instead of the physical penumbra. Already recorded in the ledger `Notes`; retained here for completeness.

### III.4 Suggested ledger change

Human-applied only. Proposed edits to the `positioned-filter-attenuation` row of `docs/validation/ledger-detector-forward-models.md`:

- `Status:` `unverified` $\rightarrow$ `rederived`.
- `Checks:` append "two independent fresh-context re-derivations (2026-08-14, 2026-09-13) agree; second pass cross-checked $\mu$ against the independent Elam tabulation (1–7 % over 1–20 keV, tabulation-level) and `ray_box_path_lengths` against arc-length quadrature over normal, oblique, tilted, side-escape, miss, parallel, interior, and corner cases; single ownership of $\Delta\Omega_p$ confirmed (`scale=1.0` on the spatial path)."
- `Source:` qualify as Henke *convention* with Chantler/FFAST tabulation, or retitle to match `henke_dispersion`'s stated backend.
- `Notes:` add (a) $\mu$ omits coherent/incoherent scattering, over-estimating $T_p$ by up to $\sim10\ \%$ (Al, 20 keV) and $\sim50\ \%$ (C, 20 keV); (b) scene validation does not forbid overlapping plates; (c) $\Delta\Omega_p$ is the small-pixel $A\cos\theta/D^2$ form, not the exact rectangular pyramid.

Not `signed-off`. That transition belongs to a human who has read the source and this diff.

## Part IV — Retained author packet and 2026-08-14 verifier record

Preserved for provenance. Markup was normalised to the [LaTeX/MyST style rules](../formatting-style.md) — the original used bare parentheses around raw LaTeX, which renders as literal text. No number, sign, factor, unit, or word of adjudication was changed.

### Claim and source (author)

For a point source, let the unit ray to detector pixel centre $p$ be $\hat{\boldsymbol u}_p$. Its exact intersection length with finite filter box $j$ is $\ell_{pj}$ in millimetres. The primary-photon transmission and pixel flux are

$$
T_p(E) = \exp\!\left[-\sum_j \mu_j(E)\ell_{pj}\right],
\qquad
F_p(E) = I_{q(p)}(E)\,\Delta\Omega_p\,T_p(E).
$$

Here $I_{q(p)}$ is the pre-filter source density for one angular tile in photons per electron per electronvolt per steradian and $\Delta\Omega_p$ is the pixel solid angle. The exponential is the Bouguer–Beer law. Elemental attenuation coefficients come from the Henke/Chantler imaginary scattering factors through the already ledgered `absorption-length` implementation. The primary tabulation source is Henke et al., *Atomic Data and Nuclear Data Tables* **54**, 181–342 (1993), DOI [10.1006/adnd.1993.1013](https://doi.org/10.1006/adnd.1993.1013).

### Implementation derivation (author)

For composition number densities $n_i$, independent elemental absorption rates add:

$$
\mu_j(E) = \sum_i \frac{1}{L_{\mathrm{abs},i}(E)}.
$$

The existing absorption-length routine returns $L_{\mathrm{abs}}$ in angstroms, hence the public helper multiplies the summed inverse length by $10^7\ \mathrm{\AA/mm}$ to return $\mathrm{mm}^{-1}$.

In plate-local coordinates, each ray is clipped against all three box slabs. The largest entry parameter and smallest exit parameter delimit the in-box interval; clipping that interval to the finite source–pixel segment gives $\ell_{pj}$. This same interval handles normal incidence, oblique thickness, side escape, and misses without a separate projected-mask approximation.

Because exponentials multiply, serial passive plates give

$$
\prod_j e^{-\mu_j\ell_{pj}}
= e^{-\sum_j\mu_j\ell_{pj}},
$$

so plate order cannot change transmission. Declared order remains part of provenance.

### Units, assumptions, and limits (author)

- $\mu_j$ has units $\mathrm{mm}^{-1}$, $\ell_{pj}$ has units mm, and the exponential argument is dimensionless.
- Zero filters, zero path length, or an uncovered pixel gives $T_p(E)=1$ exactly. For a normal ray through a full plate, $\ell = t$. For finite positive $\mu\ell$, $0 < T \leq 1$.
- A photon-energy node at exactly $E=0$ takes the photoabsorption limit $\mu_j\to+\infty$ as $E\to0^+$ (tabulated coefficients exist only for $E>0$): $T_p(0)=0$ when $\ell_{pj}>0$ for any such plate, and $T_p(0)=1$ when every $\ell_{pj}=0$, since a zero path contributes no optical depth. Added 2026-09-26 (#23), after the 2026-09-13 re-derivation.
- A homogeneous passive filter and independent primary-photon attenuation are assumed. The model excludes scattering, fluorescence, diffraction, secondary production, surface reflection, and detector charge transport.
- Pixels are sampled by centre rays from a point source. The model does not integrate the finite pixel sensitive area or an extended emission volume.

### Implementation-side anchors (author)

The focused tests pin direct composition summation against elemental absorption lengths, the angstrom-to-millimetre conversion, exact zero-filter identity, a closed-form normal-incidence case, compound additivity, and plate order invariance. Geometry tests separately pin misses, partial coverage, rotation, side escape, movement, and finite source–detector clipping.

These checks are implementation evidence, not independent validation.

### Independent re-derivation (2026-08-14)

The verifier first recorded the cited Bouguer–Beer/Henke source, the three ledgered signatures, their units and assumptions, and the limiting cases. The following derivation was completed before any implementation body or this author-prepared packet was read.

Let the source be the origin and write the finite source–pixel-centre ray as

$$
\boldsymbol{x}_p(q)=q\,\hat{\boldsymbol{u}}_p,
\qquad 0\leq q\leq d_p,
$$

where $d_p$ is the source–pixel distance in millimetres. For plate $j$, let $\boldsymbol{c}_j$ be its centre, let $\{\boldsymbol{e}_{jk}\}_{k=1}^3$ be its orthonormal local axes, and let $h_{jk}$ be its three half-extents. Along local axis $k$, points inside the box satisfy

$$
-h_{jk}\leq a_{jk}+q b_{pjk}\leq h_{jk},\qquad
a_{jk}=-\boldsymbol{c}_j\mathbin{\cdot}\boldsymbol{e}_{jk},\quad
b_{pjk}=\hat{\boldsymbol{u}}_p\mathbin{\cdot}\boldsymbol{e}_{jk}.
$$

For $b_{pjk}\ne0$, sorting $(-h_{jk}-a_{jk})/b_{pjk}$ and $(h_{jk}-a_{jk})/b_{pjk}$ gives that slab's entry and exit distances. For $b_{pjk}=0$, the interval is all distances when $|a_{jk}|\leq h_{jk}$, and empty otherwise. Intersecting all three slab intervals with $[0,d_p]$ gives $[q_{\rm in},q_{\rm out}]$, hence

$$
\ell_{pj}=\max(0,q_{\rm out}-q_{\rm in}).
$$

This derivation includes finite-segment clipping, misses, side escape, and parallel rays. A centred normal ray gives $\ell=t$; an infinite lateral plate at incidence angle $\theta$ from its normal gives $\ell=t/|\cos\theta|$.

For the $\exp(-i\omega t)$ phasor convention, write the passive Henke index as $\tilde n=1-\delta+i\beta$. Intensity propagation then gives

$$
I(\ell)=I(0)e^{-2k\beta\ell},\qquad
\beta(E)=\frac{r_e\lambda^2}{2\pi}\sum_i n_i f_{2,i}(E).
$$

With $k=2\pi/\lambda$, the independently derived linear attenuation is

$$
\mu(E)=2r_e\lambda\sum_i n_i f_{2,i}(E)
      =\sum_i L_{\mathrm{abs},i}^{-1}(E).
$$

Thus each independent passive plate obeys $dI/d\ell=-\mu I$, serial factors multiply, and

$$
T_p(E)=\exp\!\left[-\sum_j\mu_j(E)\ell_{pj}\right],\qquad
F_p(E)=I_{q(p)}(E)\,\Delta\Omega_p\,T_p(E).
$$

The units are $r_e,\lambda\,[\mathrm{\AA}]$, $n_i\,[\mathrm{\AA}^{-3}]$, and therefore $\mu\,[\mathrm{\AA}^{-1}]$. Multiplication by $10^7\ \mathrm{\AA/mm}$ produces $\mathrm{mm}^{-1}$, so every $\mu_j\ell_{pj}$ is dimensionless.

#### Independent comparison (2026-08-14)

- `ray_box_path_lengths` uses the equivalent distance parameter $q$, starts with the finite interval $[0,d_p]$, intersects the same three sorted slab intervals, and returns $\max(0,q_{\rm out}-q_{\rm in})$. No factor of $d_p$, cosine, or two is missing.
- `linear_attenuation_inv_mm` sums $L_{\mathrm{abs},i}^{-1}$ and multiplies by $10^7$, exactly matching the derived angstrom-to-millimetre conversion.
- `primary_transmission` contracts the filter index as $\sum_j\ell_{pj}\mu_j(E)$ and applies one negative exponential. The materialized pixel result multiplies the pre-filter tile density, pixel solid angle, and this transmission once each.
- A source-level numeric calculation used SciPy physical constants and `xraydb.f2_chantler` directly, without either attenuation implementation helper. At 8 and 12 keV it gave respectively $[14.55358057,4.31004892]$ mm$^{-1}$ for Si at $n=0.04994\ \mathrm{\AA}^{-3}$, versus $[14.55358041,4.31004887]$ mm$^{-1}$ from the public helper. For an Al–O medium with $n_{\rm Al}=0.02345$ and $n_{\rm O}=0.03517\ \mathrm{\AA}^{-3}$, the source calculation gave $[6.06953199,1.77018238]$ mm$^{-1}$, versus $[6.06953192,1.77018236]$ mm$^{-1}$. The maximum relative difference was $1.08\times10^{-8}$, consistent with the physical-constant precision used by the two paths.
- The focused material, transmission, and finite-box geometry tests all pass (36 tests). They cover the zero-filter and uncovered-ray identities, normal and oblique incidence, grazing side escape, finite-segment misses, compound additivity, and plate-order invariance.

Cheap filters therefore pass: $\mu\ell$ is dimensionless; passive coefficients give $0<T\leq1$; no filters, $\mu\to0$, or $\ell=0$ gives $T=1$; positive $\mu$ with $\ell\to\infty$ gives $T\to0$; and the sum is invariant under a joint plate permutation. The implementation matches the independent derivation with no factor, sign, exponent, unit, or convention divergence.

Traceability finding: `ray_box_path_lengths` is named in the ledgered claim but its docstring lacks `Validation: positioned-filter-attenuation`. This is not a physics-expression discrepancy, but the marker should be added by the owning implementation context before human sign-off.

**2026-08-14 verifier verdict:** `rederived`. Suggested human-applied ledger change: `unverified` to `rederived`, record the completed fresh-context / source-level comparison, and retain the missing-marker finding until repaired. This is not human `signed-off` status.

#### Post-validation remediation

Commit `4b40378` resolved the verifier's traceability finding by adding `Validation: positioned-filter-attenuation` to the `ray_box_path_lengths` docstring. The historical finding and verdict above are retained as the validation record. The ledger remains `unverified` pending the human-applied `unverified` to `rederived` transition; this note does not claim human sign-off.

The 2026-09-13 pass re-confirms the marker is present.

## Independent verification contract

A fresh-context verifier must derive the equations and units without using the new public attenuation helper as its oracle, compare at least one element and one compound `MediumSpec` against an external or source-level calculation, check normal, grazing/parallel, miss, zero-filter, and plate-order limits, and audit the source-to-code sign and unit conversion. The verifier records any discrepancy here and updates the ledger status. Human sign-off remains separate.
