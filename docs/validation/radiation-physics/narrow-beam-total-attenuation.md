# Validation: narrow-beam-total-attenuation

Independent re-derivation of the total narrow-beam linear attenuation coefficient and of the Elam mass-to-atomic unit chain behind its scattering term. Written from the ledger row in `docs/validation/ledger-crystallography-atomic-data.md`, the two public signatures, and the cited sources; the implementation bodies were read only after the derivation and the reference numbers below were fixed.

Signatures under review:

- `pyrite.materials.crystal.scattering_attenuation_inv_ang(element, photon_E_eV, number_density_per_ang3) -> ndarray`, in $\mathring{\mathrm A}^{-1}$
- `pyrite.materials.attenuation._mu_total_inv_ang(comp, E_eV) -> ndarray`, in $\mathring{\mathrm A}^{-1}$, with `comp = [(element, n_per_Ang3), ...]`

Every reference number in this document comes from CODATA through `scipy.constants` and from `xraydb` called directly; no PyRITE helper appears on the reference side of any comparison.

## Independent derivation

### The narrow-beam exponent counts removal, not destruction

Take a collimated monoenergetic pencil beam of $N$ photons per unit time entering a homogeneous slab of one atomic species with number density $n_a$. Over a path element $\mathrm dx$ the probability that a given photon undergoes *any* interaction with an atom is $n_a\sigma_{\rm tot}\,\mathrm dx$, with $\sigma_{\rm tot}$ the total interaction cross section per atom. In *good geometry* — the detector accepts only the forward, unscattered ray — every interaction removes the photon from that ray, whether it destroys the photon (photoabsorption) or merely deflects it (coherent or incoherent scattering). Hence

$$
\frac{\mathrm dN}{\mathrm dx}=-n_a\sigma_{\rm tot}N
\quad\Longrightarrow\quad
N(t)=N_0\,e^{-\mu_{\rm tot}t},
\qquad
\mu_{\rm tot}=n_a\sigma_{\rm tot},
$$

the Bouguer–Beer law with the *narrow-beam* (good-geometry) linear attenuation coefficient.

The partition of $\sigma_{\rm tot}$ used by Hubbell & Seltzer (NIST Standard Reference Database 126, XCOM) is additive over channels,

$$
\sigma_{\rm tot}(E)
=\sigma_{\rm photo}(E)+\sigma_{\rm coh}(E)+\sigma_{\rm incoh}(E)
+\sigma_{\rm pair}(E)+\sigma_{\rm ph.n}(E),
$$

with $\sigma_{\rm pair}$ identically zero below the nuclear-field threshold $2m_ec^2=1.022$ MeV (and $4m_ec^2$ in the electron field) and $\sigma_{\rm ph.n}$ a sub-percent giant-resonance term peaking near $10$–$30$ MeV. Both vanish across the band this claim covers, so

$$
\boxed{\;\mu_{\rm tot}(E)=n_a\left[\sigma_{\rm photo}(E)+\sigma_{\rm coh}(E)+\sigma_{\rm incoh}(E)\right]\;}
$$

and, splitting off the photoabsorption piece already owned by `absorption-length`,

$$
\mu_{\rm tot}=\underbrace{2r_e\lambda n_a f_2}_{\mu_{\rm photo}}
+\underbrace{n_a\sigma_{\rm scat}}_{\mu_{\rm scat}},
\qquad
\sigma_{\rm scat}\equiv\sigma_{\rm coh}+\sigma_{\rm incoh}.
$$

Additivity over channels is exact at the level of a first-order rate: the channels have distinct final states, so their partial widths add. Additivity over *species* is the independent-atom approximation, giving for a mixture with per-element number densities $n_i$

$$
\mu_{\rm tot}(E)=\sum_i n_i\left[\sigma_{{\rm photo},i}(E)+\sigma_{{\rm scat},i}(E)\right]
=\sum_i\left[\frac{1}{L_{{\rm abs},i}(E)}+n_i\sigma_{{\rm scat},i}(E)\right].
$$

That is what `_mu_total_inv_ang` must return for `comp = [(element, n_per_Ang3), ...]`.

### Why photoabsorption alone is the wrong exponent but the right $\beta$

For a passive medium $n=1-\delta-i\beta$ with $\beta=r_e\lambda^2n_af_2/(2\pi)$, and the field decay it implies gives the intensity coefficient $\mu=4\pi\beta/\lambda=2r_e\lambda n_af_2$. The optical theorem ties the imaginary forward amplitude to the *total* cross section, but the $f_2$ tabulations actually in use (Chantler/FFAST, Henke/CXRO) are built from photoabsorption alone, $f_2=\sigma_{\rm photo}/(2r_e\lambda)$. So $\mu_{\rm photo}$ is exactly the coefficient the refractive index needs, and folding $\mu_{\rm scat}$ into $\beta$ would double-count the elastic amplitude that $\delta$ and the coherent forward wave already carry: coherent *forward* scattering is not removal, it *is* the refractive index. The ledger's split — `absorption-length` photoabsorption-only for $\beta$, $\chi_0$, the in-medium wavevector and grazing reflectivity; `narrow-beam-total-attenuation` total for transmission exponents — is therefore the physically correct partition of ownership, not a bookkeeping convenience.

### Unit chain for the Elam scattering term

`xraydb.mu_elam(el, E, kind=...)` returns a *mass* attenuation coefficient $(\mu/\rho)$ in $\mathrm{cm^2\,g^{-1}}$. The per-atom cross section follows from the mass of one atom, $m_a=A/N_A$, with $A$ the relative atomic mass in $\mathrm{g\,mol^{-1}}$ and $N_A$ in $\mathrm{mol^{-1}}$:

$$
\sigma\;[\mathrm{cm^2\,atom^{-1}}]
=\left(\frac{\mu}{\rho}\right)\!\left[\frac{\mathrm{cm^2}}{\mathrm g}\right]
\times\frac{A}{N_A}\left[\frac{\mathrm g}{\mathrm{atom}}\right].
$$

The identity behind it is $\mu=\rho\,(\mu/\rho)=n_a\sigma$ with $n_a=\rho N_A/A$: the density cancels, as it must for an atomic datum, leaving $\sigma=(\mu/\rho)A/N_A$. The direction of the multiply is fixed by that cancellation — dividing by $A/N_A$ instead would be wrong by $(N_A/A)^2\sim10^{44}$. Converting $\mathrm{cm^2}\to\mathring{\mathrm A}^2$ uses $1\,\mathrm{cm}=10^{8}\,\mathring{\mathrm A}$, so $1\,\mathrm{cm^2}=10^{16}\,\mathring{\mathrm A}^2$:

$$
\boxed{\;\sigma_{\rm scat}(E)\;[\mathring{\mathrm A}^2]
=\left[\left(\frac{\mu}{\rho}\right)_{\rm coh}+\left(\frac{\mu}{\rho}\right)_{\rm incoh}\right]
\frac{A}{N_A}\times10^{16}\;}
$$

and finally

$$
\mu_{\rm scat}(E)\;[\mathring{\mathrm A}^{-1}]
=n_a\;[\mathring{\mathrm A}^{-3}]\times\sigma_{\rm scat}(E)\;[\mathring{\mathrm A}^{2}].
$$

Dimensionally $\mathring{\mathrm A}^{-3}\cdot\mathring{\mathrm A}^{2}=\mathring{\mathrm A}^{-1}$, the declared return unit, and the same unit as $1/L_{\rm abs}$, so the two terms are commensurable with no further conversion.

Scale sanity: for graphite at 20 keV the derivation below gives $\mu_{\rm tot}=9.72\times10^{-9}\,\mathring{\mathrm A}^{-1}=0.972\ \mathrm{cm^{-1}}$, i.e. an attenuation length of $1.03$ cm — the right order for a centimetre of graphite at 20 keV, and a factor $10^{8}$ away from what a missing or inverted $10^{16}$ would produce.

### Limiting cases

1. **Empty medium.** $n_a\to0\Rightarrow\mu_{\rm scat}\to0$ and $\mu_{\rm tot}\to\mu_{\rm photo}\to0$: transmission $\to1$.
2. **Sign.** $\sigma_{\rm coh},\sigma_{\rm incoh}\ge0$ and $n_a\ge0$, so $\mu_{\rm scat}\ge0$. Adding scattering can only raise $\mu$ and only lower the transmission; an energy at which the new term lowers the exponent would be inadmissible.
3. **Low-energy (Rayleigh/Thomson) limit.** As $E\to0$ the atomic form factor $F(q,Z)\to Z$ over the whole accessible $q$ range, so

   $$
   \sigma_{\rm coh}\to\int\frac{r_e^2}{2}\left(1+\cos^2\theta\right)Z^2\,\mathrm d\Omega
   =Z^2\,\frac{8\pi}{3}r_e^2=Z^2\sigma_{\rm T},
   $$

while $\sigma_{\rm incoh}\to0$ because the incoherent scattering function $S(q,Z)\to0$ as $q\to0$ (Pauli blocking of small momentum transfer). Both are closed-form predictions testable against the tabulation.
4. **Free-electron (Compton) limit.** Far above all binding energies $S(q,Z)\to Z$, so $\sigma_{\rm incoh}\to Z\,\sigma_{\rm KN}$ with

   $$
   \sigma_{\rm KN}(\varepsilon)=2\pi r_e^2\left\{
   \frac{1+\varepsilon}{\varepsilon^2}
   \left[\frac{2(1+\varepsilon)}{1+2\varepsilon}-\frac{\ln(1+2\varepsilon)}{\varepsilon}\right]
   +\frac{\ln(1+2\varepsilon)}{2\varepsilon}
   -\frac{1+3\varepsilon}{(1+2\varepsilon)^2}\right\},
   \qquad
   \varepsilon=\frac{E}{m_ec^2},
   $$

and simultaneously $\sigma_{\rm coh}$ collapses, since $F(q,Z)$ is confined to $q\lesssim1/a_0$ while $q_{\max}=2k$ grows.
5. **Thomson limit of (4).** $\varepsilon\to0$ gives $\sigma_{\rm KN}\to\tfrac{8}{3}\pi r_e^2=\sigma_{\rm T}$, the internal consistency check on the closed form.

Cases (3) and (4) pin the unit chain at *both* ends of the band from closed form alone, which is stronger than a single-point table comparison.

## Independent numerical reference

### The two closed-form pins on the unit chain

**Bottom of the band, $Z^2\sigma_{\rm T}$ (case 3).** With $\sigma_{\rm T}=\tfrac{8}{3}\pi r_e^2=0.6652459$ barn from CODATA $r_e$, the tabulated coherent cross section at 100 eV against the closed form:

| element | $Z$ | $\sigma_{\rm coh}(100\ \mathrm{eV})$ (barn) | $Z^2\sigma_{\rm T}$ (barn) | ratio | $\sigma_{\rm incoh}$ (barn) |
| --- | --- | --- | --- | --- | --- |
| C | 6 | $23.8845$ | $23.9489$ | $0.9973$ | $0.0030$ |
| Al | 13 | $112.4110$ | $112.4266$ | $0.9999$ | $0.0085$ |
| Si | 14 | $130.2972$ | $130.3882$ | $0.9993$ | $0.0077$ |
| Cu | 29 | $559.1101$ | $559.4718$ | $0.9994$ | $0.0078$ |

Agreement to $0.03$–$0.3\%$, and the vanishing incoherent column confirms the $S(q,Z)\to0$ half of case 3.

**Top of the band, Klein–Nishina (case 4).** Ratio $\sigma_{\rm incoh}/(Z\sigma_{\rm KN})$:

| $E$ | $\sigma_{\rm KN}$ (barn/e) | C | Al | Si |
| --- | --- | --- | --- | --- |
| 100 keV | $0.49275$ | $0.9890$ | $0.9707$ | $0.9680$ |
| 500 keV | $0.28917$ | $1.00001$ | $0.9981$ | $0.9979$ |
| 800 keV | $0.23496$ | $1.00018$ | $0.9995$ | $0.9996$ |

Within $0.2\%$ at 500 keV for all three, confirming the ledger's stated free-electron limit. Two independent closed forms, differing by eight orders of magnitude in $\sigma$ and three in $E$, both reproduce the tabulation after the same $A/N_A\times10^{16}$ conversion: the unit chain is pinned, and no constant factor error survives.

**Independent quadrature of the coherent column.** Integrating the Thomson differential cross section against the Waasmaier–Kirfel form factor,

$$
\sigma_{\rm coh}=\int_0^\pi\frac{r_e^2}{2}\left(1+\cos^2\theta\right)
\lvert F(q,Z)\rvert^2\,2\pi\sin\theta\,\mathrm d\theta,
\qquad
q=\frac{4\pi\sin(\theta/2)}{\lambda},
$$

reproduces the Elam `coh` column to $0.1\%$ for C and Al at 8, 20 and 60 keV. That is an end-to-end check of the coherent term from a different tabulation (form factors, not cross sections) and a different functional form.

### Cross sections and the graphite case

$\sigma_{\rm scat}=\sigma_{\rm coh}+\sigma_{\rm incoh}$, in barn/atom:

| element | 8 keV | 20 keV | 60 keV |
| --- | --- | --- | --- |
| C | $6.678$ | $4.474$ | $3.384$ |
| Al | $36.553$ | $15.308$ | $8.162$ |
| Si | $41.923$ | $17.458$ | $8.945$ |

For graphite at $\rho=2.26\ \mathrm{g\,cm^{-3}}$, $n_a=0.113313\,\mathring{\mathrm A}^{-3}$:

| $E$ | $\mu_{\rm photo}$ ($\mathring{\mathrm A}^{-1}$) | $\mu_{\rm scat}$ ($\mathring{\mathrm A}^{-1}$) | $\mu_{\rm tot}/\mu_{\rm photo}$ |
| --- | --- | --- | --- |
| 8 keV | $9.0515\times10^{-8}$ | $7.5670\times10^{-9}$ | $1.0836$ |
| 20 keV | $4.6540\times10^{-9}$ | $5.0696\times10^{-9}$ | $2.0893$ |
| 60 keV | $1.2472\times10^{-10}$ | $3.8341\times10^{-9}$ | $31.742$ |

The $2.0893$ independently reproduces the ledger's "graphite at 20 keV was under-stating the exponent by a factor $2.09$".

### Literature cross-check

Converting the mixed total back to a mass basis gives, for carbon at 20 keV, $\mu/\rho=0.4302\ \mathrm{cm^2\,g^{-1}}$. The Elam `total` column is $0.44197$, and the NIST XAAMDI/XCOM tabulated value for carbon at 20 keV including coherent scattering is $0.4423\ \mathrm{cm^2\,g^{-1}}$. Elam and XCOM agree to $0.07\%$, so the Elam scattering columns are a faithful stand-in for the cited XCOM decomposition, and the $-2.7\%$ residual of the *mixed* total is attributable to the photoabsorption compilation, not to the scattering term.

### Size of the term that was missing

$\mu_{\rm scat}/\mu_{\rm tot}$ at elemental solid densities:

| element ($\rho$) | 1 keV | 5 keV | 8 keV | 17 keV | 20 keV | 60 keV | 100 keV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Be (1.85) | 0.1% | 7.3% | 23.1% | 74.0% | 82.5% | 99.3% | 99.9% |
| C (2.26) | 0.1% | 2.5% | 7.7% | 40.5% | 52.1% | 96.8% | 99.3% |
| Al (2.70) | 0.2% | 0.6% | 1.7% | 7.4% | 10.3% | 66.0% | 89.3% |
| Si (2.33) | 0.2% | 0.5% | 1.4% | 6.3% | 8.7% | 60.2% | 86.5% |
| Cu (8.96) | 0.0% | 1.6% | 3.8% | 1.6% | 2.1% | 15.3% | 37.5% |
| W (19.3) | 0.3% | 1.4% | 3.3% | 2.6% | 3.3% | 14.3% | 6.5% |

The omission this claim repairs is not a small correction: for the low-$Z$ filter and substrate materials PyRITE uses, the photoabsorption-only exponent is wrong by a factor of two at 20 keV and by more than an order of magnitude at 60 keV. Limiting case (2) holds everywhere in the table.

### Mixed-compilation error

Residual of $(\mu/\rho)^{\rm Chantler}_{\rm photo}+(\mu/\rho)^{\rm Elam}_{\rm scat}$ against the all-Elam `total` column:

| element | 1 keV | 5 keV | 8 keV | 20 keV | 60 keV | 200 keV |
| --- | --- | --- | --- | --- | --- | --- |
| Be | $-9.08\%$ | $-9.22\%$ | $-6.77\%$ | $-1.38\%$ | $-0.03\%$ | $-0.00\%$ |
| C | $-6.21\%$ | $-5.66\%$ | $-5.16\%$ | $-2.65\%$ | $-0.09\%$ | $-0.00\%$ |
| Al | $-7.04\%$ | $-3.70\%$ | $-2.38\%$ | $-3.44\%$ | $-0.65\%$ | $-0.01\%$ |
| Si | $-6.43\%$ | $-3.44\%$ | $-2.01\%$ | $-3.41\%$ | $-0.65\%$ | $-0.01\%$ |
| Cu | $-2.66\%$ | $-4.79\%$ | $-3.58\%$ | $-0.92\%$ | $-1.48\%$ | $-0.32\%$ |
| W | $-4.28\%$ | $-4.71\%$ | $-3.30\%$ | $-2.00\%$ | $-1.89\%$ | $-1.45\%$ |

The residual is entirely a photoabsorption disagreement: it vanishes above $\sim60$ keV, precisely where photoabsorption stops contributing and the scattering term becomes the whole coefficient. Its sign is uniformly negative — Chantler/FFAST gives a slightly smaller photoelectric cross section than Scofield/Elam in this band. The worst value over $8$–$20$ keV for C, Al, Si is $-5.16\%$, inside the ledger's stated $6\%$ cross-compilation bound.

**Is the mixture defensible?** Yes, with the error disclosed. Three reasons.

1. The two compilations partition the *same* physical total into the *same* disjoint channels. Taking $\sigma_{\rm photo}$ from one and $\sigma_{\rm coh}+\sigma_{\rm incoh}$ from the other is neither a double count nor a gap; the mixed sum is a legitimate estimator of $\sigma_{\rm tot}$ whose error is the inter-compilation spread on the terms taken.
2. That spread is $2$–$6\%$ over $5$–$20$ keV for light elements, comparable to the spread quoted between photoabsorption compilations generally, and roughly an order of magnitude below the $109\%$ bias the scattering term removes for graphite at 20 keV. A few-percent-error model strictly dominates a factor-two-error model.
3. The mixture is forced by the architecture, correctly. $f_2$ is shared with $\beta$, $\delta$, $\chi_0$, the in-medium wavevector and the grazing reflectivity; all must use one photoabsorption normalisation. Switching only the transmission path to Elam photoabsorption would break the exact $\mu_{\rm photo}=2k\beta$ identity that `absorption-length` rests on, and Elam supplies no $f_1$/$f'$ at all, so switching *every* consumer is not available either.

The only thing the mixture must not do is claim XCOM-grade accuracy silently; the ledger and docstring both record the figure, which is the correct disclosure. One refinement: the residual reaches $-6\%$ to $-9\%$ at 1 keV for Be/C/Al/Si, i.e. worse than the "$2$–$6\%$ at $8$–$20$ keV" the ledger states once the soft-X-ray region is included.

## Comparison with the implementation

Read after the above was fixed.

### `materials/atomic.py::elam_scattering_cross_section_ang2`

```text
Eflat = np.clip(np.atleast_1d(E).ravel(), ELAM_E_MIN_EV, ELAM_E_MAX_EV)
mu_rho = np.asarray(xraydb.mu_elam(element, Eflat, "coh"), dtype=float) + np.asarray(
    xraydb.mu_elam(element, Eflat, "incoh"), dtype=float
)
# cm^2/g -> cm^2/atom -> Angstrom^2/atom.
sigma = mu_rho * (_atomic_mass_g_per_mol(element) / _AVOGADRO) * 1.0e16
```

with `ELAM_E_MIN_EV = 100.0`, `ELAM_E_MAX_EV = 8.0e5`, `from scipy.constants import Avogadro as _AVOGADRO`, and `_atomic_mass_g_per_mol` a thin `float(xraydb.atomic_mass(element))`.

Term by term this is the boxed $\sigma_{\rm scat}=[(\mu/\rho)_{\rm coh}+(\mu/\rho)_{\rm incoh}]\,A/N_A\times10^{16}$: same two channels, multiply (not divide) by $A/N_A$, $+16$ (not $-16$) in the exponent, same CODATA $N_A$. No extra $\lambda$, $2\pi$, or field-to-intensity factor — correctly, since $\sigma_{\rm scat}$ is a cross section, not an amplitude.

### `materials/crystal.py::scattering_attenuation_inv_ang`

```text
sigma = elam_scattering_cross_section_ang2(element, photon_E_eV)
return number_density_per_ang3 * sigma
```

Exactly $\mu_{\rm scat}=n_a\sigma_{\rm scat}$, shape-following `photon_E_eV`.

### `materials/attenuation.py::_mu_total_inv_ang`

```text
E_cpu = _to_cpu(E_eV)
mu = 0.0
for el, n_i in comp:
    mu = mu + 1.0 / absorption_length_ang(el, E_cpu, n_i)
    mu = mu + scattering_attenuation_inv_ang(el, E_cpu, n_i)
if is_device_array(E_eV):
    return xp.asarray(mu, dtype=REAL)
return mu
```

which is $\sum_i\left[1/L_{{\rm abs},i}+n_i\sigma_{{\rm scat},i}\right]$, the mixture form derived above, with the device handling orthogonal to the physics.

### Numeric agreement

| quantity | worst relative difference | attribution |
| --- | --- | --- |
| $\sigma_{\rm scat}$, {C, Al, Si, Cu, W} $\times$ {1, 8, 20, 60} keV | $2.2\times10^{-16}$ | float rounding; identical formula and constants |
| $\mu_{\rm scat}$, C and Al | $2.2\times10^{-16}$ | as above |
| $\mu_{\rm tot}$, graphite at 8/20/60 keV | $9.9\times10^{-9}$ | PyRITE's truncated `HC_EV_ANG` and `R_E_ANG` literals in the photoabsorption term only |
| $\mu_{\rm tot}$, two-species mixture at 5/10/30 keV | $1.1\times10^{-8}$ | as above |

The $10^{-8}$ residual is the same one already attributed in the `absorption-length` ledger row to the truncated $hc$ and $r_e$ literals ($3.5\times10^{-9}$ and $9.3\times10^{-9}$); it is inherited, not introduced here. The scattering term itself agrees to machine precision.

**Sign/monotonicity sweep.** Over 400 logarithmically spaced energies from 126 eV to 794 keV, for Be, C, Al, Si, Cu and W, $\mu_{\rm tot}-\mu_{\rm photo}>0$ at every point with zero violations; the minimum margin is $1.2\times10^{-9}\,\mathring{\mathrm A}^{-1}$ (Be). Limiting case (2) holds in the implementation, not just in the derivation.

**Verdict on the expression:** `matches`. No divergent factor, sign, exponent, unit, or convention.

## Findings on the four scoping questions

### 1. The unit chain

Verified independently at both ends of the band by closed form ($Z^2\sigma_{\rm T}$ at 100 eV, Klein–Nishina at 500 keV) and at one interior point by an independent form-factor quadrature. Correct as implemented; see the tables above.

### 2. Mixing Chantler photoabsorption with Elam scattering

Defensible, for the three reasons given above, with a measured $2$–$6\%$ penalty at $8$–$20$ keV against an order-of-magnitude bias removed. The only correction to the recorded scope is that the penalty grows to $6$–$9\%$ at 1 keV for the light elements.

### 3. The $[100\ \mathrm{eV},800\ \mathrm{keV}]$ clamp

> [!note]
> **Acted on, same day.** This section reviewed a build in which both table
> edges were clamped. Finding 3 below was accepted: `ELAM_E_MAX_EV` now
> returns NaN instead of a frozen value, so the high edge falls under the same
> out-of-domain policy as the Chantler table, and the low-edge clamp is kept
> for the reasons this section gives. The analysis is left as written; read
> the high-edge discussion as the justification for that change rather than as
> a description of current behaviour. Pinned by
> `tests/materials/test_attenuation.py::test_scattering_clamps_below_the_table_but_refuses_above_it`
> and `::test_no_packaged_brem_grid_reaches_the_elam_ceiling`.

**The clamp changes no returned value.** `xraydb.mu_elam` already saturates outside the Elam band, returning the endpoint value together with a warning. Carbon, $(\mu/\rho)_{\rm coh}+(\mu/\rho)_{\rm incoh}$ in $\mathrm{cm^2\,g^{-1}}$:

| $E$ | 5 eV | 50 eV | 99 eV | 100 eV | 101 eV |
| --- | --- | --- | --- | --- | --- |
| value | $1.197686$ | $1.197686$ | $1.197686$ | $1.197686$ | $1.197644$ |

| $E$ | 800 keV | 810 keV | 1 MeV | 10 MeV |
| --- | --- | --- | --- | --- |
| value | $0.0707563$ | $0.0707563$ | $0.0707563$ | $0.0707563$ |

So `np.clip` is numerically a no-op relative to the library call it wraps: it suppresses xraydb's out-of-range warning and makes the saturation an intentional, documented choice. It cannot introduce an error that an unclamped call would have avoided.

**Low edge — safe, and exercised in production.** Catalog `E_grid_brem` rows start at $0$, $50$ or $75$ eV, so the clamp is reached on real grids. The clamped $\sigma_{\rm scat}$ is a constant extrapolation of a term that is negligible there; $\sigma_{\rm scat}/(\sigma_{\rm photo}+\sigma_{\rm scat})$ with $\sigma_{\rm scat}$ frozen at its 100 eV value:

| element | 25 eV | 50 eV | 75 eV | 100 eV |
| --- | --- | --- | --- | --- |
| C | $5.8\times10^{-6}$ | $1.5\times10^{-5}$ | $3.0\times10^{-5}$ | $5.3\times10^{-5}$ |
| Al | $5.9\times10^{-4}$ | $4.2\times10^{-4}$ | $2.9\times10^{-5}$ | $2.1\times10^{-5}$ |
| Si | $3.3\times10^{-4}$ | $3.0\times10^{-4}$ | $3.8\times10^{-4}$ | $2.6\times10^{-5}$ |
| Mo | $8.1\times10^{-5}$ | $1.6\times10^{-4}$ | $3.6\times10^{-4}$ | $5.8\times10^{-4}$ |
| W | $2.3\times10^{-4}$ | $3.9\times10^{-4}$ | $9.0\times10^{-4}$ | $8.2\times10^{-4}$ |
| Se | $3.0\times10^{-2}$ | $4.7\times10^{-2}$ | $2.7\times10^{-4}$ | $1.3\times10^{-4}$ |

Even a $100\%$ error in the clamped value perturbs $\mu_{\rm tot}$ by less than these fractions, so the finite clamp is both harmless and strictly better than a NaN that would poison an otherwise well-tabulated soft-X-ray coefficient. Two qualifications on the docstring's "$\sim5\times10^{-5}$ of the total cross section": that figure is a low-$Z$ statement, and there are elements where it is three orders of magnitude larger — Se reaches $4.7\%$ at 50 eV, where its photoabsorption sits in a deep inter-shell minimum. Even there the error is unobservable, because at $50$ eV $\mu_{\rm tot}\gtrsim10^{-3}\,\mathring{\mathrm A}^{-1}$ gives an attenuation length of order $10^{3}\,\mathring{\mathrm A}$: transmission through any real filter or crystal is numerically zero either way. The clamp is safe on the low edge for the observable, not merely for the coefficient.

**High edge — the real exposure, currently out of reach.** Above 800 keV the situation inverts: $\mu_{\rm scat}$ is essentially all of $\mu_{\rm tot}$ (already $99.3\%$ for carbon at 100 keV) while the true Compton cross section keeps falling. Freezing $\sigma_{\rm scat}$ *over*-estimates $\mu_{\rm tot}$ by the Klein–Nishina ratio $\sigma_{\rm KN}(800\ \mathrm{keV})/\sigma_{\rm KN}(E)$:

| $E$ | 1 MeV | 2 MeV | 5 MeV | 10 MeV |
| --- | --- | --- | --- | --- |
| over-estimate factor | $1.11$ | $1.61$ | $2.83$ | $4.61$ |

and above 1.022 MeV pair production is additionally omitted, so the model is then wrong in both directions. A second, larger discontinuity sits at the Chantler ceiling: `henke_dispersion` NaNs above $966\,266.74$ eV, so $\mu_{\rm tot}$ becomes NaN there and the consumers' `nan_to_num` turns it into $\mu=0$, i.e. *perfect transmission*. That failure is silent and opposite-signed to the clamp bias.

**Can it corrupt a production grid? No, with margin.** The widest `E_grid_brem` row in `src/pyrite/data/materials.toml` stops at $262\,400$ eV (`wse2`); the next widest is $140\,700$ eV (`diamond`); the profile default is $30\,000$ eV. The physical ceiling is tighter still: the largest catalog `energy_keV` is $300$ keV of electron kinetic energy, and the bremsstrahlung endpoint cannot exceed it. So production photon energies stay below $\sim300$ keV, a factor $2.7$ under `ELAM_E_MAX_EV`, and the high-edge clamp is never exercised. The exposure is conditional on a user overriding `--brem-grid-stop` past 800 keV *and* running a beam above 800 keV; nothing in the code refuses that combination. Converting the top of the band from a silent saturation into an explicit error (or an explicit NaN, matching the photoabsorption path's policy) would close it; the low edge should keep the clamp.

### 4. Narrow-beam scoping for the two consumer families

The build-up question is quantitative, so it was measured rather than asserted. Fraction of the coherent cross section falling inside an acceptance half-angle $\theta_{\rm acc}$, from the same independent form-factor quadrature, together with the coherent share of the scattering total:

| case | $f(<1^\circ)$ | $f(<2^\circ)$ | $f(<5^\circ)$ | $f(<10^\circ)$ | $\sigma_{\rm coh}/\sigma_{\rm scat}$ |
| --- | --- | --- | --- | --- | --- |
| C, 8 keV | $0.0007$ | $0.0026$ | $0.0161$ | $0.0614$ | $0.626$ |
| C, 20 keV | $0.0021$ | $0.0083$ | $0.0486$ | $0.1560$ | $0.289$ |
| C, 60 keV | $0.0135$ | $0.0497$ | $0.1967$ | $0.3651$ | $0.058$ |
| Al, 8 keV | $0.0004$ | $0.0016$ | $0.0097$ | $0.0371$ | $0.886$ |
| Al, 20 keV | $0.0014$ | $0.0055$ | $0.0321$ | $0.1066$ | $0.599$ |
| Al, 60 keV | $0.0082$ | $0.0301$ | $0.1354$ | $0.3501$ | $0.186$ |

Rayleigh scattering at these energies is far *less* forward-peaked than the "small-angle coherent scatter" caveat suggests: the median coherent scattering angle for carbon at 20 keV is $32^\circ$, and at 60 keV still $15^\circ$.

**Filter plates.** The narrow-beam form is the standard model for a plate in a collimated line. The over-attenuation from coherent flux that stays inside the collection cone is bounded by $f(<\theta_{\rm acc})\times(\sigma_{\rm coh}/\sigma_{\rm scat})\times(\mu_{\rm scat}/\mu_{\rm tot})$. For carbon at 20 keV with a generous $5^\circ$ acceptance that is $0.0486\times0.289\times0.521=0.73\%$ of the exponent; at $10^\circ$ it is $2.3\%$. Incoherent scattering at these energies is close to isotropic and contributes negligibly to build-up. So the assumption is correctly scoped and the residual is one-sided (over-attenuating) and sub-percent for realistic geometry — materially better than the ledger's unquantified hedge implies.

**Crystal-source self-absorption and the escape factors.** Two distinct points:

- *Build-up is weaker here, not stronger.* An escape factor asks for the probability that a photon born at depth $z$ reaches the surface along a specified direction toward a distant detector. Any scattering event randomises the direction, and with a small solid angle subtended at the emission point essentially none of the scattered flux returns to the collected ray. The lever arm is longer than for a filter plate, so $\mu_{\rm tot}$ is *more* defensible for escape than for a near-detector filter.
- *Bragg coherent removal is the genuine gap.* The independent-atom $\sigma_{\rm coh}$ is an orientation-average over a free atom. In an oriented single crystal at or near a reflection condition, coherent removal from a given ray is not that average: it concentrates into the Bragg directions and can exceed the isotropic value by orders of magnitude for a ray satisfying the condition, or fall below it (anomalous transmission). PyRITE's crystal source is by construction at a reflection condition for the emitted line, so this is not hypothetical. The isotropic $\sigma_{\rm coh}$ should be read as a smooth background estimate, with the Bragg term explicitly unmodelled. Unlike the build-up bias, the sign of this residual is not known a priori.

Conclusion: correctly scoped for both families, with a quantified one-sided sub-percent bound for filters and a two-sided, orientation-dependent open condition for the crystal source.

## Summary

The derived expression and the implementation agree exactly. The three items below are scope statements about where the expression may be evaluated, not defects in it, and are offered as the remaining conditions before human sign-off:

1. `ELAM_E_MAX_EV` saturation is unguarded. It is unreachable from the current catalog (300 keV beams, 262 keV widest grid) but a user override can reach it, and beyond it the coefficient is biased high by up to $4.6\times$ while pair production is missing; above $966$ keV the total NaNs to zero attenuation instead.
2. The mixed-compilation residual is $-6\%$ to $-9\%$ at 1 keV for Be and C, outside the "$2$–$6\%$ at $8$–$20$ keV" band the ledger records.
3. Bragg coherent removal in the oriented single crystal is unmodelled and its sign is not bounded. The isotropic build-up caveat, by contrast, is now quantified at $<1\%$ of the exponent for a $5^\circ$ acceptance.
