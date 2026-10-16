# `channeling-modulated-incoherent-emission`: independent re-derivation

Fresh-context verification (2026-10-15) of the research-prototype claim (#501). This context did not write the check or the research note. Sections 1–3 were written from the ledger row and the representation claims of `cr-density-matrix-split-operator` and `cr-dechanneling-collapse` only, before reading `checks/channeling_modulated_emission.py` or the research note.

## 1. Quantity and representation

Position-local incoherent emission (nuclear bremsstrahlung, large-$q$ inner-shell ionization) is modelled as a rate proportional to the local target density $n(x)$ at the electron's transverse position. Relative to a uniform (random-orientation) density $\bar n$, the modulation factor is

$$
F(z)=\frac{\int_{\rm cell}\rho(x,x;z)\,n(x)\,dx}{\bar n\int_{\rm cell}\rho(x,x;z)\,dx}.
$$

Representation (from `cr-density-matrix-split-operator`): $\rho$ is block-diagonal in Bloch $\kappa_r=r g/n_{\rm cells}$, $g=2\pi/d$. In block $r$ the plane-wave basis is

$$
\langle x\vert r,h\rangle=\frac{e^{i(\kappa_r+hg)x}}{\sqrt d},\qquad \lvert h\rvert\le H,
$$

and $\rho_r$ is stored in the eigenbasis of $H_\perp(\kappa_r)$, with $V_r[h,n]=\langle r,h\vert r,n\rangle$. The plane-wave matrix is $P_r=V_r\rho_rV_r^\dagger$, so $\operatorname{Tr}P_r=\operatorname{Tr}\rho_r$.

## 2. Cell integral and index convention

Write $n(x)=\sum_m n_m e^{imgx}$, $n_m=d^{-1}\int_{\rm cell}n(x)e^{-imgx}dx$, $n_{-m}=n_m^*$ since $n$ is real, and $n_0=\bar n$. For a pure state $\psi(x)=\sum_h c_h e^{i(\kappa+hg)x}/\sqrt d$ with $P_{hh'}=c_hc_{h'}^*$,

$$
\lvert\psi(x)\rvert^2=\frac1d\sum_{hh'}P_{hh'}e^{i(h-h')gx}
$$

(the $\kappa$ phase cancels on the diagonal, so every block's density is $d$-periodic). Then

$$
\int_0^d\lvert\psi\rvert^2 n\,dx=\sum_{hh'}P_{hh'}\sum_m n_m\frac1d\int_0^d e^{i(h-h'+m)gx}dx=\sum_{hh'}P_{hh'}\,n_{h'-h}.
$$

With $T[h,h']=n_{h-h'}$, $n_{h'-h}=T[h',h]$ and

$$
\boxed{\int_{\rm cell}\rho(x,x)\,n(x)\,dx=\sum_r\sum_{hh'}P_{r,hh'}T[h',h]=\sum_r\operatorname{Tr}(T P_r)}.
$$

Hence $F=\sum_r\operatorname{Tr}(TP_r)/(n_0\operatorname{tr}\rho)$, matching the claim. The alternative $\operatorname{Tr}(T^{\mathsf T}P_r)=\sum P_{hh'}n_{h-h'}$ evaluates the density mirrored, $n(-x)$. Since $T$ is Hermitian, $T^{\mathsf T}=T^*$, so the two agree only when every $n_m$ is real, i.e. $n$ is even about the origin. For Si(110) with a plane at $x=0$, $T$ is real symmetric and they coincide for every state, including off-centre combs. A convention error is therefore invisible to any Si(110) test; only a density with complex harmonics discriminates (Section 6).

$T$ is the Toeplitz matrix of multiplication by $n(x)$ projected on the band. It is positive semidefinite with $\|T\|\le n_{\max}$, since $\langle c\vert T\vert c\rangle=\int n\lvert\psi\rvert^2\le n_{\max}\int\lvert\psi\rvert^2$. Ratios do not depend on the per-block normalisation, provided numerator and denominator use the same one: translation averaging removes cross-block coherences, and each block's density is $d$-periodic.

## 3. Cheap filters

| filter | result |
| --- | --- |
| units | $T$ and $n_0$ are both densities, so $F$ is dimensionless; $Y$ is a $z$-average of a dimensionless quantity. Pass. |
| uniform $n$ | $n_m=\bar n\delta_{m0}$, so $T=\bar nI$ and $F=\operatorname{Tr}P/\operatorname{tr}\rho=1$. Pass. |
| plane wave | $P=e_{h_0}e_{h_0}^\dagger$ gives $\operatorname{Tr}(TP)=T[h_0,h_0]=n_0$, so $F=1$. Pass (cutoff correction in Section 5). |
| perfect focus | $F=\int n\lvert\psi\rvert^2/(\bar n\int\lvert\psi\rvert^2)\le n_{\max}/\bar n$, with equality for $\lvert\psi\rvert^2\to\delta$ at the maximum. Pass. |
| Gaussian $n$ | For one thermally smeared plane per $d$, $n\approx\bar n\,d\,(2\pi u^2)^{-1/2}e^{-x^2/2u^2}$, so $n_{\max}/\bar n=d/(\sqrt{2\pi}u)$. For $d=1.920$ Å and the stated 9.81, $u=0.078$ Å, consistent with the Si 293 K one-dimensional rms amplitude. Plausible. |
| $Y(L)$ | Normalised yield relative to random, $Y=L^{-1}\int_0^L[\operatorname{tr}\rho\,F+(1-\operatorname{tr}\rho)]dz$: the in-subspace population weighted by $F$, plus the lost population (free, overflow, entry flux) at $F=1$. With $F\equiv1$, $Y=1$; with no loss, $Y=\langle F\rangle_z$. Consistent. Assumption: dechanneled electrons see the random density. In reality just-above-barrier electrons are still modulated (shoulders), so this is a modelling choice to flag, not an identity. |

## 4. Lattice periodicity of a registered state

A transverse state with $\psi(x+d)=\psi(x)$ has Fourier content only at $k=hg$, i.e. only in block $\kappa=0$ ($r=0$). Lattice translation $x\to x+md$ multiplies block $r$ by $e^{i\kappa_r md}$. Translation averaging over $m$ is what makes $\rho$ block-diagonal; a $\kappa=0$ pure state picks up no phase and stays the pure $\lvert\psi\rangle\langle\psi\rvert$. Fixed registration means the offset $c$ relative to the atomic planes is fixed. Averaging over continuous sub-cell offsets $c\in[0,d)$ would instead dephase the $h\neq h'$ coherences and drive $F$ to 1, so the claim depends on that registration premise.

**Gaussian comb.** For $\psi(x)=\sum_m\exp[-(x-c-md)^2/(4\sigma^2)]$ (density teeth of rms width $\sigma$), the cell coefficients are

$$
c_h=\frac1{\sqrt d}\int_0^d\psi(x)e^{-ihgx}dx=\frac1{\sqrt d}\int_{-\infty}^{\infty}e^{-(x-c)^2/4\sigma^2}e^{-ihgx}dx=\sqrt{\frac{4\pi\sigma^2}{d}}\,e^{-ihgc}\,e^{-\sigma^2g^2h^2}.
$$

The normalised amplitudes are therefore $c_h\propto e^{-ihgc-\sigma^2g^2h^2}$. Band truncation error is $\sum_{\lvert h\rvert>H}e^{-2\sigma^2g^2h^2}$; for $\sigma=0.15$ Å and $g=3.27$ Å$^{-1}$ this is negligible ($<10^{-30}$) for $H\ge 8$. For Gaussian $n$ of width $u$ and $\sigma\ll d$, $F(0)\approx d/(\sqrt{2\pi}\sqrt{u^2+\sigma^2})$. For $u=0.078$ Å and $\sigma=0.15$ Å, $F(0)\approx4.53$, consistent with the reported ≈ 4.5.

## 5. Cutoff correction to the aligned plane wave

Decompose the entry plane wave as $\psi=\phi_S+\phi_F$, with $\|\phi_F\|^2=w$ the eigen-weight above the cutoff. `cr-dechanneling-collapse` drops $\phi_F$ and the $S$–$F$ coherences, so the retained state is $\phi_S\phi_S^\dagger$ with trace $1-w$. Let $A=T-n_0I$; then $\langle\psi\vert A\vert\psi\rangle=0$ and $\|A\|\le a:=\max(n_{\max}-\bar n,\bar n-n_{\min})<n_{\max}$. Then

$$
F(0)-1=\frac{\langle\phi_S\vert A\vert\phi_S\rangle}{\bar n(1-w)}=-\frac{2\operatorname{Re}\langle\phi_S\vert A\vert\phi_F\rangle+\langle\phi_F\vert A\vert\phi_F\rangle}{\bar n(1-w)},
$$

$$
\lvert F(0)-1\rvert\le\frac{a}{\bar n}\,\frac{2\sqrt{w(1-w)}+w}{1-w}=2\frac{a}{\bar n}\sqrt w+\frac{a}{\bar n}w+O(w^{3/2}).
$$

The ledger's $2(n_{\max}/\bar n)\sqrt w$ is the leading term, with $n_{\max}$ in place of the smaller $a$. It is a strict bound whenever $2(n_{\max}-a)\sqrt w\gtrsim a\,w$, i.e. for $w\lesssim[2(n_{\max}-a)/a]^2$. For Si(110), $n_{\min}\approx0$ and $a=n_{\max}-\bar n$; with $n_{\max}/\bar n=9.81$, $(n_{\max}-a)/a=1/8.81$, so it is strict for $w\lesssim0.05$. A check that uses $2(n_{\max}/\bar n)\sqrt w$ as its tolerance is valid in that regime. The exact form above is the safe statement.

## 6. Comparison with the check (read after Sections 1–5)

`checks/channeling_modulated_emission.py` and research note §5 were read only after the sections above were written.

| item | independent | check / note | result |
| --- | --- | --- | --- |
| Toeplitz | $T[h,h']=n_{h-h'}$, $n_{-m}=n_m^*$ | `_toeplitz`: `lag = orders[:, None] - orders[None, :]`, conjugate for negative lag | match |
| harmonic convention | $n(x)=\sum_m n_m e^{imgx}$ | `_synthesize`: $n_0+2\operatorname{Re}\sum_{m\ge1}n_m e^{imgx}$ | match |
| contraction | $\sum_r\operatorname{Tr}(TP_r)$ | `einsum("hk,rkh->", toeplitz, plane)` $=\sum_r\sum_{hk}T_{hk}P_{r,kh}$ | match (not $T^{\mathsf T}$) |
| plane-wave matrix | $P_r=V_r\rho_rV_r^\dagger$ | `vectors @ rho @ conj(swapaxes(vectors))` | match |
| normalisation | $n_0\operatorname{tr}\rho$ | `coefficients[0].real * trace` | match |
| comb amplitudes | $c_h\propto e^{-ihgc-\sigma^2g^2h^2}$, block $r=0$ | `exp(-sigma**2 k**2 - 1j k centre)`, `plane[0]`; solver `kappa = arange(n_cells) g / n_cells`, so block 0 is $\kappa=0$ | match |
| entry projection | $\rho_{\rm eig}=V^\dagger PV$, drop $F$ rows/columns, entry weight $=$ dropped diagonal | `eigen = V^dagger plane V`, `keep` mask, entry = dropped diagonal | match |
| $Y(L)$ | $L^{-1}\int[\operatorname{tr}\rho F+1-\operatorname{tr}\rho]dz$ | `trapezoid(traces*factors + 1 - traces)/depths[-1]` at 100 Å samples | match |
| plane-wave gate | exact: $(a/\bar n)(2\sqrt{w(1-w)}+w)/(1-w)$ | `2 peak sqrt(w)` with $w$ = free + outside entry flux | leading-order form; strict for $w\lesssim0.05$, observed $w\le1.4\times10^{-6}$ |

No symbolic divergence was found.

## 7. Numerics

**Check run** (CPU, once): `PASS`, 2 min 48 s wall on this machine (the note says 1.5 min). Every value in the note's §5 tables and the ledger Notes reproduces to the printed digits. Highlights:

| $T$ | plane $F(0)-1$ | entry $w$ | gate $2(n_{\max}/\bar n)\sqrt w$ | comb converter vs quadrature |
| --- | --- | --- | --- | --- |
| 0.3 MeV | $2.4\times10^{-3}$ | $1.4\times10^{-6}$ | $2.3\times10^{-2}$ | 4.4845 vs 4.5295 ($w=1.6\times10^{-4}$; gate skipped) |
| 3 MeV | $2.4\times10^{-5}$ | $4.3\times10^{-9}$ | $1.3\times10^{-3}$ | 4.52981 vs 4.52951 ($w=8.2\times10^{-8}$; gate skipped) |
| 17 MeV | $<10^{-6}$ | 0 | 0 | 4.529511 vs 4.529511 (gated) |

The converter-vs-quadrature gate fires only at 17 MeV, where nothing is lost at entry, as the ledger states. At 0.3 and 3 MeV the deviations lie within the Section 5 bound for the dropped weight.

**Independent spot check** (scratch script, not committed). It uses my own Toeplitz loop over $(h,h')$, my own synthesis of $n(x)$ from `nuclear_density`, and a position-space comb over $\lvert m\rvert\le6$ on a 20 000-point grid. Si(110), 293.15 K, 17 MeV, $H=16$, 16 cells, $\sigma=0.15$ Å:

| comb centre $c$ [Å] | own $\operatorname{Tr}(TP)$ | direct quadrature | check converter |
| --- | --- | --- | --- |
| 0 (on plane) | 4.52951117 | 4.52951117 | 4.52951117 |
| 0.4 | 0.27622038 | 0.27622038 | 0.27622038 |
| 0.7 | 0.00086247 | 0.00086247 | 0.00086247 |
| $d/2$ | $9.2\times10^{-7}$ | $9.2\times10^{-7}$ | $9.2\times10^{-7}$ |

The Si(110) harmonics are real to $6\times10^{-17}$ and the plane sits at $x=0$, so this test cannot tell $T$ from $T^{\mathsf T}$. To discriminate, I used a synthetic density with $n_1\to n_1+0.3i\lvert n_1\rvert$ (asymmetric) at $c=0.4$ Å:

| | own $\operatorname{Tr}(TP)$ | own $\operatorname{Tr}(T^{\mathsf T}P)$ | direct | check |
| --- | --- | --- | --- | --- |
| asymmetric $n$ | −0.22104078 | 0.77348155 | −0.22104078 | −0.22104078 |

(The synthetic density goes negative; it serves only as a convention probe.) The check's contraction is the correct $\operatorname{Tr}(TP)$. Profile values: $n_{\max}/\bar n=9.8075$ (ledger 9.81), $n_{\min}/\bar n\approx-10^{-8}$, so $a=n_{\max}-\bar n$. The Gaussian estimate $d/(\sqrt{2\pi}\sqrt{u^2+\sigma^2})=4.53$ from Section 4 agrees with 4.5295.

The grid-convergence statement (3 MeV, $dz=50$ Å, $H=20$, 24 cells, ≤ 0.6 %) was not re-run, because it needs heavier runs than this verification's budget.

## 8. Adjudication of the note's conclusions and assumptions

- **Not promoted / no anchor.** Stated honestly. The factor is a correct expectation value, but without orientation-dependent yield data there is nothing to anchor. Taftø & Spence 1982 is marked *(verify)* as not in Zotero.
- **Genz 1996.** Used only for the qualitative statement that bremsstrahlung is enhanced under channeling, and the note says explicitly that it is not a comparison against data. Honest.
- **Wong 2024 bound.** For a paraxial planar state, $F\le n_{\max}/\bar n$ follows directly from $\|T\|\le n_{\max}$ (Section 2), so "bounded by about 10" in planar geometry is correct. The note correctly separates this from Wong's monolayer, non-paraxial, recoil-dominated enhancement, which the engine cannot represent. It claims only that the paraxial overlap part erodes in bulk. Axial (string) channeling would have a much larger $n_{\max}/\bar n$; "planar" is stated, so this is not an overreach.
- **Locality.** Flagged as the first assumption. Nuclear density as the bremsstrahlung proxy is reasonable for the screened-nucleus, large-momentum-transfer part. That is all the claim covers.
- **Inner-shell delocalization.** Flagged, with the correct scale: $\hbar v/E_K\approx1972\ {\rm eV\,Å}/1839\ {\rm eV}\approx1.07$ Å, against $d=1.92$ Å and $u=0.078$ Å. The ionization kernel is therefore wider than half a period, so a real Si K yield is strongly smeared. **Soft finding:** the note's statement that the K factor "lies between $F_{\rm nuc}$ and $F_{\rm el}$" is asserted, not derived. Convolving $n_{\rm nuc}$ with a ~1 Å kernel can flatten the modulation below $F_{\rm el}$, which is dominated by compact core electrons, toward 1. "Between $F_{\rm nuc}$ and 1, set by the impact-parameter kernel" is the defensible statement. This does not affect the ledger claim, which is restricted to large-$q$ (local) ionization.
- **Dechanneled population at $F=1$.** This is a modelling choice of $Y(L)$, not an identity. It is mild here: states between the barrier and the 300 eV margin stay in $S$ and carry their own $F$, so only population far above the barrier, or outside the band, is assigned the random value. The note states the choice but does not list it among its assumptions. Suggest adding it.
- **Registration.** The $\kappa=0$, fixed-registration premise (Section 4) is stated and carried into interface requirement 1. Averaging over sub-cell registration would erase the comb modulation; the note keeps that case separate ("registration-averaged finite probes").

Minor bookkeeping: the ledger and note say the converter matches quadrature to $10^{-7}$, while the script gates at $10^{-6}$ (observed agreement is better than $10^{-8}$). The runtime is 2.8 min here, not 1.5 min.

## Verdict

`rederived`. The cell-integral formula, Toeplitz index convention, $\operatorname{Tr}(TP)$ contraction, comb Fourier amplitudes, $\kappa=0$ block assignment and $Y(L)$ bookkeeping match the independent derivation symbolically and numerically. The plane-wave cutoff tolerance is the leading-order form of the exact bound in Section 5 and is valid in the regime exercised. The claim stays unanchored, as the note states. Human sign-off is separate.
