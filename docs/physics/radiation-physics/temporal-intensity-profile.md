# Temporal intensity profile

A profile can opt in to the temporal intensity profile $I(t)=|E(t)|^2$ of the PXR/CBS line field arriving in each observation direction. It is computed beside the line spectrum, from the same per-segment line data and under the same emission policy, and stored as `temporal_t_fs` with `temporal_intensity` (and `temporal_intensity_coherent` under `emission = coherent | both`). The profile is the field at the detector *position*, upstream of any detector response: no modelled detector resolves time, and the energy-resolution convolution discards phase.

```toml
[profiles.my_profile]
temporal_profile = true   # absent (default) computes no profile
```

or `pyrite profile set my_profile --temporal-profile`. The option is divergence-only: off, the dataset identity, case payloads, and spectra are unchanged.

## Transform and units

Each segment's spectral field is the [coherent-emission](coherent-emission.md) field

$$
E_j(E)=c_j\,F_j(E)\,\exp\!\bigl\{i\bigl[\omega(E)\,d_j-\mathbf g\cdot\mathbf r_j-\delta(E)\,\omega(E)\,L_{{\rm esc},j}\bigr]\bigr\},
\qquad d_j=t_{{\rm abs},j}-\hat{\mathbf n}\cdot\mathbf r_j ,
$$

with $c_j$ the amplitude frozen at the segment's resonance, $F_j$ its finite-time formation factor ($t_L\,{\rm sinc}$ without absorption), $d_j$ the retarded midpoint arrival time and $\omega=E/\hbar c$, all in Å with $c=1$. The unitary transform

$$
E(t)=\frac{1}{\sqrt{2\pi\hbar c}}\int E(E)\,e^{-iEt/\hbar c}\,dE
\qquad\Longrightarrow\qquad
\int|E(t)|^2\,dt=\int|E(E)|^2\,dE
$$

makes $I(t)$ carry the spectrum's photons per sr per incident electron, now per unit arrival time: the stored intensity is photons sr⁻¹ electron⁻¹ fs⁻¹. With the field's $e^{+i\omega d}$ phase the kernel $e^{-i\omega t}$ places each segment's pulse at $t=d_j$. Polarizations, reflections, and mosaic orientations stay incoherent, so their intensities add, exactly as in the spectrum.

## Incoherent policy: Doppler-compressed boxes

Under the random-phase approximation the observable is $\sum_j|E_j(t)|^2$. For a constant amplitude, the formation factor is $Q(\omega)=\int_{-t_L/2}^{t_L/2}e^{iD(\omega-\omega_{\rm res})\tau}\,d\tau$, with $D=1-\operatorname{Re}n\,\mathbf v\cdot\hat{\mathbf n}$. Its inverse transform is

$$
\frac{1}{2\pi}\int Q(\omega)\,e^{i\omega(d-t)}\,d\omega=\frac{1}{D}\,e^{-i\omega_{\rm res}(t-d)}\quad\text{for }|t-d|\le Dt_L/2,
$$

and zero outside. Each line is therefore a rectangular pulse of duration $Dt_L=2\hbar c\,a$, where $a$ is the sinc width of [the lineshape](coherent-radiation.md#finite-interaction-time). The pulse is centred on its arrival time and carries the line's whole spectral mass $w\pi/a$. Parseval holds exactly: $\int|E|^2dt=|c|^2t_L/D=\frac{1}{2\pi}\int|cQ|^2d\omega$. The kernel splits each box's mass exactly over the time bins it overlaps.

Numerical substeps of one physical flight follow each other contiguously in arrival time: consecutive midpoints are separated by $D(t_k+t_{k+1})/2$, so their boxes tile the flight's box without overlap. Squaring per substep therefore equals squaring per flight, and no flight grouping is needed in the time domain.

## Coherent policy: band-limited inverse transform

Each (reflection, orientation, polarization) row's field is rebuilt on a uniform energy grid that spans the line axis, with spacing $\delta E$. One FFT then gives $E(t)$ at spacing $\Delta t=2\pi\hbar c/(n\,\delta E)$, and discrete Parseval holds exactly on that grid. The profile is band-limited to the line axis: features shorter than about $2\pi\hbar/E_{\rm span}$ are not resolved. The rebuilt field keeps the full complex formation factor: absorption damping, the escape-path refractive phase, and its group delay. The spectrum's `sinc_cutoff` window is not applied to it.

With bunch offsets present, the ensemble average is formed in the time domain. Write $S_e$ for electron $e$'s field with its realized offset and $S_e^{\rm geo}$ for the same field without it. Let $\chi(\omega)$ be the characteristic function of the offsets: empirical $\langle e^{i(\omega A_e-B_e)}\rangle$ for an infinite slab, or analytic $e^{-(\omega\sigma_z)^2/2}$ for a finite footprint. Then

$$
I(t)=\sum_e|S_e(t)|^2+\Bigl|\mathcal F^{-1}\bigl[\chi\textstyle\sum_eS^{\rm geo}_e\bigr]\Bigr|^2-\sum_e\bigl|\mathcal F^{-1}[\chi S^{\rm geo}_e]\bigr|^2 .
$$

Its Parseval image is the spectrum's blend $(1-F)\,\text{Grouped}+F\,\text{Flat}$ with $F=|\chi|^2$. In the time domain the self terms keep each electron's realized arrival offset, so a long bunch spreads the envelope by its arrival-time distribution. Cross terms can make $I(t)$ locally negative by a Monte Carlo amount, but its integral is the non-negative blended yield.

## Time grid

One grid serves every call of a case: all observation directions, all crystalline layers, and both policies. The period $2\pi\hbar c/\delta E$ is twice the span of arrival times over every segment pulse ($d\pm t_L$, since $D\le2$), and the signal is centred in it. Band-limited coherent ringing therefore does not wrap back onto the signal. The FFT length is capped at $2^{20}$ samples; a wider line axis or longer bunch raises an error instead of allocating unbounded memory.

## Cost

The option forces the per-reflection line route for every policy. The incoherent profile adds one $O(N_{\rm line})$ box deposit per row. The coherent profile rebuilds each row's field on the $n$-point grid, $O(N_{\rm seg}\,n)$ per row, plus one $O(n\log n)$ FFT per row; with bunch offsets it adds one FFT pair per electron. GPU device kernels and electron blocking are bypassed while the option is on.

## Assumptions and limits

- Amplitudes, the $\sqrt\omega$ prefactor and the couplings are frozen at each segment's resonance (narrow-band), as in the spectrum.
- Incoherent boxes carry the row's mean escape transmission (flat box) and neglect the dispersive group delay $\sim\delta L_{\rm esc}/c$ (attoseconds or less). The coherent route keeps both.
- The incoherent box carries each line's whole mass, including tails off the line axis; the coherent profile is band-limited to the axis. Their integrals therefore agree with the stored spectra up to the out-of-axis line mass.
- Inherits every assumption and open item of [coherent PXR/CBS radiation](coherent-radiation.md) and [coherent-emission tracking](coherent-emission.md).

## Validation

[Validation: `temporal-intensity-profile`](../../validation/radiation-physics/temporal-intensity-profile.md). Implementation owner: `pyrite.montecarlo.spectrum.lines._temporal`.
