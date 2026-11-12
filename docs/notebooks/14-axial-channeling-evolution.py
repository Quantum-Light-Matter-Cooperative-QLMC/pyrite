# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "endf-parserpy==0.17.0",
#     "marimo>=0.23.3",
#     "pyrite-mc==0.32.0",
#     "matplotlib>=3.11",
#     "numpy>=2",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=False)
def _():
    import os
    from dataclasses import replace

    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite as pr
    from pyrite.config import FixedSampling
    from pyrite.detectors import EnergyBins, PointDetector
    from pyrite.experimental.quantum import AxialDiagnosticConfig, axial_diagnostic
    from pyrite.quantum import gaussian, laguerre_gauss
    from pyrite.units import ureg

    return (
        AxialDiagnosticConfig,
        EnergyBins,
        FixedSampling,
        PointDetector,
        axial_diagnostic,
        gaussian,
        laguerre_gauss,
        mo,
        np,
        plt,
        pr,
        replace,
        ureg,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Axial channeling: crystal strings, vortices, and quantum jumps

    **What makes axial channeling different from planar channeling?** An electron
    evolves in a *two-dimensional* transverse crystal potential. It can circulate
    around an atomic string, carry orbital angular momentum (OAM), and scatter
    between transverse states. Those features are hidden by a Gaussian-only
    density animation.

    We'll investigate **Si[001], 5 MeV**, in three steps, then run one of the
    few axial cases the standard simulation admits:

    1. Locate the atomic string and compare Gaussian and $\ell=+1$ vortex entries.
    2. Follow wavefunction phase, probability current, and OAM *about that string*.
    3. Turn on a small, seeded Monte Carlo wavefunction (MCWF) ensemble and inspect
       survival, purity, and transverse-energy projections.
    4. Run the calibrated **Si⟨100⟩, 3.5 MeV, 0.3 µm** case through `pr.simulate`
       and compare its on-axis channeling source with bremsstrahlung.

    **Sections 1–3 are experimental research diagnostics, not calibrated source
    predictions.** Their small periodic grids demonstrate APIs and qualitative
    mechanisms. They do **not** validate dechanneling length, photon yield,
    linewidth, or OAM-photon production. The MCWF example has intentionally poor
    Monte Carlo statistics. Section 4 is different: standard runs admit exactly
    two calibrated silicon cases, each with a stored error budget, and refuse
    every other axial case before doing any work.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Put the beam on an atomic string

    The axial continuum potential keeps reciprocal vectors perpendicular to
    the chosen crystal axis. Its minima lie near the projected atomic strings.
    The FFT grid represents a **periodic supercell**, not an isolated crystal
    channel. We center the probe and measure OAM about **string 0 at (0, 0) Å**;
    this matters because OAM about a different axis is a different observable.

    The solver configuration is deliberately *numeric* (Å, eV, radians). Quantum
    entry-state factories accept quantities, which we explicitly lower at the
    diagnostic boundary.
    """)
    return


@app.cell
def _(AxialDiagnosticConfig, gaussian, laguerre_gauss, ureg):
    # Edit the waist, energy, and recorded depths for quick experiments.
    # Keep the OAM disk inside the supercell. Increasing cell count without
    # increasing grid shape coarsens the spatial resolution.
    waist = 0.60 * ureg.angstrom
    kinetic_energy = 5 * ureg.MeV

    config = AxialDiagnosticConfig(
        crystal="silicon",
        axis=(0, 0, 1),
        kinetic_energy_eV=kinetic_energy.to("eV").magnitude,
        cells=(2, 2),
        shape=(48, 48),
        depths_ang=tuple(
            (value * ureg.nm).to("angstrom").magnitude for value in (0, 10, 20, 35, 50)
        ),
        step_ang=(5 * ureg.angstrom).magnitude,
        mode="unitary",
        backend="cpu",
        oam_string=0,
        oam_radius_ang=1.0,  # Å; disk around the selected string
        record_current=True,
    )
    # Both pure states are centered on the string at the supercell origin.
    gaussian_entry = gaussian(waist_ang=waist, dimension=2, center_ang=(0.0, 0.0))
    vortex_entry = laguerre_gauss(1, waist_ang=waist, center_ang=(0.0, 0.0))
    return config, gaussian_entry, vortex_entry


@app.cell
def _(axial_diagnostic, config, gaussian_entry, replace, vortex_entry):
    vortex = axial_diagnostic(config, entry=vortex_entry)
    # The Gaussian reference changes both amplitude profile and phase structure.
    gaussian_result = axial_diagnostic(
        replace(config, oam_string=None, record_current=False), entry=gaussian_entry
    )
    return gaussian_result, vortex


@app.cell
def _(np, plt, vortex):
    # Shift the periodically repeated string at (0,0) to the image centre.
    _grid = vortex.grid
    _shift = (_grid.shape[0] // 2, _grid.shape[1] // 2)
    _dx, _dy = (_grid.extent_ang[i] / _grid.shape[i] for i in range(2))
    _extent = (
        -_grid.extent_ang[0] / 2 - _dx / 2,
        _grid.extent_ang[0] / 2 - _dx / 2,
        -_grid.extent_ang[1] / 2 - _dy / 2,
        _grid.extent_ang[1] / 2 - _dy / 2,
    )

    def _center(values):
        return np.roll(values, _shift, axis=(0, 1)).T

    _density_0 = vortex.density_per_ang2[0]
    _density_end = vortex.density_per_ang2[-1]
    _phase_end = np.ma.masked_where(
        _density_end < _density_end.max() * 1e-3,
        np.angle(vortex.wavefunctions_per_ang[-1]),
    )
    fig_geometry, _axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for _ax, _image, _title, _cmap, _label in (
        (_axes[0, 0], vortex.potential_eV, "Axial string potential", "viridis", "U (eV)"),
        (_axes[0, 1], _density_0, "Vortex entry density", "magma", "Å⁻²"),
        (
            _axes[1, 0],
            _density_end,
            f"Density after {vortex.depths_ang[-1] / 10:g} nm",
            "magma",
            "Å⁻²",
        ),
        (_axes[1, 1], _phase_end, "Final envelope phase (masked)", "twilight", "rad"),
    ):
        _image_artist = _ax.imshow(
            _center(_image),
            origin="lower",
            extent=_extent,
            cmap=_cmap,
            aspect="equal",
            **({"vmin": -np.pi, "vmax": np.pi} if _label == "rad" else {}),
        )
        _ax.set(title=_title, xlabel="x from string (Å)", ylabel="y from string (Å)")
        fig_geometry.colorbar(_image_artist, ax=_ax, label=_label, shrink=0.8)
    _axes[0, 1].add_patch(
        plt.Circle((0, 0), vortex.config.oam_radius_ang, fill=False, ls="--", color="white", lw=1)
    )
    fig_geometry  # noqa: B018 - displayed by marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dashed circle is the **OAM measurement disk**, not a hard boundary.
    Its weight can move in or out while full-grid norm remains one. A vortex
    may display phase winding near a density zero; phase is not meaningful in
    pixels where density is negligible. Note that these snapshots display the
    *same* finite periodic state after propagation, not an isolated string.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. OAM and circulation are different diagnostics

    The vortex entry has $\exp(i\ell\phi)$ winding with $\ell=+1$ about
    the selected string. The recorded $P_\ell$ are **per-incident-electron
    weights inside the disk** for $\ell=-8,\ldots,+8$; they need not add to
    one, and their redistribution is **not** photon emission. Compare these to
    the squared *initial-state overlap* and the full periodic-grid norm.
    """)
    return


@app.cell
def _(gaussian_result, np, plt, vortex):
    depth_nm = vortex.depths_ang / 10

    def _fidelity(_result):
        _waves = _result.wavefunctions_per_ang
        return (
            np.abs(np.sum(_waves[0].conj() * _waves, axis=(1, 2)) * _result.grid.cell_area_ang2)
            ** 2
        )

    fig_diagnostics, (ax_overlap, ax_oam) = plt.subplots(
        1, 2, figsize=(11, 3.7), layout="constrained"
    )
    ax_overlap.plot(depth_nm, _fidelity(gaussian_result), "o-", label="Gaussian overlap")
    ax_overlap.plot(depth_nm, _fidelity(vortex), "s-", label="Vortex overlap")
    ax_overlap.plot(depth_nm, vortex.surviving, "k--", label="Full-grid norm")
    ax_overlap.set(xlabel="Depth (nm)", ylabel="Probability / squared overlap", ylim=(0, 1.05))
    ax_overlap.legend(fontsize=8)

    for _ell in (-1, 0, 1):
        _index = int(np.flatnonzero(vortex.oam_ells == _ell)[0])
        ax_oam.plot(depth_nm, vortex.oam_populations[:, _index], "o-", label=rf"$P_{{{_ell:+d}}}$")
    ax_oam.plot(
        depth_nm, vortex.oam_populations.sum(axis=1), "k--", label="Disk weight (reported modes)"
    )
    ax_oam.set(xlabel="Depth (nm)", ylabel="OAM weight per incident electron", ylim=(0, None))
    ax_oam.legend(fontsize=8)
    fig_diagnostics  # noqa: B018 - displayed by marimo
    return


@app.cell
def _(np, plt, vortex):
    # Transverse probability current contains phase-gradient information that
    # is lost in a density plot. This is ELECTRON current, not an X-ray pattern.
    _jx, _jy = vortex.transverse_current_per_ang2[-1]
    _grid = vortex.grid
    _shift = (_grid.shape[0] // 2, _grid.shape[1] // 2)
    _jx_center = np.roll(_jx, _shift, (0, 1)).T
    _jy_center = np.roll(_jy, _shift, (0, 1)).T
    _x = (np.arange(_grid.shape[0]) - _shift[0]) * _grid.extent_ang[0] / _grid.shape[0]
    _y = (np.arange(_grid.shape[1]) - _shift[1]) * _grid.extent_ang[1] / _grid.shape[1]
    fig_flow, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    _magnitude = np.hypot(_jx_center, _jy_center)
    _image = ax.imshow(
        _magnitude,
        origin="lower",
        aspect="equal",
        extent=(
            _x[0] - (_x[1] - _x[0]) / 2,
            _x[-1] + (_x[1] - _x[0]) / 2,
            _y[0] - (_y[1] - _y[0]) / 2,
            _y[-1] + (_y[1] - _y[0]) / 2,
        ),
    )
    _stride = 4
    ax.quiver(
        _x[::_stride],
        _y[::_stride],
        _jx_center[::_stride, ::_stride],
        _jy_center[::_stride, ::_stride],
        color="white",
        pivot="mid",
    )
    ax.set(
        xlabel="x from string (Å)",
        ylabel="y from string (Å)",
        title=f"Vortex transverse current at {vortex.depths_ang[-1] / 10:g} nm",
    )
    fig_flow.colorbar(_image, ax=ax, label="|j⊥| (Å⁻²)")
    fig_flow  # noqa: B018 - displayed by marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation.** Overlap can decrease under perfectly unitary evolution.
    $P_{+1}$ can redistribute among modes or leave the chosen disk while
    norm remains constant. Current arrows measure transverse probability flow;
    neither $\ell$ nor the direction of those arrows directly specifies the
    angular momentum or helicity of a radiated photon.

    ## 3. Add thermal/electronic scattering with MCWF

    A *different* run now enables stochastic quantum jumps. For a quick CPU
    introduction we use a coarser **24×24** grid and **12 electrons** at fixed
    random seed. Its survival, empirical purity, and transverse-energy bins
    demonstrate the recording interface, **not statistically converged physics**.
    The energy-bin basis is important: electronic jumps can change the carrier
    sector, so fixed-state-index populations are not comparable across jumps.
    """)
    return


@app.cell
def _(axial_diagnostic, config, replace, vortex_entry):
    scattering_config = replace(
        config,
        mode="mcwf",
        shape=(24, 24),
        depths_ang=(0.0, 100.0, 300.0, 500.0),  # Å
        step_ang=10.0,
        n_electrons=12,
        seed=17,
        oam_string=None,  # keep the small ensemble inexpensive
        record_current=False,
        spectral_states=6,
        spectral_representation="energy_bins",
        spectral_energy_bins=8,
    )
    scattered = axial_diagnostic(scattering_config, entry=vortex_entry, record_metrics=True)
    return (scattered,)


@app.cell
def _(np, plt, scattered):
    z = scattered.depths_ang / 10
    fig_scattering, (ax_surv, ax_spectrum) = plt.subplots(
        1, 2, figsize=(11, 3.7), layout="constrained"
    )
    ax_surv.plot(z, scattered.surviving, "o-", label="Grid survival")
    ax_surv.plot(z, scattered.metrics.purity, "s-", label="Empirical ensemble purity")
    ax_surv.plot(z, scattered.numerical_sink, "^--", label="Numerical sink")
    ax_surv.set(xlabel="Depth (nm)", ylabel="Fraction / purity estimate", ylim=(0, None))
    ax_surv.legend(fontsize=8)
    for _index in np.argsort(scattered.spectral_populations[-1])[-3:][::-1]:
        ax_spectrum.plot(
            z,
            scattered.spectral_populations[:, _index],
            "o-",
            label=str(scattered.spectral_labels[_index]),
        )
    ax_spectrum.plot(z, scattered.spectral_omitted, "s--", label="Outside computed subspace")
    ax_spectrum.plot(z, scattered.spectral_represented, "k:", label="All represented bins")
    ax_spectrum.set(xlabel="Depth (nm)", ylabel="Weight per incident electron", ylim=(0, None))
    ax_spectrum.legend(fontsize=8)
    fig_scattering  # noqa: B018 - displayed by marimo
    return


@app.cell(hide_code=True)
def _(mo, scattered):
    mo.md(f"""
    **Seeded jump counts (12 trajectories):** `{scattered.jumps}`.

    The **numerical sink** is bookkeeping, not a measured dechanneling rate.
    The spectral remainder includes states outside the computed six-state
    subspace and is **not escape**. Purity here is a finite-ensemble diagnostic;
    The randomized purity estimate can fluctuate above one at low statistics.
    Compare independent seeds, more electrons, and smaller steps before making
    even a qualitative scattering claim. A mixed ensemble has no unique
    wavefunction phase, which is why we do not plot an MCWF phase map as if it
    were a single coherent wave.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Run an admitted axial case

    The diagnostics above never produce a photon spectrum. A standard run with
    `channeling=pr.quantum.Channeling("axial", axial=...)` does, but only for
    cases held in the
    repository's calibrated lookup. Two are admitted, both Si⟨100⟩ at 3.5 MeV
    through a 0.3 µm slab:

    | Key | Beam on the ⟨100⟩ axis | Depth budget ε (95%) | Omitted in-cut free→free (upper bound) |
    | --- | --- | --- | --- |
    | `si-fibre/A` | tilt 0.088°, azimuth 225° | 3.97% | 8.06% |
    | `si-fibre/B` | exactly on axis | 9.71% | 18.9% |

    ε bounds the on-axis yield integrated over **800 eV–10 keV**, not
    individual bins. The free→free block is *omitted* and declared as a
    one-sided bound, so the computed channeling yield can be low by up to that
    fraction. Admission also requires: a monoenergetic pencil electron beam,
    an unbounded slab cut normal to the axis, CPU transport, incoherent
    emission, and an on-axis, **zero-aperture** `PointDetector` with an
    explicit energy grid inside the band. The result is a source density in
    photons / electron / eV / sr, not detector counts.

    Every other case refuses before transport. To run one anyway, set
    `Channeling(..., admission="accept-inaccurate")`, accepting results that may
    be very inaccurate. The record is then marked `unadmitted` and carries no
    ε or free→free bound.

    Each line history costs a few CPU seconds (about 4 s here), so we use
    **32 channeling histories**, seed 7. That is enough to show the shape,
    not to resolve individual lines; the calibration itself used 20 000.
    """)
    return


@app.cell
def _(EnergyBins, FixedSampling, PointDetector, np, pr):
    photon_grid_eV = np.linspace(800.0, 10_000.0, 93)  # inside the admitted band

    def fibre_run(*, energy_keV=3500.0, histories=32):
        """Si<100> slab, exactly on axis (si-fibre/B when energy_keV=3500)."""
        return pr.simulate(
            pr.Beam(
                energy_keV=energy_keV,
                transverse=pr.beams.Spot(),  # pencil beam, no divergence
                longitudinal=pr.beams.PointBunch(),  # no energy spread
            ),
            pr.geometry.Slab(
                "silicon",
                thickness_ang=3000.0,  # 0.3 µm
                width=None,
                height=None,  # laterally unbounded
                cut=pr.geometry.CrystalCut(surface_hkl=(0, 0, 1)),
                tilt_deg=0.0,
                allow_normal_incidence=True,
            ),
            PointDetector(),  # on axis, zero aperture
            physics=pr.Physics(radiative_model="uncoupled"),
            channeling=pr.quantum.Channeling("axial", axial={"version": 2, "axis_uvw": [0, 0, 1]}),
            numerics=pr.Numerics(
                sampling=FixedSampling(cxr_histories=histories, brem_histories=histories),
                energy_bins=EnergyBins(cxr=photon_grid_eV, brem=photon_grid_eV),
                seed=7,
            ),
        )

    # A nearby case refuses before any transport: 5 MeV is outside the lookup.
    try:
        fibre_run(energy_keV=5000.0, histories=4)
    except ValueError as error:
        refusal = str(error)

    fibre = fibre_run()
    return fibre, refusal


@app.cell
def _(fibre, plt):
    fig_fibre, ax_fibre = plt.subplots(figsize=(8, 3.8), layout="constrained")
    for _component, _label in (
        (fibre.channeling, "Axial channeling source"),
        (fibre.brem, "Bremsstrahlung"),
    ):
        ax_fibre.plot(_component.photon_energy_eV / 1e3, _component.density, label=_label)
    ax_fibre.set(
        xlabel="Photon energy (keV)",
        ylabel="photons / electron / eV / sr",
        yscale="log",
        title="si-fibre/B on axis, 32 histories (statistically noisy)",
    )
    ax_fibre.legend(fontsize=8)
    fig_fibre  # noqa: B018 - displayed by marimo
    return


@app.cell(hide_code=True)
def _(fibre, mo, refusal):
    _record = fibre.case["channeling"]
    _yield = fibre.channeling.integrate().inside
    mo.md(f"""
    **Stored case record.** `{_record["case_key"]}` (marker
    `{_record["model_marker"]}`) carries its own evidence: depth budget
    ε = {100 * _record["depth_budget"]:.2f}%, free→free `{_record["free_free"]}`
    with upper bound {100 * _record["free_free_upper"]:.1f}%, and
    `{_record["evidence_record"]}`. These fields persist with results and
    checkpoints.

    **Band yield.** {_yield:.3g} photons / electron / sr over 0.8–10 keV
    from 32 histories. Its Monte Carlo error is not shown here and is likely
    larger than ε; increase the histories before comparing with ε.

    **The 5 MeV refusal:** `{refusal}`

    One fibre history per line electron gives both the channeling source and
    the electron's fate: electrons that leave the axial zone continue in
    native transport (adding bremsstrahlung), retained ones exit at the back
    face. Native PXR/CBS lines are not scored inside the zone, so the fibre
    band is never counted twice.
    Replay, CUDA, block and adaptive transport, coherent emission, finite
    detectors and remote runs all refuse these cases; see the
    [admitted axial envelope](../../research/physics/axial-first-envelope-886.md)
    for the full list and the evidence needed to widen it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Where photon OAM enters

    Separately from the admitted runs, the axial code has an **experimental
    single-string line catalog**,
    polarization matrices, forward-energy Doppler shifts, and aperture-dependent
    helicity / total-angular-momentum diagnostics. These are *different* from
    the electron OAM projection above. A candidate dipole transition in the
    string-local model has $\Delta\ell=\pm1$ and photon total $J_z=\pm\hbar$;
    on-axis helicity agrees with $J_z$, while off-axis orbital and spin angular
    momentum need not be individually equal to that integer. **An electron
    vortex is not, by itself, proof of an OAM X-ray source.**

    For the **experimental, research-only** radiation displays (candidate
    line energies, depth-integrated emission strengths, helicity and far-field
    patterns), launch the channeling app from the PyRITE checkout:

    ```bash
    uv run pyrite app channeling launch
    ```

    Select **Axial (experimental)**. These displays need additional
    string-local approximations and cannot be interpreted as validated
    experimental flux. Unlike the electron $P_\ell$ above, a line's circular
    dipole amplitudes are not its population-weighted photon count.

    ## Try next

    - Change vortex charge to `-1`, then compare the OAM weights and current.
    - Run `si-fibre/A`: set `tilt_deg=0.088, tilt_azim_deg=225.0` on the slab.
      Nearby tilts refuse: admission matches the exact calibrated case.
    - Raise `histories` in section 4 (about 4 s each) before trusting
      individual line shapes.
    - Shift both sources off the string (e.g. `center_ang=(0.4, 0.0)`): OAM
      *about string 0* is no longer the same as OAM about the beam center.
    - Increase electrons to 100+ **outside this quick notebook**, repeat
      with independent seeds, and then refine `shape`, `cells`, and `step_ang`
      independently. Do not infer convergence from unitary norm alone.

    Refer to the [quantum API](../../reference/quantum-api.md) and
    [channeling validation ledger](../../validation/ledger-channeling.md)
    for the status and limitations of each diagnostic.
    """)
    return


if __name__ == "__main__":
    app.run()
