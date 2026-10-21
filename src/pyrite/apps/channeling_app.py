# /// script
# [tool.marimo.display]
# theme = "light"
# ///

"""Interactive planar and experimental axial channeling diagnostics."""

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from pyrite.apps._design import page_title, style_sheet, theme_switch

    theme_ui = theme_switch(mo)
    mo.vstack(
        [
            style_sheet(mo),
            mo.hstack(
                [
                    page_title(
                        mo,
                        "Channeling diagnostics",
                        "Explore planar states and radiation, or experimental axial wave and ensemble diagnostics.",
                        eyebrow="PyRITE physics demo",
                    ),
                    theme_ui,
                ],
                justify="space-between",
                align="start",
                wrap=True,
            ),
            mo.md(
                "Choose the planar quantum model or experimental axial diagnostics on the CPU. "
                "The plots below describe model states and probabilities; they are not detector measurements. "
                "Trajectory replay is not available for channeling cases."
            ),
        ]
    )
    return (mo,)


@app.cell
def _(mo):
    mode_ui = mo.ui.radio(["Planar", "Axial (experimental)"], value="Planar", label="Mode")
    mode_ui
    return (mode_ui,)


@app.cell
def _(mo, mode_ui):
    mo.stop(mode_ui.value != "Planar")
    crystal_ui = mo.ui.dropdown(
        {"Silicon (110)": "silicon", "Diamond (110)": "diamond"},
        value="Silicon (110)",
        label="Crystal",
    )
    energy_ui = mo.ui.slider(1, 10, value=5, step=1, label="Beam energy (MeV)")
    incidence_ui = mo.ui.slider(-5, 5, value=0, step=0.1, label="Incidence (mrad)")
    divergence_ui = mo.ui.slider(0, 5, value=1, step=0.25, label="Divergence (mrad)")
    depth_ui = mo.ui.slider(0.2, 3.0, value=1.0, step=0.2, label="Zone depth (µm)")
    mo.hstack([crystal_ui, energy_ui, incidence_ui, divergence_ui, depth_ui], wrap=True)
    return crystal_ui, depth_ui, divergence_ui, energy_ui, incidence_ui


@app.cell
def _(crystal_ui, depth_ui, divergence_ui, energy_ui, incidence_ui):
    crystal = crystal_ui.value
    energy_eV = energy_ui.value * 1e6
    incidence_rad = incidence_ui.value * 1e-3
    divergence_rad = divergence_ui.value * 1e-3
    depth_ang = depth_ui.value * 1e4
    return crystal, depth_ang, divergence_rad, energy_eV, incidence_rad


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from pyrite.apps.channeling import critical_angle_rad
    from pyrite.materials.channeling import (
        entry_populations,
        longitudinal_kinematics,
        planar_profile,
        transverse_states,
    )
    from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2
    from pyrite.materials.crystal import M_E_EV
    from pyrite.montecarlo.channeling import (
        DECHANNELING_MARGIN_EV,
        DensityMatrixSolver,
        channeling_spectrum,
        electronic_kernel,
        tds_kernel,
    )

    return (
        DECHANNELING_MARGIN_EV,
        DensityMatrixSolver,
        HBAR2_2M_EV_ANG2,
        M_E_EV,
        channeling_spectrum,
        critical_angle_rad,
        electronic_kernel,
        entry_populations,
        longitudinal_kinematics,
        np,
        planar_profile,
        plt,
        tds_kernel,
        transverse_states,
    )


@app.cell
def _(crystal, energy_eV, longitudinal_kinematics, planar_profile, transverse_states):

    profile = planar_profile(crystal, (1, 1, 0), 293.15)
    gamma, _ = longitudinal_kinematics(energy_eV)
    states = transverse_states(profile, gamma, n_plane_waves=16)
    return gamma, profile, states


@app.cell
def _(
    DECHANNELING_MARGIN_EV,
    DensityMatrixSolver,
    HBAR2_2M_EV_ANG2,
    depth_ang,
    divergence_rad,
    electronic_kernel,
    energy_eV,
    gamma,
    incidence_rad,
    np,
    profile,
    tds_kernel,
):
    # Include enough plane waves to represent tracked states above the declared
    # dechanneling cutoff. Otherwise those losses are booked as overflow.
    _, barrier_eV = profile.extrema()
    basis_order = (
        int(
            np.ceil(
                np.sqrt((barrier_eV + DECHANNELING_MARGIN_EV) * gamma / HBAR2_2M_EV_ANG2)
                / profile.g_inv_ang
            )
        )
        + 2
    )
    grid = {"n_cells": 8, "max_order": basis_order}
    kernel = tds_kernel(profile, energy_eV, **grid) + electronic_kernel(
        profile,
        energy_eV,
        n_cells=grid["n_cells"],
        max_order=grid["max_order"],
    )
    solver = DensityMatrixSolver(profile, energy_eV, kernel)
    position_ang = np.linspace(0.0, profile.period_ang, 257)
    result = solver.propagate(
        depth_ang,
        100.0,
        incidence_rad=incidence_rad,
        divergence_rad=divergence_rad,
        record_every=5,
        spatial_x_ang=position_ang,
        record_metrics=True,
    )
    return basis_order, position_ang, result, solver


@app.cell
def _(divergence_rad, energy_eV, entry_populations, np, profile):

    angles = np.linspace(-8e-3, 8e-3, 101)
    populations = entry_populations(
        profile,
        energy_eV,
        angles,
        divergence_rad=divergence_rad,
        n_plane_waves=16,
    )
    return angles, populations


@app.cell
def _(channeling_spectrum, depth_ang, divergence_rad, incidence_rad, np, solver):
    theta = 0.0
    energy_grid = np.linspace(100.0, 100_000.0, 1200)
    spectrum = channeling_spectrum(
        solver,
        depth_ang,
        energy_grid,
        theta,
        incidence_rad=incidence_rad,
        divergence_rad=divergence_rad,
        dz_ang=100.0,
        sample_ang=500.0,
    )
    return energy_grid, spectrum


@app.cell
def _(
    M_E_EV,
    angles,
    basis_order,
    critical_angle_rad,
    crystal,
    energy_eV,
    energy_grid,
    gamma,
    mo,
    np,
    plt,
    populations,
    profile,
    result,
    spectrum,
    states,
):
    bound_count = states.n_bound

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    x = np.linspace(0.0, profile.period_ang, 500)
    potential = profile.potential(x)
    axes[0, 0].plot(x, potential, color="#315c9b", lw=2)
    axes[0, 0].axhline(states.barrier_eV, color="0.4", ls="--", label="barrier (potential maximum)")
    for level in states.energies_eV[:bound_count]:
        axes[0, 0].axhline(level, color="#d35d4b", alpha=0.55, lw=0.8)
    axes[0, 0].set(
        title="Planar potential and bound levels",
        xlabel="Across one plane period (Å)",
        ylabel="Transverse energy (eV)",
    )
    axes[0, 0].legend(frameon=False)

    for band in range(min(bound_count, 5)):
        axes[0, 1].plot(angles * 1e3, populations[:, band], label=f"state {band}")
    axes[0, 1].plot(
        angles * 1e3,
        np.clip(1 - populations[:, :bound_count].sum(axis=1), 0, 1),
        color="0.25",
        ls="--",
        label="not in bound states",
    )
    psi_c = critical_angle_rad(profile, energy_eV)
    axes[0, 1].axvline(psi_c * 1e3, color="0.4", ls=":")
    axes[0, 1].axvline(-psi_c * 1e3, color="0.4", ls=":")
    axes[0, 1].set(
        title="Entry population vs incidence",
        xlabel="Incidence to plane (mrad)",
        ylabel="Population",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    z_um = result.z_ang / 1e4
    axes[0, 2].plot(z_um, result.bound_population, label="bound")
    axes[0, 2].plot(
        z_um, result.free_flux + result.reprojection_flux, label="free handoff (incl. reprojection)"
    )
    axes[0, 2].plot(z_um, result.overflow_flux, label="overflow")
    axes[0, 2].plot(z_um, result.trace, color="0.2", ls=":", label="population still in zone")
    axes[0, 2].set(
        title="Depth evolution and fate balance", xlabel="Depth (µm)", ylabel="Probability"
    )
    axes[0, 2].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(
        energy_grid / 1e3, spectrum.spectrum - spectrum.capture_spectrum, label="bound → bound"
    )
    axes[1, 0].plot(energy_grid / 1e3, spectrum.capture_spectrum, label="free → bound capture")
    axes[1, 0].set(
        title="Forward channeling-radiation spectrum",
        xlabel="Photon energy (keV)",
        ylabel="Photons / eV / sr / electron",
        yscale="log",
    )
    axes[1, 0].legend(frameon=False)

    flux = result.free_flux_by_energy
    pv = gamma * M_E_EV * (1.0 - gamma**-2)
    psi_edges_mrad = np.sqrt(2.0 * np.maximum(result.energy_edges_eV, 0) / pv) * 1e3
    handoff_total = float(flux.sum())
    if handoff_total > 0.0:
        density_per_mrad = (flux / handoff_total) / np.diff(psi_edges_mrad)
        axes[1, 1].step(psi_edges_mrad[:-1], density_per_mrad, where="post", color="#8055a5")
        axes[1, 1].set_ylabel("Conditional density (per mrad)")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No free handoff flux\nfor these settings",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_ylabel("Conditional density (per mrad)")
    axes[1, 1].set(
        title="Modeled free handoff angle",
        xlabel="Plane-normal angle magnitude (mrad; symmetric sign omitted)",
    )

    axes[1, 2].bar(
        ["entry", "free", "overflow", "bound at exit", "tracked above barrier"],
        [
            result.entry_flux,
            result.free_flux[-1] + result.reprojection_flux[-1],
            result.overflow_flux[-1],
            result.bound_population[-1],
            max(0.0, result.trace[-1] - result.bound_population[-1]),
        ],
        color=["#888888", "#8055a5", "#d99845", "#315c9b", "#67a88a"],
    )
    axes[1, 2].set(title="Exit fate fractions", ylabel="Share of incident electrons", ylim=(0, 1))
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(f"{crystal.title()} (110), {energy_eV / 1e6:g} MeV · {bound_count} bound states")
    explanation = mo.md(
        f"**Critical angle:** {psi_c * 1e3:.2f} mrad. "
        f"**Plane-wave order:** ±{basis_order} (chosen to cover the 300 eV tracking margin). "
        f"**Free handoff flux:** {handoff_total:.3g} of incident electrons. "
        f"**Probability balance defect:** {result.balance_defect[-1]:.2e}.\n\n"
        "**Reading the panels.** In the potential panel, the dashed line is the "
        "maximum of the planar potential (the barrier); each red horizontal line is "
        "a bound transverse-state energy, not a trajectory or wavefunction. States "
        "below the barrier are bound. Entry curves show how an ideal incoming beam "
        "projects onto those states as its angle changes. ‘Not in bound states’ is "
        "the remainder: continuum-like states above the barrier plus any weight "
        "outside the finite plotted state basis.\n\n"
        "In the depth panel, **population still in zone** is the solver trace: "
        "electrons still represented by its evolving state basis, including some "
        "near-barrier states. **Bound** is the subset below the barrier. **Free "
        "handoff** is accumulated flux leaving into tracked free states; **overflow** "
        "is flux scattered beyond the finite momentum basis, so its exit angle is "
        "not resolved. ‘Entry’ means weight outside the tracked subspace at the "
        "crystal face. The fate bars also separate bound electrons still present "
        "at the exit from above-barrier electrons that remain in the tracked zone.\n\n"
        "The handoff-angle panel converts the tracked free-state transverse-energy "
        "flux into angle using $E_\\perp=pv\\psi^2/2$. It shows a conditional "
        "density given that such a handoff occurred, with the symmetric positive "
        "and negative angles folded together; the total handoff share is printed "
        "above. It is a model distribution, not a measured output beam.\n\n"
        "The spectrum is the model’s forward photon spectrum: bound-to-bound "
        "transitions plus free-to-bound radiative capture. Fine ripples can come "
        "from the calculated coherent sum of Bloch-block line-shape modes within "
        "a transition, rather than from energy-bin noise. Their detailed shape is "
        "sensitive to model resolution and assumptions; compare coarser features "
        "and peak locations before interpreting small ripples as experimentally "
        "robust fringes. This view uses a point detector and does not add detector "
        "resolution or energy spread."
    )
    mo.vstack([explanation, fig])
    return


@app.cell
def _(gamma, mo, np):
    from pyrite.apps.channeling import far_field_figure

    _far = far_field_figure(gamma, np.diag([1.0, 0.0]), "Planar dipole line (x = plane normal)")
    mo.vstack(
        [
            mo.md(
                "**Planar far field.** Every planar line is a real dipole along the plane "
                "normal, so these maps are shared by all lines at this energy. It is an equal "
                "coherent sum of J_z = +1 and −1: on a fixed circular basis E₊ and E₋ "
                "mirror each other and vanish at the boosted dipole null (γθx = ±1, "
                "θy = 0). Circles inside γθ = 1 see no winding; circles enclosing both "
                "nulls wind −2 (E₊) and +2 (E₋), so the line carries no net OAM. A pure axial J_z = +1 line "
                "instead leaves E₊ flat and winds E₋ twice everywhere. Contours give the "
                "Doppler energy relative to forward; the spectrum above is the forward "
                "line content."
            ),
            _far,
        ]
    )
    return


@app.cell
def _(mo, states):
    state_ui = mo.ui.dropdown(
        {f"State {n} · {energy:.3f} eV": n for n, energy in enumerate(states.energies_eV)},
        value=f"State 0 · {states.energies_eV[0]:.3f} eV",
        label="Transverse eigenstate (Bloch k = 0 Å⁻¹)",
    )
    state_ui
    return (state_ui,)


@app.cell
def _(mo, np, plt, position_ang, profile, state_ui, states):
    _state = state_ui.value
    _psi = states.wavefunction(position_ang, _state)
    # Fix only the displayed global phase: largest plane-wave coefficient positive real.
    _coeff = states.coefficients[:, _state]
    _psi *= np.exp(-1j * np.angle(_coeff[np.argmax(np.abs(_coeff))]))
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 3.5), constrained_layout=True)
    _axes[0].plot(position_ang, profile.potential(position_ang))
    _axes[0].axhline(states.energies_eV[_state], color="#d35d4b", label=f"state {_state}")
    _axes[0].set(title="Selected level and potential", ylabel="Transverse energy (eV)")
    _axes[0].legend()
    _axes[1].plot(position_ang, _psi.real, label="Re ψ")
    _axes[1].plot(position_ang, _psi.imag, label="Im ψ", ls="--")
    _axes[1].set(title="Single eigenfunction", ylabel="Amplitude (Å⁻½)")
    _axes[1].legend()
    _axes[2].plot(position_ang, np.abs(_psi) ** 2)
    _axes[2].set(title="Eigenstate probability density", ylabel="|ψ|² (Å⁻¹)")
    for _axis in _axes:
        _axis.set_xlabel("Across one plane period (Å)")
        _axis.grid(alpha=0.2)
    mo.vstack(
        [
            mo.md(
                "Eigenfunctions have unit probability per period at **Bloch k = 0 Å⁻¹**. "
                "The displayed global phase makes the largest plane-wave coefficient real and positive; "
                "it does not affect |ψ|². Above-barrier levels are extended states."
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo, result):
    sample_ui = mo.ui.slider(
        0,
        len(result.z_ang) - 1,
        value=0,
        step=1,
        label="Recorded depth index",
        show_value=True,
    )
    conditional_ui = mo.ui.checkbox(value=False, label="Condition on remaining in the zone")
    mo.hstack([sample_ui, conditional_ui])
    return conditional_ui, sample_ui


@app.cell
def _(conditional_ui, mo, np, plt, result, sample_ui):
    _sample = sample_ui.value
    _density = result.spatial_density_per_ang
    _label = "Density per incident electron (Å⁻¹)"
    if conditional_ui.value:
        _density = np.divide(
            _density,
            result.trace[:, None],
            out=np.zeros_like(_density),
            where=result.trace[:, None] > 0,
        )
        _label = "Conditional density (Å⁻¹)"
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    _mesh = _axes[0].pcolormesh(
        result.spatial_x_ang,
        result.z_ang / 1e4,
        _density,
        shading="auto",
        cmap="viridis",
    )
    _axes[0].axhline(result.z_ang[_sample] / 1e4, color="white", ls="--")
    _axes[0].set(title="Ensemble spatial density", xlabel="Position (Å)", ylabel="Depth (µm)")
    _fig.colorbar(_mesh, ax=_axes[0], label=_label)
    _axes[1].plot(result.spatial_x_ang, _density[_sample])
    _axes[1].set(
        title=f"Depth {result.z_ang[_sample] / 1e4:.3f} µm",
        xlabel="Across one plane period (Å)",
        ylabel=_label,
    )
    mo.vstack(
        [
            mo.md(
                f"**Surviving incident-electron weight at this depth:** {result.trace[_sample]:.5f}. "
                "The heatmap is an **ensemble probability density**, including retained coherences "
                "in every Bloch block; it is not a unique ensemble wavefunction. The supercell is "
                "folded into one period. Its integral equals surviving weight by default, or one "
                "when conditioned (zero if no weight remains). Lost electrons are excluded. "
                "Changing the state or depth selectors only redraws the stored solution."
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo, mode_ui):
    mo.stop(mode_ui.value != "Axial (experimental)")
    axial_form = (
        mo.md(
            """
        **Experimental axial diagnostic** — periodic CPU model; no campaign integration.

        {crystal} {axis} {energy} {mode} {grid} {cells}

        {depth} {step} {electrons} {seed} {preset}

        {waist} {impact_x} {impact_y} {tilt_x} {tilt_y} {oam}

        Waist w means exp(-r²/w²); impact is relative to the projected cell origin,
        drawn at the center of each map. The equal ±1 mixture adds probabilities,
        with equal incident weights. Tilt is a separate carrier angle. The app
        records 41 depths and caps grid samples × steps × trajectories at 1.1e9
        (1 mm coherent at 32² and 10 Å steps, about a minute; ~300 µm MCWF with
        32 electrons). Larger studies belong on remote compute.
        """
        )
        .batch(
            crystal=mo.ui.dropdown(["silicon", "diamond"], value="silicon", label="Crystal"),
            axis=mo.ui.dropdown(["001", "110"], value="001", label="Axis"),
            energy=mo.ui.slider(1, 10, value=5, label="Energy (MeV)"),
            mode=mo.ui.dropdown(["coherent", "mcwf"], value="coherent", label="Evolution"),
            grid=mo.ui.dropdown([16, 32, 48], value=32, label="Grid samples per axis"),
            cells=mo.ui.slider(1, 3, value=1, label="Cells per axis"),
            depth=mo.ui.dropdown(
                {
                    "10 nm": 100,
                    "50 nm": 500,
                    "100 nm": 1000,
                    "500 nm": 5000,
                    "1 µm": 10_000,
                    "2 µm": 20_000,
                    "5 µm": 50_000,
                    "10 µm": 100_000,
                    "20 µm": 200_000,
                    "50 µm": 500_000,
                    "100 µm": 1_000_000,
                    "200 µm": 2_000_000,
                    "500 µm": 5_000_000,
                    "1 mm": 10_000_000,
                },
                value="1 µm",
                label="Depth",
            ),
            step=mo.ui.dropdown([1, 2, 5, 10, 20], value=5, label="Step (Å)"),
            electrons=mo.ui.slider(1, 32, value=16, label="MCWF electrons"),
            seed=mo.ui.number(0, 100000, value=0, label="Seed"),
            preset=mo.ui.dropdown(
                ["Gaussian", "Vortex +1", "Vortex -1", "Equal ±1 mixture"],
                value="Gaussian",
                label="Entry preset",
            ),
            waist=mo.ui.slider(0.2, 1.0, step=0.1, value=0.5, label="Waist (Å)"),
            impact_x=mo.ui.slider(-1, 1, step=0.1, value=0, label="Impact x (Å)"),
            impact_y=mo.ui.slider(-1, 1, step=0.1, value=0, label="Impact y (Å)"),
            tilt_x=mo.ui.slider(-1, 1, step=0.1, value=0, label="Tilt x (mrad)"),
            tilt_y=mo.ui.slider(-1, 1, step=0.1, value=0, label="Tilt y (mrad)"),
            oam=mo.ui.checkbox(value=False, label="OAM about string 0, radius 0.5 Å"),
        )
        .form(submit_button_label="Run axial diagnostic")
    )
    axial_form
    return (axial_form,)


@app.cell
def _(axial_form, mo):
    from pyrite.apps.axial_channeling import (
        axial_demo_depths,
        axial_demo_figure,
        solve_axial_demo,
    )
    from pyrite.montecarlo.channeling.axial_diagnostic import AxialDiagnosticConfig

    mo.stop(axial_form.value is None, mo.md("Choose settings and press **Run axial diagnostic**."))
    _v = axial_form.value
    axial_config = AxialDiagnosticConfig(
        crystal=_v["crystal"],
        axis=(0, 0, 1) if _v["axis"] == "001" else (1, 1, 0),
        kinetic_energy_eV=_v["energy"] * 1e6,
        cells=(_v["cells"],) * 2,
        shape=(_v["grid"],) * 2,
        depths_ang=axial_demo_depths(_v["depth"], _v["step"]),
        step_ang=_v["step"],
        tilt_rad=(_v["tilt_x"] * 1e-3, _v["tilt_y"] * 1e-3),
        mode=_v["mode"],
        n_electrons=_v["electrons"],
        seed=int(_v["seed"]),
        oam_string=0 if _v["oam"] else None,
    )
    try:
        axial_result = solve_axial_demo(
            axial_config, _v["preset"], _v["waist"], (_v["impact_x"], _v["impact_y"])
        )
    except ValueError as _error:
        mo.stop(True, mo.md(f"**Not run:** {_error}"))
    return axial_demo_figure, axial_result


@app.cell
def _(axial_result, mo):
    get_axial_frame, set_axial_frame = mo.state(0)
    axial_frames = len(axial_result.depths_ang)
    axial_play_ui = mo.ui.switch(value=False, label="Play depth")
    axial_speed_ui = mo.ui.dropdown(
        ["0.25s", "0.5s", "1s", "2s"], value="0.5s", label="Frame interval"
    )
    axial_conditional_ui = mo.ui.checkbox(
        value=False, label="Condition axial density on grid survival"
    )
    return (
        axial_conditional_ui,
        axial_frames,
        axial_play_ui,
        axial_speed_ui,
        get_axial_frame,
        set_axial_frame,
    )


@app.cell
def _(axial_play_ui, axial_speed_ui, mo):
    # The refresh element only exists (and only ticks) while playing.
    axial_tick_ui = (
        mo.ui.refresh(options=[axial_speed_ui.value], default_interval=axial_speed_ui.value)
        if axial_play_ui.value
        else None
    )
    return (axial_tick_ui,)


@app.cell
def _(axial_frames, axial_tick_ui, get_axial_frame, mo, set_axial_frame):
    mo.stop(axial_tick_ui is None)
    _ = axial_tick_ui.value  # rerun on every tick
    set_axial_frame((get_axial_frame() + 1) % axial_frames)
    return


@app.cell
def _(
    axial_conditional_ui,
    axial_frames,
    axial_play_ui,
    axial_speed_ui,
    axial_tick_ui,
    get_axial_frame,
    mo,
    set_axial_frame,
):
    axial_sample_ui = mo.ui.slider(
        0,
        axial_frames - 1,
        value=get_axial_frame(),
        label="Recorded axial depth",
        show_value=True,
        on_change=set_axial_frame,
    )
    _controls = [axial_sample_ui, axial_play_ui, axial_speed_ui, axial_conditional_ui]
    if axial_tick_ui is not None:
        _controls.append(axial_tick_ui)
    mo.hstack(_controls, wrap=True)
    return (axial_sample_ui,)


@app.cell
def _(axial_conditional_ui, axial_demo_figure, axial_result, axial_sample_ui, mo):
    _index = axial_sample_ui.value
    _fig = axial_demo_figure(axial_result, _index, axial_conditional_ui.value)
    _notes = (
        f"**{axial_result.config.mode} · depth {axial_result.depths_ang[_index] / 1e4:.4g} µm** · "
        f"surviving weight {axial_result.surviving[_index]:.6f}. "
        "Density integrates to incident-electron survival, or one when conditioned "
        "(zero if extinct). Coherent norm is not scattering survival. "
        "The finite periodic grid can wrap the probe and lose momentum transfers; "
        "compare grid, supercell and step convergence. MCWF statistics here are illustrative. "
        "Existing axial validation limits remain in the physics ledger. "
    )
    _items = [mo.md(_notes), _fig]
    if axial_result.oam_populations is not None:
        import matplotlib.pyplot as _axial_plt

        _ofig, _oax = _axial_plt.subplots()
        for _i, _ell in enumerate(axial_result.oam_ells):
            _oax.plot(
                axial_result.depths_ang / 1e4,
                axial_result.oam_populations[:, _i],
                label=f"ℓ={_ell}",
            )
        _ox, _oy = axial_result.oam_origin_ang
        _oax.set(
            xlabel="Depth (µm)",
            ylabel="Pℓ per incident electron in disk",
            title=f"OAM about string 0 at ({_ox:.2f}, {_oy:.2f}) Å; R=0.5 Å",
        )
        _oax.legend(ncol=4, fontsize=7)
        _items.extend(
            [
                mo.md(
                    "Disk OAM is truncated to |ℓ| ≤ 8; omitted disk/mode weight is not renormalized."
                ),
                _ofig,
            ]
        )
    mo.vstack(_items)
    return


@app.cell
def _(axial_result, mo):
    from pyrite.apps.axial_channeling import (
        axial_line_options,
        axial_radiation,
        axial_radiation_figures,
    )

    axial_radiation_data = axial_radiation(axial_result)
    mo.stop(
        axial_radiation_data is None,
        mo.md(
            "**Radiation:** no flat-level line catalog for this case (needs one string per "
            "cell, as on [001], and at least two flat bound levels at this energy)."
        ),
    )
    assert axial_radiation_data is not None
    _options, _strongest = axial_line_options(axial_radiation_data)
    axial_line_ui = mo.ui.dropdown(_options, value=_strongest, label="Radiation line")
    axial_line_ui
    return axial_line_ui, axial_radiation_data, axial_radiation_figures


@app.cell
def _(
    axial_line_ui,
    axial_radiation_data,
    axial_radiation_figures,
    axial_sample_ui,
    mo,
):
    _spectrum, _far, _moments = axial_radiation_figures(
        axial_radiation_data, axial_sample_ui.value, axial_line_ui.value
    )
    _items = [
        mo.md(
            "**Axial channeling radiation (string 0).** Bars are photons per incident "
            "electron emitted before the selected depth, from the level densities of string "
            "0's Wigner–Seitz cell, split by the line's J_z = ±1 content (the on-axis "
            "helicity). Maps show the selected line's far field on a fixed lab circular "
            "basis versus γθ: σ = ±1 picks a fixed circular polarizer, and a phase winding "
            "of ℓ around the axis is orbital angular momentum ℓ = J_z − σ (a pure J_z = +1 "
            "line gives a plain E₊ and an E₋ that winds twice). Phase is the leading "
            "coherency mode's; purity below one means an incoherent mix. Contours give the "
            "Doppler energy relative to forward. Dipole order; levels and lines follow "
            "`cr-axial-helicity-emission`, fields `cr-dipole-far-field-amplitude`."
        ),
        _spectrum,
    ]
    if _far is None:
        _items.append(mo.md("The selected line has no emission before this depth."))
    else:
        _items.extend(
            [
                mo.md(
                    f"Within θ ≤ 1/γ: ⟨helicity⟩ {_moments['helicity']:+.3f}, "
                    f"J_z {_moments['J_z']:+.3f}, S_z {_moments['S_z']:+.3f}, "
                    f"L_z {_moments['L_z']:+.3f} ħ per photon; "
                    f"{_moments['fraction']:.3f} of the line's photons fall in that cone."
                ),
                _far,
            ]
        )
    mo.vstack(_items)
    return


@app.cell
def _(conditional_ui, mo, result, sample_ui):
    from pyrite.apps.channeling_metrics import metric_figure, population_figure

    _pop = population_figure(
        result.z_ang,
        result.retained_band_populations,
        result.trace,
        sample_ui.value,
        conditional=conditional_ui.value,
    )
    _metrics = metric_figure(result.z_ang, result.metrics)
    mo.vstack(
        [
            mo.md(
                "**Instantaneous Bloch bands**, summed over blocks; all retained bands. "
                "Purity uses the full block-diagonal retained ensemble, conditioned on its trace. "
                "Coherence is the squared off-diagonal Frobenius norm in the instantaneous "
                "eigenbasis; it depends on that basis. Empty conditional metrics are undefined."
            ),
            _pop,
            _metrics,
        ]
    )
    return


@app.cell
def _(axial_conditional_ui, axial_result, axial_sample_ui, mo):
    from pyrite.apps.axial_channeling import axial_population_data
    from pyrite.apps.channeling_metrics import axial_accounting_figure
    from pyrite.apps.channeling_metrics import metric_figure as axial_metric_figure
    from pyrite.apps.channeling_metrics import population_figure as axial_population_figure

    _populations, _labels, _basis = axial_population_data(axial_result)
    _pop = axial_population_figure(
        axial_result.depths_ang,
        _populations,
        axial_result.surviving,
        axial_sample_ui.value,
        conditional=axial_conditional_ui.value,
        labels=_labels,
    )
    _metrics = axial_metric_figure(axial_result.depths_ang, axial_result.metrics)
    mo.vstack(
        [
            mo.md(
                f"**Population basis:** {_basis}. Degenerate levels are summed, not split arbitrarily. "
                "Purity estimates the full finite physical-position-grid ensemble, conditioned "
                "on survival. Error bars are one probe SEM (32 fixed probes), **excluding** "
                "finite-trajectory sampling error and empirical-ensemble bias. Raw estimates "
                "can exceed physical bounds. Squared coherence uses the fixed position grid. "
                "Conditional metrics are undefined when all electrons are lost. Numerical "
                "grid sinks are not physical escape; total jump counts are shown below."
            ),
            _pop,
            _metrics,
            axial_accounting_figure(axial_result),
            mo.md(f"**Total jumps:** {axial_result.jumps}"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
