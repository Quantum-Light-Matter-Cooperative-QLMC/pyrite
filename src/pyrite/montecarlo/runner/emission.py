"""Layer-resolved background emission and checkpoint bremsstrahlung repair.

Resolve runner dependencies at call time to preserve the facade test seams.
"""

from typing import Any

import numpy as np


def _brem_wide_from_segments(
    segs_b,
    E_brem,
    case,
    n_hat,
    abs_layers,
    groove=None,
    Ne=None,
    electron_band_weights=None,
):
    """Bremsstrahlung background on ``E_brem`` from already-transported brem
    segments ``segs_b``. EVERY layer radiates with its OWN composition (each
    Z^2 cross section) and self-absorbs through the WHOLE stack
    (``layers=abs_layers``); the per-layer contributions are summed. A single
    layer (``n_layers == 1``) is exactly the old single-material brem. Honors
    ``brem_chunk`` (segments per GPU matmul). Pure move of _spectrum_case's brem
    block; shared with :func:`_brem_for_case` so a brem-only repair regenerates
    the SAME multilayer background as a live sweep.

    A coupled radiative case (``radiative_model``) scores its coupled tracks
    with the BremsLib expected-value estimator; see
    :func:`_coupled_brem_from_segments`.

    ``electron_band_weights`` (internal, #361) returns each brem electron's
    in-band integral instead, summed over layers; see ``mc_brem_spectrum``."""
    from .. import runner

    brem_chunk = runner._admit_chunk(
        case.get("brem_chunk")
        or runner._RESOURCE_POLICY.brem_chunk
        or runner._adaptive_chunk(E_brem.size, intermediates=runner._EEDL_BREM_DENSE_INTERMEDIATES),
        E_brem.size,
        intermediates=runner._EEDL_BREM_DENSE_INTERMEDIATES,
    )
    n_lay = int(segs_b.get("n_layers", 1))
    if case.get("radiative_model") is not None:
        return _coupled_brem_from_segments(
            segs_b,
            E_brem,
            case,
            n_hat,
            abs_layers,
            Ne=Ne,
            brem_chunk=brem_chunk,
            electron_band_weights=electron_band_weights,
        )
    # The case records the resolved continuum: BremsLib gets one table set for
    # the whole stack (angular weighting per segment from ``v_hat``), and
    # anything else is pinned to EEDL so mc_brem_spectrum's "auto" never
    # re-resolves what identity already recorded.
    bremslib_tables = runner._case_bremslib_tables(case)
    model_kwargs: dict[str, Any] = (
        dict(cross_section_model="eedl")
        if bremslib_tables is None
        else dict(cross_section_model="bremslib", bremslib_tables=bremslib_tables)
    )
    if electron_band_weights is not None:
        model_kwargs["electron_band_weights"] = electron_band_weights

    if n_lay == 1:
        return runner.mc_brem_spectrum(
            segs_b,
            E_brem,
            composition=case["composition"],
            n_hat=n_hat,
            chunk=brem_chunk,
            layers=abs_layers,
            groove=groove,
            electron_limit=Ne,
            E_cut_keV=case.get("E_cut_brem_keV", 1.0),
            **model_kwargs,
        )
    shape = E_brem.shape if electron_band_weights is None else int(segs_b["Ne"] if Ne is None else Ne)
    brem_wide = np.zeros(shape, dtype=float)
    for L in range(n_lay):
        sL = runner._segments_in_layer(segs_b, L)
        if sL["L_ang"].size == 0:
            continue
        brem_wide = brem_wide + runner.mc_brem_spectrum(
            sL,
            E_brem,
            composition=abs_layers[L][2],
            n_hat=n_hat,
            chunk=brem_chunk,
            layers=abs_layers,
            groove=groove,
            electron_limit=Ne,
            E_cut_keV=case.get("E_cut_brem_keV", 1.0),
            **model_kwargs,
        )
    return brem_wide


def _coupled_brem_from_segments(
    segs_b, E_brem, case, n_hat, abs_layers, *, Ne, brem_chunk, electron_band_weights=None
):
    """Coupled-mode continuum: BremsLib track length on the coupled tracks.

    Each layer radiates with its own composition, like the uncoupled
    estimator, over soft and hard photon energies alike. The sampled hard
    photons shaped the tracks but are not histogrammed: at production electron
    counts they leave the continuum above the cutoff almost empty. Transport
    stopped every continuum electron at ``E_cut_brem_keV`` (the case requires
    it not to exceed the line cutoff), so no reclipping is requested.

    Validation: bremslib-coupled-expected-spectrum
    """
    from .. import runner
    from ..spectrum.brem_events import mc_coupled_brem_spectrum

    if case.get("E_cut_brem_keV", 1.0) > case.get("E_cut_lines_keV", 5.0):
        raise ValueError("coupled radiative scoring requires E_cut_brem_keV <= E_cut_lines_keV")
    tables = runner._case_bremslib_tables(case)
    if tables is None:
        raise ValueError("coupled radiative scoring requires bremsstrahlung_model='bremslib'")
    cutoff_eV = float(case["radiative_cutoff_eV"])
    n_lay = int(segs_b.get("n_layers", 1))
    layer_views = (
        [(segs_b, case["composition"])]
        if n_lay == 1
        else [
            (runner._segments_in_layer(segs_b, index), abs_layers[index][2])
            for index in range(n_lay)
        ]
    )
    per_electron = (
        {} if electron_band_weights is None else {"electron_band_weights": electron_band_weights}
    )
    brem_wide = np.zeros(E_brem.shape if electron_band_weights is None else int(Ne), dtype=float)
    for segments, composition in layer_views:
        if segments["L_ang"].size == 0:
            continue
        brem_wide = brem_wide + mc_coupled_brem_spectrum(
            segments,
            E_brem,
            composition=composition,
            n_hat=n_hat,
            chunk=brem_chunk,
            layers=abs_layers,
            electron_limit=Ne,
            cutoff_eV=cutoff_eV,
            bremslib_tables=tables,
            **per_electron,
        )
    return brem_wide


def _characteristic_from_segments(
    segs,
    E_grid,
    case,
    n_hat,
    abs_layers,
    groove=None,
    Ne=None,
):
    """Characteristic lines on the requested fine line-energy grid.

    Every material layer emits from its own EEDL subshell cross sections and
    xraydb relaxation data; photons self-absorb through the complete stack.
    The estimator uses the bremsstrahlung electron population because those
    tracks continue to the lower, background cutoff and therefore retain the
    low-energy ionization path that a line-only 5 keV cutoff would discard.
    Characteristic scoring itself enforces its documented 1 keV transport-
    validity floor even if a case requests a lower background cutoff.

    Validation: characteristic-radiation
    """
    from .. import runner

    characteristic_chunk = runner._admit_chunk(
        case.get("brem_chunk")
        or runner._RESOURCE_POLICY.brem_chunk
        or runner._adaptive_chunk(E_grid.size),
        E_grid.size,
    )
    n_lay = int(segs.get("n_layers", 1))
    if n_lay == 1:
        return runner.mc_characteristic_spectrum(
            segs,
            E_grid,
            composition=case["composition"],
            n_hat=n_hat,
            chunk=characteristic_chunk,
            layers=abs_layers,
            groove=groove,
            electron_limit=Ne,
            E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        )
    characteristic = np.zeros(E_grid.shape, dtype=float)
    for layer_index in range(n_lay):
        layer_segments = runner._segments_in_layer(segs, layer_index)
        if layer_segments["L_ang"].size == 0:
            continue
        characteristic = characteristic + runner.mc_characteristic_spectrum(
            layer_segments,
            E_grid,
            composition=abs_layers[layer_index][2],
            n_hat=n_hat,
            chunk=characteristic_chunk,
            layers=abs_layers,
            groove=groove,
            electron_limit=Ne,
            E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        )
    return characteristic


def _brem_for_case(case, E_brem):
    """Regenerate a case's bremsstrahlung background on ``E_brem`` from scratch:
    build the tilted geometry, transport ``Ne_brem`` electrons through the stack
    (``layers=abs_layers``, with the same seed as the shared live transport),
    and sum brem
    per layer via :func:`_brem_wide_from_segments`. Returns ``brem_wide``.

    This is the brem half of run_case's transport + spectrum phases factored out
    so :func:`pyrite.runs.run.repair_brem_wide` reuses the EXACT live-sweep path.
    Previously the repair rebuilt single-slab brem by hand -- ``layers=`` omitted,
    no per-layer sum, ``brem_chunk`` ignored -- silently dropping substrate
    backscatter/brem and cross-stack absorption on stacked/multilayer records.

    Validation: grazing-beam-projection
    """
    from .. import runner

    abs_layers = case.get("abs_layers")
    tilt_polar_rad = np.deg2rad(case.get("tilt_deg", 0.0))
    tilt_azim_rad = np.deg2rad(case.get("tilt_azim_deg", 0.0))
    beam, n_hat = runner.tilted_geometry(case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad)
    # Build once, then forward the identical groove through electron entry and
    # bremsstrahlung photon escape, matching the live-sweep path.
    groove = None
    if case.get("groove_spacing_ang") is not None:
        groove = runner.blazed_groove_spec(
            case["groove_spacing_ang"], case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad
        )

    Ne_brem = case["Ne_brem"]
    E_cut_brem = case.get("E_cut_brem_keV", 1.0)
    E_cut_by_electrons = np.full(Ne_brem, E_cut_brem, dtype=np.float64)

    segs_b = runner.simulate_trajectories(
        case["E0_keV"],
        case["Ne_brem"],
        case["thickness_ang"],
        E_cut_by_electrons=E_cut_by_electrons,
        composition=case["composition"],
        seed=case["seed"],
        beam_dir=beam,
        layers=abs_layers,
        **runner._beam_kwargs(case),
        crystal_width_mm=case.get("crystal_width_mm"),
        crystal_height_mm=case.get("crystal_height_mm"),
        tilt_polar_rad=tilt_polar_rad,
        tilt_azim_rad=tilt_azim_rad,
        groove=groove,
        stopping_tables=runner._case_stopping_tables(case),
        # The elastic model is transport physics: a repair replays it.
        **runner._case_elastic_kwargs(case),
        # The shell soft/hard and coupled radiative modes change the transport
        # physics, so a repair of such a case must replay them; other cases
        # are unchanged.
        **(
            dict(
                energy_model=case.get("energy_model", "frozen"),
                max_dE_frac=case.get("max_dE_frac", 0.0),
                straggling=bool(case.get("straggling", False)),
                **runner._case_inelastic_kwargs(case),
                **runner._case_radiative_kwargs(case),
            )
            if case.get("inelastic_model") is not None or case.get("radiative_model") is not None
            else {}
        ),
    )
    return runner._brem_wide_from_segments(
        segs_b,
        E_brem,
        case,
        n_hat,
        abs_layers,
        groove=groove,
    )
