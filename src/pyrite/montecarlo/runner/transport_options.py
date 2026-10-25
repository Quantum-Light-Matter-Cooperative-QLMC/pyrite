"""Transport option forwarding and opt-in capability checks."""

from ..case import refuse_channeling, refuse_transition_radiation


def _beam_kwargs(case):
    """Beam phase-space kwargs a case dict forwards to ``simulate_trajectories``:
    the transverse spot (elliptical ``beam_fwhm_mm`` / ``beam_fwhm_y_mm``) and
    the legacy longitudinal bunch (``bunch_length_fs`` / ``long_shape`` /
    ``long_offsets_fs``) or resolved ``longitudinal_distribution`` policy, the
    resolved ``transverse_distribution`` (Twiss) policy that replaces the spot,
    and ``energy_spread_frac``.
    Absent keys default to the point-bunch isotropic beam, bit-for-bit with the
    pre-BeamSpec case dict."""
    return dict(
        beam_fwhm_mm=case.get("beam_fwhm_mm"),
        beam_fwhm_y_mm=case.get("beam_fwhm_y_mm"),
        bunch_length_fs=case.get("bunch_length_fs"),
        long_shape=case.get("long_shape", "gaussian"),
        long_offsets_fs=case.get("long_offsets_fs"),
        longitudinal_distribution=case.get("longitudinal_distribution"),
        transverse_distribution=case.get("transverse_distribution"),
        energy_spread_frac=case.get("energy_spread_frac"),
        bunch_front_tilt=case.get("bunch_front_tilt"),
        **({"gdf_source": case["gdf_source"]} if "gdf_source" in case else {}),
    )


def _check_opt_in_transport(
    case, trajectory_capture, block_electrons, *, transport_core="auto", block_monitor=None
):
    """Refuse transport options an opt-in component lacks, before any work (ADR-0017 tier 3)."""
    if case.get("photon_transport") is not None:
        from .photon_aperture import prepare_photon_case

        if block_electrons is not None or block_monitor is not None:
            raise ValueError("photon transport does not support block/adaptive transport")
        prepare_photon_case(case, transport_core)
    if trajectory_capture is not None:
        refuse_channeling(case, "trajectory capture")
    if block_electrons is not None:
        refuse_channeling(case, "electron-block transport")
        refuse_transition_radiation(case, "electron-block transport")
