"""Public entry points for the CXR line spectrum and the route dispatch.

:func:`mc_spectrum` builds a :class:`~._setup.SpectrumRequest`;
:func:`_mc_spectrum` prepares it, picks the per-hkl or batched accumulation
route, and finalizes. :func:`mc_spectrum_solid_angle` integrates the result
over a detector solid angle.
"""

import numpy as np

from ...._backend import REAL, _to_cpu, xp
from . import _policy
from ._batched import _accumulate_batched
from ._per_hkl import _accumulate_per_hkl
from ._setup import SpectrumRequest, _prepare_spectrum


def mc_spectrum(
    segments,
    E_grid_eV,
    crystal,
    hkl_list,
    theta_obs_rad=np.deg2rad(119.0),
    B_ang2=None,
    use_henke=True,
    absorber_element="C",
    chunk=40000,
    n_hat=None,
    composition=None,
    beam_uvw=None,
    azimuth_rad=0.0,
    recip_miscut_rad: tuple[float, float] | None = None,
    sinc_cutoff=None,
    components=False,
    layers=None,
    mosaic_fwhm_rad=None,
    mosaic_nodes=1,
    surface_hkl: tuple[int, int, int] | None = None,
    groove=None,
    coherent=False,
    electron_limit=None,
    E_cut_keV=None,
    _table_cache=None,
    longitudinal_rms_fs=None,
    line_quadrature="node",
):
    """
    Per-electron CXR spectrum d2N/dE dOmega [photons / eV / sr / electron] on
    E_grid_eV, summed over the trajectory segments and the listed reflections.
    Per segment and reflection (Zhai SI Eqs. 5-7, nonrelativistic)::

      omega_res = beta*v_hat.g / (1 - beta*v_hat.n)
      d2N/dE/dOmega = alpha*omega/(4*pi^2*hbar*c) * abs(A)^2 * t_L^2
                      * sinc^2[(1-beta*v.n)*(omega-omega_res)*t_L/2] * T_abs

    with A = A_PXR + A_CBS per polarization (Feranchuk Eqs. 13/14 at omega_res),
    t_L = L_seg/beta, and T_abs the Beer-Lambert escape factor: its mean along
    the segment (incoherent route; exact integrated yield), or its midpoint
    value on the amplitude (coherent and flight-grouped). The finite-time sinc^2 is the centered integral of ``exp(i 2 P t)``
    over ``t_L`` (normalized-sinc convention) under a constant segment velocity
    and amplitude; at zero detuning it is ``t_L**2``.

    Validation: finite-time-lineshape

    Physical model, documented once under ``docs/physics/``:

    - escape and self-absorption (segment mean or midpoint, above; Validation:
      segment-escape-average), and the in-medium
      dispersion ``k = n(omega) omega`` that both the kinematics and the
      coherent propagation phase run on -- real part only, bulk response, so
      grazing geometry is out of scope: radiation-physics/photon-escape-and-dispersion.md
    - crystal orientation (``beam_uvw`` a direct-lattice axis along +z,
      ``surface_hkl`` a reciprocal plane normal, ``recip_miscut_rad`` tilting
      only g) and the mosaic average: geometry/transport-geometry.md,
      materials/crystal-mosaicity.md
    - coherent field summation, its two coherence scales, and the bounded cross-g
      terms dropped by keeping reflections and orientations incoherent:
      radiation-physics/coherent-emission.md

    Mutually exclusive: ``coherent`` with ``components``, ``beam_uvw`` with
    ``surface_hkl``, the Monte-Carlo mosaic with analytic ``mosaic_fwhm_eV``;
    ``coherent`` is refused for layered absorbers.

    Validation: line-absorption-tabulation, self-absorption,
    finite-transverse-crystal, blazed-groove-geometry, xray-in-medium-resonance,
    xray-in-medium-propagation-phase, surface-hkl-orientation, mosaic-mc,
    cross-reflection-coherence

    Parameters
    ----------
    segments
        Transport output mapping from :func:`simulate_trajectories`.
    E_grid_eV
        One-dimensional line photon-energy grid in eV.
    crystal, hkl_list
        Catalog crystal key and reciprocal reflections to sum.
    theta_obs_rad, n_hat
        Polar observation angle [rad], or a sample-frame direction overriding it.
    B_ang2, use_henke
        Required isotropic Debye--Waller ``B`` factor in square angstroms, and
        whether to include anomalous energy-dependent atomic form factors.
    absorber_element, composition
        Elemental or compound self-absorption description.
    chunk
        Maximum transport segments processed per spectrum chunk.
    beam_uvw, surface_hkl, azimuth_rad, recip_miscut_rad
        Crystal-orientation controls.
    sinc_cutoff
        Optional dimensionless finite-time tail cutoff; ``None`` is exact.
    components
        Return separate PXR and CBS diagonal contributions with the total.
    mosaic_fwhm_rad, mosaic_nodes
        Mosaic rocking-curve FWHM and quadrature nodes per tilt axis.
    layers, groove
        Optional film-first absorber stack, and optional blazed-groove escape
        geometry for the beam-entrance face.
    coherent
        Sum segment fields coherently instead of segment intensities.
    electron_limit, E_cut_keV
        Optional leading macro-electron count used for normalization, and
        optional post-transport electron-energy cutoff in keV.
    longitudinal_rms_fs
        Resolved Gaussian RMS bunch duration for analytic coherent averaging.
    line_quadrature
        ``"node"`` (default) samples ``sinc^2`` at each grid node. ``"bin-mean"``
        writes each node's bin mean of the closed-form integrated profile, so
        ``sum(spec * bin_width)`` is the in-window line yield at any spacing;
        bins follow ``characteristic._energy_bin_edges_and_widths``. A
        yield-only quadrature: it smooths peak height and width. Incoherent
        only; refused with ``coherent``, ``sinc_cutoff``, or numerical
        substeps. Validation: sinc-bin-integration

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        Per-electron density in photons per eV per sr. With ``components=True``,
        total, PXR-diagonal and CBS-diagonal arrays; the latter two exclude
        their interference term.

    Raises
    ------
    ValueError
        If required physical inputs or mutually exclusive controls are invalid.
    NotImplementedError
        For coherent propagation through layered absorbers.

    Validation: coherent-emission, coherent-line-spectrum,
    coherent-segment-midpoint-time, finite-footprint-longitudinal-decoherence,
    transverse-bunch-form-factor
    """
    request = SpectrumRequest(
        segments=segments,
        E_grid_eV=E_grid_eV,
        crystal=crystal,
        hkl_list=hkl_list,
        theta_obs_rad=theta_obs_rad,
        B_ang2=B_ang2,
        use_henke=use_henke,
        absorber_element=absorber_element,
        chunk=chunk,
        n_hat=n_hat,
        composition=composition,
        beam_uvw=beam_uvw,
        azimuth_rad=azimuth_rad,
        recip_miscut_rad=recip_miscut_rad,
        sinc_cutoff=sinc_cutoff,
        components=components,
        layers=layers,
        mosaic_fwhm_rad=mosaic_fwhm_rad,
        mosaic_nodes=mosaic_nodes,
        surface_hkl=surface_hkl,
        groove=groove,
        coherent=coherent,
        electron_limit=electron_limit,
        E_cut_keV=E_cut_keV,
        _table_cache=_table_cache,
        longitudinal_rms_fs=longitudinal_rms_fs,
        line_quadrature=line_quadrature,
    )
    return _mc_spectrum(request)


def _finalize_spectrum(st):
    """Phase 3: normalise the accumulated buffers to per-electron units.

    ``spec`` carries the sum over segments of every reflection and
    orientation; dividing by the electron count gives the documented
    ``d2N/dE dOmega`` per incident electron. With ``components`` the PXR and
    CBS buffers are normalised alongside it.
    """
    components = st.request.components
    Ne = st.Ne
    spec = st.spec
    spec_pxr = st.spec_pxr
    spec_cbs = st.spec_cbs

    if components:
        return _to_cpu(spec / Ne), _to_cpu(spec_pxr / Ne), _to_cpu(spec_cbs / Ne)
    return _to_cpu(spec / Ne)


def _needs_per_hkl_route(st):
    """Whether this configuration must take the per-hkl compatibility loop.

    The batched ``(n_seg, N_g)`` path covers the single-slab absorber, with or
    without a finite crystal footprint -- the escape DISTANCE is g-independent
    either way, so it hoists -- for both coherent and incoherent emission.
    Layered and grooved absorbers stay on the proven per-hkl loop. Coherent
    sinc-windowed runs use it except when the CUDA-fp32 streaming kernel can
    apply the same cutoff without materializing dense field matrices.

    The flight-grouped reduction lives there too: its segmented complex sum
    has no batched or device counterpart yet, and correctness of the default
    incoherent yield outranks the batched path's launch-count win on the
    substepped configuration.
    """
    req = st.request
    stream_handles_cutoff = (
        _policy._USE_JIT_COHERENT_STREAM
        and getattr(xp, "__name__", "") == "cupy"
        and np.dtype(REAL) == np.dtype(np.float32)
    )
    return bool(
        (req.coherent and req.sinc_cutoff is not None and not stream_handles_cutoff)
        or req.groove is not None
        or req.layers is not None
        or st.grouped
    )


def _mc_spectrum(request):
    """Run the three spectrum phases for a bound request."""
    st = _prepare_spectrum(request)
    if _needs_per_hkl_route(st):
        _accumulate_per_hkl(st)
    else:
        _accumulate_batched(st)
    return _finalize_spectrum(st)


def mc_spectrum_solid_angle(
    segments,
    E_grid_eV,
    crystal,
    hkl_list,
    *,
    n_hats,
    weights,
    groove=None,
    **kwargs,
):
    """
    Solid-angle-INTEGRATED per-electron line spectrum dN/dE [photons / eV /
    electron], already x Omega: ``sum_i weights_i * mc_spectrum(n_hat=n_hats_i)``.

    The finite detector face is tiled by detector_directions() into directions
    ``n_hats`` (sample frame) carrying solid-angle ``weights``. Because the
    resonance energy AND the amplitudes depend on n_hat, summing per-direction
    spectra yields the true, generally ASYMMETRIC integrated lineshape and the
    across-face intensity gradient -- the first-principles replacement for the
    flat-Omega + analytic aperture_fwhm_eV pair (docs/physics/detectors/detector-solid-angle.md). It
    reuses the validated single-angle mc_spectrum, so a 1-direction grid
    reproduces ``spec * Omega`` exactly (the regression anchor).

    Units: the result ALREADY includes the solid angle (the weights carry
    dOmega). When consuming it do NOT multiply by domega_sr again, and drop the
    aperture_fwhm_eV term from the detector convolution (keep the EDS-resolution
    term). This is an opt-in tool; it does not change the checkpoint pipeline's
    single-n_hat unit convention. ``**kwargs`` forward to mc_spectrum (B_ang2,
    use_henke, layers, composition, beam_uvw, azimuth_rad, mosaic_*, ...).

    groove: forwarded to mc_spectrum's blazed-groove escape model, but ONLY
    when ``n_hats`` carries a single direction (the ``n_side=1`` case of
    detector_directions() -- N == n_side**2, so N > 1 means n_side > 1).
    Tiling the detector face into multiple directions breaks the relief-facet
    parallelism the groove geometry assumes (each tile would need its own
    working-facet family), so a multi-direction grid with groove set raises
    ValueError rather than silently mixing per-tile escape paths.

    Parameters
    ----------
    segments, E_grid_eV, crystal, hkl_list
        Inputs forwarded to :func:`mc_spectrum`.
    n_hats
        Observation directions with shape ``(n_direction, 3)``.
    weights
        Matching solid-angle quadrature weights in sr.
    groove
        Optional blazed-groove geometry; supported only for one direction.
    **kwargs
        Additional keyword arguments forwarded to :func:`mc_spectrum`.

    Returns
    -------
    numpy.ndarray
        Solid-angle-integrated line spectrum in photons per incident electron
        per eV.

    Raises
    ------
    ValueError
        If direction/weight shapes differ or grooves are combined with tiling.

    Validation: blazed-groove-geometry
    """
    n_hats = np.asarray(n_hats, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if n_hats.ndim != 2 or n_hats.shape[1] != 3:
        raise ValueError("n_hats must be (N, 3)")
    if weights.shape != (n_hats.shape[0],):
        raise ValueError("weights must be (N,) matching n_hats")
    if groove is not None and n_hats.shape[0] > 1:
        raise ValueError(
            "groove escape supports n_side=1 only -- detector tiles break "
            "the relief-facet parallelism"
        )
    total = None
    for n_i, w_i in zip(n_hats, weights, strict=True):
        spec_i = np.asarray(
            mc_spectrum(segments, E_grid_eV, crystal, hkl_list, n_hat=n_i, groove=groove, **kwargs)
        )
        contrib = float(w_i) * spec_i
        total = contrib if total is None else total + contrib
    return total
