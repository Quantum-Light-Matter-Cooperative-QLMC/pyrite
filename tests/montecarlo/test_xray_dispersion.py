"""First-order Snell line roots and full dispersive coherent escape phase.

Both routes conserve tangential photon momentum on each exit face. The
coherent phase keeps delta(E), so its energy Jacobian differs from the
incoherent frozen-index sinc even though their stationary roots coincide.
Absolute roots, full-vector amplitudes, convergence/critical-angle guards,
and the residual Jacobian are pinned below.
"""

import numpy as np
import pytest

from pyrite.materials.crystal import CRYSTALS, reciprocal_g_vector, refractive_index
from pyrite.montecarlo.spectrum import mc_spectrum
from pyrite.montecarlo.spectrum.lines import _observation_direction
from pyrite.montecarlo.transport import beta_from_keV
from tests.helpers import requires_resolvable_grid

CRYSTAL = "hopg"
HKL = (0, 0, 2)
B_ANG2 = 0.4
E_KEV = 100.0
THICKNESS_ANG = 1e5
HBARC_EV_ANG = 1973.269804

# Long segment -> narrow sinc envelope, so the spectral peak resolves a shift of
# order delta * E_res (~6e-2 eV here) instead of drowning in a ~100 eV wide line.
SEG_LENGTH_ANG = 8000.0
E_GRID = np.linspace(1599.0, 1601.5, 100_001)
# A grid step of 2.5e-5 eV at 1600 eV is a fifth of float32's ulp there, and the
# interference measurements below read an absolute phase of ~3650 rad off a
# single bin -- a claim float32 cannot state at any grid spacing, since eps
# alone moves that phase by 4e-4 rad against a 1e-6 tolerance. The device
# kernels for these two ledger rows are covered by
# tests/montecarlo/test_xray_dispersion_cuda.py instead.
_needs_fp64_grid = requires_resolvable_grid(
    E_GRID, "sub-ulp in-medium line shift and absolute interference phase"
)
# Equality between two code paths does not need that resolution, so the checks
# that only compare paths use a grid every backend can represent.
_COARSE_GRID = np.arange(1599.0, 1601.5, 1.0e-3)

N_HAT = _observation_direction(np.deg2rad(119.0), None)
N_Z = float(N_HAT[2])  # detector looks upstream, out the entrance face
G_Z = float(reciprocal_g_vector(HKL, CRYSTALS[CRYSTAL]["lattice"])[0][2])
BETA = float(beta_from_keV(np.array([E_KEV]))[0])
V_DOT_G = BETA * G_Z
V_DOT_N = BETA * N_Z


def _segments():
    return {
        "r_mid": np.array([[0.0, 0.0, 5000.0]]),
        "v_hat": np.array([[0.0, 0.0, 1.0]]),
        "L_ang": np.array([SEG_LENGTH_ANG]),
        "E_keV": np.array([E_KEV]),
        "t_ang": np.zeros(1),
        "t0_ang": np.zeros(1),
        "elec_id": np.zeros(1, dtype=int),
        "layer": np.zeros(1, dtype=int),
        "Ne": 1,
        "thickness_ang": THICKNESS_ANG,
        "crystal_width_ang": 1e6,
        "crystal_height_ang": 1e6,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
    }


def _single_layer():
    """One-layer absorber stack, which routes the run onto the per-hkl path."""
    info = CRYSTALS[CRYSTAL]
    n_atoms = len(info["basis"]) / info["V_cell"]
    return [(0.0, THICKNESS_ANG, [("C", n_atoms)])]


def _spectrum(**kwargs):
    return mc_spectrum(_segments(), E_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2, **kwargs)


def _vacuum_resonance_eV():
    """``omega_res = v.g / (1 - v.n_hat)``, the retired k = omega root."""
    return HBARC_EV_ANG * V_DOT_G / (1.0 - V_DOT_N)


def _in_medium_resonance_eV(passes=6):
    """Entrance-face Snell root, solved with the pointwise refractive index.

    The root is implicit because ``Re n`` is evaluated at the resonance itself;
    the map contracts at rate ~delta ~ 1e-4 per pass, so a handful of passes is
    already at float64 rounding.
    """
    E = _vacuum_resonance_eV()
    for _ in range(passes):
        n_re = float(refractive_index(CRYSTAL, np.array([E])).real[0])
        E = HBARC_EV_ANG * V_DOT_G / (1.0 - V_DOT_N + (1.0 - n_re) * BETA / N_Z)
    return E


@_needs_fp64_grid
@pytest.mark.parametrize("layers", [None, _single_layer()], ids=["batched", "per_hkl"])
def test_line_sits_on_the_in_medium_resonance_not_the_vacuum_one(layers):
    """Validation: xray-in-medium-resonance.

    Absolute check: the spectral peak lands on the in-medium root, which sits
    ABOVE the vacuum root (the detector looks upstream, ``v.n_hat < 0``, so the
    in-medium denominator is the larger one) by the first-order shift
    ``-delta beta / (n_z (1 - v.n_hat))``.
    """
    spec = _spectrum(layers=layers)
    assert spec.max() > 0.0
    measured = float(E_GRID[spec.argmax()])

    E_vac = _vacuum_resonance_eV()
    E_med = _in_medium_resonance_eV()
    assert E_med > E_vac  # the sign the geometry demands

    # The peak is the in-medium root to well inside one grid step (2.5e-5 eV).
    np.testing.assert_allclose(measured, E_med, rtol=0.0, atol=float(E_GRID[1] - E_GRID[0]))

    # ... and the displacement from the vacuum root is the closed-form shift.
    delta = 1.0 - float(refractive_index(CRYSTAL, np.array([E_med])).real[0])
    np.testing.assert_allclose(
        (measured - E_vac) / E_vac,
        -delta * BETA / N_Z / (1.0 - V_DOT_N),
        rtol=2e-3,
    )


def test_coherent_through_a_layer_stack_is_refused():
    """The per-layer delta along the escape path is not modelled, so the coherent
    layered combination refuses rather than silently dropping the phase."""
    with pytest.raises(NotImplementedError, match="LAYERED absorber"):
        _spectrum(coherent=True, layers=_single_layer())


# --- in-medium coherent propagation phase ---------------------------------
#
# The escape leg accumulates phase at the medium's phase velocity, so segment j
# picks up ``-delta(E) omega(E) L_esc,j`` on top of the vacuum ``omega d_j``.
# Only DIFFERENCES matter, so the observable is the relative phase between two
# segments at different depths -- which is also the design brief's question:
# does a delta ~ 1e-4 accumulate into an order-unity phase over microns?


def _two_segments(z1, z2):
    seg = _segments()
    seg["r_mid"] = np.array([[0.0, 0.0, z1], [0.0, 0.0, z2]])
    seg["v_hat"] = np.tile(seg["v_hat"], (2, 1))
    for key in ("L_ang", "E_keV", "t_ang", "t0_ang"):
        seg[key] = np.tile(seg[key], 2)
    for key in ("elec_id", "layer"):
        seg[key] = np.zeros(2, dtype=int)
    return seg


def _one_segment(z):
    seg = _segments()
    seg["r_mid"] = np.array([[0.0, 0.0, z]])
    return seg


def _relative_phase(z1, z2, i_E, **kwargs):
    """Extract cos(delta phi) between the two segments' fields at grid point i_E.

    Both segments share v, so they share E_res, the sinc envelope, and the
    amplitude A; they differ only in Beer-Lambert escape weight. The coherent
    total is therefore ``I1 + I2 + 2 sqrt(I1 I2) cos(delta phi)`` with the
    single-segment runs supplying I1 and I2.
    """

    def run(seg):
        return mc_spectrum(seg, E_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2, coherent=True, **kwargs)

    i_1 = run(_one_segment(z1))[i_E]
    i_2 = run(_one_segment(z2))[i_E]
    i_both = run(_two_segments(z1, z2))[i_E]
    return (i_both - i_1 - i_2) / (2.0 * np.sqrt(i_1 * i_2))


def _phase_terms(z1, z2, E_eV):
    """Closed-form (vacuum-part, in-medium-part) of the two segments' phase gap.

    Both segments carry the same velocity, age and length, so the whole gap is
    linear in the depth separation:

      far-field retardation   ``omega d_j``     -> -omega n_z (z1 - z2)
      reciprocal harmonic     ``-g.r_j``        -> -g_z (z1 - z2)
      in-medium escape leg    ``-delta omega L`` -> -delta omega (z1 - z2)/(-n_z)

    with ``L_esc = z / (-n_z)`` out the entrance face.
    """
    dz = z1 - z2
    omega = E_eV / HBARC_EV_ANG
    delta = 1.0 - float(refractive_index(CRYSTAL, E_eV).real)
    vacuum_part = -dz * (omega * N_Z + G_Z)
    medium_part = -dz * delta * omega / (-N_Z)
    return vacuum_part, medium_part


@_needs_fp64_grid
@pytest.mark.parametrize("sinc_cutoff", [None, 200.0], ids=["batched", "per_hkl"])
def test_interference_phase_matches_the_in_medium_closed_form(sinc_cutoff):
    """Validation: xray-in-medium-propagation-phase.

    The measured two-segment interference term reproduces the closed-form total
    phase -- geometry, sign, and delta(E) on the OUTPUT grid all pinned -- and
    the in-medium leg is a required part of it: dropping that one term misses
    the measurement by far more than the agreement tolerance.
    """
    kw = dict(sinc_cutoff=sinc_cutoff)
    i_E = int(_spectrum(coherent=True, **kw).argmax())
    E = float(E_GRID[i_E])

    for z2 in (7_000.0, 10_000.0, 12_000.0, 14_900.0):
        z1 = 5_000.0
        vacuum_part, medium_part = _phase_terms(z1, z2, E)
        measured = _relative_phase(z1, z2, i_E, **kw)
        np.testing.assert_allclose(measured, np.cos(vacuum_part + medium_part), atol=1e-6)
        # The in-medium leg is not a rounding-level correction here.
        assert abs(np.cos(vacuum_part) - measured) > 1e-3


@_needs_fp64_grid
def test_micron_scale_depth_separation_inverts_the_interference():
    """The design brief's headline question, answered by the spectrum itself.

    delta ~ 1e-4 is negligible per Angstrom, but over ~1 micron of depth
    separation it reaches pi: constructive interference between the two segments
    becomes destructive. That is the whole reason this model exists.
    """
    z1, z2 = 5_000.0, 5_000.0 + 9_900.0  # escape paths ~1 micron apart
    # Read the interference on the line core (+-0.2 eV of a ~1 eV wide line),
    # at the bin where the vacuum part -- turning ~2.4 rad/eV here -- is
    # farthest from a node, so the sign flip below is unambiguous.
    peak = int(_spectrum(coherent=True).argmax())
    step = float(E_GRID[1] - E_GRID[0])
    core = range(peak - int(0.2 / step), peak + int(0.2 / step) + 1, 400)
    i_E = max(core, key=lambda i: abs(np.cos(_phase_terms(z1, z2, float(E_GRID[i]))[0])))
    E = float(E_GRID[i_E])
    vacuum_part, medium_part = _phase_terms(z1, z2, E)
    assert abs(abs(medium_part) - np.pi) < 0.05  # this geometry is tuned to half a cycle

    # A half-cycle shift flips the interference term's sign, whatever the rest of
    # the phase happened to be: cos(x +- pi) = -cos(x).
    measured = _relative_phase(z1, z2, i_E)
    assert abs(np.cos(vacuum_part)) > 0.2  # the vacuum-part term is not already ~0
    np.testing.assert_allclose(measured, -np.cos(vacuum_part), atol=0.05)


def _escape_path_resonance_eV(passes=6):
    """Root of ``v = 0`` for the coherent formation factor, by iteration.

    Along the +z flight ``L_esc = z / (-n_z)``, so the in-medium escape leg
    adds ``-delta omega / (-n_z)`` per unit depth to the vacuum phase slope:
    ``omega (1 - v.n_hat - beta delta / (-n_z)) = v.g``. This is the
    Snell-refracted normal wavevector ``k_z = omega (-n_z - delta / (-n_z))``
    of a flat exit face, not the unrefracted bulk ``1 - Re n (v.n_hat)``.
    """
    E = _vacuum_resonance_eV()
    for _ in range(passes):
        delta = 1.0 - float(refractive_index(CRYSTAL, np.array([E])).real[0])
        E = HBARC_EV_ANG * V_DOT_G / (1.0 - V_DOT_N - BETA * delta / (-N_Z))
    return E


@_needs_fp64_grid
def test_single_segment_coherent_line_sits_on_the_escape_path_root():
    """Validation: coherent-formation-absorption.

    One segment's coherent self-term is ``|t_L F|^2`` of the formation integral
    of the phase the coherent sum applies between segments, so its line centre
    is where that phase is stationary along the flight -- the escape-path
    (Snell) root shared by the incoherent route. Absorption
    keeps ``|F|^2`` even in ``v``, so it does not move the peak.
    """
    coh = mc_spectrum(_one_segment(9_000.0), E_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2, coherent=True)
    assert coh.max() > 0.0
    measured = float(E_GRID[coh.argmax()])
    step = float(E_GRID[1] - E_GRID[0])

    E_esc = _escape_path_resonance_eV()
    np.testing.assert_allclose(measured, E_esc, atol=2.0 * step)
    incoh = mc_spectrum(_one_segment(9_000.0), E_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2)
    np.testing.assert_allclose(E_GRID[incoh.argmax()], measured, rtol=0.0, atol=step)


# --- case-dict plumbing ----------------------------------------------------


def test_runner_and_cases_carry_no_dispersion_selector():
    """The model is unconditional, so nothing downstream may still select it."""
    from pyrite.campaign.sweep import Sweep, build_cases
    from pyrite.montecarlo.runner import _lines_for_segments

    case = dict(
        crystal=CRYSTAL,
        hkl_list=[HKL],
        B_ang2=B_ANG2,
        composition=None,
        E_cut_lines_keV=None,
    )
    lines = _lines_for_segments(
        _segments(), _COARSE_GRID, case, _observation_direction(np.deg2rad(119.0), None), None, None
    )
    np.testing.assert_array_equal(
        lines, mc_spectrum(_segments(), _COARSE_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2)
    )

    built = build_cases(Sweep(material="mose2", thickness_ang=1e4))[0]
    assert "xray_dispersion" not in built


# --- Non-convergence of the in-medium root -----------------------------------
#
# The contraction argument behind the three-pass fixed point assumes the X-ray
# regime, ``Re n = 1 - delta`` with ``delta ~ 1e-5-1e-3``. A segment scattered
# nearly perpendicular to ``g`` puts the vacuum root down in the optical/UV,
# where the tabulations honestly carry ``Re n > 1`` (carbon: 6.24-285 eV,
# peaking at 4.766 at 6.40 eV). ``Re n (v.n_hat)`` can then approach unity,
# ``denom`` collapses toward a spurious Cherenkov-like zero, and the map turns
# into a 2-cycle: three passes return whichever half pass three lands on, and a
# keV-scale ``E_res`` comes back attached to a ``v.g`` four orders below the
# median. The CBS amplitude's ``1/(gamma (v.g)^2)`` then produced a line total
# ten orders too large -- finite, so nothing downstream flagged it.
#
# Validation: xray-in-medium-resonance

# One UV node above unity, X-ray nodes at the usual 1-delta. Linear-in-E interp
# between nodes, matching _interp_gather1d.
_CYCLE_E_TAB = np.array([1.0, 6.5, 20.0, 3000.0, 6000.0])
_CYCLE_N_RE = np.array([1.0, 2.05, 1.0, 1.0 - 1e-5, 1.0 - 1e-5])
# Sits just inside the spurious resonance: 2.05 * v_dot_n = 0.99901.
_CYCLE_V_DOT_N = 0.999 / 2.05
# Chosen so the VACUUM root lands exactly on the 6.5 eV node.
_CYCLE_V_DOT_G = 6.5 * (1.0 - _CYCLE_V_DOT_N) / HBARC_EV_ANG


def _root(v_dot_n, v_dot_g):
    from pyrite.montecarlo.spectrum.lines import _in_medium_kinematics

    denom, n_re = _in_medium_kinematics(
        np.array([v_dot_n]), np.array([v_dot_g]), _CYCLE_N_RE, _CYCLE_E_TAB
    )
    return float(np.asarray(denom)[0]), float(np.asarray(n_re)[0])


def test_the_two_cycle_root_is_rejected_rather_than_returned():
    """The pathology itself: a root that oscillates comes back as NaN."""
    denom, _ = _root(_CYCLE_V_DOT_N, _CYCLE_V_DOT_G)
    assert np.isnan(denom)


def test_the_rejected_root_would_otherwise_have_passed_the_energy_window():
    """Why the ``E_res > 10 eV`` cut cannot be the guard.

    ``cbs-amplitude`` leans on that cut to keep ``v.g`` away from zero. It does
    not: the cut bounds ``v.g = omega denom``, so as ``denom -> 0`` it stops
    bounding ``v.g`` at all. Run the unguarded iteration by hand and confirm it
    returns a keV resonance -- comfortably inside the window -- built on a ``v.g``
    of order 1e-3.
    """
    from pyrite.montecarlo.spectrum.lines import _interp_gather1d, _interp_index

    denom = 1.0 - _CYCLE_V_DOT_N
    for _ in range(3):
        E_res = HBARC_EV_ANG * (_CYCLE_V_DOT_G / denom)
        ix, fr, blw, abv = _interp_index(np.array([E_res]), _CYCLE_E_TAB)
        n_re = float(np.asarray(_interp_gather1d(ix, fr, blw, abv, _CYCLE_N_RE))[0])
        denom = 1.0 - n_re * _CYCLE_V_DOT_N
    unguarded_E_res = HBARC_EV_ANG * (_CYCLE_V_DOT_G / denom)
    assert unguarded_E_res > 1e3
    assert _CYCLE_V_DOT_G < 1e-2


def test_a_converged_root_is_untouched_by_the_guard():
    """The control: an X-ray root contracts and must survive unchanged.

    Same tabulation, same ``v.n_hat``; only ``v.g`` moves, enough to put the
    resonance up at keV where ``Re n = 1 - 1e-5``.
    """
    v_dot_g = 3000.0 * (1.0 - _CYCLE_V_DOT_N) / HBARC_EV_ANG
    denom, n_re = _root(_CYCLE_V_DOT_N, v_dot_g)
    assert np.isfinite(denom)
    assert denom == pytest.approx(1.0 - n_re * _CYCLE_V_DOT_N, rel=0, abs=0)
    assert n_re == pytest.approx(1.0 - 1e-5, rel=1e-6)


def test_the_guard_is_inert_across_the_xray_regime():
    """Breadth control: the guard must cost nothing where the premise holds.

    Real hopg dispersion, every direction on the sphere, but the tabulation
    floored at 10 eV so ``Re n < 1`` throughout -- the regime the contraction
    argument was written for. A converged root moves ``denom`` by ~``delta**3``
    on the last pass, orders inside the tolerance, so nothing may be rejected.

    The narrower run-level claim, that the catalog's 1000 Ang production
    thickness rejects no pairs at all, is measured in the ledger rather than
    here; it needs a full transport run.
    """
    g = reciprocal_g_vector(HKL, CRYSTALS[CRYSTAL]["lattice"])[0]
    E_tab = np.geomspace(10.0, 30000.0, 512)
    n_re_tab = np.real(refractive_index(CRYSTAL, E_tab))

    rng = np.random.default_rng(0)
    v = rng.normal(size=(4096, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    v *= beta_from_keV(E_KEV)

    from pyrite.montecarlo.spectrum.lines import _in_medium_kinematics

    v_dot_g = v @ g
    denom, _ = _in_medium_kinematics(v @ N_HAT, v_dot_g, n_re_tab, E_tab)
    E_res = HBARC_EV_ANG * (v_dot_g / np.asarray(denom))
    in_window = np.isfinite(E_res) & (E_res > 10.0) & (E_res < E_tab[-1])
    assert in_window.sum() > 100
    assert np.isfinite(np.asarray(denom)[in_window]).all()


@_needs_fp64_grid
@pytest.mark.parametrize("per_hkl", [False, True], ids=["batched", "per_hkl"])
def test_side_face_line_uses_its_own_snell_root(per_hkl):
    """Validation: xray-in-medium-resonance. Box +x exit conserves z momentum."""
    seg = _one_segment(9000.0)
    seg["r_mid"][0, 0] = 4900.0
    seg["crystal_width_ang"] = 10000.0
    # grad L = (-1/n_x,0,0), so v.grad L = 0 for this z-directed track.
    E_ref = _vacuum_resonance_eV()
    kw = {"sinc_cutoff": 1.0e4} if per_hkl else {}
    for coherent in (False, True):
        spec = mc_spectrum(seg, E_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2, coherent=coherent, **kw)
        assert spec.max() > 0.0
        np.testing.assert_allclose(
            E_GRID[spec.argmax()], E_ref, rtol=0.0, atol=float(E_GRID[1] - E_GRID[0])
        )


def test_normal_exit_snell_root_is_exactly_the_bulk_root():
    from pyrite.montecarlo.spectrum.lines import _in_medium_kinematics

    table = np.array([1e3, 3e3, 1e4])
    index = np.array([0.9999, 0.99998, 0.999999])
    vdn = np.array([-BETA])
    vdg = np.array([V_DOT_G])
    bulk, _ = _in_medium_kinematics(vdn, vdg, index, table)
    snell, _ = _in_medium_kinematics(vdn, vdg, index, table, -vdn, np.ones(1))
    np.testing.assert_array_equal(snell, bulk)


def test_critical_angle_snell_root_is_rejected():
    from pyrite.montecarlo.spectrum.lines import _in_medium_kinematics

    table = np.array([1e3, 2e3, 3e3])
    index = np.full(3, 1.0 - 1e-4)
    denom, _ = _in_medium_kinematics(
        np.zeros(1), np.array([1600 / HBARC_EV_ANG]), index, table, np.zeros(1), np.array([1e4])
    )
    assert np.isnan(denom[0])


@pytest.mark.parametrize("inverse_term", [0.0, 0.2], ids=["constant-index", "dispersive-index"])
def test_dispersive_formation_integral_has_the_derivative_jacobian(inverse_term):
    """Validation: xray-in-medium-resonance, coherent-formation-absorption.

    delta(E)=A+B/E makes E*delta(E) affine exactly. Hence the phase Jacobian
    J=D-E delta' v.gradL differs from the frozen-index width D by B v.gradL/E.
    Parseval predicts the energy-integrated coherent/incoherent ratio D/J.
    """
    from pyrite.montecarlo.spectrum.lines._formation import formation_profile

    E0, duration, Dvac, vgrad, A = 1600.0, 1e5, 1.3, -1.0, 1e-4
    B = inverse_term
    J = Dvac - A * vgrad
    D = Dvac - (A + B / E0) * vgrad
    aJ = J * duration / (2 * HBARC_EV_ANG)
    aD = D * duration / (2 * HBARC_EV_ANG)
    v = np.arange(-10000.0, 10000.0 + 0.005, 0.01)
    energy = E0 + v / aJ
    E_vac = (E0 * J - B * vgrad) / Dvac
    a_vac = Dvac * duration / (2 * HBARC_EV_ANG)
    F = formation_profile(
        energy,
        (A * energy + B) / HBARC_EV_ANG,
        np.array([E_vac]),
        np.array([a_vac]),
        np.array([0.5 * vgrad * duration]),
        2 * np.ones(1),
        np.zeros(1),
        np.zeros(1),
        sinc_cutoff=None,
        xp=np,
    )[0]
    coh = np.trapezoid(abs(F) ** 2, energy) + 1 / (10000 * aJ)
    incoh = np.trapezoid(np.sinc(aD * (energy - E0) / np.pi) ** 2, energy)
    incoh += aJ / (10000 * aD**2)
    np.testing.assert_allclose(coh / incoh, D / J, rtol=1e-8)


@_needs_fp64_grid
@pytest.mark.parametrize("side_face", [False, True, None], ids=["entrance-face", "side-face", "exterior"])
def test_snell_block_and_fused_amplitudes_match_full_vectors(side_face):
    """Validation: xray-in-medium-resonance. Independent vector dot products."""
    from pyrite._backend import _to_cpu
    from pyrite.montecarlo.spectrum.lines._batched import _batched_block, _batched_tables
    from pyrite.montecarlo.spectrum.lines._kernels import _line_amp_sq_core
    from pyrite.montecarlo.spectrum.lines._setup import SpectrumRequest, _prepare_spectrum

    seg = _one_segment(9000.0)
    if side_face is None:
        seg["r_mid"][0, 2] = -9000.0
        seg.update(crystal_width_ang=None, crystal_height_ang=None)
    if side_face:
        seg["r_mid"][0, 0] = 4900.0
        seg["crystal_width_ang"] = 10000.0
    st = _prepare_spectrum(SpectrumRequest(seg, _COARSE_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2))
    bt = _batched_tables(st)
    b = _batched_block(st, bt, slice(None))
    n = np.asarray(_to_cpu(st.n_hat))
    normal = np.array([1.0, 0.0, 0.0]) if side_face else np.array([0.0, 0.0, 1.0])
    h = np.zeros(3) if side_face is None else -normal / (normal @ n)
    for row, g in enumerate(np.asarray(_to_cpu(bt.G))):
        v = np.asarray(_to_cpu(st.v_all))[0]
        omega = float(np.asarray(_to_cpu(b.omega_res))[0, row])
        delta = float(np.asarray(_to_cpu(b.delta))[0, row])
        k = omega * (n + delta * h)
        np.testing.assert_allclose(b.k_dot_v[0, row], k @ v, rtol=1e-14)
        np.testing.assert_allclose(b.k_dot_g[0, row], k @ g, rtol=1e-14)
        np.testing.assert_allclose(b.k_mag[0, row] ** 2, k @ k, rtol=1e-14)
        for e in (np.asarray(_to_cpu(bt.ES))[row], np.asarray(_to_cpu(bt.EP))[row]):
            chi = complex(b.chi_re[0, row], b.chi_im[0, row])
            u = complex(b.u_re[0, row], b.u_im[0, row])
            detuning = (k + g) @ (k + g) - k @ k
            pxr = chi / detuning * ((v @ (k + g)) * ((k + g) @ e) - (k @ k) * (v @ e))
            gamma = float(np.asarray(_to_cpu(b.gamma))[0, 0])
            cbs = (
                -u
                / (gamma * (v @ g))
                * (g @ e - (v @ g) * (v @ e) + (v @ e) * (k @ g - (k @ v) * (v @ g)) / (v @ g))
            )
            got = _line_amp_sq_core(
                chi.real,
                chi.imag,
                u.real,
                u.imag,
                b.v_dot_kg[0, row],
                g @ e,
                b.k_mag[0, row],
                v @ e,
                b.vdg[0, row],
                b.k_dot_g[0, row],
                b.k_dot_v[0, row],
                gamma,
                b.detuning[0, row],
                k @ e,
            )
            np.testing.assert_allclose(
                got, [abs(pxr + cbs) ** 2, abs(pxr) ** 2, abs(cbs) ** 2], rtol=1e-13, atol=1e-30
            )
