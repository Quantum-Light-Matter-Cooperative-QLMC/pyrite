"""Incoherent layered/grooved CXR on the batched route vs the per-hkl loop (#1194).

The batched ``(n_seg, N_g)`` route reassociates the same float reductions the
single-slab cross-route gate already tolerates (``line-hkl-batch``); the escape
physics -- per-layer exact mu(E_res) over g-independent escape pieces -- must not
move. The per-hkl loop, reachable through ``_policy._BATCH_LAYERED_ESCAPE``,
is the reference.
"""

from dataclasses import replace

import numpy as np
import pytest

import pyrite.montecarlo.spectrum.cxr._policy as _policy
from pyrite._backend import REAL, xp
from pyrite.materials.attenuation import _mu_total_inv_ang
from pyrite.materials.photon_cross_sections import photoelectric_edges
from pyrite.montecarlo import mc_cxr_spectrum
from pyrite.montecarlo.groove import blazed_groove_spec
from pyrite.montecarlo.spectrum.cxr import SpectrumRequest, _needs_per_hkl_route, _prepare_spectrum
from pyrite.montecarlo.spectrum.cxr._escape_mu import exact_mu_table
from tests.helpers import to_device, to_host

# Same cross-route latitude as ``test_spectrum_phases.BATCH_RTOL``.
BATCH_RTOL = max(1e-10, 500.0 * float(np.finfo(REAL).eps))

MOS2 = [("Mo", 0.0213), ("S", 0.0426)]
GRAPHITE = [("C", 0.1136)]
SILICON = [("Si", 0.0499)]
# Film-on-film-on-substrate stack spanning the 4000 Angstrom slab.
LAYERS = [(0.0, 900.0, MOS2), (900.0, 1800.0, SILICON), (1800.0, 4000.0, GRAPHITE)]
HKLS = [(0, 0, 2), (0, 0, 4), (1, 0, 1), (1, 1, 2)]
_TP = np.deg2rad(45.0)
GROOVE = blazed_groove_spec(20_000.0, np.pi / 2, _TP, np.pi)
GROOVE_N_HAT = np.array([np.cos(_TP), 0.0, -np.sin(_TP)])
TILTED_N_HAT = np.array([1.0, 0.0, -0.3]) / np.linalg.norm([1.0, 0.0, -0.3])


def _segments(count=48, thickness=4000.0, seed=3):
    """Fixed, varied straight rows through the whole depth: many distinct lines."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(count, 3)) * [0.25, 0.25, 1.0] + [0.0, 0.0, 2.0]
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = np.column_stack(
        [
            rng.uniform(-2e4, 2e4, count),
            rng.uniform(-2e4, 2e4, count),
            rng.uniform(50.0, thickness - 50.0, count),
        ]
    )
    n_elec = 4
    return {
        "r_mid": r,
        "v_hat": v,
        "L_ang": rng.uniform(10.0, 60.0, count),
        "E_keV": rng.uniform(15.0, 30.0, count),
        "t_ang": np.zeros(count),
        "t0_ang": np.zeros(count),
        "elec_id": np.repeat(np.arange(n_elec), count // n_elec),
        "layer": np.zeros(count, dtype=np.int16),
        "Ne": n_elec,
        "thickness_ang": thickness,
        "n_backscattered": 0,
        "n_missed": 0,
    }


def _both_routes(monkeypatch, **kwargs):
    out = []
    for batched in (True, False):
        monkeypatch.setattr(_policy, "_BATCH_LAYERED_ESCAPE", batched)
        out.append(
            np.asarray(
                mc_cxr_spectrum(_segments(), crystal="hopg", hkl_list=HKLS, B_ang2=0.8, **kwargs)
            )
        )
    return out


def _assert_routes_agree(batched, per_hkl):
    peak = float(max(np.abs(batched).max(), np.abs(per_hkl).max()))
    assert peak > 0.0
    np.testing.assert_allclose(batched, per_hkl, rtol=BATCH_RTOL, atol=BATCH_RTOL * 1e-2 * peak)


@pytest.mark.parametrize("quadrature", ["node", "bin-mean"])
@pytest.mark.parametrize(
    "geometry",
    [
        pytest.param({"layers": LAYERS, "n_hat": TILTED_N_HAT}, id="layered-exit-front"),
        pytest.param(
            {"layers": LAYERS, "n_hat": TILTED_N_HAT * [1.0, 1.0, -1.0]}, id="layered-exit-back"
        ),
        pytest.param({"groove": GROOVE, "n_hat": GROOVE_N_HAT}, id="grooved"),
    ],
)
def test_layered_and_grooved_batched_route_matches_the_per_hkl_loop(
    monkeypatch, quadrature, geometry
):
    grid = np.arange(500.0, 4000.0, 2.0)
    batched, per_hkl = _both_routes(
        monkeypatch, E_grid_eV=grid, line_quadrature=quadrature, **geometry
    )
    _assert_routes_agree(batched, per_hkl)


def test_layered_batched_route_matches_with_components_and_mosaic(monkeypatch):
    grid = np.arange(500.0, 4000.0, 2.0)
    kwargs = dict(
        E_grid_eV=grid,
        layers=LAYERS,
        n_hat=TILTED_N_HAT,
        mosaic_fwhm_rad=2e-3,
        mosaic_nodes=3,
        components=True,
    )
    monkeypatch.setattr(_policy, "_BATCH_LAYERED_ESCAPE", True)
    batched = mc_cxr_spectrum(_segments(), crystal="hopg", hkl_list=HKLS, B_ang2=0.8, **kwargs)
    monkeypatch.setattr(_policy, "_BATCH_LAYERED_ESCAPE", False)
    per_hkl = mc_cxr_spectrum(_segments(), crystal="hopg", hkl_list=HKLS, B_ang2=0.8, **kwargs)
    for b, p in zip(batched, per_hkl, strict=True):
        _assert_routes_agree(np.asarray(b), np.asarray(p))


@pytest.mark.parametrize(
    "over, expected",
    [
        ({"layers": LAYERS}, False),
        ({"groove": GROOVE, "n_hat": GROOVE_N_HAT}, False),
        ({"layers": LAYERS, "coherent": True}, True),
    ],
)
def test_route_choice_batches_incoherent_layered_and_grooved_absorbers(over, expected):
    kw = dict(
        segments=_segments(),
        E_grid_eV=np.arange(500.0, 4000.0, 2.0),
        crystal="hopg",
        hkl_list=HKLS,
        B_ang2=0.8,
        n_hat=TILTED_N_HAT,
    )
    kw.update(over)
    if kw.get("coherent") and "layers" in kw:
        # Setup refuses coherent layers; the route predicate alone is the claim.
        st = _prepare_spectrum(SpectrumRequest(**{**kw, "layers": None}))
        st.request = replace(st.request, layers=LAYERS)
    else:
        st = _prepare_spectrum(SpectrumRequest(**kw))
    assert _needs_per_hkl_route(st) is expected


@pytest.mark.parametrize("composition", [MOS2, SILICON, GRAPHITE, MOS2 + SILICON])
def test_exact_mu_table_reproduces_the_host_attenuation(composition):
    """Union-knot lin-lin folding equals the per-channel EPDL sum, edges included."""
    E = np.geomspace(5.0, 2.0e5, 20_001)
    edges = [e for element, _ in composition for e, _ in photoelectric_edges(element)]
    # Exactly on, and float32-adjacent to, every photoionization edge.
    near = [np.nextafter(np.float32(e), np.float32(s)) for e in edges for s in (0.0, np.inf)]
    E = np.sort(np.concatenate([E, edges, np.asarray(near, dtype=float)])).astype(REAL)
    expected = np.asarray(_mu_total_inv_ang(composition, E.astype(np.float64)), dtype=REAL)

    actual = to_host(exact_mu_table(composition)(to_device(E)))

    np.testing.assert_allclose(actual, expected, rtol=4.0 * float(np.finfo(REAL).eps), atol=0.0)


def test_exact_mu_table_is_nan_outside_the_epdl_band():
    table = exact_mu_table(GRAPHITE)
    out = to_host(table(xp.asarray([0.5, 2.0e11], dtype=REAL)))
    assert np.isnan(out).all()


def _host_mu_tables(request, abs_comp):
    """The pre-#1194 per-hkl escape mu: one host EPDL call per row and layer."""
    comps = [comp for _, _, comp in request.layers] if request.layers is not None else [abs_comp]
    return tuple(
        (lambda E, comp=comp: xp.asarray(_mu_total_inv_ang(comp, E), dtype=REAL)) for comp in comps
    )


@pytest.mark.parametrize(
    "geometry",
    [
        pytest.param({"layers": LAYERS, "n_hat": TILTED_N_HAT}, id="layered"),
        pytest.param({"groove": GROOVE, "n_hat": GROOVE_N_HAT}, id="grooved"),
    ],
)
def test_batched_route_matches_the_per_hkl_loop_on_host_attenuation(monkeypatch, geometry):
    """Independent reference: the per-hkl loop on the host EPDL sum, not the device table."""
    import pyrite.montecarlo.spectrum.cxr._per_hkl as per_hkl_module

    monkeypatch.setattr(per_hkl_module, "escape_mu_tables", _host_mu_tables)
    batched, per_hkl = _both_routes(
        monkeypatch, E_grid_eV=np.arange(500.0, 4000.0, 2.0), **geometry
    )
    _assert_routes_agree(batched, per_hkl)
