"""Result derived views: physical-unit flux table, typed provenance, display."""

import pickle

import numpy as np
import pytest
from scipy.constants import elementary_charge

from pyrite import CxrSpectra, Provenance, Result, SpectralDensity
from pyrite.results import FLUX_UNITS


def _result(**provenance):
    return Result(
        cxr=CxrSpectra(SpectralDensity([100.0, 300.0], [2.0, 4.0])),
        brem=SpectralDensity([100.0, 200.0, 400.0], [1.0, 1.0, 1.0]),
        characteristic=SpectralDensity([100.0, 300.0], [0.5, 0.5]),
        case={"domega_sr": 0.01, "bunch_charge_pc": 2.0, "rep_rate_hz": 1.0e4},
        provenance=provenance,
    )


def test_spectral_flux_keeps_native_nodes_and_sums_total():
    table = _result().spectral_flux(units="electron")
    assert table.columns == [
        "photon_energy_eV",
        "photon_energy_keV",
        "cxr",
        "characteristic",
        "brem",
        "total",
    ]
    np.testing.assert_array_equal(table["photon_energy_eV"], [100.0, 200.0, 300.0, 400.0])
    np.testing.assert_array_equal(table["photon_energy_keV"], [0.1, 0.2, 0.3, 0.4])
    # Exact at own nodes, linear between them, zero outside the family's grid.
    np.testing.assert_allclose(table["cxr"], [2.0, 3.0, 4.0, 0.0])
    np.testing.assert_allclose(table["brem"], [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(table["total"], [3.5, 4.5, 5.5, 1.0])


def test_spectral_flux_units_scale_by_current_and_acceptance():
    result = _result()
    raw = result.spectral_flux(units="electron")["total"].to_numpy()
    electrons_per_s = 2.0 * 10.0 * 1e-9 / elementary_charge  # Q[pC] * f[kHz] nA
    expected = {
        "sr": electrons_per_s,
        "detector": electrons_per_s * 0.01,
        "nA": 0.01 * 1e-9 / elementary_charge,
    }
    for units, scale in expected.items():
        np.testing.assert_allclose(
            result.spectral_flux(units=units)["total"].to_numpy(), raw * scale, rtol=1e-14
        )
    assert set(FLUX_UNITS) == {"detector", "nA", "sr", "electron"}
    with pytest.raises(ValueError, match="units"):
        result.spectral_flux(units="W")
    with pytest.raises(ValueError, match="coherent"):
        result.spectral_flux(mode="coherent")


def test_provenance_is_typed_and_keeps_mapping_contract():
    result = _result(identity_digest="abc", backend="cpu", custom=1)
    provenance = result.provenance
    assert isinstance(provenance, Provenance)
    assert provenance.identity_digest == "abc"
    assert provenance.backend == "cpu"
    assert provenance.adaptive_sampling is None
    assert "adaptive_sampling" not in provenance
    assert list(provenance) == ["identity_digest", "backend", "custom"]
    assert provenance["custom"] == provenance.extra["custom"] == 1
    assert provenance == {"identity_digest": "abc", "backend": "cpu", "custom": 1}
    with pytest.raises(KeyError):
        provenance["scene"]
    with pytest.raises(TypeError):
        provenance["backend"] = "cuda"  # type: ignore[index]
    assert pickle.loads(pickle.dumps(provenance)) == provenance
    assert result.scene is None and result.numerics is None


def test_result_display_summarizes_without_arrays():
    result = _result(identity_digest="0123456789abcdef")
    text = repr(result)
    assert "identity 0123456789ab" in text
    assert "brem            3 pts, 100–400 eV" in text
    assert "cxr.coherent    —" in text
    assert "| `characteristic` | 2 pts, 100–300 eV |" in result._repr_markdown_()
