"""Result derived views: typed provenance and display."""

import pickle

import pytest

from pyrite import CxrSpectra, Provenance, Result, SpectralDensity


def _result(**provenance):
    return Result(
        cxr=CxrSpectra(SpectralDensity([100.0, 300.0], [2.0, 4.0])),
        brem=SpectralDensity([100.0, 200.0, 400.0], [1.0, 1.0, 1.0]),
        characteristic=SpectralDensity([100.0, 300.0], [0.5, 0.5]),
        case={"domega_sr": 0.01, "bunch_charge_pc": 2.0, "rep_rate_hz": 1.0e4},
        provenance=provenance,
    )


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
