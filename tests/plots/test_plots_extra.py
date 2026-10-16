"""Plot backends name the ``plots`` extra when their library is absent."""

import importlib
import sys

import pytest

import pyrite._extras as extras


@pytest.mark.parametrize(
    ("backend", "library"),
    [("mpl", "matplotlib"), ("altair", "altair"), ("plotly", "plotly")],
)
def test_backend_import_without_library_raises_install_hint(monkeypatch, backend, library):
    real_find_spec = extras.find_spec
    monkeypatch.setattr(
        extras, "find_spec", lambda name: None if name == library else real_find_spec(name)
    )
    monkeypatch.delitem(sys.modules, f"pyrite.plots.{backend}", raising=False)

    with pytest.raises(extras.MissingExtraError, match=r"pyrite-mc\[plots\]") as info:
        importlib.import_module(f"pyrite.plots.{backend}")

    assert isinstance(info.value, ImportError)
    assert info.value.missing == (library,)


def test_lazy_plot_name_without_matplotlib_raises_install_hint(monkeypatch):
    import pyrite.plots

    real_find_spec = extras.find_spec
    monkeypatch.setattr(
        extras, "find_spec", lambda name: None if name == "matplotlib" else real_find_spec(name)
    )
    for name in [m for m in sys.modules if m.startswith("pyrite.plots.mpl")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(pyrite.plots.__dict__, "plot_by_energy", raising=False)

    with pytest.raises(extras.MissingExtraError, match="uv sync --extra plots"):
        pyrite.plots.plot_by_energy  # noqa: B018
