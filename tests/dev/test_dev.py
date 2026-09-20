from pyrite import _dev
from pyrite.devtools import dev_cli


def test_dev_entrypoint_delegates_to_dev_cli(monkeypatch):
    calls = []
    monkeypatch.setattr(dev_cli, "main", lambda *args, **kwargs: calls.append((args, kwargs)))

    _dev.main(["lint"])

    assert calls == [((["lint"],), {"prog_name": "pyrite-dev"})]
