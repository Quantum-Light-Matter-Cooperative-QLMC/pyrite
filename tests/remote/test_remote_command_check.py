"""Remote check tests."""

import pytest

from pyrite import remote
from pyrite.cli.commands import _remote_actions
from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.remote import (  # noqa: F401
    _queue_scripts,
    config,
    lifecycle,
    scripts,
    state,
    transport,
    viewer,
)
from tests.remote._remote_test_helpers import (
    _remote_main,
)


def test_check_materials_accepts_crystal_keys():
    remote._check_materials(["mose2", "hopg", "mos2-on-sio2-si", "silicon"])


def test_check_materials_rejects_shell_safe_unknown_catalog_keys():
    with pytest.raises(SystemExit, match="unknown material"):
        remote._check_materials(["not_in_catalog"])


@pytest.mark.parametrize("bad", ["rm -rf /", "a;b", "../etc", "a b", "", "m&n"])
def test_check_materials_rejects_injection(bad):
    with pytest.raises(SystemExit):
        remote._check_materials([bad])


def test_check_shell_tokens_preserves_checkpoint_and_synthetic_stems():
    remote._check_shell_tokens(["mose2_quick", "zhai", "20260101-000000"])


def test_check_shell_tokens_accepts_at_stems():
    """@-stems must survive the remote shell-token gate so pull/prune/slim can
    name them on the box (regression for the 2026-07-29 stem migration; ``@``
    was previously rejected by ``_SHELL_TOKEN_RE``)."""
    from pyrite.campaign.profiles import named_profile_stem

    at_stem = named_profile_stem("hopg", "full", catalog_profile="sub_100keV")
    remote._check_shell_tokens([at_stem])  # must not raise


def test_check_cli_pull_flag_skips_run(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "pull_zhai_cache", lambda: calls.append("pull"))
    monkeypatch.setattr(
        cli, "remote_check", lambda **kw: pytest.fail("--pull must not run the reproduction")
    )

    _remote_main(["check", "--pull"])

    assert calls == ["pull"]


def test_check_cli_detached_starts_queue(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **kw: calls.append(kw) or "jid")
    monkeypatch.setattr(viewer, "attach", lambda jobid: pytest.fail("no --follow: must not attach"))

    _remote_main(["check", "--detached", "--ne", "11"])

    assert calls[0]["ne"] == 11


def test_check_cli_detached_follow_attaches(monkeypatch):
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **kw: "jid")
    attached = []
    monkeypatch.setattr(viewer, "attach", attached.append)

    _remote_main(["check", "--detached", "--follow"])

    assert attached == ["jid"]


def test_check_cli_foreground_calls_remote_check(monkeypatch):
    calls = []
    monkeypatch.setattr(_remote_actions, "remote_check", lambda **kw: calls.append(kw))

    _remote_main(["check", "--ne", "11", "--refresh"])

    assert calls == [
        {
            "ne": 11,
            "ne_brem": 200,
            "ne_supp": 200,
            "tmd_azimuth": 0.0,
            "refresh": True,
            "no_sync": False,
        }
    ]


def test_check_cli_rejects_follow_without_detached(monkeypatch, capsys):
    monkeypatch.setattr(cli, "remote_check", lambda **kw: pytest.fail("must reject before running"))

    assert _remote_main(["check", "--follow"]) == 2

    assert "--follow requires --detached" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--pull", "--detached"], ["--pull", "--detached", "--follow"]])
def test_check_cli_rejects_pull_with_detached(monkeypatch, capsys, args):
    monkeypatch.setattr(
        lifecycle, "pull_zhai_cache", lambda: pytest.fail("must reject before pulling")
    )

    assert _remote_main(["check", *args]) == 2

    assert "--pull and --detached are mutually exclusive" in capsys.readouterr().err
