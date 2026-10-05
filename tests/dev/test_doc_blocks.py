"""Verify the fenced code examples in ``docs/guides/*.md`` stay live.

``docs/conf.py`` loads no doctest/notebook Sphinx extension (see the "verify
documented code blocks" task doc), so nothing else in the documentation build
notices a renamed ``pyrite``/``pyrite-dev`` command, a removed flag, or a
renamed Python API symbol used in a guide. This module is that check:

- every ``bash`` block is resolved against the live Click (``pyrite``) and
  argparse (``pyrite-dev``) command trees, or carries an explicit
  ``<!-- verify: skip (reason) -->`` marker (never both silently);
- the 4 ``python`` blocks in ``python-api-workflow.md`` execute in document
  order as one accumulating namespace (measured cost ~2.5s on CPU -- cheap
  enough to run directly here rather than split into a bind-only fast tier
  plus a separate integration-tier execution pass; see Decision 1 in the task
  doc);
- the ``toml`` block in ``sweep-profiles.md`` gets a lightweight structural
  check (parses, internal beam reference resolves).

No command is ever executed; only ``pyrite``/``pyrite-dev`` command lines are
checked, never run (they mutate workspaces, reach remotes, launch Monte Carlo
runs). The Python blocks are the one tier that *is* executed, deliberately.
"""

from pathlib import Path

import numpy as np
import pytest

from pyrite.devtools.doc_blocks import FencedBlock, extract_fenced_blocks
from pyrite.devtools.doc_cli_check import DocCommandError, check_bash_block, resolve_command
from pyrite.devtools.doc_toml_check import check_toml_block

ROOT = Path(__file__).resolve().parents[2]
GUIDES = ROOT / "docs" / "guides"
GUIDE_FILES = sorted(GUIDES.glob("*.md"))


# --- extractor -----------------------------------------------------------


def test_extractor_reports_accurate_locations_and_skip_markers(tmp_path):
    doc = tmp_path / "example.md"
    doc.write_text(
        "# Title\n"
        "\n"
        "```bash\n"
        "pyrite run standard\n"
        "```\n"
        "\n"
        "<!-- verify: skip (reason here) -->\n"
        "```bash\n"
        "git clone https://example.invalid/repo.git\n"
        "```\n",
        encoding="utf-8",
    )

    blocks = extract_fenced_blocks(doc)

    assert len(blocks) == 2
    first, second = blocks
    assert first.start_line == 3
    assert first.end_line == 5
    assert first.language == "bash"
    assert first.body == "pyrite run standard"
    assert first.skip_reason is None
    assert second.skip_reason == "reason here"


def test_extractor_raises_on_unterminated_fence(tmp_path):
    doc = tmp_path / "broken.md"
    doc.write_text("```bash\npyrite run standard\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unterminated"):
        extract_fenced_blocks(doc)


# --- tier 1: bash blocks checked against the live CLI trees --------------


def _bash_blocks() -> list[FencedBlock]:
    blocks = []
    for path in GUIDE_FILES:
        blocks.extend(b for b in extract_fenced_blocks(path) if b.language == "bash")
    return blocks


@pytest.mark.parametrize("block", _bash_blocks(), ids=lambda b: b.location)
def test_bash_block_is_checked_or_explicitly_skipped(block: FencedBlock):
    errors = check_bash_block(block)
    assert not errors, "\n".join(errors)


def test_guides_contain_bash_blocks():
    # Guards against the parametrized test above silently collecting zero
    # cases (e.g. a glob typo) and passing vacuously.
    assert _bash_blocks(), "No bash blocks collected from docs/guides/*.md"


# --- tier 2: python blocks execute as one accumulating namespace ---------


def _python_blocks() -> list[FencedBlock]:
    path = GUIDES / "python-api-workflow.md"
    return [b for b in extract_fenced_blocks(path) if b.language == "python"]


def test_python_api_workflow_executes_as_one_accumulating_namespace():
    blocks = _python_blocks()
    assert len(blocks) == 7

    namespace: dict[str, object] = {}
    for block in blocks:
        code = compile(block.body, block.location, "exec")
        exec(code, namespace)  # noqa: S102 -- executing the documented example itself

    result = namespace["result"]
    assert result.energy_eV.shape == result.spectrum.shape
    assert result.background_energy_eV.shape == result.background.shape
    assert np.all(np.isfinite(result.energy_eV))
    assert np.all(np.isfinite(result.spectrum))
    assert np.all(np.isfinite(result.background_energy_eV))
    assert np.all(np.isfinite(result.background))

    provenance = result.provenance
    assert namespace["digest"] == provenance["identity_digest"]
    assert namespace["backend"] == provenance["backend"]
    assert namespace["device"] == provenance["device"]
    assert namespace["resolved_case"] == result.case

    detected_line = namespace["detected_line"]
    assert np.shape(detected_line) == result.energy_eV.shape

    # 3 beam energies x 3 tilt angles: analytically derivable from the
    # documented axes, independent of Monte Carlo content.
    assert len(namespace["expanded"]) == 9
    assert len(namespace["cases"]) == 9

    reopened = namespace["reopened"]
    longer = namespace["longer"]
    assert reopened.digest == namespace["configured"].provenance["observation_identity_digest"]
    assert longer.identity.true_spatial_digest == reopened.identity.true_spatial_digest
    np.testing.assert_allclose(
        namespace["longer_image"], 10.0 * namespace["total_image"], rtol=1e-12
    )


# --- toml block ------------------------------------------------------------


def test_sweep_profiles_toml_blocks_are_valid_and_internally_consistent():
    path = GUIDES / "sweep-profiles.md"
    toml_blocks = [b for b in extract_fenced_blocks(path) if b.language == "toml"]
    assert len(toml_blocks) == 2

    errors = [error for block in toml_blocks for error in check_toml_block(block)]
    assert not errors, "\n".join(errors)


# --- step 8: prove the checker actually catches rot -----------------------


def _synthetic_block(body: str) -> FencedBlock:
    return FencedBlock(
        path=Path("synthetic.md"),
        start_line=10,
        end_line=12,
        language="bash",
        body=body,
        skip_reason=None,
    )


@pytest.mark.parametrize(
    ("body", "expected_fragment"),
    [
        ("pyrite frobnicate widget", "unknown subcommand 'frobnicate'"),
        ("pyrite run standard --this-flag-does-not-exist", "unknown option '--this-flag"),
        ("pyrite-dev docs --this-flag-does-not-exist", "unknown option '--this-flag"),
        ("pyrite-dev bogus-command", "unknown subcommand 'bogus-command'"),
        (
            "pyrite-dev performance analyze sub_100keV --bogus-flag",
            "unknown option '--bogus-flag'",
        ),
    ],
)
def test_checker_catches_renamed_commands_and_removed_flags(body: str, expected_fragment: str):
    block = _synthetic_block(body)

    errors = check_bash_block(block)

    assert len(errors) == 1, errors
    assert errors[0].startswith("synthetic.md:11:")
    assert expected_fragment in errors[0]


def test_block_with_no_checkable_command_and_no_skip_marker_fails():
    block = _synthetic_block("git clone https://example.invalid/repo.git")

    errors = check_bash_block(block)

    assert len(errors) == 1
    assert "no pyrite/pyrite-dev command found" in errors[0]
    assert errors[0].startswith("synthetic.md:10:")


def test_block_with_skip_marker_is_exempt_even_if_unchecked():
    block = FencedBlock(
        path=Path("synthetic.md"),
        start_line=10,
        end_line=12,
        language="bash",
        body="git clone https://example.invalid/repo.git",
        skip_reason="environment setup",
    )

    assert check_bash_block(block) == []


def test_resolve_command_rejects_unrecognized_root():
    with pytest.raises(DocCommandError, match="unrecognized command root"):
        resolve_command(["not-pyrite", "run"])
