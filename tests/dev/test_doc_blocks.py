"""Verify guide commands, including relocated material/cluster contracts.

``docs/conf.py`` loads no doctest/notebook Sphinx extension (see the "verify
documented code blocks" task doc), so nothing else in the documentation build
notices a renamed ``pyrite``/``pyrite-dev`` command, a removed flag, or a
renamed Python API symbol used in a guide. This module is that check:

- every guide ``bash`` block (plus the relocated contracts) is resolved against
  the live Click (``pyrite``) and
  argparse (``pyrite-dev``) command trees, or carries an explicit
  ``<!-- verify: skip (reason) -->`` marker (never both silently);
- the legacy Python example's downloadable source executes in the explicit slow tier;
- the canonical catalog schema's ``toml`` blocks get a lightweight structural
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
COMMAND_FILES = GUIDE_FILES + [
    ROOT / "docs/reference/material-authoring.md",
    ROOT / "docs/reference/cluster-execution.md",
]


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


@pytest.mark.parametrize(
    ("info", "language"),
    [("{figure} figures/schematic.svg", "{figure}"), ('python title="Example"', "python")],
)
def test_extractor_consumes_fence_arguments_before_next_example(tmp_path, info, language):
    doc = tmp_path / "directives.md"
    doc.write_text(
        f"```{info}\nFirst block\n```\n\n```bash\npyrite --help\n```\n",
        encoding="utf-8",
    )

    first, second = extract_fenced_blocks(doc)

    assert (first.language, first.body, first.start_line, first.end_line) == (
        language,
        "First block",
        1,
        3,
    )
    assert (second.language, second.body, second.start_line, second.end_line) == (
        "bash",
        "pyrite --help",
        5,
        7,
    )


def test_extractor_does_not_close_on_shorter_fence_or_info_string(tmp_path):
    doc = tmp_path / "nested.md"
    doc.write_text(
        "````text\n```bash\npyrite --help\n```\n````python\n`````\n",
        encoding="utf-8",
    )

    (block,) = extract_fenced_blocks(doc)

    assert block.body == "```bash\npyrite --help\n```\n````python"
    assert block.end_line == 6


# --- tier 1: bash blocks checked against the live CLI trees --------------


def _bash_blocks() -> list[FencedBlock]:
    blocks = []
    for path in COMMAND_FILES:
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


@pytest.mark.slow
def test_python_api_workflow_executes_as_one_accumulating_namespace():
    """Execute the fixed example retained for reproduction by the current guide."""
    import runpy

    path = ROOT / "docs/examples/python_api_workflow.py"
    namespace = runpy.run_path(str(path))
    data = namespace["compute"]()
    assert data["seed"] == 1
    assert data["provenance"]["backend"] == "cpu"
    for spectrum in data["spectra"].values():
        energy = np.asarray(spectrum["energy_eV"])
        density = np.asarray(spectrum["density"])
        assert energy.shape == density.shape
        assert np.all(np.diff(energy) > 0)
        assert np.isfinite(density).all() and np.all(density >= 0)


# --- toml block ------------------------------------------------------------


def test_sweep_profiles_toml_blocks_are_valid_and_internally_consistent():
    path = ROOT / "docs/repo-design/materials-catalog-schema.md"
    toml_blocks = [b for b in extract_fenced_blocks(path) if b.language == "toml"]
    assert toml_blocks, "The canonical catalog schema must retain TOML examples"

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
