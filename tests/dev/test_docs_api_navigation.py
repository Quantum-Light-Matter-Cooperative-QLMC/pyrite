"""Static API-reference ownership checks for issue #759."""

import re
import tomllib
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"
REFERENCE = DOCS / "reference" / "python-api"


def test_api_object_page_ownership_and_navigation():
    index = (REFERENCE / "index.md").read_text(encoding="utf-8")
    owners = {}
    for page in REFERENCE.glob("*.md"):
        if page.name == "index.md":
            continue
        assert f"({page.name})" in index or f"\n{page.stem}\n" in index
        content = page.read_text(encoding="utf-8")
        for block in re.findall(r"```\{eval-rst\}\n(.*?)```", content, re.S):
            if ".. autosummary::" not in block:
                continue
            assert ":toctree: ../../_autosummary" in block
            for obj in re.findall(r"^\s{3}(pyrite\.[A-Za-z_][\w.]*)$", block, re.M):
                assert obj not in owners, (obj, owners.get(obj), page.name)
                owners[obj] = page.name

    assert len(owners) >= 123
    for obj in (
        "pyrite.Beam",
        "pyrite.simulate",
        "pyrite.detectors.Detector",
        "pyrite.observations.Observation",
        "pyrite.results.Settings",
        "pyrite.quantum.QuantumSetup",
        "pyrite.storage.CaseArtifactStore",
    ):
        assert obj in owners

    legacy = (DOCS / "api.md").read_text(encoding="utf-8")
    assert ":toctree: _autosummary" not in legacy
    assert "## Physical quantity inputs" in legacy
    assert "## Root import compatibility" in legacy


def test_api_contracts_have_canonical_owners_and_legacy_anchors():
    """Preserve old deep links while making contracts canonical."""
    legacy = (DOCS / "api.md").read_text(encoding="utf-8")
    units = (REFERENCE / "units.md").read_text(encoding="utf-8")
    compatibility = (REFERENCE / "compatibility.md").read_text(encoding="utf-8")

    for heading, owner in (
        ("Canonical domain imports", "compatibility"),
        ("Physical quantity inputs", "units"),
        ("Root import compatibility", "compatibility"),
    ):
        assert f"## {heading}\n" in legacy
        slug = heading.lower().replace(" ", "-")
        assert f"reference/python-api/{owner}.md#{slug}" in legacy
        assert f"## {heading}\n" in (units if owner == "units" else compatibility)

    assert "Pint is included in the core installation." in units
    assert "Pint is included in the core installation." not in legacy
    root_table = "| Root name | Classification | Recommended import | Reason |"
    assert root_table in compatibility
    assert root_table not in legacy
    assert (
        "from pyrite.beams import Beam, TransverseSpot, LongitudinalDistribution" in compatibility
    )
    assert "from pyrite.beams import Beam, TransverseSpot, LongitudinalDistribution" not in legacy


def test_legacy_api_sections_link_to_namespace_contracts():
    """Every retained deep link reaches a section in its canonical page."""
    legacy = (DOCS / "api.md").read_text(encoding="utf-8")
    pointers = re.findall(r"\]\(reference/python-api/([\w-]+)\.md#([\w-]+)\)", legacy)
    assert len(pointers) == 19
    for owner, anchor in pointers:
        content = (REFERENCE / f"{owner}.md").read_text(encoding="utf-8")
        headings = re.findall(r"^#{1,3} (.+)$", content, re.M)
        assert anchor in [heading.lower().replace(" ", "-") for heading in headings]

    assert "(simulation-kernels)=\n## Advanced transport and radiation" in legacy
    assert ".. autosummary::" not in legacy
    assert len(legacy.splitlines()) < 110
    for owner, contract in (
        ("core", 'result.provenance["execution"]'),
        ("transport", "Depth loss is keV/bin"),
        ("radiation", "escaped incoherent PXR/CBS density per incident"),
        ("experimental", "Their aperture-integrated spectra have units"),
    ):
        assert contract in (REFERENCE / f"{owner}.md").read_text(encoding="utf-8")
        assert contract not in legacy


def test_storage_reference_covers_supported_inventory():
    """New supported storage exports must remain discoverable in reference."""
    manifest = tomllib.loads((DOCS.parent / "api-surface.toml").read_text())
    supported = manifest["package_exports"]["pyrite.storage"]
    page = (REFERENCE / "storage.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"^   pyrite\.storage\.(\w+)$", page, re.M))
    assert documented == set(supported)


def test_api_reference_covers_every_governed_target():
    """Every governed object has a generated page, allowing import aliases.

    Compare concrete inventory targets so compatibility spellings do not need
    duplicate pages. Canonical facade imports derived from root exports have
    the same coverage as their implementation or historical package paths.
    """
    manifest = tomllib.loads((DOCS.parent / "api-surface.toml").read_text())
    exports = {f"pyrite.{name}": row for name, row in manifest["root_exports"].items()}
    for module, rows in manifest["package_exports"].items():
        exports.update({f"{module}.{name}": row for name, row in rows.items()})
    for name, row in manifest["root_exports"].items():
        if row["domain"] in manifest["canonical_facades"]:
            exports[f"pyrite.{row['domain']}.{name}"] = row

    documented = {
        name
        for page in REFERENCE.glob("*.md")
        for name in re.findall(r"^   (pyrite\.[\w.]+)$", page.read_text(), re.M)
    }
    covered = {exports[name]["target"] for name in documented}
    missing = {name: row["target"] for name, row in exports.items() if row["target"] not in covered}
    assert not missing, missing
