"""Tests for the generated repository dependency map."""

from pathlib import Path

from pyrite.devtools import repo_map


def test_strong_components_collapse_cycles_into_dag_nodes() -> None:
    components = repo_map._strongly_connected_components(
        {"leaf", "left", "right", "driver"},
        {
            ("driver", "left"),
            ("left", "right"),
            ("right", "left"),
            ("right", "leaf"),
        },
    )

    assert components == [("driver",), ("leaf",), ("left", "right")]


def test_update_repo_map_replaces_only_generated_region() -> None:
    original = f"before\n{repo_map.BEGIN}\nold\n{repo_map.END}\nafter\n"
    generated = f"{repo_map.BEGIN}\nnew\n{repo_map.END}"

    assert repo_map.update_repo_map(original, generated) == (
        f"before\n{repo_map.BEGIN}\nnew\n{repo_map.END}\nafter\n"
    )


def test_checked_in_dependency_region_is_current() -> None:
    root = Path(__file__).resolve().parents[2]

    assert repo_map.write_or_check(root=root, check=True)


def test_checkpoints_do_not_import_run_drivers() -> None:
    root = Path(__file__).resolve().parents[2]

    _, dependencies = repo_map.package_dependencies(root / "src")

    assert ("checkpoints", "runs") not in dependencies


def test_observations_do_not_import_checkpoint_drivers() -> None:
    root = Path(__file__).resolve().parents[2]

    _, dependencies = repo_map.package_dependencies(root / "src")

    assert ("observations", "checkpoints") not in dependencies


def test_json_report_exposes_exact_edges_and_cycles() -> None:
    report = repo_map._report_from_edges(
        {"campaign", "observations", "checkpoints", "materials"},
        {
            ("pyrite.campaign.model", "pyrite.observations.spec"),
            ("pyrite.observations.inventory", "pyrite.checkpoints.persistence"),
            ("pyrite.checkpoints.campaign_lock", "pyrite.campaign.profiles"),
            ("pyrite.campaign.model", "pyrite.materials.crystal"),
        },
    )

    assert report["schema"] == "pyrite.package-dependencies.v1"
    assert report["cyclic_components"] == [["campaign", "checkpoints", "observations"]]
    assert report["cycle_edges"] == [
        ["campaign", "observations"],
        ["checkpoints", "campaign"],
        ["observations", "checkpoints"],
    ]
    assert report["summary"]["largest_component_size"] == 3
    assert {edge["importer"] for edge in report["module_edges"]} == {
        "pyrite.campaign.model",
        "pyrite.checkpoints.campaign_lock",
        "pyrite.observations.inventory",
    }
