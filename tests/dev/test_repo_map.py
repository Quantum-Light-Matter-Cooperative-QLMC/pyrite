"""Tests for the generated repository dependency map."""

from pathlib import Path

from cxr_mc.devtools import repo_map


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
