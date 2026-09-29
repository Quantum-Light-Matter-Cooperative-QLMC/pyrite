"""The shared Material + Checkpoint picker for the checkpoint-driven apps.

One Checkpoint dropdown names exactly one checkpoint stem: the material's
standard flat run, its blazed run, and every named-profile or ``--quick``
variant, each labelled with its face (for example ``high_energy (a7b2ce) ·
flat``). The dropdown only defaults to the standard flat run; when that is
missing it opens on a placeholder instead of silently loading another profile.
"""

from pathlib import Path

from pyrite.apps._widgets import MaterialSelect
from pyrite.apps.analyze import MaterialMenuRow, checkpoint_stem, face_menu, profile_menu

PLACEHOLDER = ""


def checkpoint_face(material: str | None, stem: str | None) -> str | None:
    """``"blazed"`` for the material's blazed stem, ``"flat"`` for any other stem."""
    if material is None or stem is None:
        return None
    return "blazed" if stem == checkpoint_stem(material, "blazed") else "flat"


def checkpoint_menu(
    material: str | None, checkpoint_dir: Path | str
) -> tuple[MaterialMenuRow, ...]:
    """Every loadable checkpoint for ``material``, standard flat first.

    Rows reuse :func:`face_menu` for the standard flat/blazed stems and
    :func:`profile_menu` for sidecar-identified variants (newest first); only
    checkpoints that exist are listed.
    """
    if not material:
        return ()
    rows: list[MaterialMenuRow] = [
        {
            "value": checkpoint_stem(material, row["value"]),
            "label": f"standard · {row['value']}",
            "disabled": False,
        }
        for row in face_menu(material, checkpoint_dir)
        if not row["disabled"]
    ]
    rows.extend(
        {"value": row["value"], "label": f"{row['label']} · flat", "disabled": False}
        for row in profile_menu(material, checkpoint_dir)
        if row["value"] != material
    )
    return tuple(rows)


def default_checkpoint(material: str | None, menu: tuple[MaterialMenuRow, ...]) -> str | None:
    """The standard flat stem when it is listed, otherwise ``None`` (no silent fallback)."""
    if material is None:
        return None
    stem = checkpoint_stem(material, "flat")
    return stem if any(row["value"] == stem for row in menu) else None


def checkpoint_picker_options(
    material: str | None, checkpoint_dir: Path | str
) -> tuple[list[MaterialMenuRow], str | None]:
    """Picker rows and initial value; a disabled placeholder leads when there is no default."""
    menu = checkpoint_menu(material, checkpoint_dir)
    initial = default_checkpoint(material, menu)
    rows = list(menu)
    if menu and initial is None:
        placeholder: MaterialMenuRow = {
            "value": PLACEHOLDER,
            "label": "— choose a checkpoint —",
            "disabled": True,
        }
        rows.insert(0, placeholder)
        initial = PLACEHOLDER
    return rows, initial


def make_checkpoint_picker(mo, material: str | None, checkpoint_dir: Path | str):
    """The Checkpoint dropdown for ``material`` (an anywidget ``MaterialSelect``)."""
    rows, initial = checkpoint_picker_options(material, checkpoint_dir)
    return mo.ui.anywidget(
        MaterialSelect(
            options=rows,
            value=initial,
            label="Checkpoint",
            disabled=not rows,
        )
    )


def checkpoint_notice(material: str | None, stem: str | None, checkpoint_dir: Path | str):
    """Why nothing is loaded yet, or ``None`` when a checkpoint is selected."""
    if material is None or stem is not None:
        return None
    menu = checkpoint_menu(material, checkpoint_dir)
    count = len(menu)
    if not count or default_checkpoint(material, menu) is not None:
        return None
    noun = "checkpoint" if count == 1 else "checkpoints"
    return (
        f"`{material}` has no standard flat checkpoint. Choose one of its {count} other "
        f"{noun} in the **Checkpoint** menu."
    )
