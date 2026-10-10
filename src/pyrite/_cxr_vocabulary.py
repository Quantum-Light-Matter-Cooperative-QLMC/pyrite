"""Catalog vocabulary adapters; resolved identity payloads keep historical keys."""

import re
from collections.abc import MutableMapping

CATALOG_FIELDS = {
    "cxr_trials": "n_electrons",
    "brem_trials": "n_electrons_brem",
    "cxr_grid_policy": "line_grid_policy",
    "E_grid_cxr": "E_grid_line",
    "E_grid_cxr_by_energy": "E_grid_line_by_energy",
}


def translate_catalog_text(text: str, *, canonical: bool = False) -> str:
    """Rename TOML keys without moving fields or changing unrelated fragments."""
    names = {old: new for new, old in CATALOG_FIELDS.items()} if canonical else dict(CATALOG_FIELDS)
    names["line_by_energy"] = "cxr_by_energy"

    def header_field(prefix):
        if not prefix.lstrip().startswith("["):
            return False
        prefix = re.sub(r'"(?:\\.|[^"\\])*"|\'[^\']*\'', "NAME", prefix.strip(" ["))
        parts = [part for part in prefix.split(".") if part]
        return (
            not parts
            or (len(parts) == 2 and parts[0] in {"profiles", "energy_grids"})
            or (len(parts) == 4 and parts[0] == "profiles" and parts[2] == "overrides")
            or (len(parts) == 3 and parts[0] == "materials" and parts[2] == "scan")
        )

    output = []
    cursor = 0
    while cursor < len(text):
        start = cursor
        character = text[cursor]
        if character == "#":
            stop = text.find("\n", cursor)
            cursor = len(text) if stop < 0 else stop
        elif character in {'"', "'"}:
            delimiter = character * (3 if text.startswith(character * 3, cursor) else 1)
            cursor += len(delimiter)
            while cursor < len(text):
                if delimiter[0] == '"' and text[cursor] == "\\":
                    cursor += 2
                elif text.startswith(delimiter, cursor):
                    break
                else:
                    cursor += 1
            inner = text[start + len(delimiter) : cursor]
            cursor = min(len(text), cursor + len(delimiter))
            prefix = text[text.rfind("\n", 0, start) + 1 : start]
            if (
                len(delimiter) == 1
                and inner in names
                and (re.match(r"\s*=", text[cursor:]) or header_field(prefix))
            ):
                output.append(delimiter + names[inner] + delimiter)
                continue
        elif character.isalpha() or character == "_":
            match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[cursor:])
            if match is None:
                cursor += 1
            else:
                token = match[0]
                cursor += len(token)
                prefix = text[text.rfind("\n", 0, start) + 1 : start]
                if token in names and (re.match(r"\s*=", text[cursor:]) or header_field(prefix)):
                    output.append(names[token])
                    continue
        else:
            cursor += 1
        output.append(text[start:cursor])
    return "".join(output)


def translate_catalog_fields(value, *, canonical: bool = False):
    """Translate mutable catalog tables in place, preserving TOML comments.

    Reject duplicate spellings even if equal; silently choosing one would make
    subsequent profile edits ambiguous. This adapter never processes persisted
    run records or identity payloads.
    """
    if isinstance(value, MutableMapping):
        if "line_by_energy" in value:
            if "cxr_by_energy" in value:
                raise ValueError("cannot specify both line_by_energy and cxr_by_energy")
            value["cxr_by_energy"] = value.pop("line_by_energy")
        for new, old in CATALOG_FIELDS.items():
            source, target = (old, new) if canonical else (new, old)
            if source in value:
                if target in value:
                    raise ValueError(f"cannot specify both {old} and {new}")
                value[target] = value.pop(source)
        for child in value.values():
            translate_catalog_fields(child, canonical=canonical)
    elif isinstance(value, (list, tuple)):
        for child in value:
            translate_catalog_fields(child, canonical=canonical)
    return value
