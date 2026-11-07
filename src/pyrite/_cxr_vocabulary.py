"""Catalog vocabulary adapters; resolved identity payloads keep historical keys."""

import re
from collections.abc import MutableMapping

#: Adaptive-sampling table fields: canonical TOML name -> stored vocabulary
#: (``pyrite._precision.STORED_KEYS``), the keys ``Precision.from_dict`` reads.
ADAPTIVE_SAMPLING_FIELDS = {
    "min_histories": "min_electrons",
    "max_histories": "max_electrons",
    "block_histories": "block_electrons",
    "max_history_share": "max_electron_share",
    "min_effective_histories": "min_effective_electrons",
    "pilot_histories": "pilot_electrons",
    "brem_min_histories": "brem_min_electrons",
    "brem_max_histories": "brem_max_electrons",
}

CATALOG_FIELDS = {
    "cxr_histories": "n_electrons",
    "brem_histories": "n_electrons_brem",
    "adaptive_sampling": "precision",
    **ADAPTIVE_SAMPLING_FIELDS,
    "cxr_grid_policy": "line_grid_policy",
    "E_grid_cxr": "E_grid_line",
    "E_grid_cxr_by_energy": "E_grid_line_by_energy",
}

_CANONICAL_KEYS = {internal: key for key, internal in CATALOG_FIELDS.items()}

#: Removed history-budget spellings (#1007) and the canonical key replacing each.
#: They error on read: an alias would keep two names for one sampling budget.
REMOVED_CATALOG_KEYS = {
    "cxr_trials": "cxr_histories",
    "brem_trials": "brem_histories",
    "cxr_electrons": "cxr_histories",
    "brem_electrons": "brem_histories",
    "n_electrons": "cxr_histories",
    "n_electrons_brem": "brem_histories",
    "precision": "adaptive_sampling",
    **{internal: key for key, internal in ADAPTIVE_SAMPLING_FIELDS.items()},
}


def catalog_key(key: str) -> str:
    """User-facing catalog spelling of a parsed field, for error messages."""
    return _CANONICAL_KEYS.get(key, key)


def _removed_key_message(key: str, prefix: str = "") -> str:
    return f"{prefix}{key}: catalog key was removed; rename it to {REMOVED_CATALOG_KEYS[key]!r}"


def translate_catalog_text(text: str, *, canonical: bool = False) -> str:
    """Rename TOML keys without moving fields or changing unrelated fragments.

    Reading (``canonical=False``) raises ``ValueError`` on a removed key;
    ``canonical=True`` renders the editable view back to canonical names.
    """
    names = {old: new for new, old in CATALOG_FIELDS.items()} if canonical else dict(CATALOG_FIELDS)
    names["line_by_energy"] = "cxr_by_energy"
    removed = () if canonical else REMOVED_CATALOG_KEYS

    def header_field(prefix):
        if not prefix.lstrip().startswith("["):
            return False
        prefix = re.sub(r'"(?:\\.|[^"\\])*"|\'[^\']*\'', "NAME", prefix.strip(" ["))
        parts = [part for part in prefix.split(".") if part]
        return (
            not parts
            or (len(parts) == 2 and parts[0] in {"profiles", "energy_grids", "numerics"})
            or (len(parts) == 4 and parts[0] == "profiles" and parts[2] == "overrides")
            or (len(parts) == 3 and parts[0] == "materials" and parts[2] == "scan")
        )

    def renamed(token, end, prefix):
        """New name of ``token`` when it sits in key position, else ``None``."""
        if token not in names and token not in removed:
            return None
        if not (re.match(r"\s*=", text[end:]) or header_field(prefix)):
            return None
        if token in removed:
            raise ValueError(_removed_key_message(token))
        return names[token]

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
            name = None if len(delimiter) != 1 else renamed(inner, cursor, prefix)
            if name is not None:
                output.append(delimiter + name + delimiter)
                continue
        elif character.isalpha() or character == "_":
            match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[cursor:])
            if match is None:
                cursor += 1
            else:
                token = match[0]
                cursor += len(token)
                prefix = text[text.rfind("\n", 0, start) + 1 : start]
                name = renamed(token, cursor, prefix)
                if name is not None:
                    output.append(name)
                    continue
        else:
            cursor += 1
        output.append(text[start:cursor])
    return "".join(output)


def translate_catalog_fields(value, *, canonical: bool = False, _path: str = ""):
    """Translate mutable catalog tables in place, preserving TOML comments.

    Reject duplicate spellings even if equal; silently choosing one would make
    subsequent profile edits ambiguous. Reading rejects removed keys. This
    adapter never processes persisted run records or identity payloads.
    """
    if isinstance(value, MutableMapping):
        if not canonical:
            for key in value:
                if key in REMOVED_CATALOG_KEYS:
                    raise ValueError(_removed_key_message(key, _path))
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
        for name, child in value.items():
            translate_catalog_fields(child, canonical=canonical, _path=f"{_path}{name}.")
    elif isinstance(value, (list, tuple)):
        for child in value:
            translate_catalog_fields(child, canonical=canonical, _path=_path)
    return value
