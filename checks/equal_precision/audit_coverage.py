"""Audit PyRITE table coverage for every frozen #510 matrix case; no transport.

Resolves the installed ELSEPA, SBETHE and BremsLib tables for each matrix
material and checks that every case energy and the electron cutoff lie inside
every table. A missing table or uncovered energy stays an explicit gap entry.
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

HERE = Path(__file__).parent


def compositions():
    """Exact number densities of the catalog entries every matrix material uses."""
    from pyrite.xsgen.sbethe.catalog import CATALOG

    return {
        "silicon": ("silicon", tuple(CATALOG.crystal("silicon").composition)),
        "tungsten": ("tungsten", tuple(CATALOG.crystal("tungsten").composition)),
        "diamond": ("diamond", tuple(CATALOG.crystal("diamond").composition)),
        "sio2": ("sio2", tuple(CATALOG.media["sio2"].composition)),
    }


def table_domains(key, composition):
    """Return each code's resolved key/digest and energy domain [keV], or its error."""
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
    from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
    from pyrite.xsgen.sbethe.catalog import resolve_composition_table

    out = {}
    try:
        entries = resolve_layer_tables(list(composition))
        out["elsepa"] = [
            {
                "element": element,
                "key": entry.tables[0].key,
                "min_keV": float(dict(entry.arrays)["energy_eV"].min()) / 1000,
                "max_keV": float(dict(entry.arrays)["energy_eV"].max()) / 1000,
            }
            for (element, _), entry in zip(composition, entries, strict=True)
        ]
    except Exception as error:  # noqa: BLE001 - every failure is retained evidence
        out["elsepa"] = {"error": f"{type(error).__name__}: {error}"}
    try:
        table = resolve_composition_table(key, list(composition))
        arrays = table.arrays()
        out["sbethe"] = {
            "key": table.key,
            "min_keV": float(arrays["stopping_energy_eV"].min()) / 1000,
            "max_keV": float(arrays["stopping_energy_eV"].max()) / 1000,
        }
    except Exception as error:  # noqa: BLE001
        out["sbethe"] = {"error": f"{type(error).__name__}: {error}"}
    elements = [element for element, _ in composition]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        brem = load_bremsstrahlung_tables(elements)
    out["bremslib"] = [
        (
            {
                "element": element,
                "key": brem[element].key,
                "min_keV": float(brem[element].minimum_incident_energy_keV),
                "max_keV": float(brem[element].maximum_incident_energy_keV),
            }
            if element in brem
            else {"element": element, "error": "missing", "warnings": len(caught)}
        )
        for element in elements
    ]
    return out


def case_gaps(case, domains, cutoff_keV):
    """List every reason this case cannot run on the installed tables."""
    gaps = []
    for code, entry in domains.items():
        for item in entry if isinstance(entry, list) else [entry]:
            label = f"{code}:{item.get('element', '')}".rstrip(":")
            if "error" in item:
                gaps.append(f"{label} unavailable")
            elif not item["min_keV"] <= cutoff_keV < case["energy_keV"] <= item["max_keV"]:
                gaps.append(f"{label} domain {item['min_keV']:g}-{item['max_keV']:g} keV")
    physics = case["physics"]
    if physics.get("inelastic_model") == "shell-soft-hard":
        from pyrite.campaign.inelastic_cases import covered_shell_materials

        if case["material"] not in covered_shell_materials():
            gaps.append(f"no conduction-band shell data for {case['material']}")
    if case.get("detector"):
        gaps.append("uncollided detector arm needs the pending weighted per-segment scorer")
    return gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=HERE / "matrix.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix_bytes = args.matrix.read_bytes()
    matrix = json.loads(matrix_bytes)
    domains = {name: table_domains(*spec) for name, spec in compositions().items()}
    cases = []
    for case in matrix["cases"]:
        gaps = case_gaps(case, domains[case["material"]], case["physics"]["electron_cutoff_keV"])
        cases.append({"id": case["id"], "covered": not gaps, "gaps": gaps})
    result = {
        "schema": "pyrite.equal-precision.coverage-audit.v1",
        "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "tables": domains,
        "n_covered": sum(c["covered"] for c in cases),
        "n_cases": len(cases),
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{result['n_covered']}/{result['n_cases']} cases covered -> {args.output}")


if __name__ == "__main__":
    main()
