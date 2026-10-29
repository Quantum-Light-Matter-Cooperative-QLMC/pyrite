"""Resolve or generate exact frozen-scene SBETHE inputs; never substitute density."""

import json
from pathlib import Path

from pyrite.xsgen._errors import TableNotFoundError
from pyrite.xsgen.sbethe.catalog import material_inputs_from_composition, resolve_composition_table
from pyrite.xsgen.sbethe.generate import generate_material


def main():
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
    prepared = {}
    for scene in manifest["scenes"]:
        key = scene["catalog_key"]
        if key in prepared:
            continue
        composition = [(scene["element"], scene["number_density_ang3"])]
        generated = False
        try:
            table = resolve_composition_table(key, composition)
        except TableNotFoundError:
            material = material_inputs_from_composition(key, composition)
            result = generate_material(
                material.key,
                material.composition,
                density_g_cm3=material.density_g_cm3,
                mean_excitation_eV=material.mean_excitation_eV,
                band_gap_eV=material.band_gap_eV,
            )
            generated = result.generated
            table = resolve_composition_table(key, composition)
        prepared[key] = {"generated": generated, "table_key": table.key, "digest": table.digest}
    print(json.dumps(prepared, indent=2))


if __name__ == "__main__":
    main()
