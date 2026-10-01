"""Per-case transport-layer material resolution: SBETHE, ELSEPA and BremsLib tables, shell keys."""


def case_shell_materials(case):
    """Catalog key of each transport layer for the shell model, ``None`` where absent.

    The case crystal for a single slab or the first layer, a radiator's
    crystal for radiator layers; any other absorber layer has no catalog key.
    """
    layers = case.get("abs_layers")
    if layers is None:
        return [str(case["crystal"])]
    radiators = case.get("layer_radiators") or [None] * len(layers)
    materials = []
    for index in range(len(layers)):
        radiator = radiators[index] if index < len(radiators) else None
        if radiator is not None:
            materials.append(str(radiator["crystal"]))
        else:
            materials.append(str(case["crystal"]) if index == 0 else None)
    return materials


def _case_inelastic_kwargs(case):
    """``simulate_trajectories`` kwargs of a case's shell soft/hard inelastic mode.

    Empty for continuous stopping. The shell soft/hard mode names one catalog
    key per transport layer (:func:`case_shell_materials`); an absorber layer
    without one raises.
    """
    model = case.get("inelastic_model")
    if model is None:
        return {}
    materials = case_shell_materials(case)
    for index, material in enumerate(materials):
        if material is None:
            raise ValueError(
                f"inelastic_model={model!r} needs a catalog material for every "
                f"layer; absorber layer {index} has none"
            )
    return dict(
        inelastic_model=model,
        inelastic_cutoff_eV=float(case["inelastic_cutoff_eV"]),
        # Opt-in secondary transport (#94); absent keeps primary-only rows.
        **(
            {"secondary_threshold_eV": float(case["secondary_threshold_eV"])}
            if case.get("secondary_threshold_eV") is not None
            else {}
        ),
        inelastic_materials=materials,
    )


def _case_stopping_table_records(case):
    """Resolve identity-matched SBETHE table records for all transport layers."""
    from ...xsgen.sbethe import resolve_composition_table

    layers = case.get("abs_layers")
    if layers is None:
        return [resolve_composition_table(str(case["crystal"]), case["composition"])]

    radiators = case.get("layer_radiators") or [None] * len(layers)
    tables = []
    for index, (_, _, composition) in enumerate(layers):
        radiator = radiators[index] if index < len(radiators) else None
        key = str(case["crystal"]) if index == 0 else None
        if radiator is not None:
            key = str(radiator["crystal"])
        table = resolve_composition_table(
            key if key is not None else f"{case['crystal']}:layer-{index}", composition
        )
        tables.append(table)
    return tables


def _case_stopping_tables(case):
    """Load SBETHE stopping arrays for all transport layers."""
    return [table.arrays() for table in _case_stopping_table_records(case)]


def _case_layer_compositions(case):
    layers = case.get("abs_layers")
    if layers is None:
        return [case["composition"]]
    return [composition for _, _, composition in layers]


def _case_elastic_entries(case):
    """Per-layer ELSEPA table entries, or ``None`` for the default elastic model."""
    if case.get("elastic_model") != "elsepa":
        return None
    from ...xsgen.elsepa.catalog import resolve_layer_tables

    return [resolve_layer_tables(composition) for composition in _case_layer_compositions(case)]


def _case_elastic_kwargs(case):
    """``simulate_trajectories`` kwargs of a case's opt-in elastic model; empty by default."""
    entries = _case_elastic_entries(case)
    if entries is None:
        return {}
    return dict(
        elastic_model="elsepa",
        elastic_tables=[[entry.arrays for entry in layer] for layer in entries],
    )


def _case_elastic_table_records(case):
    """Stored ELSEPA tables a case reads, for run identity; empty by default."""
    entries = _case_elastic_entries(case)
    if entries is None:
        return []
    return [table for layer in entries for entry in layer for table in entry.tables]


def _case_bremslib_tables(case):
    """Staged BremsLib tables for every element of a case's layers, keyed by symbol.

    ``None`` for the default EEDL continuum. Elements outside the catalogue
    (or whose table is not fetched) are absent from the mapping, so their
    emission falls back to isotropic EEDL with a warning.
    """
    if case.get("bremsstrahlung_model") != "bremslib":
        return None
    from ...xsgen.bremslib.tables import load_bremsstrahlung_tables

    elements = [
        str(element)
        for composition in _case_layer_compositions(case)
        for element in _composition_elements(composition)
    ]
    return load_bremsstrahlung_tables(elements)


def _case_radiative_kwargs(case):
    """``simulate_trajectories`` kwargs for a coupled radiative case.

    Empty for uncoupled scoring. The coupled mode reads the same BremsLib
    tables as the continuum scorer, so transport and spectrum agree on the
    table identity recorded on the rows.
    """
    model = case.get("radiative_model")
    if model is None:
        return {}
    return dict(
        radiative_model=model,
        radiative_cutoff_eV=float(case["radiative_cutoff_eV"]),
        bremslib_tables=_case_bremslib_tables(case),
    )


def _composition_elements(composition):
    return [row[0] for row in composition]


def _case_bremslib_table_records(case):
    """BremsLib tables a case reads, for run identity; empty by default."""
    tables = _case_bremslib_tables(case)
    return [] if tables is None else list(tables.values())
