"""Per-case transport-layer material resolution: SBETHE tables and shell-mode keys."""


def _case_inelastic_kwargs(case):
    """``simulate_trajectories`` kwargs of a case's opt-in inelastic mode.

    Empty for continuous stopping. The shell soft/hard mode names one catalog
    key per transport layer, resolved like :func:`_case_stopping_table_records`:
    the case crystal for a single slab or the first layer, a radiator's crystal
    for radiator layers; any other absorber layer has no shell model and raises.
    """
    model = case.get("inelastic_model")
    if model is None:
        return {}
    layers = case.get("abs_layers")
    if layers is None:
        materials = [str(case["crystal"])]
    else:
        radiators = case.get("layer_radiators") or [None] * len(layers)
        materials = []
        for index in range(len(layers)):
            radiator = radiators[index] if index < len(radiators) else None
            if radiator is not None:
                materials.append(str(radiator["crystal"]))
            elif index == 0:
                materials.append(str(case["crystal"]))
            else:
                raise ValueError(
                    f"inelastic_model={model!r} needs a catalog material for every "
                    f"layer; absorber layer {index} has none"
                )
    return dict(
        inelastic_model=model,
        inelastic_cutoff_eV=float(case["inelastic_cutoff_eV"]),
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
