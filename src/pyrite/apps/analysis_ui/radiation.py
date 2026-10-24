"""Read-only component availability and normalization context for spectral views."""

from ...results.radiation import radiation_band_metrics, two_beam_record


def radiation_panel(mo, records):
    available = [r for r in records if r.get("spec_transition") is not None]
    note = (
        f"TR/CTR stored in {len(available)}/{len(records)} cases; absent components contribute zero. "
        "Use TR/CTR alone to isolate it. Map/ranking CXR metrics include all stored additive "
        "components; their historical coherent_flux name does not mean pure coherent PXR/CBS."
    )
    if any(two_beam_record(r) for r in records):
        note += (
            " **Two-beam:** Line = signed joint-field correction, TR/CTR = TR baseline; "
            "their sum is the joint total. Removing the baseline is not DTR off."
        )
    note += " Band-yield table shows up to 12 cases, integrated over each stored line grid without detector/current scaling."
    rows = []
    for record in available[:12]:
        try:
            report = radiation_band_metrics(record)
        except ValueError as error:
            report = {"diagnostic": str(error)}
        rows.append({"case": record["case"].get("label", record["case"].get("name", "")), **report})
    return mo.accordion(
        {
            "Radiation components and band yields": mo.vstack(
                [mo.md(note), mo.ui.table(rows, selection=None)]
            )
        }
    )
