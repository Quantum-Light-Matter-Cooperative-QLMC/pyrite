"""Rewrite a current trajectory artifact into an older schema's layout.

Schema 3 compacts row fields (filters, narrowed IDs, per-history ``t0_ang``)
and adds ``/electron_index``; schema 2 stored every row field plainly at its
transported dtype; schema 1 also lacked the typed case, electron-major rows
and spectrum inputs. ``tests/data/trajectory_artifacts/schema2_hopg_lockstep.h5`` is
a genuine schema-2 capture; this helper produces older layouts of any
transport mapping the tests build.
"""

from pathlib import Path

import h5py
import numpy as np

from pyrite.montecarlo.trajectories import transport_view

SCHEMA2_FIXTURE = (
    Path(__file__).parents[1] / "data" / "trajectory_artifacts" / "schema2_hopg_lockstep.h5"
)


def downgrade(path, schema: int):
    """Rewrite ``path`` in place into the ``schema`` (1 or 2) layout."""
    with h5py.File(path, "r+") as handle:
        if int(handle.attrs["schema_version"]) >= 3:
            _plain_rows(handle)
        handle.attrs["schema_version"] = schema
        if schema == 1:
            for name in ("case/payload", "spectrum_inputs"):
                if name in handle:
                    del handle[name]
            if "row_order" in handle.attrs:
                del handle.attrs["row_order"]
    return path


def _plain_rows(handle):
    """Rebuild ``/transport`` in order with every row field stored plainly."""
    view = transport_view(handle)
    rows = {name: view[name][()] for name in view.row_fields()}
    handle.move("transport", "transport_compact")
    old = handle["transport_compact"]
    new = handle.create_group("transport", track_order=True)
    new.attrs.update(old.attrs)
    written: dict = {}
    for name in old:
        if name not in rows:
            handle.copy(old[name], new, name)
        elif old[name].id in written:
            new[name] = new[written[old[name].id]]  # alias: hard link again
        else:
            dataset = new.create_dataset(name, data=rows[name])
            for key, value in old[name].attrs.items():
                if key not in {"pyrite_dtype", "rows"}:
                    dataset.attrs[key] = value
            dataset.attrs["pyrite_kind"] = "array"
            written[old[name].id] = name
    del handle["transport_compact"]
    if "transport_order" in handle:
        order = handle["transport_order"][()].astype(np.int64)
        del handle["transport_order"]
        handle.create_dataset("transport_order", data=order)
    if "electron_index" in handle:
        del handle["electron_index"]
