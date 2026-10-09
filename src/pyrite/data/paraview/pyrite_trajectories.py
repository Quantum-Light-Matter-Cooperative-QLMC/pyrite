"""ParaView preset for PyRITE trajectory exports (``.vtp`` or scene ``.vtm``).

Runs under ParaView's Python (``pvpython``/``pvbatch``) or as a GUI macro; it
does not import PyRITE. Print its path with
``pyrite checkpoint export-trajectories --paraview-script``.

Batch / headless::

    pvbatch pyrite_trajectories.py case.vtp --screenshot case.png
    pvbatch pyrite_trajectories.py case.vtm --color generation \\
        --threshold electron_id 0 9 --screenshot closeup.png

GUI: **Macros > Import new macro...** this file, select an opened export in the
Pipeline Browser, then run the macro. Or ``paraview --script=pyrite_trajectories.py``
and open a file when prompted by an empty pipeline.

What it sets up:

* tracks coloured by a cell array (``E_start_keV`` by default; any per-segment
  field such as ``generation``, ``layer``, ``event_kind``, ``t_start_ang``);
* hidden, ready-to-enable Threshold filters on ``generation`` (secondaries),
  ``is_vacuum`` (material segments), and ``electron_id`` (one history);
* for a close-up scene ``.vtm``: target geometry as translucent context, and
  the sibling ``.instrument.vtm`` (mm) in its own render view. The two views
  are never camera-linked: their coordinates differ by 10^7 (angstrom vs mm).
"""

import argparse
import os
import sys
from xml.etree import ElementTree

from paraview import simple

TRACK_BLOCK = "tracks"


def _arguments(argv):
    parser = argparse.ArgumentParser(
        prog="pyrite_trajectories.py", description=__doc__.split("\n")[0]
    )
    parser.add_argument("export", nargs="?", help=".vtp or close-up .vtm; default: active source")
    parser.add_argument("--color", default="E_start_keV", help="cell array to colour by")
    parser.add_argument(
        "--threshold",
        nargs=3,
        action="append",
        default=[],
        metavar=("FIELD", "LOW", "HIGH"),
        help="keep cells with LOW <= FIELD <= HIGH; repeatable",
    )
    parser.add_argument("--screenshot", help="write the close-up view to this PNG and exit")
    parser.add_argument("--size", nargs=2, type=int, default=(1200, 900), metavar=("W", "H"))
    return parser.parse_args(argv)


def _cell_arrays(source):
    return set(source.CellData.keys())


def _tracks(source, path):
    """Track cells of an export: the ``tracks`` block of a scene manifest."""
    if path and path.endswith(".vtm"):
        tracks = simple.ExtractBlock(registrationName="tracks", Input=source)
        tracks.Selectors = [f"/Root/{TRACK_BLOCK}"]
        tracks.UpdatePipeline()
        return tracks
    return source


def _threshold(source, field, low, high, name, *, visible, view):
    threshold = simple.Threshold(registrationName=name, Input=source)
    threshold.Scalars = ["CELLS", field]
    threshold.LowerThreshold = float(low)
    threshold.UpperThreshold = float(high)
    threshold.ThresholdMethod = "Between"
    if visible:
        return threshold
    simple.Hide(threshold, view)
    return threshold


def _colour(display, view, source, field):
    if field not in _cell_arrays(source):
        available = ", ".join(sorted(_cell_arrays(source)))
        raise SystemExit(f"no cell array {field!r}; available: {available}")
    simple.ColorBy(display, ("CELLS", field))
    lut = simple.GetColorTransferFunction(field)
    lut.ApplyPreset("Turbo", True)
    display.RescaleTransferFunctionToDataRange(True, False)
    display.SetScalarBarVisibility(view, True)
    display.LineWidth = 2.0


def _context(source, path, view):
    """Non-track scene blocks as translucent grey context."""
    geometry = simple.ExtractBlock(registrationName="target geometry", Input=source)
    geometry.Selectors = [f"/Root/{name}" for name in _block_names(path) if name != TRACK_BLOCK]
    shown = simple.Show(geometry, view)
    shown.DiffuseColor = [0.75, 0.75, 0.75]
    shown.AmbientColor = [0.75, 0.75, 0.75]
    shown.Opacity = 0.25
    shown.ColorArrayName = ["POINTS", ""]


def _block_names(path):
    """Block names of a scene manifest, read from its XML."""
    return [node.get("name") for node in ElementTree.parse(path).iter("DataSet")]


def _side_view(view):
    """View from +y: depth (lab z, the beam axis) increases left to right."""
    view.CameraPosition = [0.0, 1.0, 0.0]
    view.CameraFocalPoint = [0.0, 0.0, 0.0]
    view.CameraViewUp = [1.0, 0.0, 0.0]
    simple.ResetCamera(view)


def main(argv):
    args = _arguments(argv)
    path = os.path.abspath(args.export) if args.export else None
    source = simple.OpenDataFile(path) if path else simple.GetActiveSource()
    if source is None:
        raise SystemExit("open a PyRITE .vtp/.vtm export, select it, then run this macro")
    if path is None:
        path = getattr(source, "FileName", None)
        path = path[0] if isinstance(path, (list, tuple)) else path
    source.UpdatePipeline()
    view = simple.GetActiveViewOrCreate("RenderView")
    tracks = _tracks(source, path)
    arrays = _cell_arrays(tracks)

    shown = tracks
    for field, low, high in args.threshold:
        shown = _threshold(
            shown, field, low, high, f"{field} {low}..{high}", visible=True, view=view
        )
    display = simple.Show(shown, view)
    if shown is not tracks:
        simple.Hide(tracks, view)
    _colour(display, view, shown, args.color)
    if tracks is not source:
        _context(source, path, view)
        simple.Hide(source, view)

    if "generation" in arrays:
        top = tracks.CellData["generation"].GetRange()[1]
        _threshold(tracks, "generation", 1, max(1, top), "secondaries", visible=False, view=view)
    if "is_vacuum" in arrays:
        _threshold(tracks, "is_vacuum", 0, 0, "material segments", visible=False, view=view)
    first = tracks.CellData["electron_id"].GetRange()[0]
    _threshold(tracks, "electron_id", first, first, "one history", visible=False, view=view)

    view.OrientationAxesVisibility = 1
    view.AxesGrid.Visibility = 1
    _side_view(view)

    instrument = (
        path[: -len(".vtm")] + ".instrument.vtm" if path and path.endswith(".vtm") else None
    )
    if instrument and os.path.exists(instrument) and not args.screenshot:
        layout = simple.GetLayout(view)
        far_view = simple.CreateView("RenderView")
        layout.SplitHorizontal(0, 0.5)
        layout.AssignView(2, far_view)
        far = simple.OpenDataFile(instrument)
        simple.Show(far, far_view).Opacity = 0.6
        far_view.OrientationAxesVisibility = 1
        far_view.AxesGrid.Visibility = 1
        _side_view(far_view)

    if args.screenshot:
        simple.SaveScreenshot(args.screenshot, view, ImageResolution=list(args.size))
    else:
        simple.Render(view)


if __name__ == "__main__":
    # A GUI macro runs with ParaView's own argv; only pvpython/pvbatch pass ours.
    main(sys.argv[1:] if os.path.basename(sys.argv[0]) == os.path.basename(__file__) else [])
