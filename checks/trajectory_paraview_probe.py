"""Optional pvpython tracks-only cost probe; see trajectory-scenes design guide."""

import json
import resource
import sys
from time import perf_counter

from paraview.simple import (
    ColorBy,
    GetActiveViewOrCreate,
    GetParaViewVersion,
    OpenDataFile,
    Render,
    ResetCamera,
    SaveScreenshot,
    Show,
    Threshold,
)

start = perf_counter()
source = OpenDataFile(sys.argv[1])
source.UpdatePipeline()
loaded = perf_counter()
view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [800, 600]
display = Show(source, view)
ColorBy(display, ("CELLS", "E_start_keV"))
ResetCamera(view)
built = perf_counter()
Render(view)
first = perf_counter()
times = []
for step in range(3):
    tick = perf_counter()
    view.CameraPosition = [value * (1.0 + step * 0.05) for value in view.CameraPosition]
    Render(view)
    times.append(perf_counter() - tick)
tick = perf_counter()
SaveScreenshot(sys.argv[2], view, ImageResolution=[800, 600])
saved = perf_counter() - tick
tick = perf_counter()
if "generation" in source.CellData.keys():
    filtered = Threshold(Input=source)
    filtered.Scalars = ["CELLS", "generation"]
    filtered.LowerThreshold = 1
    filtered.UpperThreshold = 64
    filtered.UpdatePipeline()
    cells = filtered.GetDataInformation().GetNumberOfCells()
else:
    cells = None
report = dict(
    version=str(GetParaViewVersion()),
    input=sys.argv[1],
    load_s=loaded - start,
    build_s=built - loaded,
    first_render_s=first - built,
    repeat_render_s=times,
    screenshot_s=saved,
    filter_s=perf_counter() - tick,
    secondary_cells=cells,
    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
)
with open(sys.argv[3], "w") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report))
