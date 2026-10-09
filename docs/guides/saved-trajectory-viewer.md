# Inspect saved electron showers

The optional saved viewer opens an authoritative trajectory HDF5 capture with PyVista. It selects complete histories or tracks before loading coordinates, then keeps the full selection while display filters hide or clip individual segments. The existing live Plotly preview remains available.

## Install and launch

Install the optional dependencies in the environment where rendering runs:

```bash
# Checkout; ordinary PyRITE does not require VTK/trame.
uv sync --extra trajectory-viewer
uv run --extra trajectory-viewer pyrite app trajectories inspect capture.h5
uv run --extra trajectory-viewer pyrite app trajectories launch capture.h5 --history 0 --history 1
```

For an installed distribution, install `pyrite-xray[trajectory-viewer]` and run the same `pyrite` commands. `inspect` needs no rendering dependencies. Root and nested CLI help never start a renderer.

Repeat `--history ID` or `--track ID` to select complete identities. Combining them selects their intersection. IDs are captured numeric IDs; histories group primaries and their descendants, while tracks distinguish individual secondary paths. Old captures without track metadata support history selection and explicitly report unknown track ancestry. Select event-specific arrays with repeatable `--attribute`, for example `--attribute hard_secondary_E_eV`. Available attributes are recorded in the HDF5 transport group. Identity, energy/time controls and event-kind/channel identifiers are retained automatically when available. Grooved vacuum diagnostic legs remain available through the existing VTP export; this viewer displays material transport segments only because vacuum legs lack unambiguous track ancestry.

The default admission bounds are one million selected segments and an estimated 2048 MiB working set. Nothing silently truncates a track. Increase `--max-segments` and `--memory-mib` deliberately after inspecting the size. The estimate includes a conservative renderer allowance and selected arrays/copies; it is **not** an enforced process RSS ceiling or an out-of-core promise. The server retains loaded geometry. If admission fails, reduce the history/track selection or raise the explicit limits. If the OS terminates a renderer for memory pressure, restart with fewer complete histories/tracks; the capture remains read-only.

## Browser controls and inspection

`launch` starts a trame server bound to `127.0.0.1:2722`, without automatically opening a browser. Open the printed URL. Camera interaction renders on the server and streams images; it does not transfer the shower's full geometry to the client.

Use minimum energy, maximum generation and maximum time sliders to filter displayed segments; each shows its current value. A slider applies its value when you release it (or after each arrow/Home/End key), so dragging across a large scene does not queue a refilter per step. Time is the recorded segment start age plus the captured entrance-time offset, converted from light-distance angstrom to femtoseconds, as in the trajectory data convention. Missing energy/time attributes disable their controls. With **Clip** on, the plane slider keeps the portion of each segment on the lab `x >= plane` side. Its range spans the loaded selection in the **current view's units** (angstrom or mm); switching scale recentres it.

Click a displayed track to inspect the captured segment's history/track/parent/generation and requested event attributes. The track is outlined in magenta across its full loaded length, even where filters hide part of it. The inspector reports complete and displayed segment counts and whether segment filters or plane clipping make the track partial. Changing filters or scale keeps the picked segment and refreshes its display status. Dragging to rotate does not change the selection, and a click on empty space leaves it unchanged. **Inspect parent** moves to the parent's full loaded track even if display filters hide it; it reports a primary, unknown legacy ancestry, or a parent outside the bounded selection instead. **Clear** removes the selection. **Reset camera** restores the isometric view of the current scale.

Choose **closeup** for lab-axis angstrom tracks and a selection-fitted target crop, or **instrument** for lab-axis millimetres and recorded target/filter/detector context. Tracks are rescaled with the view; angstrom and mm coordinates are never merged. Scene metadata, poses and geometry come from the capture, not the current catalog. Legacy captures have no invented downstream filters or detector. The same shared geometry owner builds VTK scene exports and the viewer's geometry-only blocks.

## Export

```bash
uv run --extra trajectory-viewer pyrite app trajectories export capture.h5 shower.png --history 0
uv run --extra trajectory-viewer pyrite app trajectories export capture.h5 orbit.gif --history 0 --frames 30 --fps 10
uv run --extra trajectory-viewer pyrite app trajectories export capture.h5 instrument.png --scale instrument
```

CLI PNG and GIF exports are 1000 × 700 pixels; browser exports use the browser view's current size. GIF is a camera orbit of the current scene, with 2–120 frames and 1–30 fps; it does not simulate new transport or advance the captured shower. The CLI exports the full bounded selection, while browser/desktop exports preserve the current display filters and camera. Every output has an adjacent `<output>.json` recording capture identity, provenance, selected stored-row ranges, fields, frame, units, sample transform, scale, filters and image size; GIF also records frame count/rate. Existing output or sidecar requires `--overwrite`. A temporary directory holds the complete export before publication; image and JSON replacements are individually atomic, not a transactional pair.

Browser **Screenshot** and **Orbit GIF** (30 frames, 10 fps) download the rendered file to the browser as `pyrite-trajectories.png` or `pyrite-trajectories-orbit.gif`. Each also keeps a uniquely named server copy and its provenance sidecar under `trajectory-exports/` (configurable with `--output-dir`); the **Export ready** status names both and the file size. A failed export reports its error and starts no download. The GIF takes several seconds on large scenes, during which the view does not update.

## Desktop, headless and remote hosts

`launch --native` opens a desktop VTK window. Energy/generation/time sliders use the same filter owner. PyVista's picking gesture outlines the track and prints captured identity to the terminal. Keys: `s` screenshot, `m` orbit GIF, `c` close-up, `i` instrument, `x` toggle the lab x=0 clipping plane, `p` inspect the selected parent. Desktop interaction requires a working graphical display; use PNG export or the browser server on a headless host.

Headless PNG/GIF export and the browser server use an offscreen render window. VTK still needs a usable OpenGL renderer: current wheels can use EGL/OSMesa, while other installations may require an X display/Xvfb. Software rendering can be selected with `LIBGL_ALWAYS_SOFTWARE=1` on Mesa. Set `MPLCONFIGDIR` to a writable directory if matplotlib's optional rendering dependency cannot write its cache. WSL needs WSLg for a desktop window or an operational offscreen software/accelerated renderer. Changing the backend does not make a missing graphics runtime work.

On an SSH host:

```bash
# Rendering host
pyrite app trajectories launch capture.h5 --history 0 --port 2722
# Client machine
ssh -L 2722:127.0.0.1:2722 HOST
# Open http://127.0.0.1:2722 locally.
```

The localhost server and SSH tunnel are the supported single-user deployment. Multi-user/public hosting requires a separately configured trame launcher/proxy and authentication. Captures and exports remain on the rendering host.

## Measured limits

[Saved-viewer measurements](../repo-design/storage/trajectory-scenes.md#repeated-workstation-and-remote-measurements) retain three fresh trials each at about 119k and 1.01 million synthetic segments, locally and through an actual SSH tunnel. At 1.01 million, energy updates appeared in 0.4–1.5 s and instrument scale-switch medians ranged from 1.4 to 5.3 s across measured deployments. Sampled server RSS was 1.4–1.7 GiB, with 1.1–1.2 GiB summed Chrome-process RSS. The admission estimate is not a hard memory cap.

One bounded native and one connected-browser trial also completed at roughly 10 million segments on the lab host; these are exploratory successes, not a production ceiling. WSLg selected AMD by default; the NVIDIA arm explicitly used `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` and verified the RTX 5080 renderer. Device-wide VRAM samples include other users/desktop allocations and do not attribute memory to a process. Larger inputs repeat one shower's geometry and overdraw. Independent large captures, other networks, per-process VRAM and interactive human review remain open. These measurements use the existing linear energy colormap.
