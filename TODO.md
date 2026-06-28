# TODO

**jupyter to marimo.** *in progress on this feature branch (feature/marimo-transfer)*
Here we are exploring the move to marimo from jupyter notebooks,
primarily to improve data visualization and interactivity.
As part of this, we are experimenting with replacing matplotlib with altair
due to the extreme sluggishness and compute-weight of the matplotlib plots

Also covering notebook/repo reorg (PA #3 deferred steps):

3. ~~**Fix `checkpoint_path_for` to anchor to repo root.**~~ Done — `_DEFAULT_CHECKPOINT_DIR` in `src/cxr_mc/run.py` anchors to repo root via `Path(__file__).resolve().parents[2]`.

4. ~~**Move notebooks to `notebooks/`.**~~ Done — `scan.ipynb`, `analysis.ipynb` (+ `.md` pairs) moved to `notebooks/`; `repo_map.md`, `README.md`, `docs/running-on-a-cluster.md`, and the `notebook`/`repo-orientation` skills updated.
