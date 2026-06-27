# TODO

**jupyter to marimo.** *in progress on this feature branch (feature/marimo-transfer)*
Here we are exploring the move to marimo from jupyter notebooks,
primarily to improve data visualization and interactivity.
As part of this, we are experimenting with replacing matplotlib with altair
due to the extreme sluggishness and compute-weight of the matplotlib plots

Also covering notebook/repo reorg (PA #3 deferred steps):

3. **Fix `checkpoint_path_for` to anchor to repo root.** Currently uses a relative
   `checkpoints/` default resolved from the kernel cwd — so moving notebooks breaks
   checkpoint lookup. Fix: anchor with `Path(__file__).parents[N]` or an
   `CXR_CHECKPOINT_DIR` env-var override in `src/cxr_mc/run.py`.

4. **Move notebooks to `notebooks/`.** Once step 3 is in place, move
   `scan.ipynb`, `analysis.ipynb` (and their `.md` pairs) from the repo root
   into a `notebooks/` subdir. Update `repo_map.md`, `README.md`, and the
   `notebook` and `repo-orientation` skills to reflect the new paths.
