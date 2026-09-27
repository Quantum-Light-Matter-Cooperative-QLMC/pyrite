# PyRITE core

- The `pyrite-xray` distribution owns `src/pyrite/`, packaged data, `pyrite`,
  `pyrite-dev`, compatibility executables `cxr`/`cxr-dev`, apps, and tests.
- Read `docs/repo_map.md` before source exploration. Use Serena for definitions/references/call sites; use `rg` or direct reads for exact text and non-code.
- Package data resolve through `pyrite.DATA_DIR`; imports must work from any cwd.
- Keep domain logic in owning `src/pyrite/` modules, Click wiring in `cli/commands/`, and marimo apps thin/output-free.
- Main's `TODO.md` is authoritative; tracked agent work lives in `agentdocs/`,
  with branch detail in `agentdocs/tasks/<full-branch-name>/`.
- Toolchain and dependencies: `mem:tech_stack`. Commands: `mem:suggested_commands`. Design rules: `mem:conventions`. Completion gates: `mem:task_completion`. Remote SLURM RAM limits: `mem:remote_cluster_memory`.
