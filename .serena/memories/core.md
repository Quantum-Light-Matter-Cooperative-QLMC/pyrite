# PyRITE core

- `AGENTS.md` is canonical for commands, test selection, work rules, labels,
  and task dispatch; skills live in `.agents/skills`. Do not restate them here.
- Read the `docs/repo_map.md` front page (then the `docs/repo-map/` detail page for the area) before source exploration. Use Serena for
  definitions/references/call sites across the CLI/domain boundary; use `rg`
  or direct reads for exact text, docs, data, and generated files.
- The server starts with `--project-from-cwd`, so each worktree is already
  active; skip `activate_project`, onboarding, and the manual unless
  troubleshooting Serena. Language server: `python_ty`.
- Package data resolve through `pyrite.DATA_DIR`; imports must work from any cwd.
- Remote SLURM RAM policy: `mem:remote_cluster_memory`. Memory upkeep:
  `mem:memory_maintenance`.
