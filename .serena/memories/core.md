# cxr-mc core

- One publishable distribution owns `src/cxr_mc/`, packaged data, `cxr`, `cxr-dev`, apps, and tests.
- Read `docs/repo_map.md` before source exploration. Use Serena for definitions/references/call sites; use `rg` or direct reads for exact text and non-code.
- Package data resolve through `cxr_mc.DATA_DIR`; imports must work from any cwd.
- Keep domain logic in owning `src/cxr_mc/` modules, Click wiring in `cli/commands/`, and marimo apps thin/output-free.
- Main's `TODO.md` is authoritative; branch detail belongs in `tasks/<full-branch-name>/`.
- Toolchain and dependencies: `mem:tech_stack`. Commands: `mem:suggested_commands`. Design rules: `mem:conventions`. Completion gates: `mem:task_completion`.
