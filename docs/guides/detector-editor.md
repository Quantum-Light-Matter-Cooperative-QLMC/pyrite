# Terminal detector editor (prototype)

`pyrite tui` opens a keyboard-driven terminal editor for named detectors. It
is an optional front end over the same services as `pyrite detector`: the
same catalog selection, read-only bundled entries, validation and TOML
serialization. Everything it does can also be done with the CLI, which stays
the supported interface for scripts. Plotting stays in the `pyrite app`
notebooks.

## Install

The editor needs the optional `tui` extra (Textual 8.2.8 or newer; tested
with 8.2.8):

<!-- verify: skip (package installation, not a pyrite command) -->
```bash
pip install "pyrite-mc[tui]"
# or, in a checkout
uv sync --extra tui
```

A checkout's plain `uv sync` already includes it. Without the extra,
`pyrite tui` exits with status 1 and prints that install command. The rest of
PyRITE, including `pyrite -h` and `pyrite tui -h`, never imports Textual.

## Use

```bash
pyrite tui                           # selected catalog, as pyrite detector
pyrite --catalog ./lab-catalog tui   # a complete external catalog
```

The browser lists each detector with its type (`scalar` or `planar`), the
layer it comes from (`bundled`, `user` or `catalog`) and its label. With no
complete catalog selected, bundled detectors are read-only: press `c` to clone
one into the user layer (`catalog.user` / `PYRITE_USER_CATALOG`). A complete
external catalog is edited in place.

| Screen | Keys |
| --- | --- |
| Browse | up/down select, enter or `e` edit, `n` new, `c` clone, `?` help, `q` or ctrl+q quit |
| Editor | tab/shift+tab move, ctrl+r review the diff, escape cancel |
| Review | ctrl+s save, escape back to the editor |

Each field label shows its unit and the matching `pyrite detector` option,
for example `Distance [mm]  --distance-mm`. Inputs use the CLI's domains:
pairs take two numbers (`0.055 0.055`), measured edges are comma-separated,
and a blank field means "use the model default". Field errors appear under
the field as you type; combinations such as a measured range without a bin
width are reported on review. A new detector chooses its type first; fields
of the other type are hidden but keep what you typed if you switch back.

The review screen shows the canonical TOML diff and the equivalent CLI
commands, for example:

```text
pyrite detector create clone lab_chip --from pixelated_reference
pyrite detector set lab_chip --yes --distance-mm 500.0
```

Clones and edits apply only the fields you changed, like `pyrite detector
set`.

## What is written, and when

Nothing is written while you type, cancel, go back from the review, or when
validation fails. Save validates the proposed catalog with the same loader a
run uses, then writes atomically. If the catalog changed since the editor read
it, for example through `pyrite detector` in another shell, Save is refused and
the newer catalog is kept; cancel the edit and open it again to start from it. Quitting (`q`, or
ctrl+q anywhere) and interruption write nothing; an interrupt signal
(SIGINT) makes `pyrite tui` exit with status 130.

`pyrite tui` needs an interactive terminal on both stdin and stdout; under a
pipe it exits with status 2 and points to `pyrite detector`.

## Limits of the prototype

- Detectors only: scalar and planar (pixel) entries. Profiles, beams,
  materials, filters and profile detector membership stay in the CLI.
- No rename or delete; use `pyrite detector rename|delete`, which update or
  protect profile references.
- An existing optional section or field cannot be cleared here; use
  `pyrite detector reset NAME SECTION`.
- One detector per review. There is no multi-object draft, no simulation
  launch, no data download and no plotting.
- Color is decorative: errors carry an `error:` prefix and `NO_COLOR` is
  honored. Very small terminals scroll the form; tested down to 40 x 14.

The editor's usability against the CLI has not yet been evaluated with users.
That evaluation decides whether the editor grows to other catalog objects.
