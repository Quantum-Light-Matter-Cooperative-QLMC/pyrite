# Persistent shell completion

PyRITE provides Click tab-completion for Bash, zsh, and Fish. Completion has
two persistent parts:

1. `pyrite` must be an executable on `PATH` in every shell session.
2. The shell must load PyRITE's generated completion script.

`uv sync` creates the project environment, and `uv run pyrite ...` exposes its
entry point only for that command. Installing completion through
`uv run pyrite completion install` does not make the bare `pyrite` command
persistent, so the completion installer rejects that project-only state unless
another `pyrite` executable is already available outside the active environment.
Activating `.venv` manually in each session is not the recommended user
installation.

## Install the persistent command

From a PyRITE checkout, install the distribution as an
[isolated uv tool](https://docs.astral.sh/uv/guides/tools/#installing-tools):

```bash
uv tool install .
uv tool update-shell
exec "$SHELL"
```

`uv tool install` keeps PyRITE and its dependencies in a dedicated environment
while placing its console scripts in uv's tool executable directory. The
command is then available without activating the repository `.venv`.
The example installs the base CPU distribution; select the same accelerator
extra documented in the [README installation
matrix](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite#install)
when the persistent command should use a GPU backend.

For a development checkout whose persistent command should immediately follow
source changes, use an editable tool installation:

```bash
uv tool install --editable .
uv tool update-shell
exec "$SHELL"
```

The tool environment is for interactive command availability. Use the locked
project environment and canonical `uv run pyrite-dev ...` commands for tests,
linting, documentation builds, and other contributor verification.

## Install completion

After `command -v pyrite` succeeds in a new shell, install completion:

```bash
pyrite completion install
exec "$SHELL"
```

Pass the shell explicitly when `$SHELL` does not identify the intended shell:

```bash
pyrite completion install --shell zsh
```

The command generates a completion script once under PyRITE's platform user
data directory, then adds a small managed source block to the shell's standard
startup file. This follows [Click's cached-script
guidance](https://click.palletsprojects.com/en/stable/shell-completion/#enabling-completion)
and avoids invoking PyRITE to regenerate the registration script during every
shell startup. `--completion-file` and `--rc-file` override the two paths when
a managed environment requires custom locations.

Rerunning the command refreshes the generated script and does not duplicate the
managed startup block. Run it again after an upgrade if the shell registration
format needs refreshing.

## zsh behavior

By default, zsh installation updates `$ZDOTDIR/.zshrc` when `ZDOTDIR` is set,
or `~/.zshrc` otherwise. The managed block initializes zsh's completion system
with `compinit` only when `compdef` is not already available, then sources the
generated PyRITE script. Existing zsh frameworks that initialize completion
earlier therefore retain ownership of their completion configuration.

Check the installation in a fresh shell:

```zsh
command -v pyrite
whence -w _pyrite_completion
```

The first command must report uv's persistent tool executable. The second
should report `_pyrite_completion: function` after `.zshrc` has loaded.

## Remove completion

Remove both the managed startup block and generated script with:

```bash
pyrite completion remove
```

This does not uninstall PyRITE. To remove the persistent uv tool afterward:

```bash
uv tool uninstall pyrite-xray
```

## Troubleshooting

If a new shell reports `command not found: pyrite`, run `uv tool update-shell`
and restart the shell. Do not solve this by repeatedly activating the project
virtual environment unless that environment is intentionally managed by the
shell configuration.

If zsh completion was installed before this cached-script implementation,
rerun `pyrite completion install --shell zsh`. The installer replaces the
older managed `eval "$(_PYRITE_COMPLETE=zsh_source pyrite)"` block.
