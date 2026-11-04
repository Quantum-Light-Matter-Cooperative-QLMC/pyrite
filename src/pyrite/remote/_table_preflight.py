"""Table verification command for remote jobs."""

from . import config


def _tables_preflight_block(allow_physics_fallback=False) -> str:
    """Bash that fails the job fast when pinned tables, datasets or sdbase are missing.

    Runs ``pyrite tables verify`` (table manifest and payload digests; the EEDL, EADL and EPDL
    files hashed in full; SBETHE's ``sdbase/``) once per chain, guarded like
    the dependency sync by ``$JOBDIR/.tables_ok``, before any transport runs. A miss records
    ``FAILED (tables)`` and exits 1; the verify output, which names the fix,
    lands in the job log."""
    required = "elsepa,sbethe-tables,eedl,eadl,epdl,sbethe"
    # BremsLib is required by model-aware scan preflight, not this global
    # inventory: explicit EEDL/uncoupled jobs do not consume those tables.
    verify = f"{config.remote_runtime_env()} {config.shell_remote_uv()} run --no-sync pyrite tables verify --require {required}"
    return f"""if [ ! -f "$JOBDIR/.tables_ok" ]; then
  if {verify} >> "$JOBDIR/log" 2>&1; then
    : > "$JOBDIR/.tables_ok"
  else
    echo "FAILED (tables) $(date -Is)" > "$JOBDIR/state"
    exit 1
  fi
fi"""
