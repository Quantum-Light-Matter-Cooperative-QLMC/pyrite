"""Remote checkpoint and completed-job reclamation."""

from ..console import output as _cli_core
from . import config, scripts, state, transport
from . import pull as pulling


def _refuse_if_busy(materials, quick):
    """Abort `start` if a live job is already producing any checkpoint this run
    would write. Two runs writing the same `<stem>.pkl` share one `<stem>.pkl.tmp`
    and race on `os.replace` -- the first rename consumes the temp, the second
    dies with FileNotFoundError (see run.py:_checkpoint_save). Comparing *stems*
    (material, or material_quick) not bare materials lets a `--quick` smoke test
    run alongside a full sweep of the same material, since they write different
    files."""
    wanted = set(scripts._stems(materials, quick))
    busy = [
        (jid, sorted(clash))
        for jid, jquick, jmats in state._live_jobs()
        if (clash := wanted.intersection(scripts._stems(jmats, jquick)))
    ]
    if busy:
        detail = "\n".join(f"  job {jid} is running -> {', '.join(s)}" for jid, s in busy)
        raise SystemExit(
            "refusing to start: a live job is already producing the same "
            "checkpoint(s), and two runs writing one <stem>.pkl race on its "
            f".tmp and crash.\n{detail}\n"
            "monitor it (pyrite job attach <jobid>) or stop it "
            "(pyrite remote stop <material>) first, or run different materials."
        )


def clear_remote(materials, yes=False, catalog_profile="standard"):
    """Delete one or more materials' accumulated checkpoints on the box: the
    legacy ``checkpoints/<material>/`` and ``checkpoints/<material>_quick/``
    plus the current full/survey stems for each standard-profile material, or
    every manifest-confirmed identity
    generation owned by ``catalog_profile``. Accepts a single crystal key or a list.

    Refuses (before touching anything) if a live job or a pre-submission
    reservation protects any stem.  Without ``yes`` this is a safe dry preview:
    it prints exactly which files exist and would be deleted, then stops. With
    ``yes`` it ``rm -f``s them and reports what went."""
    if isinstance(materials, str):
        materials = [materials]
    transport._check_materials(materials)  # interpolated into a remote shell command
    label = f"profile={catalog_profile}" if catalog_profile != "standard" else ", ".join(materials)
    if catalog_profile != "standard":
        current_stems = [
            stem
            for fidelity in ("full", "survey")
            for stem in scripts._stems(
                materials,
                False,
                fidelity,
                catalog_profile=catalog_profile,
            )
        ]
        stems = _profile_checkpoint_stems(catalog_profile, current_stems=current_stems)
    else:
        # Legacy bare stems plus the stems current standard runs write, which
        # are hashed once a profile runs adaptive counts (the #361 default).
        stems = list(
            dict.fromkeys(
                [
                    *(stem for m in materials for stem in (m, f"{m}_quick")),
                    *(
                        stem
                        for fidelity in ("full", "survey")
                        for stem in scripts._stems(materials, False, fidelity)
                    ),
                ]
            )
        )
    wanted = set(stems)
    live_jobs = state._live_jobs()
    live_profiles = (
        state._job_profiles([jobid for jobid, _quick, _materials in live_jobs])
        if catalog_profile != "standard" and live_jobs
        else {}
    )
    if catalog_profile != "standard":
        busy = [
            (jid, sorted(jmats))
            for jid, _jquick, jmats in live_jobs
            if live_profiles.get(jid) == catalog_profile
        ]
    else:
        busy = [
            (jid, sorted(clash))
            for jid, jquick, jmats in live_jobs
            if (clash := wanted.intersection(scripts._stems(jmats, jquick)))
        ]
    if busy:
        detail = "\n".join(f"  job {jid} is producing -> {', '.join(s)}" for jid, s in busy)
        raise SystemExit(
            "refusing to clear: a live job is still producing one of these "
            f"checkpoints, and clearing it would race a running sweep.\n{detail}\n"
            "stop it (pyrite remote stop <material>) first, or wait for it to finish."
        )
    if yes:
        return _clear_exact_remote_stems(stems, label)
    reservations = state._reservation_holders(stems)
    if reservations:
        detail = "\n".join(
            f"  reservation {owner} protects -> {stem}" for stem, owner in reservations
        )
        raise SystemExit(
            "refusing to clear: a checkpoint reservation is still active, which can "
            "belong to a submission whose SLURM outcome is not yet known.\n"
            f"{detail}\n"
            "wait for submission to resolve, or stop the recorded job before clearing."
        )
    # which stems actually exist on the box; the `|| true` keeps a missing last
    # stem's failed `[ -f ]` from becoming the loop's -- and hence ssh's -- exit
    # status, which would make _ssh_capture abort the whole clear
    stem_names = " ".join(stems)
    legacy_names = " ".join(f"{stem}.pkl" for stem in stems)
    listing = (
        f"cd {config.shell_remote_path('checkpoints')} 2>/dev/null || exit 0; "
        f": legacy-candidates {legacy_names}; "
        f"for stem in {stem_names}; do "
        '[ -d "$stem" ] && echo "$stem/" || true; '
        '[ -f "$stem.pkl" ] && echo "$stem.pkl" || true; done'
    )
    existing = transport._ssh_capture(listing).split()
    if not existing:
        print(f"(nothing to clear for {label})")
        return
    print("would delete on the box:")
    for f in existing:
        print(f"  checkpoints/{f}")
    if not _cli_core.confirm_destructive(yes, "Delete these remote checkpoints?"):
        return
    return _clear_exact_remote_stems(stems, label)


def _clear_exact_remote_stems(stems, label):
    """Reserve and delete the exact stems previously selected or previewed."""
    outcome = transport._ssh_capture(
        scripts._clear_checkpoint_stems_command(f"clear-{scripts._new_jobid()}", stems)
    ).splitlines()
    reservations = [line.split("\t", 1)[1] for line in outcome if line.startswith("RESERVED\t")]
    if reservations:
        raise SystemExit(
            "refusing to clear: a checkpoint reservation is still active for "
            f"{', '.join(reservations)}. Wait for submission to resolve, or stop "
            "the recorded job before clearing."
        )
    existing = [line.split("\t", 1)[1] for line in outcome if line.startswith("CLEARED\t")]
    if not existing:
        print(f"(nothing to clear for {label})")
        return
    print("cleared on the box:")
    for name in existing:
        print(f"  checkpoints/{name}")


def _profile_checkpoint_stems(catalog_profile, *, current_stems=()):
    """Return exact remote stems owned by a named catalog profile.

    Current predicted stems remain candidates even before they have a manifest.
    Identity-qualified stems are accepted only when their remote manifest
    confirms profile ownership. Detector IDs may replace the profile in the
    readable ``@`` label, which is never deletion authority by itself.
    """
    from ..campaign.profiles import _VARIANT_STEM_RE

    stems = set(current_stems)
    names = transport._ssh_capture(scripts._list_checkpoint_dirs_command()).split()
    for name in names:
        match = _VARIANT_STEM_RE.fullmatch(name)
        if match is None:
            continue
        remote_meta = pulling._remote_meta_json(name)
        if remote_meta is None:
            continue
        identity = remote_meta[1].get("dataset_identity")
        if isinstance(identity, dict) and identity.get("catalog_profile") == catalog_profile:
            stems.add(name)
    return sorted(stems)


def clear_all_remote(yes=False):
    """Empty the box's ``checkpoints/`` directory: delete every ``*.pkl`` file
    under it (recursively, so per-reproduction subdirs are included too).

    Refuses (before touching anything) if any live job is running or any
    checkpoint reservation is held: both signal an in-flight sweep whose output
    a blanket clear would destroy or race. Without ``yes`` this is a safe dry
    preview: it lists the files that would be deleted, then stops. With ``yes``
    it deletes them and reports the count."""
    live = state._live_jobs()
    if live:
        detail = "\n".join(
            f"  job {jid} is producing -> {', '.join(sorted(scripts._stems(jmats, jquick)))}"
            for jid, jquick, jmats in live
        )
        raise SystemExit(
            "refusing to clear --all: live job(s) are still producing checkpoints, "
            f"and clearing would race running sweeps.\n{detail}\n"
            "stop them (pyrite remote stop --all) first, or wait for them to finish."
        )
    _now, reservations = state._reservation_ledger()
    if reservations:
        detail = "\n".join(
            f"  reservation {jobid} protects -> {stem}" for stem, jobid, _mtime in reservations
        )
        raise SystemExit(
            "refusing to clear --all: a checkpoint reservation is still active, which can "
            "belong to a submission whose SLURM outcome is not yet known.\n"
            f"{detail}\n"
            "wait for submission to resolve, reap orphans (pyrite remote reap), or stop the "
            "recorded job before clearing."
        )
    # list first (dry preview), then delete only under --yes. `|| true` keeps a
    # missing checkpoints/ dir or a find failure from becoming ssh's exit status.
    listing = (
        f"cd {config.shell_remote_path('checkpoints')} 2>/dev/null || exit 0; "
        r'find . -type f \( -name "*.h5" -o -name "*.pkl" \) 2>/dev/null | sed "s|^\./||" | sort || true'
    )
    existing = transport._ssh_capture(listing).split()
    if not existing:
        print("(nothing to clear: checkpoints/ holds no .h5 or .pkl files)")
        return
    print(f"would delete on the box -- {len(existing)} file(s):")
    for f in existing:
        print(f"  checkpoints/{f}")
    if not _cli_core.confirm_destructive(yes, "Delete all remote checkpoint files?"):
        return
    transport._ssh_capture(
        f"cd {config.shell_remote_path('checkpoints')} 2>/dev/null || exit 0; "
        r'find . -type f \( -name "*.h5" -o -name "*.pkl" \) -delete'
    )
    print(f"cleared on the box: {len(existing)} checkpoint file(s) under checkpoints/")


def prune_remote(
    *,
    all_profiles: bool = False,
    catalog_profile: str | None = None,
    yes: bool = False,
):
    """Preview or prune stale records and obsolete profile generations remotely."""
    from ..checkpoints.checkpoint_cleanup import _targets

    targets = _targets(all_profiles, catalog_profile)
    current_stems = [target.stem for target in targets]
    obsolete_stems = []
    if catalog_profile is not None and catalog_profile != "standard":
        profile_stems = _profile_checkpoint_stems(
            catalog_profile,
            current_stems=current_stems,
        )
        obsolete_stems = sorted(set(profile_stems).difference(current_stems))
    stems = sorted(set(current_stems).union(obsolete_stems))
    live = state._live_jobs()
    if live:
        detail = "\n".join(
            f"  job {jobid} is producing -> {', '.join(sorted(scripts._stems(materials, quick)))}"
            for jobid, quick, materials in live
        )
        raise SystemExit(
            "refusing remote prune: live job(s) are producing checkpoints.\n"
            f"{detail}\n"
            "wait for completion, or stop them with pyrite remote stop --all."
        )
    reservations = state._reservation_holders(stems)
    if reservations:
        detail = "\n".join(
            f"  reservation {owner} protects -> {stem}" for stem, owner in reservations
        )
        raise SystemExit(
            "refusing remote prune: checkpoint reservation still active.\n"
            f"{detail}\n"
            "wait for submission to resolve, reap orphans, or stop its recorded job."
        )
    output = _run_remote_prune(
        stems,
        all_profiles=all_profiles,
        catalog_profile=catalog_profile,
        obsolete_stems=obsolete_stems,
        yes=yes,
    )
    if (
        not yes
        and (
            "would prune" in output.lower()
            or "would delete obsolete profile checkpoint" in output.lower()
        )
        and _cli_core.confirm_destructive(False, "Delete these stale remote checkpoint records?")
    ):
        return _run_remote_prune(
            stems,
            all_profiles=all_profiles,
            catalog_profile=catalog_profile,
            obsolete_stems=obsolete_stems,
            yes=True,
        )


def _run_remote_prune(
    stems,
    *,
    all_profiles,
    catalog_profile,
    obsolete_stems,
    yes,
):
    """Run remote pruning against one exact, already-resolved stem selection."""
    output = transport._ssh_capture(
        scripts._prune_checkpoint_stems_command(
            f"prune-{scripts._new_jobid()}",
            stems,
            all_profiles=all_profiles,
            catalog_profile=catalog_profile,
            obsolete_stems=obsolete_stems,
            yes=yes,
        )
    )
    if output:
        print(output)
    return output


def prune_job_dirs(*, profile=None, all_jobs=False, yes=False):
    """Delete terminal (done/FAILED/cancelled) job directories on the box.

    Preview unless ``yes``. Selection is exactly one of ``profile`` (its
    ``NAME``/``NAME-N`` family) or ``all_jobs`` (every job directory). A live
    chain is never removed -- the remote command skips any directory whose
    latest recorded SLURM id is still in squeue -- so this is safe to run while
    other jobs (including a fresh submission of the same profile) are live.
    """
    if all_jobs == (profile is not None):
        raise SystemExit("prune-jobs needs exactly one of --profile NAME or --all")
    if profile is not None:
        transport._check_shell_tokens([profile])
    records = [
        line.split("\t")
        for line in transport._ssh_capture(
            scripts._prune_job_dirs_command(profile=profile, all_jobs=all_jobs, yes=yes)
        ).splitlines()
        if "\t" in line
    ]
    scope = f"profile {profile!r}" if profile is not None else "all profiles"
    live_kept = sorted(r[1] for r in records if r[0] == "KEPT" and r[2] == "live")
    if yes:
        pruned = [r[1] for r in records if r[0] == "PRUNED"]
        if pruned:
            print("pruned job directories on the box:")
            for jid in pruned:
                print(f"  jobs/{jid}")
        else:
            print(f"(no terminal job directories to prune for {scope})")
    else:
        candidates = [(r[1], r[2]) for r in records if r[0] == "WOULD-PRUNE"]
        if candidates:
            print("would delete on the box:")
            for jid, st in candidates:
                print(f"  jobs/{jid}  [{st}]")
            if _cli_core.confirm_destructive(False, "Delete these terminal job directories?"):
                return prune_job_dirs(profile=profile, all_jobs=all_jobs, yes=True)
        else:
            print(f"(no terminal job directories to prune for {scope})")
    if live_kept:
        print(f"kept {len(live_kept)} live job(s): {', '.join(live_kept)}")
