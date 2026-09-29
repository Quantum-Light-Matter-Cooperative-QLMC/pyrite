"""Remote checkpoint and validation-cache retrieval."""

import json
import sys
import time
from pathlib import Path

from ..checkpoints import archive
from . import config, scripts, transport


def _resolve_survey_stems(stems):
    """Expand bare canonical-material stems to also include any matching
    identity-qualified survey checkpoint directories on the box
    (``<material>--survey-<hash>/``), so ``pyrite remote pull <material>`` finds
    a survey checkpoint without the caller needing to know its hash suffix.

    On-disk names stay hash-based -- this only discovers them, via one remote
    listing of ``checkpoints/``, matched against the existing
    ``profiles._VARIANT_STEM_RE``. Already-qualified stems (a stem that
    already fullmatches that regex) and ``_quick`` stems pass through
    unexpanded: a stem that already names an exact variant, or a quick smoke
    checkpoint, never has a survey sibling worth auto-discovering."""
    from ..campaign.profiles import _VARIANT_STEM_RE

    bare = {
        stem
        for stem in stems
        if not stem.endswith("_quick") and _VARIANT_STEM_RE.fullmatch(stem) is None
    }
    if not bare:
        return list(stems)
    names = transport._ssh_capture(scripts._list_checkpoint_dirs_command()).split()
    resolved = list(stems)
    for name in sorted(names):
        match = _VARIANT_STEM_RE.fullmatch(name)
        # A survey variant reads as ``--survey-`` (legacy ``fidelity`` group) or
        # ``@survey-`` (@-stem ``label`` group for a standard-profile survey run).
        survey_token = None if match is None else (match["fidelity"] or match["label"])
        if match is None or survey_token != "survey" or match["material"] not in bare:
            continue
        if name in resolved:
            continue
        resolved.append(name)
        print(f"also pulling identity-qualified survey checkpoint -> checkpoints/{name}/")
    return resolved


def _split_profile_selector(stem):
    """Split a ``MATERIAL@PROFILE`` pull selector into ``(material, profile)``,
    or ``None`` for a plain stem (no ``@``, or a full ``@``-stem).

    A resolved on-disk @-stem (``<material>@<label>-<digest>``, matching
    :data:`~pyrite.campaign.profiles._VARIANT_STEM_RE`) is already an exact checkpoint,
    not a profile query, so it passes through literally -- only a bare
    ``MATERIAL@PROFILE`` with no digest tail is treated as a selector to
    resolve via meta.json."""
    from ..campaign.profiles import _VARIANT_STEM_RE

    if "@" not in stem or _VARIANT_STEM_RE.fullmatch(stem) is not None:
        return None
    material, _, profile = stem.partition("@")
    if not material or not profile:
        raise SystemExit(
            f"invalid MATERIAL@PROFILE selector {stem!r}: expected both a material "
            "and a profile name either side of '@'"
        )
    return material, profile


def _remote_meta_json(stem):
    """Fetch and parse one remote checkpoint's ``meta.json`` plus its mtime, or
    ``None`` when the file is absent or unreadable. One ssh round trip: a
    leading ``stat`` line disambiguates "missing" from "empty" without a
    second connection."""
    path = config.remote_path("checkpoints", stem, "meta.json")
    path_q = config.shell_arg(path)
    raw = transport._ssh_capture(f"[ -f {path_q} ] || exit 0; stat -c %Y {path_q}; cat {path_q}")
    mtime_line, _, body = raw.partition("\n")
    if not mtime_line.strip():
        return None
    try:
        mtime = float(mtime_line)
        meta = json.loads(body)
    except ValueError, TypeError, json.JSONDecodeError:
        return None
    return mtime, meta


def _profile_pull_candidates(material):
    """Every on-disk stem for MATERIAL: the bare canonical directory (if
    present) plus every identity-qualified ``<material>--<fidelity>-<hash>``
    variant -- the stem name alone never carries the catalog profile, so this
    only narrows by material; :func:`resolve_profile_stem` reads each
    candidate's meta.json to filter by profile."""
    from ..campaign.profiles import _VARIANT_STEM_RE

    names = set(transport._ssh_capture(scripts._list_checkpoint_dirs_command()).split())
    candidates = [material] if material in names else []
    for name in sorted(names):
        match = _VARIANT_STEM_RE.fullmatch(name)
        if match is not None and match["material"] == material and name not in candidates:
            candidates.append(name)
        elif name.startswith(f"{material}_quick_") and name not in candidates:
            candidates.append(name)
    return candidates


def resolve_profile_stem(
    material, catalog_profile, *, fidelity=None, hash_prefix=None, detector_id=None
):
    """Resolve a ``MATERIAL@PROFILE`` pull selector to one exact on-disk
    checkpoint stem.

    Decision 3/Phase 3 design: on-disk stems stay hash-based and never encode
    the catalog profile, so this reads each candidate directory's
    ``meta.json`` -> ``dataset_identity.catalog_profile`` remotely (one ssh
    round trip per candidate; materials rarely carry more than a handful of
    variants). The newest match wins unless ``hash_prefix`` (``--hash``) pins
    one; ties print every alternative hash so a caller can pin explicitly.
    """
    candidates = _profile_pull_candidates(material)
    found = []
    for stem in candidates:
        fetched = _remote_meta_json(stem)
        if fetched is None:
            continue
        mtime, meta = fetched
        identity = meta.get("dataset_identity") or {}
        profile = identity.get("catalog_profile") or "standard"
        digest = str(identity.get("parameter_sha256", ""))
        found.append(
            (
                stem,
                profile,
                str(identity.get("fidelity", "full")),
                digest,
                mtime,
                identity.get("detector_id"),
            )
        )
    matches = [
        item
        for item in found
        if item[1] == catalog_profile
        and (fidelity is None or item[2] == fidelity)
        and (detector_id is None or item[5] == detector_id)
    ]
    if hash_prefix is not None:
        matches = [item for item in matches if item[3].startswith(hash_prefix)]
    if not matches:
        available = sorted(
            {
                f"{profile}/{found_fidelity}:{digest[:12]}"
                for _, profile, found_fidelity, digest, _, _ in found
            }
        )
        detail = f"; found on the box: {', '.join(available)}" if available else "; none found"
        selector = f"{material}@{catalog_profile}"
        if detector_id is not None:
            selector += f" detector {detector_id}"
        if hash_prefix is not None:
            selector += f" --hash {hash_prefix}"
        raise SystemExit(f"no remote checkpoint matches {selector}{detail}")
    matches.sort(key=lambda item: item[4], reverse=True)
    if len(matches) > 1 and hash_prefix is None:
        alternates = ", ".join(item[3][:12] for item in matches[1:])
        print(
            f"{material}@{catalog_profile}: {len(matches)} hashes found on the box; "
            f"pulling newest ({matches[0][3][:12]}); pin another with "
            f"--hash (alternates: {alternates})"
        )
    return matches[0][0]


def pull(
    stems,
    grid=False,
    drop_wide_brem=False,
    downcast=False,
    level9=False,
    no_sync=False,
    dataset=None,
    force=False,
    summary=None,
    hash_prefix=None,
):
    """Fetch checkpoints/<stem>.pkl back from the box for each stem (stem =
    material, or material_quick for a --quick run).

    A bare material stem also picks up any identity-qualified survey
    checkpoint the box holds for it (``<material>--survey-<hash>/``, see
    :func:`_resolve_survey_stems`) -- skipped for a ``dataset`` merge pull
    (``--brem-only``/``--line-only``), since those require an existing local
    checkpoint to merge into and a freshly discovered stem would not have one.

    Every pull runs the transfer through ``pyrite checkpoint slim`` on the box, which encodes
    straight to that ssh session's stdout (``-o -``) -- the box's compress pass
    overlaps the wire instead of staging a whole temp artifact on box disk
    first. ``grid`` filters to just the material's current grid (plus the
    optional byte trimmers); ``level9`` recompresses at the codec's maximum
    level (:data:`_checkpoint_io.MAX_LEVEL`) instead of the default a live sweep
    writes at -- lossless, just smaller for the wire, and only worth it on a
    slow link, since the level costs far more box CPU than it saves bytes.
    ``sync_code()`` runs first (unless ``no_sync``) so the box rebuilds the grid
    from the same ``config.py`` the laptop has, and has the ``-o -`` and
    ``--compresslevel`` flags at all -- closing sync drift."""
    if dataset not in (None, "brem", "line"):
        raise ValueError("dataset must be None, 'brem', or 'line'")
    stems = list(stems)
    if not stems:
        raise ValueError("stems must contain at least one checkpoint stem")
    # MATERIAL@PROFILE selectors split and resolve to an exact stem here,
    # before the shell-token check below -- '@' is not a safe interpolation
    # token, and only the resolved stem ever reaches a remote command.
    selectors = [_split_profile_selector(stem) for stem in stems]
    qualified = [selector for selector in selectors if selector is not None]
    if hash_prefix is not None and len(qualified) != 1:
        raise SystemExit("--hash requires exactly one MATERIAL@PROFILE selector to pull")
    if qualified:
        from ..campaign.profiles import detector_variant
        from ..materials import CATALOG, load_material_catalog

        resolved_stems = []
        for stem, selector in zip(stems, selectors, strict=True):
            if selector is None:
                resolved_stems.append(stem)
                continue
            material, profile = selector
            transport._check_shell_tokens([material])
            try:
                catalog = (
                    CATALOG if profile == "standard" else load_material_catalog(profile=profile)
                )
                detector_ids = tuple(catalog.profile_detector_set(profile))
            except KeyError, ValueError:
                resolved_stems.append(
                    resolve_profile_stem(material, profile, hash_prefix=hash_prefix)
                )
                continue
            if hash_prefix is not None:
                resolved_stems.append(
                    resolve_profile_stem(material, profile, hash_prefix=hash_prefix)
                )
                continue
            for detector_id in detector_ids:
                selected_id = (
                    detector_id if detector_variant(profile, detector_id) is not None else None
                )
                resolved_stems.append(
                    resolve_profile_stem(
                        material,
                        profile,
                        hash_prefix=hash_prefix,
                        detector_id=selected_id,
                    )
                )
        stems = resolved_stems
    transport._check_shell_tokens(stems)
    if dataset is None:
        stems = _resolve_survey_stems(stems)
        transport._check_shell_tokens(stems)
    dest = config.LOCAL_ROOT / "checkpoints"
    dest.mkdir(exist_ok=True)
    use_slim = True  # component directories are projected to one transfer pickle
    if use_slim and not no_sync:
        transport.sync_code()  # box must rebuild the grid from the same config.py
    failed = []
    completed = []
    failure_errors = {}
    for stem in stems:
        local = dest / stem
        try:
            from ..campaign import profiles
            from ..campaign.profiles import identity_from_stem
            from ..checkpoints import _checkpoint_io, _checkpoint_store
            from ..results import merge_dataset
            from ..runs.run import _manifest_save

            # identity_from_stem(stem, dest) covers the bare-canonical/_quick
            # stems and any re-pull with an already-registered local sidecar
            # with no extra ssh call. It cannot, on a stem's FIRST pull,
            # reconstruct a variant whose digest depends on state the catalog
            # alone doesn't carry (a coherent/both emission run, a
            # since-edited named profile -- see identity_from_stem's
            # docstring) -- so for that one case only, fall back to the box's
            # authoritative meta.json (one extra ssh round trip, gated on the
            # variant-stem pattern so canonical/_quick/merge pulls never pay
            # it -- see test_pull_quick_stem_does_not_resolve_survey_siblings
            # and test_pull_dataset_merge_skips_survey_discovery).
            resolved_identity = identity_from_stem(stem, dest)
            if resolved_identity is None and (
                profiles._VARIANT_STEM_RE.fullmatch(stem) or "_quick_" in stem
            ):
                remote_meta = _remote_meta_json(stem)
                if remote_meta is not None:
                    resolved_identity = remote_meta[1].get("dataset_identity")

            flags = ""
            if grid:
                flags += " --grid"
            if drop_wide_brem:
                flags += " --drop-wide-brem"
            if downcast:
                flags += " --downcast"
            if level9:
                flags += f" --compresslevel {_checkpoint_io.MAX_LEVEL}"
            if dataset is not None:
                flags += f" --{dataset}-only"
            ckpt = config.remote_path("checkpoints", stem)
            incoming_local = dest / f".{stem}.incoming.pkl"
            # `-o -` streams the encoded artifact straight down this ssh
            # session's stdout (slim's own report goes to stderr). HDF5 needs
            # random-access output, so the box still stages the container in its
            # own tempdir and only the zstd frame overlaps the transfer -- there
            # is no encode/transfer overlap to claim. The older
            # write-temp-then-cat form staged that temp beside the checkpoint
            # instead; a nonzero slim exit still fails the pull, because ssh
            # propagates the remote command's status.
            remote_transfer = (
                f"cd {config.shell_remote_dir()} && "
                f"{config.remote_runtime_env()} {config.shell_remote_uv()} run --no-sync pyrite checkpoint slim "
                f"{config.shell_arg(ckpt)}{flags} -o -"
            )
            # Timed so a slow pull is attributable: this covers box slim CPU +
            # wire, the local decode+split below is what remains. If the rate
            # here sits near the link speed the wire is the wall; if it sits far
            # below it, the box's compress pass is.
            started = time.monotonic()
            transport._ssh_download(
                remote_transfer, incoming_local, label=f"Pulling checkpoint '{stem}'..."
            )
            elapsed = max(time.monotonic() - started, 1e-9)
            transferred = incoming_local.stat().st_size / 1e6
            print(
                f"transferred {transferred:.1f} MB in {elapsed:.1f}s ({transferred / elapsed:.1f} MB/s)"
            )

            incoming = _checkpoint_io.load(str(incoming_local))
            incoming_local.unlink(missing_ok=True)
            if dataset is not None:
                if not _checkpoint_store.checkpoint_exists(stem, dest):
                    raise FileNotFoundError(f"no local checkpoints/{stem}/ to merge into")
                archive.archive_checkpoint(stem, force=True)
                base = _checkpoint_store.load(stem, dest)
                n_merged, n_skipped = merge_dataset(base, incoming, dataset, force=force)
                components = ("characteristic", "line") if dataset == "line" else (dataset,)
                _checkpoint_store.save(stem, dest, base, components=components)
                _manifest_save(str(local), base, resolved_identity)
                print(
                    f"merged {dataset} ({n_merged} rec, skipped {n_skipped}) -> checkpoints/{stem}/"
                )
            else:
                _checkpoint_store.save(stem, dest, incoming)
                _manifest_save(str(local), incoming, resolved_identity)
                label = "+".join(filter(None, ["grid" if grid else "", "level9" if level9 else ""]))
                detail = f" ({label})" if label else ""
                print(f"pulled{detail} -> checkpoints/{stem}/")
            completed.append(stem)
        except (Exception, SystemExit) as exc:
            failed.append(stem)
            detail = str(exc) or type(exc).__name__
            failure_errors[stem] = detail
            print(
                f"warning: could not pull checkpoint {stem!r}: {detail}; continuing",
                file=sys.stderr,
            )
    if summary is not None:
        summary.update(completed=completed, failed=failed, errors=failure_errors)
    if failed:
        raise SystemExit(
            f"remote pull failed for {len(failed)} of {len(stems)} requested "
            f"checkpoint(s): {', '.join(failed)}"
        )


def pull_zhai_cache():
    """Fetch every cache file under checkpoints/zhai_reproduction/ from the
    box.

    Lists remote filenames first (like clear_remote's listing step) rather
    than `scp -r`, which double-nests the directory when the local destination
    already exists -- listing + per-file scp is unambiguous either way."""
    remote_dir = config.remote_path("checkpoints", "zhai_reproduction")
    remote_dir_word = config.shell_arg(remote_dir)
    listing = (
        f"[ -d {remote_dir_word} ] || exit 0; "
        f"find {remote_dir_word} -maxdepth 1 -type f -name '*.pkl' -printf '%f\\n'"
    )
    names = [Path(p).name for p in transport._ssh_capture(listing).split()]
    if not names:
        print("(no zhai cache files on the box -- run `pyrite run --preset zhai --remote` first)")
        return
    dest = config.LOCAL_ROOT / "checkpoints" / "zhai_reproduction"
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        transport._run(
            config.scp_argv(config.scp_remote_path(f"{remote_dir}/{name}"), str(dest / name)),
            label=f"Pulling {name}...",
        )
    print(f"pulled -> checkpoints/zhai_reproduction/ ({len(names)} cache files)")
