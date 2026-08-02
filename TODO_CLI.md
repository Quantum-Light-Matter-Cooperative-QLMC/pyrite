# CLI-specific TODO List

## Stream-of-Consciousness Thoughts/Questions

We need to examine what has been done by others with similar needs, where many different types of operations can be performed, both locally and remotely. We really need to figure out a consistent command structure:

- What lives top-level, and how is the hierarchy defined
- Which functionalities get a dedicated command vs. a flag
- How do we really properly set the boundaries between a `profile`, a `material`, `energy-grid`, etc., and what things should be mutable vs. immutable and where and when
- How do we structure things for maximal reusability without conflict, making best use of caches wherever possible
- Should remote-related commands go under their own subcommand, e.g., `cxr remote`, or should they be a flag on a standard command, or something else entirely?

It seems like `git` is a good place to start for some of the `profile` and `remote`-related things, but the similarities only go so far. Other heavily validated/public-facing/well-regarded python-based simulation libraries are probably another good source. We need to send out an agent just to do a deep-research session on the generally-accepted best practices -- what do people recommend, and what has been done in well-regarded existing codebases similar to ours.

`remote` is an integral part of MY workflow, so I don't want it sitting around as a hard-to-access dev tool that takes a bunch of typing to use, but it's also not very usable for the general public in its current state.

## High Priority

1. ALL commands that specify parameters like --energy, --thickness, etc., should use the SINGULAR form. Some commands currently have those flags saying '--energies', etc. Needs consistency.
2. We need to add `[coherent|incoherent|both]` to `cxr profile set`
   1. I think we can also put it in `cxr profile add`, and if someone has individually added both, the profile will just automatically switch to using both
3. Should each individual submission type (performance, run/case, energy-grid, rebrem, reline, etc.) have its own submission/pull commands? Or should they be unified? We need to standardize one way or another.
   1. `energy-grid` has its own `submit` but not its own `pull`, `performance` has its own `pull` and `list` but not its own `submit`, while `run` has all three for itself.
4. energy-line allows for azmiuths from 0 to 360 degrees instead of the hard limit set to (90, 270).
5. `cxr energy-grid`:
    1. Why do we have an energy-grid specific `job` command? seems a little pointless. maybe i'm wrong tho.
    2. If this is configured as a material-specific characteristic, e.g., everything here sets a material-specific parameter, then it seems like it should live under `cxr materials`
    3. `regen-golden` has no reason to live here. It should be under `materials` or just top-level
6. Places where commands are forced to be resubmitted as `<command> --yes` should:
    1. Accept shortened `-y`
    2. Provide a prompt [y/N] rather than exiting and demanding resubmission
7. `cxr [remote] run -p` still requires BOTH a single materials AND a profile. This makes genuinely no sense. Just take a profile, or a material, or both, but why limit to single material?
8. We should add `cxr remote fetch`, with similar functionality to `git fetch`, e.g., it will check what profiles are on the remote.
   1. Even if the local doesn't have a copy, of, say, `sub_100keV`, user can still run `cxr remote pull sub_100keV` and it will create a local version of that profile (complete with both the profile configuration and data) if it didn't already exist.
   2. If one did exist locally, but the two configs disagree, there should be a warning to skip or overwrite

## Low Priority

1. `cxr` alone should pull up the help message, just like -h/--help
2. **Block command name reuse as object names** Maybe already implemented, but it seems a possibility that a user might unintentionally name a profile or some other object one of the command names unintentionally when misusing it, e.g., `cxr profile create set hopg` might create a profile named 'set'. Seems an easy-ish thing to just block outright, no duplicating command names to avoid confusion. Low priority.
