# General issues/notes

1) change default polar tilt angle in `cxr analysis` from 0 deg to a median value, like 45 deg., for all mats.
2) `cxr remote status` should return the progress bar, `cxr remote attach` should be updated to just be the continuous-monitor version of `status` (and should contain the various `-v/vv` options as well)
3) clean up printed messages to terminal for various `cxr remote <command>` commands (e.g., `cxr remote start` spits out a whole bash script, `cxr remote status -vv` spits out a divide by zero runtime warning which is presumably from something else deeper in repo)
4) # ! IMPORTANT FOR VALIDATION ! # Confirm no more placeholder debye-waller factors remain in any crystals