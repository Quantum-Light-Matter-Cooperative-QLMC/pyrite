#!/bin/sh
# Disposable Linux CLI exercise. Run in a subshell; keep your own defaults intact.
set -eu
campaign_dir=$(mktemp -d)
trap 'rm -rf "$campaign_dir"' EXIT
export XDG_CONFIG_HOME="$campaign_dir/config"
export PYRITE_USER_CATALOG="$campaign_dir/catalog"
export PYRITE_HOME="$campaign_dir/workspace"
unset PYRITE_CATALOG PYRITE_PROFILE
export NO_COLOR=1

# Clone a complete instrument setup, then narrow its campaign ranges.
pyrite beam create lab-beam --from example --transverse-fwhm-mm 0.1
pyrite profile create lab-scan --from quickstart --beam lab-beam \
  --material hopg --energy 30,60 --thickness 50000
pyrite config set profile.current lab-scan

# Inspect the saved default, the effective ranges, and their sources.
pyrite config list profile
pyrite profile ranges show lab-scan hopg
