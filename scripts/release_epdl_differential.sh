#!/usr/bin/env bash
# Build, publish and verify the tables-epdl-differential-1 GitHub release.
# Run from the repository root with a gh login that can create releases.
# Follows docs/repo-design/data-distribution-and-repository-size.md, "Hosted table releases".
set -euo pipefail

REPO=Quantum-Light-Matter-Cooperative-QLMC/pyrite
TAG=tables-epdl-differential-1
ASSET=epdl2025_differential.npz
TAPE_URL=https://nuclear.llnl.gov/EPICS/ENDF2025/EPDL2025.ALL
TAPE_SHA=59bbd8c559685dda0bf0de2762bc43126f599cd154d635940f17b6a59c1c43fd
# Pinned by pyrite.datasets.EPDL_DIFFERENTIAL and photon_differential.EPDL_DIFFERENTIAL_SHA256.
PIN_SHA=d36a50d1add5c6f0b3df80a108e83aa8db8a4822445ca192534f6bc0a55bb166
OUT=build/xsgen-release
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/pyrite-uv-cache}

sha() { sha256sum "$1" | cut -d' ' -f1; }

mkdir -p "$OUT"
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  echo "release $TAG already exists; never replace an asset under an existing tag" >&2
  exit 1
fi

# 1. Upstream tape, SHA-verified.
if [[ ! -f "$OUT/EPDL2025.ALL" ]]; then
  curl -fsSL -o "$OUT/EPDL2025.ALL" "$TAPE_URL"
fi
[[ "$(sha "$OUT/EPDL2025.ALL")" == "$TAPE_SHA" ]] || { echo "EPDL2025.ALL digest mismatch" >&2; exit 1; }

# 2. Deterministic derived table; must reproduce the committed pin.
uv run python scripts/release_epdl_table.py \
  --source "$OUT/EPDL2025.ALL" --table differential --output "$OUT/$ASSET" --write
GOT=$(sha "$OUT/$ASSET")
[[ "$GOT" == "$PIN_SHA" ]] || { echo "built $GOT, pinned $PIN_SHA: use a new tag and repin" >&2; exit 1; }
SIZE=$(stat -c %s "$OUT/$ASSET")

# 3. Notes: provenance, attribution, licence, modifications.
NOTES=$(mktemp)
cat >"$NOTES" <<EOF
PyRITE EPDL2025 photon-transport table (subshell photoionization and scattering functions), installed with \`pyrite tables fetch epdl-differential\`.

- Asset: \`$ASSET\` ($SIZE B), SHA-256 \`$PIN_SHA\`
- Pinned by \`pyrite.datasets.EPDL_DIFFERENTIAL\` and \`pyrite.materials.photon_differential.EPDL_DIFFERENTIAL_SHA256\`
- Built by \`scripts/release_epdl_table.py --table differential\` from \`EPDL2025.ALL\` (SHA-256 \`$TAPE_SHA\`), deterministic output.
- The attenuation table \`tables-epdl-1\` and its digest are unchanged.

## Attribution
Derived from EPICS2025 EPDL by D. E. Cullen, Livermore Evaluated Photon Data Library, NDS-IAEA-225, distributed by the IAEA Nuclear Data Section; source $TAPE_URL
Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

**Modifications:** keeps, for Z = 1–100, the File 23 subshell photoionization cross sections (MT 534–599, with each subshell's binding energy) and the File 27 coherent form factor (MT 502) and incoherent scattering function (MT 504). The MT 505/506 anomalous scattering factors are not kept. Knots were removed where lin-lin interpolation through the kept knots reproduces every removed upstream value to 5e-4 relative; zero values and edge pairs are kept verbatim, and values are stored as float32. See THIRD-PARTY-NOTICES.md.
EOF

# 4. Publish without moving the Latest badge.
gh release create "$TAG" "$OUT/$ASSET" --repo "$REPO" \
  --title "EPDL2025 photon differential table 1" --notes-file "$NOTES" --latest=false
rm -f "$NOTES"

# 5. Download back and check against the pin.
BACK=$(gh release download "$TAG" --repo "$REPO" -p "$ASSET" -O - | sha256sum | cut -d' ' -f1)
[[ "$BACK" == "$PIN_SHA" ]] || { echo "downloaded asset digest $BACK != pin" >&2; exit 1; }

# 6. Install through the public URL into a scratch workspace, as CI does.
SCRATCH=$(mktemp -d)
PYRITE_HOME="$SCRATCH" uv run pyrite tables fetch epdl-differential
rm -rf "$SCRATCH"
echo "published and verified $TAG"
