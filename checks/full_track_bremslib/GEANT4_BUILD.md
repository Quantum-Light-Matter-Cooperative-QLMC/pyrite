# Rebuild the Geant4 issue #182 reference

The input source is Geant4 tag `v11.4.2`, commit
`8cc04f65977807f1848da7b958c421cd5e162f26`. This recipe runs in a
scratch directory containing a copy of this check's `.mac` files and
`geant4-testem5.patch`. The completed modified TestEm5 source is also retained
in `~/dev/geant4/issue-182/source/` on the original workstation.

```bash
git clone --filter=blob:none --sparse https://github.com/Geant4/geant4.git source
git -C source checkout 8cc04f65977807f1848da7b958c421cd5e162f26
git -C source sparse-checkout set examples/extended/electromagnetic/TestEm5
git -C source apply --unidiff-zero ../geant4-testem5.patch
micromamba create -y -p "$PWD/geant4-env" -c conda-forge \
  geant4=11.4.2 expat zlib freetype gxx_linux-64
cmake -S source/examples/extended/electromagnetic/TestEm5 -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGeant4_DIR="$PWD/geant4-env/lib/cmake/Geant4" \
  -DCMAKE_CXX_COMPILER="$PWD/geant4-env/bin/x86_64-conda-linux-gnu-g++"
cmake --build build -j 4
cp w_300kev.mac si_300kev.mac w_800kev.mac si_800kev.mac build/
for case in w_300kev si_300kev w_800kev si_800kev; do
  (cd build && micromamba run -p "$OLDPWD/geant4-env" ./TestEm5 "$case.mac" > "$case.log" 2>&1)
done
```

For an exact Linux package set, use
`micromamba create -y -p "$PWD/geant4-env" -f geant4-conda-explicit.txt`
in place of the package-name install command. The explicit file records the
resolved conda-forge package URLs from the original run.

Geant4's packaged data are selected by `micromamba run`. The macros set the
thread count, seeds, material, geometry, physics, cuts, histograms and event
count. Histogram 62 selects `eBrem` photons at creation; histogram 3 counts
all photons. Inspect the `Lowest e+e- kinetic energy` line in each log to
confirm the 10 keV stop setting. Save the `*_h1_h3.csv` and `*_h1_h62.csv`
files with the logs; these are the raw comparison inputs.

## Emission benchmark

Use `geant4-testem5-emission.patch` in place of `geant4-testem5.patch`
(it contains that change as well), build as above into a separate build
directory, and run the four `*_emission.mac` macros. Add
`-DCMAKE_PREFIX_PATH="$PWD/geant4-env"` to the CMake configure line if
CLHEP is not found. Save each `*_emission.log`, gzip it, and keep
`*_emission_h1_h63.csv`. The four runs of 100,000 events take a few minutes
on one core each.
