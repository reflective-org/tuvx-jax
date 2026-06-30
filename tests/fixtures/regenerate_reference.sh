#!/usr/bin/env bash
# Regenerate the Fortran reference output used by the validation tests.
# The .nc fixtures are git-ignored (large, derived); rebuild them from the TUV-x Fortran build.
#
# Usage:  bash regenerate_reference.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"          # tuv-x repo root
BUILD="$REPO/build"

if [[ ! -x "$BUILD/tuv-x" ]]; then
  echo "Building TUV-x Fortran first..."
  mkdir -p "$BUILD"
  ( cd "$BUILD" && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 )
fi

echo "Running tuv-x on examples/tuv_5_4.json ..."
( cd "$BUILD" && ./tuv-x examples/tuv_5_4.json )
cp "$BUILD/photolysis_rate_constants.nc" "$HERE/tuv_5_4_reference.nc"
echo "Wrote $HERE/tuv_5_4_reference.nc"

# no-aerosol variant: exact radiation-field / J validation for the ported (non-aerosol) radiators
echo "Running tuv-x on tuv_5_4_no_aerosol.json ..."
cp "$HERE/tuv_5_4_no_aerosol.json" "$BUILD/examples/"
( cd "$BUILD" && ./tuv-x examples/tuv_5_4_no_aerosol.json )
cp "$BUILD/photolysis_rate_constants.nc" "$HERE/tuv_5_4_no_aerosol_reference.nc"
echo "Wrote $HERE/tuv_5_4_no_aerosol_reference.nc"
