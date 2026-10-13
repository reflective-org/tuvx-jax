# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Regenerate ``tuv_5_4_no_aerosol_sza_reference.nc`` from a Fortran TUV-x build.

Usage::

    python tests/fixtures/regenerate_sza_reference.py /path/to/tuv-x/build

The Fortran ``tuv-x`` executable is run in a temporary directory (the build tree is not touched) on
``tuv_5_4_no_aerosol_sza.json``: the validated ``tuv_5_4_no_aerosol.json`` with only these changes

* the time grid / solar zenith angle / Earth-Sun distance profiles are replaced by fixed values
  (SZA = 0, 30, 60, 80 degrees; Earth-Sun distance = 1 AU) so the reference covers low sun;
* ``"__output": true`` is set on the cross section and quantum yield of :data:`DIAG_REACTIONS`, so
  the Fortran also writes its sigma(level, wavelength) and phi(level, wavelength) arrays.

``data/`` in the temporary directory is a symlink to this repository's bundled ``data/`` so the
Fortran reads exactly the data the Python port reads. The output is copied (values untouched) into a
compact, compressed fixture: the J profiles of every reaction at every SZA, plus the diagnostic
sigma/phi arrays at the first SZA only (they do not depend on SZA; this is asserted).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import netCDF4
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE_CFG = HERE / "tuv_5_4_no_aerosol.json"
SZA_CFG = HERE / "tuv_5_4_no_aerosol_sza.json"
OUT = HERE / "tuv_5_4_no_aerosol_sza_reference.nc"

SZA_DEG = [0.0, 30.0, 60.0, 80.0]

#: Reactions whose cross section and quantum yield arrays the Fortran writes as diagnostics
#: (the recipes ported for the MCM tropospheric mechanism, plus reactions sharing those recipes).
DIAG_REACTIONS = [
    "CH2O+hv->H+HCO",
    "CH2O+hv->H2+CO",
    "CH3ONO2+hv->CH3O+NO2",
    "C2H5ONO2+hv->C2H5O+NO2",
    "PAN+hv->CH3CO(OO)+NO2",
    "CH3CH2ONO2+hv->CH3CH2O+NO2",
    "CH3CHONO2CH3+hv->CH3CHOCH3+NO2",
    "nitro_acetone+hv->Products",
    "CH3COCH2CH3+hv->CH3CO+CH2CH3",
    "CH3COCHO+hv->CH3CO+HCO",
    "MVK+hv->Products",
    "CH3COCH3+hv->CH3CO+CH3",
    "CH3CHO+hv->CH3+HCO",
    "C2H5CHO+hv->C2H5+HCO",
    "CH2CHCHO+hv->Products",
    "NOCl+hv->NO+Cl",
    "CFC-113+hv->Products",
]


def make_config() -> dict:
    cfg = json.loads(BASE_CFG.read_text())
    time_grid = {"name": "time", "units": "hours"}
    for g in cfg["grids"]:
        if g["name"] == "time":
            g["values"] = [float(i) for i in range(len(SZA_DEG))]
    for i, p in enumerate(cfg["profiles"]):
        if p["name"] == "solar zenith angle":
            cfg["profiles"][i] = {"name": "solar zenith angle", "type": "from config file",
                                  "units": "degrees", "values": SZA_DEG, "grid": time_grid}
        elif p["name"] == "Earth-Sun distance":
            cfg["profiles"][i] = {"name": "Earth-Sun distance", "type": "from config file",
                                  "units": "AU", "uniform value": 1.0, "grid": time_grid}
    names = {r["name"] for r in cfg["photolysis"]["reactions"]}
    missing = set(DIAG_REACTIONS) - names
    if missing:
        raise SystemExit(f"reactions not in {BASE_CFG.name}: {sorted(missing)}")
    for r in cfg["photolysis"]["reactions"]:
        if r["name"] in DIAG_REACTIONS:
            r["cross section"]["__output"] = True
            r["quantum yield"]["__output"] = True
    return cfg


def subset(src: Path, dst: Path, reaction_names) -> None:
    """Copy the J profiles (all SZAs) and sigma/phi diagnostics (first SZA) into ``dst``."""
    with netCDF4.Dataset(src) as ds, netCDF4.Dataset(dst, "w") as out:
        out.description = ("Fortran TUV-x output for tests/fixtures/tuv_5_4_no_aerosol_sza.json "
                           "(subset by tests/fixtures/regenerate_sza_reference.py)")
        for dim in ("vertical_level", "wavelength", "time"):
            out.createDimension(dim, ds.dimensions[dim].size)
        keep = ["altitude", "wavelength", "solar zenith angle", "Earth-Sun distance", "temperature"]
        keep += [n for n in reaction_names if n in ds.variables]
        for name in keep:
            v = ds.variables[name]
            o = out.createVariable(name, "f8", v.dimensions, zlib=True, complevel=9)
            o[:] = v[:]
        for name in ds.variables:
            if not name.startswith(("cross section ", "quantum yield ")):
                continue
            arr = np.array(ds.variables[name][:])  # (wavelength, vertical_level, time)
            assert np.array_equal(arr, np.repeat(arr[:, :, :1], arr.shape[2], axis=2)), name
            o = out.createVariable(name, "f8", ("wavelength", "vertical_level"),
                                   zlib=True, complevel=9)
            o[:] = np.ascontiguousarray(arr[:, :, 0])


def main(build_dir: str) -> None:
    exe = Path(build_dir).resolve() / "tuv-x"
    if not exe.exists():
        raise SystemExit(f"no tuv-x executable at {exe}")
    cfg = make_config()
    SZA_CFG.write_text(json.dumps(cfg, indent=2) + "\n")
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        (work / "data").symlink_to(REPO / "data")
        (work / SZA_CFG.name).write_text(SZA_CFG.read_text())
        subprocess.run([str(exe), SZA_CFG.name], cwd=work, check=True)
        names = [r["name"] for r in cfg["photolysis"]["reactions"]]
        subset(work / "photolysis_rate_constants.nc", OUT, names)
    print(f"Wrote {SZA_CFG} and {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
