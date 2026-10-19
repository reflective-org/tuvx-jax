# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Stage 2: data loaders round-trip the bundled TUV-x data files."""

from pathlib import Path

import numpy as np
import pytest

from tuvx_photolysis import data

DATA = Path(__file__).resolve().parents[1] / "data"


def test_load_cross_section_base():
    xs = data.load_cross_section(DATA / "cross_sections" / "HNO4_1.nc")
    assert xs.wavelength.shape == (54,)
    assert xs.parameters.shape == (54, 1)
    assert xs.n_params == 1
    assert xs.values.shape == (54,)
    assert np.all(xs.values >= 0.0)
    assert xs.units == "cm^2"
    # wavelength monotonically increasing
    assert np.all(np.diff(xs.wavelength) > 0)


def test_load_cross_section_with_temperature():
    xs = data.load_cross_section(DATA / "cross_sections" / "O3_1.nc")
    assert xs.wavelength.shape == (63501,)
    assert xs.parameters.shape == (63501, 1)
    assert xs.temperature is not None and xs.temperature.shape == (1,)


def test_load_quantum_yield_multiparam():
    qy = data.load_quantum_yield(DATA / "quantum_yields" / "CH2O_1.nc")
    assert qy.wavelength.shape == (112,)
    assert qy.parameters.shape == (112, 2)
    assert qy.n_params == 2
    with pytest.raises(ValueError):
        _ = qy.values  # not defined for multi-parameter data


def test_tabulated_from_arrays_roundtrip():
    # The JPL-swap-in path: build the same container directly from arrays.
    wl = np.linspace(200.0, 400.0, 11)
    sigma = np.exp(-((wl - 300.0) ** 2) / 100.0) * 1e-20
    td = data.TabulatedData(wavelength=wl, parameters=sigma, units="cm^2")
    assert td.parameters.shape == (11, 1)
    np.testing.assert_allclose(td.values, sigma)


def test_load_profiles():
    # All start at the surface and increase monotonically. ussa.dens reaches 120 km, ussa.ozone
    # 74 km. ussa.temp caps with a high-altitude exospheric sentinel row (altitude 1e10) — the
    # loader stays faithful to the file; sentinel handling happens at grid-interpolation time.
    for name in ("ussa.dens", "ussa.temp", "ussa.ozone"):
        prof = data.load_profile_csv(DATA / "profiles" / "atmosphere" / name)
        assert prof.altitude_km[0] == 0.0
        assert np.all(np.diff(prof.altitude_km) > 0)
        assert np.all(prof.values > 0)
    assert data.load_profile_csv(DATA / "profiles" / "atmosphere" / "ussa.dens").altitude_km[-1] == pytest.approx(120.0)
    assert data.load_profile_csv(DATA / "profiles" / "atmosphere" / "ussa.ozone").altitude_km[-1] == pytest.approx(74.0)
    temp = data.load_profile_csv(DATA / "profiles" / "atmosphere" / "ussa.temp")
    assert temp.altitude_km[-1] == pytest.approx(1e10)  # exospheric sentinel
    assert 270.0 < temp.values[0] < 300.0  # surface temperature ~288 K


def test_load_wavelength_grid():
    edges = data.load_wavelength_grid(DATA / "grids" / "wavelength" / "combined.grid")
    assert edges.shape == (157,)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == pytest.approx(120.0)


def test_load_solar_flux():
    flux = data.load_solar_flux(DATA / "profiles" / "solar" / "neckel.flx")
    assert flux.wavelength.shape == flux.flux.shape
    assert np.all(flux.flux > 0)
    assert flux.wavelength[0] == pytest.approx(330.5)


def test_loads_from_a_folder_with_non_ascii_characters(tmp_path):
    # Windows: the netCDF/HDF5 C library cannot open such paths itself (a user folder named
    # "José"); the loader reads the bytes in Python, so the folder name does not matter.
    folder = tmp_path / "cache-José-データ"
    folder.mkdir()
    copy = folder / "O2_1.nc"
    copy.write_bytes((DATA / "cross_sections" / "O2_1.nc").read_bytes())
    xs = data.load_cross_section(copy)
    ref = data.load_cross_section(DATA / "cross_sections" / "O2_1.nc")
    np.testing.assert_array_equal(xs.parameters, ref.parameters)
    np.testing.assert_array_equal(xs.wavelength, ref.wavelength)


def test_every_data_file_opens_from_memory_with_identical_contents():
    # from_netcdf hands netCDF the file's bytes (plus zero padding); every bundled file must
    # open that way and give exactly the arrays netCDF reads from the file on disk
    import netCDF4
    files = sorted(DATA.rglob("*.nc"))
    assert len(files) > 100
    for path in files:
        buffer = path.read_bytes() + bytes(data._MEMORY_PADDING_BYTES)
        with netCDF4.Dataset(str(path)) as disk, \
                netCDF4.Dataset(path.name, mode="r", memory=buffer) as mem:
            assert set(mem.variables) == set(disk.variables), path.name
            for name in disk.variables:
                np.testing.assert_array_equal(np.array(mem.variables[name][:]),
                                              np.array(disk.variables[name][:]),
                                              err_msg=f"{path.name}:{name}")
