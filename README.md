# tuvx-jax

A Python/JAX port of [NCAR's TUV-x](https://github.com/NCAR/tuv-x) actinic-flux and
photolysis-rate-constant pipeline. Given altitude / latitude / longitude / time-of-year and
the bundled JPL/IUPAC cross sections and quantum yields, it solves the radiation field
(delta-Eddington two-stream) and integrates `J = ∫ F(λ,z)·σ(λ,T)·φ(λ) dλ` per reaction.

**Validated to machine precision against the Fortran TUV-x** for the bundled no-aerosol
configuration (including the O2 Lyman-α/Schumann-Runge bands); see `validation/`. Double
precision throughout (`jax_enable_x64` is set on import).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                    # the full validation suite vs Fortran reference fixtures
python validation/validate_plots.py       # comparison plots vs the Fortran TUV-x outputs
```

The bundled TUV-x/JPL data in `data/` makes the package runnable with no dependence on the
original Fortran repo at runtime.

## Usage

```python
from tuvx_photolysis import PhotolysisCalculator

calc = PhotolysisCalculator.from_tuvx_json(
    "examples/tuv_5_4_no_aerosol.json",   # TUV-x JSON config
    data_root=".",                        # directory containing data/
)
J = calc.rate_constants_profile(solar_zenith_angle_deg=45.0, earth_sun_distance=1.0)
# J: {reaction name: J profile [s^-1] on calc.height_edges_km}
# (calc.rate_constants(latitude, longitude, year, month, day, hour_utc, ...) computes
#  the solar position for you; branching=True applies JPL product-branching yields)
```

`PhotolysisCalculator` (`api.py`) orchestrates the pipeline: `geometry` (spherical, slant
paths) → `radiators` (air/O2/O3 optical properties) → `solver` (delta-Eddington two-stream
radiation field) → `photolysis` (actinic flux) → per-reaction wavelength integration → J.
`cross_section.py` / `quantum_yield.py` build the σ·φ product per reaction; `special.py`
holds the non-tabulated recipes. Reactions whose recipes aren't ported are recorded in
`skipped_reactions`, never silently dropped.

Each module's docstring cites the authoritative Fortran source it ports. **Treat the
Fortran as ground truth** when changing physics.

## J coverage: MCM tropospheric (limonene) mechanism

Photolysis frequencies used by the tropospheric MCM limonene mechanism (KPP `constants_mcm`
naming, MCM J number), and where each comes from with `examples/tuv_5_4_no_aerosol.json`.
The TUV-x key is the exact key returned by `rate_constants` / `rate_constants_profile`. All
24 keys (22 direct, 2 totals for MCM branching) are supported, none in `skipped_reactions`;
`tests/test_mcm_coverage.py` checks this table against the calculator.

| MCM J name | MCM J | TUV-x key returned by `rate_constants` |
|---|---|---|
| `O3_O1D` | 1 | `O3+hv->O2+O(1D)` |
| `O3_O3P` | 2 | `O3+hv->O2+O(3P)` |
| `H2O2` | 3 | `H2O2+hv->OH+OH` |
| `NO2` | 4 | `NO2+hv->NO+O(3P)` |
| `NO3_NO` | 5 | `NO3+hv->NO+O2` |
| `NO3_NO2` | 6 | `NO3+hv->NO2+O(3P)` |
| `HONO` | 7 | `HNO2+hv->OH+NO` |
| `HNO3` | 8 | `HNO3+hv->OH+NO2` |
| `HCHO_H` | 11 | `CH2O+hv->H+HCO` |
| `HCHO_H2` | 12 | `CH2O+hv->H2+CO` |
| `C3H7CHO_HCO` | 15 | MCM parameterization (no TUV-x recipe) |
| `IPRCHO` | 17 | MCM parameterization (no TUV-x recipe) |
| `MACR_HCO` | 18 | MCM branching of `CH2=C(CH3)CHO+hv->Products` |
| `MACR_H` | 19 | MCM branching of `CH2=C(CH3)CHO+hv->Products` |
| `C5HPALD1` | 20 | MCM parameterization (no TUV-x recipe) |
| `CH3COCH3` | 21 | `CH3COCH3+hv->CH3CO+CH3` |
| `MEK` | 22 | `CH3COCH2CH3+hv->CH3CO+CH2CH3` |
| `MVK_C2H3` | 24 | MCM branching of `MVK+hv->Products` |
| `GLYOX_H2` | 31 | `CHOCHO+hv->H2+CO+CO` |
| `GLYOX_HCHO` | 32 | `CHOCHO+hv->CH2O+CO` |
| `GLYOX_HCO` | 33 | `CHOCHO+hv->HCO+HCO` |
| `MGLYOX` | 34 | `CH3COCHO+hv->CH3CO+HCO` |
| `BIACET` | 35 | `CH3COCOCH3+hv->Products` |
| `CH3OOH` | 41 | `CH3OOH+hv->CH3O+OH` |
| `CH3NO3` | 51 | `CH3ONO2+hv->CH3O+NO2` |
| `NC3H7NO3` | 53 | `nC3H7ONO2+hv->C3H7O+NO2` |
| `IC3H7NO3` | 54 | `CH3CHONO2CH3+hv->CH3CHOCH3+NO2` |
| `NOA` | 56 | `nitro_acetone+hv->Products` |

* **No TUV-x recipe**: butanal (J15), isobutyraldehyde (J17) and the HPALDs (J20) have no
  cross-section/quantum-yield data in TUV-x; they are left to the MCM clear-sky parameterization
  rather than invented here.
* **MCM branching**: TUV-x carries one total channel for methacrolein (constant Φ = 0.01) and for
  MVK (`mvk.F90` total yield). The MCM channels (J18/J19, J24) are split downstream with MCM
  branching ratios.
* Only what TUV-x itself defines is returned; values follow the Fortran exactly, including its
  recipe quirks (documented in the docstrings). These are the TUV 5.4 recipes, not necessarily the
  MCM's own cross sections/yields.

### Validation of the tropospheric recipes

The recipes ported for this mechanism are CH2O (`ch2o.F90` cross section and H2+CO quantum
yield; the H+HCO channel is the base quantum yield on parameter 1 of `CH2O_1.nc`), CH3ONO2
(`ch3ono2-ch3o_no2.F90`), RONO2 (`rono2.F90`, e.g. i-C3H7ONO2), nitroxy-acetone
(`nitroxy_acetone.F90`), MEK (`ch3coch2ch3-ch3co_ch2ch3.F90`), methylglyoxal (`ch3cocho.F90`), MVK
(`mvk.F90`) and acetone (`acetone-ch3co_ch3.F90` cross section and quantum yield). They are
checked against Fortran TUV-x 0.15.0 output for `tests/fixtures/tuv_5_4_no_aerosol_sza.json`
(the validated config at fixed SZA 0/30/60/80 deg, 1 AU, with σ/φ diagnostics; regenerate with
`python tests/fixtures/regenerate_sza_reference.py /path/to/tuv-x/build`):

* σ(level, λ) and φ(level, λ) on all 121 levels x 156 wavelength cells: `rtol = 1e-12`
  (observed ≤ 7e-15, i.e. round-off);
* J at 0-15 km (including the surface) for each SZA: `rtol = 1e-7`. Observed 1.75e-8 for every
  reaction, the same SZA-independent offset every long-validated reaction shows against the
  Fortran (a common scalar, not a recipe error).

Reactions sharing these recipe types (C2H5ONO2, PAN, PPN, CH3CH2ONO2) are enabled too and
checked the same way, or against the original two-time reference in `test_photolysis.py`.

## Provenance

This repo was split (history-preserving) out of the coupled SAI plume-modeling project
[`reflective-org/SANDBOX`](https://github.com/reflective-org/SANDBOX), where it drives the
photolysis of a stratospheric chemistry box model via a thin adapter. It has no dependency
on that project.
