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

## Provenance

This repo was split (history-preserving) out of the coupled SAI plume-modeling project
[`reflective-org/SANDBOX`](https://github.com/reflective-org/SANDBOX), where it drives the
photolysis of a stratospheric chemistry box model via a thin adapter. It has no dependency
on that project.
