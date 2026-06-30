# Validation: Python (JAX) port vs. Fortran TUV-x

`validate_plots.py` compares the `tuvx_photolysis` port against the Fortran reference
(`tuv_5_4_no_aerosol_reference.nc`, regenerate via `../tests/fixtures/regenerate_reference.sh`) and
writes the PNGs below. Run with `python validation/validate_plots.py`.

| Plot | What it shows |
|---|---|
| `radiation_field.png` | Total actinic flux spectrum, Python vs Fortran, at 0/20/50 km. The ratio panel is exactly 1.0 outside the Lyman-α/Schumann-Runge band (red); the only deviation is inside that deferred-LA/SR region. |
| `j_profiles.png` | J(altitude) for representative reactions (O3, NO2, CH2O, HNO4); Python overlays Fortran. |
| `j_scatter.png` | Python J vs Fortran J for every reaction dominated by λ ≥ 206 nm — all 3334 points lie on the 1:1 line across 8 orders of magnitude. |
| `j_error_vs_uv.png` | Median J error vs each reaction's deep-UV (<206 nm) fraction. Error is ~1e-8 (machine precision) at zero deep-UV fraction and rises monotonically with it — i.e. the *only* error source is the deferred LA/SR bands, not a hidden bug. |

## Summary

Outside the deferred Lyman-α/Schumann-Runge region, the port reproduces the Fortran TUV-x to
machine precision: solar geometry, the O3 cross section, the full delta-Eddington radiation field,
and the per-reaction photolysis rate constants. Reactions whose photolysis depends on λ < ~206 nm
will match once the LA/SR O2 parameterization (a tracked follow-up) is added.
