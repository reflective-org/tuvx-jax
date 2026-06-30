# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Stage 1 smoke test: the package and all module stubs import cleanly."""

import importlib

import tuvx_photolysis


def test_version():
    assert tuvx_photolysis.__version__ == "0.1.0"


def test_modules_import():
    for mod in ["data", "grids", "geometry", "radiators", "solver", "photolysis", "api"]:
        importlib.import_module(f"tuvx_photolysis.{mod}")
