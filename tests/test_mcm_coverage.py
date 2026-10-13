# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The README "J coverage" table (MCM J name -> TUV-x key) is backed by the calculator.

Every TUV-x key the table names (directly, or as the total an MCM branching is applied to) must be
a reaction the calculator returns for ``examples/tuv_5_4_no_aerosol.json`` -- not skipped -- with a
positive surface J by day. Rows without a TUV-x recipe must name no key.
"""

import re
from pathlib import Path

import pytest

from tuvx_photolysis import PhotolysisCalculator

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "examples" / "tuv_5_4_no_aerosol.json"
NO_RECIPE = "MCM parameterization (no TUV-x recipe)"


def _coverage_rows():
    text = (REPO / "README.md").read_text()
    section = text.split("## J coverage", 1)[1].split("\n#", 1)[0]
    return re.findall(r"^\| `(\w+)` \| (\d+) \| (.+?) \|$", section, flags=re.M)


ROWS = _coverage_rows()


@pytest.fixture(scope="module")
def calc():
    return PhotolysisCalculator.from_tuvx_json(CFG, data_root=REPO)


@pytest.fixture(scope="module")
def surface_j(calc):
    return calc.rate_constants(0.0, 0.0, 2002, 3, 21, 12.0, altitude_km=0.0)


def test_table_lists_the_28_mechanism_photolysis_rates():
    names = [r[0] for r in ROWS]
    assert len(names) == 28 and len(set(names)) == 28


@pytest.mark.parametrize("mcm_name,mcm_j,source", ROWS, ids=[r[0] for r in ROWS])
def test_row_is_backed_by_calculator(calc, surface_j, mcm_name, mcm_j, source):
    keys = re.findall(r"`([^`]+)`", source)
    if source == NO_RECIPE:
        assert not keys
        return
    assert len(keys) == 1 and (source == f"`{keys[0]}`" or source.startswith("MCM branching of"))
    key = keys[0]
    assert key in calc.reaction_names(), calc.skipped_reactions.get(key, "not in config")
    assert key not in calc.skipped_reactions
    assert surface_j[key] > 0.0
