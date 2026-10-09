#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-10-03 17:08:38
Author: joh
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clinical_df():
    df = pd.DataFrame(
        data={
            "CENTER": ["MSK", "CCH", "KYW", "CGM"],
            "BM_BLAST": [2.0, 25.0, np.nan, 8.0],
            "WBC": [4.0, 30.0, 5.0, np.nan],
            "ANC": [2.0, 1.0, 3.0, 0.5],
            "MONOCYTES": [0.5, np.nan, 0.3, 0.2],
            "HB": [11.0, 7.5, np.nan, 9.0],
            "PLT": [200.0, 30.0, 400.0, 90.0],
            "CYTOGENETICS": [
                "46,XX[20]",
                "45,XY,-7[15/20]/46,XY[5/20]",
                "Normal",
                "40~46,XX,der(6)t(1;6)(q24;p24) with multiple abnormalities [24]",
            ],
        },
        index=pd.Index(["P1", "P2", "P3", "P4"], name="ID")
    )


@pytest.fixture
def molecular_df():
    return pd.DataFrame(
        {
            "GENE": ["TP53", "TP53", "SF3B1", "FLT3", "MLL"],
            "EFFECT": ["non_synonymous_codon", "stop_gained", "non_synonymous_codon", "ITD", "PTD"],
            "PROTEIN_CHANGE": ["p.R273H", "p.Q52*", "p.K700E", "p.ITD", "p.PTD"],
            "VAF": [0.30, 0.25, 0.40, 0.10, 0.20],
        },
        index=pd.Index(["P2", "P2", "P1", "P4", "P4"], name="ID"),
    )

@pytest.fixture
def target_df():
    return pd.DataFrame(
        {"OS_YEARS": [5.0, 0.8, 3.0, 1.2], "OS_STATUS": [0, 1, 0, 1]},
        index=pd.Index(["P1", "P2", "P3", "P4"], name="ID"),
    )
