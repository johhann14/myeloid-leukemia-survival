#!/usr/bin/env python3
# coding: utf-8

"""
Creation date: 2026-09-21 19:00:20
Author: joh
"""

import pandas as pd
import numpy as np


from pathlib import Path

def clinical_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = df[list(cfg.columns)].copy()
    if cfg.get("aml_flag", False):
        print(f"AML flag triggered")
        out["aml_blast20"] = (df["BM_BLAST"] >= 20).astype(float)
    return out