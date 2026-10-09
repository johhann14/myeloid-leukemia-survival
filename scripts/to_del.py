#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-09-29 00:07:09
Author: joh
"""

import pandas as pd




df = pd.read_csv("/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/experiments/results copy.csv", index_col=0)

COLUMNS = [
            "config",
            "model",
            "lb_score",
            "cv_mean","cv_std","train_mean","fold_scores",
            "n_features","feature_groups","description"
        ]


df = df[COLUMNS]

df.to_csv("/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/experiments/results_col.csv")

