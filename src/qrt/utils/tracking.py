#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-09-28 18:19:59
Author: joh
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
import os

ROOT = "/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/experiments"


def start_run(cfg, cfg_path):

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{os.path.basename(cfg_path).split('.')[0]}"

    run_dir = f"{ROOT}/runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    sink_id = logger.add(
                    f"{run_dir}/train.log",
                    level="DEBUG", 
                    format="{time:HH:mm:ss} | {level: <7} | {message}"
    )

    OmegaConf.save(cfg, f"{run_dir}/config.yaml")

    return {
        "id": run_id,
        "dir": run_dir,
        "sink": sink_id,
        "cfg": cfg,
        "cfg_path": cfg_path
    }

COLUMNS = [
            "run_id",
            "config",
            "model",
            "lb_score",
            "cv_mean","cv_std","train_mean","fold_scores",
            "n_features","feature_groups","description"
]


def end_run(run, val_scores, train_scores, oof, n_features, risk_test=None):
    cfg = run["cfg"]
    oof.rename("risk_score").to_csv(f"{run['dir']}/oof.csv")
    if risk_test is not None:
        risk_test.rename("risk_score").to_csv(f"{run['dir']}/submission.csv")
    row = {
        "run_id": run["id"],
        "config": Path(run["cfg_path"]).name,
        "description": cfg.experiment.get("description", ""),
        "model": cfg.model.name,
        "feature_groups": "+".join(cfg.features.groups.keys()),
        "n_features": n_features,
        "cv_mean": round(np.mean(val_scores), 4),
        "cv_std": round(np.std(val_scores), 4),
        "train_mean": round(np.mean(train_scores), 4),
        "fold_scores": "|".join(f"{s:.4f}" for s in val_scores),
        "lb_score": "",
    }
    csv_path = f"{ROOT}/results.csv"
    df_row = pd.DataFrame([row])[COLUMNS]
    df_row.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    logger.success(f"CV {row['cv_mean']:.4f} ± {row['cv_std']:.4f} | train {row['train_mean']:.4f}")
    logger.remove(run["sink"])  
