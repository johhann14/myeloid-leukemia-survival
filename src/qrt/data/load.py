#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-09-23 14:48:18
Author: joh
"""

import os 
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


target_train_csv_path = "/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/data/raw/train/target_train.csv"


def load(csv_path: str, dropna: bool = False) -> pd.DataFrame:
    target_train_df = pd.read_csv(target_train_csv_path, index_col=0)
    clean_target_df = target_train_df.dropna(subset=['OS_YEARS', 'OS_STATUS'])

    df = pd.read_csv(csv_path, index_col=0)

    if dropna:
        df = df[df.index.isin(clean_target_df.index)]

    return df



def make_stratified_fold_group_centered(train_target_df: pd.DataFrame, clinical_train_df: pd.DataFrame, n_bins: int, seed):
    """
    The dataframe indexes should already be non NaN values
    """
    train_target_df = train_target_df.copy()
    probs = np.linspace(0, 1, n_bins + 1)
    dead_years = train_target_df.loc[train_target_df["OS_STATUS"] == 1, "OS_YEARS"]
    censored_years = train_target_df.loc[train_target_df["OS_STATUS"] == 0, "OS_YEARS"]

    quantiles_dead = np.unique(np.quantile(dead_years, probs))
    quantiles_censored = np.unique(np.quantile(censored_years, probs))
    # print(len(quantiles_censored))
    # print(quantiles_censored)
    # print(quantiles_dead)
    # print(len(quantiles_censored))
    # print(dead_years.shape)

    train_target_df["stratified_label"] = pd.NA
    dead_bins = np.digitize(dead_years, quantiles_dead[1:-1])
    censored_bins = np.digitize(censored_years, quantiles_censored[1:-1])

    train_target_df.loc[train_target_df["OS_STATUS"] == 1, 'stratified_label'] = "d_" + dead_bins.astype(str)
    train_target_df.loc[train_target_df["OS_STATUS"] == 0, 'stratified_label'] = "c_" + censored_bins.astype(str)


    # print(f"REsulsts")
    # print(np.unique(censored_bins))
    # print(dead_bins.shape)

    # number of labels 2 * (n_bins-2)
    # d = {} 

    # for i in df.index:
    #     if i in index_censored:
    #         p = "c"
    #         ind = np.where(index_censored == i)[0][0]
    #         s = censored_bins[ind]
    #         c = f"{p}_{s}"
    #     else:
    #         p = "d"
    #         ind = np.where(index_dead == i)[0][0]
    #         s = dead_bins[ind]
    #         c = f"{p}_{s}"

    #     d[i] = c


    X = train_target_df.index
    y = train_target_df["stratified_label"]
    groups = clinical_train_df.loc[train_target_df.index, "CENTER"]
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_target_df["fold"] = -1
    for i , (train_index, val_index) in enumerate(skf.split(X, y, groups)):
        val_uid = train_target_df.index[val_index]
        train_target_df.loc[val_uid, 'fold'] = i

    train_target_df = train_target_df.drop(columns=["stratified_label", "OS_YEARS", "OS_STATUS"])
    train_target_df["fold"] = pd.to_numeric(train_target_df["fold"])
    return train_target_df



def make_stratified_fold_aml(train_target_df, clinical_df, seed, n_splits=5):
    aml = (clinical_df.loc[y.index, "BM_BLAST"] >= 20).astype(int).astype(str)
    strata = train_target_df.OS_STATUS.astype(int).astype(str) + "_" + pd.qcut(train_target_df.OS_YEARS, 4, labels=False).astype(str) + "_" + aml
    f = pd.Series(-1, index=y.index)
    for i, (_, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(y.index, strata)):
        f.iloc[va] = i
    return f

if __name__ == "__main__":
    from qrt.features import build
    cfg = build.load_config(yaml_path="/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/configs/baseline.yaml")
    df = load(f'{cfg.data.raw_dir}/train/target_train.csv', dropna=True)
    clinical_train_df = load(f"{cfg.data.raw_dir}/train/clinical_train.csv", dropna=True)
    d = make_stratified_fold_group_centered(df, clinical_train_df=clinical_train_df, n_bins=4, seed=cfg.experiment.seed)
    print(f"ok : {len(df)}")
    print(d)

    d.to_csv("/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/data/preprocessed/stratified_fold.csv")




