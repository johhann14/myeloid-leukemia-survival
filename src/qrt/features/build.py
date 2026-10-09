#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-09-23 14:35:00
Author: joh
"""

from qrt.features import molecular, clinical, cytogenetics, ipssm
from qrt.data import load
import yaml
import pandas as pd
from omegaconf import OmegaConf

def load_config(yaml_path: str):
    cfg = OmegaConf.load(yaml_path)
    if not cfg:
        raise ValueError(f"Config is empty: {yaml_path}")
    return cfg



def build_stateless(clinical_df: pd.DataFrame, molecular_df: pd.DataFrame, cfg) -> pd.DataFrame:
    groups = cfg.features.groups   
    features_list = []
    if 'cytogenetics' in cfg.features.groups or "ipssm" in cfg.features.groups:
        cyto_df = cytogenetics.cytogenetics_features(clinical_df, cfg.features.groups.cytogenetics)
        features_list.append(cyto_df)

    if 'clinical' in cfg.features.groups:
        features_list.append(clinical.clinical_features(clinical_df, cfg.features.groups.clinical))

    if 'molecular' in cfg.features.groups:
        features_list.append(molecular.molecular_features(molecular_df=molecular_df, index=clinical_df.index, cfg=cfg.features.groups.molecular))
    
    if "ipssm" in groups:
        features_list.append(ipssm.ipssm_features(clinical_df, molecular_df, cyto_df, groups.ipssm))


    X = pd.concat(features_list, axis=1)
    assert not X.columns.duplicated().any(), X.columns[X.columns.duplicated()].tolist()
    return X


# def build_features(config: OmegaConf, split: str) -> pd.DataFrame:
#     data_dir = f"{config.data.raw_dir}/{split}"
#     clinical_csv_path = f"{data_dir}/clinical_{split}.csv"
#     dropna = True if split == "train" else False
#     clinical_df = load.load(csv_path=clinical_csv_path, dropna=True)
#     features_list = []

#     if 'cytogenetics' in config.features.groups:
#         features_list.append(cytogenetics.cytogenetics_features(clinical_df, config.features.groups.cytogenetics))

#     if 'clinical' in config.features.groups:
#         features_list.append(clinical.clinical_features(clinical_df, config.features.groups.clinical))
#     return pd.concat(features_list, axis=1)


if __name__ == "__main__":
    config_path = "/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/configs/baseline.yaml"
    config_path = "/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/configs/molecular_baseline.yaml"
    cfg = load_config(config_path)
    print(cfg)
    clinical_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/clinical_train.csv")
    molecular_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/molecular_train.csv")
    print(f"everything loaded")
    train_features = build_stateless(clinical_df=clinical_df, molecular_df=molecular_df, cfg=cfg)
    print(train_features[:5])
    print(train_features.columns)

            


        






