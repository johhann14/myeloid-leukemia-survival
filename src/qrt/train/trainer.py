#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-09-23 13:37:38
Author: joh
"""

from qrt.features import build
from qrt.data import load
from qrt.models.base import cox, xgb

from sksurv.util import Surv
import numpy as np
from sklearn.impute import SimpleImputer

from sksurv.metrics import concordance_index_censored , concordance_index_ipcw
from qrt.models import registry

from loguru import logger
from qrt.utils import tracking
import pandas as pd
import fire
from scipy.stats import rankdata


RUNS = "experiments/runs"


def evaluate(y_train, y_test, risk, tau=7):
    y_train = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', y_train)
    y_test = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', y_test)

    return concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0]


def run_cv(cfg, X, y, clinical):
    fold_df = load.make_stratified_fold_group_centered(
                                train_target_df=y,
                                clinical_train_df=clinical,
                                n_bins=4,
                                seed=cfg.experiment.seed
            )
    val_scores, train_scores = [], []
    oof = pd.Series(np.nan, index=X.index)

    for f in sorted(fold_df["fold"].unique()):
        tr_uids = fold_df[fold_df["fold"] != f].index
        val_uids = fold_df[fold_df["fold"] == f].index
        model = registry.build_wrapper(cfg)
        X_tr, y_tr = X.loc[tr_uids], y.loc[tr_uids]
        model.fit(X_tr, y_tr)

        X_val = X.loc[val_uids]
        y_val = y.loc[val_uids]

        risk = model.predict(X_val)
        oof.loc[val_uids] = risk

        cox_index_val = evaluate(y_tr, y_val, risk, tau=7)
        cox_index_train = evaluate(y_tr, y_tr, model.predict(X_tr), tau=7)

        val_scores.append(cox_index_val)
        train_scores.append(cox_index_train)

    return train_scores, val_scores, oof


def run_full_train_test(cfg, X_train, y, X_test):
    missing = set(X_train.columns) ^ set(X_test.columns)
    assert not missing, f"columns differs train/test: {missing}"

    X_test = X_test[X_train.columns]
    model = registry.build_wrapper(cfg)
    model.fit(X_train, y)

    risk = pd.Series(model.predict(X_test), index=X_test.index, name="risk_score")

    return risk



def run_experiment(cfg_path, submission=False):
    logger.info(f"Running exp | Submission: {submission}")
    cfg = build.load_config(cfg_path)
    run = tracking.start_run(cfg, cfg_path)

    try:
        clinical_train_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/clinical_train.csv", dropna=True)
        molecular_train_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/molecular_train.csv", dropna=True)
        clinical_test_df = load.load(csv_path=f"{cfg.data.raw_dir}/test/clinical_test.csv", dropna=False)
        molecular_test_df = load.load(csv_path=f"{cfg.data.raw_dir}/test/molecular_test.csv", dropna=False)
        target_train_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/target_train.csv", dropna=True)
        train_stateless_features = build.build_stateless(clinical_df=clinical_train_df, molecular_df=molecular_train_df, cfg=cfg)
        test_stateless_features = build.build_stateless(clinical_df=clinical_test_df, molecular_df=molecular_test_df, cfg=cfg)

        X_train = train_stateless_features.copy()
        X_test = test_stateless_features.copy()
        y_train = target_train_df.copy()
        logger.info(f"Train columns: {X_train.columns}")
        logger.info(f"Test columns: {X_test.columns}")

        logger.info(f"Running CV..")
        train_scores, val_scores, oof = run_cv(cfg, X_train, y_train, clinical_train_df)
        if submission:
            logger.info(f"Running fullt train/test")
            risk_test = run_full_train_test(cfg, X_train, y_train, X_test)
        else:
            risk_test = None

        tracking.end_run(
            run=run,
            val_scores=val_scores,
            train_scores=train_scores,
            oof=oof,
            n_features=X_train.shape[1],
            risk_test=risk_test,
        )

    except Exception as e:
        logger.exception("Run failed: {e}")
        logger.remove(run["sink"])
        raise




def make_submission(cfg_yaml: str):
    logger.info(f"Training on the whole train and infering on the test")
    cfg = build.load_config(cfg_yaml)
    run = tracking.start_run(cfg=cfg, cfg_path=cfg_yaml)

    try:
        clinical_train_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/clinical_train.csv", dropna=True)
        molecular_train_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/molecular_train.csv", dropna=True)
        clinical_test_df = load.load(csv_path=f"{cfg.data.raw_dir}/test/clinical_test.csv", dropna=False)
        molecular_test_df = load.load(csv_path=f"{cfg.data.raw_dir}/test/molecular_test.csv", dropna=False)
        target_train_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/target_train.csv", dropna=True)

        train_stateless_features = build.build_stateless(clinical_df=clinical_train_df, molecular_df=molecular_train_df, cfg=cfg)
        test_stateless_features = build.build_stateless(clinical_df=clinical_test_df, molecular_df=molecular_test_df, cfg=cfg)

        # training
        
        X_train = train_stateless_features.copy()
        X_test = test_stateless_features.copy()
        logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
        y_train_df = target_train_df.copy()

        wrapper = registry.build_wrapper(cfg)
        wrapper.fit(X_train, y_train_df)
        risk_test = wrapper.predict(X_test)
        cox_index_train = evaluate(y_train_df, y_train_df, risk=wrapper.predict(X_train), tau=7)
        oof = pd.Series(risk_test, index=X_test.index)

        logger.debug(f"OKKKK")
        logger.debug(f"RISK_test: {risk_test[:10]}")
        tracking.end_run(run, np.zeros(2), [cox_index_train], oof, n_features=train_stateless_features.shape[1])
    except Exception as e:
        logger.exception(f"Exception raise: {e}")
        raise



def run_training(cfg_yaml: str, fold):
    logger.info(f"Running: {cfg_yaml} on fold: {fold}")
    cfg = build.load_config(cfg_yaml)
    run = tracking.start_run(cfg=cfg, cfg_path=cfg_yaml)
    try:
        clinical_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/clinical_train.csv", dropna=True)
        molecular_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/molecular_train.csv", dropna=True)

        target_train_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/target_train.csv", dropna=True)
        fold_df = load.make_stratified_fold_group_centered(target_train_df, clinical_train_df=clinical_df, n_bins=4, seed=cfg.experiment.seed)

        stateless_features = build.build_stateless(clinical_df=clinical_df, molecular_df=molecular_df, cfg=cfg)
        logger.info(f"Head stateless features")
        logger.debug(stateless_features[:3])
        logger.info(f"Col stateless features: {stateless_features.columns}")
        if isinstance(fold, int):
            fold = [fold]

        train_scores = []
        test_scores = []
        oof = pd.Series(np.nan, index=fold_df.index)
        for f in fold:    
            train_uids = fold_df[fold_df["fold"] != f].index
            test_uids = fold_df[fold_df["fold"] == f].index
            # features = ["BM_BLAST", "HB", "PLT", "cyto_complexity_score"]
            X_train = stateless_features.loc[train_uids]
            X_test = stateless_features.loc[test_uids]

            y_train_df = target_train_df.loc[train_uids]
            y_test_df = target_train_df.loc[test_uids]
            y_train = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', target_train_df.loc[train_uids])
            y_test = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', target_train_df.loc[test_uids])

            wrapper = registry.build_wrapper(cfg)
            logger.info(f"Fold: {fold}, num_train: {len(train_uids)}, num_val: {len(test_uids)}")

            wrapper.fit(X_train, y_train_df)
            risk_test = wrapper.predict(X_test)
            oof.loc[test_uids] = risk_test
            cox_index_train = evaluate(y_train_df, y_train_df, risk=wrapper.predict(X_train), tau=7)
            cox_index_test = evaluate(y_train_df, y_test_df, risk=risk_test, tau=7)
            train_scores.append(cox_index_train)
            test_scores.append(cox_index_test)
            logger.info(f"train {cox_index_train}:.4f | val {cox_index_test}:.4f")

        tracking.end_run(run, test_scores, train_scores, oof, n_features=stateless_features.shape[1])

    except Exception:
        logger.exception("Run failed")
        logger.remove(run["sink"])
        raise    
#         X_train[['BM_BLAST', 'HB', 'PLT']] = imputer.fit_transform(X_train[['BM_BLAST', 'HB', 'PLT']])
#         X_test[['BM_BLAST', 'HB', 'PLT']] = imputer.transform(X_test[['BM_BLAST', 'HB', 'PLT']])

#         status_train = target_train_df.loc[train_uids, 'OS_STATUS'].astype(bool).values
#         time_train = target_train_df.loc[train_uids, 'OS_YEARS'].clip(lower=1e-4).values
#         y_train_xgb = np.where(status_train, time_train, -time_train)
    
#         model = cox(alpha=1e-2)
# #         model = xgb.XGBRegressor(
# #             objective="survival:cox",
# #             eval_metric="cox-nloglik",
# #             n_estimators=600,          # Moins d'arbres (ou utilise early stopping)
# #             learning_rate=0.03,        # Reste sur un pas d'apprentissage doux
# #             max_depth=3,               # Réduit de 4 à 3 pour limiter la mémorisation
# #             min_child_weight=15,       # Clé : empêche d'isoler des patients uniques
# #             subsample=0.7,             # Bagging sur les patients
# #             colsample_bytree=0.6,      # Sous-échantillonnage des colonnes
# #             reg_lambda=3.0,            # Régularisation L2 forte
# #             reg_alpha=1.0,             # Régularisation L1 (sparsity)
# #             random_state=42
# # )
#         # model.fit(X_train, y_train_xgb)
#         model.fit(X_train, y_train)

#         cox_cindex_train = concordance_index_ipcw(y_train, y_train, model.predict(X_train), tau=7)[0]
#         cox_cindex_test = concordance_index_ipcw(y_train, y_test, model.predict(X_test), tau=7)[0]
#         print(f"Cox Proportional Hazard Model Concordance Index IPCW on train: {cox_cindex_train:.2f}")
#         print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.2f}")
if __name__ == "__main__":

    # run_training("/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/configs/baseline.yaml", [0,1,2,3,4])
    # run_training("/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/configs/molecular_baseline.yaml", [0,1,2,3,4])

    fire.Fire(run_experiment)