import fire
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import rankdata

from qrt.data import load
from qrt.features import build
from qrt.train.trainer import evaluate
from qrt.utils import tracking

RUNS_DIR = f"{tracking.ROOT}/runs"


def read_preds(run_id, name):
    return pd.read_csv(f"{RUNS_DIR}/{run_id}/{name}", index_col=0).iloc[:, 0]


def rank_in_groups(s, groups):
    out = pd.Series(np.nan, index=s.index)
    for g in groups.unique():
        idx = groups.index[groups == g]
        out.loc[idx] = rankdata(s.loc[idx]) / len(idx)
    return out


def run_ensemble(cfg_path, submission=True):
    cfg = build.load_config(cfg_path)
    run = tracking.start_run(cfg, cfg_path)
    try:
        runs, w = list(cfg.ensemble.runs), np.asarray(cfg.ensemble.weights, float)
        assert len(runs) == len(w), "un poids par run"
        w = w / w.sum()

        y = load.load(f"{cfg.data.raw_dir}/train/target_train.csv", dropna=True)
        clinical_train_df = load.load(f"{cfg.data.raw_dir}/train/clinical_train.csv", dropna=True)
        fold_df = load.make_stratified_fold_group_centered(y, clinical_train_df=clinical_train_df, n_bins=4, seed=cfg.experiment.seed)
        folds = fold_df["fold"].loc[y.index]

        oof = sum(wi * rank_in_groups(read_preds(r, "oof.csv").loc[y.index], folds) for wi, r in zip(w, runs))

        val_scores = []
        for f in sorted(folds.unique()):
            tr, va = folds.index[folds != f], folds.index[folds == f]
            val_scores.append(evaluate(y.loc[tr], y.loc[va], oof.loc[va].values, tau=7))
            logger.info(f"fold {f} | val {val_scores[-1]:.4f}")

        risk_test = None
        if submission:
            subs = [read_preds(r, "submission.csv") for r in runs]
            idx = subs[0].index
            one_group = pd.Series(0, index=idx)
            risk_test = sum(wi * rank_in_groups(s.loc[idx], one_group) for wi, s in zip(w, subs)).rename("risk_score")

        tracking.end_run(run=run,
                          val_scores=val_scores,
                            train_scores=[np.nan], 
                            oof=oof, 
                            n_features=len(runs), 
                            risk_test=risk_test)
        
    except Exception:
        logger.exception("Ensemble failed")
        logger.remove(run["sink"])
        raise


if __name__ == "__main__":
    fire.Fire(run_ensemble)