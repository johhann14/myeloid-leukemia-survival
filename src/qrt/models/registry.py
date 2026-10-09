#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-09-28 16:34:13
Author: joh
"""


from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.impute import SimpleImputer
from omegaconf import OmegaConf

from qrt.data import load
from qrt.features import build
from sksurv.metrics import concordance_index_censored , concordance_index_ipcw
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import numpy as np



class WrapperRSF():

    def __init__(
            self, 
            n_estimators=500, 
            min_samples_leaf=15, 
            min_samples_split=10, 
            max_features="sqrt",
            max_depth=None, 
            impute="median", 
            n_jobs=6,
            random_state=0
    ):
        self.model = RandomSurvivalForest(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features,
                                n_jobs=n_jobs,
                                random_state=random_state,
                                low_memory=True
        )
        self.imp = SimpleImputer(strategy=impute)

    def _transform(self, X, fit=False):
        X = X.astype("float32")
        return self.imp.fit_transform(X) if fit else self.imp.transform(X)

    def fit(self, X, y_df):
        self.columns = list(X.columns)
        y = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_df)
        self.model.fit(self._transform(X, fit=True), y)
        return self


    def predict(self, X):
        return self.model.predict(self._transform(X[self.columns]))
        

class WrapperXGBAFT:
    """
    Modelise the time of the event
    """
    def __init__(self, n_estimators=600, learning_rate=0.02, max_depth=3, min_child_weight=15,
                 subsample=0.8, colsample_bytree=0.6, reg_lambda=3.0, dist="logistic", scale=1.2, seeds=(0,)):
        self.n_estimators = n_estimators
        self.params = {"objective": "survival:aft", "aft_loss_distribution": dist,
                       "aft_loss_distribution_scale": scale, "eta": learning_rate, "max_depth": max_depth,
                       "min_child_weight": min_child_weight, "subsample": subsample,
                       "colsample_bytree": colsample_bytree, "lambda": reg_lambda}

        self.seeds = list(seeds)

    def fit(self, X, y_df):
        t = y_df["OS_YEARS"].clip(lower=1e-3).values
        dtrain = xgb.DMatrix(X)                                    
        dtrain.set_float_info("label_lower_bound", t)
        dtrain.set_float_info("label_upper_bound", np.where(y_df["OS_STATUS"].values == 1, t, np.inf))
        self.columns = list(X.columns)
        self.boosters = [
            xgb.train({**self.params, "seed": s}, dtrain, num_boost_round=self.n_estimators) 
            for s in self.seeds
        ]
        return self

    def predict(self, X):
        d = xgb.DMatrix(X[self.columns].astype("float32"))
        return -np.mean([b.predict(d, output_margin=True) for b in self.boosters], axis=0)                                       
                                       

class WrapperCoxPHModel:

    def __init__(self, alpha, impute, scale, columns):
        self.imp = SimpleImputer(strategy=impute) if impute else None
        self.scale = StandardScaler() if scale else None
        self.model = CoxPHSurvivalAnalysis(alpha=alpha)
        self.columns = columns

    def _transform(self, X, fit=False):
        X = X.copy()
        cols = X.columns if self.columns == 'all' else list(self.columns)
        if self.imp:
            X[cols] = self.imp.fit_transform(X[cols]) if fit else self.imp.transform(X[cols])
        if self.scale:
            X = self.scale.fit_transform(X) if fit else self.scale.transform(X)
        return X

    def fit(self, X, y_df):
        X_preprocessed = self._transform(X, fit=True)
        y = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', y_df)

        self.model.fit(X_preprocessed, y)
        return self


    def predict(self, X):
        X_preprocessed = self._transform(X, fit=False)
        return self.model.predict(X_preprocessed)


class WrapperXGBCox:

    def __init__(self, n_estimators=500, learning_rate=0.02, max_depth=3, min_child_weight=15,
                 subsample=0.8, colsample_bytree=0.6, reg_lambda=3.0, seeds=(0, 1, 2)):
        self.params = dict(objective="survival:cox", n_estimators=n_estimators, learning_rate=learning_rate,
                           max_depth=max_depth, min_child_weight=min_child_weight, subsample=subsample,
                           colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, n_jobs=6)
        self.seeds = list(seeds)
        self.models = []

    @staticmethod
    def _signed_time(y_df):
        t = y_df["OS_YEARS"].clip(lower=1e-4).values                
        return np.where(y_df["OS_STATUS"].values == 1, t, -t)        


    def fit(self, X, y_df):
        self.columns = list(X.columns)
        y = self._signed_time(y_df)
        self.models = [xgb.XGBRegressor(random_state=s, **self.params).fit(X, y) for s in self.seeds]
        return self

    def predict(self, X):
        X = X[self.columns]
        return np.mean([np.log(m.predict(X)) for m in self.models], axis=0)



class WrapperCoxNet:
    def __init__(self, alpha, l1_ratio=0.5, max_iter=100000, impute="median", scale=True, columns="all"):
            self.imp = SimpleImputer(strategy=impute) if impute else None
            self.scale = StandardScaler() if scale else None
            self.model = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=l1_ratio, max_iter=max_iter)
            self.columns = columns
    
    def _transform(self, X, fit=False):
        X = X.copy()
        cols = X.columns if self.columns == 'all' else list(self.columns)
        if self.imp:
            X[cols] = self.imp.fit_transform(X[cols]) if fit else self.imp.transform(X[cols])
        if self.scale:
            X = self.scale.fit_transform(X) if fit else self.scale.transform(X)
        return X

    def fit(self, X, y_df):
        X_preprocessed = self._transform(X, fit=True)
        y = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', y_df)

        self.model.fit(X_preprocessed, y)
        return self


    def predict(self, X):
        X_preprocessed = self._transform(X, fit=False)
        return self.model.predict(X_preprocessed)



MODELS = {"cox": WrapperCoxPHModel,
          "coxnet": WrapperCoxNet,
          "xgb_aft": WrapperXGBAFT,
          "xgb_cox": WrapperXGBCox,
          "rsf": WrapperRSF,
        }

def build_wrapper(cfg):
    return MODELS[cfg.model.name](**OmegaConf.to_container(cfg.model.params))


if __name__ == "__main__":
    cfg = build.load_config(yaml_path='/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/configs/molecular_baseline.yaml')


    wrapper = build_wrapper(cfg)
    print(f"OK")