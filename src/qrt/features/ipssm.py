#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-09-30 15:46:12
Author: joh
"""


import pandas as pd
import numpy as np
from qrt.data import load
import json  


def clinical_variable(clinical_df: pd.DataFrame) -> pd.DataFrame:
    clinical_df = clinical_df.copy()
    bm_blast_series = clinical_df["BM_BLAST"]
    # bm_blast_series = bm_blast_series.apply
    bm_blast_series = bm_blast_series.clip(upper=20) / 5
    plt_series = clinical_df["PLT"]
    # plt_series = plt_series.apply(lambda x: min(x, 250) /100)
    plt_series = plt_series.clip(upper=250) /100

    hb_series = clinical_df["HB"]
    # hb_series = hb_series.apply(lambda x: min(max(4, x), 20))
    hb_series = hb_series.clip(lower=4, upper=20)


    return pd.DataFrame(data={
                            "BLAST5": bm_blast_series,
                            "TRANSF_PLT100": plt_series,
                            "HB1": hb_series,
                        },
                        index=clinical_df.index
    )




GENE_STANDALONE = [
    "NPM1",
    "RUNX1",
    "NRAS",
    "ETV6",
    "IDH2",
    "CBL",
    "EZH2",
    "U2AF1",
    "SRSF2",
    "DNMT3A",
    "ASXL1",
    "KRAS"
]

GENE_RESIDUAL = [
    "BCOR",
    "BCORL1",
    "CEBPA",
    "ETNK1",
    "GATA2",
    "GNB1",
    "IDH1",
    "NF1",
    "PHF6",
    "PPM1D",
    "PRPF8",
    "PTPN11",
    "SETBP1",
    "STAG2",
    "WT1"
]
def gene_standalone(molecular_df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    molecular_df = molecular_df.copy()
    m = molecular_df[molecular_df["GENE"].isin(GENE_STANDALONE)]

    flags = pd.crosstab(m.index, m["GENE"]).clip(upper=1).reindex(columns=GENE_STANDALONE, fill_value=0).add_prefix("mut_")
    flags = flags.reindex(index=index, fill_value=0)

    mll_ptd = molecular_df[(molecular_df["GENE"] == "MLL") & (molecular_df["EFFECT"]=="PTD")]
    mll_ptd_ids = set(mll_ptd.index)
    flags["mut_MLL_PTD"] = (flags.index.isin(mll_ptd_ids)).astype(int) 


    # flt3_itd_tkd = molecular_df[(molecular_df["GENE"] == "FLT3") & (molecular_df["EFFECT"].isin(["ITD", "TKD"]))]
    # flt3_itd_tkd_ids = set(flt3_itd_tkd.index)
    # flags["mut_FLT3_ITD_TKD"] = (flags.index.isin(flt3_itd_tkd_ids)).astype(int)

    flt3 = molecular_df[molecular_df["GENE"] == "FLT3"]
    pc = flt3["PROTEIN_CHANGE"].fillna("")
    pos = pc.str.extract(r"p\.[A-Z*]?(\d+)", expand=False).astype(float)
    is_itd = (flt3["EFFECT"] == "ITD") | (flt3["EFFECT"].isin(["inframe_codon_gain", "inframe_variant"]) & pos.between(570, 620))
    is_tkd = pos.isin([835, 836, 841, 842])
    flags["mut_FLT3_ITD_TKD"] = flags.index.isin(flt3[is_itd | is_tkd].index).astype(int)


    residuals = molecular_df[molecular_df["GENE"].isin(GENE_RESIDUAL)]
    residuals = residuals.reset_index(level=0)
    # ID1 GENE 1 GENE 2
    residuals = residuals.groupby("ID")["GENE"].nunique().clip(upper=2)
    residuals = residuals.reindex(index=index, fill_value=0).rename("mut_residuals")


    return pd.concat((flags, residuals), axis=1)



# TP53 multi
# FLT3 ITD TKD
# SF3B15Q

def gene_cyto_dependant(molecular_df: pd.DataFrame, cyto_df: pd.DataFrame):

    # TP53 multi
    tp53 = molecular_df[molecular_df["GENE"] == "TP53"]
    n_tp53 = tp53.groupby(level=0).size().reindex(cyto_df.index, fill_value=0)

    tp53_vaf_max = tp53.groupby(level=0)['VAF'].max().reindex(index=cyto_df.index, fill_value=0)

    tp53_vaf_flag = tp53_vaf_max > 0.55


    tp53_multi = ( ((n_tp53 >=2) | ((n_tp53 >=1))
                    &
                 (tp53_vaf_flag | (cyto_df["cyto_abn17p"] == 1))))



    #SF3B1_5q

    sf3b1 = molecular_df[molecular_df["GENE"] == "SF3B1"]
    sf3b1_ids = set(sf3b1.index)

    sf3b1 = pd.Series(1, index=sf3b1_ids).reindex(index=cyto_df.index, fill_value=0)
    # print(sf3b1)
    del_7 = (cyto_df["cyto_del7q"] == 1) | (cyto_df["cyto_minus7"] == 1)

    sf3b1_5q = (sf3b1) & (cyto_df["cyto_del5q"] == 1) & (~del_7) & (cyto_df["cyto_complex"]==0)
    # print(sf3b1_5q)
    # print(sf3b1_5q.value_counts(dropna=False))
    #TP53 multi
    comut_df = molecular_df["GENE"].isin(["SRSF2", "STAG2", "BCOR", "BCORL1", "RUNX1", "NRAS"])

    comut = comut_df.groupby(level=0).any()
    comut = comut.reindex(index=cyto_df.index, fill_value=False)
    del_5q = cyto_df["cyto_del5q"] == 1
    sf3b1_alpha = sf3b1 & ~comut & ~del_5q

    return pd.DataFrame(
        data={
            "TP53_multi": tp53_multi.astype(int),
            "SF3B1_5q": sf3b1_5q.astype(int),
            "SF3B1_alpha": sf3b1_alpha.astype(int),
         
        },
        index=cyto_df.index
    )

    


BETA = {
    "BLAST5": 0.352,           # min(BM_BLAST, 20) / 5
    "TRANSF_PLT100": -0.222,   # min(PLT, 250) / 100
    "HB1": -0.171,             
    "CYTOVEC": 0.287,        

    "TP53multi": 1.18,
    "MLL_PTD": 0.798,
    "FLT3": 0.798,            
    "SF3B1_5q": 0.504,
    "NPM1": 0.430,
    "RUNX1": 0.423,
    "NRAS": 0.417,
    "ETV6": 0.391,
    "IDH2": 0.379,
    "CBL": 0.295,
    "EZH2": 0.270,
    "U2AF1": 0.247,
    "SRSF2": 0.239,
    "DNMT3A": 0.221,
    "ASXL1": 0.213,
    "KRAS": 0.202,
    "SF3B1_alpha": -0.0794,
    "nRes2": 0.231,            
}

MEAN = {
    "BLAST5": 0.922,
    "TRANSF_PLT100": 1.4,
    "HB1": 9.87,
    "CYTOVEC": 1.39,
    "TP53multi": 0.0710,
    "MLL_PTD": 0.0247,
    "FLT3": 0.0108,
    "SF3B1_5q": 0.0166,
    "NPM1": 0.0112,
    "RUNX1": 0.1260,
    "NRAS": 0.0362,
    "ETV6": 0.0216,
    "IDH2": 0.0429,
    "CBL": 0.0473,
    "EZH2": 0.0588,
    "U2AF1": 0.0866,
    "SRSF2": 0.1580,
    "DNMT3A": 0.1610,
    "ASXL1": 0.2520,
    "KRAS": 0.0271,
    "SF3B1_alpha": 0.1860,
    "nRes2": 0.3880,
}

RENAME = {
    "mut_FLT3_ITD_TKD": "FLT3",
    "mut_MLL_PTD": "MLL_PTD",
    "mut_residuals": "nRes2",
    "TP53_multi": "TP53multi",
    "cyto_ipssr": "CYTOVEC",
}



def ipssm_score(ipssm_variables: dict) -> float:
    import numpy as np
    s = 1.15467

    for k,v in ipssm_variables.items():
        k = k.replace("mut_", "")
        beta = BETA[k]
        mean = MEAN[k]

        s += beta * (v-mean) / np.log(2)
    
    return s

def ipssm_score(x: pd.DataFrame) -> pd.Series:
    x = x.rename(columns=RENAME).rename(columns=lambda c: c.replace("mut_", ""))
    missing = set(BETA) - set(x.columns)
    assert not missing, f"IPSSM var missing : {missing}"

    terms = sum(BETA[k] * (x[k] - MEAN[k]).fillna(0) for k in BETA)   
    return (terms / np.log(2)).rename("ipssm_score")


IPSSM_EXTRA = ["TP53multi", "FLT3", "MLL_PTD", "SF3B1_5q", "SF3B1_alpha", "nRes2", "NPM1", "ETV6"]


def ipssm_features(clinical_df, molecular_df, cyto_df, cfg) -> pd.DataFrame:
    x = pd.concat([
        clinical_variable(clinical_df),
        gene_standalone(molecular_df, clinical_df.index),
        gene_cyto_dependant(molecular_df, cyto_df),
        cyto_df[["cyto_ipssr"]],
    ], axis=1)
    out = ipssm_score(x).to_frame()

    if cfg.get("keep_variables", False):
        x = x.rename(columns=RENAME).rename(columns=lambda c: c.replace("mut_", ""))
        out = out.join(x[IPSSM_EXTRA].add_prefix("ipssm_"))
    return out

if __name__ == "__main__":
    from qrt.features import build, cytogenetics

    cfg = build.load_config(yaml_path="/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/configs/baseline.yaml")
    # molecular_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/molecular_train.csv", dropna=True)
    # clinical_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/clinical_train.csv", dropna=True)

    # cyto_df = cytogenetics.parsing_cytogenetics_v1(clinical_df)

    # a = clinical_variable(clinical_df=clinical_df)
    # b = gene_standalone(molecular_df=molecular_df, index=clinical_df.index)
    # c = gene_cyto_dependant(molecular_df, cyto_df)
    # d = cyto_df["cyto_ipssr"]
    # ipssm_df = pd.concat((a,b,c,d), axis=1)
    # print(ipssm_df.columns)
    # s = ipssm_score(ipssm_df).rename("risk_score")
    # print(s)
    # s.to_csv("/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/data/preprocessed/train_ipssm.csv")

    # risk = s
    # target_df = load.load(f"{cfg.data.raw_dir}/train/target_train.csv", dropna=True)
    # fold_df = load.make_stratified_fold(target_df, clinical_train_df=clinical_df, n_bins=4, seed=cfg.experiment.seed)

    # scores = []

    # from qrt.train.trainer import evaluate
    # for f in sorted(fold_df["fold"].unique()):
    #     tr = fold_df[fold_df["fold"] != f].index
    #     val= fold_df[fold_df["fold"] == f].index
    #     scores.append(evaluate(target_df.loc[tr], target_df.loc[val], risk.loc[val].values, tau=7))

    # print(np.mean(scores), np.round(scores, 4))

    # TEST

    molecular_df = load.load(csv_path=f"{cfg.data.raw_dir}/test/molecular_test.csv", dropna=False)
    clinical_df = load.load(csv_path=f"{cfg.data.raw_dir}/test/clinical_test.csv", dropna=False)

    cyto_df = cytogenetics.parsing_cytogenetics_v1(clinical_df)

    a = clinical_variable(clinical_df=clinical_df)
    b = gene_standalone(molecular_df=molecular_df, index=clinical_df.index)
    c = gene_cyto_dependant(molecular_df, cyto_df)
    d = cyto_df["cyto_ipssr"]
    ipssm_df = pd.concat((a,b,c,d), axis=1)
    print(ipssm_df.columns)
    s = ipssm_score(ipssm_df).rename("risk_score")
    print(s)
    s.to_csv("/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/data/preprocessed/test_ipssm.csv")

    # risk = s
    # target_df = load.load(f"{cfg.data.raw_dir}/train/target_train.csv", dropna=True)
    # fold_df = load.make_stratified_fold(target_df, clinical_train_df=clinical_df, n_bins=4, seed=cfg.experiment.seed)

    # scores = []

    # from qrt.train.trainer import evaluate
    # for f in sorted(fold_df["fold"].unique()):
    #     tr = fold_df[fold_df["fold"] != f].index
    #     val= fold_df[fold_df["fold"] == f].index
    #     scores.append(evaluate(target_df.loc[tr], target_df.loc[val], risk.loc[val].values, tau=7))

    # print(np.mean(scores), np.round(scores, 4))

    
    
        