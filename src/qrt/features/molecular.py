#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-09-24 17:23:01
Author: joh
"""

import pandas as pd
import numpy as np
from qrt.features import build
from qrt.data import load
import json  

def get_top_gene_mutations(molecular_df: pd.DataFrame):
    """
    Get the top gene that concern the most unique patient
    """
    molecular_df = molecular_df.copy()
    molecular_df = molecular_df.reset_index()
    d = molecular_df.groupby("GENE")["ID"].nunique()

    top_genes = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
    # print(top_genes)
    top_genes = list(top_genes.keys())[:20]


    return top_genes


def get_top_gene_mutations_v1(molecular_df: pd.DataFrame) -> list:
    molecular_df = molecular_df.copy()
    molecular_df = molecular_df.reset_index()
    d = molecular_df.groupby("GENE")["ID"].nunique()
   
    return d.index[d>=20].to_list()

def write_top_gene_mutations(top_genes: list, dst_path: str) -> None:
    with open(dst_path, "w") as f:
        json.dump(top_genes, f, indent=4)

def load_top_gene_mutations(src_path: str) -> list:
    with open(src_path, "r") as f:
        d = json.load(f)

    return d

    


def molecular_features(molecular_df: pd.DataFrame, index: pd.Index, cfg) -> pd.DataFrame:

    # top_genes = get_top_gene_mutations(molecular_df)

    top_genes = load_top_gene_mutations(cfg.top_genes)
    # print(top_genes)
    m = molecular_df[molecular_df["GENE"].isin(top_genes)]

    # rows : m.index, col: GENE, [i,j] = number of occurence 
    flags = pd.crosstab(m.index, m["GENE"]).clip(upper=1).reindex(columns=top_genes, fill_value=0).add_prefix("mut_")
    # print(flags )

    # this pattern : 
    # id1 GENE1 val
    #    GENE2 val

    # id2 GENE 2 val 
    #     GENE 7 val

    # 2D indexing -> calcule the stats for this group and then unpack it 
    id_gene_df = m.groupby([m.index, "GENE"])

    # id_gene_vaf_mean = id_gene_df["VAF"].mean().unstack().add_prefix("vaf_mean_")
    id_gene_vaf_max = id_gene_df["VAF"].max().unstack().reindex(columns=top_genes).add_prefix("vaf_max_")


    id_df = molecular_df.groupby(level=0)
    stats = pd.DataFrame({
        "mol_n_mut": id_df.size(),
        "mol_n_genes": id_df["GENE"].nunique(),
        "mol_max_vaf": id_df["VAF"].max(),
    })

    out = pd.concat([flags, id_gene_vaf_max, stats], axis=1).reindex(index=index)
    out = out.fillna(0)
    out["mol_no_mutation"] = (out["mol_n_mut"]  == 0).astype(int)

    return out


if __name__ == "__main__":


    cfg = build.load_config(yaml_path="/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/configs/baseline.yaml")
    clinical_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/clinical_train.csv", dropna=True)
    molecular_df = load.load(csv_path=f"{cfg.data.raw_dir}/train/molecular_train.csv", dropna=True)

    # top_gene= get_top_gene_mutations(molecular_df)
    # write_top_gene_mutations(top_genes=top_gene, dst_path='/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/data/preprocessed/top_gene.json')

    # top_gene = load_top_gene_mutations("/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/data/preprocessed/top_gene.json")
    top_genes = get_top_gene_mutations_v1(molecular_df)
    # out = molecular_features(molecular_df=molecular_df, index=clinical_df.index)
    write_top_gene_mutations(top_genes=top_genes, dst_path='/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/data/preprocessed/top_gene_v1.json')
    # print(out[:3])
    # print(out.columns)
    # print(out["mut_BCOR"].value_counts(dropna=False))


    # out.to_csv("/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/data/preprocessed/molecular_stats.csv")





