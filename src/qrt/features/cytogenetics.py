#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-09-23 15:30:51
Author: joh
"""

#!/usr/bin/env python3
# coding: utf-8

"""
Creation date: 2026-09-21 19:00:20
Author: joh
"""

import pandas as pd
import numpy as np

from qrt.data import load
from pathlib import Path
import re

def parsing_cytogenetics_v0(df: pd.DataFrame) -> pd.DataFrame:
    cyto_series = df['CYTOGENETICS'].fillna('Unknown').astype(str).str.strip().str.upper()

    new_df = pd.DataFrame(index=df.index)

    new_df['cyto_is_unknown'] = cyto_series.eq('UNKNOWN').astype(int)

    new_df['cyto_is_normal'] = cyto_series.str.contains(
        r'^46,\s*(XX|XY)\s*(?:\[\d+\])?$', regex=True
    ).astype(int)

    new_df['cyto_sex_karyotype'] = np.where(
        cyto_series.str.contains('XX'), 'F',
        np.where(cyto_series.str.contains('XY'), 'M', 'Unknown')
    )
    new_df = pd.get_dummies(new_df, columns=['cyto_sex_karyotype'], prefix='cyto_sex', drop_first=False)

    new_df['cyto_chromosome_count'] = cyto_series.str.extract(r'^(\d{2})')[0].astype(float)
    new_df['cyto_chromosome_count'] = new_df['cyto_chromosome_count'].fillna(46.0)

    new_df['cyto_t_8_21'] = cyto_series.str.contains(r'T\(8;21\)').astype(int)
    new_df['cyto_inv_16'] = cyto_series.str.contains(r'INV\(16\)').astype(int)
    new_df['cyto_t_15_17'] = cyto_series.str.contains(r'T\(15;17\)').astype(int)

    new_df['cyto_monosomy_7'] = cyto_series.str.contains(r'(?<!\w)-7\b|\bDEL\(7\)').astype(int)
    new_df['cyto_monosomy_5'] = cyto_series.str.contains(r'(?<!\w)-5\b|\bDEL\(5\)').astype(int)
    new_df['cyto_trisomy_8'] = cyto_series.str.contains(r'\+8\b').astype(int)

    is_labeled_complex = cyto_series.eq('COMPLEX')

    def clone_complexity(s: str) -> int:
        if s in ('UNKNOWN', 'COMPLEX'):
            return 0
        max_tokens = 0
        for clone in s.split('/'):
            tokens = [t for t in clone.split(',') if t.strip()]
            abn_tokens = tokens[2:] if len(tokens) > 2 else []
            max_tokens = max(max_tokens, len(abn_tokens))
        return max_tokens

    new_df['cyto_complexity_score'] = cyto_series.apply(clone_complexity)
    new_df['cyto_is_complex'] = ((new_df['cyto_complexity_score'] >= 3) | is_labeled_complex).astype(int)
    # return new_df["cyto_complexity_score"]
    return new_df



WORDS = {"NORMAL": "46,XX", "COMPLEX": "COMPLEX",
         "COMPLEXKARYOTYPE": "COMPLEX", "COMPLEXKARYPTYPE": "COMPLEX"}

def normalize(s) -> str:
    if pd.isna(s):
        return "UNKNOWN"
    s = str(s).upper().replace(" ", "")
    # s = re.sub(r"\[[\d/]+\]|\{[\d/]+\}", "", s)
    s = re.sub("\[[^\]]*\]", "", s)
    s = re.sub(r"\{[^}]*\}", "", s)
    if re.fullmatch(r'[A-Z]+', s):
        return WORDS.get(s, "UNKNOWN")
    return s

def normalize_v1(s) -> str:
    if pd.isna(s):
        return "UNKNOWN"
    s = str(s).upper()
    s = re.sub(r"\s*\bWITH\b\s*", ",", s)
    s = s.replace(" ", "")
    # s = re.sub(r"\[[\d/]+\]|\{[\d/]+\}", "", s)
    s = re.sub("\[[^\]]*\]", "", s)
    s = re.sub(r"\{[^}]*\}", "", s)
    if re.fullmatch(r'[A-Z]+', s):
        return WORDS.get(s, "UNKNOWN")
    return s

def make_set_karyotype_anomalies(s: str) -> set:  
# Orignal: 47~49,xy,+21,+21,+22[cp11]/46,xy[10]
# normalised: 47~49,XY,+21,+21,+22/46,XY

    if s in ('UNKNOWN', "COMPLEX"):
        return set()

    out = set()

    for clone in s.split('/'):
        tokens = clone.split(',')
        # print(tokens)
        # tokens_anomalies = tokens[2:] # skip n_chromosome, sexual_chromose
        
        # for token in tokens_anomalies:
        #     out.add(token)
        for token in tokens[1:]:
            if token in ["IDEM", "SL", "SDL"] or re.fullmatch(r"X{1,3}Y{0,2}|XY", token):
                continue
            out.add(token)
            
                       

    return out

def has(abn: set, pattern: str) -> bool:
    return any(re.fullmatch(pattern, t) for t in abn)

def lesion_flags(abn: set) -> dict:
    return {
        "minus_y":  has(abn, r"-Y"),
        "del11q":   has(abn, r"DEL\(11\)\(Q.*"),
        "del5q":    has(abn, r"DEL\(5\)\(Q.*|DEL\(5Q\)"),
        "del12p":   has(abn, r"DEL\(12\)\(P.*"),
        "del20q":   has(abn, r"DEL\(20\)\(Q.*|DEL\(20Q\)"),
        "del7q":    has(abn, r"DEL\(7\)\(Q.*|DEL\(7Q\)"),
        "plus8":    has(abn, r"\+8"),
        "plus19":   has(abn, r"\+19"),
        "i17q":     has(abn, r"I\(17\)\(Q.*|I\(17Q\)"),
        "minus7":   has(abn, r"-7"),
        "abn3q":    has(abn, r"INV\(3\).*|T\(3;\d+\).*|T\(\d+;3\).*|DEL\(3\)\(Q.*"),
        "abn17p": has(abn, r"-17|DEL\(17\)\(P.*|DEL\(17P\).*|I\(17\)\(Q.*|I\(17Q\).*"),
    }

def ipssr_category(cytogenetic: str, anormalities: set) -> float:
    if cytogenetic == "UNKNOWN":
        return np.nan
    n = len(anormalities)
    f = lesion_flags(anormalities)

    if cytogenetic == "COMPLEX" or n >3:
        return 4
    if n==3:
        return 3
    if f["minus7"] or f["abn3q"]:                   
        return 3
    if n == 2 and f["del7q"]:                       
        return 3
    if n == 0:                                      
        return 1
    if n == 1:
        if f["minus_y"] or f["del11q"]:             
            return 0
        if f["del5q"] or f["del12p"] or f["del20q"]:  
            return 1
        return 2                                    
    if f["del5q"]:                                  
        return 1
    return 2
    
def parsing_cytogenetics_v1(df: pd.DataFrame) -> pd.DataFrame:
    # GROUP 
    # very good: -Y, del(11q)
    # good: Normal, del(5q), del(12p), del(20q), double including del(5q)
    # intermediate: del(7q), +8, +19, i(17q), any other single or double independent clones
    # poor: −7, inv(3)/t(3q)/del(3q), double including −7/del(7q), complex: 3 abnormalities
    # very poor: Complex: > 3 abnormalities
    
    cytogenetics_normalized = df["CYTOGENETICS"].map(normalize)
    karyotype_anomalies = cytogenetics_normalized.map(make_set_karyotype_anomalies)

    out = pd.DataFrame(index=df.index)
    out["cyto_is_unknown"] = (cytogenetics_normalized == "UNKNOWN").astype(int)
    out["cyto_n_abn"] = karyotype_anomalies.map(len)
    out["cyto_complex"] =  ((out["cyto_n_abn"] >= 3) | (cytogenetics_normalized == "COMPLEX")).astype(int)


    tmp = karyotype_anomalies.map(lesion_flags) # une serie qui a un dict a chaque eleemnt 
    tmp = tmp.to_list() # list of dict

    flags = pd.DataFrame(tmp, index=df.index)#each dict is a row of the df et each key is a col

    out = out.join(flags.add_prefix("cyto_"))
    out["cyto_ipssr"] = [ipssr_category(iscn, anormality) for iscn, anormality in zip(cytogenetics_normalized, karyotype_anomalies)]
    return out

def parsing_cytogenetics_v2(df: pd.DataFrame) -> pd.DataFrame:
    # GROUP 
    # very good: -Y, del(11q)
    # good: Normal, del(5q), del(12p), del(20q), double including del(5q)
    # intermediate: del(7q), +8, +19, i(17q), any other single or double independent clones
    # poor: −7, inv(3)/t(3q)/del(3q), double including −7/del(7q), complex: 3 abnormalities
    # very poor: Complex: > 3 abnormalities
    
    cytogenetics_normalized = df["CYTOGENETICS"].map(normalize_v1)
    karyotype_anomalies = cytogenetics_normalized.map(make_set_karyotype_anomalies)

    out = pd.DataFrame(index=df.index)
    out["cyto_is_unknown"] = (cytogenetics_normalized == "UNKNOWN").astype(int)
    out["cyto_n_abn"] = karyotype_anomalies.map(len)
    out["cyto_complex"] =  ((out["cyto_n_abn"] >= 3) | (cytogenetics_normalized == "COMPLEX")).astype(int)


    tmp = karyotype_anomalies.map(lesion_flags) # une serie qui a un dict a chaque eleemnt 
    tmp = tmp.to_list() # list of dict

    flags = pd.DataFrame(tmp, index=df.index)#each dict is a row of the df et each key is a col

    out = out.join(flags.add_prefix("cyto_"))
    out["cyto_ipssr"] = [ipssr_category(iscn, anormality) for iscn, anormality in zip(cytogenetics_normalized, karyotype_anomalies)]
    return out

parser = {
            "v0": parsing_cytogenetics_v0,
            "v1": parsing_cytogenetics_v1,
            "v2": parsing_cytogenetics_v2,
        }


def cytogenetics_features(df: pd.DataFrame, cfg):

    return parser[cfg.version](df)

if __name__ == "__main__":

    # csv_path = f'{Path(__file__).parent.parent.parent}/input_data/X_train/clinical_train.csv'
    csv_path = "/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/data/raw/train/clinical_train.csv"
    csv_path = '/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/data/raw/test/clinical_test.csv'
    df = pd.read_csv(csv_path, index_col=0)
    # df = load.load(csv_path, dropna=True)
    df = load.load(csv_path, dropna=False)


    # new_df = parsing_cytogenetics_v0(df)
    # new_df.to_csv(f"/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/input_data/cytogenetics_train.csv")
    # print(f"ok")
    # out = parsing_cytogenetics_v1(df)
    # out.to_csv(f"/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/data/preprocessed/train_cytogenetics_v1.csv")
    ct = df["CYTOGENETICS"]
    print(f"OKOKOK")
    print()
    for i in ct:
        if "with" in str(i):
            print("\n")
            print(f"Orignal: {i}")
            print(f"normalised: {normalize_v1(i)}")
    # for i in ct:

    #     print(f"Orignal: {i}")
    #     i = normalize(i)
    #     print(f"tokens: {make_set_karyotype_anomalies(i)}")
