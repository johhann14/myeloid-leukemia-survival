#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-09-28 20:59:16
Author: joh
"""


from qrt.train import trainer

import fire


CONFIGS_ROOT = "/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/configs"

def main(cfg_file: str, fold=[0,1,2,3,4]):
    trainer.run_training(cfg_yaml=f"{CONFIGS_ROOT}/{cfg_file}", fold=fold)

if __name__ == "__main__":
    fire.Fire(main)