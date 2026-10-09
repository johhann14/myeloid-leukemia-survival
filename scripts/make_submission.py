#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-09-28 22:06:10
Author: joh
"""

import fire
from qrt.train import trainer

CONFIGS_ROOT = "/home/joh/Sync/pro/data-challenge_ens/qrt_leukemia/configs"

def main(cfg_file: str):
    trainer.make_submission(cfg_yaml=f"{CONFIGS_ROOT}/{cfg_file}")


if __name__ == "__main__":
    fire.Fire(main)