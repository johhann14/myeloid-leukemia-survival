#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-10-03 17:05:11
Author: joh
"""

import pytest
from qrt.features import cytogenetics as cg


@pytest.mark.parametrize("raw, expected", [
    ("46,XY[20]/47,XY,+8[5]", "46,XY/47,XY,+8"),        
    ("45,XX,-7[15/20]/46,XX[5/20]", "45,XX,-7/46,XX"),
    ("Normal", "46,XX"),
    ("46,xy,del(20)(q12)[2]/46,xy[18]", "46,XY,DEL(20)(Q12)/46,XY")
])
def test_normalize_v1(raw, expected):
    assert cg.normalize_v1(raw) == expected

@pytest.mark.parametrize("cyto_string, karyotype", [
    ("46,XY,DEL(20)(Q12)/46,XY", {"DEL(20)(Q12)"})
])
def test_make_set_karyotype_anomalies(cyto_string, karyotype):
    assert cg.make_set_karyotype_anomalies(cyto_string) == karyotype
