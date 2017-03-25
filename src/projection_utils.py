#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:04:29 2017

@author: camillejandot
"""
import numpy as np

def proj(Ap):
    inv_inner = np.linalg.inv(Ap.T.dot(Ap))
    return Ap.dot(inv_inner).dot(Ap.T)

def orth_projection(list_Pi):
    orth_projs = []
    sum_Pi = np.zeros(list_Pi[0].shape)
    for Pi in list_Pi :
        sum_Pi += Pi
    inv = np.linalg.pinv(sum_Pi)
    for Pi in list_Pi :
        orth_projs.append(Pi.dot(inv))
    return orth_projs