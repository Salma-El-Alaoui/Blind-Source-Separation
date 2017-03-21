#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:00:09 2017

@author: camillejandot
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import fastica


# Load ECG data
df = pd.read_table('../data/ecg/foetal_ecg.dat', sep="\s+", header=None)

# Display the 8 channels
channels = []
for i in range(8):
    channel = df[i+1]
    channels.append(channel)
    plt.figure()
    plt.plot(channel[:500])
    plt.title('Channel ' + str(i+1))
#%%
X = channels[:3]
W = fastica(n_components=2).fit(X).components_
print(W.shape)
