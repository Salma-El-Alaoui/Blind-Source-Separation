#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:00:09 2017

@author: camillejandot
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from data_utils import Image, ECG_data, Audio
from projection_utils import proj, orth_projection
from fastICA import fastICA
from jade import jadeR

"""
Script that enables to perform Multidimensional ICA, adapted from Cardoso's 
paper Multidimensional Independent Component Analysis (1)

- Available implementations of ICA:
    - algorithm = 'jade'
    - algorithm = 'fastICA'
- Available experiments:
    - experiment = 'ecg3' (3 channels of ecg, separating mother from baby's
    signal, same as in Cardoso's paper (1))
    
"""
#%%
# Defining parameters of the experiment
algorithm = 'jade'
experiment = 'audio'
verbose = False


# Running experiment
if experiment == 'ecg3':
    xlim = 1000
    data = ECG_data(verbose=verbose).load()
    channels_3 = np.asarray(data[:3])
    
    if algorithm == 'jade':
        unmixing_mat = np.asarray(jadeR(channels_3))
    elif algorithm == 'fastICA':
        unmixing_mat,_,_ = fastICA(channels_3)
    else:
        print('Algorithm ', algorithm, ' is not implemented: using jade')
    A_hat = np.linalg.inv(unmixing_mat)
    
    # Plotting results of ICA
    y = np.dot(unmixing_mat,channels_3)
    
    plt.figure(figsize=(15.0, 4.0))
    n_mixtures = 3
    for i in range(n_mixtures):
        plt.subplot(1, n_mixtures, i+1)
        plt.plot(y[i,:],linewidth=2)
        plt.xlim([0,xlim])
        plt.title('y for source ' + str(i))
        plt.suptitle("Recovered Sources with ICA")
    plt.show()
    
    
    if algorithm == 'jade':
        c_mother = [0, 1] 
        c_foetus = 2
    elif algorithm == 'fastICA':
        c_mother = [2, 1] 
        c_foetus = 0
    
    a_foetus = A_hat[:,c_foetus]
    Pi_f = 1/(np.linalg.norm(a_foetus))**2 * np.outer(a_foetus, a_foetus)
    
    a_mother = A_hat[:, c_mother]
    Pi_m = proj(a_mother)
    
    # Orthogonal projections
    list_Pi = [Pi_f,Pi_m]
    orth_projs = orth_projection(list_Pi)

    # MICA
    
    mica_mother = orth_projs[1].dot(channels_3)
    mica_foetus = orth_projs[0].dot(channels_3)
    
    plt.figure(figsize=(15.0, 4.0))
    for i in range(n_mixtures):
        plt.subplot(1, n_mixtures, i+1)
        plt.plot(mica_mother[i,:],linewidth=2)
        plt.xlim([0,xlim])
    plt.suptitle('Mother MICA Component')
    
    plt.figure(figsize=(15.0, 4.0))
    for i in range(n_mixtures):
        plt.subplot(1, n_mixtures, i+1)
        plt.plot(mica_foetus[i,:],linewidth=2)
        plt.xlim([0,xlim])
    plt.suptitle('Foetus MICA Component')

    