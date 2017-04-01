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
    - experiment = 'emma' (separating image of Emma Watson from picture of grass)
    
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
    
    # Orthogonal projection
    
    ## Attempt of KMeans to create 2 clusters - not satisfying
    #from sklearn.cluster import KMeans
    #import pywt
    #n_clusters = 2 
    #distance = 'euclidean'
    #y_dwt = []
    #for i in range(len(y)):
    #    cA, cD = pywt.dwt(y[i], 'haar')
    #    list(cA).extend(list(cD))
    #    y_dwt.append(cA)
    #    
    #y_clusters = KMeans(n_clusters=n_clusters).fit_predict(np.asarray(y_dwt))
    #clusters_belonging = []
    #
    #for cluster in range(n_clusters):
    #    clusters_belonging.append(np.where(y_clusters==cluster))
    
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

#%%

elif experiment == 'audio':
     #Load data
    audio_gl_path = '../data/audio/'
    paths = [audio_gl_path + 'LetItBe1.wav',audio_gl_path + 'LetItBe2.wav',
             audio_gl_path + 'LetItBe3.wav',audio_gl_path + 'Thunder1.wav',
             audio_gl_path + 'Thunder2.wav',audio_gl_path + 'Thunder3.wav',]
    audio = Audio(paths=paths)._load_tracks()
    mixture_1 = audio.mix_tracks(weights=[1./12,1./8,2./6,1./6,1./6,1./8],load=False)
    mixture_2 = audio.mix_tracks(load=False)
    mixture_3 = audio.mix_tracks(weights=[1./6,1./8,2./6,1./6,1./8,1./12],load=False)
    
    mixtures = np.array([mixture_1[:,0].flatten(),mixture_2[:,0].flatten(),mixture_3[:,0].flatten()])
    
    # Performing ICA
    if algorithm == 'jade':
        unmixing_mat = np.asarray(jadeR(mixtures))
    elif algorithm == 'fastICA':
        unmixing_mat,_,_ = fastICA(mixtures,n_iter=100)
        
    A_hat = np.linalg.inv(unmixing_mat)
    y = np.dot(unmixing_mat,mixtures)
    
        
    for i in range(3):
        plt.figure()
        plt.plot(y[i,:].reshape(mixture_1[:,0].shape))
        plt.title('y for source ' + str(i))

    audio_gl_path = '../data/audio/FastICAResult'
    wavfile.write(audio_gl_path + 'f_test_1.wav',rate=44100,data=y[0,:])
    wavfile.write(audio_gl_path + 'f_test_2.wav',rate=44100,data=y[1,:])
    wavfile.write(audio_gl_path + 'f_test_3.wav',rate=44100,data=y[2,:])
    
#    
#         #Orthogonal projections
#        
#    c_emma = [1,2]
#    c_grass = 0
#    
#    a_grass = A_hat[:,c_grass]
#    a_emma = A_hat[:, c_emma]
#    
#    Pi_emma = proj(a_emma)
#    Pi_grass = 1/(np.linalg.norm(a_grass))**2 * np.outer(a_grass, a_grass)
#  
#    
#    list_Pi = [Pi_grass,Pi_emma]
#    orth_projs = orth_projection(list_Pi)
#    
#    mica_emma = orth_projs[1].dot(mixtures)
#    mica_grass = orth_projs[0].dot(mixtures)
#    
#    wavfile.write(audio_gl_path + 'f_test_adc_1.wav',rate=44100,data=mica_grass[0])
#    wavfile.write(audio_gl_path + 'f_test_beatles_1.wav',rate=44100,data=mica_emma[0])
#    wavfile.write(audio_gl_path + 'f_test_adc_2.wav',rate=44100,data=mica_grass[0])
#    wavfile.write(audio_gl_path + 'f_test_beatles_2.wav',rate=44100,data=mica_emma[0])
#    wavfile.write(audio_gl_path + 'f_test_adc_3.wav',rate=44100,data=mica_grass[0])
#    wavfile.write(audio_gl_path + 'f_test_beatles_3.wav',rate=44100,data=mica_emma[0])
#    
#    wavfile.write(audio_gl_path + 'f_mixture_1_th_lib.wav',rate=44100,data=mixture_1[0])
#    wavfile.write(audio_gl_path + 'f_mixture_2_th_lib.wav',rate=44100,data=mixture_2[0])
#    wavfile.write(audio_gl_path + 'f_mixture_3_th_lib.wav',rate=44100,data=mixture_3[0])    