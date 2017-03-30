#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:00:09 2017

@author: camillejandot
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from data_utils import Image, ECG_data
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
    
    data = ECG_data(verbose=verbose).load()
    channels_3 = np.asarray(data[:3])
    
    if algorithm == 'jade':
        unmixing_mat = np.asarray(jadeR(channels_3))
    elif algorithm == 'fastICA':
        unmixing_mat,_,_ = fastICA(channels_3.T)
    else:
        print('Algorithm ', algorithm, ' is not implemented: using jade')
    A_hat = np.linalg.inv(unmixing_mat)
    
    # Plotting results of ICA
    y = np.dot(unmixing_mat,channels_3)
    
    for i in range(3):
        plt.figure()
        plt.plot(y[i,:])
        plt.title('y for source ' + str(i))
        
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
        c_mother = [0, 2] 
        c_foetus = 1
    
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
    
    plt.figure()
    plt.plot(mica_mother.T,linewidth=0.6)
    plt.xlim([0,500])
    plt.title('mother')
    plt.figure()
    plt.plot(mica_foetus.T,linewidth=0.6)
    plt.xlim([0,500])
    plt.title('foetus')

elif experiment == 'emma':
    # Load data
    im_gl_path = '../data/image/'
    paths = [im_gl_path + 'grass.jpeg',im_gl_path + 'emma.jpeg']
    
    # Mixing images
    mixture_1 = Image(paths=paths).mix_images([0.5,0.5])/255.
    mixture_2 = Image(paths=paths).mix_images(weights=[0.7,0.3],verbose=1)/255.
    mixture_3 = Image(paths=paths).mix_images(weights=[0.35,0.65],verbose=1)/255.
    
    mixtures = np.array([mixture_1.flatten(),mixture_2.flatten(),mixture_3.flatten()])
    
    # Performing ICA
    if algorithm == 'jade':
        unmixing_mat, _,_ = jadeR(mixtures)
    elif algorithm == 'fastICA':
        unmixing_mat, _,_ = fastICA(mixtures.T,n_iter=100)
    unmixing_mat = np.asarray(jadeR(mixtures))
    A_hat = np.linalg.inv(unmixing_mat)
    y = np.dot(unmixing_mat,mixtures)
    
    for i in range(3):
        plt.figure()
        plt.imshow(y[i,:].reshape(mixture_1.shape),cmap='gray')
        plt.title('y for source ' + str(i))

    if algorithm == 'jade':
        c_emma = 0
        c_grass = [1,2]
    elif algorithm == 'fastICA':
        c_grass = [1,2]
        c_emma = 0
    
    # Orthogonal projections
    a_grass = A_hat[:,c_grass]
    a_emma = A_hat[:, c_emma]
    Pi_emma = 1/(np.linalg.norm(a_emma))**2 * np.outer(a_emma, a_emma)
    Pi_grass = proj(a_grass)
    
    list_Pi = [Pi_grass,Pi_emma]
    orth_projs = orth_projection(list_Pi)
    
    mica_emma = orth_projs[1].dot(mixtures)
    mica_grass = orth_projs[0].dot(mixtures)
    
    # Plotting final pictures
    plt.figure()
    plt.imshow(mica_emma[0].reshape(mixture_1.shape),cmap='gray')
    plt.title('emma')
    plt.figure()
    plt.imshow(mica_emma[1].reshape(mixture_1.shape),cmap='gray')
    plt.title('emma')
    plt.figure()
    plt.imshow(mica_emma[2].reshape(mixture_1.shape),cmap='gray')
    plt.title('emma')
    
    plt.figure()
    plt.imshow(mica_grass[0].reshape(mixture_1.shape),cmap='gray')
    plt.title('grass')
    plt.figure()
    plt.imshow(mica_grass[1].reshape(mixture_1.shape),cmap='gray')
    plt.title('grass')
    plt.figure()
    plt.imshow(mica_grass[2].reshape(mixture_1.shape),cmap='gray')
    plt.title('grass')

elif experiment == 'test':
    # Load data
    im_gl_path = '../data/image/'
    paths = [im_gl_path + 'grass.jpeg',im_gl_path + 'emma.jpeg',im_gl_path + 'lena.jpeg']
    
    # Mixing images
    mixture_1 = Image(paths=paths).mix_images([0.2,0.5,0.3])/255.
    mixture_2 = Image(paths=paths).mix_images(weights=[0.6,0.2,0.2],verbose=1)/255.
    mixture_3 = Image(paths=paths).mix_images(weights=[0.5,0.35,0.15],verbose=1)/255.
    
    mixtures = np.array([mixture_1.flatten(),mixture_2.flatten(),mixture_3.flatten()])
    
    # Performing ICA
    if algorithm == 'jade':
        unmixing_mat, _,_ = jadeR(mixtures)
    elif algorithm == 'fastICA':
        unmixing_mat, _,_ = fastICA(mixtures.T,n_iter=100)
    unmixing_mat = np.asarray(jadeR(mixtures))
    A_hat = np.linalg.inv(unmixing_mat)
    y = np.dot(unmixing_mat,mixtures)
    
    for i in range(3):
        plt.figure()
        plt.imshow(y[i,:].reshape(mixture_1.shape),cmap='gray')
        plt.title('y for source ' + str(i))

    if algorithm == 'jade':
        c_emma = 0
        c_grass = [1,2]
    elif algorithm == 'fastICA':
        c_grass = [1,2]
        c_emma = 0
        
    
    # Orthogonal projections
        

    c_emma = 0
    c_grass = 2
    c_lena = 1
    
    a_grass = A_hat[:,c_grass]
    a_emma = A_hat[:, c_emma]
    a_lena = A_hat[:, c_lena]
    
    Pi_emma = 1/(np.linalg.norm(a_emma))**2 * np.outer(a_emma, a_emma)
    Pi_grass = 1/(np.linalg.norm(a_grass))**2 * np.outer(a_grass, a_grass)
    Pi_lena = 1/(np.linalg.norm(a_lena))**2 * np.outer(a_lena, a_lena)
    
    list_Pi = [Pi_grass,Pi_emma,Pi_lena]
    orth_projs = orth_projection(list_Pi)
    
    mica_lena = orth_projs[2].dot(mixtures)
    mica_emma = orth_projs[1].dot(mixtures)
    mica_grass = orth_projs[0].dot(mixtures)
    
    # Plotting final pictures
    plt.figure()
    plt.imshow(mica_emma[0].reshape(mixture_1.shape),cmap='gray')
    plt.title('emma')
    plt.figure()
    plt.imshow(mica_emma[1].reshape(mixture_1.shape),cmap='gray')
    plt.title('emma')
    plt.figure()
    plt.imshow(mica_emma[2].reshape(mixture_1.shape),cmap='gray')
    plt.title('emma')
    
    plt.figure()
    plt.imshow(mica_grass[0].reshape(mixture_1.shape),cmap='gray')
    plt.title('grass')
    plt.figure()
    plt.imshow(mica_grass[1].reshape(mixture_1.shape),cmap='gray')
    plt.title('grass')
    plt.figure()
    plt.imshow(mica_grass[2].reshape(mixture_1.shape),cmap='gray')
    plt.title('grass')

    
    plt.figure()
    plt.imshow(mica_lena[0].reshape(mixture_1.shape),cmap='gray')
    plt.title('lena')
    plt.figure()
    plt.imshow(mica_lena[1].reshape(mixture_1.shape),cmap='gray')
    plt.title('lena')
    plt.figure()
    plt.imshow(mica_lena[2].reshape(mixture_1.shape),cmap='gray')
    plt.title('lena')
    
elif experiment == 'test2':
    # Load data
    im_gl_path = '../data/image/'
    paths = [im_gl_path + 'grass.jpeg',im_gl_path + 'emma.jpeg',im_gl_path + 'emma_2.jpeg']
    
    # Mixing images
    mixture_1 = Image(paths=paths).mix_images([0.2,0.5,0.3])/255.
    mixture_2 = Image(paths=paths).mix_images(weights=[0.6,0.2,0.2],verbose=1)/255.
    mixture_3 = Image(paths=paths).mix_images(weights=[0.5,0.35,0.15],verbose=1)/255.
    
    mixtures = np.array([mixture_1.flatten(),mixture_2.flatten(),mixture_3.flatten()])
    
    # Performing ICA
    if algorithm == 'jade':
        unmixing_mat, _,_ = jadeR(mixtures)
    elif algorithm == 'fastICA':
        unmixing_mat, _,_ = fastICA(mixtures.T,n_iter=100)
    unmixing_mat = np.asarray(jadeR(mixtures))
    A_hat = np.linalg.inv(unmixing_mat)
    y = np.dot(unmixing_mat,mixtures)
    
    for i in range(3):
        plt.figure()
        plt.imshow(y[i,:].reshape(mixture_1.shape),cmap='gray')
        plt.title('y for source ' + str(i))


    # Orthogonal projections
        
    c_emma = [0,2]
    c_grass = 1
    
    a_grass = A_hat[:,c_grass]
    a_emma = A_hat[:, c_emma]
    
    Pi_emma = proj(a_emma)
    Pi_grass = 1/(np.linalg.norm(a_grass))**2 * np.outer(a_grass, a_grass)
  
    
    list_Pi = [Pi_grass,Pi_emma]
    orth_projs = orth_projection(list_Pi)
    
    mica_emma = orth_projs[1].dot(mixtures)
    mica_grass = orth_projs[0].dot(mixtures)
    
    # Plotting final pictures
    plt.figure()
    plt.imshow(mica_emma[0].reshape(mixture_1.shape),cmap='gray')
    plt.title('emma')
    plt.figure()
    plt.imshow(mica_emma[1].reshape(mixture_1.shape),cmap='gray')
    plt.title('emma')
    plt.figure()
    plt.imshow(mica_emma[2].reshape(mixture_1.shape),cmap='gray')
    plt.title('emma')
    
    plt.figure()
    plt.imshow(mica_grass[0].reshape(mixture_1.shape),cmap='gray')
    plt.title('grass')
    plt.figure()
    plt.imshow(mica_grass[1].reshape(mixture_1.shape),cmap='gray')
    plt.title('grass')
    plt.figure()
    plt.imshow(mica_grass[2].reshape(mixture_1.shape),cmap='gray')
    plt.title('grass')

 
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
    unmixing_mat = np.asarray(jadeR(mixtures))
    A_hat = np.linalg.inv(unmixing_mat)
    y = np.dot(unmixing_mat,mixtures)
    
        
    for i in range(3):
        plt.figure()
        plt.plot(y[i,:].reshape(mixture_1[:,0].shape))
        plt.title('y for source ' + str(i))

    wavfile.write(audio_gl_path + 'test_1.wav',rate=44100,data=y[0,:])
    wavfile.write(audio_gl_path + 'test_2.wav',rate=44100,data=y[1,:])
    wavfile.write(audio_gl_path + 'test_3.wav',rate=44100,data=y[2,:])
    
    
         #Orthogonal projections
        
    c_emma = [1,2]
    c_grass = 0
    
    a_grass = A_hat[:,c_grass]
    a_emma = A_hat[:, c_emma]
    
    Pi_emma = proj(a_emma)
    Pi_grass = 1/(np.linalg.norm(a_grass))**2 * np.outer(a_grass, a_grass)
  
    
    list_Pi = [Pi_grass,Pi_emma]
    orth_projs = orth_projection(list_Pi)
    
    mica_emma = orth_projs[1].dot(mixtures)
    mica_grass = orth_projs[0].dot(mixtures)
    
    wavfile.write(audio_gl_path + 'test_adc_1.wav',rate=44100,data=mica_grass[0])
    wavfile.write(audio_gl_path + 'test_beatles_1.wav',rate=44100,data=mica_emma[0])
    wavfile.write(audio_gl_path + 'test_adc_2.wav',rate=44100,data=mica_grass[0])
    wavfile.write(audio_gl_path + 'test_beatles_2.wav',rate=44100,data=mica_emma[0])
    wavfile.write(audio_gl_path + 'test_adc_3.wav',rate=44100,data=mica_grass[0])
    wavfile.write(audio_gl_path + 'test_beatles_3.wav',rate=44100,data=mica_emma[0])
    
#%%
    wavfile.write(audio_gl_path + 'mixture_1_th_lib.wav',rate=44100,data=mixture_1[0])
    wavfile.write(audio_gl_path + 'mixture_2_th_lib.wav',rate=44100,data=mixture_2[0])
    wavfile.write(audio_gl_path + 'mixture_3_th_lib.wav',rate=44100,data=mixture_3[0])    