#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:32:15 2017

@author: salma

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from data_utils import Image, ECG_data, Audio
from projection_utils import proj, orth_projection
from fastICA import fastICA, fastISA
from jade import jadeR

n_sources = 3
sub_dim = 2
method = 'mica'
algorithm ='fastICA'

# Loading Data
im = Image(nb_images=n_sources)
mixtures, mixing = im.mix_images(dimension=n_sources, verbose=1)
sources = im.get_sources()
    
# Performing ICA
if method == 'mica':
    if algorithm == 'jade':
        unmixing_mat = np.asarray(jadeR(mixtures))
    elif algorithm == 'fastICA':
        unmixing_mat, _ ,_ = fastICA(mixtures, init=False, A_init=mixing, n_iter=50)
    A_hat = np.linalg.inv(unmixing_mat)
    y = np.dot(unmixing_mat, mixtures)
    
    plt.figure(figsize=(15.0, 4.0))
    for plot_num in range(n_sources):
        plt.subplot(1, n_sources, plot_num+1)
        plt.imshow(y[plot_num, :].reshape(im.get_shape()),cmap='gray')
        plt.axis('off')
        plt.title('y for source ' + str(plot_num))
        plt.suptitle("Recovered Sources with ICA")
    plt.show()
        
    # Orthogonal projections    
    c_emma = [0,1]
    c_grass = 2
    
    a_grass = A_hat[:,c_grass]
    a_emma = A_hat[:, c_emma]
    
    Pi_emma = proj(a_emma)
    Pi_grass = 1/(np.linalg.norm(a_grass))**2 * np.outer(a_grass, a_grass)
      
    list_Pi = [Pi_grass,Pi_emma]
    orth_projs = orth_projection(list_Pi)
    
    mica_grass = orth_projs[0].dot(mixtures)
    mica_emma = orth_projs[1].dot(mixtures)
    
    # Plotting final pictures
    plt.figure(figsize=(15.0, 4.0))
    for plot_num in range(n_sources):
        plt.subplot(1, n_sources, plot_num+1)
        plt.imshow(mica_emma[plot_num].reshape(im.get_shape()),cmap='gray')
        plt.axis('off')
        plt.suptitle("Emma MICA Component")
        
    plt.figure(figsize=(15.0, 4.0))
    for plot_num in range(n_sources):
        plt.subplot(1, n_sources, plot_num+1)
        plt.imshow(mica_grass[plot_num].reshape(im.get_shape()),cmap='gray')
        plt.axis('off')
        plt.suptitle("Grass MICA Component")

else:
    W,S,R = fastISA(X=mixtures, dim=n_sources, red_dim=mixtures.shape[0], T=mixtures.shape[1], sub_dim=sub_dim, maxiter=15, seed=5, A_init=mixing)
    plt.figure(figsize=(15.0, 4.0))
    for plot_num in range(n_sources):
        plt.subplot(1, n_sources, plot_num+1)
        plt.imshow(S[plot_num, :].reshape(im.get_shape()),cmap='gray')
        plt.axis('off')
        plt.suptitle("Recovered Sources with fastISA")
    plt.show()