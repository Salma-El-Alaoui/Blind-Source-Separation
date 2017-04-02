#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:32:15 2017

@author: salma

"""
import matplotlib.pyplot as plt
import numpy as np
from data_utils import Image, load_hog_features
from projection_utils import proj, orth_projection
from fastICA import fastICA, fastISA
from jade import jadeR
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import chi2_kernel

n_sources = 4
sub_dim = 2
method = 'fastISA'
algorithm ='fastICA'
mixing_matrix = np.array([[ 0.9703649 ,  0.72929818,  0.18619978,  0.92128597],
       [ 0.47784494,  0.05356984,  0.8321572 ,  0.89070084],
       [ 0.34709615,  0.95119286,  0.20143702,  0.10954151],
       [ 0.89497312,  0.38542305,  0.51929272,  0.32574721]])

unmixing_init_jade = np.array([[  2.26871417,  2.19459843,  -0.34084106, -4.48947479],
 [  3.86424994, -3.7989024,   -3.05250899,  1.08584109],
 [  3.33557399, -8.76012108,  -7.53106942,  16.48697795],
 [  0.94020928,  -1.54942086, -11.75755656,   6.68037556]])

unmixing_init_fastICA = np.array([[-0.81662635, -0.4594127,   0.31846124,  0.14367955],
 [-0.48541563,  0.27400536, -0.82638635, -0.07986443],
 [ 0.30755096, -0.82643013, -0.46181555,  0.09566635],
 [-0.04507106, -0.16955894,  0.06677054, -0.982222  ]])

init_fastISA = np.linalg.inv(unmixing_init_fastICA)

# Loading Data
im = Image(nb_images=n_sources)
mixtures, mixing = im.mix_images(dimension=n_sources, verbose=1, mixing_matrix=mixing_matrix)
sources = im.get_sources()
    
# Performing ICA
if method == 'mica' or method =='ica':
    if algorithm == 'jade':
        unmixing_mat = np.asarray(jadeR(mixtures))
    elif algorithm == 'fastICA':
        unmixing_mat, _, _ = fastICA(mixtures, init=False, A_init=mixing, n_iter=50)
    print(unmixing_mat)
    A_hat = np.linalg.inv(unmixing_mat)
    y = np.dot(unmixing_mat, mixtures)
    
    plt.figure(figsize=(15.0, 4.0))
    for plot_num in range(n_sources):
        plt.subplot(1, n_sources, plot_num+1)
        plt.imshow(y[plot_num, :].reshape(im.get_shape()),cmap='gray')
        plt.axis('off')
        plt.title('y for source ' + str(plot_num))
        #plt.suptitle("Recovered Sources with ICA")
    plt.show()
    
    hog_features = load_hog_features(y, equalize=False)
    X = chi2_kernel(hog_features, hog_features, gamma=0.01)
    kmeans = KMeans(n_clusters=sub_dim, random_state=0).fit(X)
    labels = kmeans.labels_
    # Orthogonal projections 
    if method == 'mica':
        if n_sources == 3:
            c_emma = np.where(labels == 0)[0]
            c_grass = np.where(labels == 1)[0][0]
            
        elif n_sources == 4:
            c_emma = np.where(labels == 0)[0]
            c_grass = np.where(labels == 1)[0]
        
        a_grass = A_hat[:,c_grass]
        a_emma = A_hat[:, c_emma]
        
        Pi_emma = proj(a_emma)
        if n_sources == 3:
            Pi_grass = 1/(np.linalg.norm(a_grass))**2 * np.outer(a_grass, a_grass)
        elif n_sources == 4:
            Pi_grass = proj(a_grass)
          
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
            #plt.suptitle("Emma MICA Component")
            
        plt.figure(figsize=(15.0, 4.0))
        for plot_num in range(n_sources):
            plt.subplot(1, n_sources, plot_num+1)
            plt.imshow(mica_grass[plot_num].reshape(im.get_shape()),cmap='gray')
            plt.axis('off')
            #plt.suptitle("Grass MICA Component")

elif method == 'fastISA':
    W,S,R = fastISA(X=mixtures, dim=n_sources, red_dim=mixtures.shape[0], T=mixtures.shape[1], sub_dim=sub_dim, maxiter=15, seed=5,
                    A_init=init_fastISA)
    plt.figure(figsize=(15.0, 4.0))
    for plot_num in range(n_sources):
        plt.subplot(1, n_sources, plot_num+1)
        plt.imshow(S[plot_num, :].reshape(im.get_shape()),cmap='gray')
        plt.axis('off')
        #plt.suptitle("Recovered Sources with fastISA")
    plt.show()