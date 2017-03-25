#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:32:09 2017

@author: camillejandot
"""
from numpy.linalg import norm
import numpy as np
from random import shuffle
from fastdtw import fastdtw

class KMeans:
    """
    Implements KMeans algorithm.
    Two distances are available:
        - euclidean ('euclidean')
        - dynamic time warping ('dtw') 
        (https://en.wikipedia.org/wiki/Dynamic_time_warping)
        
    """
    def __init__(self,n_clusters=2,n_iter=10,distance='euclidean'):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.centroids = 0
        self.distance = distance

## Using fastdtw implementation instead (faster)
     
    def _DTW_distance(self,x,y):
        m = len(x)
        n = len(y)
        D = np.zeros((m,n))
        for i in range(m):
            D[i,0] = np.inf
        for i in range(n):
            D[0,i] = np.inf

        for i in range(m):
            print(i)
            for j in range(n):
                cost = np.abs(x[i]-y[j])
                D[i,j] = cost + np.min([D[i-1,j],D[i,j-1],D[i-1,j-1]])
        
        return D[m,n]
    
    def _assign_to_nearest_centroid(self,X,centroids):
        (n,d) = X.shape 
        assigned_centroids = np.zeros(n)
        # For each object...
        for i in range(n):
            obj = X[i]
            dists = np.zeros(self.n_clusters)
            # ... compute distance of the object to each centroid,...
            for i_centroid in range(len(centroids)):
                centroid = centroids[i_centroid]
                if self.distance == 'euclidean':
                    dists[i_centroid] = norm(obj-centroid,2)
                elif self.distance == 'dtw':
                    #dists[i_centroid] = self._DTW_distance(obj,centroid)
                    dists[i_centroid],_ = fastdtw(obj,centroid)
                else:
                    #print('distance ',self.distance,' not implemented: using euclidean)
                    dists[i_centroid] = norm(obj-centroid,2)
            #... and assign obj to nearest centroid.
            assigned_centroids[i] = np.argmin(dists)
        return assigned_centroids
    
    def fit(self,X):
        # Initialization of centroids with points of the distribution.
        (n,d) = X.shape 
        indices = np.arange(n)
        shuffle(indices)
        centroids = X[indices[:self.n_clusters]]
        for it in range(self.n_iter):
            # Assign points to nearest centroid
            assigned_centroids = self._assign_to_nearest_centroid(X,centroids)
                    
             # For each cluster, update centroid.
            for i_centroid in range(len(centroids)):
                 indices_centroid = np.where(assigned_centroids==i_centroid)
                 points_in_cluster = X[indices_centroid]
                 centroids[i_centroid] = points_in_cluster.mean(axis=0)
                 
        self.centroids = centroids
        return self
        
    def predict(self,X):
        return self._assign_to_nearest_centroid(X,self.centroids)
    
    def fit_predict(self,X):
        return self.fit(X).predict(X)
            
    

    

