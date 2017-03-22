#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:11:50 2017

@author: salma
"""


import numpy as np
import matplotlib.pyplot as plt

#shift the column-wise mean to 0
def center(X):
    centered =  X - np.mean(X, axis =0)
    return np.transpose(centered)

def whiten(X):
    Y = np.transpose(X)
    N, p = Y.shape
    Y= Y/np.sqrt(N-1)
    U, s, V = np.linalg.svd(Y, full_matrices=False)
    S= np.diag(s)
    #np.allclose(X, np.dot(U, np.dot(S, V)))
    Yt = np.transpose(Y)
    YtY = np.dot(Yt, Y)
    #diagonal matrix of eigen values of XXt:
    D = S**2
    #orthogonal matrix of eigen vectors of XXt:
    E = np.transpose(V)
    ED = np.dot(E,D)
    #print(np.dot(E, np.transpose(E)))
    #print(np.allclose(XXt, np.dot(ED, np.transpose(E))))
    R = np.dot(E, np.dot(np.linalg.inv(S), np.transpose(E)))
    #Whitened mixture
    Xtilda = np.dot(R,X)
    #print(np.diag(np.dot(Xtilda,np.transpose(Xtilda))))
    return Xtilda, E, D

def fastICA(X):
    X = center(X)
    X, _, _ = whiten(X)
    p, N = X.shape
    W = np.zeros((p,p))
    iterations = 10
    #number of componenets
    for i in range(p):
        #random initialisation
        W[i,:] = np.sqrt(1) * np.random.randn(p)
        for k in range(iterations):
            wtold = np.transpose(W[i,:])
            g = np.tanh(np.dot(wtold,X))
            gPrime = np.ones((1,N))- np.multiply(np.tanh(np.dot(wtold,X)), np.tanh(np.dot(wtold,X)))
            w = 1/N*np.dot(X, np.transpose(g))- np.mean(gPrime)*W[i,:]
            w = w/np.sqrt(np.dot(np.transpose(w),w))
            if i == 1:
                w = w - np.dot(W[0,:], np.dot(np.transpose(w),W[0,:]))
                w = w/np.sqrt(np.dot(np.transpose(w),w))
            #check convergence:
            #if np.allclose(1, np.dot(W[i,:],w)):
                #print(np.dot(W[i,:],w))
                #W[i,:] = w
            #print("iteration",k, "  ",np.dot(W[i,:],w))
            W[i,:] = w
    S = np.dot(W,X)
    A = np.linalg.inv(W)
    return W,S,A