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

def whiten(X,red_dim=None):
    
    Y = np.transpose(X)
    N, p = Y.shape
    Y = Y/np.sqrt(N-1)
    U, s, V = np.linalg.svd(Y, full_matrices=False) 
    #np.allclose(X, np.dot(U, np.dot(S, V)))
    Yt = np.transpose(Y)
    YtY = np.dot(Yt, Y)
    print('V ',V.shape)
    
    if red_dim == None:
        S = np.diag(s)
        # Diagonal matrix of eigen values of XXt
        #D = S**2
        # Orthogonal matrix of eigen vectors of XXt
        E = np.transpose(V)
        print('E ', E.shape)

    
    else:
        order_eigen  = np.argsort(-s)
        s_ordered_red = s[order_eigen][:red_dim]
        # Diagonal matrix of eigen values of XXt (first red_dim eigenvalues)
        S = np.diag(s_ordered_red)
        #D = S**2
        # Orthogonal matrix of eigen vectors of XXt
        V_ordered_red = V[order_eigen][:red_dim]
        print('v ord red ', V_ordered_red.shape)
        E = np.transpose(V_ordered_red)
        print('E ', E.shape)
        

    R = np.dot(E, np.dot(np.linalg.inv(S), np.transpose(E)))
    print('R ', R.shape)
    R_inv = np.linalg.inv(R) #np.dot(np.transpose(E), np.dot(S, E))
    print('Rinv ', R_inv.shape)
    # Whitened mixture
    Xtilda = np.dot(R,X)
    print('x tild a ', Xtilda.shape)
    return Xtilda, R, R_inv
        
def fastICA(X,n_iter=10):
    X = center(X)
    X, _, _ = whiten(X)
    p, N = X.shape
    W = np.zeros((p,p))
    iterations = n_iter
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

def orthogonalize(W):
    M = np.dot(W,W.T)
    ret = np.real(np.linalg.inv(scipy.linalg.sqrtm(M))).dot(W)
    return ret
    

def fastISA(X, dim, red_dim, T, sub_dim, maxiter, seed):
    
    Xtilda, R, R_inv = whiten(X)
    
    block_matrix=np.zeros(dim)
    for i in range(dim//sub_dim):
        begin_block=(i-1)*sub_dim
        for j in range(sub_dim):
            for k in range(sub_dim):
                block_matrix[begin_block+j,begin_block+k]=1
                
    W = np.random.rand(dim,red_dim) 
    W = orthogonalize(W)
    
    for i in range(maxiter):
        s = np.dot(W,X) 
        s_square = s**2
        block_subspace = block_matrix.dot(s_square)
        
        gamma = 0.1
        g =  (gamma + block_subspace)**(-1/2.)
        g_prime = -1/2.*(gamma + block_subspace)**(-3/2.)
        
        #W = 


