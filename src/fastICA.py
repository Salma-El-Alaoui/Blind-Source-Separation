#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:11:50 2017

@author: salma
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy


def center(X):
    """
     shift the row-wise mean to 0
     
    """
    centered =  X - np.mean(X, axis=1).reshape(len(X),1)
    return centered

    
def whiten(X, zca=True, red_dim=None):  
    if zca:
        # equivalent to taking U instead of V
        Y = np.transpose(X)
        N, p = Y.shape
        Y = Y/np.sqrt(N-1)
        U, s, V = np.linalg.svd(Y, full_matrices=False)
        
        if not red_dim:
            S = np.diag(s)
            E = np.transpose(V)  
        else:
            order_eigen  = np.argsort(-s)
            s_ordered_red = s[order_eigen][:red_dim]
            S = np.diag(s_ordered_red)
            V_ordered_red = V[order_eigen][:red_dim]
            E = np.transpose(V_ordered_red)
            
        R = np.dot(E, np.dot(np.linalg.inv(S), np.transpose(E)))
        R_inv = np.linalg.inv(R)
        X_whitened = np.dot(R,X)
    
    else:
        X_t = X
        covarianceMatrix = X_t.dot(X_t.T)/X.shape[1]
        s,E = np.linalg.eig(covarianceMatrix)
        s = s.real
        E = E.real
        order_eigen  = np.argsort(-s)
        s_ord_red = s[order_eigen][:red_dim]
        E_ord_red = (E.T[order_eigen][:red_dim]).T
        E = E_ord_red
        S = (np.diag(s_ord_red**(-0.5)))
        R = np.dot(S,E.T)
        R_inv = np.dot(E,S)
        X_whitened = np.dot(R,X_t)
        
    return X_whitened, R, R_inv


def fastICA(X,n_iter=10,init=False,W_init=None):
    X = center(X)
    X, _, _ = whiten(X)
    p, N = X.shape
    W = np.zeros((p,p))
    if init:
        noise = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p)/p, size = p)
        W = W_init + 0.*noise
    iterations = n_iter
    #number of componenets
    for i in range(p):
        #random initialisation
        if not init:
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


def fastISA(X, dim, red_dim, T, sub_dim, maxiter, seed, A_init):
     
    X_whitened, R, R_inv = whiten(X, zca=True, red_dim=red_dim)
    
    X = X_whitened.copy()
    block_matrix = np.zeros((dim,dim))
    for i in range(dim//sub_dim):
        begin_block = i*sub_dim 
        for j in range(sub_dim):
            for k in range(sub_dim):
                block_matrix[begin_block+j,begin_block+k] = 1
                
    W = np.linalg.inv(np.dot(R, A_init)) +  np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim)/dim, size = dim)
    W_orth = orthogonalize(W.copy())
    W = W_orth.copy()
    for i in range(maxiter):
        s = np.dot(W,X)
        s_square = s**2
        block_subspace = block_matrix.dot(s_square)
        gamma = 0.1
        g =  (gamma + block_subspace)**(-1/2.)
        g_prime = -1/2.* (gamma + block_subspace)**(-3/2.)
        men = np.mean((g + 2.*g_prime*s_square),axis = 1)
        men = men.reshape((len(men),1))
        W_new = (s * g).dot(X.T)/float(T) - W *(men.dot(np.ones((1, W.shape[1]))))
        W_orth = orthogonalize(W_new.copy())
        W = W_orth.copy()
    S = np.dot(W,X)
    
    return W,S, R

#%%
from data_utils import gen_super_gauss

#A = scipy.io.loadmat("A.mat")['M']
#X = scipy.io.loadmat("X.mat")['X']
#super_gauss = scipy.io.loadmat("supergauss.mat")['S']
A, X, super_gauss = gen_super_gauss(dim=20, red_dim=20, T=10000, sub_dim=4, seed=5)
W_true = np.linalg.inv(A)
W,S,R = fastISA(X=X, dim=20, red_dim=20, T=10000, sub_dim=4, maxiter=15, seed=5, A_init=A)
print(W_true-W)
plt.figure()
plt.imshow(np.dot(np.dot(W, R), A), cmap='gray', vmin=-1, vmax=1)
scipy.io.savemat('result.mat', {'arr':np.dot(np.dot(W, R), A)})

