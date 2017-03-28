#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:11:50 2017

@author: salma
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import normalize


#shift the column-wise mean to 0
def center(X):
    centered =  X - np.mean(X, axis =0)
    return np.transpose(centered)

#def whiten(X, red_dim = None):
#    Y = np.transpose(X)
#    N, p = Y.shape
#    Y= Y/np.sqrt(p)
#    U, s, V = np.linalg.svd(Y, full_matrices=False)
#    S= np.diag(s)
#    #np.allclose(X, np.dot(U, np.dot(S, V)))
#    Yt = np.transpose(Y)
#    YtY = np.dot(Yt, Y)
#    #diagonal matrix of eigen values of XXt:
#    D = S**2
#    #orthogonal matrix of eigen vectors of XXt:
#    E = np.transpose(V)
#    ED = np.dot(E,D)
#    #print(np.dot(E, np.transpose(E)))
#    #print(np.allclose(XXt, np.dot(ED, np.transpose(E))))
#    R = np.dot(E, np.dot(np.linalg.inv(S), np.transpose(E)))
#    R_inv = np.dot(np.transpose(E), np.dot(S, E))
#    #Whitened mixture
#    Xtilda = np.dot(R,X)
#    #print(np.diag(np.dot(Xtilda,np.transpose(Xtilda))))
#    return Xtilda, R, R_inv


def whiten(X,red_dim=None,zca=True):
    
    if zca:
        Y = np.transpose(X)
        N, p = Y.shape
        Y = Y/np.sqrt(N-1)
        U, s, V = np.linalg.svd(Y, full_matrices=False) 
        #np.allclose(X, np.dot(U, np.dot(S, V)))
        #Yt = np.transpose(Y)
        #YtY = np.dot(Yt, Y)
        
        if red_dim == None:
            S = np.diag(s)
            # Diagonal matrix of eigen values of XXt
            #D = S**2
            # Orthogonal matrix of eigen vectors of XXt
            E = np.transpose(V)
    
        
        else:
            order_eigen  = np.argsort(-s)
            s_ordered_red = s[order_eigen][:red_dim]
            # Diagonal matrix of eigen values of XXt (first red_dim eigenvalues)
            S = np.diag(s_ordered_red)
            #D = S**2
            # Orthogonal matrix of eigen vectors of XXt
            V_ordered_red = V[order_eigen][:red_dim]
            E = np.transpose(V_ordered_red)
            
    
        R = np.dot(E, np.dot(np.linalg.inv(S), np.transpose(E)))
        R_inv = np.linalg.inv(R) #np.dot(np.transpose(E), np.dot(S, E))
        # Whitened mixture
        X_whitened = np.dot(R,X)
    
    else:
        X_t = X
        
        X_t = X_t/np.sqrt(X_t.shape[1])

        covarianceMatrix = X_t.dot(X_t.T)
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

def power_minus_half(M):
    ret = np.real(np.linalg.inv(scipy.linalg.sqrtm(M)))
    return ret

def power_minus_3_2(M):
    ret = np.dot(np.linalg.inv(M),power_minus_half(M))
    return ret

def fastISA(X, dim, red_dim, T, sub_dim, maxiter, seed, W_init):
     
    X_whitened, R, R_inv = whiten(X,red_dim=red_dim,zca=False)
    
    X = X_whitened.copy()
    block_matrix = np.zeros((dim,dim))
    for i in range(dim//sub_dim):
        begin_block = i*sub_dim 
        for j in range(sub_dim):
            for k in range(sub_dim):
                block_matrix[begin_block+j,begin_block+k] = 1
                
    W = W_init +  np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    print(np.random.multivariate_normal(mean=np.zeros(dim), cov=0.001*np.eye(dim)))
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
        plt.figure()
        plt.imshow(men,cmap='gray')

        W_new = (s*g).dot(X.T)/float(T) - W*(men.dot(np.ones((1,W.shape[1]))))
        W_orth = orthogonalize(W_new.copy())
        W = W_orth.copy()
    S = np.dot(W,X)
    
    return W,S

#%%
from data_utils import gen_super_gauss

A, X, super_gauss = gen_super_gauss(15, 15, 50, 5, 5)
W_true = np.linalg.inv(A)
W,S = fastISA(X, 15, 15, 50, 5, 20, 5, W_true)
print(W_true-W)
plt.figure()
plt.imshow(A.T.dot(W),cmap='gray')

