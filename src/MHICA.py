#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:31:14 2017

@author: camillejandot
"""

from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt
from data_utils import ECG_data
#%%

class cfMHICA:
    """
    Implementation based on MULTIDIMENSIONAL INDEPENDENT COMPONENT ANALYSIS 
    USING CHARACTERISTIC FUNCTIONS, Fabian J. Theis
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.212.7954&rep=rep1&type=pdf
    """
    
    def __init__(self,k=2,n=2):
        self.k = k
        self.n = n
    
    def _estimate_hessians(self,X,n_estimates=1,N=4):
        #n_samples,_ = X.shape
        hessians = []
        for i in range(n_estimates):
            #x = rand(n_samples*self.k)
            x = rand(self.n*self.k)
            
            # Compute characteristic function
            char_func = 0
            for i in range(N):
                char_func += np.exp(x.dot(X[i]))
            char_func /= float(N)
            
            # Compute gradient
            gradient = np.zeros(N)
            for i in range(N):
                gradient +=  np.exp(x.dot(X[i])) * X[i]
            gradient /= float(N)
            
            # Compute Hessian
            hessian = np.zeros((N,N))
            for i in range(N):
                hessian +=  np.exp(x.dot(X[i])) * X[i].outer(X[i])
            hessian /= float(N)
            
            # Compute Hessian Log
            hessian_log = hessian / char_func - gradient.outer(gradient) / char_func**2
            hessians.append(hessian_log)
        
        return hessians
    
    def _joint_block_diagonalization(self):
        pass
    
    def _finite_diff_hessian(self):
        pass
  #%%
if __name__ == '__main__':
    
    ## ECG
#    data = ECG_data().load()
#    channels_3 = np.asarray(data[:3])
#    
#    hessians = cfMHICA()._estimate_hessians(channels_3)


    ## Toy signals 
    sig_1 = [np.sin(0.1*i/100) for i in range(100000)]
    sig_2 = list(np.exp(np.array(sig_1)))
    sig_3 = [2*((0.007*i/100 + 0.5) - np.floor(0.007*i/100 + 0.5)) - 1 for i in range(100000)]
    sig_4 = list((np.array(sig_3) + 0.5)**2)
    
    sigs = [sig_1,sig_2,sig_3,sig_4]
    for sig in sigs:
        plt.figure()
        plt.plot(sig)
    
    sigs_array = np.asarray(sigs)
    hessians = cfMHICA()._estimate_hessians(sigs_array)
    