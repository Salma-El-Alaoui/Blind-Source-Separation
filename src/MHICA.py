#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:31:14 2017

@author: camillejandot
"""

from numpy.random import rand
import numpy as np
from data_utils import ECG_data

class cfMHICA:
    """
    Implementation based on MULTIDIMENSIONAL INDEPENDENT COMPONENT ANALYSIS 
    USING CHARACTERISTIC FUNCTIONS, Fabian J. Theis
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.212.7954&rep=rep1&type=pdf
    """
    
    def __init__(self,k=2):
        self.k = k
    
    def _estimate_hessians(self,X,n_estimates=1):
        n_samples,_ = X.shape
        hessians = []
        for i in range(n_estimates):
            x = rand(n_samples*self.k)
            
            # Compute characteristic function
            char_func = 0
            for i in range(n_samples):
                char_func += np.exp(x.dot(X[i]))
            char_func /= float(n_samples)
            
            # Compute gradient
            gradient = np.zeros(n_samples)
            for i in range(n_samples):
                gradient +=  np.exp(x.dot(X[i])) * X[i]
            gradient /= float(n_samples)
            
            # Compute Hessian
            hessian = np.zeros((n_samples,n_samples))
            for i in range(n_samples):
                hessian +=  np.exp(x.dot(X[i])) * X[i].outer(X[i])
            hessian /= float(n_samples)
            
            # Compute Hessian Log
            hessian_log = hessian / char_func - gradient.outer(gradient) / char_func**2
            hessians.append(hessian_log)
        
        return hessians
    
    def _joint_block_diagonalization(self):
        pass
    
  
if __name__ == '__main__':
    data = ECG_data().load()
    channels_3 = np.asarray(data[:3])
    
    hessians = cfMHICA()._estimate_hessians(channels_3)