#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:06:24 2017

@author: camillejandot
"""

import numpy as np
import matplotlib.pyplot as plt

def img_hist(img):
    shape = img.shape
    hist = np.zeros(256)
    for i in range(shape[0]):
        for j in range(shape[1]):
            hist[img[i,j]] += 1
    return hist/(shape[0]*shape[1])

def cum_sum(hist):
    return [sum(hist[:i+1]) for i in range(len(hist))]

def equal_hist(img0):
    img = img0.copy()
    shape = img.shape
    hist = img_hist(img)
    cum_dist = np.array(cum_sum(hist)) 
    transf = np.uint8(255 * cum_dist) 
    img_after_eq = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            img_after_eq[i,j] = transf[img[i,j]]
    hist_after_eq = img_hist(img_after_eq)
    return img_after_eq,hist,hist_after_eq
    
def plot_histograms(hist,hist_after_eq):
    plt.figure()
    plt.plot(hist)
    plt.title('Histogram before equalization') 
    
    plt.figure()
    plt.plot(hist_after_eq)
    plt.title('Histogram after equalization')
    
    plt.show()
    
def equalize_item(item,rgb=False,verbose=False,report=False):
#    img = np.zeros((32,32,3))
#
#    img[:,:,0] = item[:1024].reshape(32,32)
#    img[:,:,1] = item[1024:2048].reshape(32,32)
#    img[:,:,2] = item[2048:].reshape(32,32)
#    
#    min_r = np.ones((32,32))*img[:,:,0].min() 
#    min_g = np.ones((32,32))*img[:,:,1].min() 
#    min_b = np.ones((32,32))*img[:,:,2].min() 
#    
#    img[:,:,0] -= min_r
#    img[:,:,1] -= min_g
#    img[:,:,2] -= min_b
#    
#    img[:,:,0] *= 255.
#    img[:,:,1] *= 255.
#    img[:,:,2] *= 255.
    
    if not rgb:
        #img_grey = np.uint8((0.2126 * img[:,:,0]) + np.uint8(0.7152 * img[:,:,1]) + np.uint8(0.0722 * img[:,:,2]))
        input_grey = item #img_grey.copy()
        
        eq_img_grey, hist_grey, eq_hist_grey = equal_hist(input_grey)
    
        if verbose:
            plt.figure()
            plt.imshow(img_grey,cmap='gray',interpolation=None)
            if report:
                plt.axis('off')
            if not report:
                plt.title('Before equalization (grayscale)')
            plt.figure()
            plt.imshow(eq_img_grey,cmap='gray',interpolation=None)
            if report:
                plt.axis('off')
            if not report:
                plt.title('After equalization (grayscale)')
            
        return eq_img_grey
    
    else:
        eq_img_r, hist_r, eq_hist_r = equal_hist(img[:,:,0])
        eq_img_g, hist_g, eq_hist_g = equal_hist(img[:,:,1])
        eq_img_b, hist_b, eq_hist_b = equal_hist(img[:,:,2])
        color_img = np.zeros((32,32,3))
        color_img[:,:,0] = eq_img_r
        color_img[:,:,1] = eq_img_g
        color_img[:,:,2] = eq_img_b
        
        if verbose:
            plt.figure()
            plt.imshow(img[:,:,0],cmap='gray',interpolation=None)
            plt.title('Before equalization (grayscale)')
            plt.figure()
            plt.imshow(eq_img_r,cmap='gray',interpolation=None)
            plt.title('After equalization (grayscale)')
            
            plt.figure()
            plt.imshow(img[:,:,1],cmap='gray',interpolation=None)
            plt.title('Before equalization (grayscale)')
            plt.figure()
            plt.imshow(eq_img_g,cmap='gray',interpolation=None)
            plt.title('After equalization (grayscale)')
            
            plt.figure()
            plt.imshow(img[:,:,2],cmap='gray',interpolation=None)
            plt.title('Before equalization (grayscale)')
            plt.figure()
            plt.imshow(eq_img_b,cmap='gray',interpolation=None)
            plt.title('After equalization (grayscale)')
            
        return color_img
