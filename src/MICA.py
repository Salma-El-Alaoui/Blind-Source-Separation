#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:00:09 2017

@author: camillejandot
"""

import matplotlib.pyplot as plt
import pandas as pd
from jade import jadeR
from functools import reduce
import numpy as np
from data_utils import Image
from fastICA import fastICA
#%%
# Load ECG data
df = pd.read_table('../data/ecg/foetal_ecg.dat', sep="\s+", header=None)

# Display the 8 channels
channels = []
for i in range(8):
    channel = df[i+1]
    channels.append(channel.values)
    plt.figure()
    plt.plot(channel[:500])
    plt.title('Channel ' + str(i+1))
#%%
    
channels_3 = np.array(channels[:3])
#unmixing_mat = np.asarray(jadeR(channels_3))
unmixing_mat, _,_ = fastICA(channels_3.T)
print(unmixing_mat)
A_hat = np.linalg.inv(unmixing_mat)
#%% plotting results of JADE
y = np.dot(unmixing_mat,channels_3)

for i in range(3):
    plt.figure()
    plt.plot(y[i,:])
    #plt.ylim([-10,10])
    plt.title('y for source ' + str(i))
#%% projection computations
c_mother = [0, 2] #[0,1]
c_foetus = 1 #2

a_foetus = A_hat[:,c_foetus]
Pi_f = 1/(np.linalg.norm(a_foetus))**2 * np.outer(a_foetus, a_foetus)

def proj(Ap):
    inv_inner = np.linalg.inv(Ap.T.dot(Ap))
    return Ap.dot(inv_inner).dot(Ap.T)

a_mother = A_hat[:, c_mother]
Pi_m = proj(a_mother)
#%% orthogonal computations

def orth_projection(list_Pi):
    orth_projs = []
    sum_Pi = np.zeros(list_Pi[0].shape)
    for Pi in list_Pi :
        sum_Pi += Pi
    inv = np.linalg.pinv(sum_Pi)
    for Pi in list_Pi :
        orth_projs.append(Pi.dot(inv))
    return orth_projs

list_Pi = [Pi_f,Pi_m]
orth_projs = orth_projection(list_Pi)

#%% MICA

mica_mother = orth_projs[1].dot(channels_3)
mica_foetus = orth_projs[0].dot(channels_3)

plt.figure()
plt.plot(mica_mother.T,linewidth=0.6)
plt.xlim([0,500])
#plt.ylim([-80,120])
plt.title('mother')
plt.figure()
plt.plot(mica_foetus.T,linewidth=0.6)
plt.xlim([0,500])
#plt.ylim([-80,120])
plt.title('foetus')

#%% MICA on images

import cv2

im_gl_path = '../data/image/'
paths = [im_gl_path + 'grass.jpeg',im_gl_path + 'emma.jpeg']
mixture_1 = Image(paths=paths).mix_images([0.5,0.5])
mixture_2 = Image(paths=paths).mix_images(weights=[0.6,0.4],verbose=1)
mixture_3 = Image(paths=paths).mix_images(weights=[0.15,0.85],verbose=1)

mixtures = np.array([mixture_1.flatten(),mixture_2.flatten(),mixture_3.flatten()])
unmixing_mat, _,_ = fastICA(mixtures.T)
#unmixing_mat = np.asarray(jadeR(mixtures))
A_hat = np.linalg.inv(unmixing_mat)

y = np.dot(unmixing_mat,mixtures)

for i in range(3):
    plt.figure()
    plt.imshow(y[i,:].reshape(mixture_1.shape),cmap='gray')
    plt.title('y for source ' + str(i))

#mixture_1 = cv2.imread("../data/image/blend1.png", 0)
#mixture_2 = cv2.imread("../data/image/blend2.png", 0)
#mixtures = np.array([mixture_1.flatten(),mixture_2.flatten()])

#unmixing_mat = np.asarray(jadeR(mixtures))
#unmixing_mat, _,_ = fastICA(mixtures.T)
#A_hat = np.linalg.inv(unmixing_mat)
#%%    

#c_mother = [0] #[0,1]
#c_foetus = 1 #2

c_mother = [1,2]
c_foetus = 0


a_foetus = A_hat[:,c_foetus]
Pi_f = 1/(np.linalg.norm(a_foetus))**2 * np.outer(a_foetus, a_foetus)

def proj(Ap):
    inv_inner = np.linalg.inv(Ap.T.dot(Ap))
    return Ap.dot(inv_inner).dot(Ap.T)

a_mother = A_hat[:, c_mother]
Pi_m = proj(a_mother)

list_Pi = [Pi_f,Pi_m]
orth_projs = orth_projection(list_Pi)

mica_mother = orth_projs[1].dot(mixtures)
mica_foetus = orth_projs[0].dot(mixtures)

plt.figure()
plt.imshow(mica_mother[0].reshape(mixture_1.shape),cmap='gray')
plt.title('mother')
plt.figure()
plt.imshow(mica_mother[1].reshape(mixture_1.shape),cmap='gray')
plt.title('mother')
plt.figure()
plt.imshow(mica_mother[2].reshape(mixture_1.shape),cmap='gray')
plt.title('mother')

plt.figure()
plt.imshow(mica_foetus[0].reshape(mixture_1.shape),cmap='gray')
plt.title('foetus')
plt.figure()
plt.imshow(mica_foetus[1].reshape(mixture_1.shape),cmap='gray')
plt.title('foetus')
plt.figure()
plt.imshow(mica_foetus[2].reshape(mixture_1.shape),cmap='gray')
plt.title('foetus')