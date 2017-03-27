#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:54:22 2017

@author: camillejandot
"""
from scipy.io import wavfile
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import imresize

class ECG_data:
    def __init__(self,verbose=False):
        self.verbose = verbose
    
    def load(self):
        df = pd.read_table('../data/ecg/foetal_ecg.dat', sep="\s+", header=None)
        channels = []
        for i in range(8):
            channel = df[i+1]
            channels.append(channel.values)
            if self.verbose:
                plt.figure()
                plt.plot(channel[:500])
                plt.title('Channel ' + str(i+1))
        return channels
        
class Audio:
    """
    Implementation not complete
    Not tested, probably buggy
    """
    
    def __init__(self,paths,verbose=True):
        self.paths = paths
        self.rates = []
        self.verbose = verbose
        
    def _load_tracks(self):
        tracks = []
        lengths = []
        for path in self.paths:
            rate, data = wavfile.read(path)
            self.rates.append(rate)
            tracks.append(data)
            lengths.append(len(data))
        min_size = min(lengths)
        sub_tracks = []
        for track in tracks:
            sub_tracks.append(track[:min_size])
        self.tracks = sub_tracks
        
        if self.verbose:
            for i,track in enumerate(tracks):
                plt.figure()
                plt.plot(track)
                plt.title('Track '+str(i))
                plt.xlim(0,1000)
        
        #wavfile.write('test.wav',rate=44100,data=tracks[0]/2.)
        #wavfile.write('test2.wav',rate=44100,data=tracks[1]/2.)
        return self
    
    def mix_tracks(self,weights=None):
        self._load_tracks()
        if weights:
            mixed_track = np.zeros(self.tracks[0].shape)
            for i,track in enumerate(self.tracks):
                mixed_track += weights[i] * track
            self.mixed_track = mixed_track / np.array(weights).sum()
        else:
            mixed_track = np.zeros(self.tracks[0].shape)
            for track in self.tracks:
                mixed_track += track
            self.mixed_track = mixed_track / len(self.tracks)
            
        if self.verbose:
            plt.figure()
            plt.plot(track)
            plt.title('Mixed track')
            plt.xlim(0,100000)
        return self.mixed_track
    
    
class Image:
    
    def __init__(self,paths,shape=225):
        self.paths = paths
        self.shape = shape
        
    def _load_images(self):
        images = []
        for path in self.paths:
            image = imread(path)
            image_bw = imresize(image[:,:,0],(self.shape,self.shape))
            images.append(image_bw)
        self.images = images

    
    def mix_images(self,weights=None,verbose=2):
        """
        Loads images in paths and mix them
        Images to be mixed should be of equal size
        verbose : (2: displays source images + mixed image, 1: displays only 
        source image, 0: displays nothing)
        """
        self._load_images()
        if weights:
            mixed_image = np.zeros(self.images[0].shape)
            for i,image in enumerate(self.images):
                mixed_image += weights[i] * image
            self.mixed_image = mixed_image / np.array(weights).sum()
        else:
            mixed_image = np.zeros(self.images[0].shape)
            for image in self.images:
                mixed_image += image
            self.mixed_image = mixed_image / len(self.images)
            
        if verbose == 2:
            for i,image in enumerate(self.images):
                plt.figure()
                plt.imshow(image,cmap='gray')
                plt.title('Source image ' + str(i+1))
            plt.figure()
            plt.imshow(self.mixed_image,cmap='gray')
            plt.title('Mixed image')
        elif verbose == 1:
            plt.figure()
            plt.imshow(self.mixed_image,cmap='gray')
            plt.title('Mixed image')
        return self.mixed_image
    
def gen_super_gauss(dim, red_dim, T, sub_dim, seed):
    n_subspace = dim % sub_dim
    np.random.seed(seed)
    gaussians = np.random.standard_normal((dim,T))
    
    for i in range(n_subspace):
        block = np.zeros(sub_dim, T)
        for j in range(sub_dim):
            block[j,:] = np.random.rand(1,T)
        cols = (i)*sub_dim + np.arange(sub_dim)
        gaussians[cols, :] = gaussians[cols, :]*block
    
    normalization_const = np.sqrt(np.sum(np.sum(gaussians**2))/(dim*T))
    
    super_gauss = gaussians / normalization_const
    
    A = np.random.rand(dim,dim)
    X = np.dot(A,super_gauss)
    
    return  A, X, super_gauss
            
        
        
if __name__ == '__main__':

    data_type = 'image'
    
    if data_type == 'image':
        im_gl_path = '../data/image/'
        im_paths = [im_gl_path + 'lena.jpeg',im_gl_path + 'emma.jpeg']
        weights = [0.5,1.3]
        mixed_image = Image(im_paths).mix_images(weights=weights,verbose=2)
    
    elif data_type == 'audio':
        audio_gl_path = '../data/audio/'
        audio_paths = [audio_gl_path + 'bach.wav',audio_gl_path + 'dream.wav']
        mixed_track = Audio(audio_paths).mix_tracks()
        wavfile.write(audio_gl_path + 'mix1.wav',rate=44100,data=mixed_track/100.)
    
    elif data_type == 'test':
        im_gl_path = '../data/image/'
        im_paths = [im_gl_path + 'lena.jpeg',im_gl_path + 'emma.jpeg']
        weights = [0.5,1.3]
        mixed_image = Image(im_paths).mix_images(weights=weights,verbose=2)
        print(gen_super_gauss(4,3,100,2,10))

    

