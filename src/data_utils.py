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
    
    def mix_tracks(self,weights=None,load=True):
        if load:
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
    
    def __init__(self, nb_images=3, shape=225):
        im_gl_path = '../data/image/'
        if nb_images == 2:
            ims = ['grass.jpeg', 'emma.jpeg']
        if nb_images == 3:
            ims = ['grass.jpeg', 'emma.jpeg', 'emma_2.jpeg']
        if nb_images == 4:
            ims = ['grass.jpeg', 'grass_2.jpeg', 'emma.jpeg','emma_2.jpeg']
            
        self.paths =[im_gl_path + i for i in ims]
        self.shape = shape
    
    def _load_images(self):
        images = []
        for path in self.paths:
            image = imread(path)
            if image.ndim==3:
                image_bw = imresize(image[:,:,0],(self.shape,self.shape))
            else:
                image_bw = imresize(image,(self.shape,self.shape))
            images.append(image_bw)
        self.images = images
    
        
    def get_shape(self):
        return (self.shape, self.shape)
    
        
    def get_sources(self):
        self._load_images()
        images_flat = [i.flatten()/float(self.shape) for i in self.images]
        return np.array(images_flat)
    
        
    def mix_images(self, dimension=3, verbose=1, mixing_matrix=None):
        """
        * Loads images in paths and mix them
        * Images to be mixed should be of equal size
        * verbose : (2: displays source images + mixed image, 1: displays only 
        source image, 0: displays nothing)
        * if no weights are given, a random mixing matrix is used
        
        """
        if not mixing_matrix:
           mixing_matrix = np.random.rand(dimension, dimension)
        sources = self.get_sources()   
        mixture = np.dot(mixing_matrix, sources)
        if verbose:
            plt.figure(figsize=(15.0, 4.0))
            for plot_num, image in enumerate(self.images):
                plt.subplot(1, len(self.images), plot_num+1)
                plt.imshow(image, cmap='gray')
                plt.axis('off')
            plt.suptitle("Source images")
            plt.show()
            
            plt.figure(figsize=(15.0, 4.0))
            for plot_num in range(dimension):
                plt.subplot(1, dimension, plot_num+1)
                plt.imshow(mixture[plot_num,:].reshape(self.get_shape()), cmap='gray')
                plt.axis('off')
            plt.suptitle("Mixtures")  
            plt.show()
            
        return mixture, mixing_matrix
    
def gen_super_gauss(dim, red_dim, T, sub_dim, seed):
    n_subspace = int(np.floor(dim / sub_dim))
    
    np.random.seed(seed)
    gaussians = np.random.standard_normal((dim,T))
    
    for i in range(n_subspace):
        block = np.ones((sub_dim, T))
        for j in range(sub_dim):
            block[j,:] = np.random.rand(1,T)
        cols = (i)*sub_dim + np.arange(sub_dim)
        gaussians[cols, :] = gaussians[cols, :]*block
    
    normalization_const = np.sqrt(np.sum(np.sum(gaussians**2))/(dim*T))
    
    super_gauss = gaussians / normalization_const
    
    A = np.random.rand(dim,dim)
    #A = np.arange(dim**2).reshape(dim,dim)
    X = np.dot(A,super_gauss)
    
    return  A, X, super_gauss
            
        
        
if __name__ == '__main__':

    data_type = 'image'
    
    if data_type == 'image':
        mixture = Image(nb_images=4).mix_images(dimension=4, verbose=1)
    
    elif data_type == 'audio':
        audio_gl_path = '../data/audio/'
        audio_paths = [audio_gl_path + 'LetItBe1.wav',audio_gl_path + 'LetItBe2.wav',audio_gl_path + 'LetItBe3.wav',
                       audio_gl_path + 'Highway1.wav',audio_gl_path + 'Highway2.wav',audio_gl_path + 'Highway3.wav']
        mixed_track = Audio(audio_paths).mix_tracks()
        wavfile.write(audio_gl_path + 'MixLibHth.wav',rate=44100,data=mixed_track)
    
    elif data_type == 'test':
        im_gl_path = '../data/image/'
        im_paths = [im_gl_path + 'lena.jpeg',im_gl_path + 'emma.jpeg']
        weights = [0.5,1.3]
        mixed_image = Image(im_paths).mix_images(weights=weights,verbose=2)
        print(gen_super_gauss(4,3,100,2,10))

    

