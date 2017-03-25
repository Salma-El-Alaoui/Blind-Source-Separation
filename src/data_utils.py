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
    
    def __init__(self,paths):
        self.paths = paths
        
    def _load_tracks(self):
        tracks = []
        for path in self.paths:
            tracks.append(wavfile.read(path))
        self.tracks = tracks
    
    def mix_tracks(self,weights=None):
        self.load_tracks()
        if weights:
            pass
        else:
            self.mixed_track = self.tracks.mean()
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
            
        
        
if __name__ == '__main__':
    im_gl_path = '../data/image/'
    im_paths = [im_gl_path + 'lena.jpeg',im_gl_path + 'emma.jpeg']
    weights = [0.5,1.3]
    mixed_image = Image(im_paths).mix_images(weights=weights,verbose=2)

