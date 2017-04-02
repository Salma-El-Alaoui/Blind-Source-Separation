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
from HistogramOrientedGradient import HistogramOrientedGradient
from equalization import equalize_item


class ECG_data:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def load(self):
        df = pd.read_table('../data/ecg/foetal_ecg.dat', sep="\s+", header=None)
        channels = []
        for i in range(8):
            channel = df[i + 1]
            channels.append(channel.values)
            if self.verbose:
                plt.figure()
                plt.plot(channel[:500])
                plt.title('Channel ' + str(i + 1))
        return channels


class Audio:
    """
    Implementation not complete
    Not tested, probably buggy
    """

    def __init__(self, nb_tracks=2):

        audio_gl_path = '../data/audio/'
        if nb_tracks == 2:
            tracks = ['LetItBe.wav', 'Thunderstruck.wav']
        if nb_tracks == 4:
            tracks = ['LetItBe1.wav', 'LetItBe2.wav', 'Thunder1.wav',
                      'Thunder2.wav']

        self.paths = [audio_gl_path + i for i in tracks]
        self.rates = []

    def load_tracks(self, verbose=False):
        tracks = []
        lengths = []
        for path in self.paths:
            rate, data = wavfile.read(path)
            self.rates.append(rate)
            tracks.append(data[:, 0])
            lengths.append(len(data))
        min_size = min(lengths)
        min_size = min_size / 10
        sub_tracks = []
        for track in tracks:
            sub_tracks.append(track[:min_size])
        self.tracks = sub_tracks

        if verbose:
            for i, track in enumerate(tracks):
                plt.figure()
                plt.plot(track)
                plt.title('Track ' + str(i))
                plt.xlim(0, 1000)

        return self

    def get_sources(self):
        return np.array(self.tracks)

    def mix_tracks(self, mixing_matrix=None, load=True, dimension=3, verbose=True, write=False):
        if load:
            self._load_tracks()
        if not (type(mixing_matrix) is np.ndarray):
            mixing_matrix = np.random.rand(dimension, len(self.tracks))

        sources = self.get_sources()
        mixture = np.dot(mixing_matrix, sources)
        if verbose:
            plt.figure(figsize=(15.0, 4.0))
            for plot_num, track in enumerate(self.tracks):
                plt.subplot(1, len(self.tracks), plot_num + 1)
                plt.plot(track)
                plt.xlim(0, 100000)
            plt.suptitle("Source images")
            plt.show()

            plt.figure(figsize=(15.0, 4.0))
            for plot_num in range(dimension):
                plt.subplot(1, dimension, plot_num + 1)
                plt.plot(mixture[plot_num, :])
                plt.xlim(0, 100000)
            plt.suptitle("Mixtures")
            plt.show()
            if write:
                audio_results_path = '../results/audio/'
                file_name = audio_results_path + str(dimension) + '_' + 'mixtures_' + str(plot_num) + '.wav'
                wavfile.write(file_name, rate=44100, data=mixture[plot_num, :])

        return mixture, mixing_matrix


class Image:
    def __init__(self, nb_images=3, shape=224):
        im_gl_path = '../data/image/'
        if nb_images == 2:
            ims = ['grass.jpeg', 'emma.jpeg']
        if nb_images == 3:
            ims = ['grass.jpeg', 'emma.jpeg', 'emma_2.jpeg']
        if nb_images == 4:
            ims = ['grass.jpeg', 'grass_2.jpeg', 'emma.jpeg', 'emma_2.jpeg']

        self.paths = [im_gl_path + i for i in ims]
        self.shape = shape

    def _load_images(self):
        images = []
        for path in self.paths:
            image = imread(path)
            if image.ndim == 3:
                image_bw = imresize(image[:, :, 0], (self.shape, self.shape))
            else:
                image_bw = imresize(image, (self.shape, self.shape))
            images.append(image_bw)
        self.images = images

    def get_shape(self):
        return (self.shape, self.shape)

    def get_sources(self):
        self._load_images()
        images_flat = [i.flatten() / float(self.shape) for i in self.images]
        return np.array(images_flat)

    def mix_images(self, dimension=3, verbose=1, mixing_matrix=None):
        """
        * Loads images in paths and mix them
        * Images to be mixed should be of equal size
        * verbose : (2: displays source images + mixed image, 1: displays only 
        source image, 0: displays nothing)
        * if no weights are given, a random mixing matrix is used
        
        """
        sources = self.get_sources()
        if not (type(mixing_matrix) is np.ndarray):
            mixing_matrix = np.random.rand(dimension, len(self.images))
        mixture = np.dot(mixing_matrix, sources)
        if verbose:
            plt.figure(figsize=(15.0, 4.0))
            for plot_num, image in enumerate(self.images):
                plt.subplot(1, len(self.images), plot_num + 1)
                plt.imshow(image, cmap='gray')
                plt.axis('off')
            plt.suptitle("Source images")
            plt.show()

            plt.figure(figsize=(15.0, 4.0))
            for plot_num in range(dimension):
                plt.subplot(1, dimension, plot_num + 1)
                plt.imshow(mixture[plot_num, :].reshape(self.get_shape()), cmap='gray')
                plt.axis('off')
            plt.suptitle("Mixtures")
            plt.show()

        return mixture, mixing_matrix


def gen_super_gauss(dim, red_dim, T, sub_dim, seed):
    n_subspace = int(np.floor(dim / sub_dim))

    np.random.seed(seed)
    gaussians = np.random.standard_normal((dim, T))

    for i in range(n_subspace):
        block = np.ones((sub_dim, T))
        for j in range(sub_dim):
            block[j, :] = np.random.rand(1, T)
        cols = (i) * sub_dim + np.arange(sub_dim)
        gaussians[cols, :] = gaussians[cols, :] * block

    normalization_const = np.sqrt(np.sum(np.sum(gaussians ** 2)) / (dim * T))

    super_gauss = gaussians / normalization_const

    A = np.random.rand(dim, dim)
    X = np.dot(A, super_gauss)

    return A, X, super_gauss


def vec_to_img(item, rgb=False):
    length = int(np.sqrt(item.shape[0]))
    img = np.zeros((length, length, 3))
    for i in range(3):
        img[:, :, i] = item.reshape(length, length)

    if rgb:
        return img

    else:
        ret = (0.2126 * img[:, :, 0]) + (0.7152 * img[:, :, 1]) + (0.0722 * img[:, :, 2])
        return ret


def rgb_to_opprgb(item):
    img_opp = np.zeros((32, 32, 3))
    opp_mult = np.array([[1., 0., 1, 13983], [1., -0.39465, -0.58060], [1., 2.03211, 0.]])
    img = vec_to_img(item, rgb=True)

    img_opp[:, :, 0] = opp_mult[0, 0] * img[:, :, 0] + opp_mult[0, 1] * img[:, :, 1] + opp_mult[0, 2] * img[:, :, 2]
    img_opp[:, :, 1] = opp_mult[1, 0] * img[:, :, 0] + opp_mult[1, 1] * img[:, :, 1] + opp_mult[1, 2] * img[:, :, 2]
    img_opp[:, :, 2] = opp_mult[2, 0] * img[:, :, 0] + opp_mult[2, 1] * img[:, :, 1] + opp_mult[2, 2] * img[:, :, 2]

    return img_opp


def load_hog_features(data_train, rgb=False, equalize=True, yuv=False, n_cells_hog=8, signed=True, ):
    hist_train = []
    hog = HistogramOrientedGradient()

    for id_img in range(len(data_train)):
        image = data_train[id_img]
        if equalize:
            img = equalize_item(image, rgb=rgb, verbose=False)
        elif yuv:
            img = rgb_to_opprgb(image)
        else:
            img = vec_to_img(image, rgb=rgb)
        hist_train.append(hog._build_histogram(img))
    X_train = np.array(hist_train)
    return X_train


if __name__ == '__main__':

    data_type = 'image'

    if data_type == 'image':
        mixture, mixing = Image(nb_images=4).mix_images(dimension=4, verbose=1)

    elif data_type == 'audio':
        audio = Audio(nb_tracks=6).load_tracks()
        mixture, mixing = audio.mix_tracks(load=False, dimension=3, verbose=True)

    elif data_type == 'test':
        im_gl_path = '../data/image/'
        im_paths = [im_gl_path + 'lena.jpeg', im_gl_path + 'emma.jpeg']
        weights = [0.5, 1.3]
        mixed_image = Image(im_paths).mix_images(weights=weights, verbose=2)
