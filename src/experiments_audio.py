#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:35:05 2017

@author: salma
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from data_utils import Audio
from projection_utils import proj, orth_projection
from fastICA import fastICA, fastISA, amari_index, whiten, center
from jade import jadeR

n_sources = 4
n_mixtures = 4
sub_dim = 2
method = 'fastISA'
algorithm = 'jade'
audio_results_path = '../results/audio/'

mixing_matrix = np.array([[1. / 12, 1. / 8, 2. / 6, 1. / 6],
                          [1 / 6, 1 / 6, 1 / 6, 1 / 6],
                          [0.2, 0.3, 0.4, 0.5],
                          [2. / 6, 1. / 6, 1. / 12, 1. / 8]])

mixing_matrix_2sources = np.array([[1./12,1./6],
                                    [2./6,1./8]])

unmixing_matrix_jade = np.array([[3.82935198e-05, 1.94283716e-03, -6.20992848e-04,
                                  -1.72438304e-03],
                                 [7.74224894e-03, -7.21363959e-03, -1.25648552e-03,
                                  2.47264094e-03],
                                 [3.26140695e-04, -1.16772451e-03, 1.40646669e-03,
                                  -1.10893463e-03],
                                 [1.98454452e-02, -1.06123617e-02, -1.09244890e-02,
                                  -2.72918548e-04]])

init_fastISA = np.linalg.inv(unmixing_matrix_jade)

sum_rows = mixing_matrix.sum(axis=1)
mixing_matrix = mixing_matrix / sum_rows.reshape(mixing_matrix.shape[0], 1)

# Loading Data
audio = Audio(nb_tracks=n_sources).load_tracks()
mixtures, mixing = audio.mix_tracks(load=False, dimension=n_mixtures, verbose=True, mixing_matrix=mixing_matrix)
_, R_init, _ = whiten(center(mixtures), zca=False)

# Performing ICA
if method == 'mica' or method == 'ica':
    if algorithm == 'jade':
        unmixing_mat = np.asarray(jadeR(mixtures))
        print("\n\n amari_index Jade", amari_index(np.dot(np.dot(unmixing_matrix_jade, R_init), mixing_matrix), 2))

    elif algorithm == 'fastICA':
        unmixing_mat, _, _ = fastICA(mixtures, init=False, A_init=mixing, n_iter=50)
    A_hat = np.linalg.inv(unmixing_mat)
    y = np.dot(unmixing_mat, mixtures)

    plt.figure(figsize=(15.0, 4.0))
    for plot_num in range(n_mixtures):
        plt.subplot(1, n_sources, plot_num + 1)
        file_name = audio_results_path + 'y_source_jade' + str(plot_num) + '.wav'
        wavfile.write(file_name, rate=44100, data=y[plot_num, :])
        plt.plot(y[plot_num, :])
        plt.title('y for source ' + str(plot_num))
    plt.suptitle("Recovered Sources with ICA")
    plt.show()

    # Orthogonal projections 
    if method == 'mica':
        if n_mixtures == 4:
            c_beatles = [1, 2]
            c_acdc = [0, 3]

        a_acdc = A_hat[:, c_acdc]
        a_beatles = A_hat[:, c_beatles]

        Pi_beatles = proj(a_beatles)
        Pi_acdc = proj(a_acdc)

        list_Pi = [Pi_acdc, Pi_beatles]
        orth_projs = orth_projection(list_Pi)

        mica_acdc = orth_projs[0].dot(mixtures)
        mica_beatles = orth_projs[1].dot(mixtures)

        # Plotting final pictures
        plt.figure(figsize=(15.0, 4.0))
        for plot_num in range(n_mixtures):
            plt.subplot(1, n_sources, plot_num + 1)
            file_name = audio_results_path + 'component_beatles_' + str(plot_num) + '_jade' + '.wav'
            wavfile.write(file_name, rate=44100, data=mica_beatles[plot_num] / 100)
            plt.plot(mica_beatles[plot_num])
        plt.suptitle("beatles MICA Component")
        plt.show()

        plt.figure(figsize=(15.0, 4.0))
        for plot_num in range(n_mixtures):
            plt.subplot(1, n_sources, plot_num + 1)
            file_name = audio_results_path + 'component_acdc_' + str(plot_num) + '_jade' + '.wav'
            wavfile.write(file_name, rate=44100, data=mica_acdc[plot_num] / 100)
            plt.plot(mica_acdc[plot_num])
        plt.suptitle("acdc MICA Component")
        plt.show()

elif method == 'fastISA':
    W, S, R = fastISA(X=mixtures, dim=mixtures.shape[0], red_dim=mixtures.shape[0], T=mixtures.shape[1],
                      sub_dim=sub_dim, maxiter=15, seed=5, A_init=init_fastISA)
    plt.figure(figsize=(15.0, 4.0))
    for plot_num in range(n_mixtures):
        plt.subplot(1, n_sources, plot_num + 1)
        plt.plot(S[plot_num, :])
        file_name = audio_results_path + 'fastISA_' + str(plot_num) + '.wav'
        wavfile.write(file_name, rate=44100, data=S[plot_num, :])
    plt.suptitle("Recovered Sources with fastISA")
    plt.show()
    print("amari_index ", amari_index(np.dot(np.dot(W, R), mixing_matrix), 2))
