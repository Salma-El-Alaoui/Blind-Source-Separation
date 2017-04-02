#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: camillejandot
"""
import numpy as np


class HistogramOrientedGradient():
    def __init__(self, cell_size=4, n_bins=18, signed=True, img_size=224):
        self.n_cells = int(img_size / cell_size)  # nb of cells (one 1 axis)
        self.cell_size = cell_size  # nb of pixel in cell (along 1 axis)
        self.n_bins = n_bins
        self.img_size = img_size
        self.signed = signed

    def _grad(self, img):
        # Compute gradient wrt 1st dimension
        grad_x = np.zeros(img.shape)
        grad_x[:, 1:-1] = img[:, 2:] - img[:, :-2]
        grad_x[:, 0] = img[:, 1] - img[:, 0]
        grad_x[:, -1] = img[:, -1] - img[:, -2]

        # Compute gradient wrt 2nd dimension
        grad_y = np.zeros(img.shape)
        grad_y[1:-1, :] = img[2:, :] - img[:-2, :]
        grad_y[0, :] = img[1, :] - img[0, :]
        grad_y[-1, :] = img[-1, :] - img[-2, :]

        return grad_x, grad_y

    def _compute_magnitude(self, grad_x, grad_y):
        return np.sqrt(grad_x ** 2 + grad_y ** 2)

    def _compute_orientation(self, grad_x, grad_y):
        orientation_rad = np.arctan2(grad_y, grad_x)
        orientation_deg = (orientation_rad * 180 / (2 * np.pi)) % 360
        return orientation_deg

    def _orientation_interpolation(self, orientation):
        n_bins = self.n_bins
        if self.signed:
            angle_max = 360.
        else:
            angle_max = 180.
        angle_bin = angle_max / n_bins
        nums_bin_low = orientation // angle_bin
        nums_bin_high = (nums_bin_low + 1.) % n_bins

        weights_high = (orientation % angle_bin) / float(angle_bin)
        hist = np.zeros((orientation.shape[0], orientation.shape[1], n_bins))

        for i in range(n_bins):
            hist[:, :, i] = np.where(nums_bin_low == i, (1. - weights_high), 0)
            hist[:, :, i] += np.where(nums_bin_high == i, weights_high, 0)

        return hist

    def _average_per_cell(self, img):
        average = np.zeros((self.n_cells, self.n_cells, self.n_bins))
        for i in range(0, img.shape[0], self.cell_size):
            for j in range(0, img.shape[0], self.cell_size):
                ind_i = i // self.cell_size
                ind_j = j // self.cell_size
                average[ind_i, ind_j, :] = img[i:i + self.cell_size, j:j + self.cell_size, :].mean(axis=0).mean(axis=0)
        return average

    def _normalize_histogram(self, raw_histogram):
        pass

    def _build_histogram(self, im):
        if im.ndim == 2:
            grad_x, grad_y = self._grad(im)
            orientation = self._compute_orientation(grad_x, grad_y)
            magnitude = self._compute_magnitude(grad_x, grad_y)
            orientation_interpolation = self._orientation_interpolation(orientation)
            hist = self._average_per_cell(orientation_interpolation * magnitude[:, :, np.newaxis]).flatten()
            hist /= np.linalg.norm(hist)
            feat = list(hist)
        elif im.ndim == 3:
            feat = []
            for i in range(im.shape[2]):
                grad_x, grad_y = self._grad(im[:, :, i])
                orientation = self._compute_orientation(grad_x, grad_y)
                magnitude = self._compute_magnitude(grad_x, grad_y)
                orientation_interpolation = self._orientation_interpolation(orientation)
                hist = self._average_per_cell(orientation_interpolation * magnitude[:, :, np.newaxis]).flatten()
                hist /= np.linalg.norm(hist)
                feat.extend(list(hist))
        else:
            raise Exception("Reshape, please :)")
        return feat

    def extract_features(self, X):
        if X[0].ndim == 2:
            feat = np.zeros((len(X), self.n_cells * self.n_cells * self.n_bins))

        elif X[0].ndim == 3:
            feat = np.zeros((len(X), self.n_cells * self.n_cells * self.n_bins * X[0].shape[2]))

        else:
            raise Exception("Reshape, please :)")

        for i, item in enumerate(X):
            feat[i] = self._build_histogram(item)

        return feat
