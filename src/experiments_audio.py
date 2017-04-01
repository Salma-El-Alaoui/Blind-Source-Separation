#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:35:05 2017

@author: salma
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from data_utils import Image, ECG_data, Audio
from projection_utils import proj, orth_projection
from fastICA import fastICA
from jade import jadeR


