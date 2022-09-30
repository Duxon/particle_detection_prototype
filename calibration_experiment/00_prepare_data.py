#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:43:14 2022

@author: Jacob Seifert  - j.seifert@uu.nl
"""

#%% Imports

import matplotlib.pyplot as plt
import numpy as np

#%% Functions

def viz_particle(data, x, y, w):
    plt.figure(figsize=(5,2))

    plt.subplot(131)
    plt.imshow(data[0, x:x+w, y:y+w])
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(data[data.shape[0]//2, x:x+w, y:y+w])
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(data[-1, x:x+w, y:y+w])
    plt.axis('off')
    plt.show()

def crop_particle(data, x, y, w):
    N = data.shape[0]
    cropped_data = np.zeros(shape=(N, w, w), dtype=data.dtype)
    for i in range(N):
        cropped_data[i, :, :] = data[i, x:x+w, y:y+w]
    return cropped_data



#%% Load raw data

data = np.load('./data/raw_data.npz')['arr_0']

# z vector
z_vec = np.linspace(-data.shape[0]//2 * 20, data.shape[0]//2 * 20,
                    data.shape[0], endpoint=False)

#%% Select particle

x = 45   # x limit
y = 430  # y limit
w = 88   # window size

# viz_particle(data, x, y, w)
cropped_data = crop_particle(data, x, y, w)

#%% Save data

np.savez_compressed('./data/processed.npz', features=cropped_data,
                    labels=z_vec)



















# EOF