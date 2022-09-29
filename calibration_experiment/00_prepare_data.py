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


#%% Load raw data

data = np.load('./data/raw_data.npz')['arr_0']

# z vector
z_vec = np.linspace(-data.shape[0]//2 * 20, data.shape[0]//2 * 20,
                    data.shape[0])


#%% Select particle

x = 45   # x limit
y = 430  # y limit
w = 88   # window size

viz_particle(data, x, y, w)





















# EOF