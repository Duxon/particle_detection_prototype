# -*- coding: utf-8 -*-
"""
Created on 06.02.2023

@author: Jacob Seifert, j.seifert@uu.nl, derduxon@gmail.com
"""

#%% Importing functions

import matplotlib.pyplot as plt
import numpy as np
import os

#%% Load data

print('loading data...')

data = np.load('./data/230206_182958_images.npz')
background = data.f.background
imgs = data.f.data

print('...finished load data.')

#%% create video using matplotlib

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


frames = [] # for storing the generated images
fig = plt.figure(figsize=(8.5, 5))
ax = plt.gca()

for i in range(imgs.shape[0]):
# for i in range(20):
    if i%10 == 0:
        print('{}/{}'.format(i, imgs.shape[0]))
    img = imgs[i] - background
    img[img<0] = 0
    frame = ax.imshow(img, cmap='inferno', animated=True)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    frames.append([frame])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)
ani.save('./fig/vid_out.mp4')
plt.show()
