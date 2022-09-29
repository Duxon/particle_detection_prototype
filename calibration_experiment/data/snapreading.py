

"""
Show the image by the method snap in UUtrack software

Created on Mon Jun 22 01:00:07 2020

@author: Zhang101
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import trackpy

"""
To import our library with functions you might need to put the functionsDFSM.py 
file in the same folder as this file or tell your code to search it"""
import DFSM as dfsm


#%%
# Settings
# Configure settings, make folders, open data files and declare some functions:

# Configure:

folder = "./"
filename = "Snap"

extension = ".hdf5"
needsbinning = True 


#%% 
# Import data and quickly show some properties of the file

# Import data
data = dfsm.ImportHDF5data(folder+filename+extension)

# fig = plt.figure(figsize=(15,15))

imgNum = len(data.getkeys())

# beforeL = (np.array(data[imgNum, 0]).T)# imgNum is the number th of image in the file
# log = np.log(beforeL)
# ax1 = fig.add_subplot(1,1,1)
# """
# the iamge is the gray values, and probably you need a log scale to show it"""
# ax1.imshow(beforeL,cmap= 'gray', vmin = 20000, vmax= 70000)
# #ax1 = fig.add_subplot(2,1,2)
# # #ax1.imshow(log,cmap= 'gray', vmin = 7.5, vmax= 10)


# %% z-space

z_vec = np.linspace(-imgNum*10, imgNum*10, imgNum)

# print(z_vec)

#%% convert to numpy array

export_data = np.zeros((imgNum, 600, 600), dtype=np.uint16)

i = 0
for m in data.getkeys():
    print(m)
    img = np.array(data[i, 0], dtype=np.uint16).T[170:770, 160:760]
    export_data[i, :, :] = img
    
    plt.figure()
    plt.imshow(img)
    z = int(z_vec[i])
    plt.title(f'z depth = {z} nm')
    plt.axis('off')
    plt.savefig('./out/' + str(i) + '.png', dpi=300)
    plt.show()
    
    i += 1
    
# %% store

np.savez_compressed('raw_data.npz', export_data)

