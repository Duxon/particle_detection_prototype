# -*- coding: utf-8 -*-
"""
Created on 02.02.2023

@author: Jacob Seifert, j.seifert@uu.nl
"""

#%% Importing functions

import functions.my_lab_fun as my_lab_fun
import matplotlib.pyplot as plt
import numpy as np
import functions.BaslerCam as BaslerCam
import time
from datetime import datetime, date

#%% Measurement settings

exposure_time = 5e-1     # exposure time in s. shortest possible: 29 us
number_of_frames = 1000
rest_time = 0.01    # waiting time between frames [in s]
today = date.today()
    
#%% Setting up Basler Camera
    
serial = 21958367
cam = BaslerCam.BaslerCamera(serial, bins=1)
cam._cam.properties['BlackLevel'] = 20

#%% take test image

test_image = cam.getImage(exposure_time)

plt.figure()
plt.imshow(test_image)
plt.colorbar()

plt.show()
    
#%% Testing continuous images

for i in range(10):
    plt.figure()
    plt.imshow((cam.getImage(exposure_time)))
    plt.colorbar()
    plt.show()
    
#%% Calibration routine

user_in_calibration = input('Would you like to recalibrate? y/[n]\n')
calib_path = './background_calibration.npz'

if user_in_calibration == 'y' or user_in_calibration == 'yes':
    # Background routine
    print('Starting calibration routine...\n'
          'First: Background measurement. Block the beam!')
    input('Press ENTER to start background measurement now.')
    background = np.zeros((1200, 1600))
    bg_iter = 100 
    for i in range(bg_iter):
        print('Progress: {}/{}'.format(i, bg_iter))
        background += cam.getImage(exposure_time)
    background /= bg_iter
    print(background)
    print(background.dtype)
    background = np.round(background).astype(np.int16)

    # Saving calibration_file
    np.savez_compressed(calib_path,
                background=background,
                exposure_time=exposure_time,
                date=today.strftime("%b-%d-%Y"))
    print('Calibration file saved.')

else:
    loaded_calibration = np.load(calib_path)
    print('Loading calibration from {}'.format(loaded_calibration.f.date))
    background = loaded_calibration.f.background

#%% Recording the measurements
    
data = np.zeros(shape=(number_of_frames, 1200, 1600), dtype=np.int16)    

input('Press ENTER to start measurement in 30 seconds.')
time.sleep(0)

for i in range(number_of_frames):
    print('Frame: {}/{}'.format(i+1, number_of_frames))
    
    # take data
    img = cam.getImage(exposure_time)
    data[i, :, :] = img
    
    # plot image
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.title('{}/{}'.format(i+1, number_of_frames))
    plt.show()
    
    # wait before next frame
    time.sleep(rest_time)
    
#%% Store the data
    
print('Data collection finished. Saving data as *.npz file...')
    
data_path = './data/'
now = datetime.now() 
file_string = today.strftime("%y%m%d_") + now.strftime("%H%M%S_images.npz")
np.savez_compressed(data_path+file_string,
                    data=data,
                    background=background)
    

#%% Close Camera

cam.closeCam()
print('Done. Session closed.')

# EOF
