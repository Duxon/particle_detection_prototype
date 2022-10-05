#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:43:14 2022

@author: Jacob Seifert  - j.seifert@uu.nl
"""

#%% Imports

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

#%% Functions

def normalize_labels(labels):
    return (labels + 1120) / (1100 + 1120)

def label_to_nm(y):
    return y * (1100 + 1120) - 1120


#%% Load data

data = np.load('./data/processed.npz')

images = data.f.features
images = images/65535.0  # 16-bit data
images = np.expand_dims(images, -1)
labels = data.f.labels
labels = normalize_labels(labels)


ds = tf.data.Dataset.from_tensor_slices((images, labels))  # dataset
ds = ds.shuffle(1000).batch(1)


images_val = data.f.features_val/65535.0
images_val = np.expand_dims(images_val, -1)
labels_val = normalize_labels(data.f.labels_val)


#%% create model

model = keras.Sequential([
    keras.layers.Conv2D(32, 3, strides=2, padding='same',
                  activation='relu', input_shape=(88, 88, 1)),
    keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
    keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'),
    keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(1, activation='relu')
])


model.summary()

loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

model.compile(loss=loss, optimizer=optimizer)


# %% Train model

history = model.fit(images, labels, epochs=150,
                    validation_data=(images_val, 
                                     labels_val))


# %% visualize predictions on validation

predictions = model(images_val).numpy()

plt.figure()
plt.plot(label_to_nm(labels_val), label='true values')
plt.plot(label_to_nm(predictions), label='predicted values')
plt.legend()
plt.title('Validation using previously unseen dataset')
plt.ylabel('z position [nm]')
plt.xlabel('# of prediction example')
plt.show()











# EOF