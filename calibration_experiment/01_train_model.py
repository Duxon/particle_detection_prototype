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
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


model.summary()

loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

model.compile(loss=loss, optimizer=optimizer)


# %% Train model

history = model.fit(images, labels, epochs=100,
                    validation_data=(images_val, 
                                     labels_val))


# %% visualize predictions on validation

from matplotlib import gridspec

predictions = model(images_val).numpy()

fig = plt.figure(figsize=(6, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax0.plot(label_to_nm(labels_val), label='true values')
ax0.plot(label_to_nm(predictions), label='predicted values')
ax0.set_title('validation using previously unseen dataset')
ax0.set_ylabel('z position [nm]')
ax0.legend()

residuals = - label_to_nm(labels_val) + label_to_nm(predictions).squeeze()
ax1.plot(residuals, 'C3', marker='x')
ax1.set_ylabel('residuals [nm]')
ax1.set_xlabel('# of prediction example')

ax0.grid()
ax1.grid()

plt.savefig('./fig/validation_with_residuals.png', dpi=300)
plt.show()


#%% plot history

fig = plt.figure(figsize=(6, 5))
plt.semilogy(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy during training')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'])#, loc='upper left')

plt.savefig('./fig/loss_history.png', dpi=300)
plt.show()




# EOF