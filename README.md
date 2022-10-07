# Particle Detection Prototype

The Particle Detection Prototype is an optical device to measure the size distribution of nanosized particles in water.
We are building this device at the [Lili's Proto Lab](https://lilis-protolab.sites.uu.nl/) at Utrecht University.

In this repository, we are documenting the current state and software that is developed to make the device work.

## Optical Setup

![setup](https://github.com/Duxon/particle_detection_prototype/blob/main/media/preliminary_setup.png)

> needs some adjustments!

## Inference model

![inference_model](https://github.com/Duxon/particle_detection_prototype/blob/main/media/inference_model.png)

## Validation experiment for PSF engineering

We are using a cylindrical lens in our imaging system to introduce an astigmatic aberration to our point-spread-function (PSF) of our imaging system.
To evaluate how well the z postition of a particle (with z = 0 nm defined as in focus) is encoded in the shape of the PSF, we are imaging nanoholes in a dark-field microscope. By using a piezo stage, we have fine control of the z position (label) which influenes our particle images (features). The training results are shown below.

![z-stack](https://github.com/Duxon/particle_detection_prototype/blob/main/calibration_experiment/data/out/z-stack.gif)

![predicting z positition](https://github.com/Duxon/particle_detection_prototype/blob/main/calibration_experiment/fig/validation_with_residuals.png)
![loss and validation loss](https://github.com/Duxon/particle_detection_prototype/blob/main/calibration_experiment/fig/loss_history.png)

You can run the analysis of the data by executing all scripts in [this directoy](https://github.com/Duxon/particle_detection_prototype/tree/main/calibration_experiment) in the numbered order.

## Particle detection

We want to locate scattering particles in a dark-field measurement in real-time to crop their signal and perform an intensity and z-position analysis on each particle individually. For that, we could use visual transformer networks as described in https://arxiv.org/abs/2010.11929. By changing the last output layer from an object classification vector to a vector of length 4 (for each coordinate of an object location box), we can use this visual transformer design to locate objects in complex environments and with noise. I've followed the implementation from https://keras.io/examples/vision/object_detection_using_vision_transformer/ to identify motorbikes after training on roughly 700 annotated images. The results are promising and the inference time is very fast, enabling real-time object detection.

![visual_transformer](https://github.com/Duxon/particle_detection_prototype/blob/main/particle_tracking/figures/visual_transformers.png)

![results](https://github.com/Duxon/particle_detection_prototype/blob/main/particle_tracking/figures/bikes.png)
 
Next steps: 

* The current transformer is limited to locating only one object per image. We do not only expect a larger number of particles per image, but to make it worse we have an unknown number of particles. Thus, the current visual transformer implementation needs to be extended. Perhaps we can use this resource from Facebook AI: https://alcinos.github.io/detr_page/
	
* To train the visual transformer network, we need to annotate a bunch of scattering images by hand once we have our setup. Probably >100 but <1000. That should be possible in a reasonable time. However, there are also services where you can pay a relatively small price for data labeling (like https://scale.com/)

