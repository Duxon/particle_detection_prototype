# Particle Detection Prototype

The Particle Detection Prototype is an optical device to measure the size distribution of nanosized particles in water.
We are building this device at the [Lili's Proto Lab](https://lilis-protolab.sites.uu.nl/) at Utrecht University.

In this repository, we are documenting the current state and software that is developed to make the device work.

## Validation experiment for PSF engineering

We are using a cylindrical lens in our imaging system to introduce an astigmatic aberration to our point-spread-function (PSF) of our imaging system.
To evaluate how well the z postition of a particle (with z = 0 nm defined as in focus) is encoded in the shape of the PSF, we are imaging nanoholes in a dark-field microscope. By using a piezo stage, we have fine control of the z position (label) which influenes our particle images (features). The training results are shown below.

![z-stack](https://github.com/Duxon/particle_detection_prototype/blob/main/calibration_experiment/data/out/z-stack.gif)

![predicting z positition](https://github.com/Duxon/particle_detection_prototype/blob/main/calibration_experiment/fig/validation_with_residuals.png)
![loss and validation loss](https://github.com/Duxon/particle_detection_prototype/blob/main/calibration_experiment/fig/loss_history.png)

You can run the analysis of the data by executing all scripts in [this directoy](https://github.com/Duxon/particle_detection_prototype/tree/main/calibration_experiment) in the numbered order.
