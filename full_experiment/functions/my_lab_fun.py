#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:27:15 2019

In this file, I'll define a set of useful functions for working in the lab.

@author: Jacob Seifert, j.seifert@uu.nl
"""


#%% Importing libraries and defining functions

def get_probe_location(probe):
    import scipy.optimize as opt
    import numpy as np
    import matplotlib.pyplot as plt


    def twoD_Gaussian(x_y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = x_y
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                                + c*((y-yo)**2)))
        return g.ravel()

    #%% Preparations

    image_size = probe.shape[0]  # only square images allowed for now

    # Create x and y indices
    x = np.linspace(0, image_size, image_size)
    y = np.linspace(0, image_size, image_size)
    x, y = np.meshgrid(x, y)

    #create data
    #data = twoD_Gaussian((x, y), 10, 300, 312, 300, 300, 0, 10)
    data = probe.ravel()

    # plot twoD_Gaussian data generated above
    plt.figure()
    plt.imshow(data.reshape(image_size, image_size))
    plt.colorbar()

    #%% Fitting

    # parameter structure: Amplitude, X, Y, sigma X, sigma Y, 0, offset
    initial_guess = (1e3, np.int_(image_size/2), np.int_(image_size/2),
                     np.int_(image_size/4), np.int_(image_size/4), 0, 10)

    data_noisy = data + 0.2*np.random.normal(size=data.shape)

    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data_noisy, p0=initial_guess)

    #%% Plotting fitting result

    data_fitted = twoD_Gaussian((x, y), *popt)
    data_fitted = data_fitted.reshape(image_size, image_size)
    data_noisy= data_noisy.reshape(image_size, image_size)

    plt.imshow(data_noisy)
    plt.contour(data_fitted, cmap=plt.cm.Reds, linewidths = 0.5)
    plt.colorbar()
    plt.title('Gaussian fit (contours)')
    plt.show()
    print('Fit parameters:\nAmplitude = {}\nX = {}, Y = {}\
          \nSigmaX = {}, SigmaY = {}\
          \nOffset = {}'.format(popt[0], popt[2], popt[1], popt[3],
                                popt[4], popt[6]))

    x_fit = popt[2]
    y_fit = popt[1]

    return x_fit, y_fit


def center_cropping(image, image_size):
    ''' Crops the image to an image_size^2 square. The center is preserved.'''

    import numpy as np

    x_px_total = image.shape[0]
    y_px_total = image.shape[1]

    size_diff_y = np.int((y_px_total - image_size) / 2)
    size_diff_x = np.int((x_px_total - image_size) / 2)
    if size_diff_x == 0:
        d = image[:, size_diff_y:-size_diff_y]
    elif size_diff_y == 0:
        d = image[size_diff_x:-size_diff_x, :]
    else:
        d = image[size_diff_x:-size_diff_x, size_diff_y:-size_diff_y]

    return d


def pixel_matching(coords, dl=6.45):
    """ Returns new coordinates which all obey the locations of the pixels
        on the camera. This enhances our ptychography reconstruction because
        we are avoiding rounding inaccuracies inbetween scanning points. """

    import numpy as np

    # Assuming that dr is given in um and coords are given in mm.
    dl /= 1e3  # now in mm

    retracted_coords = np.rint(coords / dl)
    return retracted_coords * dl


def overlap_area(R, d):
    """ Returns the overlap between two circles of radius r
        with distance d to each other. """
    import numpy as np
    A = 2 * R**2 * np.arccos(d/(2*R)) - (d/2) * np.sqrt(4*R**2-d**2)
    return A / (np.pi * R**2)


def get_poisson_disk_sample(r_min=0.24, k_max=100, w=3, h=3,
                            start_pos=(0, 0), dl=6.45):
    """ @author:
        https://github.com/scipython/scipython-maths/tree/master/poisson_disc_sampled_noise
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Choose up to k points around each reference point as candidates for a new
    # sample point
    k = k_max

    # Minimum distance between samples
    r = r_min

    width, height = w, h

    # How many temperatures for simulated annealing should be used?
    opt_steps = 5e7

    # Cell side length
    a = r/np.sqrt(2)
    # Number of cells in the x- and y-directions of the grid
    nx, ny = int(width / a) + 1, int(height / a) + 1

    # A list of coordinates in the grid of cells
    coords_list = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    # Initilalize the dictionary of cells: each key is a cell's coordinates, the
    # corresponding value is the index of that cell's point's coordinates in the
    # samples list (or None if the cell is empty).
    cells = {coords: None for coords in coords_list}


    def get_cell_coords(pt):
        """Get the coordinates of the cell that pt = (x,y) falls in."""

        return int(pt[0] // a), int(pt[1] // a)


    def get_neighbours(coords):
        """Return the indexes of points in cells neighbouring cell at coords.

        For the cell at coords = (x,y), return the indexes of points in the cells
        with neighbouring coordinates illustrated below: ie those cells that could
        contain points closer than r.

                                         ooo
                                        ooooo
                                        ooXoo
                                        ooooo
                                         ooo

        """

        dxdy = [(-1,-2),(0,-2),(1,-2),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),
                (-2,0),(-1,0),(1,0),(2,0),(-2,1),(-1,1),(0,1),(1,1),(2,1),
                (-1,2),(0,2),(1,2),(0,0)]
        neighbours = []
        for dx, dy in dxdy:
            neighbour_coords = coords[0] + dx, coords[1] + dy
            if not (0 <= neighbour_coords[0] < nx and
                    0 <= neighbour_coords[1] < ny):
                # We're off the grid: no neighbours here.
                continue
            neighbour_cell = cells[neighbour_coords]
            if neighbour_cell is not None:
                # This cell is occupied: store this index of the contained point.
                neighbours.append(neighbour_cell)
        return neighbours

    def point_valid(pt):
        """Is pt a valid point to emit as a sample?

        It must be no closer than r from any other point: check the cells in its
        immediate neighbourhood.

        """

        cell_coords = get_cell_coords(pt)
        for idx in get_neighbours(cell_coords):
            nearby_pt = samples[idx]
            # Squared distance between or candidate point, pt, and this nearby_pt.
            distance2 = (nearby_pt[0]-pt[0])**2 + (nearby_pt[1]-pt[1])**2
            if distance2 < r**2:
                # The points are too close, so pt is not a candidate.
                return False
        # All points tested: if we're here, pt is valid
        return True

    def get_point(k, refpt):
        """Try to find a candidate point relative to refpt to emit in the sample.

        We draw up to k points from the annulus of inner radius r, outer radius 2r
        around the reference point, refpt. If none of them are suitable (because
        they're too close to existing points in the sample), return False.
        Otherwise, return the pt.

        """
        i = 0
        while i < k:
            rho, theta = np.random.uniform(r, 2*r), np.random.uniform(0, 2*np.pi)
            pt = refpt[0] + rho*np.cos(theta), refpt[1] + rho*np.sin(theta)
            if not (0 <= pt[0] < width and 0 <= pt[1] < height):
                # This point falls outside the domain, so try again.
                continue
            if point_valid(pt):
                return pt
            i += 1
        # We failed to find a suitable point in the vicinity of refpt.
        return False

    # Pick a random point to start with.
    pt = (np.random.uniform(0, width), np.random.uniform(0, height))
    samples = [pt]
    # Our first sample is indexed at 0 in the samples list...
    cells[get_cell_coords(pt)] = 0
    # ... and it is active, in the sense that we're going to look for more points
    # in its neighbourhood.
    active = [0]

    nsamples = 1
    # As long as there are points in the active list, keep trying to find samples.
    while active:
        # choose a random "reference" point from the active list.
        idx = np.random.choice(active)
        refpt = samples[idx]
        # Try to pick a new point relative to the reference point.
        pt = get_point(k, refpt)
        if pt:
            # Point pt is valid: add it to the samples list and mark it as active
            samples.append(pt)
            nsamples += 1
            active.append(len(samples)-1)
            cells[get_cell_coords(pt)] = len(samples) - 1
        else:
            # We had to give up looking for valid points near refpt, so remove it
            # from the list of "active" points.
            active.remove(idx)

#    plt.scatter(*zip(*samples), color='r', alpha=0.6, lw=0)
#    plt.xlim(0, width)
#    plt.ylim(0, height)
#    plt.axis('off')
#    plt.show()


    # repack into my own coordinate layout
    pos = np.zeros((2, len(samples)))
    for c in range(len(samples)):
        pos[0, c] = samples[c][0]
        pos[1, c] = samples[c][1]

    # matching the scanning pattern to the grid of pixels on our camera
    pos = pixel_matching(pos, dl)
    # path optimization
    pos = solve_TSP(pos, opt_steps)

#    plt.plot(pos[0, :], pos[1, :], '-o')
#    plt.axes().set_aspect('equal', 'datalim')
#    plt.grid()
#    plt.xlabel('x (mm)')
#    plt.ylabel('y (mm)')
#    plt.title('optimized scanning path')
#    plt.show()

    grid_phys = np.copy(pos)
    grid_phys[0, :] += start_pos[0] - w/2
    grid_phys[1, :] += start_pos[1] - h/2

    return pos.T * 1e3, grid_phys


def get_stingray_image(exposure_s):
    ''' Captures and returns an image from the stingray camera manufactured by
        Allied Vision.
    '''
    from pymba import Vimba
    import time
    import matplotlib.pyplot as plt
    import numpy as np

    exposure = exposure_s * 1e6  # conversion to microseconds

    # start Vimba
    with Vimba() as vimba:
        # get system object
        system = vimba.getSystem()

        # list available cameras (after enabling discovery for GigE cameras)
        if system.GeVTLIsPresent:
            system.runFeatureCommand("GeVDiscoveryAllOnce")
            time.sleep(0.2)
        cameraIds = vimba.getCameraIds()

        # get and open a camera
        camera0 = vimba.getCamera(cameraIds[0])
        camera0.openCamera()

        # Set up the camera with the right parameters
        # Includes examples of setting up the output TTLs
        camera0.PixelFormat = 'Mono16'
        camera0.Height = camera0.HeightMax
        camera0.Width = camera0.WidthMax
        camera0.ExposureTime = exposure  # microseconds

        # set the value of a feature
        camera0.AcquisitionMode = 'SingleFrame'

        # create new frames for the camera
        frame0 = camera0.getFrame()    # creates a frame

        # announce frame
        frame0.announceFrame()

        # capture a camera images
        camera0.startCapture()
        frame0.queueFrameCapture()
        camera0.runFeatureCommand('AcquisitionStart')
        camera0.runFeatureCommand('AcquisitionStop')
        frame0.waitFrameCapture()

        # save image data
        img_data = np.ndarray(buffer=frame0.getBufferByteData(),
                              dtype=np.uint16,
                              shape=(frame0.height, frame0.width))

        # clean up after capture
        camera0.endCapture()
        camera0.revokeAllFrames()
        camera0.closeCamera()

        # Account for Allied Vision's vertical inversion
        img_data = np.flip(img_data, axis=0)

        # visualize image and print some feedback
        print('Aquired image with resolution {}. Max. value: {}, '
              'Min. value: {}'.format(img_data.shape,
                                      np.max(img_data),
                                      np.min(img_data)))

        plt.imshow(img_data)
        plt.colorbar()
        plt.show()

        return img_data.copy()


def get_noisy_mesh(grid_length, start_pos, step_length, noise_factor=0):
    ''' Returns a quadratic grid with edge length grid_length,
        relative to the absolute zero position (start_pos).
        The output format is a (2, grid_length) list
        with a set of grid_length * (x, y) coordinates.
        Noise factor alternates exact coordinates around their center in 100%.
    '''
    import numpy as np

    total_positions = grid_length**2
    grid = np.zeros((2, total_positions), dtype=float)

    i = 0
    for x in range(grid_length):
        for y in range(grid_length):
            grid[:, i] = [(x + noise_factor * (np.random.rand(1) - 0.5))
                          * step_length,
                          (y + noise_factor * (np.random.rand(1) - 0.5))
                          * step_length]
            i += 1

    grid_phys = np.copy(grid)
    grid_phys[0, :] += start_pos[0] 
    grid_phys[1, :] += start_pos[1]

    return grid, grid_phys


def solve_TSP(coords, T_steps=1e6):
    """ Takes an array of coordinates and returns it ordered along
        the shortest path between each point using Simulated Annealing.
        T_steps is the count of steps for the temperatur control parameter.
    """
    import random
    import numpy
    import math
    import copy

    total_count = coords.shape[1]
    cities = coords.transpose().tolist()  # cities analogy
    tour = list(range(total_count))
    random.shuffle(tour)

    # simulated annealing:
    for T in numpy.linspace(0, 0.1, num=T_steps)[::-1]:
        [i, j] = sorted(random.sample(range(total_count), 2))
        newTour = tour[:i]+tour[j:j+1]+tour[i+1:j]+tour[i:i+1]+tour[j+1:]

        oldDistances = []
        newDistances = []
        for k in [j, j-1, i, i-1]:
            old = 0
            new = 0
            for d in [0, 1]:
                old += (cities[tour[(k+1) % total_count]][d] -
                        cities[tour[k % total_count]][d])**2
                new += (cities[newTour[(k+1) % total_count]][d] -
                        cities[newTour[k % total_count]][d])**2
            oldDistances.append(math.sqrt(old))
            newDistances.append(math.sqrt(new))
        oldDistance = sum(oldDistances)
        newDistance = sum(newDistances)

        if newDistance < oldDistance:
            tour = copy.copy(newTour)
        elif math.exp((oldDistance - newDistance) / (T + 1e-12)) > random.random():
            tour = copy.copy(newTour)

    # repacking the solution
    solution = numpy.zeros((2, total_count))
    for m in range(2):
        for n in range(total_count):
            solution[m, n] = cities[tour[n]][m]

    return solution

 #%%

def get_fermat_spiral(total_points, start_pos, radius, return_ind=0):
    """ Returns 2D scanning coordinates with 'optimal uniformness' for
        ptychography. According to Huang et al. (2014), this might be
        a Fermat spiral.
        INPUTS:
            - total_points: total scanning points (result will be round-ish)
            - start_pos: offset for the spiral center
            - pixel_size: pixel size of the camera (in um)
            - radius: max. "radius" of the spiral (in mm)
            - return_ind: returns a mask with for selecting ::return_ind
    """
    import numpy as np
    import sys

    if np.min(start_pos) < radius:
        print('WARNING: Inputs allow physical scanning positions < 0.')
        print('Stopping script.')
        sys.exit()

    fermat_spiral = np.zeros((2, total_points))

    # Fermat spiral definition:
    n = np.linspace(1, total_points, total_points)
    c = radius / np.sqrt(total_points)
    r = c * np.sqrt(n)
    theta_0 = np.deg2rad(137.505)
    theta = n * theta_0

    # Converting to cartesian coordinates:
    fermat_spiral[0, :] = r * np.cos(theta)
    fermat_spiral[1, :] = r * np.sin(theta)

    # optimizing path
    print('Solving the Traveling Salesman Problem...')
    old_spiral = np.copy(fermat_spiral)
    fermat_spiral = solve_TSP(fermat_spiral, 1e7)  # 1e6 starts to work

    # creating physical coordinates for the scanning stage:
    stage_coordinates = np.copy(fermat_spiral)
    stage_coordinates[0, :] += start_pos[0]
    stage_coordinates[1, :] += start_pos[1]

    if return_ind == 0:
        return fermat_spiral.T * 1e3, stage_coordinates
    else: # return indices of every third from original spiral
        ind = []
        masker = old_spiral[:, ::return_ind]
        for row in range(fermat_spiral.shape[1]):
            check = False
            for k in range(masker.shape[1]):
                # if fermat_spiral[:, row] == masker[:, k]:
                if np.array_equal(fermat_spiral[:, row], masker[:, k]):
                    check = True
            if check == True:
                ind.append(True)
            else:
                ind.append(False)

        return fermat_spiral.T * 1e3, stage_coordinates, ind
    
# %%

def viz(image, title='none'):
    ''' Crudely visializes a 2D image, array etc. '''
    import matplotlib.pyplot as plt
    import numpy as np

    if np.iscomplex(image[0, 0]):
        image = np.abs(image)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    plt.imshow(image)
    if title != 'none':
        plt.title(title)
    plt.show()

    return


def save_ptychography_data(coordinates, diffraction_patterns, out_path,
                           probe_xy, background, blank_sample, cam_distance, 
                           bit_depth, dl, lambda0, hdf5=True):
    ''' Saves coordinates and corresponding diffraction patterns
        in the directory out_path (and compressed).
        Additionally, a timestamp, log.txt and copy of the data gathering
        script are created and stored.
    '''
    import os
    import datetime
    import numpy as np
    import h5py

    # If not existent, create output path
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # generate filename and save the data
    time0 = str(datetime.datetime.now())       # format: '2019-01-02 16:37:16'
    timestamp = time0.replace(' ', '-')[:-7]   # 2019-01-02-16:37:16
    timestamp = timestamp.replace(':', '-')        # 2019-01-02-16-37-16


    print('Compressing and saving data...')
    if hdf5:
        # in hdf5, I want to store everything in units of meter!
        save_path = '{}{}_{}.hdf5'.format(out_path, timestamp, 'ptychoMeasurement')
        gzip_compression = 4  # between 1 and 10
        datatype = 'f4'       # float32 is more than enough precision

        with h5py.File(save_path, 'w') as f:
            f.create_dataset('ptychogram', data=diffraction_patterns,
                             dtype=datatype, compression=gzip_compression)
            f.create_dataset('encoder', data=coordinates/1e6,
                             dtype=datatype, compression=gzip_compression)
            f.create_dataset('wavelength', shape=(1,), data=lambda0/1e6,
                              dtype=datatype, compression=gzip_compression)
            f.create_dataset('zo', shape=(1,), data=cam_distance/1e6,
                             dtype=datatype, compression=gzip_compression)
            f.create_dataset('dxd', shape=(1,), data=dl/1e6,
                             dtype=datatype, compression=gzip_compression)
            # optional datasets
            f.create_dataset('background', data=background,
                             dtype=datatype, compression=gzip_compression)
            f.create_dataset('bit_depth', shape=(1,), data=bit_depth,
                             dtype='i8', compression=gzip_compression)
            
    else:
        save_path = '{}{}_{}.npz'.format(out_path, timestamp, 'ptychoMeasurement')
        np.savez_compressed(save_path, diff_pat=diffraction_patterns,
	                        scan_pos=coordinates, probe_location=probe_xy,
	                        background=background, blank_sample=blank_sample,
	                        dist_d=cam_distance)

    # generate log.txt
    logpath = '{}{}_log.txt'.format(out_path, timestamp)
    comment = input('What\'s your comment?\n')

    log_message = ('time           : {}'.format(time0[:-7])
#                   + '\nexecuted script: {}'.format(script_source)
                   + '\ndata        : {}'.format(save_path)
                   + '\ncomment      : {}'.format(comment))
    print('### Log message ###\n' + log_message)

    # save and print log.txt
    print(log_message,  file=open(logpath, 'w'))
    return

def save_ptychography_data_offAxis(coordinates, diffraction_patterns,
                                   diffraction_patterns_offAxis, out_path,
                                   probe_xy, background, background_offAxis,
                                   blank_sample, distances):
    ''' Saves coordinates and corresponding diffraction patterns
        in the directory out_path (and compressed).
        Additionally, a timestamp, log.txt and copy of the data gathering
        script are created and stored.
    '''
    import os
    import datetime
    import numpy as np

    # If not existent, create output path
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # generate filename and save the data
    time0 = str(datetime.datetime.now())       # format: '2019-01-02 16:37:16'
    timestamp = time0.replace(' ', '-')[:-7]   # 2019-01-02-16:37:16
    timestamp = timestamp.replace(':', '-')        # 2019-01-02-16-37-16

    save_path = '{}{}_{}.npz'.format(out_path, timestamp, 'ptychoMeasurement')

    print('Compressing and saving data...')
    np.savez_compressed(save_path, diff_pat=diffraction_patterns,
                        scan_pos=coordinates, probe_location=probe_xy,
                        diffraction_patterns_offAxis=diffraction_patterns_offAxis,
                        background_offAxis=background_offAxis,
                        background=background, blank_sample=blank_sample,
                        dist_p=distances[0], dist_d=distances[1])

    # generate log.txt
    logpath = '{}{}_log.txt'.format(out_path, timestamp)
    comment = input('What\'s your comment?\n')

    log_message = ('time           : {}'.format(time0[:-7])
#                   + '\nexecuted script: {}'.format(script_source)
                   + '\ndata        : {}'.format(save_path)
                   + '\ncomment      : {}'.format(comment))
    print('### Log message ###\n' + log_message)

    # save and print log.txt
    print(log_message,  file=open(logpath, 'w'))
    return


