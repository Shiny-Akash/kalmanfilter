#!/usr/bin/env python3

"""runs the kalman filter on the given system"""

# using KalmanFilter from filterpy module
from filterpy.kalman import KalmanFilter

# numpy for array manipulations
import numpy as np

# initialize the filter
kfilter = KalmanFilter(dim_x=1, dim_z=1)

# initialize parameters
kfilter.x = np.array([0])  # state vector
kfilter.P = kfilter.P * 100  # covariance matrix of x
kfilter.Q = kfilter.Q * 0.01 # environment noise
kfilter.H = np.array([1]) # sensor model
kfilter.F = np.array([1]) # state transition matrix
kfilter.B = np.array([1]) # input matrix

