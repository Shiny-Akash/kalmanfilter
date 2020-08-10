#!/usr/bin/env python3

'''runs the kalman filter on the given system'''

# using KalmanFilter from filterpy module
from filterpy.kalman import KalmanFilter
# numpy for array manipulations
import numpy as np

# initialize the filter
kfilter = KalmanFilter(dim_x=1, dim_z=1)

# initialize parameters
kfilter.x = np.array([0]) # state vector
kfilter.P = np.array([100]) # covariance matrix
