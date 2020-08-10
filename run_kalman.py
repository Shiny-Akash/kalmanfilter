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
kfilter.Q = kfilter.Q * 0.01  # environment noise
kfilter.H = np.array([1])  # sensor model
kfilter.F = np.array([1])  # state transition matrix
kfilter.B = np.array([1])  # input matrix

# sensor data
data_length = 10
truth = np.arange(1, data_length + 1)
r1 = 0.5
r2 = 0.9
z1 = truth + np.random.normal(0, r1, data_length)
z2 = truth + np.random.normal(0, r2, data_length)

# run the loop
for i in range(data_length):
    kfilter.predict(u=1)
    print(
        "Predicted State : [{:10.6f}] Measured State : [{:10.6f}] Actual State : [{:10.6f}]".format(
            kfilter.x[0], z1[i], truth[i]
        )
    )
    kfilter.update(z=z1[i], R=r1)
    kfilter.update(z=z2[i], R=r2)
