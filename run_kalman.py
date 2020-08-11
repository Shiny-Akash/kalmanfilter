#!/usr/bin/env python3

"""runs the kalman filter on the given system"""

# using KalmanFilter from filterpy module
from filterpy.kalman import KalmanFilter

import numpy as np
import matplotlib.pyplot as plt

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

estimations = []
# run the loop
for i in range(data_length):
    kfilter.predict(u=1)
    kfilter.update(z=z1[i], R=r1)
    kfilter.update(z=z2[i], R=r2)

    print(
        "Estimated State : [{:10.6f}] Measured State : [{:10.6f}] Actual State : [{:10.6f}]".format(
            kfilter.x[0], z1[i], truth[i]
        )
    )

    # store info for plotting
    estimations.append(kfilter.x)


x_axis = list(range(data_length))

plt.figure(1)
plt.plot(x_axis, truth, x_axis, z1, x_axis, z2, x_axis, estimations)
plt.legend(['truth', 'measurement1', 'measurement2', 'estimation'])

plt.figure(2)
estimations_error = [abs(x-y) for x, y in zip(estimations, truth)]
measurement1_error = [abs(x-y) for x, y in zip(z1, truth)]
measurement2_error = [abs(x-y) for x, y in zip(z2, truth)]

plt.plot(x_axis, estimations_error, x_axis, measurement1_error, x_axis, measurement2_error)
plt.legend(['estimations', 'measurement1', 'measurement2'])
plt.title('error plot')
plt.ylim([0,5])
plt.show()
