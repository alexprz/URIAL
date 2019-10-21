"""Simulate signal from design matrix and noise."""

import scipy
import numpy as np


def simulate_signal(n_voxels, X, G, sigma_noise, s):
    '''
        Performs the simulation of a first level analysis.

        Args:
            n_voxels (int): number of voxels
            X (array of shape (n_time_points, n_conditions)): design matrix used to create the true signal from the simulated Betas coefficients and gaussian noise
            G (array of shape (n_conditions, n_conditions): Covariance matrix. Conditions are supposed to have the similarity structure determined by G.
            sigma_noise (float): Standard deviation of the Gaussian noise added to the signal obtained by multiplying X by the Betas.

        Returns:
            T (array of shape (n_time_points, n_voxels)): Array storing the simulated noised signal
            B (array of shape (n_conditions, n_voxels)): Array storing the simulated true Betas
            R (array of shape (n_time_points, n_voxels)): Array storing the simulated true residuals (R = S-X*B)
    '''
    n_time_points, n_conditions = X.shape

    B = np.random.multivariate_normal(np.zeros(n_conditions), G, n_voxels).T

    T = np.dot(X, B)
    Eps = np.random.normal(0, sigma_noise, (T.shape))

    T_noised = T + Eps

    # Spatial convolution:
    if s > 0:
        T_noised = scipy.ndimage.filters.gaussian_filter1d(T_noised, s, axis=1)


    # R =
    # R = T - T_noised

    return T, T_noised, B
