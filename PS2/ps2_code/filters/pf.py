"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle


class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, bearing_std, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, bearing_std)
        
        # Inialize numeb of particles
        self.num_particles = num_particles
        # Initialize particle around previous state with given distribution
        self.X = np.random.multivariate_normal(self.mu, self.Sigma, num_particles)
        # Assign initial weights to each particle
        self.weight = np.zeros(self.num_particles)

    def predict(self, u):

        # 1. Implement control to each of the particle (Propagate each particle)
        for i in range(self.num_particles):
            self.X[i, :] = sample_from_odometry(self.X[i, :], u, self._alphas)
            self.X[i, 2] = wrap_angle(self.X[i, 2])

        # 2. Calculate bel_bar by calculating prior weights of each particle
        self._state_bar = get_gaussian_statistics(self.X)

    def update(self, z):
        
        # 1. Assign landmark id
        lm_id = int(z[1])

        # 2. Get observation for each particle
        z_bar = np.zeros(self.num_particles)
        for particle in range(self.num_particles):
            z_bar[particle] = get_observation(self.X[particle],lm_id)[0]

        # 3. Update weights for each particle
        for particle in range(self.num_particles):
            self.weight[particle] = gaussian.pdf(z_bar[particle], loc=z[0], scale=np.sqrt(self._Q))
        self.weight += 1.e-200 # To prevent dividing by zero
        self.weight /= sum(self.weight) # Normalizing

        # 4. Resampling (algorithm according to ProbRob)
        R = uniform(low=0, high=1 / self.num_particles)
        tmp = self.weight[0]
        X = np.empty((self.num_particles, self.state_dim))
        ind = 0
        for m in range(self.num_particles):
            U = R + m / self.num_particles
            while U > tmp:
                ind += 1
                tmp += self.weight[ind]
            # Correction
            X[m, :] = self.X[ind, :]

        # 5. Update
        self.X = X
        self._state = get_gaussian_statistics(self.X)