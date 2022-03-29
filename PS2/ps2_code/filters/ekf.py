"""
This file implements the Extended Kalman Filter.
"""

from cgitb import reset
from distutils.log import debug
import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle

class EKF(LocalizationFilter):

    def calculate_G_t(self, mu, u):
        # Returns jacobian G_t
        res = np.array([[1, 0, (-u[1] * np.sin(u[0] + mu[2]))[0]],
                        [0, 1,  (u[1] * np.cos(u[0] + mu[2]))[0]],
                        [0, 0, 1]])
        return res

    def calculate_V_t(self, mu, u):
        # Returns jacobian V_t
        res = np.array([[(-u[1] * np.sin(u[0] + mu[2]))[0], (np.cos(u[0] + mu[2]))[0], 0],
                        [( u[1] * np.cos(u[0] + mu[2]))[0], (np.sin(u[0] + mu[2]))[0], 0],
                        [1, 0, 1]])
        return res

    def calculate_R_t(self, V_t, M_t):
        # Returns covariance matrix R_t
        res = V_t @ M_t @ V_t.T
        return res

    def calculate_H_t(self, mu, lm_id):
        # Returns jacobian H_t
        m_x = self._field_map.landmarks_poses_x[lm_id]
        m_y = self._field_map.landmarks_poses_y[lm_id]
        mu_x = mu[0][0]
        mu_y = mu[1][0]
        res = np.array([(m_y - mu_y) / ((m_x-mu_x)**2 + (m_y-mu_y)**2), 
                        -(m_x - mu_x) / ((m_x-mu_x)**2 + (m_y-mu_y)**2), 
                        -1])
        return res

    def predict(self, u):
        # 1. Assign mu and Sigma
        self._state_bar.mu = self.mu[np.newaxis].T
        self._state_bar.Sigma = self.Sigma

        # 2. Calculate G_t, V_t, M_t, R_t
        G_t = self.calculate_G_t(self._state_bar.mu, u)
        V_t = self.calculate_V_t(self._state_bar.mu, u)

        # Task D, set1, set2
        set_1 = 2 * self._alphas
        set_2 = 0.5 * self._alphas

        M_t = get_motion_noise_covariance(u, self._alphas)
        R_t = self.calculate_R_t(V_t, M_t)

        # 3. Prediction of the mean of the state
        prediction_state_mu = np.array([self._state_bar.mu[0][0], self._state_bar.mu[1][0], self._state_bar.mu[2][0]])
        self._state_bar.mu = get_prediction(prediction_state_mu, u)[np.newaxis].T
        # Wrap angle on mu
        self._state_bar.mu[2][0] = wrap_angle(self._state_bar.mu[2][0])

        # 4. Prediction of the covariance of the state
        self._state_bar.Sigma = G_t @ self._state_bar.Sigma @ G_t.T + R_t

    def update(self, z):

        # 1. Assign landmark id
        lm_id = int(z[1])

        # 2. Get expected observation
        z_bar = get_expected_observation(self.mu_bar, lm_id)

        # 3. Calculate H_t
        H_t = self.calculate_H_t(self._state_bar.mu, lm_id)

        # Task D, set 1, set 2
        Q1 = 0.6**2
        Q2 = 0.1**2

        # 4. Calculate S_t
        S_t = H_t @ self._state_bar.Sigma @ H_t.T + self._Q

        # 5. Calculate Kalman gain K
        K_t = self._state_bar.Sigma @ H_t.T * S_t**(-1)

        # 6. Update on mu
        self._state.mu = self._state_bar.mu + (K_t * wrap_angle((z - z_bar)[0]))[np.newaxis].T
        # Wrap angle on mu
        self._state.mu[2][0] = wrap_angle(self._state.mu[2][0])

        # 7. Update on covariance
        self._state.Sigma = np.asarray((np.eye(3) - np.asmatrix(K_t).T @ np.asmatrix(H_t)) @ self._state_bar.Sigma)