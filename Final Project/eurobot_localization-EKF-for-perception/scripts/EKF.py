#!/usr/bin/env python3

"""
This file implements the Extended Kalman Filter.
"""

from core_functions import *
import numpy as np
import rospy
from tf.transformations import euler_from_quaternion
from localization_filter import LocalizationFilter
from task import get_observation as get_expected_observation
from task import get_prediction
from task import wrap_angle

# Beacon parameters
BEAC_R = rospy.get_param("beacons_radius")

# Lidar parameters
LIDAR_DELTA_ANGLE = (np.pi / 180) / 4
LIDAR_START_ANGLE = -(np.pi / 2 + np.pi / 4)

max_dist = rospy.get_param("max_dist")
min_sin = rospy.get_param("min_sin")


WORLD_X = rospy.get_param("world_x")
WORLD_Y = rospy.get_param("world_y")
WORLD_BORDER = rospy.get_param("world_border")
BEAC_L= rospy.get_param("beac_l")
BEAC_BORDER = rospy.get_param("beac_border")
YELLOW_BEACONS = np.array([ [WORLD_X + WORLD_BORDER + BEAC_BORDER + BEAC_L / 2., WORLD_Y / 2.],
                                    [-(WORLD_BORDER + BEAC_BORDER + BEAC_L / 2.), WORLD_Y - BEAC_L / 2.],
                                    [-(WORLD_BORDER + BEAC_BORDER + BEAC_L / 2.), BEAC_L / 2.]])

class EKF(LocalizationFilter):

    def calculate_H_t(self, mu, lm_id):
        # Returns jacobian H_t
        m_x = YELLOW_BEACONS[lm_id, 0]
        m_y = YELLOW_BEACONS[lm_id, 1]
        mu_x = mu[0][0]
        mu_y = mu[1][0]
        range_ = np.array([-(m_x - mu_x) / np.sqrt((m_x - mu_x) ** 2 + (m_y - mu_y) ** 2),
                           -(m_y - mu_y) / np.sqrt((m_x - mu_x) ** 2 + (m_y - mu_y) ** 2),
                           0])

        bearing = np.array([(m_y - mu_y) / ((m_x - mu_x) ** 2 + (m_y - mu_y) ** 2),
                            -(m_x - mu_x) / ((m_x - mu_x) ** 2 + (m_y - mu_y) ** 2),
                            -1])
        res = np.vstack([range_, bearing])
        return res

    def predict(self, u):

        if (u is not None):
            # 1. Assign mu and Sigma
            self._state_bar.mu = self.mu[np.newaxis].T
            self._state_bar.Sigma = self.Sigma

            # 2. Prediction of the mean of the state
            prediction_state_mu = np.array(
                [self._state_bar.mu[0][0], self._state_bar.mu[1][0], self._state_bar.mu[2][0]])
            self._state_bar.mu = cvt_local2global(u, prediction_state_mu)[
                np.newaxis].T
            # Wrap angle on mu
            self._state_bar.mu[2][0] = wrap_angle(self._state_bar.mu[2][0])

            # 4. Prediction of the covariance of the state
            R_t = np.diag([self._alphas[0], self._alphas[1], self._alphas[2]])
            self._state_bar.Sigma = self._state_bar.Sigma + R_t

            # self._state = self._state_bar

            # return self._state

    def update(self, z):  # в Z у нас несколько биконов

        if (len(z) != 0):
            global_beacons = cvt_local2global(z, self.mu_bar)
            for i in range(global_beacons.shape[0]):

                lm_id = np.argmin(np.linalg.norm(YELLOW_BEACONS - global_beacons[i], axis=1))
                range_ = np.sqrt(z[i, 0] ** 2 + z[i, 1] ** 2)
                bearing = wrap_angle(np.arctan2(z[i, 1], z[i, 0]))
                z_real = np.array([[range_], [bearing]])
                # 2. Get expected observation

                z_bar = get_expected_observation(self.mu_bar, lm_id, YELLOW_BEACONS)

                # 3. Calculate H_t
                H_t = self.calculate_H_t(self._state_bar.mu, lm_id)
                # 4. Calculate S_t
                S_t = H_t @ self._state_bar.Sigma @ H_t.T + self._Q
                # 5. Calculate Kalman gain K
                K_t = self._state_bar.Sigma @ H_t.T @ np.linalg.inv(S_t)
                # 6. Calculate Innovation vector
                z_inn = z_real - z_bar
                z_inn[1] = wrap_angle(z_inn[1])
                # 7. Update on mu
                self._state_bar.mu = self._state_bar.mu + (K_t @ z_inn)
                # Wrap angle on mu
                self._state_bar.mu[2, 0] = wrap_angle(self._state_bar.mu[2, 0])
                # 8. Update on covariance
                self._state_bar.Sigma = (np.eye(3) - K_t @ H_t) @ self._state_bar.Sigma


        self._state = self._state_bar
        return self._state

        # Additional Sausar functions

    def get_landmarks(self, scan, intense):
        """Returns filtrated lidar data"""
        ranges = np.array(scan.ranges)
        ind = self.filter_scan(scan, intense)
        final_ind = np.where((np.arange(ranges.shape[0]) * ind) > 0)[0]
        angles = (LIDAR_DELTA_ANGLE * final_ind + LIDAR_START_ANGLE) % (2 * np.pi)
        distances = ranges[final_ind]
        return angles, distances

    def filter_scan(self, scan, intense):
        ranges = np.array(scan.ranges)
        intensities = np.array(scan.intensities)
        cloud = cvt_ros_scan2points(scan)
        index0 = (intensities > intense) & (ranges < max_dist)
        index1 = self.alpha_filter(cloud, min_sin)
        index = index0 * index1
        return np.where(index, ranges, 0)

    @staticmethod
    def alpha_filter(cloud, min_sin_angle):
        x, y = cloud.T
        x0, y0 = 0, 0
        x1, y1 = np.roll(x, 1), np.roll(y, 1)  # one element from the end of x1, y1 is placed im the beginning
        cos_angle = ((x - x0) * (x - x1) + (y - y0) * (y - y1)) / (np.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0))
                                                                   * np.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1)))
        sin_angle = np.sqrt(1 - cos_angle * cos_angle)
        index = sin_angle >= min_sin_angle
        return index
