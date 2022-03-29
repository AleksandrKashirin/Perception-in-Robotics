#!/usr/bin/env python3
from core_functions import *
import numpy as np

import rospy
from tf.transformations import euler_from_quaternion
from scipy import interpolate

BEAC_R = rospy.get_param("beacons_radius")

# parameters of lidar
LIDAR_DELTA_ANGLE = (np.pi / 180) / 4
LIDAR_START_ANGLE = -(np.pi / 2 + np.pi / 4)


class ParticleFilter:
    def __init__(self, world_beacons, start_x=0.293, start_y=0.425, start_angle=3 * np.pi / 2, color='orange'):

        self.start_coords = np.array([start_x, start_y, start_angle])
        # self.color = color
        if color == 'yellow':
            self.beacons = world_beacons
        else:
            self.beacons = world_beacons
        self.particles_num_from_measurement_model = rospy.get_param("particles_num_from_measurement_model")
        self.particles_num = rospy.get_param("particles_num")
        self.distance_noise = rospy.get_param("distance_noise")
        self.angle_noise = rospy.get_param("angle_noise")
        self.max_dist = rospy.get_param("max_dist")
        self.distance_noise_1_beacon = rospy.get_param("distance_noise_1_beacon")
        self.angle_noise_1_beacon = rospy.get_param("angle_noise_1_beacon")
        self.sigma_r = rospy.get_param("sigma_r")
        self.num_seeing_beacons = rospy.get_param("num_seeing_beacons")
        # Create Particles
        x = np.random.normal(start_x, self.distance_noise, self.particles_num)
        y = np.random.normal(start_y, self.distance_noise, self.particles_num)
        angle = np.random.normal(start_angle, self.angle_noise, self.particles_num) % (2 * np.pi)
        self.particles = np.array([x, y, angle]).T
        self.landmarks = np.zeros((2, 0))
        self.sigma_phi = rospy.get_param("sigma_phi")
        self.min_sin = rospy.get_param("min_sin")
        self.r_lid = np.array([])
        self.phi_lid = np.array([])
        self.beacon_ind = np.array([])

    @staticmethod
    def gaus(x, mu=0, sigma=1.):
        """calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma"""
        return np.exp(- ((x - mu) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(2.0 * np.pi * (sigma ** 2))

    @staticmethod
    def p_trans(a, d):
        x_beac = d * np.cos(a)
        y_beac = d * np.sin(a)
        return x_beac, y_beac

    def localization(self, delta, beacons, global_coords):
        self.motion_model(delta)
        self.particles = self.measurement_model(self.particles, beacons, global_coords)
        main_robot = get_gaussian_statistics(self.particles)
        return main_robot

    def measurement_model(self, particles, beacons, global_coords):
        self.landmarks = beacons
        # particles_measurement_model = self.get_particle_measurement_model(beacons)
        # print "Particles shape", particles.shape
        # print "meas particles ", particles_measurement_model.shape
        # if beacons.shape[0] > 0:
        #     particles = np.concatenate((particles[:(self.particles_num - self.particles_num_from_measurement_model)], particles_measurement_model), axis=0)
        # print "Particles shape", particles.shape
        weights = self.weights(beacons, particles, global_coords)
        inds = self.resample(weights, self.particles_num)
        # self.min_cost_function = np.mean(self.cost_function)
        particles = particles[inds, :]
        return particles

    def get_particle_measurement_model(self, landmarks):
        if landmarks.shape[0] != 0:
            beacons = cvt_global2local(self.beacons[np.newaxis, :], self.particles[:, np.newaxis])
            buf_beacons = beacons[0, :]
            distance_landmark_beacons = np.sqrt((landmarks[np.newaxis, np.newaxis, :, 0].T - buf_beacons[:, 0]) ** 2 +
                                                (landmarks[np.newaxis, np.newaxis, :, 1].T - buf_beacons[:, 1]) ** 2)
            self.beacon_ind = np.argpartition(distance_landmark_beacons[:, 0, :], 2)[:, 0]
            r = (np.sqrt((landmarks[np.newaxis, :, 1]) ** 2 + (landmarks[np.newaxis, :, 0]) ** 2)).T
            phi = np.arctan2(landmarks[np.newaxis, :, 1], landmarks[np.newaxis, :, 0])
            phi = wrap_angle(phi).T
            r_lid = r + np.random.normal(0, self.sigma_r, self.particles_num_from_measurement_model)
            phi_lid = phi + np.random.normal(0, self.sigma_phi, self.particles_num_from_measurement_model)
            phi_lid = wrap_angle(phi_lid)
            if (len(self.beacon_ind) > 0):
                y_lid = np.random.uniform(0, 2 * np.pi, self.particles_num_from_measurement_model)
                x = self.beacons[self.beacon_ind[0], 0] + r_lid[0, :5] * np.cos(y_lid)
                y = self.beacons[self.beacon_ind[0], 1] + r_lid[0, :5] * np.sin(y_lid)
                theta = wrap_angle(y_lid - np.pi - phi_lid[0, :self.particles_num_from_measurement_model])
                index = (x < 3) & (x > 0) & (y < 2) & (y > 0)
                return np.array([x, y, theta]).T[index]
            else:
                x = np.zeros(self.particles_num_from_measurement_model)
                y = np.zeros(self.particles_num_from_measurement_model)
                theta = np.zeros(self.particles_num_from_measurement_model)
                return np.array([x, y, theta]).T
        else:
            x = np.zeros(self.particles_num_from_measurement_model)
            y = np.zeros(self.particles_num_from_measurement_model)
            theta = np.zeros(self.particles_num_from_measurement_model)
            return np.array([x, y, theta]).T

    @staticmethod
    def get_random(cov_x, cov_y, cov_theta):
        return np.array([np.random.normal(0, cov_x), np.random.normal(0, cov_y), np.random.normal(0, cov_theta)])

    def motion_model(self, delta):
        d_n = self.distance_noise
        a_n = self.angle_noise
        x_noise_0 = []
        y_noise_0 = []
        angle_noise_0 = []

        x_noise_1 = []
        y_noise_1 = []
        angle_noise_1 = []

        angle_noise_2 = []

        sensor_amount = 3
        sensor0 = 0.2
        sensor1 = 0.4
        sensor2 = 0.4
        delta_lidar = delta[0]
        delta_wheel = delta[1]
        delta_imu = delta[2]
        try:
            if delta_lidar is None:
                sensor1 += sensor0 / (sensor_amount - 1)
                sensor2 += sensor0 / (sensor_amount - 1)
                sensor0 = 0
                sensor_amount -= 1

            if delta_wheel is None:
                sensor0 += sensor1 / (sensor_amount - 1)
                sensor2 += sensor1 / (sensor_amount - 1)
                sensor1 = 0
                sensor_amount -= 1

            if delta_imu is None:
                sensor0 += sensor2 / (sensor_amount - 1)
                sensor1 += sensor2 / (sensor_amount - 1)
                sensor2 = 0
                sensor_amount -= 1
        except ZeroDivisionError:
            pass

        if delta_lidar is not None:
            x_noise_0 = np.random.normal(delta_lidar[0], d_n, int(sensor0 * self.particles_num))
            y_noise_0 = np.random.normal(delta_lidar[1], d_n, int(sensor0 * self.particles_num))
            angle_noise_0 = np.random.normal(delta_lidar[2], a_n, int(sensor0 * self.particles_num))

        if delta_wheel is not None:
            cov_x = 0.005 + 0.06 * delta_wheel[0] ** 2
            cov_y = 0.005 + 0.05 * delta_wheel[1] ** 2
            cov_theta = 0.005 + 0.8 * delta_wheel[2] ** 2

            x_noise_1 = np.random.normal(delta_wheel[0], cov_x,
                                         int((sensor1 + sensor2) * self.particles_num))
            y_noise_1 = np.random.normal(delta_wheel[1], cov_y,
                                         int((sensor1 + sensor2) * self.particles_num))
            angle_noise_1 = np.random.normal(delta_wheel[2], cov_theta, int(sensor1 * self.particles_num))

        if delta_imu is not None:

            cov_theta = 0.005 + 0.06 * delta_wheel[2] ** 2

            angle_noise_2 = np.random.normal(delta_imu[2], cov_theta, int(sensor2 * self.particles_num))

        try:
            x_noise = np.concatenate((x_noise_0, x_noise_1))
            y_noise = np.concatenate((y_noise_0, y_noise_1))
            angle_noise = np.concatenate((angle_noise_0, angle_noise_1, angle_noise_2))

            move_point = np.array([x_noise, y_noise, angle_noise]).T
            self.particles = cvt_local2global(move_point, self.particles)
        except (RuntimeError, ValueError) as e:
            pass

    def resample(self, weights, n=1000):
        indices_buf = []
        weights = np.array(weights)
        c = weights[0]
        j = 0
        m = n
        M = 1. / m
        r = np.random.uniform(0, M)
        for i in range(m):
            u = r + i * M
            while u > c:
                j += 1
                c = c + weights[j]
            indices_buf.append(j)
        return indices_buf

    def calculate_main(self):
        x = np.mean(self.particles[:, 0])
        y = np.mean(self.particles[:, 1])
        zero_elem = self.particles[0, 2]
        # this helps if particles angles are close to 0 or 2*pi
        temporary = ((self.particles[:, 2] - zero_elem + np.pi) % (2.0 * np.pi)) + zero_elem - np.pi
        angle = np.mean(temporary)
        # angle = np.arctan2(1 / self.particles.shape[0] * np.sum(np.sin(self.particles[:, 2])),
        #                     1 / self.particles.shape[0] * np.sum(np.cos(self.particles[:, 2])))
        return np.array((x, y, angle))

    def weights(self, landmarks, particles, global_coords):
        """Calculate particle weights based on their pose and landmarks"""
        if landmarks.shape[0] != 0:
            beacons = cvt_global2local(self.beacons[np.newaxis, :], particles[:, np.newaxis])
            distance_landmark_beacons = np.linalg.norm(beacons[:, np.newaxis, :, :] -
                                                       landmarks[np.newaxis, :, np.newaxis, :], axis=3)
            self.beacon_ind = np.argmin(distance_landmark_beacons, axis=2)
            beacons = beacons[np.arange(beacons.shape[0])[:, np.newaxis], self.beacon_ind]
            # dist_from_closest_beacon_landmark = np.linalg.norm(beacons - landmarks[np.newaxis, :, :2], axis=2)

            # landmarks = landmarks[np.where(dist_from_closest_beacon_landmark[0, :] < 4 * BEAC_R)[0], :]
            # beacons = beacons[:, np.where(dist_from_closest_beacon_landmark[0, :] < 4 * BEAC_R)[0], :]

            r = (np.sqrt((landmarks[np.newaxis, :, 1]) ** 2 + (landmarks[np.newaxis, :, 0]) ** 2)).T
            phi = np.arctan2(landmarks[np.newaxis, :, 1], landmarks[np.newaxis, :, 0])
            phi = wrap_angle(phi).T
            self.r_lid = (np.sqrt(((beacons[:, :, 1]) ** 2 + (beacons[:, :, 0]) ** 2))).T
            self.phi_lid = (np.arctan2(beacons[:, :, 1], beacons[:, :, 0])).T
            self.phi_lid = wrap_angle(self.phi_lid)

            r_diff = r - self.r_lid
            phi_diff = wrap_angle(phi - self.phi_lid)

            weights = self.gaus(np.log(1 + r_diff), mu=0, sigma=self.sigma_r) * self.gaus(np.log(1 + phi_diff),
                                                                                          mu=0, sigma=self.sigma_phi)

            weights = (np.product(weights, axis=0))

            if np.sum(weights) > 0:
                if landmarks.shape[0] != 0:
                    weights = weights ** (1. / landmarks.shape[0])
                    weights /= np.sum(weights)
            else:
                weights = np.ones(particles.shape[0], dtype=float) / particles.shape[0]
        else:
            weights = np.ones(particles.shape[0], dtype=float) / particles.shape[0]

        if global_coords is not None:
            x_diff = particles[:, 0] - global_coords[1]
            y_diff = particles[:, 1] - global_coords[2]
            theta_diff = particles[:, 2] - global_coords[3]

            weights *= self.gaus(x_diff, mu=0, sigma=self.sigma_r) * self.gaus(y_diff, mu=0,
                                                                                     sigma=self.sigma_r) * self.gaus(
                theta_diff, mu=0, sigma=self.sigma_phi * 100000)
            weights /= np.sum(weights)

        return weights

    def filter_scan(self, scan, intense):
        ranges = np.array(scan.ranges)
        intensities = np.array(scan.intensities)
        cloud = cvt_ros_scan2points(scan)
        index0 = (intensities > intense) & (ranges < self.max_dist)
        index1 = self.alpha_filter(cloud, self.min_sin)
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

    def get_landmarks(self, scan, intense):
        """Returns filtrated lidar data"""
        ranges = np.array(scan.ranges)
        ind = self.filter_scan(scan, intense)
        final_ind = np.where((np.arange(ranges.shape[0]) * ind) > 0)[0]
        angles = (LIDAR_DELTA_ANGLE * final_ind + LIDAR_START_ANGLE) % (2 * np.pi)
        distances = ranges[final_ind]
        return angles, distances


class InterpPose:

    def __init__(self):
        self.pose_vals = np.zeros(3)
        self.pose = np.zeros(3)

    def add_value(self, pose, stamp):
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z,
             pose.orientation.w]
        angle = wrap_angle(euler_from_quaternion(q)[2] % (2 * np.pi))

        # if len(self.pose_vals) >= 25:
        #     self.pose_vals.popleft()
        self.pose_vals = np.array(
            [pose.position.x, pose.position.y, wrap_angle(angle)])

    def get_iterp_pose(self, time):
        return self.pose_vals
        # if len(self.pose_vals) <= 2:
        #     return np.array([0, 0, 0])
        # odom_arr = np.array(self.pose_vals)
        # odom_x = interpolate.interp1d(odom_arr[:, 0], odom_arr[:, 1], fill_value='extrapolate')
        # odom_y = interpolate.interp1d(odom_arr[:, 0], odom_arr[:, 2], fill_value='extrapolate')
        # odom_theta = interpolate.interp1d(odom_arr[:, 0], unfold_angles(odom_arr[:, 3]), fill_value='extrapolate')
        # return np.array([odom_x(time), odom_y(time), wrap_angle(odom_theta(time))])


class Gaussian(object):
    """
    Represents a multi-variate Gaussian distribution representing the state of the robot.
    """

    def __init__(self, mu, sigma):
        """
        Sets the internal mean and covariance of the Gaussian distribution.

        :param mu: A 1-D numpy array (size 3x1) of the mean (format: [x, y, theta]).
        :param Sigma: A 2-D numpy ndarray (size 3x3) of the covariance matrix.
        """

        assert isinstance(mu, np.ndarray)
        assert isinstance(sigma, np.ndarray)
        assert sigma.shape == (3, 3)

        if mu.ndim < 1:
            raise ValueError('The mean must be a 1D numpy ndarray of size 3.')
        elif mu.shape == (3,):
            # This transforms the 1D initial state mean into a 2D vector of size 3x1.
            mu = mu[np.newaxis].T
        elif mu.shape != (3, 1):
            raise ValueError('The mean must be a vector of size 3x1.')

        self.mu = mu
        self.sigma = sigma


def get_gaussian_statistics(samples):
    """
    Computes the parameters of the samples assuming the samples are part of a Gaussian distribution.

    :param samples: The samples of which the Gaussian statistics will be computed (shape: N x 3).
    :return: Gaussian object from utils.objects with the mean and covariance initialized.
    """

    assert isinstance(samples, np.ndarray)
    assert samples.shape[1] == 3

    # Compute the mean along the axis of the samples.
    mu = np.mean(samples, axis=0)

    # Compute mean of angles.
    angles = samples[:, 2]
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    mu[2] = np.arctan2(sin_sum, cos_sum)

    # Compute the samples covariance.
    mu_0 = samples - np.tile(mu, (samples.shape[0], 1))
    mu_0[:, 2] = np.array([wrap_angle(angle) for angle in mu_0[:, 2]])
    sigma = mu_0.T @ mu_0 / (samples.shape[0] - 1)

    return Gaussian(mu, sigma)
