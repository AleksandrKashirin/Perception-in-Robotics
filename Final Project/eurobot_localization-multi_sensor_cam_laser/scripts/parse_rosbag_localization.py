#!/usr/bin/env python

import rosbag
import tf_conversions
import matplotlib.pyplot as plt
import numpy as np
# import scipy


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


rosbag_file = rosbag.Bag("2021-02-19-14-19-02.bag")

robot_name = "main_robot"  # or main_robot or secondary_robot
show_speed = False

data_x_list = []
data_y_list = []
data_angle_list = []
data_time_list = []
scan_time_list = []

data_odom_x_list = []
data_odom_y_list = []
data_odom_angle_list = []
data_odom_time_list = []

data_pose_x_list = []
data_pose_y_list = []
data_pose_angle_list = []
data_pose_time_list = []

stm_data = np.zeros((0, 2))
stm_data_speed_x = np.zeros((0, 2))
stm_data_speed_y = np.zeros((0, 2))
stm_data_speed = np.zeros((0, 2))
stm_data_speed_angle = np.zeros((0, 2))

time = 0
speed = 0
speed_x = 0
speed_y = 0
speed_omega = 0
stm_time = 0

speed_x_list = []
speed_y_list = []
speed_omega_list = []
speed_list = []

max_intenses = np.zeros((0, 2))

for topic, msg, t in rosbag_file.read_messages():
    if topic == "/tf" and msg.transforms:

        # =======================POSITION==========================
        if msg.transforms[0].header.frame_id == robot_name + "_odom":
            data_x = msg.transforms[0].transform.translation.x
            data_y = msg.transforms[0].transform.translation.y

            data_time = msg.transforms[0].header.stamp.to_sec()

            q_x = msg.transforms[0].transform.rotation.x
            q_y = msg.transforms[0].transform.rotation.y
            q_z = msg.transforms[0].transform.rotation.z
            q_w = msg.transforms[0].transform.rotation.w
            q = (q_x, q_y, q_z, q_w)

            stm_time = data_time

            data_angle = wrap_angle(tf_conversions.transformations.euler_from_quaternion(q)[2])

            data_x_list.append(data_x)
            data_y_list.append(data_y)
            data_time_list.append(data_time)
            data_angle_list.append(data_angle)

        # print(msg.transforms[0].header.frame_id)

    if topic == "/" + robot_name + "/odom":
        data_odom_x = msg.pose.pose.position.x
        data_odom_y = msg.pose.pose.position.y

        data_odom_time = msg.header.stamp.to_sec()

        q_odom_x = msg.pose.pose.orientation.x
        q_odom_y = msg.pose.pose.orientation.y
        q_odom_z = msg.pose.pose.orientation.z
        q_odom_w = msg.pose.pose.orientation.w
        q_odom = (q_odom_x, q_odom_y, q_odom_z, q_odom_w)

        data_odom_angle = wrap_angle(tf_conversions.transformations.euler_from_quaternion(q_odom)[2])

        data_odom_x_list.append(data_odom_x)
        data_odom_y_list.append(data_odom_y)
        data_odom_angle_list.append(data_odom_angle)
        data_odom_time_list.append(data_odom_time)

        stm_time = data_odom_time

        # print(msg.header.stamp)

    if topic == "/" + robot_name + "/pose2D":
        data_pose_x = msg.x
        data_pose_y = msg.y

        # data_pose_time = msg.header.stamp.to_sec()

        q_pose = msg.theta

        data_pose_angle = q_pose

        data_pose_x_list.append(data_pose_x)
        data_pose_y_list.append(data_pose_y)
        data_pose_angle_list.append(data_pose_angle)
        # data_pose_time_list.append(data_pose_time)
        # print(msg.header.stamp)

    # =======================LIDAR_DATA==========================
    if topic == "/" + robot_name + "/scan":
        scan_time = msg.header.stamp.to_sec()
        scan_ranges = list(msg.ranges)
        scan_intensities = list(msg.intensities)
        scan_data = zip(*sorted(zip(scan_intensities, scan_ranges), reverse=True))

        scan_data = np.array(scan_data)

        max_intense = scan_data[:, 0].T

        # for data_range, data_intensity in zip(scan_data[1, :20], scan_data[0, :20]):
        # if data_range < 2.2:
        # print(data_range)
        # plt.scatter(data_range, data_intensity, c='g', s=15, edgecolors='none')

        max_intenses = np.vstack((max_intenses, max_intense))

        # plt.show()

        scan_time_list.append(scan_time)

    # =======================SPEED==========================
    if topic == "/" + robot_name + "/stm/command":
        data_splitted = msg.data.split()
        if len(data_splitted) == 5:
            speed_x = float(data_splitted[2])
            speed_y = float(data_splitted[3])
            speed = np.sqrt(speed_x**2 + speed_y**2)
            speed_omega = float(data_splitted[4])
            speed_x_list.append(speed_x)
            speed_y_list.append(speed_y)
            speed_omega_list.append(speed_omega)
            speed_list.append(speed)
            print(data_splitted)

    '''
    if topic == "/" + robot_name + "/stm/response":
        print(msg.data)
        stm_time = 1
    else:
        stm_time = 0
    '''

    # stm_data = np.vstack((stm_data, (np.array([time, stm_time]))))
    stm_data_speed_x = np.vstack((stm_data_speed_x, (np.array([stm_time, speed_x]))))
    stm_data_speed_y = np.vstack((stm_data_speed_y, (np.array([stm_time, speed_y]))))
    stm_data_speed = np.vstack((stm_data_speed, (np.array([stm_time, speed]))))
    stm_data_speed_angle = np.vstack((stm_data_speed_angle, (np.array([stm_time, speed_omega]))))
    time += 1

rosbag_file.close()

x = stm_data[:, 0]
y = stm_data[:, 1]


# fig, ax = plt.subplots()
# plt.plot(x, y, marker='.', lw=1)
# d = scipy.zeros(len(y))
# ax.fill_between(x, y, y2=0, color='b', alpha=0.3)
#
# plt.show()


if show_speed:
    fig, axs = plt.subplots(4, figsize=(15, 20))
else:
    fig, axs = plt.subplots(3, figsize=(15, 20))

axs[0].set_title('Axis X', fontsize=20, style='italic')
axs[0].scatter(data_time_list, data_x_list, c='b', s=3, edgecolors='none', label='laser')
axs[0].scatter(data_odom_time_list, data_odom_x_list, c='r', s=3, edgecolors='none', label='odom')

# axs[0].scatter(scan_time_list, np.ones(len(scan_time_list)), c='g', s=5, edgecolors='none', label='scan')
if show_speed:
    axs[0].plot(stm_data_speed_x[:, 0], stm_data_speed_x[:, 1], c='g', label='speed x')
    axs[0].plot(stm_data_speed[:, 0], stm_data_speed[:, 1], c='black', label='speed linear')
axs[0].legend(loc='best')

axs[0].set_xlabel("t", fontsize=20, style='italic')
axs[0].set_ylabel("x(t)", fontsize=20, style='italic')

axs[0].grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


axs[1].set_title('Axis Y', fontsize=20, style='italic')
axs[1].plot(data_odom_x_list, data_odom_y_list, label='odom')
axs[1].scatter(data_time_list, data_y_list, c='b', s=3, edgecolors='none', label='laser')
# axs[1].scatter(data_odom_time_list, data_odom_y_list, c='r', s=3, edgecolors='none', label='odom')
if show_speed:
    axs[1].plot(stm_data_speed_y[:, 0], stm_data_speed_y[:, 1], c='g', label='speed y')
axs[1].legend(loc='best')


axs[1].set_xlabel("t", fontsize=20, style='italic')
axs[1].set_ylabel("y(t)", fontsize=20, style='italic')

axs[1].grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


axs[2].set_title('Angle', fontsize=20, style='italic')
# start_point_laser = data_angle_list[0]
# start_point_odom = data_odom_angle_list[0]
axs[2].scatter(data_time_list, data_angle_list, c='b', s=3, edgecolors='none', label='laser')
axs[2].scatter(data_odom_time_list, data_odom_angle_list, c='r', s=3, edgecolors='none', label='odom')
axs[2].scatter(data_odom_time_list, data_odom_y_list, c='black', s=3, edgecolors='none', label='IMU')
if show_speed:
    axs[2].plot(stm_data_speed_angle[:, 0], stm_data_speed_angle[:, 1], c='g', label='speed angle')
axs[2].legend(loc='best')

axs[2].set_xlabel("t", fontsize=20, style='italic')
axs[2].set_ylabel("angle(t)", fontsize=20, style='italic')

axs[2].grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


if show_speed:
    axs[3].set_title('Speed', fontsize=20, style='italic')
    axs[3].plot(speed_list, c='b', label='speed_linear')
    axs[3].plot(speed_x_list, c='black', label='speed_x')
    axs[3].plot(speed_y_list, c='g', label='speed_y')
    axs[3].plot(speed_omega_list, c='r', label='speed_angle')
    axs[3].legend(loc='best')

    axs[3].set_xlabel("t", fontsize=20, style='italic')
    axs[3].set_ylabel("speed", fontsize=20, style='italic')

    axs[3].grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


fig.suptitle('Filtered lidar odometry VS raw wheel odometry', fontsize=25, style='italic', linespacing=100)


np.savez('Data_for_perception', X_laser_odometry=data_x_list, Y_laser_odometry=data_y_list,
         x_wheels_odometry=data_odom_x_list, y_wheels_odometry=data_odom_y_list,
         data_odom_time_list=data_odom_time_list, data_time_list=data_time_list,
         data_angle_list=data_angle_list, data_odom_angle_list=data_odom_angle_list)

# plt.figure(figsize=(20, 15))
# plt.plot(data_time_list, data_y_list, c='g', s=5, edgecolors='none')
# plt.scatter(data_odom_time_list[400:500],data_odom_y_list[400:500], c='r', s=3, edgecolors='none')

plt.legend(loc='best')

plt.show()

plt.figure(figsize=(12, 10))
plt.plot(data_odom_x_list, data_odom_y_list, label='odom')
plt.plot(data_x_list, data_y_list, label='lidar')
plt.axis('equal')
plt.show()
ranges = []
intensities = []

'''
for range, intensivity in zip(max_intenses[:, 1], max_intenses[:, 0]):
    if range < 1.0:
        plt.scatter(range, intensivity, c='g', s=8, edgecolors='none')
        ranges.append(range)
        intensities.append(intensities)


# y_mean = [np.mean(intensities) for i in ranges]
# f1 = interp1d(ranges, intensities, kind='nearest')

plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
# plt.plot(ranges, y_mean, c='r')
plt.show()
'''

# print(data_odom_angle_list[200:300])
# print(data_odom_time_list[0])

