#!/usr/bin/env python3
import numpy as np
import scipy.optimize
import warnings

import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import TransformStamped, PoseArray, Pose, Quaternion, PoseStamped, PoseWithCovarianceStamped, \
    Pose2D
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import tf_conversions

from core_functions import cvt_global2local, cvt_local2global, find_src, wrap_angle, unfold_angles
from np_particle import ParticleFilter, InterpPose

PF_RATE = rospy.get_param("rate")  # 20
BEAC_R = rospy.get_param("beacons_radius")
WORLD_X = rospy.get_param("world_x")
WORLD_Y = rospy.get_param("world_y")
WORLD_BORDER = rospy.get_param("world_border")
BEAC_L = rospy.get_param("beac_l")
BEAC_BORDER = rospy.get_param("beac_border")

YELLOW_BEACONS = np.array([[WORLD_X + WORLD_BORDER + BEAC_BORDER + BEAC_L / 2., WORLD_Y / 2.],
                           [-(WORLD_BORDER + BEAC_BORDER + BEAC_L / 2.), WORLD_Y - BEAC_L / 2.],
                           [-(WORLD_BORDER + BEAC_BORDER + BEAC_L / 2.), BEAC_L / 2.]])

BLUE_BEACONS = np.array([[-(WORLD_BORDER + BEAC_BORDER + BEAC_L / 2.), WORLD_Y / 2.],
                         [WORLD_X + WORLD_BORDER + BEAC_BORDER + BEAC_L / 2., WORLD_Y - BEAC_L / 2.],
                         [WORLD_X + WORLD_BORDER + BEAC_BORDER + BEAC_L / 2., BEAC_L / 2.]])

beacon_assumption_dist = 0.2

terms = [
    -8.6548337881836648e-003,
    1.2311644873210734e+000,
    -1.1211778518213729e+000,
    2.5215401727391451e+000,
    -2.9919937073658538e+000,
    2.0019044463325177e+000,
    -7.5987024304760820e-001,
    1.5263965297147822e-001,
    -1.2598767142069580e-002
]


def regress(x):
    t = 1
    r = 0
    for c in terms:
        r += c * t
        t *= x
    return r


def dict_append(var, val, dic):
    if var in dic:
        dic[var].append(val)
    else:
        dic[var] = [val]


class PFNode(object):
    def __init__(self):

        self.beacon_ranges = []

        # scan params
        self.robot_name = rospy.get_param("robot_name")
        self.alpha = rospy.get_param("alpha")
        self.scan_offset = float(rospy.get_param("scan_offset"))  # intrinsic lidar offset
        self.beacon_radius = rospy.get_param("beacons_radius")
        self.beacon_range = rospy.get_param("beacon_range")
        self.min_range = rospy.get_param("min_range")
        self.min_points_per_beacon = rospy.get_param("min_point_per_beacon")
        # lidar_coords are external offset related to robot frame
        self.lidar_coords = np.array(
            [rospy.get_param("lidar_x"), rospy.get_param("lidar_y"), rospy.get_param("lidar_a")])

        # start attributes
        self.prev_side_status = None
        self.color = None
        self.beacons = []
        self.world_beacons = []
        self.scan = None
        self.last_odom = np.zeros(3)
        self.pf = None  # later on init Particle Filter here

        # times
        self.timer = None
        self.prev_lidar_time = rospy.Time.now()
        self.lidar_time = rospy.Time.now()
        self.scan_time = rospy.Time.now()  # only for extrapolation

        # ros publishers
        self.particle_pub = rospy.Publisher("my_particles", PoseArray, queue_size=1)
        self.beacons_publisher = rospy.Publisher("my_beacons", MarkerArray, queue_size=1)
        self.landmark_publisher = rospy.Publisher("landmarks", MarkerArray, queue_size=2)
        self.pose_pub = rospy.Publisher("/%s/filtered_coords" % self.robot_name, PoseWithCovarianceStamped,
                                        queue_size=1)
        # tf2 listeners
        # self.lidar_buffer = tf2_ros.Buffer()
        # self.listener_lidar = tf2_ros.TransformListener(self.lidar_buffer)
        # # tf2 buffer containing wheeled odometry (sent from STM)
        # self.wheel_odom_buffer = tf2_ros.Buffer()
        # self.listener_odom = tf2_ros.TransformListener(self.wheel_odom_buffer)
        # tf2 broadcaster
        self.broadcast_odom_frame = tf2_ros.TransformBroadcaster()
        # tf2 listener to obtain coords for beacons filtering
        # self.tfBuffer = tf2_ros.Buffer()
        # self.lis = tf2_ros.TransformListener(self.tfBuffer)

        self.robot_odom_coords = None
        self.prev_robot_odom = None

        # FIXME stop recording
        self.record_data = bool(rospy.get_param("/%s/record_data" % self.robot_name, 0))

        self.wheel_pose = np.zeros(4)
        self.lidar_pose = np.zeros(4)
        self.imu_pose = np.zeros(4)
        self.camera_pose = np.zeros(4)

        self.prev_wheel_odom = np.zeros(3)
        self.odom_arr = np.zeros(3)
        self.odom_time = np.zeros(1)

        self.prev_wheel_odom_coords = np.zeros(4)
        self.prev_lidar_odom_coords = np.zeros(4)
        self.prev_imu_coords = np.zeros(4)

        self.dict_pf = {}

        self.time_start = rospy.Time.now().to_sec()
        self.data_prev = False
        self.flag = True

        self.robot_filtered_coords = np.zeros(3)
        # ros subscribers
        # rospy.Subscriber("/tf", TransformStamped, self.callback_frame, queue_size=1)
        rospy.Subscriber("stm/side_status", String, self.callback_side, queue_size=1)
        rospy.Subscriber("/%s/scan" % self.robot_name, LaserScan, self.scan_callback, queue_size=1)
        # rospy.Subscriber("/%s/imu/data_raw" % self.robot_name, Imu, self.imu_callback,
        #                  queue_size=1)

        rospy.Subscriber("/%s/pose_stamped" % self.robot_name, PoseStamped, self.lidar_odom_callback,
                         queue_size=1)

        rospy.Subscriber("/%s/odom" % self.robot_name, Odometry, self.wheel_odom_callback,
                         queue_size=1)
        rospy.Subscriber("/%s/camera/pose" % self.robot_name, PoseStamped, self.camera_pose_callback,
                         queue_size=1)

        rospy.Subscriber("/%s/imu/data_raw" % self.robot_name, Imu, self.imu_callback,
                         queue_size=1)

    # TODO for testing
    #     rospy.Subscriber("/stm/start_status", String, self.start_callback, queue_size=1)
    #
    # def start_callback(self, data):
    #     if data.data == "1" and not self.data_prev:
    #         self.data_prev = True
    #         self.time_start = rospy.Time.now().to_sec()

    def imu_callback(self, data):

        q = [data.orientation.x, data.orientation.y, data.orientation.z,
             data.orientation.w]
        angle = -wrap_angle(euler_from_quaternion(q)[2] % (2 * np.pi))
        self.imu_pose = np.array(
            [data.header.stamp.to_sec(), 0, 0, wrap_angle(angle)])

    def camera_pose_callback(self, data):
        q = [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z,
             data.pose.orientation.w]
        angle = wrap_angle(euler_from_quaternion(q)[2] % (2 * np.pi))
        self.camera_pose = np.array(
            [data.header.stamp.to_sec(), data.pose.position.x, data.pose.position.y, wrap_angle(angle)])

    def wheel_odom_callback(self, data):
        q = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z,
             data.pose.pose.orientation.w]
        angle = wrap_angle(euler_from_quaternion(q)[2] % (2 * np.pi))
        self.wheel_pose = np.array(
            [data.header.stamp.to_sec(), data.pose.pose.position.x, data.pose.pose.position.y, wrap_angle(angle)])

    def lidar_odom_callback(self, data):
        q = [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z,
             data.pose.orientation.w]
        angle = wrap_angle(euler_from_quaternion(q)[2] % (2 * np.pi))
        self.lidar_pose = np.array(
            [data.header.stamp.to_sec(), data.pose.position.x, data.pose.position.y, wrap_angle(angle)])

    def scan_callback(self, data):
        if self.pf is None:
            rospy.logwarn("PF: Didn't get side status")
            return 0
        scan_time = data.header.stamp
        scan_time_in_norm = data.header.stamp.to_sec() - self.time_start
        lidar_odom_coords = self.lidar_pose
        wheel_odom_coords = self.wheel_pose
        imu_coords = self.imu_pose
        camera_coords = self.camera_pose

        header = data.header
        points = self.point_cloud_from_scan(data)
        beacons, color = self.beacons_detection(points)
        # self.publish_beacons(beacons, header, color)
        if beacons.shape[0] != 0:
            ind = self.beacons_filter(beacons)
            beacons = beacons[ind]
            self.publish_beacons(beacons, header, np.array([0, 1, 0]))

        self.prev_robot_odom = self.robot_odom_coords

        delta_lidar = cvt_global2local(lidar_odom_coords[1:4], self.prev_lidar_odom_coords[1:4])
        delta_odom = cvt_global2local(wheel_odom_coords[1:4], self.prev_wheel_odom_coords[1:4])
        delta_imu = cvt_global2local(imu_coords[1:4], self.prev_imu_coords[1:4])
        # delta_lidar += np.random.normal(0, 0.1, 3)
        # # delta_lidar = np.zeros(3)

        if abs(delta_imu[2]) >= 0.6 or scan_time.to_sec() - imu_coords[0] >= 0.5:
            # rospy.logwarn("imu error %s" % delta_imu)
            delta_imu = None

        if np.linalg.norm(delta_lidar[:2]) >= 0.1 or abs(delta_lidar[2]) >= 0.6 or scan_time.to_sec() - \
                lidar_odom_coords[0] >= 0.5:
            # rospy.logwarn("Lidar odom error %s" % delta_lidar)
            delta_lidar = None

        if np.linalg.norm(delta_odom[:2]) >= 0.1 or abs(delta_odom[2]) >= 0.6 or scan_time.to_sec() - wheel_odom_coords[
            0] >= 0.5:
            # rospy.logwarn("wheel odom error %s" % delta_odom)
            delta_odom = None

        if scan_time.to_sec() - self.camera_pose[0] >= 0.5:
            # rospy.logwarn("camera pose error %s" % camera_coords)
            camera_coords = None

        filtered_gaussian = self.pf.localization(np.array([delta_lidar, delta_odom, delta_imu], dtype=object), beacons,
                                                 camera_coords)
        self.robot_filtered_coords = find_src(filtered_gaussian.mu.T[0], self.lidar_coords)
        odom_frame_coords = cvt_global2local(self.robot_filtered_coords, self.start_pos)

        # self.pf_poses.append(np.array([rospy.get_time(), self.robot_filtered_coords[0], self.robot_filtered_coords[1], self.robot_filtered_coords[2]]))

        self.publish_odom_frame_robot(odom_frame_coords, scan_time)

        self.publish_position(self.robot_filtered_coords, filtered_gaussian.sigma, scan_time)
        self.prev_lidar_odom_coords = lidar_odom_coords.copy()
        self.prev_wheel_odom_coords = wheel_odom_coords.copy()
        self.prev_imu_coords = imu_coords.copy()

    def publish_position(self, filtered_pose, sigma, header_stamp):
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = header_stamp
        pose.header.frame_id = "map"
        pose.pose.pose.position.x = filtered_pose[0]
        pose.pose.pose.position.y = filtered_pose[1]
        pose.pose.pose.position.z = 0
        pose.pose.pose.orientation = Quaternion(
            *tf_conversions.transformations.quaternion_from_euler(0, 0, filtered_pose[2]))
        pose.pose.covariance = [np.sqrt(sigma[0, 0]), 0, 0, 0, 0, 0,
                                0, np.sqrt(sigma[1, 1]), 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, np.sqrt(sigma[2, 2])]
        self.pose_pub.publish(pose)

    def callback_side(self, side):
        # STM always sent side status
        if self.prev_side_status != side.data:
            if side.data == "1":
                rospy.logwarn("blue_side")
                self.color = "blue"
                # start pos in global frame "map"
                self.start_pos = np.array(rospy.get_param("start_blue"))
                self.robot_odom_coords = self.start_pos
                self.prev_robot_odom = self.start_pos
                beacons = BLUE_BEACONS
            else:
                rospy.logwarn("yellow_side")
                self.color = "yellow"
                self.start_pos = np.array(rospy.get_param("start_yellow"))
                self.robot_odom_coords = self.start_pos
                self.prev_robot_odom = self.start_pos
                beacons = YELLOW_BEACONS
            self.world_beacons = beacons
            self.prev_side_status = side.data
            self.robot_filtered_coords = self.start_pos
            self.pf = ParticleFilter(self.world_beacons, color=self.color, start_x=self.start_pos[0],
                                     start_y=self.start_pos[1],
                                     start_angle=self.start_pos[2])

    # def get_odom(self):
    #     """
    #     get wheeled odometry from tf buffer in current time
    #     used only when wheels odometry posted
    #     returns the last available transformation from tf frames
    #     """
    #     try:
    #         t = self.wheel_odom_buffer.lookup_transform('%s_odom' % self.robot_name, '%s_laser' % self.robot_name,
    #                                                     rospy.Time(0))
    #         q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
    #         yaw = tf_conversions.transformations.euler_from_quaternion(q)[2]
    #         return True, np.array([t.transform.translation.x, t.transform.translation.y, yaw])
    #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    #         rospy.logwarn("Transform for PF with error")
    #
    #     return False, np.array([0, 0, 0])

    def point_cloud_from_scan(self, scan):
        distances_to_beac = np.linalg.norm(self.robot_filtered_coords[:2] - self.world_beacons, axis=1)
        closest_beac = min(distances_to_beac)
        if closest_beac < 0.4:
            intensity = 2200
        else:
            intensity = 4000
        # ???????
        angles, ranges = self.pf.get_landmarks(scan, intensity)
        # # ranges += self.scan_offset
        # ranges = [r + self.scan_offset if r > 1.5 else r + 0.04 for r in ranges]
        ranges = [regress(r) for r in ranges]
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.array([x, y]).T
        return points

    def beacons_detection(self, point_cloud):
        points_number = point_cloud.shape[0]
        marked_points = [False] * points_number
        beacons = []
        for i in range(points_number):
            if not marked_points[i]:
                nearest_points = np.linalg.norm(point_cloud - point_cloud[i], axis=1) < self.beacon_range
                # or statement
                marked_points = nearest_points | marked_points
                if np.count_nonzero(nearest_points) >= self.min_points_per_beacon:
                    beacons.append(self.find_beacon(point_cloud[nearest_points], self.beacon_radius))
        beacons = np.array(beacons)
        # filter beacons
        # is_belong = self.beacons_filter(beacons)
        # beacons = beacons[is_belong]
        color = np.array([1, 0, 0])
        return np.array(beacons), color

    def beacons_filter(self, beacons):
        """ to filter found beacons that do not belong to expected beacons (yellow or blue ones)"""
        beac_num = beacons.shape[0]
        beacons = cvt_local2global(beacons, self.robot_filtered_coords.copy())
        is_belong = [False] * beac_num
        if self.color == "blue":
            for i in range(3):
                distance = np.linalg.norm(beacons - BLUE_BEACONS[i], axis=1)
                if np.sum(distance < beacon_assumption_dist) > 1:
                    ind_min = np.argmin(distance)
                    is_beacon_from_list = [False] * beac_num
                    is_beacon_from_list[ind_min] = True
                else:
                    is_beacon_from_list = distance < beacon_assumption_dist
                is_belong = np.array(is_belong) | np.array(is_beacon_from_list)
        else:
            for i in range(3):
                distance = np.linalg.norm(beacons - YELLOW_BEACONS[i], axis=1)
                if np.sum(distance < beacon_assumption_dist) > 1:
                    # rospy.logwarn("MANY BEACONS AT THE SAME PLACE")
                    ind_min = np.argmin(distance)
                    is_beacon_from_list = [False] * beac_num
                    is_beacon_from_list[ind_min] = True
                else:
                    # rospy.logwarn("Onither PLACE")
                    is_beacon_from_list = distance < beacon_assumption_dist
                # is_beacon_from_list = np.linalg.norm(beacons - YELLOW_BEACONS[i], axis=1) < 0.2
                is_belong = np.array(is_belong) | np.array(is_beacon_from_list)
        # beacons = beacons[is_belong]
        return is_belong

    def point_extrapolation(self, prev_coords, curr_coords, prev_time, curr_time, next_time):
        """
        basic proportions: curr_time - prev_time : prev_coords -> curr_coords
        next_time - prev_time: prev_coords -> extrapolated_coords
        """
        dt21 = curr_time.to_sec() - prev_time.to_sec()
        dt31 = next_time.to_sec() - prev_time.to_sec()
        if np.abs(dt21) > 1E-6:
            delta_extrapolated = cvt_global2local(curr_coords, prev_coords) * dt31 / dt21
        else:
            delta_extrapolated = np.array([0, 0, 0])
        return cvt_local2global(delta_extrapolated, prev_coords)

    def get_odom_frame(self, filtered_coords, odom_coords):
        odom = find_src(filtered_coords, odom_coords)
        self.last_odom[:2] = (1 - self.alpha) * self.last_odom[:2] + self.alpha * odom[:2]
        self.last_odom[2] = odom[2]
        return self.last_odom

    @staticmethod
    def find_beacon(points, beacon_radius):
        """
            Search for beacon center point
        """
        # def fun(x):
        #     scs = np.sum((points - x) * (-x), axis=1) / np.linalg.norm(x)
        #     return np.abs(np.linalg.norm(points - x, axis=1) - beacon_radius) - np.where(scs < 0, scs, 0)
        #
        # # rospy.logwarn(points)
        #
        # res = scipy.optimize.least_squares(fun, points[0])
        # return np.array(res.x)
        x = []
        y = []
        for point in range(1, len(points) - 1):
            try:
                k1 = -((points[point, 0] - points[0, 0]) / (points[point, 1] - points[0, 1]))
                k2 = -((points[-1, 0] - points[point, 0]) / (points[-1, 1] - points[point, 1]))

                b1 = points[point, 1] - k1 * points[point, 0]
                b2 = points[point, 1] - k2 * points[point, 0]

                beacon_x = (b2 - b1) / (k1 - k2)
                x.append(beacon_x)
                y.append(k1 * beacon_x + b1)
            except RuntimeWarning:
                continue

        x = np.mean(x)
        y = np.mean(y)

        return np.array([x, y])

    @staticmethod
    def tf_to_coords(odom):
        q = [odom.transform.rotation.x, odom.transform.rotation.y,
             odom.transform.rotation.z, odom.transform.rotation.w]
        yaw = tf_conversions.transformations.euler_from_quaternion(q)[2]
        x = odom.transform.translation.x
        y = odom.transform.translation.y
        return np.array([x, y, yaw])

    @staticmethod
    def particle_to_pose(self, particle):
        pose = Pose()
        pose.position.x = particle[0]
        pose.position.y = particle[1]
        pose.position.z = 1
        pose.orientation = Quaternion(*tf_conversions.transformations.quaternion_from_euler(0, 0, particle[2]))
        return pose

    def publish_beacons(self, beacons, header, color):
        markers = []
        for i, beacon in enumerate(beacons):
            marker = Marker()
            marker.header = header
            marker.ns = "beacons"
            marker.id = i
            marker.type = 3
            marker.pose.position.x = beacon[0]
            marker.pose.position.y = beacon[1]
            marker.pose.position.z = 0
            marker.pose.orientation.w = 1
            marker.scale.x = 2 * self.beacon_radius
            marker.scale.y = 2 * self.beacon_radius
            marker.scale.z = 0.2
            marker.color.a = 1
            marker.color.g = color[1]
            marker.color.b = color[0]
            marker.color.r = color[2]
            marker.lifetime = rospy.Duration(0.3)
            markers.append(marker)
        self.beacons_publisher.publish(markers)

    def publish_odom_frame(self, coords, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "map"
        t.child_frame_id = "%s_odom" % self.robot_name
        t.transform.translation.x = coords[0]
        t.transform.translation.y = coords[1]
        t.transform.translation.z = 0.0
        q = tf_conversions.transformations.quaternion_from_euler(0, 0, coords[2])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.broadcast_odom_frame.sendTransform(t)

    def publish_odom_frame_robot(self, coords, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "%s_odom" % self.robot_name
        t.child_frame_id = self.robot_name
        t.transform.translation.x = coords[0]
        t.transform.translation.y = coords[1]
        t.transform.translation.z = 0.0
        q = tf_conversions.transformations.quaternion_from_euler(0, 0, coords[2])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.broadcast_odom_frame.sendTransform(t)

    def publish_particles(self):
        pose_viz = PoseArray()
        pose_viz.header.stamp = rospy.Time.now()
        pose_viz.header.frame_id = "map"
        pose_viz.poses = self.particles_to_poses()
        self.particle_pub.publish(pose_viz)

    def particles_to_poses(self):
        return list(map(self.particle_to_pose, self.pf.particles))

    def particle_to_pose(self, particle):
        pose = Pose()
        pose.position.x = particle[0]
        pose.position.y = particle[1]
        pose.position.z = 1
        pose.orientation = Quaternion(*tf_conversions.transformations.quaternion_from_euler(0, 0, particle[2]))
        return pose


def shutdown():
    rospy.logwarn("Particle filter shutting down")


if __name__ == '__main__':
    try:
        rospy.on_shutdown(shutdown)
        rospy.init_node('particle_filter_node', anonymous=True)
        warnings.filterwarnings("error")
        pf_node = PFNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
