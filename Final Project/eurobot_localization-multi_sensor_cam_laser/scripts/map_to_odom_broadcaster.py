#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String
import tf_conversions


class FixedTFBroadcaster:

    def __init__(self):
        self.pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
        self.robot_name = rospy.get_param("robot_name")
        self.prev_side_status = None
        rospy.Subscriber("/%s/stm/side_status" % self.robot_name, String, self.callback_side, queue_size=1)
        self.odom_pose = np.zeros(3)
        self.timer = rospy.Timer(rospy.Duration(0, 10), self.broadcast_odom)

    def broadcast_odom(self, event):

        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = "map"
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = "%s_odom" % self.robot_name
        t.transform.translation.x = self.odom_pose[0]
        t.transform.translation.y = self.odom_pose[1]
        t.transform.translation.z = 0.0
        q = tf_conversions.transformations.quaternion_from_euler(0, 0, self.odom_pose[2])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        tfm = tf2_msgs.msg.TFMessage([t])
        self.pub_tf.publish(tfm)

        # t = geometry_msgs.msg.TransformStamped()
        # t.header.frame_id = "%s" % self.robot_name
        # t.header.stamp = rospy.Time.now()
        # t.child_frame_id = "%s_laser" % self.robot_name
        # t.transform.translation.x = 0
        # t.transform.translation.y = 0
        # t.transform.translation.z = 1
        #
        # t.transform.rotation.x = 0
        # t.transform.rotation.y = 0
        # t.transform.rotation.z = 0
        # t.transform.rotation.w = 1
        #
        # tfm = tf2_msgs.msg.TFMessage([t])
        # self.pub_tf.publish(tfm)

    def callback_side(self, side):
        # STM always sent side status
        if self.prev_side_status != side.data:
            if side.data == "1":
                start_pos = np.array(rospy.get_param("start_blue"))
                self.odom_pose = start_pos
            else:
                start_pos = np.array(rospy.get_param("start_yellow"))
                self.odom_pose = start_pos
            self.prev_side_status = side.data


if __name__ == '__main__':
    try:
        rospy.init_node('fixed_tf2_broadcaster')
        tfb = FixedTFBroadcaster()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
