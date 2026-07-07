#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified TF broadcaster for the obstacle-avoidance stack.

Builds the tree:

    map --(dynamic, from /odom)--> odom --(static)--> base_link --> {gps, imu}

The lidar frame (velodyne) is attached under base_link by a separate
static_transform_publisher in the launch file, so the full chain
velodyne -> base_link -> odom -> map is available for TF lookups
(e.g. obstacle_aware_local_path transforms cluster points into map).

No morai_msgs dependency: the vehicle pose is taken from nav_msgs/Odometry
published by gpsimu_parser.py.
"""

import rospy
import tf
from nav_msgs.msg import Odometry


class TFBroadcaster:
    def __init__(self):
        rospy.init_node("tf_broadcaster", anonymous=True)

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.gps_frame = rospy.get_param("~gps_frame", "gps")
        self.imu_frame = rospy.get_param("~imu_frame", "imu")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")

        # Static mounting offsets (meters) relative to the parent frame.
        # odom -> base_link is usually identity; base_link -> {gps,imu} are the
        # sensor mounting positions on the vehicle (default 0 if unknown).
        self.base_offset = self._xyz("base")          # odom -> base_link
        self.gps_offset = self._xyz("gps")            # base_link -> gps
        self.imu_offset = self._xyz("imu")            # base_link -> imu
        self.no_rotation = (0.0, 0.0, 0.0, 1.0)

        self.br = tf.TransformBroadcaster()
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)

        rospy.loginfo(
            "tf_broadcaster ready: %s -> %s (dynamic) -> %s -> {%s, %s}",
            self.map_frame,
            self.odom_frame,
            self.base_frame,
            self.gps_frame,
            self.imu_frame,
        )
        rospy.spin()

    def _xyz(self, name):
        return (
            float(rospy.get_param("~%s_x" % name, 0.0)),
            float(rospy.get_param("~%s_y" % name, 0.0)),
            float(rospy.get_param("~%s_z" % name, 0.0)),
        )

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        stamp = rospy.Time.now()

        # map -> odom: dynamic vehicle pose from /odom
        self.br.sendTransform(
            (position.x, position.y, position.z),
            (orientation.x, orientation.y, orientation.z, orientation.w),
            stamp,
            self.odom_frame,
            self.map_frame,
        )

        # odom -> base_link: static (identity + optional offset)
        self.br.sendTransform(
            self.base_offset, self.no_rotation, stamp, self.base_frame, self.odom_frame
        )

        # base_link -> gps / imu: static sensor mounting offsets
        self.br.sendTransform(
            self.gps_offset, self.no_rotation, stamp, self.gps_frame, self.base_frame
        )
        self.br.sendTransform(
            self.imu_offset, self.no_rotation, stamp, self.imu_frame, self.base_frame
        )


if __name__ == "__main__":
    try:
        TFBroadcaster()
    except rospy.ROSInterruptException:
        pass
