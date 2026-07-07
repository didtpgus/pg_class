#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from morai_msgs.msg import EgoVehicleStatus
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


class OdomToEgoStatus:
    def __init__(self):
        rospy.init_node("odom_to_ego_status", anonymous=True)

        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.status_topic = rospy.get_param("~status_topic", "/Ego_topic")
        self.velocity_alpha = float(rospy.get_param("~velocity_alpha", 0.25))
        self.max_reasonable_speed = float(rospy.get_param("~max_reasonable_speed", 30.0))

        self.prev_x = None
        self.prev_y = None
        self.prev_t = None
        self.filtered_speed = 0.0

        self.pub = rospy.Publisher(self.status_topic, EgoVehicleStatus, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)

        rospy.loginfo(
            "Odom to EgoVehicleStatus ready: odom=%s status=%s alpha=%.2f",
            self.odom_topic,
            self.status_topic,
            self.velocity_alpha,
        )
        rospy.spin()

    def odom_callback(self, msg):
        now = msg.header.stamp if not msg.header.stamp.is_zero() else rospy.Time.now()
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        speed = self.filtered_speed
        if self.prev_x is not None and self.prev_y is not None and self.prev_t is not None:
            dt = (now - self.prev_t).to_sec()
            if dt > 1e-3:
                dx = position.x - self.prev_x
                dy = position.y - self.prev_y
                measured_speed = math.hypot(dx, dy) / dt
                if measured_speed <= self.max_reasonable_speed:
                    speed = (
                        (1.0 - self.velocity_alpha) * self.filtered_speed
                        + self.velocity_alpha * measured_speed
                    )

        self.filtered_speed = speed
        self.prev_x = position.x
        self.prev_y = position.y
        self.prev_t = now

        _, _, yaw = euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ])

        status = EgoVehicleStatus()
        status.header.stamp = now
        status.header.frame_id = msg.header.frame_id
        status.position.x = position.x
        status.position.y = position.y
        status.position.z = position.z
        status.velocity.x = self.filtered_speed
        status.heading = math.degrees(yaw)

        self.pub.publish(status)
        rospy.loginfo_throttle(
            1.0,
            "Fake /Ego_topic from /odom: speed=%.2fm/s (%.1fkph)",
            self.filtered_speed,
            self.filtered_speed * 3.6,
        )


if __name__ == "__main__":
    try:
        OdomToEgoStatus()
    except rospy.ROSInterruptException:
        pass
