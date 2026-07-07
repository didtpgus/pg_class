#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf
from nav_msgs.msg import Odometry


class EgoTFPublisher:
    def __init__(self):
        rospy.init_node("ego_tf_pub", anonymous=True)
        self.parent_frame = rospy.get_param("~parent_frame", "map")
        self.child_frame = rospy.get_param("~child_frame", "base_link")
        self.extra_child_frame = rospy.get_param("~extra_child_frame", "Ego")
        self.z = float(rospy.get_param("~z", 0.0))
        self.br = tf.TransformBroadcaster()
        rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1)
        rospy.spin()

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        translation = (position.x, position.y, self.z)
        rotation = (orientation.x, orientation.y, orientation.z, orientation.w)
        stamp = rospy.Time.now()

        self.br.sendTransform(translation, rotation, stamp, self.child_frame, self.parent_frame)
        if self.extra_child_frame:
            self.br.sendTransform(translation, rotation, stamp, self.extra_child_frame, self.parent_frame)


if __name__ == "__main__":
    try:
        EgoTFPublisher()
    except rospy.ROSInterruptException:
        pass
