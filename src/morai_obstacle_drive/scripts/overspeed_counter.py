#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overspeed counter: counts how many times the TRUE speed (EgoVehicleStatus.velocity)
crosses above a threshold (default 20 kph) -- one count per excursion, edge-triggered
with hysteresis so a speed hovering at the line isn't double-counted. Dev-only (needs
/Ego_topic, which is forbidden in the actual run).

    rosrun morai_obstacle_drive overspeed_counter.py
    rosrun morai_obstacle_drive overspeed_counter.py _threshold_kph:=20 _hysteresis_kph:=0.5
"""
import rospy
from morai_msgs.msg import EgoVehicleStatus


class OverspeedCounter:
    def __init__(self):
        rospy.init_node("overspeed_counter")
        self.thresh = float(rospy.get_param("~threshold_kph", 20.0))
        # must drop this far below the threshold before another crossing is counted
        self.hyst = float(rospy.get_param("~hysteresis_kph", 0.5))
        self.topic = rospy.get_param("~ego_topic", "/Ego_topic")

        self.count = 0
        self.above = False          # currently in an over-threshold excursion
        self.peak = 0.0             # peak of the current excursion
        self.session_peak = 0.0     # highest speed seen overall

        rospy.Subscriber(self.topic, EgoVehicleStatus, self.cb, queue_size=20)
        rospy.Timer(rospy.Duration(3.0), self.status)
        rospy.on_shutdown(self.final)
        rospy.loginfo("overspeed_counter: counting crossings > %.1f kph on %s (hyst %.1f)",
                      self.thresh, self.topic, self.hyst)
        rospy.spin()

    def cb(self, msg):
        v = msg.velocity.x * 3.6          # velocity.x is m/s -> kph
        if v > self.session_peak:
            self.session_peak = v
        if not self.above:
            if v > self.thresh:
                self.above = True
                self.peak = v
                self.count += 1
                rospy.logwarn("=== OVERSPEED #%d: crossed %.0f kph (now %.1f) ===",
                              self.count, self.thresh, v)
        else:
            if v > self.peak:
                self.peak = v
            if v < self.thresh - self.hyst:   # dropped clearly below -> excursion over
                self.above = False
                rospy.logwarn("    #%d ended, peak %.1f kph  |  total over %.0f = %d",
                              self.count, self.peak, self.thresh, self.count)

    def status(self, _evt):
        state = "ABOVE (peak %.1f)" % self.peak if self.above else "ok"
        rospy.loginfo("overspeed count = %d over %.0f kph  [now %s, session peak %.1f]",
                      self.count, self.thresh, state, self.session_peak)

    def final(self):
        rospy.logwarn("==== FINAL: %d overspeed(s) over %.0f kph, session peak %.1f kph ====",
                      self.count, self.thresh, self.session_peak)


if __name__ == "__main__":
    try:
        OverspeedCounter()
    except rospy.ROSInterruptException:
        pass
