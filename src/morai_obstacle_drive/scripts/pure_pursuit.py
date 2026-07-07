#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import atan2, cos, sin, sqrt, pi, degrees

import numpy as np
from geometry_msgs.msg import Point
from morai_msgs.msg import CtrlCmd
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float32
from tf.transformations import euler_from_quaternion

# Optional: MORAI gear/event service, for shifting to Parking (P) at the goal.
try:
    from morai_msgs.msg import EventInfo
    from morai_msgs.srv import MoraiEventCmdSrv
    _HAVE_EVENT_CMD = True
except Exception:
    _HAVE_EVENT_CMD = False


class PIDControl:
    """Plain textbook PID (P + clamped-I + D). Output is a throttle/brake command
    in [-1, 1]: positive -> accel, negative -> brake."""

    def __init__(self):
        # Re-verified by closed-loop sweep after the gpsimu speed_scale calibration:
        # P~0.04 is the flat optimum, i=0 is best (accurate speed needs no integral).
        self.p_gain = rospy.get_param("~p_gain", 0.03)
        self.i_gain = rospy.get_param("~i_gain", 0.0)
        self.d_gain = rospy.get_param("~d_gain", 0.00)
        # Clamp on the integral term (throttle units) to bound windup.
        self.integral_limit = rospy.get_param("~integral_limit", 0.3)
        self.prev_error = 0.0
        self.i_control = 0.0

    def reset(self):
        self.prev_error = 0.0
        self.i_control = 0.0

    def update_gains(self):
        # Re-read gains live so they can be tuned via rosparam without a restart.
        self.p_gain = rospy.get_param("~p_gain", self.p_gain)
        self.i_gain = rospy.get_param("~i_gain", self.i_gain)
        self.d_gain = rospy.get_param("~d_gain", self.d_gain)
        self.integral_limit = rospy.get_param("~integral_limit", self.integral_limit)

    def pid(self, target_vel_kph, current_vel_kph, dt):
        dt = max(dt, 1e-3)
        error = target_vel_kph - current_vel_kph

        p_control = self.p_gain * error
        self.i_control += self.i_gain * error * dt
        self.i_control = max(-self.integral_limit, min(self.integral_limit, self.i_control))
        d_control = self.d_gain * (error - self.prev_error) / dt
        self.prev_error = error

        return p_control + self.i_control + d_control


class PurePursuitPIDCurvature:
    def __init__(self):
        rospy.init_node("pure_pursuit_pid_curvature", anonymous=True)

        rospy.Subscriber("local_path", Path, self.path_callback)
        rospy.Subscriber("odom", Odometry, self.odom_callback)
        # Traffic-light node raises this flag on a red light in a straight section.
        # While True we hand the control topic over to it and only brake here, so
        # the two publishers don't fight over ctrl_cmd_0.
        rospy.Subscriber("/traffic_light_stop", Bool, self.traffic_stop_callback)
        self.traffic_stop = False

        self.ctrl_cmd_pub = rospy.Publisher("ctrl_cmd_0", CtrlCmd, queue_size=1)
        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 1

        # Debug speeds (kph) for PlotJuggler
        self.target_speed_pub = rospy.Publisher("target_speed", Float32, queue_size=1)
        self.current_speed_pub = rospy.Publisher("current_speed", Float32, queue_size=1)

        self.is_path = False
        self.is_odom = False

        self.path = Path()
        self.vehicle_yaw = 0.0
        self.odom_pitch = 0.0          # rad, vehicle tilt from odom attitude (road grade)
        self.current_position = Point()
        self.current_vel = 0.0  # m/s, taken from /odom twist
        self.forward_point = Point()

        self.vehicle_length = rospy.get_param("~vehicle_length", 1.0)
        self.min_lfd = rospy.get_param("~min_lfd", 1.5)
        self.max_lfd = rospy.get_param("~max_lfd", 4.0)
        # lfd-vs-speed shaping exponent (>1 = sub-linear: lfd stays short across
        # the low/mid band and only climbs near max speed). 1.0 = linear.
        # ~3.4 gives lfd ~1.8m at 13kph for min/max_lfd = 1.0/3.0, max_speed 17.
        self.lfd_speed_exp = rospy.get_param("~lfd_speed_exp", 3.4)

        self.max_speed_kph = rospy.get_param("~max_speed_kph", 17.0)
        self.min_curve_speed_kph = rospy.get_param("~min_curve_speed_kph", 14.0)
        self.lateral_acc_limit = rospy.get_param("~lateral_acc_limit", 1.4)
        self.curvature_lookahead = rospy.get_param("~curvature_lookahead", 13)
        # Explicit cap on the commanded target speed (applied after the
        # curvature-based target is computed). Keeps the speed setpoint bounded
        # here rather than relying on max_speed_kph inside curvature_to_speed.
        self.target_speed = rospy.get_param("~target_speed", self.max_speed_kph)
        # Limit how fast the target speed may DROP (kph/s). Smooths the instant
        # target fall at corner entry (17->13.5) so the PID doesn't brake-spike
        # and overshoot far below target. Acceleration stays instant.
        self.target_decel_rate = rospy.get_param("~target_decel_rate", 5.0)
        # Low-pass (EMA) on the curvature-based target speed. The curvature estimate
        # (top-K over a shifting lookahead window) is jumpy, so the raw target
        # oscillates (e.g. 14<->17) -> the controller brakes then re-accelerates ->
        # curve surge. Smoothing gives a stable target. Lower = smoother, more lag.
        self.target_filter_alpha = rospy.get_param("~target_filter_alpha", 0.15)
        self.filtered_target_vel = 0.0
        # Velocity feedforward: base throttle (per kph of target) to HOLD the
        # target speed, so the P term only trims small errors instead of settling
        # ~2 kph low (P-only steady-state error). No integrator -> no windup.
        self.speed_ff_gain = rospy.get_param("~speed_ff_gain", 0.013)
        # Steering feedforward: extra throttle proportional to |steering| to
        # compensate cornering drag (tire scrub) in curves -- so the car holds the
        # curve target speed instead of bleeding down, WITHOUT a global integrator.
        # Straights (near-zero steering) get almost none.
        self.steer_ff_gain = rospy.get_param("~steer_ff_gain", 0.0)  # throttle per rad
        # Overspeed limiter: this low-drag plant coasts above the target cap and
        # the PID won't brake it down (integral pumped positive by corners cancels
        # the P brake). Force a brake proportional to the excess, ONLY near the cap
        # (straights) so corner braking is left to the normal PID.
        self.overspeed_brake_gain = rospy.get_param("~overspeed_brake_gain", 1.0)
        self.overspeed_deadband = rospy.get_param("~overspeed_deadband", 0.0)
        # Minimum brake FLOOR applied the instant an overspeed is detected (added on
        # top of the proportional term). The proportional brake is tiny near the cap
        # (gain*small_excess), and MORAI ignores tiny brake commands (a brake
        # dead-zone), so on a downhill the speed runs away until the command finally
        # grows big enough to bite (~20 kph). A floor makes the brake bite at once.
        self.overspeed_min_brake = rospy.get_param("~overspeed_min_brake", 0.5)
        # Anticipate an ACCELERATING overspeed: if the car is already past the
        # throttle-cut point (cap - release_margin) yet still speeding up, that push
        # is external (downhill/tailwind) -> use the full upward trend in the
        # look-ahead so the brake fires BEFORE crossing the cap. On a flat straight
        # the car always decelerates above the cut point (throttle is off), so this
        # never triggers there -> no extra braking on flat.
        self.overspeed_anticipate = rospy.get_param("~overspeed_anticipate", True)
        # Coast-down anticipation: while over the cap, throttle is cut, so the car
        # already decelerates on drag. We look this many seconds ahead using the
        # speed trend and ease the overspeed brake off early instead of braking
        # until the speed is fully back down (which over-decelerates -> surge).
        self.overspeed_lookahead = rospy.get_param("~overspeed_lookahead", 0.7)  # s
        self.overspeed_rate_alpha = rospy.get_param("~overspeed_rate_alpha", 0.5)  # rate EMA
        # Downhill pre-brake (crest-triggered grade cap). Grade = SMOOTHED odom pitch
        # (gravity-referenced IMU attitude gpsimu passes through). On this track uphill =
        # -grade, descent = +grade, and the plain is a gentle descent. A CREST state
        # machine fires only the steep descent after a steep climb: ARM when grade climbs
        # past dh_arm_pitch (negative deg), FIRE when it then crosses up through
        # dh_fire_pitch (~0) = topped out into the descent, hold dh_hold_time s. The
        # gentle plain never reaches dh_arm_pitch -> never fires (no laptime cost). While
        # held, cut = dh_cap_gain KPH per deg of descent past dh_pitch_dead, up to
        # dh_cap_max, SLEWED (down dh_apply_rate / up dh_recover). dh_pitch_sign flips if
        # a rig reports the grade the other way.
        self.dh_pitch_tau = rospy.get_param("~dh_pitch_tau", 0.8)       # s, pitch smoothing
        self.dh_pitch_sign = rospy.get_param("~dh_pitch_sign", 1.0)    # +1: uphill=-grade, descent=+grade
        self.dh_arm_pitch = rospy.get_param("~dh_arm_pitch", -2.3)     # deg; climb past this arms the crest
        self.dh_fire_pitch = rospy.get_param("~dh_fire_pitch", 0.0)    # deg; armed + cross up through this fires
        self.dh_hold_time = rospy.get_param("~dh_hold_time", 4.0)      # s; how long the cut holds after a crest
        self.dh_pitch_dead = rospy.get_param("~dh_pitch_dead", 0.5)    # deg descent before the cut ramps in
        self.dh_cap_gain = rospy.get_param("~dh_cap_gain", 2.0)         # kph reduction per deg past dead
        self.dh_cap_max = rospy.get_param("~dh_cap_max", 4.0)           # kph
        self.dh_apply_rate = rospy.get_param("~dh_apply_rate", 6.0)    # kph/s, cap comes down
        self.dh_recover = rospy.get_param("~dh_recover", 3.0)           # kph/s, cap goes back up
        self.dh_reduction = 0.0                                          # current cap reduction (kph)
        self.dh_pitch_lp = 0.0                                           # smoothed grade (deg)
        self.dh_armed = False                                            # climbed past arm -> ready to fire
        self.dh_hold_timer = 0.0                                         # s left holding the cut after a crest
        self.dh_prev_active = False                                      # for ON/OFF transition logging
        # Predictive ACCEL (throttle release): anticipate reaching target on the
        # current acceleration this many seconds ahead and ease the throttle early,
        # so the car coasts up to target (less overshoot -> less braking) and coasts
        # DOWN into curves on drag instead of braking. Larger = release earlier.
        self.accel_lookahead = rospy.get_param("~accel_lookahead", 0.2)  # s (overshoot easing)
        # Separate horizon for the BELOW-target-and-decelerating predictive accel
        # (catch-up): larger -> more throttle sooner to arrest a drop under target.
        # Independent from accel_lookahead so tuning catch-up doesn't slow the
        # overshoot easing (and vice versa).
        self.decel_accel_lookahead = rospy.get_param("~decel_accel_lookahead", 0.2)  # s
        # Release the throttle once the (predicted) speed is within this many kph
        # BELOW target -> coast the last bit up, no overshoot past target.
        self.throttle_release_margin = rospy.get_param("~throttle_release_margin", 1.4)  # kph
        # Floor CATCH-UP boost: SEPARATE from the below-target predictive catch-up
        # (decel_accel_lookahead). Engages ONLY when the actual speed drops below the
        # MIN curve speed floor (min_curve_speed_kph) -- i.e. under even the slowest
        # curve target (hard cornering / avoidance drag) -- and adds throttle
        # proportional to how far under, so it recovers fast. Zero effect at/above the
        # floor, so it never touches normal cruise or curve holding.
        self.catchup_gain = rospy.get_param("~catchup_gain", 0.15)     # throttle per kph below floor (past deadband)
        self.catchup_deadband = rospy.get_param("~catchup_deadband", 0.0)  # kph below floor before it engages
        self.catchup_max = rospy.get_param("~catchup_max", 0.15)        # cap on the boost (accel units)
        self.prev_vel_kph = None
        self.vel_rate_filt = 0.0
        # Brake slew-rate limit (1/s): max change of the brake command per second,
        # so the brake ramps instead of spiking at corner entry.
        self.brake_rate = rospy.get_param("~brake_rate", 0.001)
        self.prev_brake_cmd = 0.0
        # Throttle deadband: outputs below this are treated as coast (accel=0), so
        # accel and brake never chatter/co-fire near the target speed.
        self.accel_deadband = rospy.get_param("~accel_deadband", 0.02)
        # Accel jerk limit (accel-command units / s): max change of the throttle
        # command per second, so acceleration ramps smoothly instead of jumping
        # ("확확"). Lower = smoother/softer, higher = snappier.
        self.accel_jerk_limit = rospy.get_param("~accel_jerk_limit", 0.2)
        # Separate slew limit for the DOWNWARD (throttle-release / lift-off)
        # direction of the accel command. Defaults to accel_jerk_limit if unset,
        # so behavior is unchanged until tuned. Lower = softer lift-off (coasts
        # off throttle more gently), higher = snappier release.
        self.accel_release_jerk_limit = rospy.get_param(
            "~accel_release_jerk_limit", 6.0)
        self.prev_accel_cmd = 0.0

        # Terminal stop: within stop_decel_dist of the path end, follow a constant-
        # deceleration profile v = sqrt(2*stop_decel*remaining) so speed hits 0
        # exactly AT the endpoint (instead of the lfd fallback stopping ~lfd short).
        # Kept short (stop_decel_dist) since laptime matters.
        self.stop_decel_dist = rospy.get_param("~stop_decel_dist", 3.0)  # m before goal to brake
        self.stop_decel = rospy.get_param("~stop_decel", 3.2)            # m/s^2 profile decel
        self.stop_brake_gain = rospy.get_param("~stop_brake_gain", 1.0)  # brake per m/s over profile
        self.stop_tol = rospy.get_param("~stop_tol", 0.1)               # m; within this = arrived
        # GPS position reporting delay (s): the ego position lags by this, so the terminal
        # stop dead-reckons the position forward (current_vel*pos_delay along heading) so
        # the latch fires at the TRUE position instead of overshooting. ~ the measured
        # GPS speed lag before IMU fusion (~0.28 s).
        self.pos_delay = rospy.get_param("~pos_delay", 0.0)            # s
        # Arrived-and-stopped latch: if within stop_arrive_dist (the 0.3 m spec) of the
        # goal AND essentially stopped, latch brake=1 + Parking even if outside stop_tol.
        self.stop_arrive_dist = rospy.get_param("~stop_arrive_dist", 0.3)  # m
        self.stop_zero_speed = rospy.get_param("~stop_zero_speed", 0.15)   # m/s (~0.5 kph = stopped)
        # Near the goal, latch a full stop as soon as the car is essentially
        # stopped (speed below this, m/s), even if it halts just outside stop_tol.
        # Otherwise the profile coasts (brake=0) below the goal and the car creeps.
        self.stop_hold_speed = rospy.get_param("~stop_hold_speed", 0.3)  # m/s (~1 kph)
        self.stop_hold_dist = rospy.get_param("~stop_hold_dist", 0.6)    # m; low-speed latch only within this
        self.stopped = False   # latched once we arrive at the goal -> hold brake=1
        # On arrival, optionally shift MORAI to Parking (P) via the event service so the
        # car is mechanically held (brake=1 is also kept as a fallback). Gear enum is
        # MORAI's EventInfo: P=1 R=2 N=3 D=4; option is the change bitmask (2 = gear).
        self.park_at_goal = bool(rospy.get_param("~park_at_goal", True))
        self.park_gear = int(rospy.get_param("~park_gear", 1))        # 1 = Parking
        self.park_option = int(rospy.get_param("~park_option", 2))    # 2 = gear bit
        self.parked = False
        self._event_srv = None
        if self.park_at_goal and _HAVE_EVENT_CMD:
            self._event_srv = rospy.ServiceProxy("/Service_MoraiEventCmd", MoraiEventCmdSrv)
        elif self.park_at_goal:
            rospy.logwarn("park_at_goal set but morai_msgs EventCmd unavailable -> disabled")

        self.pid_controller = PIDControl()
        self.last_time = rospy.Time.now()
        self.last_gain_refresh = rospy.Time.now()

        rate = rospy.Rate(50)   # publish /current_speed & control at 50Hz (IMU rate)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - self.last_time).to_sec()
            self.last_time = now

            # Refresh tunable params live (~2 Hz) so gains can be swept via
            # rosparam set without restarting the node.
            if (now - self.last_gain_refresh).to_sec() > 0.5:
                self.pid_controller.update_gains()
                self.target_speed = rospy.get_param("~target_speed", self.target_speed)
                self.min_curve_speed_kph = rospy.get_param("~min_curve_speed_kph", self.min_curve_speed_kph)
                self.target_decel_rate = rospy.get_param("~target_decel_rate", self.target_decel_rate)
                self.target_filter_alpha = rospy.get_param("~target_filter_alpha", self.target_filter_alpha)
                self.brake_rate = rospy.get_param("~brake_rate", self.brake_rate)
                self.overspeed_brake_gain = rospy.get_param("~overspeed_brake_gain", self.overspeed_brake_gain)
                self.overspeed_deadband = rospy.get_param("~overspeed_deadband", self.overspeed_deadband)
                self.overspeed_min_brake = rospy.get_param("~overspeed_min_brake", self.overspeed_min_brake)
                self.overspeed_anticipate = rospy.get_param("~overspeed_anticipate", self.overspeed_anticipate)
                self.overspeed_lookahead = rospy.get_param("~overspeed_lookahead", self.overspeed_lookahead)
                self.overspeed_rate_alpha = rospy.get_param("~overspeed_rate_alpha", self.overspeed_rate_alpha)
                self.dh_pitch_tau = rospy.get_param("~dh_pitch_tau", self.dh_pitch_tau)
                self.dh_pitch_sign = rospy.get_param("~dh_pitch_sign", self.dh_pitch_sign)
                self.dh_arm_pitch = rospy.get_param("~dh_arm_pitch", self.dh_arm_pitch)
                self.dh_fire_pitch = rospy.get_param("~dh_fire_pitch", self.dh_fire_pitch)
                self.dh_hold_time = rospy.get_param("~dh_hold_time", self.dh_hold_time)
                self.dh_pitch_dead = rospy.get_param("~dh_pitch_dead", self.dh_pitch_dead)
                self.dh_cap_gain = rospy.get_param("~dh_cap_gain", self.dh_cap_gain)
                self.dh_cap_max = rospy.get_param("~dh_cap_max", self.dh_cap_max)
                self.dh_apply_rate = rospy.get_param("~dh_apply_rate", self.dh_apply_rate)
                self.dh_recover = rospy.get_param("~dh_recover", self.dh_recover)
                self.speed_ff_gain = rospy.get_param("~speed_ff_gain", self.speed_ff_gain)
                self.steer_ff_gain = rospy.get_param("~steer_ff_gain", self.steer_ff_gain)
                self.accel_lookahead = rospy.get_param("~accel_lookahead", self.accel_lookahead)
                self.decel_accel_lookahead = rospy.get_param("~decel_accel_lookahead", self.decel_accel_lookahead)
                self.throttle_release_margin = rospy.get_param("~throttle_release_margin", self.throttle_release_margin)
                # terminal-stop params (live-tunable so the goal stop can be dialed in
                # without a restart each lap)
                self.stop_tol = rospy.get_param("~stop_tol", self.stop_tol)
                self.pos_delay = rospy.get_param("~pos_delay", self.pos_delay)
                self.stop_decel_dist = rospy.get_param("~stop_decel_dist", self.stop_decel_dist)
                self.stop_decel = rospy.get_param("~stop_decel", self.stop_decel)
                self.stop_arrive_dist = rospy.get_param("~stop_arrive_dist", self.stop_arrive_dist)
                self.stop_zero_speed = rospy.get_param("~stop_zero_speed", self.stop_zero_speed)
                self.catchup_gain = rospy.get_param("~catchup_gain", self.catchup_gain)
                self.catchup_deadband = rospy.get_param("~catchup_deadband", self.catchup_deadband)
                self.catchup_max = rospy.get_param("~catchup_max", self.catchup_max)
                self.accel_jerk_limit = rospy.get_param("~accel_jerk_limit", self.accel_jerk_limit)
                self.accel_release_jerk_limit = rospy.get_param("~accel_release_jerk_limit", self.accel_release_jerk_limit)
                self.last_gain_refresh = now

            if self.traffic_stop:
                # Red light on a straight section: the traffic-light node owns the
                # control topic and commands the stop. We only brake (no accel) so
                # we never push the car forward against it.
                self.brake_for_traffic_light()
            elif self.is_path and self.is_odom:
                self.control(dt)
            else:
                self.publish_stop_if_needed()
                self.print_waiting_status()

            rate.sleep()

    def control(self, dt):
        current_vel_kph = self.current_vel * 3.6  # self.current_vel is m/s from /odom twist

        # Terminal approach: once the path end (goal) is within stop_decel_dist,
        # take over and decelerate to a full stop AT the endpoint.
        if self.path.poses:
            goal = self.path.poses[-1].pose.position
            # Delay-compensate the ego position for the goal distance: the GPS position
            # lags ~pos_delay s, so the car is really current_vel*pos_delay further along
            # its heading. Without this the stop LATCH fires late -> overshoot. The term
            # is small at the low terminal speed and vanishes at rest (0.3 m spec).
            cx = self.current_position.x + self.current_vel * cos(self.vehicle_yaw) * self.pos_delay
            cy = self.current_position.y + self.current_vel * sin(self.vehicle_yaw) * self.pos_delay
            dist_to_goal = sqrt((goal.x - cx) ** 2 + (goal.y - cy) ** 2)
            if dist_to_goal <= self.stop_decel_dist:
                self.terminal_stop(goal, dist_to_goal, dt)
                return
            # Goal is far again (e.g. a new/longer path) -> clear the stop latch.
            self.stopped = False

        lfd = self.calc_lfd(current_vel_kph)
        forward_point, local_forward_point = self.find_forward_point(lfd)

        if forward_point is None:
            self.ctrl_cmd_msg.steering = 0.0
            # Immediate full brake, bypassing the brake slew-rate limit so we stop
            # right away when the path is lost (no forward point to track).
            self.ctrl_cmd_msg.accel = 0.0
            self.ctrl_cmd_msg.brake = 1.0
            self.prev_brake_cmd = 1.0
            self.prev_accel_cmd = 0.0
            self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
            rospy.logwarn_throttle(1, "No forward point found on local_path")
            return

        self.forward_point = forward_point
        theta = atan2(local_forward_point[1], local_forward_point[0])
        self.ctrl_cmd_msg.steering = atan2(2.0 * self.vehicle_length * sin(theta), lfd)

        curvature = self.calc_path_curvature(self.path)
        raw_target_vel = min(self.curvature_to_speed(curvature), self.target_speed)

        # Low-pass the target to kill curvature-estimate oscillation (jumpy target
        # -> brake/accel surge in curves). Smooth in BOTH directions so the car
        # decelerates once into a curve and holds, and re-accelerates gradually.
        self.filtered_target_vel += self.target_filter_alpha * (raw_target_vel - self.filtered_target_vel)
        target_vel = self.filtered_target_vel

        # Speed trend (kph/s), lightly filtered. Shared by the predictive accel
        # (throttle release) and the predictive brake below.
        if self.prev_vel_kph is not None:
            rate = (current_vel_kph - self.prev_vel_kph) / max(dt, 1e-3)
            self.vel_rate_filt += self.overspeed_rate_alpha * (rate - self.vel_rate_filt)
        self.prev_vel_kph = current_vel_kph

        # Predictive throttle: feed the PID the speed the car is HEADING for
        # (current + lookahead * trend), not just the current speed. Two horizons:
        #  - BELOW target AND decelerating -> CATCH-UP: decel_accel_lookahead * the
        #    (negative) trend lowers the predicted speed -> MORE throttle, arresting
        #    a drop under target (e.g. cornering drag under the curve target).
        #  - otherwise (accelerating toward, or above target) -> OVERSHOOT EASING:
        #    accel_lookahead * upward trend only, so throttle eases near target and
        #    curve ENTRY still coasts down.
        if current_vel_kph < target_vel and self.vel_rate_filt < 0.0:
            predicted_up = current_vel_kph + self.decel_accel_lookahead * self.vel_rate_filt
        else:
            predicted_up = current_vel_kph + self.accel_lookahead * max(0.0, self.vel_rate_filt)
        # Steering feedforward adds throttle in curves (|steering| large) to beat
        # cornering drag; ~0 on straights. Holds curve speed with no integrator.
        steer_ff = self.steer_ff_gain * abs(self.ctrl_cmd_msg.steering)
        throttle = max(0.0, self.pid_controller.pid(target_vel, predicted_up, dt)
                       + self.speed_ff_gain * target_vel + steer_ff)
        # Release the accel once the predicted speed is within throttle_release_margin
        # of target -> coast the last bit up, no overshoot.
        if predicted_up >= target_vel - self.throttle_release_margin:
            throttle = 0.0

        # Floor CATCH-UP: SEPARATE from the below-target predictive catch-up
        # (decel_accel_lookahead) -- this one engages ONLY when the actual speed drops
        # below the MIN curve speed floor (min_curve_speed_kph), i.e. we've sagged under
        # even the slowest curve target (hard cornering/avoidance drag). Adds throttle
        # proportional to how far under the floor so it climbs back fast.
        # GATED to when the target is AT the floor (the curve law clamped raw_target_vel
        # down to min_curve_speed_kph = the sharpest curves/avoidance). Everywhere else
        # -- straights (raw = max), mild curves, and the standing-start launch -- the
        # gate is off, so it can't super-charge the launch and blow past the cap to 20.
        sag = self.min_curve_speed_kph - current_vel_kph
        at_floor = raw_target_vel <= self.min_curve_speed_kph + 0.1
        if at_floor and sag > self.catchup_deadband:
            throttle += min(self.catchup_max,
                            self.catchup_gain * (sag - self.catchup_deadband))

        # --- DOWNHILL pre-brake via a CREST-triggered grade cap ---
        # gpsimu passes the gravity-referenced IMU attitude into odom, so odom PITCH is
        # the real road grade once smoothed. On THIS track: uphill = -pitch, descent =
        # +pitch, and the "plain" is itself a GENTLE descent (small +). We only want to
        # pre-brake the STEEP descent right after a STEEP CLIMB (a hill crest) -- NOT the
        # gentle plain. So a crest state machine:
        #   ARM  once the smoothed grade climbs past dh_arm_pitch (a negative deg, e.g.
        #        -2.3) -> we're on a real uphill.
        #   FIRE when, already armed, the grade crosses up through dh_fire_pitch (~0) ->
        #        we've topped out and started descending. Hold the cut for dh_hold_time s
        #        (the descent), releasing early if we start climbing again.
        # The gentle plain never reaches dh_arm_pitch, so it never fires -> no laptime
        # cost. While held, cut = dh_cap_gain KPH per deg of descent past dh_pitch_dead,
        # capped at dh_cap_max, SLEWED (down dh_apply_rate / up dh_recover) so no chatter.
        grade = degrees(self.odom_pitch) * self.dh_pitch_sign   # uphill -, descent +
        p_ema = min(1.0, dt / self.dh_pitch_tau) if dt > 0 else 0.0
        self.dh_pitch_lp += p_ema * (grade - self.dh_pitch_lp)
        g = self.dh_pitch_lp
        if g <= self.dh_arm_pitch:                 # climbed a real hill -> arm
            self.dh_armed = True
        if self.dh_armed and g >= self.dh_fire_pitch:   # crested into the descent -> fire
            self.dh_armed = False
            self.dh_hold_timer = self.dh_hold_time
        if self.dh_hold_timer > 0.0:
            self.dh_hold_timer = max(0.0, self.dh_hold_timer - dt)
            if g <= self.dh_arm_pitch:             # climbing again -> descent over
                self.dh_hold_timer = 0.0
        active = self.dh_hold_timer > 0.0
        if active:
            target_red = min(self.dh_cap_max,
                             self.dh_cap_gain * max(0.0, g - self.dh_pitch_dead))
        else:
            target_red = 0.0
        if target_red > self.dh_reduction:
            self.dh_reduction = min(target_red, self.dh_reduction + self.dh_apply_rate * dt)
        else:
            self.dh_reduction = max(target_red, self.dh_reduction - self.dh_recover * dt)
        os_cap = self.target_speed - self.dh_reduction

        # --- DOWNHILL detection LOG (crest state machine on smoothed grade) ---
        dh_active = self.dh_reduction > 0.1
        if dh_active and not self.dh_prev_active:
            rospy.logwarn("=== DOWNHILL ON (crest) >> grade=%+.2fdeg cap %.1f->%.1f cur=%.0f hold=%.1fs ==="
                          % (g, self.target_speed, os_cap, current_vel_kph, self.dh_hold_timer))
        elif not dh_active and self.dh_prev_active:
            rospy.logwarn("=== DOWNHILL OFF <<< cap back to %.1f ===" % self.target_speed)
        elif dh_active:
            rospy.loginfo_throttle(0.5, "  [downhill] grade=%+.2f cap=%.1f cur=%.0f hold=%.1fs"
                                   % (g, os_cap, current_vel_kph, self.dh_hold_timer))
        else:
            # grade (uphill -, descent +) and whether we're ARMED (past a real climb)
            rospy.loginfo_throttle(0.5, "  grade=%+.2fdeg armed=%d (arm<=%.1f fire>=%.1f) - normal"
                                   % (g, int(self.dh_armed), self.dh_arm_pitch, self.dh_fire_pitch))
        self.dh_prev_active = dh_active

        # Overspeed BRAKE, coast-down aware, measured against os_cap (grade-adaptive
        # top-speed limit), not the curvature-lowered curve target.
        overspeed_brake = 0.0
        if (self.overspeed_anticipate
                and current_vel_kph > os_cap - self.throttle_release_margin
                and self.vel_rate_filt > 0.0):
            # past the cut point but STILL accelerating = external push (downhill) ->
            # anticipate the overspeed and brake early.
            os_trend = self.vel_rate_filt
        else:
            # coast-down ease-off: only the (negative) trend, so the brake releases
            # early as the car decelerates back under the cap.
            os_trend = min(0.0, self.vel_rate_filt)
        predicted_down = current_vel_kph + self.overspeed_lookahead * os_trend
        excess = predicted_down - os_cap
        if excess > self.overspeed_deadband:
            # floor + proportional: the floor clears MORAI's brake dead-zone so the
            # command bites immediately instead of ramping up from ~0.
            overspeed_brake = self.overspeed_min_brake + self.overspeed_brake_gain * (excess - self.overspeed_deadband)

        if overspeed_brake > 0.0:
            self.ctrl_cmd_msg.accel = 0.0
            self.ctrl_cmd_msg.brake = min(1.0, overspeed_brake)
            self.prev_brake_cmd = self.ctrl_cmd_msg.brake
            # decay the accel memory so throttle ramps up smoothly when braking ends
            self.prev_accel_cmd = max(0.0, self.prev_accel_cmd - self.accel_release_jerk_limit * max(dt, 1e-3))
        else:
            self.apply_output(throttle, dt)

        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
        self.target_speed_pub.publish(Float32(data=target_vel))
        self.current_speed_pub.publish(Float32(data=round(current_vel_kph)))  # debug topic: round to int for readability

        # (per-cycle control debug log removed to keep the console clean for the
        #  DOWNHILL logs; re-enable if needed.)

    def engage_parking(self):
        # Shift MORAI to Parking via the event service (one-shot, non-fatal on failure).
        if self._event_srv is None:
            return
        try:
            rospy.wait_for_service("/Service_MoraiEventCmd", timeout=0.5)
            ev = EventInfo()
            ev.option = self.park_option   # bitmask: 2 = change gear only
            ev.gear = self.park_gear       # 1 = P
            self._event_srv(ev)
            rospy.logwarn("=== PARKED at goal: gear->P (gear=%d option=%d) ===",
                          self.park_gear, self.park_option)
        except Exception as exc:
            rospy.logwarn("park: gear change failed (%s) -- holding brake=1 instead", exc)

    def terminal_stop(self, goal, dist_to_goal, dt):
        # Steer toward the actual endpoint (not the lfd point) and brake along a
        # constant-deceleration profile so the car stops exactly at the goal.
        dx = goal.x - self.current_position.x
        dy = goal.y - self.current_position.y
        local_x = cos(self.vehicle_yaw) * dx + sin(self.vehicle_yaw) * dy
        local_y = -sin(self.vehicle_yaw) * dx + cos(self.vehicle_yaw) * dy
        ld = max(dist_to_goal, 1.0)   # floor so a near-zero distance can't blow up steering
        theta = atan2(local_y, local_x)
        self.ctrl_cmd_msg.steering = atan2(2.0 * self.vehicle_length * sin(theta), ld)

        remaining = dist_to_goal - self.stop_tol
        # LATCH a full stop (brake=1 hold + shift to P) when EITHER:
        #  (a) within stop_tol of the goal (the precise arrival), OR
        #  (b) ARRIVED-AND-STOPPED: the car is within stop_arrive_dist (0.3 m spec) of the
        #      goal AND essentially stopped (speed < stop_zero_speed). Needed because the
        #      car often halts just SHORT of stop_tol (~stop_tol+0.05); without (b) it
        #      would never latch -> no brake hold, no Parking, and it could creep/roll.
        if (dist_to_goal <= self.stop_tol
                or (dist_to_goal <= self.stop_arrive_dist
                    and self.current_vel < self.stop_zero_speed)):
            self.stopped = True
        if self.stopped:
            self.ctrl_cmd_msg.steering = 0.0
            self.ctrl_cmd_msg.accel = 0.0
            self.ctrl_cmd_msg.brake = 1.0
            self.prev_brake_cmd = 1.0
            self.prev_accel_cmd = 0.0
            self.pid_controller.reset()
            if self.park_at_goal and not self.parked:
                self.engage_parking()
                self.parked = True   # one-shot; brake=1 still held regardless
        else:
            # v_allow = sqrt(2*a*remaining) reaches 0 exactly at the goal. Brake
            # (bypassing the slew limit) when above it; coast when below.
            v_allow = sqrt(2.0 * self.stop_decel * remaining)   # m/s
            self.ctrl_cmd_msg.accel = 0.0
            if self.current_vel > v_allow:
                brake = self.stop_brake_gain * (self.current_vel - v_allow)
                self.ctrl_cmd_msg.brake = max(0.0, min(1.0, brake))
            else:
                self.ctrl_cmd_msg.brake = 0.0
            self.prev_brake_cmd = self.ctrl_cmd_msg.brake

        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
        self.current_speed_pub.publish(Float32(data=round(self.current_vel * 3.6)))  # debug topic: round to int
        rospy.loginfo_throttle(
            0.3, "terminal stop: dist={:.2f}m v={:.1f}kph brake={:.2f} latched={}".format(
                dist_to_goal, self.current_vel * 3.6, self.ctrl_cmd_msg.brake, self.stopped))

    def calc_lfd(self, current_vel_kph):
        # Speed band spans 0 -> max_speed_kph. The ratio is raised to
        # lfd_speed_exp so lfd grows SUB-linearly: it stays short across the
        # low/mid band and only climbs toward max_lfd near max speed.
        # (0 kph -> min_lfd, max_speed_kph -> max_lfd.)
        speed_range = max(self.max_speed_kph, 1e-3)
        speed_ratio = current_vel_kph / speed_range
        speed_ratio = max(0.0, min(1.0, speed_ratio))
        shaped_ratio = speed_ratio ** self.lfd_speed_exp
        lfd = self.min_lfd + (self.max_lfd - self.min_lfd) * shaped_ratio
        return max(self.min_lfd, min(self.max_lfd, lfd))

    def find_forward_point(self, lfd):
        translation = [self.current_position.x, self.current_position.y]
        trans_matrix = np.array(
            [
                [cos(self.vehicle_yaw), -sin(self.vehicle_yaw), translation[0]],
                [sin(self.vehicle_yaw), cos(self.vehicle_yaw), translation[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        det_trans_matrix = np.array(
            [
                [trans_matrix[0][0], trans_matrix[1][0], -(trans_matrix[0][0] * translation[0] + trans_matrix[1][0] * translation[1])],
                [trans_matrix[0][1], trans_matrix[1][1], -(trans_matrix[0][1] * translation[0] + trans_matrix[1][1] * translation[1])],
                [0.0, 0.0, 1.0],
            ]
        )

        for pose in self.path.poses:
            path_point = pose.pose.position
            global_path_point = np.array([path_point.x, path_point.y, 1.0])
            local_path_point = det_trans_matrix.dot(global_path_point)

            if local_path_point[0] > 0.0:
                distance = sqrt(local_path_point[0] ** 2 + local_path_point[1] ** 2)
                if distance >= lfd:
                    return path_point, local_path_point

        return None, None

    def calc_path_curvature(self, path):
        points = []
        for pose in path.poses:
            path_point = pose.pose.position
            dx = path_point.x - self.current_position.x
            dy = path_point.y - self.current_position.y
            local_x = cos(self.vehicle_yaw) * dx + sin(self.vehicle_yaw) * dy
            if local_x <= 0.0:
                continue

            points.append((path_point.x, path_point.y))
            if len(points) >= self.curvature_lookahead:
                break

        if len(points) < 3:
            return 0.0

        curvatures = []
        for p1, p2, p3 in zip(points, points[1:], points[2:]):
            a = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            b = sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2)
            c = sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)
            if a < 1e-3 or b < 1e-3 or c < 1e-3:
                continue

            area2 = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
            curvature = 2.0 * area2 / (a * b * c)
            curvatures.append(curvature)

        if not curvatures:
            return 0.0

        curvatures.sort(reverse=True)
        top_count = max(1, min(5, len(curvatures)))
        return sum(curvatures[:top_count]) / top_count

    def curvature_to_speed(self, curvature):
        if curvature < 1e-4:
            return self.max_speed_kph

        speed_mps = sqrt(self.lateral_acc_limit / curvature)
        speed_kph = speed_mps * 3.6
        return max(self.min_curve_speed_kph, min(self.max_speed_kph, speed_kph))

    def apply_output(self, output, dt):
        # Direct map: positive output -> accel, negative -> brake. The brake
        # command is slew-rate-limited (brake_rate, 1/s) so it can't spike
        # instantly at corner entry, without slowing the target ramp.
        output = max(-1.0, min(1.0, output))

        # --- brake first (slew-limited rise, immediate release) ---
        target_brake = max(0.0, -output)
        max_delta = self.brake_rate * max(dt, 1e-3)
        # 상승(코너 진입 스파이크)만 슬루 제한하고, 해제는 즉시.
        # 그래야 초록불로 정지가 해제될 때 브레이크가 걸려 있지 않아 바로 출발한다.
        if target_brake > self.prev_brake_cmd:
            brake = min(self.prev_brake_cmd + max_delta, target_brake)
        else:
            brake = target_brake
        self.prev_brake_cmd = brake
        self.ctrl_cmd_msg.brake = brake

        # --- accel only when NOT braking (mutual exclusion, brake dominant) ---
        # A small coast deadband keeps tiny positive outputs from blipping the
        # throttle right after/around braking, so accel and brake never appear
        # to fire together near the setpoint.
        accel = max(0.0, output)
        if brake > 0.0 or accel < self.accel_deadband:
            accel = 0.0
        # Jerk limit: slew-rate limit the accel command so acceleration ramps
        # smoothly instead of jumping (bounded jerk). Up (press) and down
        # (release/lift-off) use SEPARATE limits so throttle-off can be softened
        # independently of throttle-on.
        dt_eff = max(dt, 1e-3)
        if accel > self.prev_accel_cmd:
            accel = min(self.prev_accel_cmd + self.accel_jerk_limit * dt_eff, accel)
        else:
            accel = max(self.prev_accel_cmd - self.accel_release_jerk_limit * dt_eff, accel)
        self.prev_accel_cmd = accel
        self.ctrl_cmd_msg.accel = accel

    def brake_for_traffic_light(self):
        # Hold current steering, cut throttle, full brake. Reset PID/brake state so
        # there's no windup or stale slew when normal control resumes on green.
        self.ctrl_cmd_msg.accel = 0.0
        self.ctrl_cmd_msg.brake = 1.0
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
        self.pid_controller.reset()
        self.prev_brake_cmd = 1.0
        self.prev_accel_cmd = 0.0
        self.filtered_target_vel = 0.0
        rospy.loginfo_throttle(1.0, "🔴 traffic-light stop: braking (control handed to traffic node)")

    def publish_stop_if_needed(self):
        if self.is_odom:
            return
        self.ctrl_cmd_msg.steering = 0.0
        self.ctrl_cmd_msg.accel = 0.0
        self.ctrl_cmd_msg.brake = 0.0
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

    def print_waiting_status(self):
        if not self.is_path:
            rospy.loginfo_throttle(1, "[1] can't subscribe '/local_path' topic...")
        if not self.is_odom:
            rospy.loginfo_throttle(1, "[2] can't subscribe '/odom' topic...")

    def traffic_stop_callback(self, msg):
        self.traffic_stop = msg.data

    def path_callback(self, msg):
        self.is_path = True
        self.path = msg

    def odom_callback(self, msg):
        self.is_odom = True
        odom_quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        # gpsimu copies the (gravity-referenced) /imu attitude into odom, so pitch here
        # is the real vehicle tilt = road grade once smoothed: +uphill, -downhill.
        _, self.odom_pitch, self.vehicle_yaw = euler_from_quaternion(odom_quaternion)
        self.current_position.x = msg.pose.pose.position.x
        self.current_position.y = msg.pose.pose.position.y

        # /odom twist carries the forward speed in m/s (ROS convention);
        # control() converts it to kph via * 3.6.
        self.current_vel = msg.twist.twist.linear.x


if __name__ == "__main__":
    try:
        PurePursuitPIDCurvature()
    except rospy.ROSInterruptException:
        pass
