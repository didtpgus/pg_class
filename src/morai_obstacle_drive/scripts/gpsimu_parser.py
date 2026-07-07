#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import rospy
import math
from sensor_msgs.msg import Imu
from morai_msgs.msg import GPSMessage
from nav_msgs.msg import Odometry
from pyproj import Proj


class GPSIMUParser:
    def __init__(self):
        rospy.init_node('GPS_IMU_parser', anonymous=True)
        self.gps_topic = rospy.get_param("~gps_topic", "/gps")
        self.imu_topic = rospy.get_param("~imu_topic", "/imu")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.publish_hz = rospy.get_param("~publish_hz", 50.0)
        self.gps_timeout = rospy.get_param("~gps_timeout", 1.0)
        self.imu_timeout = rospy.get_param("~imu_timeout", 1.0)
        self.utm_zone = rospy.get_param("~utm_zone", 52)
        # --- Speed estimation: IMU/GPS complementary filter ---
        # predict: integrate IMU longitudinal accel every cycle (instant, no lag)
        # correct: nudge toward the GPS position-delta speed on each new fix
        #          (slow but unbiased -> cancels the IMU integration drift).
        # gps_correct_gain: blend factor toward the GPS speed per fix (0..1).
        #   Higher = trusts GPS more (less drift, a little more noise/lag).
        self.gps_correct_gain = float(rospy.get_param("~gps_correct_gain", 0.2))
        # IMU accel with magnitude below this (m/s^2) is treated as 0 so sensor
        # noise / a small gravity leak doesn't integrate into creep at standstill.
        self.imu_accel_deadband = float(rospy.get_param("~imu_accel_deadband", 0.07))
        # Complementary filter: IMU accel integration provides the FAST, low-lag,
        # smooth speed transient; the windowed GPS speed slowly corrects it toward
        # the true absolute (kills IMU bias/drift). Breaks the GPS-only "low lag XOR
        # low noise" tradeoff. Set imu_fusion=false to fall back to pure GPS (EMA).
        self.imu_fusion = bool(rospy.get_param("~imu_fusion", True))
        # Flip if MORAI's body-x accel sign is opposite (forward accel should be +).
        self.imu_accel_sign = float(rospy.get_param("~imu_accel_sign", 1.0))
        # Low-pass on the RAW IMU accel before it is integrated. MORAI's body-x
        # accel is noisy (seen swinging -13..+7 m/s^2), so integrating it raw makes
        # the fused speed change too fast/jittery. Filtering the ACCEL smooths the
        # speed's rate-of-change with far less lag than filtering the speed itself.
        # 1 = raw, lower = smoother. Live-tunable.
        self.imu_accel_alpha = float(rospy.get_param("~imu_accel_alpha", 0.7))
        self.imu_accel_filt = 0.0
        # Estimated IMU longitudinal-accel bias (m/s^2), adapted slowly from the
        # persistent GPS-vs-IMU speed error and subtracted in the predict step so the
        # integration doesn't drift. imu_bias_gain = adaptation speed (per GPS fix);
        # higher = removes drift faster but can fight real sustained accel.
        self.imu_accel_bias = float(rospy.get_param("~imu_bias_init", 0.06))  # seed ~measured bias
        self.imu_bias_gain = float(rospy.get_param("~imu_bias_gain", 0.02))  # (legacy; observer uses tau_b)
        self.max_reasonable_speed = float(rospy.get_param("~max_reasonable_speed", 7.0))
        # Spike rejection: cap the per-GPS-fix speed change to a physically
        # plausible max_accel * dt. Kills 6->30 kph measurement jumps (bad GPS
        # fixes / dt jitter) without adding filter lag.
        self.max_accel = float(rospy.get_param("~max_accel", 5.0))  # m/s^2
        # Calibration scale for the published speed. The GPS-derived speed read
        # ~90% of the MORAI speedometer (residual timing scale), so multiply by
        # ~1.11 to match. Tune if a paired reading shows a different ratio.
        self.speed_scale = float(rospy.get_param("~speed_scale", 1.5))
        # Light low-pass on the raw GPS-delta speed. 1.0 = raw (no filter, snappy
        # but jittery); lower = smoother but a little lag. Sweet spot between the
        # old heavy filter (~3 s lag) and raw. Live-tunable via rosparam.
        self.speed_filter_alpha = float(rospy.get_param("~speed_filter_alpha", 0.5))
        # Constant SIM-time GPS period used as the velocity dt, instead of the GPS
        # header.stamp difference. MORAI stamps GPS with WALL-clock time, so a
        # stamp-based dt = wall dt -> speed = true_speed * real_time_factor, which
        # varies per run (RTF changes with CPU/GPU load). GPS is generated at a
        # FIXED sim rate, so dividing the position delta by that fixed period gives
        # an RTF-INDEPENDENT speed (any constant offset is absorbed by speed_scale).
        # Set to 1 / (sim GPS rate). ~20 Hz observed at RTF~=1 -> 0.05 s.
        self.gps_dt = float(rospy.get_param("~gps_dt", 0.05))  # s
        self.last_param_refresh = rospy.Time(0)

        # Large queue so bursty GPS delivery isn't dropped at the transport layer:
        # every fix reaches the callback (in order), so the velocity window sees
        # CONSECUTIVE fixes (constant sim-dt is then exact). queue_size=1 was
        # dropping ~40% of fixes under load -> window over-counted motion -> speed
        # spiked to ~26 kph. tcp_nodelay lowers delivery latency/bunching.
        self.gps_sub = rospy.Subscriber(self.gps_topic, GPSMessage, self.navsat_callback,
                                        queue_size=100, tcp_nodelay=True)
        self.imu_sub = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=20)
        self.odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=1)

        # initialize
        self.x, self.y = None, None
        self.lat = None
        self.lon = None
        self.e_o = 0.0
        self.n_o = 0.0
        self.last_gps_stamp = None
        self.last_imu_stamp = None
        self.has_gps = False
        self.has_imu = False

        # previous pose/time for velocity estimation.
        # Velocity is updated only when a NEW GPS fix arrives (tracked by
        # vel_last_gps_stamp); the publish loop runs faster than GPS, so
        # recomputing every cycle would inject zeros (dx=0) and decay the
        # speed toward 0. Between fixes we simply hold the last value.
        # ================= Lightweight GPS/IMU speed observer (alpha-beta) =========
        # State is in SI (m/s) end-to-end; converted to kph only at consumers. A
        # 2-state constant-gain observer (speed v = output_speed, accel bias): IMU
        # accel integration gives the fast/low-lag transient BETWEEN fixes, the GPS
        # speed slowly corrects the absolute + the bias. NOT an EKF (no covariance).
        #
        # The hard part is GPS TIME: stamps are wall-clock, delivery is bursty (1 ms)
        # + stalls (139 ms) + drops (~half under load); no sim clock. So we recover
        # the SIM-time each interval spans and use it for BOTH branches (shared scale
        # -> one calibration fixes both). Key quantity T1 = the single-period wall
        # interval (~ sim_period / RTF); it lets us count how many sim-periods an
        # interval spans (n) and convert IMU wall-time to sim-time.
        self.output_speed = 0.0          # v estimate (m/s), published
        self.gps_prev_fix = None         # (x, y, z, wall_stamp_sec) of the last accepted fix
        self.warm_samples = 0            # bootstrap counter
        self.gps_data_stamp = None
        self.alt = 0.0                   # latest GPS altitude (m), for 3D distance

        # --- GPS-only speed (no IMU integration), RTF-independent count-based ---
        self.gps_win = []                # recent per-fix distances (m) for the averaged speed
        self.n_win = int(rospy.get_param("~n_win", 5))                   # fixes averaged (smoothing)
        # sim seconds per GPS fix (1/sim_gps_rate). THE calibration: RTF-independent, so
        # once set vs Ego it holds across load. speed = (Sum dist / N) / sim_period.
        self.sim_period = float(rospy.get_param("~sim_period", 0.0568))  # s (~1/17.6 Hz)
        self.use_altitude = bool(rospy.get_param("~use_altitude", True))  # 3D dist on slopes
        self.dup_dist = float(rospy.get_param("~dup_dist", 0.01))        # <this m between fixes = dup -> skip
        # LEAD (delay) compensation: output = v_meas + lead_tau * d(v_meas)/dt. 0 = raw
        # GPS (no lead). Raise to cancel the GPS speed lag (tune vs Ego velocity in dev).
        self.lead_tau = float(rospy.get_param("~lead_tau", 0.0))         # s of lead
        self.rate_alpha = float(rospy.get_param("~rate_alpha", 0.3))     # EMA on the speed rate
        # IMU predict (fills the GPS reporting delay -> low lag). imu_fusion toggles it.
        self.imu_since_fix = 0           # IMU msgs counted in the current GPS interval
        self.sim_dt_per_imu = None       # SIM seconds per IMU msg (= sim_period / count)
        self.gps_correct_gain = float(rospy.get_param("~gps_correct_gain", 0.2))  # GPS pull per fix
        self.bias_gain = float(rospy.get_param("~bias_gain", 0.02))      # accel-bias adapt per fix
        # final low-pass on the output speed (1.0 = none, lower = smoother/laggier).
        self.speed_filter_alpha = float(rospy.get_param("~speed_filter_alpha", 0.5))
        self.prev_vmeas = 0.0
        self.prev_vmeas_t = None
        self.vmeas_rate = 0.0

        self.proj_UTM = Proj(proj='utm', zone=self.utm_zone, ellps='WGS84', preserve_units=False)

        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = 'map'
        self.odom_msg.child_frame_id = 'base_link'

        rospy.loginfo(
            "GPS/IMU parser ready: gps=%s imu=%s odom=%s publish_hz=%.1f timeout gps=%.1fs imu=%.1fs",
            self.gps_topic,
            self.imu_topic,
            self.odom_topic,
            self.publish_hz,
            self.gps_timeout,
            self.imu_timeout,
        )

        rate = rospy.Rate(self.publish_hz)
        while not rospy.is_shutdown():
            # Re-read the speed filter live (~2 Hz) so it can be tuned via rosparam.
            now = rospy.Time.now()
            if (now - self.last_param_refresh).to_sec() > 0.5:
                # GPS-only speed params, live-tunable via rosparam.
                self.speed_scale = float(rospy.get_param("~speed_scale", self.speed_scale))
                self.max_reasonable_speed = float(rospy.get_param("~max_reasonable_speed", self.max_reasonable_speed))
                self.n_win = int(rospy.get_param("~n_win", self.n_win))
                self.sim_period = float(rospy.get_param("~sim_period", self.sim_period))
                self.use_altitude = bool(rospy.get_param("~use_altitude", self.use_altitude))
                self.dup_dist = float(rospy.get_param("~dup_dist", self.dup_dist))
                self.lead_tau = float(rospy.get_param("~lead_tau", self.lead_tau))
                self.rate_alpha = float(rospy.get_param("~rate_alpha", self.rate_alpha))
                self.speed_filter_alpha = float(rospy.get_param("~speed_filter_alpha", self.speed_filter_alpha))
                self.imu_fusion = bool(rospy.get_param("~imu_fusion", self.imu_fusion))
                self.gps_correct_gain = float(rospy.get_param("~gps_correct_gain", self.gps_correct_gain))
                self.bias_gain = float(rospy.get_param("~bias_gain", self.bias_gain))
                self.imu_accel_alpha = float(rospy.get_param("~imu_accel_alpha", self.imu_accel_alpha))
                self.imu_accel_deadband = float(rospy.get_param("~imu_accel_deadband", self.imu_accel_deadband))
                self.imu_accel_sign = float(rospy.get_param("~imu_accel_sign", self.imu_accel_sign))
                self.last_param_refresh = now
            if self.ready_to_publish():
                self.convertLL2UTM()
                self.odom_pub.publish(self.odom_msg)
                # (per-publish /odom log removed to keep the console clean)
            else:
                self.log_waiting_status()
            rate.sleep()

    def navsat_callback(self, gps_msg):
        self.lat = gps_msg.latitude
        self.lon = gps_msg.longitude
        self.alt = gps_msg.altitude          # for 3D distance on slopes
        self.e_o = gps_msg.eastOffset
        self.n_o = gps_msg.northOffset
        self.last_gps_stamp = rospy.Time.now()   # wall clock, for staleness checks
        stamp = gps_msg.header.stamp
        self.gps_data_stamp = stamp if not stamp.is_zero() else self.last_gps_stamp
        self.has_gps = True

        # Compute this fix's UTM position HERE (once per received fix) and feed the
        # velocity window, so EVERY fix is captured consecutively -- even in a burst
        # where two fixes land between 50 Hz loop ticks (the old path converted only
        # the latest in the loop and silently skipped the intermediate one, which
        # looked like a dropped fix to the window). The callback is serialized, so
        # gps_hist / output_speed are touched by a single thread here.
        try:
            xy = self.proj_UTM(self.lon, self.lat)
        except Exception:
            return
        if xy[0] == float("inf") or xy[1] == float("inf"):
            return
        vx = xy[0] - self.e_o
        vy = xy[1] - self.n_o
        self._push_gps_fix(vx, vy, self.alt, self.gps_data_stamp)

    def _push_gps_fix(self, x, y, z, stamp):
        # ===== GPS-ONLY speed (no IMU): windowed 3D distance / wall_dt, lead-compensated.
        # RTF~1 here, and the GPS header stamp is wall-clock, so wall_dt == elapsed sim
        # time -> dist/wall_dt is the true speed for every fix regardless of the bursty
        # ~17 Hz delivery (a burst has small dist AND small wall_dt; a stall both large).
        # 3D distance (incl. altitude) so it isn't under-read on slopes (= velocity.x).
        t = stamp.to_sec()
        if self.gps_prev_fix is None:
            self.gps_prev_fix = (x, y, z, t)
            return
        px, py, pz, pt = self.gps_prev_fix
        wall_dt = t - pt
        if wall_dt <= 1e-4:            # same tick / non-monotonic -> ignore
            return
        dz = (z - pz) if self.use_altitude else 0.0
        dist = ((x - px) ** 2 + (y - py) ** 2 + dz ** 2) ** 0.5

        # duplicate / zero-displacement guard (a re-published stale pos != 0 speed)
        if dist < self.dup_dist:
            self.gps_prev_fix = (x, y, z, t)
            return
        self.warm_samples = min(self.warm_samples + 1, 1000)

        # COUNT-BASED speed (RTF-INDEPENDENT). Each accepted fix spans ~one sim GPS
        # period, so N_fixes * sim_period == elapsed SIM time no matter how the ~17 Hz
        # delivery is wall-jittered OR how slow the sim runs. dist/wall_dt would instead
        # read true*RTF, and RTF here is ~0.7 and LOAD-dependent -> a fixed scale can't
        # track it. Averaging N fixes also smooths. Calibrate sim_period vs Ego velocity
        # once (dev); then it holds across load in competition (no Ego needed).
        self.gps_win.append(dist)
        if len(self.gps_win) > self.n_win:
            self.gps_win.pop(0)
        n = len(self.gps_win)
        v_meas = (sum(self.gps_win) / (n * self.sim_period)) * self.speed_scale

        if 0.0 <= v_meas <= 1.5 * self.max_reasonable_speed:
            if self.warm_samples < 3:
                self.output_speed = v_meas                   # warmup: seed straight from GPS
            elif self.imu_fusion:
                # FUSION: GPS is the slow, absolute CORRECT toward the accurate (count-
                # based, RTF-independent) speed; the IMU predict (imu_callback) supplies
                # the fast low-lag motion BETWEEN fixes -> cancels the ~0.3 s GPS
                # reporting delay so the overspeed brake reacts in time. A light bias
                # state absorbs residual IMU drift (safe now: GPS target is correct).
                innov = v_meas - self.output_speed
                self.output_speed = max(0.0, min(self.max_reasonable_speed,
                                                 self.output_speed + self.gps_correct_gain * innov))
                self.imu_accel_bias = max(-1.0, min(1.0, self.imu_accel_bias - self.bias_gain * innov))
            else:
                # GPS-ONLY: smoothed GPS (optional lead) + final EMA.
                raw = v_meas
                if self.prev_vmeas_t is not None:
                    dtv = t - self.prev_vmeas_t
                    if dtv > 1e-3:
                        rate = (v_meas - self.prev_vmeas) / dtv
                        self.vmeas_rate += self.rate_alpha * (rate - self.vmeas_rate)
                    raw = v_meas + self.lead_tau * self.vmeas_rate
                self.output_speed += self.speed_filter_alpha * (raw - self.output_speed)
                self.output_speed = max(0.0, min(self.max_reasonable_speed, self.output_speed))
            self.prev_vmeas = v_meas
            self.prev_vmeas_t = t
        # distribute this interval's SIM time (sim_period) across its IMU msgs, so the IMU
        # predict integrates in SIM seconds (RTF-consistent), not wall seconds.
        if self.imu_since_fix > 0:
            self.sim_dt_per_imu = self.sim_period / self.imu_since_fix
        self.imu_since_fix = 0
        self.gps_prev_fix = (x, y, z, t)

    def convertLL2UTM(self):    
        xy_zone = self.proj_UTM(self.lon, self.lat)


        if self.lon == 0 and self.lat == 0:
            self.x = 0.0
            self.y = 0.0
        else:
            self.x = xy_zone[0] - self.e_o
            self.y = xy_zone[1] - self.n_o

        now = rospy.get_rostime()
        self.update_velocity(self.x, self.y)

        self.odom_msg.header.stamp = now
        self.odom_msg.pose.pose.position.x = self.x
        self.odom_msg.pose.pose.position.y = self.y
        self.odom_msg.pose.pose.position.z = 0.

    def update_velocity(self, x, y):
        # Speed is computed per-GPS-fix in navsat_callback/_push_gps_fix (windowed
        # median), so every fix is captured even in a burst. Here (50 Hz loop) we
        # just publish the latest value into the odom twist.
        self.odom_msg.twist.twist.linear.x = self.output_speed
        self.odom_msg.twist.twist.linear.y = 0.0
        self.odom_msg.twist.twist.linear.z = 0.0

    def imu_callback(self, data):
        if data.orientation.w == 0:
            self.odom_msg.pose.pose.orientation.x = 0.0
            self.odom_msg.pose.pose.orientation.y = 0.0
            self.odom_msg.pose.pose.orientation.z = 0.0
            self.odom_msg.pose.pose.orientation.w = 1.0
        else:
            self.odom_msg.pose.pose.orientation.x = data.orientation.x
            self.odom_msg.pose.pose.orientation.y = data.orientation.y
            self.odom_msg.pose.pose.orientation.z = data.orientation.z
            self.odom_msg.pose.pose.orientation.w = data.orientation.w

        # PREDICT (only if imu_fusion): dead-reckon the speed with the IMU forward accel
        # between GPS fixes -> fast, low-lag (fills the ~0.3 s GPS reporting delay). Runs
        # at the IMU rate; integrates over sim_dt_per_imu (SIM seconds, from the count-
        # based interval) so it's RTF-consistent and shares the count-based calibration.
        # (Orientation is set above and is used regardless of imu_fusion.)
        self.imu_accel_filt += self.imu_accel_alpha * (
            data.linear_acceleration.x * self.imu_accel_sign - self.imu_accel_filt)
        self.imu_since_fix += 1
        if self.imu_fusion and self.warm_samples >= 3 and self.sim_dt_per_imu is not None:
            a = self.imu_accel_filt - self.imu_accel_bias
            if abs(a) < self.imu_accel_deadband:
                a = 0.0
            self.output_speed = max(0.0, min(self.max_reasonable_speed,
                                             self.output_speed + a * self.sim_dt_per_imu))
        self.last_imu_stamp = rospy.Time.now()   # wall clock, for staleness checks
        self.has_imu = True

    def age(self, stamp):
        if stamp is None:
            return float("inf")
        return (rospy.Time.now() - stamp).to_sec()

    def ready_to_publish(self):
        return (
            self.has_gps
            and self.has_imu
            and self.age(self.last_gps_stamp) <= self.gps_timeout
            and self.age(self.last_imu_stamp) <= self.imu_timeout
        )

    def log_waiting_status(self):
        if not self.has_gps:
            rospy.logwarn_throttle(1.0, "Waiting for GPS topic: %s", self.gps_topic)
        elif self.age(self.last_gps_stamp) > self.gps_timeout:
            rospy.logwarn_throttle(
                1.0,
                "GPS data stale: age=%.2fs timeout=%.2fs",
                self.age(self.last_gps_stamp),
                self.gps_timeout,
            )

        if not self.has_imu:
            rospy.logwarn_throttle(1.0, "Waiting for IMU topic: %s", self.imu_topic)
        elif self.age(self.last_imu_stamp) > self.imu_timeout:
            rospy.logwarn_throttle(
                1.0,
                "IMU data stale: age=%.2fs timeout=%.2fs",
                self.age(self.last_imu_stamp),
                self.imu_timeout,
            )

if __name__ == '__main__':
    try:
        GPS_IMU_parser = GPSIMUParser()
    except rospy.ROSInterruptException:
        pass
