#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight Frenet local path planner (single-segment quintic fan).

A trimmed port of the local_path_selector "optimal frenet sampler":
  - Reference : global path CSV (x y). A window ahead of the ego is resampled
                uniformly. Curvature (kappa) is treated as 0 (not needed / lighter).
  - Candidates: N_D lateral end-offsets. For each, a quintic d(s) is built from the
                ego Frenet state (d0, d0', 0) to (d1, 0, 0) over [0, SF].
  - Collision : each candidate is checked with SAT (ego OBB swept along the path vs
                each obstacle OBB). Colliding / over-curved candidates are invalid.
  - Selection : minimum-cost valid candidate (deviation + curvature + optional side
                bias). Published as nav_msgs/Path on /local_path.

Obstacles come from /lidar_clusters (MarkerArray, sphere centroids) -> (x, y, radius)
in map, so NO morai ground-truth object list is used. numpy only (no GPU).
"""

import math
import os

import numpy as np
import rospy
import rospkg
import tf
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray


# ----- quintic helpers (single segment, t in [0, 1]) -----
_A = np.array(
    [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0],
     [1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5], [0, 0, 2, 6, 12, 20]],
    dtype=np.float64,
)
_AINV = np.linalg.inv(_A).astype(np.float32)


def quintic_coeffs(d0, d0p, d0pp, d1, d1p, d1pp):
    """d0/d0p/d0pp are scalars, d1/d1p/d1pp are (C,) arrays. Returns (C, 6)."""
    C = d1.shape[0]
    b = np.stack((
        np.full(C, d0, np.float32), np.full(C, d0p, np.float32), np.full(C, d0pp, np.float32),
        d1.astype(np.float32), d1p.astype(np.float32), d1pp.astype(np.float32),
    ), axis=0)
    return (_AINV @ b).T  # (C, 6)


def horner(coef, t):
    """coef (C,6), t (N,) -> (C, N)."""
    r = np.broadcast_to(coef[:, 5:6], (coef.shape[0], t.size)).copy()
    for i in range(4, -1, -1):
        r = r * t + coef[:, i:i + 1]
    return r


def dpp_eval(coef, t, dt2):
    a2, a3, a4, a5 = coef[:, 2:3], coef[:, 3:4], coef[:, 4:5], coef[:, 5:6]
    return dt2 * (2 * a2 + 6 * a3 * t + 12 * a4 * (t ** 2) + 20 * a5 * (t ** 3))


def sat_overlap_batch(Ce, ue, ve, Le, We, Cb, ub, vb, eb_u, eb_v):
    """SAT OBB-vs-OBB. Ce/ue/ve: (M,2). Cb/ub/vb: (2,). Returns (M,) bool overlap."""
    D = Ce - Cb[None, :]
    axes = [ue, ve, np.broadcast_to(ub, ue.shape), np.broadcast_to(vb, ue.shape)]
    collide = np.ones((Ce.shape[0],), dtype=bool)
    for L in axes:
        Ra = Le * np.abs(np.sum(ue * L, axis=1)) + We * np.abs(np.sum(ve * L, axis=1))
        Rb = eb_u * np.abs(ub[0] * L[:, 0] + ub[1] * L[:, 1]) + eb_v * np.abs(vb[0] * L[:, 0] + vb[1] * L[:, 1])
        sep = np.abs(D[:, 0] * L[:, 0] + D[:, 1] * L[:, 1]) > (Ra + Rb)
        collide &= ~sep
    return collide


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class FrenetLocalPath:
    def __init__(self):
        rospy.init_node("frenet_local_path", anonymous=True)

        self.frame_id = rospy.get_param("~frame_id", "map")
        self.path_file = rospy.get_param("~path_file", "2021042015_양세현_c_track.txt")

        # sampling / geometry
        self.SF = float(rospy.get_param("~sf", 5.0))            # lookahead [m]
        self.NPTS = int(rospy.get_param("~n_pts", 80))           # points along path
        self.N_D = int(rospy.get_param("~n_d", 9))               # lateral candidates
        self.D_MAX = float(rospy.get_param("~d_max", 2.5))       # max |lateral offset| [m]
        self.KAP_TH = float(rospy.get_param("~kappa_thresh", 1.0))
        self.DPS_CLIP = float(rospy.get_param("~dps_clip", 0.5))
        # Complete the lateral shift ~reach_buffer before the nearest obstacle
        # (clamped to >= reach_min), then hold the offset. Otherwise the quintic
        # smears the shift over the whole horizon and is barely displaced at the
        # obstacle, so every candidate collides -> no avoidance.
        self.reach_buffer = float(rospy.get_param("~reach_buffer", 2.0))
        self.reach_min = float(rospy.get_param("~reach_min", 4.0))

        # ego footprint (half length / width) [m]
        self.EGO_HALF_L = float(rospy.get_param("~ego_half_l", 0.785))
        self.EGO_HALF_W = float(rospy.get_param("~ego_half_w", 0.590))
        # Each obstacle (cluster point) is modeled as a fixed axis-aligned box of
        # obs_box_w (lateral) x obs_box_l (longitudinal), centered on the point.
        self.OBS_HALF_X = 0.5 * float(rospy.get_param("~obs_box_w", 2.0))
        self.OBS_HALF_Y = 0.5 * float(rospy.get_param("~obs_box_l", 1.6))
        self.OBS_RAD = math.hypot(self.OBS_HALF_X, self.OBS_HALF_Y)  # for window/clearance

        # costs
        self.W_D = float(rospy.get_param("~w_d", 0.2))
        self.W_K = float(rospy.get_param("~w_k", 0.1))
        self.side_bias = float(rospy.get_param("~side_bias", 0.0))  # >0 prefers left
        # Temporal consistency (hysteresis): penalize changing the chosen lateral
        # offset from the previous cycle so the selection is "sticky" and doesn't
        # chatter between adjacent candidates on small obstacle/ego jitter. Scaled
        # by NPTS at use so it is directly comparable to W_D. 0 = no hysteresis.
        self.w_consistency = float(rospy.get_param("~w_consistency", 0.7))

        # Only obstacles whose cluster centroid is within this lateral distance
        # (|Frenet d|, i.e. perpendicular offset from the global path) get an
        # avoidance path. Clusters farther off to the side (parked cars, walls,
        # scenery) are ignored so we don't swerve for things outside our lane.
        self.avoid_lateral_thresh = float(rospy.get_param("~avoid_lateral_thresh", 1.45))
        # Asymmetric side reach: obstacles can be in the OPPOSITE (left) lane, so allow a
        # larger LEFT threshold than the near/right one. Uses a SIGNED offset (+ = left of
        # travel direction); avoid_left_sign flips it if this map's left comes out
        # negative. Both default to avoid_lateral_thresh (= old symmetric behavior).
        self.avoid_lat_left = float(rospy.get_param("~avoid_lat_left", self.avoid_lateral_thresh))
        self.avoid_lat_right = float(rospy.get_param("~avoid_lat_right", self.avoid_lateral_thresh))
        self.avoid_left_sign = float(rospy.get_param("~avoid_left_sign", 1.0))
        # Ignore obstacles the ego has already driven past: their centroid is more
        # than this far BEHIND the ego (along its heading). Without this, a cluster
        # that is now beside/behind us keeps "active" non-empty and delays the
        # return to the global path. ~= rear overhang + obstacle half-length so we
        # only drop it once it is fully behind the vehicle.
        self.rear_ignore_dist = float(rospy.get_param("~rear_ignore_dist", 1.0))

        # obstacle io
        self.obstacle_topic = rospy.get_param("~obstacle_topic", "/lidar_clusters")
        self.obstacle_timeout = float(rospy.get_param("~obstacle_timeout", 0.5))
        self.use_odom_transform_fallback = bool(rospy.get_param("~use_odom_transform_fallback", True))

        self.DS = self.SF / self.NPTS
        self.Sq = np.linspace(0.0, self.SF, self.NPTS, dtype=np.float32)

        # Candidate lateral offsets. +d = left. "left" only samples [0, D_MAX] so
        # avoidance never steers right; "right" -> [-D_MAX, 0]; "both" -> symmetric.
        self.avoid_side = str(rospy.get_param("~avoid_side", "left")).strip().lower()
        if self.avoid_side == "left":
            self.levels = np.linspace(0.0, self.D_MAX, self.N_D, dtype=np.float32)
        elif self.avoid_side == "right":
            self.levels = np.linspace(-self.D_MAX, 0.0, self.N_D, dtype=np.float32)
        else:  # "both"
            self.levels = np.linspace(-self.D_MAX, self.D_MAX, self.N_D, dtype=np.float32)

        self.gx, self.gy, self.gs = self.load_path()
        self.cur_idx = 0

        self.odom = None
        self.obstacles = []          # list of (x, y) in map frame
        self.last_obstacle_stamp = rospy.Time(0)
        self.prev_d1 = 0.0           # last chosen lateral end-offset (for hysteresis)
        self.tf_listener = tf.TransformListener()

        self.global_path_pub = rospy.Publisher("/global_path", Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher("/local_path", Path, queue_size=1)
        self.samples_pub = rospy.Publisher("/sampled_paths", Path, queue_size=1)
        self.obs_viz_pub = rospy.Publisher("/obstacle_boxes", MarkerArray, queue_size=1)

        rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(self.obstacle_topic, MarkerArray, self.obstacle_callback, queue_size=1)

        self.global_path_msg = self.build_global_path_msg()
        rospy.loginfo(
            "Frenet local path ready: %d global pts, SF=%.1fm NPTS=%d N_D=%d",
            len(self.gx), self.SF, self.NPTS, self.N_D,
        )

    # ---------- setup ----------
    def load_path(self):
        pkg_path = rospkg.RosPack().get_path("morai_obstacle_drive")
        full = os.path.join(pkg_path, "path", self.path_file)
        xs, ys = [], []
        with open(full, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
        gx = np.array(xs, np.float64)
        gy = np.array(ys, np.float64)
        ds = np.hypot(np.diff(gx), np.diff(gy))
        gs = np.concatenate([[0.0], np.cumsum(ds)])
        return gx, gy, gs

    def build_global_path_msg(self):
        msg = Path()
        msg.header.frame_id = self.frame_id
        for x, y in zip(self.gx, self.gy):
            p = PoseStamped()
            p.header.frame_id = self.frame_id
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.orientation.w = 1.0
            msg.poses.append(p)
        return msg

    # ---------- callbacks ----------
    def odom_callback(self, msg):
        self.odom = msg

    def odom_yaw(self):
        q = self.odom.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    def lidar_local_to_map(self, marker):
        if self.odom is None:
            return None
        ego = self.odom.pose.pose.position
        yaw = self.odom_yaw()
        lx, ly = marker.pose.position.x, marker.pose.position.y
        pt = PointStamped()
        pt.header.stamp = rospy.Time.now()
        pt.header.frame_id = self.frame_id
        pt.point.x = ego.x + math.cos(yaw) * lx - math.sin(yaw) * ly
        pt.point.y = ego.y + math.sin(yaw) * lx + math.cos(yaw) * ly
        return pt

    def obstacle_callback(self, msg):
        obstacles = []
        for marker in msg.markers:
            if marker.action == Marker.DELETE:
                continue
            pt = PointStamped()
            pt.header = marker.header
            pt.point = marker.pose.position
            try:
                if pt.header.frame_id and pt.header.frame_id.strip("/") != self.frame_id.strip("/"):
                    pt = self.tf_listener.transformPoint(self.frame_id, pt)
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
                if not self.use_odom_transform_fallback:
                    rospy.logwarn_throttle(2.0, "Obstacle TF failed: %s", exc)
                    continue
                pt = self.lidar_local_to_map(marker)
                if pt is None:
                    continue
            # Ignore clusters outside our lane band. ASYMMETRIC: reach farther to the
            # LEFT (opposite lane may hold obstacles) than to the right. d>0 = left.
            d = self.global_signed_offset(pt.point.x, pt.point.y)
            if d > self.avoid_lat_left or d < -self.avoid_lat_right:
                continue
            # Each cluster point becomes a fixed box (obs_box_w x obs_box_l);
            # actual footprint is applied in the SAT check, not here.
            obstacles.append((pt.point.x, pt.point.y))
        self.obstacles = obstacles
        self.last_obstacle_stamp = rospy.Time.now()

    def global_lateral_offset(self, ox, oy):
        # Perpendicular distance (m) from a map point to the global path,
        # approximated by the distance to the nearest global path vertex. The
        # path is dense, so this closely matches the true Frenet |d|.
        return float(np.sqrt(np.min((self.gx - ox) ** 2 + (self.gy - oy) ** 2)))

    def global_signed_offset(self, ox, oy):
        # SIGNED perpendicular offset (m) from the global path: + = LEFT of the travel
        # direction, - = RIGHT. Sign from the tangent x offset cross product at the
        # nearest vertex; avoid_left_sign flips the convention if this map's left comes
        # out negative. Lets the side band be asymmetric (see avoid_lat_left/right).
        i = int(np.argmin((self.gx - ox) ** 2 + (self.gy - oy) ** 2))
        j0 = max(i - 1, 0)
        j1 = min(i + 1, len(self.gx) - 1)
        tx = self.gx[j1] - self.gx[j0]
        ty = self.gy[j1] - self.gy[j0]
        tn = (tx * tx + ty * ty) ** 0.5
        if tn < 1e-6:
            return 0.0
        ovx = ox - self.gx[i]
        ovy = oy - self.gy[i]
        return float(self.avoid_left_sign * (tx * ovy - ty * ovx) / tn)

    def obstacles_active(self):
        if not self.obstacles:
            return []
        if (rospy.Time.now() - self.last_obstacle_stamp).to_sec() > self.obstacle_timeout:
            return []
        return self.obstacles

    # ---------- reference ----------
    def nearest_index(self, x, y):
        n = len(self.gx)
        lo = max(self.cur_idx - 5, 0)
        hi = min(self.cur_idx + 200, n)
        seg = (self.gx[lo:hi] - x) ** 2 + (self.gy[lo:hi] - y) ** 2
        idx = lo + int(np.argmin(seg))
        if seg[idx - lo] > 25.0:  # lost -> global search
            idx = int(np.argmin((self.gx - x) ** 2 + (self.gy - y) ** 2))
        self.cur_idx = idx
        return idx

    def build_reference(self, x, y):
        i0 = self.nearest_index(x, y)
        s0 = self.gs[i0]
        s_query = np.clip(s0 + self.Sq, self.gs[0], self.gs[-1])
        rx = np.interp(s_query, self.gs, self.gx).astype(np.float32)
        ry = np.interp(s_query, self.gs, self.gy).astype(np.float32)
        # heading from finite differences
        dx = np.gradient(rx)
        dy = np.gradient(ry)
        psi = np.arctan2(dy, dx)
        return rx, ry, np.sin(psi).astype(np.float32), np.cos(psi).astype(np.float32)

    # ---------- planning ----------
    def plan(self):
        pos = self.odom.pose.pose.position
        rx, ry, sp, cp = self.build_reference(pos.x, pos.y)
        active = self.obstacles_active()

        # Drop obstacles the ego has already passed (centroid behind the rear),
        # so "active" empties right after the pass and we return to the line
        # instead of avoiding a cluster now beside/behind us.
        if active:
            yaw = self.odom_yaw()
            cy, sy = math.cos(yaw), math.sin(yaw)
            active = [(ox, oy) for (ox, oy) in active
                      if (cy * (ox - pos.x) + sy * (oy - pos.y)) > -self.rear_ignore_dist]

        # Lateral maneuver length: finish the shift ~reach_buffer before the nearest
        # obstacle (clamped to >= reach_min), then hold. This keeps candidates fully
        # displaced AT the obstacle instead of smeared over the whole horizon.
        if active:
            obs_s = min(int(np.argmin((rx - ox) ** 2 + (ry - oy) ** 2)) * self.DS
                        for (ox, oy) in active)
            L_reach = clamp(obs_s - self.reach_buffer, self.reach_min, self.SF)
        else:
            L_reach = self.SF
        tq = np.clip(self.Sq / L_reach, 0.0, 1.0).astype(np.float32)
        dt2 = np.float32(1.0 / (L_reach ** 2))

        # ego Frenet state at s=0
        yaw_ref0 = math.atan2(float(sp[0]), float(cp[0]))
        d0 = -(pos.x - float(rx[0])) * math.sin(yaw_ref0) + (pos.y - float(ry[0])) * math.cos(yaw_ref0)
        yaw_ego = self.odom_yaw()
        d0p = clamp(math.tan(yaw_ego - yaw_ref0), -self.DPS_CLIP, self.DPS_CLIP)

        # candidate quintics: (d0, d0p, 0) -> (d1, 0, 0)
        d1 = self.levels
        zeros = np.zeros_like(d1)
        coef = quintic_coeffs(d0, d0p, 0.0, d1, zeros, zeros)   # (C, 6)
        d_all = horner(coef, tq)                                # (C, NPTS)
        dpp_all = dpp_eval(coef, tq, dt2)                       # (C, NPTS) ~ d''(s)

        # cartesian
        X = rx[None, :] - d_all * sp[None, :]
        Y = ry[None, :] + d_all * cp[None, :]

        # d'(s) via finite difference (for footprint heading)
        dp_all = np.empty_like(d_all)
        dp_all[:, 1:-1] = (d_all[:, 2:] - d_all[:, :-2]) / (2 * self.DS)
        dp_all[:, 0] = (d_all[:, 1] - d_all[:, 0]) / self.DS
        dp_all[:, -1] = (d_all[:, -1] - d_all[:, -2]) / self.DS

        C = d_all.shape[0]
        valid = np.max(np.abs(dpp_all), axis=1) <= self.KAP_TH   # curvature bound (kappa_ref=0)
        cost = self.W_D * np.sum(np.abs(d_all), axis=1) + self.W_K * np.sum(dpp_all ** 2, axis=1)
        cost = cost - self.side_bias * np.sum(d_all, axis=1)     # >0 side_bias favors +d (left)
        # Hysteresis: bias toward the offset chosen last cycle so the selection
        # doesn't flip between near-equal candidates. NPTS scaling makes
        # w_consistency comparable to W_D (a per-point deviation weight).
        # ONLY while an obstacle is active: with no obstacle the single sensible
        # choice is the centerline, and a standing hysteresis term would out-weigh
        # the (half-as-large) return-path deviation cost and lock in the offset,
        # so the car never comes back to the global path.
        if active:
            cost = cost + self.w_consistency * self.NPTS * np.abs(self.levels - self.prev_d1)

        # SAT collision check per obstacle. Each obstacle is a fixed axis-aligned
        # box (OBS_HALF_X x OBS_HALF_Y, i.e. obs_box_w x obs_box_l) centered on the
        # cluster point; ego is its own OBB swept along the candidate path.
        yaw_ref = np.arctan2(sp, cp)
        obs_ub = np.array([1.0, 0.0], np.float32)
        obs_vb = np.array([0.0, 1.0], np.float32)
        clearance = np.full(C, np.inf, np.float32)               # min ego-center -> obstacle dist
        for (ox, oy) in active:
            oidx = int(np.argmin((rx - ox) ** 2 + (ry - oy) ** 2))
            half = int((self.EGO_HALF_L + self.OBS_RAD) / self.DS) + 2
            lo = max(0, oidx - half)
            hi = min(self.NPTS, oidx + half + 1)
            if hi <= lo:
                continue
            idxs = np.arange(lo, hi)

            dxs = X[:, idxs] - ox
            dys = Y[:, idxs] - oy
            clearance = np.minimum(clearance, np.sqrt(dxs ** 2 + dys ** 2).min(axis=1) - self.OBS_RAD)

            yaw = yaw_ref[None, idxs] + np.arctan(np.clip(dp_all[:, idxs], -self.DPS_CLIP, self.DPS_CLIP))
            ue = np.stack([np.cos(yaw), np.sin(yaw)], axis=2)     # (C, W, 2)
            ve = np.stack([-ue[:, :, 1], ue[:, :, 0]], axis=2)
            Ce = np.stack([X[:, idxs], Y[:, idxs]], axis=2)       # (C, W, 2)

            W = idxs.size
            coll = sat_overlap_batch(
                Ce.reshape(C * W, 2), ue.reshape(C * W, 2), ve.reshape(C * W, 2),
                self.EGO_HALF_L, self.EGO_HALF_W,
                np.array([ox, oy], np.float32), obs_ub, obs_vb,
                self.OBS_HALF_X, self.OBS_HALF_Y,
            ).reshape(C, W).any(axis=1)
            valid &= ~coll

        if np.any(valid):
            best = int(np.where(valid, cost, np.inf).argmin())
        elif active:
            # No collision-free candidate: steer to the most-open side instead of
            # falling back to the (blocked) centerline.
            best = int(np.argmax(clearance))
            rospy.logwarn_throttle(1.0, "No collision-free candidate; max-clearance fallback d1=%.2f", float(self.levels[best]))
        else:
            best = int(cost.argmin())

        self.prev_d1 = float(self.levels[best])   # remember choice for next-cycle hysteresis
        avoiding = bool(active) and abs(float(d_all[best, self.NPTS // 2])) > 0.15
        self.publish_samples(X, Y)
        return self.to_path(X[best], Y[best]), avoiding

    def to_path(self, xs, ys):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame_id
        for x, y in zip(xs, ys):
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.orientation.w = 1.0
            path.poses.append(p)
        return path

    def publish_obstacle_boxes(self):
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.frame_id
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for i, (ox, oy) in enumerate(self.obstacles_active()):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = rospy.Time.now()
            m.ns = "obstacle_boxes"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(ox)
            m.pose.position.y = float(oy)
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = 2.0 * self.OBS_HALF_X   # map-x extent (= obs_box_w)
            m.scale.y = 2.0 * self.OBS_HALF_Y   # map-y extent (= obs_box_l)
            m.scale.z = 0.6
            m.color.r = 1.0
            m.color.g = 0.3
            m.color.b = 0.0
            m.color.a = 0.4
            arr.markers.append(m)
        self.obs_viz_pub.publish(arr)

    def publish_samples(self, X, Y):
        if self.samples_pub.get_num_connections() == 0:
            return
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = rospy.Time.now()
        for c in range(X.shape[0]):
            for i in range(0, X.shape[1], 4):
                p = PoseStamped()
                p.pose.position.x = float(X[c, i])
                p.pose.position.y = float(Y[c, i])
                p.pose.orientation.w = 1.0
                msg.poses.append(p)
        self.samples_pub.publish(msg)

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.global_path_msg.header.stamp = rospy.Time.now()
            self.global_path_pub.publish(self.global_path_msg)

            if self.odom is None:
                rospy.loginfo_throttle(1.0, "Waiting for /odom topic...")
                rate.sleep()
                continue

            self.publish_obstacle_boxes()
            local_path, avoiding = self.plan()
            self.local_path_pub.publish(local_path)
            # (per-cycle local_path log removed to keep the console clean)
            rate.sleep()


if __name__ == "__main__":
    try:
        FrenetLocalPath().run()
    except rospy.ROSInterruptException:
        pass
