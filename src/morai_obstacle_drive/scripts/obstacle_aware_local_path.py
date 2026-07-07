#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os

import rospy
import rospkg
import tf
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value, low, high):
    return max(low, min(high, value))


def smoothstep(t):
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class ObstacleAwareLocalPath:
    def __init__(self):
        rospy.init_node("obstacle_aware_local_path", anonymous=True)

        self.frame_id = rospy.get_param("~frame_id", "map")
        self.path_file = rospy.get_param("~path_file", "2021042015_양세현_c_track.txt")
        self.local_path_size = int(rospy.get_param("~local_path_size", 70))
        self.search_back_size = int(rospy.get_param("~search_back_size", 5))
        self.search_forward_size = int(rospy.get_param("~search_forward_size", 140))

        self.obstacle_topic = rospy.get_param("~obstacle_topic", "/lidar_clusters")
        self.obstacle_timeout = float(rospy.get_param("~obstacle_timeout", 0.5))
        self.avoid_lookahead = float(rospy.get_param("~avoid_lookahead", 25.0))
        self.avoid_start_margin = float(rospy.get_param("~avoid_start_margin", 4.0))
        self.avoid_end_margin = float(rospy.get_param("~avoid_end_margin", 9.0))
        self.avoid_lateral_offset = float(rospy.get_param("~avoid_lateral_offset", 2.0))
        # Which side to steer around obstacles: "left" (default), "right", or
        # "auto" (opposite side of the obstacle). Path normal (nx,ny) points left,
        # so a positive offset shifts the path to the left.
        self.avoid_side = str(rospy.get_param("~avoid_side", "left")).strip().lower()
        self.path_block_width = float(rospy.get_param("~path_block_width", 2.0))
        self.obstacle_inflate = float(rospy.get_param("~obstacle_inflate", 0.45))
        self.min_obstacle_s = float(rospy.get_param("~min_obstacle_s", 0.5))
        self.use_odom_transform_fallback = bool(rospy.get_param("~use_odom_transform_fallback", True))

        self.global_path = self.load_path()
        self.ref_points = self.build_reference_frame(self.global_path)
        self.odom = None
        self.current_waypoint = 0
        self.obstacles = []
        self.closest_obstacle = None
        self.raw_obstacle_count = 0
        self.last_obstacle_stamp = rospy.Time(0)
        self.tf_listener = tf.TransformListener()

        self.global_path_pub = rospy.Publisher("/global_path", Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher("/local_path", Path, queue_size=1)
        self.avoid_viz_pub = rospy.Publisher("/avoidance_debug", MarkerArray, queue_size=1)

        rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(self.obstacle_topic, MarkerArray, self.obstacle_callback, queue_size=1)

        rospy.loginfo(
            "Obstacle-aware local path ready: %d global points, %.1fm Frenet reference",
            len(self.global_path.poses),
            self.ref_points[-1]["s"] if self.ref_points else 0.0,
        )

    def load_path(self):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("morai_obstacle_drive")
        full_path = os.path.join(pkg_path, "path", self.path_file)

        path = Path()
        path.header.frame_id = self.frame_id

        with open(full_path, "r") as path_file:
            for line in path_file:
                parts = line.split()
                if len(parts) < 2:
                    continue
                pose = PoseStamped()
                pose.header.frame_id = self.frame_id
                pose.pose.position.x = float(parts[0])
                pose.pose.position.y = float(parts[1])
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)

        return path

    def build_reference_frame(self, path):
        points = [[pose.pose.position.x, pose.pose.position.y] for pose in path.poses]
        if not points:
            return []

        s_vals = [0.0]
        for p0, p1 in zip(points, points[1:]):
            s_vals.append(s_vals[-1] + math.hypot(p1[0] - p0[0], p1[1] - p0[1]))

        ref_points = []
        for idx, point in enumerate(points):
            if len(points) == 1:
                yaw = 0.0
            elif idx == 0:
                p0, p1 = points[0], points[1]
                yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            elif idx == len(points) - 1:
                p0, p1 = points[-2], points[-1]
                yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            else:
                p0, p1 = points[idx - 1], points[idx + 1]
                yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])

            ref_points.append({
                "x": point[0],
                "y": point[1],
                "s": s_vals[idx],
                "yaw": yaw,
                "sin": math.sin(yaw),
                "cos": math.cos(yaw),
                "nx": -math.sin(yaw),
                "ny": math.cos(yaw),
            })

        return ref_points

    def odom_callback(self, msg):
        self.odom = msg

    def odom_yaw(self):
        if self.odom is None:
            return 0.0
        q = self.odom.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    def lidar_local_to_map(self, marker):
        if self.odom is None:
            return None

        ego = self.odom.pose.pose.position
        yaw = self.odom_yaw()
        lx = marker.pose.position.x
        ly = marker.pose.position.y

        point = PointStamped()
        point.header.stamp = rospy.Time.now()
        point.header.frame_id = self.frame_id
        point.point.x = ego.x + math.cos(yaw) * lx - math.sin(yaw) * ly
        point.point.y = ego.y + math.sin(yaw) * lx + math.cos(yaw) * ly
        point.point.z = marker.pose.position.z
        return point

    def obstacle_callback(self, msg):
        transformed = []
        self.raw_obstacle_count = 0
        for marker in msg.markers:
            if marker.action == Marker.DELETE:
                continue

            self.raw_obstacle_count += 1
            point = PointStamped()
            point.header = marker.header
            point.point = marker.pose.position

            try:
                if point.header.frame_id and point.header.frame_id.strip("/") != self.frame_id.strip("/"):
                    point = self.tf_listener.transformPoint(self.frame_id, point)
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
                if not self.use_odom_transform_fallback:
                    rospy.logwarn_throttle(2.0, "Obstacle TF transform failed: %s", exc)
                    continue

                fallback_point = self.lidar_local_to_map(marker)
                if fallback_point is None:
                    rospy.logwarn_throttle(2.0, "Obstacle TF transform failed and /odom is not ready: %s", exc)
                    continue

                point = fallback_point
                rospy.logwarn_throttle(
                    2.0,
                    "Obstacle TF failed (%s). Using /odom yaw fallback for lidar-local obstacle coordinates.",
                    exc,
                )

            half_width = 0.5 * max(marker.scale.x, marker.scale.y, 0.3) + self.obstacle_inflate
            transformed.append((point.point.x, point.point.y, half_width))

        self.obstacles = transformed
        self.last_obstacle_stamp = rospy.Time.now()

    def nearest_waypoint(self):
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        path_len = len(self.ref_points)

        start_idx = max(self.current_waypoint - self.search_back_size, 0)
        end_idx = min(self.current_waypoint + self.search_forward_size, path_len)
        idx, dist = self.find_nearest_in_range(x, y, start_idx, end_idx)
        if dist > 5.0:
            idx, dist = self.find_nearest_in_range(x, y, 0, path_len)
        return idx, dist

    def find_nearest_in_range(self, x, y, start_idx, end_idx):
        best_idx = start_idx
        best_dist = float("inf")
        for idx in range(start_idx, end_idx):
            ref = self.ref_points[idx]
            dist = math.hypot(x - ref["x"], y - ref["y"])
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        return best_idx, best_dist

    def build_base_local_points(self, start_idx):
        end_idx = min(start_idx + self.local_path_size, len(self.ref_points))
        local_refs = []
        for idx in range(start_idx, end_idx):
            ref = self.ref_points[idx]
            local_refs.append({
                "x": ref["x"],
                "y": ref["y"],
                "s_global": ref["s"],
                "s": ref["s"] - self.ref_points[start_idx]["s"],
                "yaw": ref["yaw"],
                "nx": ref["nx"],
                "ny": ref["ny"],
            })
        return local_refs

    def project_to_frenet(self, x, y, refs):
        if not refs:
            return None
        if len(refs) == 1:
            ref = refs[0]
            dx = x - ref["x"]
            dy = y - ref["y"]
            return {
                "s": 0.0,
                "d": dx * ref["nx"] + dy * ref["ny"],
                "idx": 0,
                "distance": math.hypot(dx, dy),
                "x_ref": ref["x"],
                "y_ref": ref["y"],
                "nx": ref["nx"],
                "ny": ref["ny"],
            }

        best = None
        for idx in range(len(refs) - 1):
            p0 = refs[idx]
            p1 = refs[idx + 1]
            vx = p1["x"] - p0["x"]
            vy = p1["y"] - p0["y"]
            seg_len_sq = vx * vx + vy * vy
            if seg_len_sq < 1e-9:
                continue

            wx = x - p0["x"]
            wy = y - p0["y"]
            t = clamp((wx * vx + wy * vy) / seg_len_sq, 0.0, 1.0)
            proj_x = p0["x"] + t * vx
            proj_y = p0["y"] + t * vy
            seg_len = math.sqrt(seg_len_sq)
            yaw = math.atan2(vy, vx)
            nx = -math.sin(yaw)
            ny = math.cos(yaw)
            dx = x - proj_x
            dy = y - proj_y
            d = dx * nx + dy * ny
            dist = math.hypot(dx, dy)
            s = p0["s"] + t * seg_len

            candidate = {
                "s": s,
                "d": d,
                "idx": idx,
                "distance": dist,
                "x_ref": proj_x,
                "y_ref": proj_y,
                "nx": nx,
                "ny": ny,
            }
            if best is None or candidate["distance"] < best["distance"]:
                best = candidate

        return best

    def select_blocking_obstacle(self, refs):
        self.closest_obstacle = None
        if not refs or not self.obstacles:
            return None
        if (rospy.Time.now() - self.last_obstacle_stamp).to_sec() > self.obstacle_timeout:
            return None

        best = None
        for ox, oy, radius in self.obstacles:
            projection = self.project_to_frenet(ox, oy, refs)
            if projection is None:
                continue

            obs_s = projection["s"]
            obs_d = projection["d"]
            block_width = self.path_block_width + radius
            candidate = {
                "x": ox,
                "y": oy,
                "idx": projection["idx"],
                "s": obs_s,
                "d": obs_d,
                "distance": projection["distance"],
                "radius": radius,
                "block_width": block_width,
            }
            if self.closest_obstacle is None or candidate["distance"] < self.closest_obstacle["distance"]:
                self.closest_obstacle = candidate

            if self.min_obstacle_s <= obs_s <= self.avoid_lookahead and abs(obs_d) <= block_width:
                if best is None or candidate["s"] < best["s"]:
                    best = candidate

        return best

    def apply_avoidance(self, refs, obstacle):
        if obstacle is None:
            return refs, False

        if self.avoid_side == "right":
            direction = -1.0
        elif self.avoid_side == "auto":
            direction = -1.0 if obstacle["d"] >= 0.0 else 1.0
        else:  # "left" (default): always steer to the left side
            direction = 1.0
        offset = direction * self.avoid_lateral_offset
        start_s = max(0.0, obstacle["s"] - self.avoid_start_margin)
        end_s = min(refs[-1]["s"], obstacle["s"] + self.avoid_end_margin)
        span = max(end_s - start_s, 1e-3)

        avoided = []
        for ref in refs:
            s = ref["s"]
            point = dict(ref)
            if start_s <= s <= end_s:
                phase = (s - start_s) / span
                envelope = math.sin(math.pi * smoothstep(phase))
                point["x"] = ref["x"] + offset * envelope * ref["nx"]
                point["y"] = ref["y"] + offset * envelope * ref["ny"]
            avoided.append(point)

        return avoided, True

    def points_to_path(self, refs):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame_id
        for ref in refs:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = ref["x"]
            pose.pose.position.y = ref["y"]
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path

    def publish_debug(self, obstacle, avoiding):
        markers = MarkerArray()
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id
        marker.ns = "avoidance_state"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD if obstacle else Marker.DELETE
        if obstacle:
            marker.pose.position.x = obstacle["x"]
            marker.pose.position.y = obstacle["y"]
            marker.scale.x = marker.scale.y = marker.scale.z = max(0.4, obstacle["radius"] * 2.0)
            marker.color.r = 1.0 if avoiding else 0.0
            marker.color.g = 0.3
            marker.color.b = 0.0
            marker.color.a = 0.9
        markers.markers.append(marker)
        self.avoid_viz_pub.publish(markers)

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.global_path.header.stamp = rospy.Time.now()
            self.global_path_pub.publish(self.global_path)

            if self.odom is None:
                rospy.loginfo_throttle(1.0, "Waiting for /odom topic...")
                rate.sleep()
                continue

            self.current_waypoint, nearest_dist = self.nearest_waypoint()
            base_refs = self.build_base_local_points(self.current_waypoint)
            obstacle = self.select_blocking_obstacle(base_refs)
            local_refs, avoiding = self.apply_avoidance(base_refs, obstacle)

            self.local_path_pub.publish(self.points_to_path(local_refs))
            self.publish_debug(obstacle, avoiding)

            if avoiding:
                rospy.loginfo_throttle(
                    0.5,
                    "Avoiding obstacle: waypoint=%d obs_s=%.1fm obs_d=%.1fm nearest=%.2fm",
                    self.current_waypoint,
                    obstacle["s"],
                    obstacle["d"],
                    nearest_dist,
                )
            else:
                if self.closest_obstacle:
                    rospy.loginfo_throttle(
                        1.0,
                        (
                            "Normal local path: waypoint=%d nearest=%.2fm raw_obs=%d mapped_obs=%d "
                            "closest_s=%.1fm closest_d=%.1fm gate_s=[%.1f,%.1f] gate_d=%.1fm"
                        ),
                        self.current_waypoint,
                        nearest_dist,
                        self.raw_obstacle_count,
                        len(self.obstacles),
                        self.closest_obstacle["s"],
                        self.closest_obstacle["d"],
                        self.min_obstacle_s,
                        self.avoid_lookahead,
                        self.closest_obstacle["block_width"],
                    )
                else:
                    rospy.loginfo_throttle(
                        1.0,
                        "Normal local path: waypoint=%d nearest=%.2fm raw_obs=%d mapped_obs=%d",
                        self.current_waypoint,
                        nearest_dist,
                        self.raw_obstacle_count,
                        len(self.obstacles),
                    )

            rate.sleep()


if __name__ == "__main__":
    try:
        ObstacleAwareLocalPath().run()
    except rospy.ROSInterruptException:
        pass
