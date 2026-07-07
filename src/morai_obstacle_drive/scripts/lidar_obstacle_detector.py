#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2
from sklearn.linear_model import RANSACRegressor
from visualization_msgs.msg import Marker, MarkerArray

import hdbscan


def get_ros_params():
    params = {
        "ransac_threshold": rospy.get_param("~ransac_threshold", 0.2),
        "min_cluster_size": rospy.get_param("~min_cluster_size", 25),
        "min_samples": rospy.get_param("~min_samples", 5),
        "cluster_selection_epsilon": rospy.get_param("~cluster_selection_epsilon", 0.1),
        "lidar_topic": rospy.get_param("~lidar_topic", "/velodyne_points"),
        "roi_x_min": rospy.get_param("~roi_x_min", -3.0),
        "roi_x_max": rospy.get_param("~roi_x_max", 20.0),
        "roi_y_min": rospy.get_param("~roi_y_min", -3.0),
        "roi_y_max": rospy.get_param("~roi_y_max", 3.0),
        "roi_z_min": rospy.get_param("~roi_z_min", -2.0),
        "roi_z_max": rospy.get_param("~roi_z_max", 0.5),
        "max_length": rospy.get_param("~max_length", 2.0),
        "max_width": rospy.get_param("~max_width", 2.0),
        "min_height": rospy.get_param("~min_height", 0.25),
        "voxel_size": rospy.get_param("~voxel_size", 0.1),
        "merge_gap": rospy.get_param("~merge_gap", 1.0),
    }

    rospy.loginfo("[lidar_obstacle_detector] Parameters:")
    for key, value in params.items():
        rospy.loginfo("  %s: %s", key, value)
    return params


def voxel_downsample(points, voxel_size):
    discrete = np.floor(points / voxel_size)
    _, idx = np.unique(discrete, axis=0, return_index=True)
    return points[idx]


def remove_ground_ransac(points, threshold):
    x = points[:, :2]
    y = points[:, 2]
    ransac = RANSACRegressor(residual_threshold=threshold)
    try:
        ransac.fit(x, y)
        residuals = np.abs(y - ransac.predict(x))
        return points[residuals > threshold]
    except Exception:
        return points


def merge_close_clusters(points, labels, max_gap):
    unique_labels = set(labels)
    unique_labels.discard(-1)
    centroids = {}
    for label in unique_labels:
        centroids[label] = np.mean(points[labels == label], axis=0)

    merged_labels = {}
    merged = set()
    for label_a in unique_labels:
        if label_a in merged:
            continue
        merged_labels[label_a] = label_a
        for label_b in unique_labels:
            if label_a == label_b or label_b in merged:
                continue
            dist = np.linalg.norm(centroids[label_a][:2] - centroids[label_b][:2])
            if dist < max_gap:
                merged_labels[label_b] = label_a
                merged.add(label_b)

    return np.array([merged_labels.get(label, -1) if label != -1 else -1 for label in labels])


class LidarObstacleDetector:
    def __init__(self):
        rospy.init_node("lidar_obstacle_detector", anonymous=True)
        self.params = get_ros_params()
        self.prev_obstacle_marker_ids = set()
        self.prev_cluster_marker_ids = set()

        self.obstacle_pub = rospy.Publisher("/lidar_obstacles", MarkerArray, queue_size=1)
        self.cluster_pub = rospy.Publisher("/lidar_clusters", MarkerArray, queue_size=1)
        self.downsample_pub = rospy.Publisher("/downsampled_points", PointCloud2, queue_size=1)
        self.cluster_pos_pub = rospy.Publisher("/cluster_positions", PointStamped, queue_size=10)
        rospy.Subscriber(self.params["lidar_topic"], PointCloud2, self.pointcloud_callback, queue_size=1)

        rospy.loginfo("LiDAR obstacle detector ready on %s", self.params["lidar_topic"])

    def publish_points(self, points, frame_id):
        header = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=frame_id)
        self.downsample_pub.publish(pc2.create_cloud_xyz32(header, points))

    def valid_bbox(self, cluster_points):
        x_len = np.ptp(cluster_points[:, 0])
        y_len = np.ptp(cluster_points[:, 1])
        z_len = np.ptp(cluster_points[:, 2])
        return (
            x_len < self.params["max_length"]
            and y_len < self.params["max_width"]
            and z_len >= self.params["min_height"]
        )

    def make_delete_markers(self, marker_array, frame_id, curr_marker_ids, prev_marker_ids, namespace):
        removed_ids = prev_marker_ids - curr_marker_ids
        for marker_id in removed_ids:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = namespace
            marker.id = marker_id
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        return curr_marker_ids

    def pointcloud_callback(self, msg):
        points = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )])
        if len(points) == 0:
            return

        p = self.params
        mask = (
            (p["roi_x_min"] <= points[:, 0]) & (points[:, 0] <= p["roi_x_max"])
            & (p["roi_y_min"] <= points[:, 1]) & (points[:, 1] <= p["roi_y_max"])
            & (p["roi_z_min"] <= points[:, 2]) & (points[:, 2] <= p["roi_z_max"])
        )
        roi_points = points[mask]
        if len(roi_points) == 0:
            return

        downsampled = voxel_downsample(roi_points, p["voxel_size"])
        if len(downsampled) == 0:
            return

        # Match lidar_pkg/scripts/lidar_clustering.py: RANSAC is applied to ROI points.
        non_ground = remove_ground_ransac(roi_points, p["ransac_threshold"])
        self.publish_points(non_ground, msg.header.frame_id)
        if len(non_ground) == 0:
            return

        # HDBSCAN's Boruvka KDTree queries k (~min_samples) neighbors and raises
        # "k must be <= number of training points" when too few points remain.
        # Below the minimum needed to form a cluster, treat everything as noise.
        min_cluster_points = max(int(p["min_cluster_size"]), int(p["min_samples"]) + 1)
        if len(non_ground) < min_cluster_points:
            labels = np.full(len(non_ground), -1, dtype=int)
        else:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=p["min_cluster_size"],
                min_samples=p["min_samples"],
                cluster_selection_epsilon=p["cluster_selection_epsilon"],
            )
            labels = clusterer.fit_predict(non_ground)
            labels = merge_close_clusters(non_ground, labels, p["merge_gap"])

        obstacle_markers = MarkerArray()
        cluster_markers = MarkerArray()
        curr_marker_ids = set()
        marker_id = 0

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            cluster_points = non_ground[labels == cluster_id]
            if len(cluster_points) == 0 or not self.valid_bbox(cluster_points):
                continue

            mins = np.min(cluster_points, axis=0)
            maxs = np.max(cluster_points, axis=0)
            centroid = np.mean(cluster_points, axis=0)
            box_centroid = (mins + maxs) * 0.5
            scale = np.maximum(maxs - mins, np.array([0.25, 0.25, 0.25]))

            box = Marker()
            box.header.frame_id = msg.header.frame_id
            box.header.stamp = rospy.Time.now()
            box.ns = "lidar_obstacles"
            box.id = marker_id
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = float(box_centroid[0])
            box.pose.position.y = float(box_centroid[1])
            box.pose.position.z = float(box_centroid[2])
            box.pose.orientation.w = 1.0
            box.scale.x = float(scale[0])
            box.scale.y = float(scale[1])
            box.scale.z = float(scale[2])
            box.color.r = 1.0
            box.color.g = 0.2
            box.color.b = 0.0
            box.color.a = 0.55
            obstacle_markers.markers.append(box)

            sphere = Marker()
            sphere.header = box.header
            sphere.ns = "lidar_clusters"
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(centroid[0])
            sphere.pose.position.y = float(centroid[1])
            sphere.pose.position.z = float(centroid[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
            sphere.color.r = 0.0
            sphere.color.g = 1.0
            sphere.color.b = 0.0
            sphere.color.a = 0.8
            cluster_markers.markers.append(sphere)

            pt_msg = PointStamped()
            pt_msg.header = box.header
            pt_msg.point = sphere.pose.position
            self.cluster_pos_pub.publish(pt_msg)

            curr_marker_ids.add(marker_id)
            marker_id += 1

        self.prev_obstacle_marker_ids = self.make_delete_markers(
            obstacle_markers,
            msg.header.frame_id,
            curr_marker_ids,
            self.prev_obstacle_marker_ids,
            "lidar_obstacles",
        )
        self.prev_cluster_marker_ids = self.make_delete_markers(
            cluster_markers,
            msg.header.frame_id,
            curr_marker_ids,
            self.prev_cluster_marker_ids,
            "lidar_clusters",
        )
        self.obstacle_pub.publish(obstacle_markers)
        self.cluster_pub.publish(cluster_markers)


if __name__ == "__main__":
    try:
        LidarObstacleDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
