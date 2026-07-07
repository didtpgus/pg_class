#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from sklearn.linear_model import RANSACRegressor
import hdbscan
import std_msgs.msg

def get_ros_params():
    params = {
        "ransac_threshold": rospy.get_param("~ransac_threshold", 0.3),
        "min_cluster_size": rospy.get_param("~min_cluster_size", 5),
        "min_samples": rospy.get_param("~min_samples", 3),
        "cluster_selection_epsilon": rospy.get_param("~cluster_selection_epsilon", 1.5),
        "lidar_topic": rospy.get_param("~lidar_topic", "/velodyne_points"),
        "roi_x_min": rospy.get_param("~roi_x_min", -15.0),
        "roi_x_max": rospy.get_param("~roi_x_max", 15.0),
        "roi_y_min": rospy.get_param("~roi_y_min", -6.0),
        "roi_y_max": rospy.get_param("~roi_y_max", 6.0),
        "roi_z_min": rospy.get_param("~roi_z_min", -5.0),
        "roi_z_max": rospy.get_param("~roi_z_max", 5.0),
    }

    rospy.loginfo("[lidar_clustering_node] Parameters:")
    for key, value in params.items():
        rospy.loginfo(f"  {key}: {value}")
    return params

prev_marker_ids = set() 

# Voxel 기반 다운샘플링
def voxel_downsample(points, voxel_size=0.1):
    discrete = np.floor(points / voxel_size)
    _, idx = np.unique(discrete, axis=0, return_index=True)
    return points[idx]

# RANSAC으로 지면 제거
def remove_ground_ransac(points, threshold=0.3):
    X = points[:, :2]
    y = points[:, 2]
    ransac = RANSACRegressor(residual_threshold=threshold)
    try:
        ransac.fit(X, y)
        z_pred = ransac.predict(X)
        residuals = np.abs(y - z_pred)
        mask = residuals > threshold
        return points[mask]
    except:
        return points

# 크기가 임계값 이상인 클러스터 제거
def filter_by_bounding_box(cluster_points, max_length=5.0, max_width=2.5):
    x_len = np.ptp(cluster_points[:, 0])
    y_len = np.ptp(cluster_points[:, 1])
    return (x_len < max_length) and (y_len < max_width)

# 다운샘플링된 포인트 퍼블리시
def publish_downsampled_points(points, frame_id):
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    cloud_msg = pc2.create_cloud_xyz32(header, points)
    downsample_pub.publish(cloud_msg)

# 가까운 클러스터 병합
def merge_close_clusters(points, labels, max_gap=1.0):
    unique_labels = set(labels)
    unique_labels.discard(-1)
    centroids = {}

    # 클러스터 중심 계산
    for l in unique_labels:
        cluster_pts = points[labels == l]
        centroids[l] = np.mean(cluster_pts, axis=0)

    merged_labels = dict()
    merged = set()

    # 중심 거리 기준으로 병합
    for l1 in unique_labels:
        if l1 in merged:
            continue
        merged_labels[l1] = l1
        for l2 in unique_labels:
            if l1 == l2 or l2 in merged:
                continue
            dist = np.linalg.norm(centroids[l1][:2] - centroids[l2][:2])
            if dist < max_gap:
                merged_labels[l2] = l1
                merged.add(l2)

    # 병합된 라벨 적용
    new_labels = np.array([merged_labels.get(lbl, -1) if lbl != -1 else -1 for lbl in labels])
    return new_labels

# PointCloud 콜백 함수
def pointcloud_callback(msg):
    global prev_marker_ids

    # PointCloud2를 numpy로 변환
    points = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(
        msg, field_names=("x", "y", "z"), skip_nans=True)])
    if len(points) == 0:
        return

    # ROI 필터링
    x_cond = (params["roi_x_min"] <= points[:, 0]) & (points[:, 0] <= params["roi_x_max"])
    y_cond = (params["roi_y_min"] <= points[:, 1]) & (points[:, 1] <= params["roi_y_max"])
    z_cond = (params["roi_z_min"] <= points[:, 2]) & (points[:, 2] <= params["roi_z_max"])
    roi_mask = x_cond & y_cond & z_cond
    roi_points = points[roi_mask]
    if len(roi_points) == 0:
        return

    # 다운샘플링
    downsampled_points = voxel_downsample(roi_points, voxel_size=0.1)
    if len(downsampled_points) == 0:
        return

    # 지면 제거
    non_ground_points = remove_ground_ransac(roi_points, threshold=params["ransac_threshold"])
    publish_downsampled_points(non_ground_points, msg.header.frame_id)
    
    if len(non_ground_points) == 0:
        return

    # 클러스터링 수행
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        cluster_selection_epsilon=params["cluster_selection_epsilon"],
    )
    labels = clusterer.fit_predict(non_ground_points)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # 가까운 클러스터 병합
    labels = merge_close_clusters(non_ground_points, labels, max_gap=1.0)
    merged_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    rospy.loginfo(f"HDBSCAN: {n_clusters} clusters detected before merge")
    rospy.loginfo(f"After merging close clusters: {merged_clusters} clusters")

    marker_array = MarkerArray()
    marker_id = 0
    curr_marker_ids = set()

    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        cluster_points = non_ground_points[labels == cluster_id]
        if len(cluster_points) == 0:
            continue
        if not filter_by_bounding_box(cluster_points):
            continue

        centroid = np.mean(cluster_points, axis=0)

        # RViz Marker 설정
        marker = Marker()
        marker.header.frame_id = msg.header.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "lidar_clusters"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = centroid[0]
        marker.pose.position.y = centroid[1]
        marker.pose.position.z = centroid[2]
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        marker_array.markers.append(marker)

        # 각 클러스터 중심 위치 퍼블리시
        pt_msg = PointStamped()
        pt_msg.header.frame_id = msg.header.frame_id
        pt_msg.header.stamp = rospy.Time.now()
        pt_msg.point.x = centroid[0]
        pt_msg.point.y = centroid[1]
        pt_msg.point.z = centroid[2]
        cluster_pos_pub.publish(pt_msg)

        curr_marker_ids.add(marker_id)
        marker_id += 1

    # 사라진 마커는 삭제 처리
    removed_ids = prev_marker_ids - curr_marker_ids
    for rem_id in removed_ids:
        del_marker = Marker()
        del_marker.header.frame_id = msg.header.frame_id
        del_marker.header.stamp = rospy.Time.now()
        del_marker.ns = "lidar_clusters"
        del_marker.id = rem_id
        del_marker.action = Marker.DELETE
        marker_array.markers.append(del_marker)

    prev_marker_ids = curr_marker_ids
    marker_pub.publish(marker_array)

# ROS 노드 초기화 및 실행
if __name__ == "__main__":
    rospy.init_node("lidar_clustering_node")

    params = get_ros_params()

    marker_pub = rospy.Publisher("/lidar_clusters", MarkerArray, queue_size=1)
    downsample_pub = rospy.Publisher("/downsampled_points", PointCloud2, queue_size=1)
    cluster_pos_pub = rospy.Publisher("/cluster_positions", PointStamped, queue_size=10)

    rospy.Subscriber(params["lidar_topic"], PointCloud2, pointcloud_callback)
    rospy.spin()
