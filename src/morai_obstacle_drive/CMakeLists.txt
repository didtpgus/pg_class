cmake_minimum_required(VERSION 3.0.2)
project(morai_obstacle_drive)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  morai_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  tf
  visualization_msgs
  yolo
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/gpsimu_parser.py
  scripts/ego_tf_pub.py
  scripts/odom_to_ego_status.py
  scripts/lidar_obstacle_detector.py
  scripts/obstacle_aware_local_path.py
  scripts/pure_pursuit.py
  scripts/yolo_detection.py
  scripts/traffic_light_control.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch path
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
