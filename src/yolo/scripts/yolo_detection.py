#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import os

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO

from yolo.msg import YoloDetection, YoloDetectionArray

class YoloNode:
    def __init__(self):
        rospy.init_node('yolo_detection_node', anonymous=True)

        model_path = rospy.get_param('~model_path', 'yolov8n.pt')
        image_topic = rospy.get_param('~image_topic', '/usb_cam/image_raw/compressed')
        self.debug = rospy.get_param('~debug', False)

        if not os.path.exists(model_path):
            rospy.logerr(f"YOLO model file not found at {model_path}")
            return

        self.detection_pub = rospy.Publisher('yolo_detections', YoloDetectionArray, queue_size=10)

        self.bridge = CvBridge()
        self.model = YOLO(model_path)

        rospy.Subscriber(image_topic, CompressedImage, self.image_callback, queue_size=1)
        
        rospy.loginfo(f"YOLO detection node started. Subscribing to {image_topic}")
        rospy.spin()

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                rospy.logwarn("Failed to decode image, skipping frame.")
                return

            results = self.model(cv_image, conf=0.6, verbose=False)[0] # conf: confidence threshold, verbose: terminal output

            detection_array_msg = YoloDetectionArray()
            detection_array_msg.header = msg.header

            for box in results.boxes:
                detection_msg = YoloDetection()

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())

                detection_msg.class_name = self.model.names[cls_id]
                detection_msg.confidence = conf
                detection_msg.x = x1
                detection_msg.y = y1
                detection_msg.width = x2 - x1
                detection_msg.height = y2 - y1

                detection_array_msg.detections.append(detection_msg)

            if len(detection_array_msg.detections) > 0:
                self.detection_pub.publish(detection_array_msg)

            if self.debug:
                debug_image = cv_image.copy()
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    label = f"{self.model.names[cls_id]} {conf:.2f}"
                    cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(debug_image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("YOLO Detection", debug_image)
                cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        YoloNode()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
