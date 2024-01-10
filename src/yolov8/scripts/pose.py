#!/usr/bin/env python3
import sys
import os
yoloPath = '/home/nvidia/xjb/algorithm_ws/src/yolov8/'
sys.path.append(yoloPath) 
from ultralytics import YOLO
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from yolov8.msg import keypoint, pose
# import numpy as np
import threading

# Load a model
model = YOLO('/home/nvidia/xjb/ultralytics-main/weights/yolov8n-pose.pt')  # load an official model
img = None

def callback(imgmsg):
    global img
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
    delay = rospy.Time.now().to_sec() - imgmsg.header.stamp.to_sec()
    print("传图延时=", delay)
    print("---------------------------\n")


def pose_detect():
    global img
    global model
    print('enter detect')
    while True:
        if img is None:
            continue
        results = model.predict(source=img, half=True, show=False)  # predict on an image, stream=True
        pose_result = pose()
        pose_result.header.stamp = rospy.Time.now()
        for result in results:
            # print(result.keypoints)
            pose_result.shape = result.orig_shape
            for kp in result.keypoints:
                # print(result.boxes.xyxyn.tolist())
                keypoint_msg = keypoint()
                xyn = kp.xyn.cpu().numpy()[0]
                # print(xyn)
                keypoint_msg.x = xyn[:, 0]
                keypoint_msg.y = xyn[:, 1]
                keypoint_msg.conf = kp.conf.tolist()[0]
                pose_result.keypoints.append(keypoint_msg)
        
        pose_result_pub.publish(pose_result)
        img = None

if __name__ == "__main__":
    rospy.init_node('yolov8_pose')
    sub = rospy.Subscriber("/xavier_camera_image", Image, callback)
    pose_result_pub = rospy.Publisher("pose_result", pose, queue_size=1)

    # 创建子线程执行pose函数
    t = threading.Thread(target=pose_detect, daemon=True)
    t.start()
    rospy.spin()