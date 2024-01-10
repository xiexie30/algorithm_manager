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
from yolov8.msg import yolo
from yolov8.msg import Box
# import numpy as np

# Load a model
model = YOLO('/home/nvidia/xjb/ultralytics-main/weights/yolov8m.pt')  # load an official model

def callback(imgmsg):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
    # cv2.imshow("listener", img)
    # cv2.waitKey(3)

    # Predict with the model
    # results = model.predict(source=img, half=True, show=False)[0]
    # results = model.predict(source=img, half=True, show=False)
    # print(results)
    print('enter callback: ', rospy.Time.now().to_sec(), '   ', imgmsg.header.stamp.to_sec())
    results = model.predict(source=img, half=True, show=False)  # predict on an image, stream=True
    yolo_result = yolo()
    yolo_result.header.stamp = rospy.Time.now()
    for result in results:
        yolo_result.shape = result.orig_shape
        yolo_result.class_names = list(result.names.values())
        for box in result.boxes:
            # print(result.boxes.xyxyn.tolist())
            yolo_Box = Box()
            yolo_Box.x1 = float(box.xyxyn.tolist()[0][0])
            yolo_Box.y1 = float(box.xyxyn.tolist()[0][1])
            yolo_Box.x2 = float(box.xyxyn.tolist()[0][2])
            yolo_Box.y2 = float(box.xyxyn.tolist()[0][3])
            yolo_Box.conf = float(box.conf.item())
            # print(box.cls.item())
            yolo_Box.cls = int(box.cls.item())
            # yolo_Box.id = 0
            # print(type(cls))
            # print(cls.dtype)
            yolo_result.boxes.append(yolo_Box)#获取检测结果
    
    yolo_result_pub.publish(yolo_result) #发布检测结果

    print(imgmsg.header)
    print(rospy.Time.now().to_sec(), '   ', imgmsg.header.stamp.to_sec())
    delay = rospy.Time.now().to_sec() - imgmsg.header.stamp.to_sec()
    print("传图延时=", delay)
    print("---------------------------\n")


if __name__ == "__main__":
    rospy.init_node('yolov8')
    sub = rospy.Subscriber("/xavier_camera_image", Image, callback)

    yolo_result_pub = rospy.Publisher("yolo_result", yolo, queue_size=1)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()