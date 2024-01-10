#!/usr/bin/env python3
#coding=UTF-8
import sys
import os
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from yolov8.msg import yolo
from yolov8.msg import Box

# img = cv2.image
bridge = CvBridge()

def img_sub_callback(imgmsg):
    # img = image()
    global img
    img = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
    # cv2.imshow("listener", img)
    # cv2.waitKey(3)


def yolo_result_pub_callback(yolo_result_msg):
    global img
    if img is None:
        return
    print('当前时间：', rospy.Time.now().to_sec(), '   发送时间：', yolo_result_msg.header.stamp.to_sec())
    boxes = yolo_result_msg.boxes
    for box in boxes:
        x1 = int(box.x1 * img.shape[1])
        y1 = int(box.y1 * img.shape[0])
        x2 = int(box.x2 * img.shape[1])
        y2 = int(box.y2 * img.shape[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, yolo_result_msg.class_names[box.cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.imshow("listener", img)
    # cv2.waitKey(1)


if __name__ == "__main__":
    try:
        rospy.init_node('test_img_sub')

        sub = rospy.Subscriber("/xavier_camera_image", Image, img_sub_callback)

        yolo_result_sub = rospy.Subscriber("yolo_result", yolo, yolo_result_pub_callback)

        rospy.spin()
    except rospy.ROSInterruptException:
        pass