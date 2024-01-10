#!/usr/bin/env python
#coding=UTF-8
# license removed for brevity
import os
import rospy
import sys
from sensor_msgs.msg import Image
# sys.path.remove('/opt/ros//lib/python2.7/dist-packages')
import cv2
from cv_bridge import CvBridge

def talker():
    pub = rospy.Publisher('/xavier_camera_image', Image, queue_size=1)
    rospy.init_node('image_publisher_py', anonymous=True)
    rate = rospy.Rate(30)
    bridge = CvBridge()
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    #  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("图像的宽度=", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("图像的高度=", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("帧率=", cap.get(cv2.CAP_PROP_FPS))
    while not rospy.is_shutdown():
        ret, img = cap.read()
        # cv2.imshow("talker", img)
        # cv2.waitKey(3)
        imgmsg = bridge.cv2_to_imgmsg(img, "bgr8")
        imgmsg.header.stamp = rospy.Time.now()
        pub.publish(imgmsg)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

