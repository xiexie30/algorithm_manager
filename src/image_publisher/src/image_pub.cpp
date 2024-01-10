#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/imgcodecs/legacy/constants_c.h"

// TODO: 比较opencv encode和cv_bridge::CvImage两种方式哪个快

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_publisher");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("/xavier_camera_image", 1);

    cv::VideoCapture capture;
    capture.open(0, cv::CAP_V4L2);
    // capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G')); 
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640); //图像的宽，需要相机支持此宽
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480); //图像的高，需要相机支持此高
    capture.set(cv::CAP_PROP_FPS, 30);

    if (!capture.isOpened()) {
        ROS_ERROR("VideoCapture open failed!!!");
        return -1;
    }
    
    std::cout << "图像的宽度=" << capture.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "图像的高度=" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "帧率=" << capture.get(cv::CAP_PROP_FPS) << std::endl;
    
    cv::Mat frame;
    ros::Rate loop_rate(30);
    while (nh.ok()) {
        capture.read(frame);
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
        pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    capture.release(); //释放相机捕获对象
    return 0;
}
