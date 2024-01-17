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
    std_msgs::Header image_header;
    image_header.seq = 0;
    image_header.frame_id = "earth"; // 我这里选择earth类型的坐标系
    while (nh.ok()) {
        capture.read(frame);
        if (!frame.empty()) {
            image_header.seq++;
            image_header.stamp = ros::Time::now();
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(image_header, "bgr8", frame).toImageMsg();
            pub.publish(msg);
        }
        ros::spinOnce();
        loop_rate.sleep();
    }

    capture.release(); //释放相机捕获对象
    return 0;
}
