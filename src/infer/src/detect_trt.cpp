#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <thread>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>
#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"
// ros msg头文件
#include "yolov8/yolo.h"
#include "yolov8/Box.h"

// 使用image/compressed压缩图像的命令为：
    // rosrun my_image_transport image_subscriber _image_transport:=compressed

cv::Mat img;
cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
ros::Publisher yolov8_trt_result_pub;

static const char *cocolabels[] = {"person",        "bicycle",      "car",
                                   "motorcycle",    "airplane",     "bus",
                                   "train",         "truck",        "boat",
                                   "traffic light", "fire hydrant", "stop sign",
                                   "parking meter", "bench",        "bird",
                                   "cat",           "dog",          "horse",
                                   "sheep",         "cow",          "elephant",
                                   "bear",          "zebra",        "giraffe",
                                   "backpack",      "umbrella",     "handbag",
                                   "tie",           "suitcase",     "frisbee",
                                   "skis",          "snowboard",    "sports ball",
                                   "kite",          "baseball bat", "baseball glove",
                                   "skateboard",    "surfboard",    "tennis racket",
                                   "bottle",        "wine glass",   "cup",
                                   "fork",          "knife",        "spoon",
                                   "bowl",          "banana",       "apple",
                                   "sandwich",      "orange",       "broccoli",
                                   "carrot",        "hot dog",      "pizza",
                                   "donut",         "cake",         "chair",
                                   "couch",         "potted plant", "bed",
                                   "dining table",  "toilet",       "tv",
                                   "laptop",        "mouse",        "remote",
                                   "keyboard",      "cell phone",   "microwave",
                                   "oven",          "toaster",      "sink",
                                   "refrigerator",  "book",         "clock",
                                   "vase",          "scissors",     "teddy bear",
                                   "hair drier",    "toothbrush"};

static const char *visdronelabels[] = {"pedestrian", "person", "car", "van", "bus", "truck", "motor",
                                      "bicycle", "awning-tricycle", "tricycle"};

std::vector<std::string> labels(cocolabels, cocolabels + sizeof(cocolabels) / sizeof(cocolabels[0]));

inline yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

// 线程函数，执行推理
void runInference(bool imshow = false)
{
    std::cout << "enter detect" << std::endl;

    while (true)
    {
        if (img.empty()) {
            // std::cout << "img is empty" << std::endl;
            continue;
        }

        // 执行推理
        auto results = cpmi.commit(cvimg(img)).get();
        std::cout << results.size() << " results" << std::endl;

        // 发布检测结果
        yolov8::yolo yolo_result;
        yolo_result.header.stamp = ros::Time::now();
        yolo_result.shape = {img.cols, img.rows};
        yolo_result.class_names = labels;
        for (const auto &box : results)
        {
            // std::cout << box.left << " " << box.top << " " << box.right << " " << box.bottom << " " << box.confidence << " " << box.class_label << std::endl;
            yolov8::Box yolo_Box;
            yolo_Box.x1 = box.left / img.cols;
            yolo_Box.y1 = box.top / img.rows;
            yolo_Box.x2 = box.right / img.cols;
            yolo_Box.y2 = box.bottom / img.rows;
            yolo_Box.conf = box.confidence;
            yolo_Box.cls = box.class_label;
            yolo_result.boxes.push_back(yolo_Box);
        }
        yolov8_trt_result_pub.publish(yolo_result);

        // 显示结果
        if (imshow)
        {
            cv::Mat show_img = img.clone();
            for (const auto &box : results)
            {
                cv::rectangle(show_img, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom),
                              cv::Scalar(0, 0, 255), 2);
                cv::putText(show_img, cv::format("%s %.2f", cocolabels[box.class_label], box.confidence),
                            cv::Point(box.left, box.top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
            }
            cv::imshow("detect_trt_result", show_img);
            cv::waitKey(1);
        }
        img.release();
    }
}

// 图像订阅回调函数
void imageCallback(const sensor_msgs::ImageConstPtr& msg, bool imshow=false)
{
  try
  {
    img = cv_bridge::toCvShare(msg, "bgr8")->image;
    if (img.empty()) {
      ROS_ERROR("image is empty!!");
      return;
    }

    // 显示延时
    double time_now = ros::Time::now().toSec();
    // float delay = (time_now.nsec - msg->header.stamp.nsec) / 1.0 / 1e9;
    double delay = ros::Time::now().toSec() - msg->header.stamp.toSec();
    ROS_INFO("Time:%d.%d; Delay:%f; Seq:%d; Frame:%s:\n\t",msg->header.stamp.sec,msg->header.stamp.nsec, delay, 
        msg->header.seq,msg->header.frame_id.c_str());
    
    if (imshow) {
      cv::imshow("view", img);
    }
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "yolov8_trt_inference_node");
    ros::NodeHandle nh;

    // 初始化参数
    std::string model = "yolov8m-640-fp16"; // 模型名称
    yolo::Type type = yolo::Type::V8;        // yolo版本
    bool imshow = false;                      // 是否显示图像
    if (argc > 1 && argv[1]) {
        model = std::string(argv[1]);
    }
    if (argc > 2 && argv[2] && (strcmp(argv[2], "v7") == 0 || strcmp(argv[2], "7") == 0)) {
        type = yolo::Type::V7;
    }
    if (argc > 3 && (strcmp(argv[3], "imshow") == 0 || strcmp(argv[3], "show") == 0)) {
        imshow = true;
    }
    // if (imshow) {
    //     cv::namedWindow("detect_trt_result");
    //     cv::startWindowThread();
    // }

    // 初始化模型
    std::string model_file = cv::format("/home/nvidia/xjb/algorithm_ws/src/infer/workspace/%s.engine", model.c_str());
    int max_infer_batch = 1;
    bool ok = cpmi.start([model_file, type]() { return yolo::load(model_file, type); }, max_infer_batch);
    if (!ok) {
        ROS_ERROR("Failed to load model %s", model_file.c_str());
        return -1;
    }

    // 订阅图像数据
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub = it.subscribe("/xavier_camera_image", 1, boost::bind(&imageCallback, _1, imshow));
    // ros::Subscriber image_sub = nh.subscribe("/xavier_camera_image", 1, boost::bind(&imageCallback, _1, imshow));
    // 发布检测结果
    yolov8_trt_result_pub = nh.advertise<yolov8::yolo>("/yolov8_trt/result", 1);

    // 创建新线程来运行推理
    std::thread inferenceThread(runInference, imshow);
    inferenceThread.detach();

    ros::spin();

    // if (imshow)
    //     cv::destroyWindow("detect_trt_result");

    return 0;
}

