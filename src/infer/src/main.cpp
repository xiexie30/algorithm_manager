
#include <opencv2/opencv.hpp>

#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"

using namespace std;

shared_future<yolo::BoxArray> prev_future;

const int width = 640;
const int height = 640;
const int outputH = 640;

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

yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

static double timestamp_now_float() {
    return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
}

void perf(const string& model, yolo::Type type) {
  printf("prtf begin!!\n");
  int max_infer_batch = 16;
  int batch = 16;
  std::vector<cv::Mat> images{cv::imread("inference/car.jpg"), cv::imread("inference/gril.jpg"),
                              cv::imread("inference/group.jpg")};

  for (int i = images.size(); i < batch; ++i) images.push_back(images[i % 3]);

  string model_file = cv::format("%s.engine", model.c_str());
  cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
  bool ok = cpmi.start([model_file, type] { return yolo::load(model_file, type); },
                       max_infer_batch);

  if (!ok) return;

  std::vector<yolo::Image> yoloimages(images.size());
  std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);

  trt::Timer timer;
  for (int i = 0; i < 20; ++i) {
    timer.start();
    cpmi.commits(yoloimages).back().get();
    timer.stop("BATCH16");
  }

  for (int i = 0; i < 20; ++i) {
    timer.start();
    cpmi.commit(yoloimages[0]).get();
    timer.stop("BATCH1");
  }
}

void batch_inference(const string& model, yolo::Type type) {
  printf("batch_inference!!\n");
  std::vector<cv::Mat> images{cv::imread("inference/car.jpg"), cv::imread("inference/gril.jpg"),
                              cv::imread("inference/group.jpg")};
  string model_file = cv::format("%s.engine", model.c_str());
  auto yolo = yolo::load(model_file, type);
  if (yolo == nullptr) return;

  std::vector<yolo::Image> yoloimages(images.size());
  std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);
  auto batched_result = yolo->forwards(yoloimages);
  for (int ib = 0; ib < (int)batched_result.size(); ++ib) {
    auto &objs = batched_result[ib];
    auto &image = images[ib];
    for (auto &obj : objs) {
      uint8_t b, g, r;
      tie(b, g, r) = yolo::random_color(obj.class_label);
      cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                    cv::Scalar(b, g, r), 5);

      auto name = cocolabels[obj.class_label];
      auto caption = cv::format("%s %.2f", name, obj.confidence);
      int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
      cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                    cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
      cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
                  16);
    }
    printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
    cv::imwrite(cv::format("Result%d.jpg", ib), image);
  }
}

void single_inference(const string& model, yolo::Type type) {
  printf("single_inference!!\n");
  cv::Mat image = cv::imread("inference/car.jpg");
  string model_file = cv::format("%s.engine", model.c_str());
  auto yolo = yolo::load(model_file, type);
  if (yolo == nullptr) return;

  auto objs = yolo->forward(cvimg(image));
  int i = 0;
  for (auto &obj : objs) {
    uint8_t b, g, r;
    tie(b, g, r) = yolo::random_color(obj.class_label);
    cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                  cv::Scalar(b, g, r), 5);

    auto name = cocolabels[obj.class_label];
    auto caption = cv::format("%s %.2f", name, obj.confidence);
    int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                  cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
    cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);

    if (obj.seg) {
      cv::imwrite(cv::format("%d_mask.jpg", i),
                  cv::Mat(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data));
      i++;
    }
  }

  printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
  cv::imwrite("Result.jpg", image);
}

void cam_detect(const string& model, yolo::Type type, const char *show) {
  printf("Camera detect begin!!\n");
  //printf("%s\n", show);
  bool viewImg = false;
  if (show && strcmp(show, "view-img") == 0) viewImg = true;
  
  cv::VideoCapture cap(-1);
  //cv::VideoCapture cap("rtsp://admin:scuimage508@192.168.1.7:554/");
  //cout << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "  " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << endl;
  cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
  //cap.set(cv::CAP_PROP_FPS, 30);
  //cout << cap.get(cv::CAP_PROP_FPS);
  if(!cap.isOpened()) {
      printf("camera open failed\n");
      return;
  }
  else {
      printf("camera is opened\n");
  }
  string model_file = cv::format("%s.engine", model.c_str());
  auto yolo = yolo::load(model_file, type);
  if (yolo == nullptr) return;

  cv::Mat image;
  int nnn = 0;
  while (nnn < 1000) {
      nnn++;
      cap.read(image);
      //cout << image.cols << "   " << image.rows << endl;
      if (image.empty()) continue;
      
      auto begin_timer = timestamp_now_float();
      auto objs = yolo->forward(cvimg(image));
      float inference_time = timestamp_now_float() - begin_timer;
      
      for (auto &obj : objs) {
        // printf("class label: %d, conf: %.2f, left: %.2f, top: %.2f, right: %.2f, bottom: %.2f\n", obj.class_label, obj.left, obj.top, obj.right, obj.bottom);
        uint8_t b, g, r;
        tie(b, g, r) = yolo::random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                      cv::Scalar(b, g, r), round(outputH * 0.005));
      
        auto name = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                      cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), ceil(outputH * 0.001), 16);
      }

      auto time1 = timestamp_now_float() - begin_timer;
      if (viewImg && !image.empty()) {
        cv::imshow("frame", image);
        cv::waitKey(1);
        //if(cv::waitKey(1) >= 0) {
        //    break;
        //}
      }
      printf("%d objects, inference time per image: %.2f ms, %.2f ms, total: %.2f ms, fps = %.2f \n", objs.size(), inference_time, time1, timestamp_now_float() - begin_timer, (1 / inference_time) * 1000);
  }
}

void detect_save(const string& model, yolo::Type type, const char *save) {
  printf("Save result!!\n");
  bool saveVid = false;
  if (save && strcmp(save, "save") == 0) saveVid = true;
  
  string cap_path = "../videos/" + to_string(outputH) + ".mp4";
  string outputVideoPath = "../videos/" + to_string(outputH) + "-" + model + ".avi";
  cv::VideoCapture cap(cap_path);
  cv::VideoWriter outputVideo;
  outputVideo.open(outputVideoPath, cv::VideoWriter::fourcc('D','I','V','X'), 30.0, cv::Size(width, outputH));
  if(!cap.isOpened()) {
      printf("VideoCapture open failed\n");
      return;
  }
  else {
      printf("VideoCapture is opened\n");
  }
  
  string model_file = cv::format("%s.engine", model.c_str());
  auto yolo = yolo::load(model_file, type);
  if (yolo == nullptr) return;
  cv::Mat image;
  while (cap.isOpened()) {
      cap.read(image);
      if (image.empty()) {
        cap.release();
        continue;
      }
      
      auto begin_timer = timestamp_now_float();
      auto objs = yolo->forward(cvimg(image));
      float inference_time = timestamp_now_float() - begin_timer;
      
      for (auto &obj : objs) {
        // printf("class label: %d, conf: %.2f, left: %.2f, top: %.2f, right: %.2f, bottom: %.2f\n", obj.class_label, obj.left, obj.top, obj.right, obj.bottom);
        uint8_t b, g, r;
        tie(b, g, r) = yolo::random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                      cv::Scalar(b, g, r), round(outputH * 0.004));
      
        auto name = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        float scalar = image.rows * 0.001;
        int width = cv::getTextSize(caption, 0, scalar, 2, nullptr).width + 2;
        cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 23),
                      cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
        int thickness = ceil(image.rows * 0.001);
        int baseLine = ceil(image.rows * 0.01 / 8) * 8;
        cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, scalar, cv::Scalar::all(0), thickness, 16);
      }

      auto time1 = timestamp_now_float() - begin_timer;
      if (saveVid && !image.empty()) {
        outputVideo.write(image);
        //cv::imshow("frame", image);
        //cv::waitKey(1);
        //if(cv::waitKey(1) >= 0) {
        //    break;
        //}
      }
      printf("%d objects, inference time per image: %.2f ms, %.2f ms, total: %.2f ms, fps = %.2f \n", objs.size(), inference_time, time1, timestamp_now_float() - begin_timer, (1 / inference_time) * 1000);
  }
  outputVideo.release();
}


void cam_detect_high_perf(const string& model, yolo::Type type, const char *show) {
  printf("Camera detect begin!!\n");
  //printf("%s\n", show);
  bool viewImg = false;
  if (show && strcmp(show, "view-img") == 0) viewImg = true;
  
  cv::VideoCapture cap(0);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
  //cap.set(cv::CAP_PROP_FPS, 30);
  //cout << cap.get(cv::CAP_PROP_FPS);
  if(!cap.isOpened()) {
      printf("camera open failed\n");
      return;
  }
  else {
      printf("camera is opened\n");
  }
  
  int max_infer_batch = 1;
  //int batch = 1;
  cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
  bool ok = cpmi.start([] { return yolo::load("yolov7-s2-fp16.engine", yolo::Type::V7); },
                       max_infer_batch);
                       
  cv::Mat image;
  cv::Mat prev_image;
  int nnn = 0;
  while (nnn < 1000) {
      nnn++;
      cap.read(image);
      
      if (prev_future.valid()) {
        auto begin_timer = timestamp_now_float();
        auto objs = prev_future.get();
        float inference_time = timestamp_now_float() - begin_timer;
        
        for (auto &obj : objs) {
          // printf("class label: %d, conf: %.2f, left: %.2f, top: %.2f, right: %.2f, bottom: %.2f\n", obj.class_label, obj.left, obj.top, obj.right, obj.bottom);
          uint8_t b, g, r;
          tie(b, g, r) = yolo::random_color(obj.class_label);
          cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                        cv::Scalar(b, g, r), round(outputH * 0.005));
        
          auto name = cocolabels[obj.class_label];
          auto caption = cv::format("%s %.2f", name, obj.confidence);
          int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
          cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                        cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
          cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), ceil(outputH * 0.001), 16);
        }
  
        auto time1 = timestamp_now_float() - begin_timer;
        if (viewImg) {
          cv::imshow("frame", image);
          if(cv::waitKey(1) >= 0) {
              break;
          }
        }
        printf("%d objects, inference time per image: %.2f ms, %.2f ms, total: %.2f ms, fps = %.2f \n", objs.size(), inference_time, time1, timestamp_now_float() - begin_timer, (1 / inference_time) * 1000);
      }
      image.copyTo(prev_image);
      prev_future = cpmi.commit(cvimg(image));
  }
}


int main(int argc, const char *argv[]) {
  // initLibNvInferPlugins(nullptr, "");
  // bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
  string mode;
  string model = "prune-db-yolov8m-640-fp16";
  yolo::Type type = yolo::Type::V8;
  //if (argc > 1 && strcmp(argv[1], "save") == 0) 
  //    detect_save(model, yolo::Type::V8, argv[1]);
  //else
  //    cam_detect(model, yolo::Type::V8, argv[1]);
  //cam_detect_high_perf("yolov7-s2-int8", yolo::Type::V7, argv[1]);
  //cam_detect("yolov7-tiny", yolo::Type::V7, argv[1]);

  // ./infer perf prune-db-yolov8m-640-fp16 v8
  // ./infer cam_detect yolov7-640-fp16 v7 view-img

  if (argv[1]) mode = string(argv[1]);
  if (argv[2]) model = string(argv[2]);
  if (argv[3] && (strcmp(argv[3], "v7") == 0 || strcmp(argv[3], "7") == 0)) type = yolo::Type::V7;

  if (mode == "perf") {
    perf(model, type);
  }
  else if (mode == "batch_infer") {
    batch_inference(model, type);
  }
  else if (mode == "single_infer") {
    single_inference(model, type);
  }
  else if (mode == "cam_detect") {
    cam_detect(model, type, argv[4]); // view-img
  }
  else if (mode == "cam_save") {
    detect_save(model, type, argv[4]); // view-img
  }
  else {
    return 0;
  }
  //batch_inference();
  //single_inference();
  return 0;
}
