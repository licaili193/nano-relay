#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include <signal.h>
#include <unistd.h>

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "nvmpi.h"
#include "udp_send_socket.h"

DEFINE_string(foreign_addr, "127.0.0.1", "Foreign address");
DEFINE_int32(foreign_port, 6000, "Foreign port");
DEFINE_int32(camera, 0, "Camera number");

constexpr int width = 640 * 3;
constexpr int height = 480;
constexpr size_t bitrate = 800000;
constexpr size_t idr_interval = 10;
constexpr float framerate = 10.f;

std::atomic_bool running;

void myHandler(int s){
  running.store(false);
}


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = myHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  // Video capture
  cv::VideoCapture cap;
  if (!cap.open(FLAGS_camera)) {
    LOG(FATAL) << "Cannot open camera";
  }

  // Create codec
  nvEncParam param;
  param.width = width;
  param.height = height;
  param.profile = 0; // V4L2_MPEG_VIDEO_H265_PROFILE_MAIN
  param.level = 3; // V4L2_MPEG_VIDEO_H265_LEVEL_2_0_HIGH_TIER
  param.bitrate = bitrate;
  param.peak_bitrate = bitrate;
	param.enableLossless = 0;
	param.mode_vbr = 0; // V4L2_MPEG_VIDEO_BITRATE_MODE_CBR
	param.insert_spspps_idr = 1;
	param.iframe_interval = idr_interval;
	param.idr_interval = idr_interval;
	param.fps_n = static_cast<unsigned int>(framerate);
	param.fps_d = 1;
	param.capture_num = 1;
	param.max_b_frames = 0;
	param.refs = 0;
	param.qmax = -1;
	param.qmin = -1;
	param.hw_preset_type = 1; // V4L2_ENC_HW_PRESET_ULTRAFAST
  param.use_extend_color_format = false;
  param.vbv_size = 0;

  nvmpictx* ctx = nvmpi_create_encoder(NV_VIDEO_CodingHEVC, &param);
  nvPacket packet;

  // Create socket
  std::unique_ptr<relay::communication::UDPSendSocket> modi_sock =  
      std::unique_ptr<relay::communication::UDPSendSocket>(
          new relay::communication::UDPSendSocket(0, 10, FLAGS_foreign_addr, FLAGS_foreign_port));

  running.store(true);

  while (running.load()) {
    auto start = std::chrono::system_clock::now();

    // Capture frame
    cv::Mat mat;
    cap >> mat;        
    if (mat.rows <= 0 || mat.cols <= 0) {
      break;
    }

    // Process frame
    double aspect_ratio = 
        static_cast<double>(mat.cols) / static_cast<double>(mat.rows);
    if (aspect_ratio > (double)width / (double)height) {
      double scale = (double)height / static_cast<double>(mat.rows);  
      cv::resize(
          mat, mat, cv::Size(0, 0), scale, scale, cv::INTER_LINEAR);
      mat = mat(cv::Rect((mat.cols - width) / 2, 0, width, height));
    } else {
      double scale = (double)width / static_cast<double>(mat.cols);  
      cv::resize(
          mat, mat, cv::Size(0, 0), scale, scale, cv::INTER_LINEAR);
      mat = mat(cv::Rect(0, (mat.rows - height) / 2, width, height));
    }

    cv::cvtColor(mat, mat, CV_BGR2YUV_I420);

    // Encode frame
    nvFrame frame;
    memset(&frame, 0, sizeof(nvFrame));
    frame.payload[0] = const_cast<unsigned char*>(mat.data);
    frame.payload_size[0] = width * height;
    frame.payload[1] = const_cast<unsigned char*>(mat.data + width * height); 
    frame.payload_size[1] = width * height / 4;
    frame.payload[2] = const_cast<unsigned char*>(mat.data + width * height * 5 / 4); 
    frame.payload_size[2] = width * height / 4;    
    frame.flags = 0;
    frame.type = NV_PIX_YUV420;
    frame.width = width;
    frame.height = height;

    int ret = nvmpi_encoder_put_frame(ctx, &frame);

    while (true) {
      memset(&packet, 0, sizeof(nvPacket));
      if (nvmpi_encoder_get_packet(ctx, &packet) == 0) {
        size_t result_size = packet.payload_size;
        modi_sock->push(result_size, reinterpret_cast<char*>(packet.payload));
      } else {
        break;
      }
    }

    std::this_thread::sleep_until(start + std::chrono::milliseconds(100));
  }

  return 0;
}