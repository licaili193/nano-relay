
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include <stdio.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>
#include <signal.h>
#include <poll.h>

#include "NvUtils.h"
#include "nvbuf_utils.h"
#include "NvEglRenderer.h"

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cudaEGL.h"

#include "nvmpi.h"
#include "udp_send_socket.h"
#include "image_processing.h"

#include "camera_grabber.h"

// TODO: clean the include files

DEFINE_string(foreign_addr, "127.0.0.1", "Foreign address");
DEFINE_int32(foreign_port, 6000, "Foreign port");
DEFINE_string(camera, "/dev/video0", "Camera name");

constexpr int width = 640;
constexpr int height = 480;
constexpr size_t bitrate = 800000;
constexpr size_t idr_interval = 10;
constexpr float framerate = 30.f;

std::atomic_bool running;

void myHandler(int s){
  running.store(false);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  struct sigaction sig_action;
  struct pollfd fds[1];
  NvBufferTransformParams transParams;

  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = myHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

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

  nvmpictx* nvm_ctx = nvmpi_create_encoder(NV_VIDEO_CodingHEVC, &param);
  nvPacket packet;

  // Create socket
  std::unique_ptr<relay::communication::UDPSendSocket> modi_sock =  
      std::unique_ptr<relay::communication::UDPSendSocket>(
          new relay::communication::UDPSendSocket(0, 10, FLAGS_foreign_addr, FLAGS_foreign_port));

  auto d_img_yuyv = image_processing::allocateImageYUYV(width, height);
  CHECK(d_img_yuyv);
  auto d_img_yuv420 = image_processing::allocateImageYUV(width, height);
  CHECK(d_img_yuv420);
  auto d_img_yuv420_s = image_processing::allocateImageYUV(width, height);
  CHECK(d_img_yuv420_s);
  unsigned char* img_yuv420 = new unsigned char[width * height * 3 / 2];
  CHECK(img_yuv420);

  CameraGrabber grabber(d_img_yuyv, FLAGS_camera, width, height);
  
  running.store(true);
  grabber.startStream();

  /* Wait for camera event with timeout = 5000 ms */
  while (running.load()) {
    if (!grabber.newImage()) {
      continue;
    }

    image_processing::yuyv2YUV(width, height, d_img_yuyv, d_img_yuv420);
    image_processing::shuffleYUV(width, height, d_img_yuv420, d_img_yuv420_s);
    image_processing::downloadImageYUV(width, height, d_img_yuv420_s, img_yuv420);

    // Encode frame
    nvFrame frame;
    memset(&frame, 0, sizeof(nvFrame));
    frame.payload[0] = img_yuv420;
    frame.payload_size[0] = width * height;
    frame.payload[1] = img_yuv420 + width * height; 
    frame.payload_size[1] = width * height / 4;
    frame.payload[2] = img_yuv420 + width * height * 5 /4; 
    frame.payload_size[2] = width * height / 4;    
    frame.flags = 0;
    frame.type = NV_PIX_YUV420;
    frame.width = width;
    frame.height = height;

    int ret = nvmpi_encoder_put_frame(nvm_ctx, &frame);

    while (true) {
      memset(&packet, 0, sizeof(nvPacket));
      if (nvmpi_encoder_get_packet(nvm_ctx, &packet) == 0) {
        size_t result_size = packet.payload_size;
        modi_sock->push(result_size, reinterpret_cast<char*>(packet.payload));
      } else {
        break;
      }
    }
  }

  grabber.stopStream();
  cudaFree(d_img_yuyv);
  cudaFree(d_img_yuv420);
  cudaFree(d_img_yuv420_s);
  delete[] img_yuv420;

  LOG(INFO) << "Finished cleanly";

  return 0;
}