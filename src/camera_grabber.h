#ifndef __CAMERA_GRABBER__
#define __CAMERA_GRABBER__

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

#define V4L2_BUFFERS_NUM 4

namespace {
struct nv_buffer {
    /* User accessible pointer */
    unsigned char* start;
    /* Buffer length */
    unsigned int size;
    /* File descriptor of NvBuffer */
    int dmabuff_fd;
};

struct context_t {
    /* Camera v4l2 context */
    char cam_devname[256];
    char cam_file[16];
    int cam_fd = -1;
    unsigned int cam_pixfmt = V4L2_PIX_FMT_YUYV;
    unsigned int cam_w = 640;
    unsigned int cam_h = 480;
    unsigned int frame = 0;
    unsigned int save_n_frame = 0;

    /* Global buffer ptr */
    nv_buffer* g_buff = nullptr;
    bool capture_dmabuf = true;

    int render_dmabuf_fd;
    EGLDisplay egl_display = EGL_NO_DISPLAY;
    EGLImageKHR egl_image = NULL;
};

/* Correlate v4l2 pixel format and NvBuffer color format */
struct nv_color_fmt {
    unsigned int v4l2_pixfmt;
    NvBufferColorFormat nvbuff_color;
};

static nv_color_fmt nvcolor_fmt[] =
{
    /* TODO: add more pixel format mapping */
    {V4L2_PIX_FMT_UYVY, NvBufferColorFormat_UYVY},
    {V4L2_PIX_FMT_VYUY, NvBufferColorFormat_VYUY},
    {V4L2_PIX_FMT_YUYV, NvBufferColorFormat_YUYV},
    {V4L2_PIX_FMT_YVYU, NvBufferColorFormat_YVYU},
    {V4L2_PIX_FMT_GREY, NvBufferColorFormat_GRAY8},
    {V4L2_PIX_FMT_YUV420M, NvBufferColorFormat_YUV420},
};

NvBufferColorFormat get_nvbuff_color_fmt(unsigned int v4l2_pixfmt) {
    unsigned i;

    for (i = 0; i < sizeof(nvcolor_fmt) / sizeof(nvcolor_fmt[0]); i++)
    {
        if (v4l2_pixfmt == nvcolor_fmt[i].v4l2_pixfmt)
            return nvcolor_fmt[i].nvbuff_color;
    }

    return NvBufferColorFormat_Invalid;
}

bool request_camera_buff(context_t *ctx) {
    /* Request camera v4l2 buffer */
    struct v4l2_requestbuffers rb;
    memset(&rb, 0, sizeof(rb));
    rb.count = V4L2_BUFFERS_NUM;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_DMABUF;
    if (ioctl(ctx->cam_fd, VIDIOC_REQBUFS, &rb) < 0) {
        LOG(FATAL) << "Failed to request v4l2 buffers: " << strerror(errno);
    }
    if (rb.count != V4L2_BUFFERS_NUM) {
        LOG(FATAL) << "V4l2 buffer number is not as desired";
    }

    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++)
    {
        struct v4l2_buffer buf;

        /* Query camera v4l2 buf length */
        memset(&buf, 0, sizeof buf);
        buf.index = index;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_DMABUF;

        if (ioctl(ctx->cam_fd, VIDIOC_QUERYBUF, &buf) < 0) {
            LOG(FATAL) << "Failed to query buff: " << strerror(errno);
        }

        /* TODO: add support for multi-planer
           Enqueue empty v4l2 buff into camera capture plane */
        buf.m.fd = (unsigned long)ctx->g_buff[index].dmabuff_fd;
        ctx->g_buff[index].size = buf.length;

        if (ioctl(ctx->cam_fd, VIDIOC_QBUF, &buf) < 0) {
            LOG(FATAL) << "Failed to enqueue buffers: " << strerror(errno);
        }
    }

    return true;
}

bool request_camera_buff_mmap(context_t *ctx) {
    /* Request camera v4l2 buffer */
    struct v4l2_requestbuffers rb;
    memset(&rb, 0, sizeof(rb));
    rb.count = V4L2_BUFFERS_NUM;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_MMAP;
    if (ioctl(ctx->cam_fd, VIDIOC_REQBUFS, &rb) < 0) {
        LOG(FATAL) << "Failed to request v4l2 buffers: " << strerror(errno);
    }
    if (rb.count != V4L2_BUFFERS_NUM) {
        LOG(FATAL) << "V4l2 buffer number is not as desired";
    }

    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++) {
        struct v4l2_buffer buf;

        /* Query camera v4l2 buf length */
        memset(&buf, 0, sizeof buf);
        buf.index = index;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(ctx->cam_fd, VIDIOC_QUERYBUF, &buf) < 0) {
            LOG(FATAL) << "Failed to query buff: " << strerror(errno);
        }

        ctx->g_buff[index].size = buf.length;
        ctx->g_buff[index].start = (unsigned char *)
            mmap (NULL /* start anywhere */,
                    buf.length,
                    PROT_READ | PROT_WRITE /* required */,
                    MAP_SHARED /* recommended */,
                    ctx->cam_fd, buf.m.offset);
        if (MAP_FAILED == ctx->g_buff[index].start)
            LOG(FATAL) << "Failed to map buffers";

        if (ioctl(ctx->cam_fd, VIDIOC_QBUF, &buf) < 0)
            LOG(FATAL) << "Failed to enqueue buffers: " << strerror(errno);
    }

    return true;
}
}

class CameraGrabber {
 public:
  CameraGrabber(CUcontext* cuda_ctx, uint8_t* d_img_yuyv, std::string camera_name, int width, int height, size_t panel_width, size_t panel_offset) {
    cuda_ctx_ = cuda_ctx;
    camera_name_ = camera_name;
    // TODO: unify the setting with camera name
    context_.cam_w = width;
    context_.cam_h = height;
    panel_width_ = panel_width;
    panel_offset_ = panel_offset;
    d_img_yuyv_ = d_img_yuyv;
    running_.store(false);
    new_image_.store(false);
  }

  ~CameraGrabber() {
    if (t_.joinable()) {
      t_.join();
    }
  }

  void startStream() {
    std::thread t(&CameraGrabber::worker, this);
    std::swap(t, t_);
  }

  void stopStream() {
    running_.store(false);
  }

  bool newImage() {
    bool res = new_image_.load();
    if (res) {
      new_image_.store(false);
    }
    return res;
  }

 private:
  bool initializeCamera(std::string camera_name);

  void worker();

  bool prepareBuffers();

  void handleEglImage(void *pEGLImage, size_t width, size_t height);

  void cleanup();

  CUcontext* cuda_ctx_ = nullptr;

  context_t context_;
  uint8_t* d_img_yuyv_ = nullptr;
  std::string camera_name_ = "/dev/video0";

  size_t panel_width_ = 640;
  size_t panel_offset_ = 0;

  std::atomic_bool running_;
  std::thread t_;
  std::atomic_bool new_image_;
};

#endif
