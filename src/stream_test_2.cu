
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

DEFINE_string(foreign_addr, "127.0.0.1", "Foreign address");
DEFINE_int32(foreign_port, 6000, "Foreign port");
DEFINE_int32(camera, 0, "Camera number");

constexpr int width = 640;
constexpr int height = 480;
constexpr size_t bitrate = 800000;
constexpr size_t idr_interval = 10;
constexpr float framerate = 30.f;

#define V4L2_BUFFERS_NUM 4

std::atomic_bool running;

void myHandler(int s){
  running.store(false);
}

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
    const char* cam_devname = "/dev/video0";
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

bool camera_initialize(context_t* ctx) {
    struct v4l2_format fmt;

    /* Open camera device */
    ctx->cam_fd = open(ctx->cam_devname, O_RDWR);
    if (ctx->cam_fd == -1) {
        LOG(FATAL) << "Failed to open camera device" << strerror(errno);
    }

    /* Set camera output format */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = ctx->cam_w;
    fmt.fmt.pix.height = ctx->cam_h;
    fmt.fmt.pix.pixelformat = ctx->cam_pixfmt;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    if (ioctl(ctx->cam_fd, VIDIOC_S_FMT, &fmt) < 0) {
        LOG(FATAL) << "Failed to set camera output format" << strerror(errno);
    }

    /* Get the real format in case the desired is not supported */
    memset(&fmt, 0, sizeof fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_G_FMT, &fmt) < 0) {
        LOG(FATAL) << "Failed to get camera output format" << strerror(errno);
    }
    if (fmt.fmt.pix.width != ctx->cam_w ||
        fmt.fmt.pix.height != ctx->cam_h ||
        fmt.fmt.pix.pixelformat != ctx->cam_pixfmt) {
        LOG(WARNING) << "The desired format is not supported";
        ctx->cam_w = fmt.fmt.pix.width;
        ctx->cam_h = fmt.fmt.pix.height;
        ctx->cam_pixfmt =fmt.fmt.pix.pixelformat;
    }

    struct v4l2_streamparm streamparm;
    memset (&streamparm, 0x00, sizeof (struct v4l2_streamparm));
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(ctx->cam_fd, VIDIOC_G_PARM, &streamparm);

    LOG(INFO) << "Camera ouput format: " 
              << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height
              << " stride: " << fmt.fmt.pix.bytesperline 
              << " image size: " << fmt.fmt.pix.sizeimage
              << " framerate: " << streamparm.parm.capture.timeperframe.denominator << "/"
              << streamparm.parm.capture.timeperframe.numerator;

    return true;
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

bool prepare_buffers(context_t* ctx) {
    NvBufferCreateParams input_params = {0};

    /* Allocate global buffer context */
    ctx->g_buff = (nv_buffer*)malloc(V4L2_BUFFERS_NUM * sizeof(nv_buffer));
    if (ctx->g_buff == NULL) {
        LOG(FATAL) << "Failed to allocate global buffer context";
    }

    input_params.payloadType = NvBufferPayload_SurfArray;
    input_params.width = ctx->cam_w;
    input_params.height = ctx->cam_h;
    input_params.layout = NvBufferLayout_Pitch;

    /* Create buffer and provide it with camera */
    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++)
    {
        int fd;
        NvBufferParams params = {0};

        input_params.colorFormat = get_nvbuff_color_fmt(ctx->cam_pixfmt);
        input_params.nvbuf_tag = NvBufferTag_CAMERA;
        if (-1 == NvBufferCreateEx(&fd, &input_params)) {
            LOG(FATAL) << "Failed to create NvBuffer";
        }

        ctx->g_buff[index].dmabuff_fd = fd;

        if (-1 == NvBufferGetParams(fd, &params)) {
            LOG(FATAL) << "Failed to get NvBuffer parameters";
        }

        if (ctx->cam_pixfmt == V4L2_PIX_FMT_GREY &&
            params.pitch[0] != params.width[0]) {
          LOG(WARNING) << "Disabled DAM";
          ctx->capture_dmabuf = false;
        }

        /* TODO: add multi-planar support
           Currently only supports YUV422 interlaced single-planar */
        if (ctx->capture_dmabuf) {
            if (-1 == NvBufferMemMap(ctx->g_buff[index].dmabuff_fd, 0, NvBufferMem_Read_Write,
                        (void**)&ctx->g_buff[index].start)) {
                LOG(FATAL) << "Failed to map buffer";
            }
        }
    }

    input_params.colorFormat = get_nvbuff_color_fmt(V4L2_PIX_FMT_YUYV);
    input_params.nvbuf_tag = NvBufferTag_VIDEO_CONVERT;
    /* Create Render buffer */
    if (-1 == NvBufferCreateEx(&ctx->render_dmabuf_fd, &input_params)) {
        LOG(FATAL) << "Failed to create NvBuffer";
    }

    if (ctx->capture_dmabuf) {
        if (!request_camera_buff(ctx)) {
            LOG(FATAL) << "Failed to set up camera buff";
        }
    } else {
        if (!request_camera_buff_mmap(ctx)) {
            LOG(FATAL) << "Failed to set up camera buff";
        }
    }

    LOG(INFO) << "Succeed in preparing stream buffers";
    return true;
}

bool start_stream(context_t * ctx) {
    enum v4l2_buf_type type;

    /* Start v4l2 streaming */
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_STREAMON, &type) < 0) {
        LOG(FATAL) << "Failed to start streaming: " << strerror(errno);
    }

    usleep(200);

    LOG(INFO) << "Camera video streaming on ...";
    return true;
}

bool stop_stream(context_t * ctx) {
    enum v4l2_buf_type type;

    /* Stop v4l2 streaming */
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_STREAMOFF, &type)) {
        LOG(FATAL) << "Failed to stop streaming: " << strerror(errno);
    }

    LOG(INFO) << "Camera video streaming off ...";
    return true;
}

void handle_egl_image(void *pEGLImage, size_t width, size_t height, 
                      unsigned char*& img_yuv420, uint8_t*& d_img_yuv420, uint8_t*& d_img_yuv420_s) {
    EGLImageKHR *pImage = (EGLImageKHR *)pEGLImage;
    EGLImageKHR& image = *pImage;
    CUresult status;
    CUeglFrame eglFrame;
    CUgraphicsResource pResource = NULL;

    cudaFree(0);
    status = cuGraphicsEGLRegisterImage(&pResource, image,
                CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (status != CUDA_SUCCESS)
    {
        LOG(INFO) << "cuGraphicsEGLRegisterImage failed:" << status;
        return;
    }

    status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
    if (status != CUDA_SUCCESS)
    {
        LOG(INFO) << "cuGraphicsSubResourceGetMappedArray failed";
    }

    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        LOG(INFO) << "cuCtxSynchronize failed";
    }

    image_processing::yuyv2YUV(width, height, (uint8_t*)eglFrame.frame.pPitch[0], d_img_yuv420);
    image_processing::shuffleYUV(width, height, d_img_yuv420, d_img_yuv420_s);
    image_processing::downloadImageYUV(width, height, d_img_yuv420_s, img_yuv420);

    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        LOG(INFO) << "cuCtxSynchronize failed after memcpy";
    }

    status = cuGraphicsUnregisterResource(pResource);
    if (status != CUDA_SUCCESS)
    {
        LOG(INFO) << "cuGraphicsEGLUnRegisterResource failed";
    }
}

void cleanup(context_t* ctx) {
    if (ctx->cam_fd > 0) {
        close(ctx->cam_fd);
    }

    if (ctx->g_buff != NULL) {
        for (unsigned i = 0; i < V4L2_BUFFERS_NUM; i++) {
            if (ctx->g_buff[i].dmabuff_fd) {
                NvBufferDestroy(ctx->g_buff[i].dmabuff_fd);
            }
        }
        free(ctx->g_buff);
    }

    NvBufferDestroy(ctx->render_dmabuf_fd);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  context_t ctx;
  camera_initialize(&ctx);
  prepare_buffers(&ctx);

  start_stream(&ctx);

  struct sigaction sig_action;
  struct pollfd fds[1];
  NvBufferTransformParams transParams;

  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = myHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  /* Init the NvBufferTransformParams */
  memset(&transParams, 0, sizeof(transParams));
  transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
  transParams.transform_filter = NvBufferTransform_Filter_Smart;

  fds[0].fd = ctx.cam_fd;
  fds[0].events = POLLIN;

  /* Get defalut EGL display */
  // ctx.egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  // if (ctx.egl_display == EGL_NO_DISPLAY) {
  //   LOG(FATAL) << "Failed to get EGL display connection";
  // }

  /* Init EGL display connection */
  // if (!eglInitialize(ctx.egl_display, NULL, NULL)) {
  //   LOG(FATAL) << "Failed to initialize EGL display connection";
  // }

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

  // auto d_img_yuyv = image_processing::allocateImageYUYV(width, height);
  // CHECK(d_img_yuyv);
  auto d_img_yuv420 = image_processing::allocateImageYUV(width, height);
  CHECK(d_img_yuv420);
  auto d_img_yuv420_s = image_processing::allocateImageYUV(width, height);
  CHECK(d_img_yuv420_s);
  unsigned char* img_yuv420 = new unsigned char[width * height * 3 / 2];
  CHECK(img_yuv420);

  running.store(true);

  /* Wait for camera event with timeout = 5000 ms */
  while (poll(fds, 1, 5000) > 0 && running.load()) {
    if (fds[0].revents & POLLIN) {
      struct v4l2_buffer v4l2_buf;

      /* Dequeue a camera buff */
      memset(&v4l2_buf, 0, sizeof(v4l2_buf));
      v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      if (ctx.capture_dmabuf) {
        v4l2_buf.memory = V4L2_MEMORY_DMABUF;
      } else {
        v4l2_buf.memory = V4L2_MEMORY_MMAP;
      }
      if (ioctl(ctx.cam_fd, VIDIOC_DQBUF, &v4l2_buf) < 0) {
        LOG(FATAL) << "Failed to dequeue camera buff: " << strerror(errno);
      }

      ctx.frame++;
      
      if (ctx.capture_dmabuf) {
        /* Cache sync for VIC operation since the data is from CPU */
        NvBufferMemSyncForDevice(ctx.g_buff[v4l2_buf.index].dmabuff_fd, 0,
            (void**)&ctx.g_buff[v4l2_buf.index].start);
      } else {
        /* Copies raw buffer plane contents to an NvBuffer plane */
        Raw2NvBuffer(ctx.g_buff[v4l2_buf.index].start, 0,
            ctx.cam_w, ctx.cam_h, ctx.g_buff[v4l2_buf.index].dmabuff_fd);
      }

      /*  Convert the camera buffer from YUV422 to YUV420P */
      // if (-1 == NvBufferTransform(
      //     ctx.g_buff[v4l2_buf.index].dmabuff_fd, ctx.render_dmabuf_fd, &transParams)) {
      //   LOG(FATAL) << "Failed to convert the buffer";
      // }

      /* Create EGLImage from dmabuf fd */
      ctx.egl_image = NvEGLImageFromFd(ctx.egl_display, ctx.g_buff[v4l2_buf.index].dmabuff_fd);
      if (ctx.egl_image == NULL) {
        LOG(FATAL) << "Failed to map dmabuf fd to EGLImage";
      }

      // Customize handle image
      handle_egl_image(&ctx.egl_image, width, height, img_yuv420, d_img_yuv420, d_img_yuv420_s);

      /* Destroy EGLImage */
      NvDestroyEGLImage(ctx.egl_display, ctx.egl_image);
      ctx.egl_image = NULL;

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

      /* Enqueue camera buffer back to driver */
      if (ioctl(ctx.cam_fd, VIDIOC_QBUF, &v4l2_buf)) {
        LOG(FATAL) << "Failed to queue camera buffers: " << strerror(errno);
      }
    }
  }

  cleanup(&ctx);
  // cudaFree(d_img_yuyv);
  cudaFree(d_img_yuv420);
  cudaFree(d_img_yuv420_s);
  delete[] img_yuv420;

  LOG(INFO) << "Finished cleanly";

  return 0;
}