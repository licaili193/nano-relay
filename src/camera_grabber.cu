#include "camera_grabber.h"

bool CameraGrabber::initializeCamera(std::string camera_name) {
  context_t* ctx = &context_;

  sprintf(ctx->cam_devname, "%s", camera_name.c_str());
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
    // TODO: to handle it gracefully - stream a grey screen if camera unable to initialize
  }

  /* Get the real format in case the desired is not supported */
  memset(&fmt, 0, sizeof fmt);
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(ctx->cam_fd, VIDIOC_G_FMT, &fmt) < 0) {
    LOG(FATAL) << "Failed to get camera output format" << strerror(errno);
    // TODO: to handle it gracefully - stream a grey screen if camera unable to initialize
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

  return prepareBuffers();
}

bool CameraGrabber::prepareBuffers() {
  context_t* ctx = &context_;

  NvBufferCreateParams input_params = {0};

  /* Allocate global buffer context */
  ctx->g_buff = (nv_buffer*)malloc(V4L2_BUFFERS_NUM * sizeof(nv_buffer));
  if (ctx->g_buff == NULL) {
    LOG(FATAL) << "Failed to allocate global buffer context";
    // TODO: to handle it gracefully - stream a grey screen if camera unable to initialize
  }

  input_params.payloadType = NvBufferPayload_SurfArray;
  input_params.width = ctx->cam_w;
  input_params.height = ctx->cam_h;
  input_params.layout = NvBufferLayout_Pitch;

  /* Create buffer and provide it with camera */
  for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++) {
    int fd;
    NvBufferParams params = {0};

    input_params.colorFormat = get_nvbuff_color_fmt(ctx->cam_pixfmt);
    input_params.nvbuf_tag = NvBufferTag_CAMERA;
    if (-1 == NvBufferCreateEx(&fd, &input_params)) {
      LOG(FATAL) << "Failed to create NvBuffer";
      // TODO: to handle it gracefully - stream a grey screen if camera unable to initialize
    }

    ctx->g_buff[index].dmabuff_fd = fd;

    if (-1 == NvBufferGetParams(fd, &params)) {
      LOG(FATAL) << "Failed to get NvBuffer parameters";
      // TODO: to handle it gracefully - stream a grey screen if camera unable to initialize
    }

    if (ctx->cam_pixfmt == V4L2_PIX_FMT_GREY && params.pitch[0] != params.width[0]) {
      LOG(WARNING) << "Disabled DAM";
      ctx->capture_dmabuf = false;
    }

    /* TODO: add multi-planar support
       Currently only supports YUV422 interlaced single-planar */
    if (ctx->capture_dmabuf) {
      if (-1 == NvBufferMemMap(ctx->g_buff[index].dmabuff_fd, 0, NvBufferMem_Read_Write,
                              (void**)&ctx->g_buff[index].start)) {
        LOG(FATAL) << "Failed to map buffer";
        // TODO: to handle it gracefully - stream a grey screen if camera unable to initialize
      }
    }
  }

  input_params.colorFormat = get_nvbuff_color_fmt(V4L2_PIX_FMT_YUYV);
  input_params.nvbuf_tag = NvBufferTag_VIDEO_CONVERT;
  /* Create Render buffer */
  if (-1 == NvBufferCreateEx(&ctx->render_dmabuf_fd, &input_params)) {
    LOG(FATAL) << "Failed to create NvBuffer";
    // TODO: to handle it gracefully - stream a grey screen if camera unable to initialize
  }

  if (ctx->capture_dmabuf) {
    if (!request_camera_buff(ctx)) {
      LOG(FATAL) << "Failed to set up camera buff";
      // TODO: to handle it gracefully - stream a grey screen if camera unable to initialize
    }
  } else {
    if (!request_camera_buff_mmap(ctx)) {
      LOG(FATAL) << "Failed to set up camera buff";
      // TODO: to handle it gracefully - stream a grey screen if camera unable to initialize
    }
  }

  LOG(INFO) << "Succeed in preparing stream buffers";
  return true;
}

void CameraGrabber::worker() {
  context_t& ctx = context_;

  enum v4l2_buf_type type;

  /* Start v4l2 streaming */
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(ctx.cam_fd, VIDIOC_STREAMON, &type) < 0) {
    LOG(FATAL) << "Failed to start streaming: " << strerror(errno);
    // TODO: to handle it gracefully - stream a grey screen if camera unable to initialize
  }

  usleep(200);

  LOG(INFO) << "Camera video streaming on ...";
  
  struct pollfd fds[1];
  NvBufferTransformParams transParams;

  /* Init the NvBufferTransformParams */
  memset(&transParams, 0, sizeof(transParams));
  transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
  transParams.transform_filter = NvBufferTransform_Filter_Smart;

  fds[0].fd = ctx.cam_fd;
  fds[0].events = POLLIN;

  running_.store(true);
  while (poll(fds, 1, 5000) > 0 &&running_.load()) {
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
        // TODO: Do we need to handle it gracefully
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
  
      /* Create EGLImage from dmabuf fd */
      ctx.egl_image = NvEGLImageFromFd(ctx.egl_display, ctx.g_buff[v4l2_buf.index].dmabuff_fd);
      if (ctx.egl_image == NULL) {
        LOG(FATAL) << "Failed to map dmabuf fd to EGLImage";
        // TODO: Do we need to handle it gracefully
      }
  
      // Customize handle image
      // TODO: TO CHANGE HERE!!!
      handleEglImage(&ctx.egl_image, ctx.cam_w, ctx.cam_h, d_img_yuyv_);
  
      /* Destroy EGLImage */
      NvDestroyEGLImage(ctx.egl_display, ctx.egl_image);
      ctx.egl_image = NULL;

      /* Enqueue camera buffer back to driver */
      if (ioctl(ctx.cam_fd, VIDIOC_QBUF, &v4l2_buf)) {
        LOG(FATAL) << "Failed to queue camera buffers: " << strerror(errno);
        // TODO: Do we need to handle it gracefully
      }
    }
  }

  enum v4l2_buf_type buf_type;
  /* Stop v4l2 streaming */
  buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(ctx.cam_fd, VIDIOC_STREAMOFF, &buf_type)) {
    LOG(FATAL) << "Failed to stop streaming: " << strerror(errno);
    // TODO: Do we need to handle it gracefully
  }

  LOG(INFO) << "Camera video streaming off ...";

  cleanup();
}

void CameraGrabber::handleEglImage(void *pEGLImage, size_t width, size_t height, uint8_t* d_img_yuyv) {
  EGLImageKHR *pImage = (EGLImageKHR *)pEGLImage;
  EGLImageKHR& image = *pImage;
  CUresult status;
  CUeglFrame eglFrame;
  CUgraphicsResource pResource = NULL;

  cudaFree(0);
  status = cuGraphicsEGLRegisterImage(&pResource, image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
  if (status != CUDA_SUCCESS) {
    LOG(INFO) << "cuGraphicsEGLRegisterImage failed:" << status;
    return;
  }

  status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
  if (status != CUDA_SUCCESS) {
    LOG(INFO) << "cuGraphicsSubResourceGetMappedArray failed";
  }

  status = cuCtxSynchronize();
  if (status != CUDA_SUCCESS) {
    LOG(INFO) << "cuCtxSynchronize failed";
  }

  cudaMemcpy(d_img_yuyv, (uint8_t*)eglFrame.frame.pPitch[0], width * height * 2, cudaMemcpyDeviceToDevice);
  new_image_.store(true);

  status = cuCtxSynchronize();
  if (status != CUDA_SUCCESS) {
    LOG(INFO) << "cuCtxSynchronize failed after memcpy";
  }

  status = cuGraphicsUnregisterResource(pResource);
  if (status != CUDA_SUCCESS) {
    LOG(INFO) << "cuGraphicsEGLUnRegisterResource failed";
  }
}

void CameraGrabber::cleanup() {
  context_t* ctx = &context_;
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
