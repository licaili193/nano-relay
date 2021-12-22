#include "image_processing.h"

namespace image_processing {

namespace {

__device__ float clamp(float x, float a, float b) {
  return max(a, min(b, x));
}

// Thread per block: 1D - 512
// Blocks per grid: 2D - height, (width + 511) / 512
__global__ void copyPixel3(size_t width, size_t height, uint8_t* d_img_1, uint8_t* d_img_2, uint8_t* d_img_3, uint8_t* d_img_out) {
  size_t i = blockIdx.x;
  size_t j = blockIdx.y * 512 + threadIdx.x;

  if (i < height && j < width) {
    size_t index_single = i * width + j;
    size_t index_1 = i * 3 * width + j;
    size_t index_2 = i * 3 * width + j + width;
    size_t index_3 = i * 3 * width + j + 2 * width;
    d_img_out[3 * index_1] = d_img_1[3 * index_single];
    d_img_out[3 * index_1 + 1] = d_img_1[3 * index_single + 1];
    d_img_out[3 * index_1 + 2] = d_img_1[3 * index_single + 2];
    d_img_out[3 * index_2] = d_img_2[3 * index_single];
    d_img_out[3 * index_2 + 1] = d_img_2[3 * index_single + 1];
    d_img_out[3 * index_2 + 2] = d_img_2[3 * index_single + 2];
    d_img_out[3 * index_3] = d_img_3[3 * index_single];
    d_img_out[3 * index_3 + 1] = d_img_3[3 * index_single + 1];
    d_img_out[3 * index_3 + 2] = d_img_3[3 * index_single + 2];
  }
}

__device__ void rgb2YUVPixel(uint8_t r, uint8_t g, uint8_t b, uint8_t& y, uint8_t& u, uint8_t& v) {
  float Y = (0.257 * (float)r) + (0.504 * (float)g) + (0.098 * (float)b) + 16;
  float V = (0.439 * (float)r) - (0.368 * (float)g) - (0.071 * (float)b) + 128;
  float U = -(0.148 * (float)r) - (0.291 * (float)g) + (0.439 * (float)b) + 128;
  y = (uint8_t)(clamp(Y, 0, 255));
  u = (uint8_t)(clamp(U, 0, 255));
  v = (uint8_t)(clamp(V, 0, 255));
}

// Thread per block: 1D - 512
// Blocks per grid: 2D - height / 2, (width / 2 + 511) / 512
__global__ void rgb2YUV4Pixel(size_t width, size_t height, uint8_t* d_img, uint8_t* d_img_out) {
  size_t i = blockIdx.x;
  size_t j = blockIdx.y * 512 + threadIdx.x;

  if (i < height / 2 && j < width / 2) {
    // Pixel 1
    size_t i_1 = 2 * i;
    size_t j_1 = 2 * j;
    size_t index_1 = i_1 * width + j_1;
    uint8_t y_1, u_1, v_1;
    rgb2YUVPixel(d_img[3 * index_1], d_img[3 * index_1 + 1], d_img[3 * index_1 + 2], y_1, u_1, v_1);
    d_img_out[index_1] = y_1;

    // Pixel 2
    size_t i_2 = 2 * i;
    size_t j_2 = 2 * j + 1;
    size_t index_2 = i_2 * width + j_2;
    uint8_t y_2, u_2, v_2;
    rgb2YUVPixel(d_img[3 * index_2], d_img[3 * index_2 + 1], d_img[3 * index_2 + 2], y_2, u_2, v_2);
    d_img_out[index_2] = y_2;

    // Pixel 3
    size_t i_3 = 2 * i + 1;
    size_t j_3 = 2 * j;
    size_t index_3 = i_3 * width + j_3;
    uint8_t y_3, u_3, v_3;
    rgb2YUVPixel(d_img[3 * index_3], d_img[3 * index_3 + 1], d_img[3 * index_3 + 2], y_3, u_3, v_3);
    d_img_out[index_3] = y_3;

    // Pixel 4
    size_t i_4 = 2 * i + 1;
    size_t j_4 = 2 * j + 1;
    size_t index_4 = i_4 * width + j_4;
    uint8_t y_4, u_4, v_4;
    rgb2YUVPixel(d_img[3 * index_4], d_img[3 * index_4 + 1], d_img[3 * index_4 + 2], y_4, u_4, v_4);
    d_img_out[index_4] = y_4;
    
    d_img_out[width * height + i * width / 2 + j] = u_1 / 4 + u_2 / 4 + u_3 / 4 + u_4 / 4;
    d_img_out[width * height + width * height / 4 + i * width / 2 + j] = v_1 / 4 + v_2 / 4 + v_3 / 4 + v_4 / 4;
  }
}

// Thread per block: 1D - 512
// Blocks per grid: 2D - height, (width / 2 + 511) / 512
__global__ void yuyv2YUV2Pixel(size_t width, size_t height, uint8_t* d_img, uint8_t* d_img_out) {
    size_t i = blockIdx.x;
    size_t j = blockIdx.y * 512 + threadIdx.x;
  
    if (i < height && j < width / 2) {
      d_img_out[i * width + 2 * j] = d_img[2 * width * i + 4 * j];
      d_img_out[i * width + 2 * j + 1] = d_img[2 * width * i + 4 * j + 2];
      d_img_out[width * height + i / 2 * width / 2 + j] = d_img[2 * width * i + 4 * j + 1];
      d_img_out[width * height + width * height / 4 + i / 2 * width / 2 + j] = d_img[2 * width * i + 4 * j + 3];
    }
  }

// Thread per block: 1D - 512
// Blocks per grid: 2D - height / 2, (width / 2 + 511) / 512
__global__ void shuffleYUVPixel(size_t width, size_t height, uint8_t* d_img, uint8_t* d_img_out) {
  size_t i = blockIdx.x;
  size_t j = blockIdx.y * 512 + threadIdx.x;

  if (i < height / 2 && j < width / 2) {
    uint8_t U = d_img[width * height + i * width / 2 + j];
    uint8_t V = d_img[width * height * 5 / 4 + i * width / 2 + j];
    if (j % 2 == 0) {
      d_img_out[width * height + i * width / 4 + j / 2] = U;
      d_img_out[width * height + height / 4 * width / 2 + i * width / 4 + j / 2] = V;
    } else {
      d_img_out[width * height * 5 / 4 + i * width / 4 + j / 2] = U;
      d_img_out[width * height * 5 / 4 + height / 4 * width / 2 + i * width / 4 + j / 2] = V;
    }
  }
}

}

uint8_t* allocateImage(size_t width, size_t height) {
  uint8_t* d_res = nullptr;
  cudaMalloc(&d_res, width * height * 3);
  return d_res;  
}

uint8_t* allocateImageYUV(size_t width, size_t height) {
  uint8_t* d_res = nullptr;
  cudaMalloc(&d_res, width * height * 3 / 2);
  return d_res;  
}

uint8_t* allocateImageYUYV(size_t width, size_t height) {
  uint8_t* d_res = nullptr;
  cudaMalloc(&d_res, width * height * 2);
  return d_res; 
}

uint8_t* uploadImage(size_t width, size_t height, uint8_t* img) {
  uint8_t* d_res = allocateImage(width, height);
  CHECK(d_res);
  cudaMemcpy(d_res, img, width * height * 3, cudaMemcpyHostToDevice);
  return d_res;  
}

void uploadImage(size_t width, size_t height, uint8_t* img, uint8_t* d_img) {
  CHECK(img);
  CHECK(d_img);
  cudaMemcpy(d_img, img, width * height * 3, cudaMemcpyHostToDevice);
}

void downloadImage(size_t width, size_t height, uint8_t* d_img, uint8_t* img) {
  cudaMemcpy(img, d_img, width * height * 3, cudaMemcpyDeviceToHost);
}

void downloadImageYUV(size_t width, size_t height, uint8_t* d_img, uint8_t* img) {
  cudaMemcpy(img, d_img, width * height * 3 / 2, cudaMemcpyDeviceToHost);
}

void copyImage(uint8_t* d_dst, uint8_t* d_src, size_t size) {
  cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
}

void freeImage(uint8_t* d_img) {
  cudaFree(d_img);
}

void combineThreeImages(size_t width, size_t height, uint8_t* d_img_1, uint8_t* d_img_2, uint8_t* d_img_3, uint8_t* d_img_out) {
  CHECK(d_img_1);
  CHECK(d_img_2);
  CHECK(d_img_3);
  CHECK(d_img_out);
  dim3 gridDims(height, (width + 511) / 512);
  copyPixel3<<<gridDims, 512>>>(width, height, d_img_1, d_img_2, d_img_3, d_img_out);
}

void rgb2YUV(size_t width, size_t height, uint8_t* d_img, uint8_t* d_img_out) {
  CHECK(d_img);
  CHECK(d_img_out);
  dim3 gridDims(height / 2, (width / 2 + 511) / 512);
  rgb2YUV4Pixel<<<gridDims, 512>>>(width, height, d_img, d_img_out);
}

void yuyv2YUV(size_t width, size_t height, uint8_t* d_img, uint8_t* d_img_out) {
  CHECK(d_img);
  CHECK(d_img_out);
  dim3 gridDims(height, (width / 2 + 511) / 512);
  yuyv2YUV2Pixel<<<gridDims, 512>>>(width, height, d_img, d_img_out);
}

void shuffleYUV(size_t width, size_t height, uint8_t* d_img, uint8_t* d_img_out) {
  CHECK(d_img);
  CHECK(d_img_out);

  // Copy the lumination first
  copyImage(d_img_out, d_img, width * height);

  dim3 gridDims(height / 2, (width / 2 + 511) / 512);
  shuffleYUVPixel<<<gridDims, 512>>>(width, height, d_img, d_img_out);
}

}
