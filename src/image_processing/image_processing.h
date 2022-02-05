#ifndef __IMAGE_PROCESSING__
#define __IMAGE_PROCESSING__

#include <glog/logging.h>

namespace image_processing {

// Support U8C3 images only
uint8_t* allocateImage(size_t width, size_t height);

uint8_t* allocateImageYUV(size_t width, size_t height);

uint8_t* allocateImageYUYV(size_t width, size_t height);

uint8_t* uploadImage(size_t width, size_t height, uint8_t* img);

void uploadImage(size_t width, size_t height, uint8_t* img, uint8_t* d_img);

void downloadImage(size_t width, size_t height, uint8_t* d_img, uint8_t* img);

void downloadImageYUV(size_t width, size_t height, uint8_t* d_img, uint8_t* img);

void copyImage(uint8_t* d_dst, uint8_t* d_src, size_t size);

void freeImage(uint8_t* d_img);

void combineThreeImages(size_t width, size_t height, uint8_t* d_img_1, uint8_t* d_img_2, uint8_t* d_img_3, uint8_t* d_img_out);

void rgb2YUV(size_t width, size_t height, uint8_t* d_img, uint8_t* d_img_out);

void yuyv2YUV(size_t width, size_t height, uint8_t* d_img, uint8_t* d_img_out);

void shuffleYUV(size_t width, size_t height, uint8_t* d_img, uint8_t* d_img_out);

void copyYUYVWithOffset(uint8_t* d_dst, size_t width_dst, uint8_t* d_src, size_t width, size_t height, size_t offset);

}

#endif
