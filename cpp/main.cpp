#include "metal-conv.hpp"
#include <iostream>

int main() {

  float inputArray[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float kernelArray[] = {5.0f, 4.0f};

  const Mat2d<float> input = {inputArray, 5, 1};
  const Mat2d<float> kernel = {kernelArray, 2, 1};
  Mat2d<float>* output = new Mat2d<float>();

  MetalConv* metalConv = new MetalConv();

  char c;
  while (true) {
    if (c == 'q') {
      break;
    }
    metalConv->conv2d(&input, &kernel, output);
    for (int i = 0; i < output->height; i++) {
      for (int j = 0; j < output->width; j++) {
        std::cout << output->data[i * output->width + j] << " ";
      }
      std::cout << std::endl;
    }

    std::cin.get(c);
  }

  delete metalConv;
}