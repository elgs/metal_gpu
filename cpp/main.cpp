#include "metal-conv.hpp"
#include <iostream>

void printOutput(const Mat2d<float>& output) {
  printf("output: %d x %d\n", output.width, output.height);
  for (int i = 0; i < output.height; ++i) {
    for (int j = 0; j < output.width; ++j) {
      printf("%f ", output.data[i * output.width + j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main() {

  float inputArray[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const Mat2d<float> input = {inputArray, 4, 4};

  float kernelArray[] = {1, 2, 3, 4};
  const Mat2d<float> kernel = {kernelArray, 2, 2};
  Mat2d<float> output;

  MetalConv* metalConv = new MetalConv();

  char c;
  while (true) {
    if (c == 'q') {
      break;
    }
    metalConv->conv2d(&input, &kernel, &output, 1, 1, 1, 1);
    printOutput(output);

    metalConv->maxPool(&input, 2, 2, &output);
    printOutput(output);

    metalConv->avgPool(&input, 2, 2, &output);
    printOutput(output);

    float inputArray[] = {1, 2, 3, 4, 5, 6, 7};
    const Mat2d<float> input = {inputArray, 7, 1};
    const float f = metalConv->reduceSum(&input, 2);
    printf("reduceSum: %f\n", f);

    std::cin.get(c);
  }

  delete metalConv;
}