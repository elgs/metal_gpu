#include "metal-conv.hpp"
#include <iostream>
#include <thread>

struct Benchmark {
  bool stopped = false;
  std::string title = "";
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  std::chrono::duration<double> elapsed_seconds;
  Benchmark(std::string title = "") {
    this->title = title;
    start = std::chrono::high_resolution_clock::now();
  }
  void lap() {
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    printf("[%s] elapsed time: %f s\n", this->title.c_str(), elapsed_seconds.count());
    start = std::chrono::high_resolution_clock::now();
  }

  void stop() {
    if (stopped) {
      return;
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    printf("[%s] elapsed time: %f s\n", this->title.c_str(), elapsed_seconds.count());
    stopped = true;
  }

  ~Benchmark() {
    if (stopped) {
      return;
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    printf("[%s] elapsed time: %f s\n", this->title.c_str(), elapsed_seconds.count());
  }
};

void randomMat2d(Mat2d<float>* mat, unsigned int width, unsigned int height) {
  mat->width = width;
  mat->height = height;
  mat->data = new float[width * height];
  for (unsigned int i = 0; i < mat->height; ++i) {
    for (unsigned int j = 0; j < mat->width; ++j) {
      mat->data[i * mat->width + j] = (float)rand() / (float)RAND_MAX;
    }
  }
}

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
  srand(time(NULL));

  // float inputArray[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  // const Mat2d<float> input = {inputArray, 4, 4};
  Mat2d<float> input;
  randomMat2d(&input, 20'000, 20'000);

  float kernelArray[] = {1, 2, 3, 4};
  const Mat2d<float> kernel = {kernelArray, 2, 2};

  // Mat2d<float> input2;
  // randomMat2d(&input2, 1'000'000'000, 1);

  MetalConv* metalConv = new MetalConv();

  char c;
  while (true) {
    if (c == 'q') {
      break;
    }

    int NUM_THREADS = 8;

    std::thread threads[NUM_THREADS];
    Benchmark benchConv2d("conv2d");
    printf("conv2d threads started: ");
    for (int i = 0; i < NUM_THREADS; ++i) {
      printf("%d ", i);
      threads[i] = std::thread([&]() {
        Mat2d<float> output;
        metalConv->conv2d(&input, &kernel, &output, 1, 1, 1, 1);
      });
    }
    printf("\n");
    printf("conv2d threads joined: ");
    for (int i = 0; i < NUM_THREADS; ++i) {
      printf("%d ", i);
      threads[i].join();
    }
    printf("\n");
    benchConv2d.stop();

    // printOutput(output);

    Benchmark benchMaxPool("maxPool");
    printf("maxpool threads started: ");
    for (int i = 0; i < NUM_THREADS; ++i) {
      printf("%d ", i);
      threads[i] = std::thread([&]() {
        Mat2d<float> output;
        metalConv->maxPool(&input, 2, 2, &output);
      });
    }
    printf("\n");
    printf("maxpool threads joined: ");
    for (int i = 0; i < NUM_THREADS; ++i) {
      printf("%d ", i);
      threads[i].join();
    }
    printf("\n");
    benchMaxPool.stop();
    // printOutput(output);

    Benchmark benchAvgPool("avgPool");
    printf("avgpool threads started: ");
    for (int i = 0; i < NUM_THREADS; ++i) {
      printf("%d ", i);
      threads[i] = std::thread([&]() {
        Mat2d<float> output;
        metalConv->avgPool(&input, 2, 2, &output);
      });
    }
    printf("\n");
    printf("avgpool threads joined: ");
    for (int i = 0; i < NUM_THREADS; ++i) {
      printf("%d ", i);
      threads[i].join();
    }
    printf("\n");
    benchAvgPool.stop();

    // printOutput(output);

    // Benchmark benchReduce;
    // const float f = metalConv->reduceSum(&input2, 256);
    // benchReduce.lap();
    // printf("reduceSum: %f\n", f);

    // const float f2 = metalConv->reduceSumCPU(&input2);
    // benchReduce.stop();
    // printf("reduceSumCPU: %f\n", f2);

    std::cin.get(c);
  }

  delete[] input.data;
  // delete[] input2.data;
  delete metalConv;
}