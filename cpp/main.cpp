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
  randomMat2d(&input, 2'000, 2'000);
  printf("Input: %d x %d\n", input.width, input.height);

  Mat2d<float> kernel;
  randomMat2d(&kernel, 64, 64);
  printf("Conv2d kernel: %d x %d\n", kernel.width, kernel.height);

  // float kernelArray[] = {1, 2, 3, 4};
  // const Mat2d<float> kernel = {kernelArray, 2, 2};

  // Mat2d<float> input2;
  // randomMat2d(&input2, 1'000'000'000, 1);

  MetalConv* metalConv = new MetalConv();

  char c;
  while (true) {
    if (c == 'q') {
      break;
    }

    int NUM_THREADS = 1;
    std::thread threads[NUM_THREADS];

    Benchmark benchConv2dGPU("conv2d GPU");
    for (int i = 0; i < NUM_THREADS; ++i) {
      threads[i] = std::thread([&]() {
        Mat2d<float> output;
        metalConv->conv2d(&input, &kernel, &output);
      });
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
      threads[i].join();
    }
    benchConv2dGPU.stop();

    Benchmark benchConv2dCPU("conv2d CPU");
    for (int i = 0; i < NUM_THREADS; ++i) {
      threads[i] = std::thread([&]() {
        Mat2d<float> output;
        metalConv->conv2dCPU(&input, &kernel, &output);
        delete[] output.data;
      });
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
      threads[i].join();
    }
    benchConv2dCPU.stop();

    // printOutput(output);

    printf("MaxPool kernel: %d x %d\n", 4, 4);
    Benchmark benchMaxPoolGPU("maxPool GPU");
    for (int i = 0; i < NUM_THREADS; ++i) {
      threads[i] = std::thread([&]() {
        Mat2d<float> output;
        metalConv->maxPool(&input, 4, 4, &output);
      });
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
      threads[i].join();
    }
    benchMaxPoolGPU.stop();

    Benchmark benchMaxPoolCPU("maxPool CPU");
    for (int i = 0; i < NUM_THREADS; ++i) {
      threads[i] = std::thread([&]() {
        Mat2d<float> output;
        metalConv->maxPoolCPU(&input, 4, 4, &output);
        delete[] output.data;
      });
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
      threads[i].join();
    }
    benchMaxPoolCPU.stop();
    // printOutput(output);

    printf("AvgPool kernel: %d x %d\n", 4, 4);
    Benchmark benchAvgPoolGPU("avgPool GPU");
    for (int i = 0; i < NUM_THREADS; ++i) {
      threads[i] = std::thread([&]() {
        Mat2d<float> output;
        metalConv->avgPool(&input, 4, 4, &output);
      });
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
      threads[i].join();
    }
    benchAvgPoolGPU.stop();

    Benchmark benchAvgPoolCPU("avgPool CPU");
    for (int i = 0; i < NUM_THREADS; ++i) {
      threads[i] = std::thread([&]() {
        Mat2d<float> output;
        metalConv->avgPoolCPU(&input, 4, 4, &output);
        delete[] output.data;
      });
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
      threads[i].join();
    }
    benchAvgPoolCPU.stop();

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