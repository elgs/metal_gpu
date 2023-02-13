#include <iostream>

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
