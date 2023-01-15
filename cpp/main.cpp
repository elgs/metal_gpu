#include <iostream>
#include "metal-conv.hpp"

void hook() {
  char c;
  std::cin.get(c);
  while (c != '\n') {
    std::cout << c;
    std::cin.get(c);
  }
}

int main() {
  MetalConv* metalConv = new MetalConv();
  metalConv->conv2d();
  metalConv->conv2d();
  hook();
  delete metalConv;
}