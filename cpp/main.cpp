#include "metal-conv.hpp"
#include <iostream>

int main() {
  MetalConv* metalConv = new MetalConv();

  char c;
  while (true) {
    if (c == 'q') {
      break;
    }
    metalConv->conv2d();
    std::cin.get(c);
  }

  delete metalConv;
}