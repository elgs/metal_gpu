#!/bin/bash

clang++ -O3 -std=c++20 -fno-objc-arc -I ./metal-cpp main.cpp -framework Foundation -framework Metal -framework MetalKit
NSZombieEnabled=YES OBJC_DEBUG_MISSING_POOLS=YES 
./a.out