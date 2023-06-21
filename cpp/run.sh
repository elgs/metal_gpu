#!/bin/bash

clang++ -O3 -std=c++2b -fno-objc-arc -I ./metal-cpp main.cpp -framework Foundation -framework Metal -framework MetalKit
NSZombieEnabled=NO OBJC_DEBUG_MISSING_POOLS=NO ./a.out