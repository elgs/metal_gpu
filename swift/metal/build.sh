#!/bin/bash

xcrun -sdk macosx metal compute.metal -c -o compute.air
xcrun -sdk macosx metallib compute.air -o compute.metallib