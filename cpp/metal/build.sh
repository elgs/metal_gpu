#!/bin/bash

xcrun -sdk macosx metal libconv.metal -c -o libconv.air 
xcrun -sdk macosx metallib libconv.air -o libconv.metallib