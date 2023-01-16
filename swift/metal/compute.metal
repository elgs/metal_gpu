#include <metal_stdlib>
using namespace metal;

[[kernel]] void add(const device float* arr1  [[ buffer(0) ]],
                const device float* arr2  [[ buffer(1) ]],
                const device float* arr3  [[ buffer(2) ]],
                device float* resultArray [[ buffer(3) ]],
                uint   index [[ thread_position_in_grid ]]) {
  resultArray[index] = arr1[index] + arr2[index] + arr3[index];
}
