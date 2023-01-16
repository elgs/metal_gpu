#define NEGATIVE_INFINITY (-1.0f / 0.0f)

#include <metal_stdlib>
using namespace metal;

[[kernel]] void conv2d(
  const device float* in   [[ buffer(0) ]],
  const device int& inWidth [[ buffer(1) ]],
  const device int& inHeight [[ buffer(2) ]],

  const device float* ker  [[ buffer(3) ]],
  const device int& kerWidth [[ buffer(4) ]],
  const device int& kerHeight [[ buffer(5) ]],

        device float* out  [[ buffer(6) ]],
  const device int& outWidth [[ buffer(7) ]],
  const device int& outHeight [[ buffer(8) ]],

  const device int& strideX [[ buffer(9) ]],
  const device int& strideY [[ buffer(10) ]],

  const device int& paddingX [[ buffer(11) ]],
  const device int& paddingY [[ buffer(12) ]],

  const        uint   index   [[ thread_position_in_grid ]]
) {
  float sum = 0.0f;
  
  const int ox = index % outWidth;
  const int oy = index / outWidth;

  for (int ky = 0; ky < kerHeight; ++ky) {
    for (int kx = 0; kx < kerWidth; ++kx) {
      const int ix = ox * strideX + kx - paddingX;
      const int iy = oy * strideY + ky - paddingY;
      if (ix >= 0 && iy >= 0 && ix < inWidth && iy < inHeight) {
        sum += in[iy * inWidth + ix] * ker[ky * kerWidth + kx];
      }
    }
  }
  out[index] = sum;
}

[[kernel]] void maxPool(
  const device float* in   [[ buffer(0) ]],
  const device int& inWidth [[ buffer(1) ]],
  const device int& inHeight [[ buffer(2) ]],

  const device int& kerWidth [[ buffer(3) ]],
  const device int& kerHeight [[ buffer(4) ]],

        device float* out  [[ buffer(5) ]],
  const device int& outWidth [[ buffer(6) ]],
  const device int& outHeight [[ buffer(7) ]],

  const device int& strideX [[ buffer(8) ]],
  const device int& strideY [[ buffer(9) ]],

  const device int& paddingX [[ buffer(10) ]],
  const device int& paddingY [[ buffer(11) ]],

  const        uint   index   [[ thread_position_in_grid ]]
) {
  float max = NEGATIVE_INFINITY;
  
  const int ox = index % outWidth;
  const int oy = index / outWidth;

  for (int ky = 0; ky < kerHeight; ++ky) {
    for (int kx = 0; kx < kerWidth; ++kx) {
      const int ix = ox * strideX + kx - paddingX;
      const int iy = oy * strideY + ky - paddingY;
      if (ix >= 0 && iy >= 0 && ix < inWidth && iy < inHeight) {
        const float tmp = in[iy * inWidth + ix];
        max = max > tmp ? max : tmp;
      }
    }
  }
  out[index] = max;
}