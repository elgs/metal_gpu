#define NEGATIVE_INFINITY (-1.0f / 0.0f)

#include <metal_stdlib>
using namespace metal;

[[kernel]] void conv2d(
  const device float*   in         [[ buffer(0) ]],
  const device uint&    inWidth    [[ buffer(1) ]],
  const device uint&    inHeight   [[ buffer(2) ]],

  const device float*   ker        [[ buffer(3) ]],
  const device uint&    kerWidth   [[ buffer(4) ]],
  const device uint&    kerHeight  [[ buffer(5) ]],

        device float*   out        [[ buffer(6) ]],
  const device uint&    outWidth   [[ buffer(7) ]],
  const device uint&    outHeight  [[ buffer(8) ]],

  const device uint&    strideX    [[ buffer(9) ]],
  const device uint&    strideY    [[ buffer(10) ]],

  const device uint&    paddingX   [[ buffer(11) ]],
  const device uint&    paddingY   [[ buffer(12) ]],

  const        uint     index      [[ thread_position_in_grid ]]
) {
  float sum = 0.0f;
  
  const uint ox = index % outWidth;
  const uint oy = index / outWidth;

  for (uint ky = 0; ky < kerHeight; ++ky) {
    for (uint kx = 0; kx < kerWidth; ++kx) {
      const uint ix = ox * strideX + kx - paddingX;
      const uint iy = oy * strideY + ky - paddingY;
      if (ix >= 0 && iy >= 0 && ix < inWidth && iy < inHeight) {
        sum += in[iy * inWidth + ix] * ker[ky * kerWidth + kx];
      }
    }
  }
  out[index] = sum;
}

[[kernel]] void maxPool(
  const device float*   in         [[ buffer(0) ]],
  const device uint&    inWidth    [[ buffer(1) ]],
  const device uint&    inHeight   [[ buffer(2) ]],

  const device uint&    kerWidth   [[ buffer(3) ]],
  const device uint&    kerHeight  [[ buffer(4) ]],

        device float*   out        [[ buffer(5) ]],
  const device uint&    outWidth   [[ buffer(6) ]],
  const device uint&    outHeight  [[ buffer(7) ]],

  const device uint&    strideX    [[ buffer(8) ]],
  const device uint&    strideY    [[ buffer(9) ]],

  const device uint&    paddingX   [[ buffer(10) ]],
  const device uint&    paddingY   [[ buffer(11) ]],

  const        uint     index      [[ thread_position_in_grid ]]
) {
  float max = NEGATIVE_INFINITY;
  
  const uint ox = index % outWidth;
  const uint oy = index / outWidth;

  for (uint ky = 0; ky < kerHeight; ++ky) {
    for (uint kx = 0; kx < kerWidth; ++kx) {
      const uint ix = ox * strideX + kx - paddingX;
      const uint iy = oy * strideY + ky - paddingY;
      if (ix >= 0 && iy >= 0 && ix < inWidth && iy < inHeight) {
        const float tmp = in[iy * inWidth + ix];
        max = max > tmp ? max : tmp;
      }
    }
  }
  out[index] = max;
}

[[kernel]] void avgPool(
  const device float*   in        [[ buffer(0) ]],
  const device uint&    inWidth   [[ buffer(1) ]],
  const device uint&    inHeight  [[ buffer(2) ]],

  const device uint&    kerWidth  [[ buffer(3) ]],
  const device uint&    kerHeight [[ buffer(4) ]],

        device float*   out       [[ buffer(5) ]],
  const device uint&    outWidth  [[ buffer(6) ]],
  const device uint&    outHeight [[ buffer(7) ]],

  const device uint&    strideX   [[ buffer(8) ]],
  const device uint&    strideY   [[ buffer(9) ]],

  const device uint&    paddingX  [[ buffer(10) ]],
  const device uint&    paddingY  [[ buffer(11) ]],

  const        uint     index     [[ thread_position_in_grid ]]
) {
  float sum = 0.0f;
  
  const uint ox = index % outWidth;
  const uint oy = index / outWidth;

  for (uint ky = 0; ky < kerHeight; ++ky) {
    for (uint kx = 0; kx < kerWidth; ++kx) {
      const uint ix = ox * strideX + kx - paddingX;
      const uint iy = oy * strideY + ky - paddingY;
      if (ix >= 0 && iy >= 0 && ix < inWidth && iy < inHeight) {
        sum += in[iy * inWidth + ix];
      }
    }
  }
  out[index] = sum / (kerWidth * kerHeight);
}

[[kernel]] void reduceSum(
  const device float* in        [[ buffer(0) ]],
  const device uint&  inLength  [[ buffer(1) ]],
  const device uint&  width     [[ buffer(2) ]],
        device float* out       [[ buffer(3) ]],
  const        uint   index     [[ thread_position_in_grid ]]
) {
  float sum = 0.0f;
  for (uint i = 0; i < width && index * width + i < inLength; ++i) {
    sum += in[index * width + i];
  }
  out[index] = sum;
}