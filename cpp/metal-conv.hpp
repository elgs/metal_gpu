#ifndef METAL_CONV_HPP
#define METAL_CONV_HPP

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

void handleErrors(void* data, NS::Error* pError) {
  if (!data && pError) {
    printf("%s", pError->localizedDescription()->utf8String());
    assert(false);
  }
}

const char* kernelSrc = R"(
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
    
    // x, y in output grid
    const int x = index % outWidth;
    const int y = index / outWidth;

    for (int ky = 0; ky < kerHeight; ++ky) {
      for (int kx = 0; kx < kerWidth; ++kx) {
        const int ix = x * strideX + kx - paddingX;
        const int iy = y * strideY + ky - paddingY;
        if (ix >= 0 && iy >= 0 && ix < inWidth && iy < inHeight) {
          sum += in[iy * inWidth + ix] * ker[ky * kerWidth + kx];
        }
      }
    }
    out[index] = sum;
  }
)";

template <typename T>
struct Mat2d {
  T* data;
  unsigned int width;
  unsigned int height;
};

class MetalConv {
public:
  MetalConv();
  ~MetalConv();
  void conv2d(
      const Mat2d<float>* input,
      const Mat2d<float>* kernel,
      Mat2d<float>* output,
      const unsigned int strideX = 1,
      const unsigned int strideY = 1,
      const unsigned int paddingX = 0,
      const unsigned int paddingY = 0);

private:
  NS::AutoreleasePool* pPool;
  MTL::Device* pDevice;
  MTL::Library* pLibrary;
  MTL::Function* pFunction;
  MTL::ComputePipelineState* pComputePipelineState;
  MTL::CommandQueue* pCommandQueue;
};

MetalConv::~MetalConv() {
  pCommandQueue->release();
  pComputePipelineState->release();
  pFunction->release();
  pLibrary->release();
  pDevice->release();
  pPool->release();
}

MetalConv::MetalConv() {
  srand(time(0));
  NS::Error* pError = nullptr;
  pPool = NS::AutoreleasePool::alloc()->init();

  pDevice = MTL::CreateSystemDefaultDevice();
  pLibrary = pDevice->newLibrary(NS::String::string(kernelSrc, NS::UTF8StringEncoding), nullptr, &pError);
  handleErrors(pLibrary, pError);
  assert(pLibrary != nullptr);

  pFunction = pLibrary->newFunction(NS::String::string("conv2d", NS::UTF8StringEncoding));
  // MTL::Library* pLibrary = pDevice->newLibrary(NS::String::string("../metal/compute.metallib", NS::UTF8StringEncoding), &pError);
  pComputePipelineState = pDevice->newComputePipelineState(pFunction, &pError);
  handleErrors(pComputePipelineState, pError);

  pCommandQueue = pDevice->newCommandQueue();
}

void MetalConv::conv2d(
    const Mat2d<float>* input,
    const Mat2d<float>* kernel,
    Mat2d<float>* output,
    const unsigned int strideX,
    const unsigned int strideY,
    const unsigned int paddingX,
    const unsigned int paddingY) {
  auto start = std::chrono::steady_clock::now();

  output->width = (input->width - kernel->width + 2 * paddingX) / strideX + 1;
  output->height = (input->height - kernel->height + 2 * paddingY) / strideY + 1;

  MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
  MTL::ComputeCommandEncoder* pComputeCommandEncoder = pCommandBuffer->computeCommandEncoder();
  pComputeCommandEncoder->setComputePipelineState(pComputePipelineState);

  MTL::Buffer* inputBuffer = pDevice->newBuffer(input->data, sizeof(float) * input->width * input->height, MTL::ResourceStorageModeShared);
  MTL::Buffer* kernelBuffer = pDevice->newBuffer(kernel->data, sizeof(float) * kernel->width * kernel->height, MTL::ResourceStorageModeShared);
  MTL::Buffer* outputBuffer = pDevice->newBuffer(sizeof(float) * output->width * output->height, MTL::ResourceStorageModeShared);
  pComputeCommandEncoder->setBuffer(inputBuffer, 0, 0);
  pComputeCommandEncoder->setBytes(&input->width, sizeof(int), 1);
  pComputeCommandEncoder->setBytes(&input->height, sizeof(int), 2);

  pComputeCommandEncoder->setBuffer(kernelBuffer, 0, 3);
  pComputeCommandEncoder->setBytes(&kernel->width, sizeof(int), 4);
  pComputeCommandEncoder->setBytes(&kernel->height, sizeof(int), 5);

  pComputeCommandEncoder->setBuffer(outputBuffer, 0, 6);
  pComputeCommandEncoder->setBytes(&output->width, sizeof(int), 7);
  pComputeCommandEncoder->setBytes(&output->height, sizeof(int), 8);

  pComputeCommandEncoder->setBytes(&strideX, sizeof(int), 9);
  pComputeCommandEncoder->setBytes(&strideY, sizeof(int), 10);

  pComputeCommandEncoder->setBytes(&paddingX, sizeof(int), 11);
  pComputeCommandEncoder->setBytes(&paddingY, sizeof(int), 12);

  MTL::Size gridSize = MTL::Size(output->width * output->height, 1, 1);
  // on M1 Pro Max
  NS::UInteger maxTotalThreadsPerThreadgroup = pComputePipelineState->maxTotalThreadsPerThreadgroup(); // 1024
  // NS::UInteger threadExecutionWidth = pComputePipelineState->threadExecutionWidth(); // 32
  MTL::Size threadgroupSize(maxTotalThreadsPerThreadgroup, 1, 1);
  pComputeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
  pComputeCommandEncoder->endEncoding();

  auto callback = [output, outputBuffer](MTL::CommandBuffer* pCommandBuffer) {
    output->data = (float*)outputBuffer->contents();
  };

  // pCommandBuffer->addCompletedHandler(callback);

  pCommandBuffer->commit();
  pCommandBuffer->waitUntilCompleted();
  callback(pCommandBuffer);
}

#endif // METAL_CONV_HPP
