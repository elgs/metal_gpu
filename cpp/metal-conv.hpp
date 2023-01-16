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

// const char* kernelSrc = R"(
// )";

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

  void maxPool(
      const Mat2d<float>* input,
      const unsigned int kernelWidth,
      const unsigned int kernelHeight,
      Mat2d<float>* output,
      const unsigned int strideX = 1,
      const unsigned int strideY = 1,
      const unsigned int paddingX = 0,
      const unsigned int paddingY = 0);

private:
  NS::AutoreleasePool* pPool;
  MTL::Device* pDevice;
  MTL::Library* pLibrary;
  MTL::Function* pFunctionConv2d;
  MTL::Function* pFunctionMaxPool;
  MTL::ComputePipelineState* pComputePipelineStateConv2d;
  MTL::ComputePipelineState* pComputePipelineStateMaxPool;
  MTL::CommandQueue* pCommandQueue;
};

MetalConv::~MetalConv() {
  pCommandQueue->release();
  pComputePipelineStateConv2d->release();
  pComputePipelineStateMaxPool->release();
  pFunctionConv2d->release();
  pFunctionMaxPool->release();
  pLibrary->release();
  pDevice->release();
  pPool->release();
}

MetalConv::MetalConv() {
  NS::Error* pError = nullptr;
  pPool = NS::AutoreleasePool::alloc()->init();

  pDevice = MTL::CreateSystemDefaultDevice();
  // pLibrary = pDevice->newLibrary(NS::String::string(kernelSrc, NS::UTF8StringEncoding), nullptr, &pError);
  MTL::Library* pLibrary = pDevice->newLibrary(NS::String::string("./metal/libconv.metallib", NS::UTF8StringEncoding), &pError);
  handleErrors(pLibrary, pError);
  assert(pLibrary != nullptr);

  pFunctionConv2d = pLibrary->newFunction(NS::String::string("conv2d", NS::UTF8StringEncoding));
  pComputePipelineStateConv2d = pDevice->newComputePipelineState(pFunctionConv2d, &pError);
  handleErrors(pComputePipelineStateConv2d, pError);

  pFunctionMaxPool = pLibrary->newFunction(NS::String::string("maxPool", NS::UTF8StringEncoding));
  pComputePipelineStateMaxPool = pDevice->newComputePipelineState(pFunctionMaxPool, &pError);
  handleErrors(pComputePipelineStateMaxPool, pError);

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
  pComputeCommandEncoder->setComputePipelineState(pComputePipelineStateConv2d);

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
  NS::UInteger maxTotalThreadsPerThreadgroup = pComputePipelineStateConv2d->maxTotalThreadsPerThreadgroup(); // 1024
  // NS::UInteger threadExecutionWidth = pComputePipelineStateConv2d->threadExecutionWidth(); // 32
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

void MetalConv::maxPool(
    const Mat2d<float>* input,
    const unsigned int kernelWidth,
    const unsigned int kernelHeight,
    Mat2d<float>* output,
    const unsigned int strideX,
    const unsigned int strideY,
    const unsigned int paddingX,
    const unsigned int paddingY) {
  auto start = std::chrono::steady_clock::now();

  output->width = (input->width - kernelWidth + 2 * paddingX) / strideX + 1;
  output->height = (input->height - kernelHeight + 2 * paddingY) / strideY + 1;

  MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
  MTL::ComputeCommandEncoder* pComputeCommandEncoder = pCommandBuffer->computeCommandEncoder();
  pComputeCommandEncoder->setComputePipelineState(pComputePipelineStateMaxPool);

  MTL::Buffer* inputBuffer = pDevice->newBuffer(input->data, sizeof(float) * input->width * input->height, MTL::ResourceStorageModeShared);
  MTL::Buffer* outputBuffer = pDevice->newBuffer(sizeof(float) * output->width * output->height, MTL::ResourceStorageModeShared);
  pComputeCommandEncoder->setBuffer(inputBuffer, 0, 0);
  pComputeCommandEncoder->setBytes(&input->width, sizeof(int), 1);
  pComputeCommandEncoder->setBytes(&input->height, sizeof(int), 2);

  pComputeCommandEncoder->setBytes(&kernelWidth, sizeof(int), 3);
  pComputeCommandEncoder->setBytes(&kernelHeight, sizeof(int), 4);

  pComputeCommandEncoder->setBuffer(outputBuffer, 0, 5);
  pComputeCommandEncoder->setBytes(&output->width, sizeof(int), 6);
  pComputeCommandEncoder->setBytes(&output->height, sizeof(int), 7);

  pComputeCommandEncoder->setBytes(&strideX, sizeof(int), 8);
  pComputeCommandEncoder->setBytes(&strideY, sizeof(int), 9);

  pComputeCommandEncoder->setBytes(&paddingX, sizeof(int), 10);
  pComputeCommandEncoder->setBytes(&paddingY, sizeof(int), 11);

  MTL::Size gridSize = MTL::Size(output->width * output->height, 1, 1);
  // on M1 Pro Max
  NS::UInteger maxTotalThreadsPerThreadgroup = pComputePipelineStateMaxPool->maxTotalThreadsPerThreadgroup(); // 1024
  // NS::UInteger threadExecutionWidth = pComputePipelineStateMaxPool->threadExecutionWidth(); // 32
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
