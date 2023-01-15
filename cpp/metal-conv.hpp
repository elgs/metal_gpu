#ifndef METAL_CONV_HPP
#define METAL_CONV_HPP

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

void handleErrors(NS::Error* pError) {
  if (pError) {
    printf("%s", pError->localizedDescription()->utf8String());
    assert(false);
  }
}

const char* kernelSrc = R"(
  #include <metal_stdlib>
  using namespace metal;

  [[kernel]] void add(const device float* arr1  [[ buffer(0) ]],
                  const device float* arr2  [[ buffer(1) ]],
                  device float* resultArray [[ buffer(2) ]],
                  uint   index [[ thread_position_in_grid ]]) {
    resultArray[index] = arr1[index] + arr2[index];
  }
)";

const unsigned int ARRAY_LENGTH = 1000000;
const unsigned int BUFFER_SIZE = ARRAY_LENGTH * sizeof(float);

void generateRandomFloatData(MTL::Buffer* buffer) {
  float* dataPtr = (float*)buffer->contents();
  // printf("%p\n", dataPtr);
  for (unsigned long int i = 0; i < ARRAY_LENGTH; ++i) {
    dataPtr[i] = float(rand() % 10);
  }
}

class MetalConv {
public:
  MetalConv();
  ~MetalConv();
  void conv2d();

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
  NS::Error* pError = nullptr;
  pPool = NS::AutoreleasePool::alloc()->init();

  pDevice = MTL::CreateSystemDefaultDevice();
  pLibrary = pDevice->newLibrary(NS::String::string(kernelSrc, NS::UTF8StringEncoding), nullptr, &pError);
  if (!pLibrary) {
    handleErrors(pError);
  }
  pFunction = pLibrary->newFunction(NS::String::string("add", NS::UTF8StringEncoding));
  // MTL::Library* pLibrary = pDevice->newLibrary(NS::String::string("../metal/compute.metallib", NS::UTF8StringEncoding), &pError);
  pComputePipelineState = pDevice->newComputePipelineState(pFunction, &pError);
  if (!pComputePipelineState) {
    handleErrors(pError);
  }

  pCommandQueue = pDevice->newCommandQueue();
}

void MetalConv::conv2d() {
  NS::AutoreleasePool* pPoolLocal = NS::AutoreleasePool::alloc()->init();
  MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
  MTL::ComputeCommandEncoder* pComputeCommandEncoder = pCommandBuffer->computeCommandEncoder();
  pComputeCommandEncoder->setComputePipelineState(pComputePipelineState);

  MTL::Buffer* arr1Buf = pDevice->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModeShared);
  MTL::Buffer* arr2Buf = pDevice->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModeShared);
  MTL::Buffer* resultBuf = pDevice->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModeShared);
  pComputeCommandEncoder->setBuffer(arr1Buf, 0, 0);
  pComputeCommandEncoder->setBuffer(arr2Buf, 0, 1);
  pComputeCommandEncoder->setBuffer(resultBuf, 0, 2);

  MTL::Size gridSize = MTL::Size(ARRAY_LENGTH, 1, 1);
  // on M1 Pro Max
  NS::UInteger maxTotalThreadsPerThreadgroup = pComputePipelineState->maxTotalThreadsPerThreadgroup(); // 1024
  // NS::UInteger threadExecutionWidth = pComputePipelineState->threadExecutionWidth(); // 32
  MTL::Size threadgroupSize(maxTotalThreadsPerThreadgroup, 1, 1);
  pComputeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);

  // prepare data
  generateRandomFloatData(arr1Buf);
  generateRandomFloatData(arr2Buf);
  // end of prepare data

  pComputeCommandEncoder->endEncoding();

  // copy pointers
  pCommandBuffer->addCompletedHandler([arr1Buf, arr2Buf, resultBuf](MTL::CommandBuffer* pCommandBuffer) {
    float* r1 = (float*)arr1Buf->contents();
    float* r2 = (float*)arr2Buf->contents();
    float* result = (float*)resultBuf->contents();
    // printf("%p, %p, %p\n", r1, r2, result);
    for (int i = 0; i < (ARRAY_LENGTH > 5 ? 5 : ARRAY_LENGTH); i++) {
      printf("%f + %f = %f\n", r1[i], r2[i], result[i]);
    }
    printf("done\n");
  });

  pCommandBuffer->commit();
  pPoolLocal->release();
}

#endif // METAL_CONV_HPP
