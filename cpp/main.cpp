#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

const char* kernelSrc = R"(
  #include <metal_stdlib>
  using namespace metal;

  [[kernel]] void add(const device float* arr1  [[ buffer(0) ]],
                  const device float* arr2  [[ buffer(1) ]],
                  const device float* arr3  [[ buffer(2) ]],
                  device float* resultArray [[ buffer(3) ]],
                  uint   index [[ thread_position_in_grid ]]) {
    resultArray[index] = arr1[index] + arr2[index] + arr3[index];
  }
)";

void handleErrors(NS::Error* pError) {
  if (pError) {
    printf("%s", pError->localizedDescription()->utf8String());
    assert(false);
  }
}

const unsigned int ARRAY_LENGTH = 1000000;
const unsigned int BUFFER_SIZE = ARRAY_LENGTH * sizeof(float);

void generateRandomFloatData(MTL::Buffer* buffer) {
  float* dataPtr = (float*)buffer->contents();
  for (unsigned long int i = 0; i < ARRAY_LENGTH; ++i) {
    dataPtr[i] = float(rand() % 10);
  }
}

int main() {
  NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

  NS::Error* pError = nullptr;

  MTL::Device* pDevice = MTL::CreateSystemDefaultDevice();

  MTL::Library* pLibrary = pDevice->newLibrary(NS::String::string(kernelSrc, NS::UTF8StringEncoding), nullptr, &pError);
  // MTL::Library* pLibrary = pDevice->newLibrary(NS::String::string("../metal/compute.metallib", NS::UTF8StringEncoding), &pError);
  if (!pLibrary) {
    handleErrors(pError);
  }

  MTL::Function* pFunction = pLibrary->newFunction(NS::String::string("add", NS::UTF8StringEncoding));

  MTL::CommandQueue* pCommandQueue = pDevice->newCommandQueue();
  MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
  MTL::ComputeCommandEncoder* pComputeCommandEncoder = pCommandBuffer->computeCommandEncoder();

  MTL::ComputePipelineState* pComputePipelineState = pDevice->newComputePipelineState(pFunction, &pError);
  if (!pComputePipelineState) {
    handleErrors(pError);
  }
  pComputeCommandEncoder->setComputePipelineState(pComputePipelineState);

  MTL::Buffer* arr1Buf = pDevice->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModeShared);
  MTL::Buffer* arr2Buf = pDevice->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModeShared);
  MTL::Buffer* arr3Buf = pDevice->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModeShared);
  MTL::Buffer* resultBuf = pDevice->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModeShared);
  generateRandomFloatData(arr1Buf);
  generateRandomFloatData(arr2Buf);
  generateRandomFloatData(arr3Buf);

  pComputeCommandEncoder->setBuffer(arr1Buf, 0, 0);
  pComputeCommandEncoder->setBuffer(arr2Buf, 0, 1);
  pComputeCommandEncoder->setBuffer(arr3Buf, 0, 2);
  pComputeCommandEncoder->setBuffer(resultBuf, 0, 3);

  MTL::Size gridSize = MTL::Size(ARRAY_LENGTH, 1, 1);
  NS::UInteger threadGroupSize = pComputePipelineState->maxTotalThreadsPerThreadgroup();
  MTL::Size threadgroupSize(threadGroupSize, 1, 1);

  pComputeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);

  pComputeCommandEncoder->endEncoding();
  pCommandBuffer->commit();

  pCommandBuffer->waitUntilCompleted();

  // get result
  float* resultData = (float*)resultBuf->contents();

  float* r1 = (float*)arr1Buf->contents();
  float* r2 = (float*)arr2Buf->contents();
  float* r3 = (float*)arr3Buf->contents();
  float* result = (float*)resultBuf->contents();

  for (int i = 0; i < 10; i++) {
    printf("%f + %f + %f = %f\n", r1[i], r2[i], r3[i], result[i]);
  }
  /////////////////////////////////////////////////////////////////////////////

  arr1Buf->release();
  arr2Buf->release();
  arr3Buf->release();
  resultBuf->release();

  // pCommandBuffer->release();
  // pComputeCommandEncoder->release();
  // pCommandQueue->release();
  // pComputePipelineState->release();
  // pFunction->release();
  // pLibrary->release();
  // pDevice->release();

  pPool->release();
}
