#ifndef METAL_CONV_HPP
#define METAL_CONV_HPP

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>

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

  void conv2dCPU(
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

  void maxPoolCPU(
      const Mat2d<float>* input,
      const unsigned int kernelWidth,
      const unsigned int kernelHeight,
      Mat2d<float>* output,
      const unsigned int strideX = 1,
      const unsigned int strideY = 1,
      const unsigned int paddingX = 0,
      const unsigned int paddingY = 0);

  void avgPool(
      const Mat2d<float>* input,
      const unsigned int kernelWidth,
      const unsigned int kernelHeight,
      Mat2d<float>* output,
      const unsigned int strideX = 1,
      const unsigned int strideY = 1,
      const unsigned int paddingX = 0,
      const unsigned int paddingY = 0);

  void avgPoolCPU(
      const Mat2d<float>* input,
      const unsigned int kernelWidth,
      const unsigned int kernelHeight,
      Mat2d<float>* output,
      const unsigned int strideX = 1,
      const unsigned int strideY = 1,
      const unsigned int paddingX = 0,
      const unsigned int paddingY = 0);

  double reduceSum(
      const Mat2d<float>* input,
      const unsigned int width);

  double reduceSumCPU(const Mat2d<float>* input);

private:
  NS::AutoreleasePool* pPool;
  MTL::Device* pDevice;
  MTL::Library* pLibrary;
  MTL::Function* pFunctionConv2d;
  MTL::Function* pFunctionMaxPool;
  MTL::Function* pFunctionAvgPool;
  MTL::Function* pFunctionReduce;
  MTL::ComputePipelineState* pComputePipelineStateConv2d;
  MTL::ComputePipelineState* pComputePipelineStateMaxPool;
  MTL::ComputePipelineState* pComputePipelineStateAvgPool;
  MTL::ComputePipelineState* pComputePipelineStateReduce;
  MTL::CommandQueue* pCommandQueue;
};

MetalConv::~MetalConv() {
  pCommandQueue->release();
  pComputePipelineStateConv2d->release();
  pComputePipelineStateMaxPool->release();
  pComputePipelineStateAvgPool->release();
  pComputePipelineStateReduce->release();
  pFunctionConv2d->release();
  pFunctionMaxPool->release();
  pFunctionAvgPool->release();
  pFunctionReduce->release();
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

  pFunctionAvgPool = pLibrary->newFunction(NS::String::string("avgPool", NS::UTF8StringEncoding));
  pComputePipelineStateAvgPool = pDevice->newComputePipelineState(pFunctionAvgPool, &pError);
  handleErrors(pComputePipelineStateAvgPool, pError);

  pFunctionReduce = pLibrary->newFunction(NS::String::string("reduceSum", NS::UTF8StringEncoding));
  pComputePipelineStateReduce = pDevice->newComputePipelineState(pFunctionReduce, &pError);
  handleErrors(pComputePipelineStateReduce, pError);

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

  if (input->width < kernel->width || input->height < kernel->height) {
    std::cout << "Input size must be greater than kernel size" << std::endl;
    return;
  }

  if (strideX == 0 || strideY == 0) {
    std::cout << "Stride must be greater than 0" << std::endl;
    return;
  }

  output->width = (input->width - kernel->width + 2 * paddingX) / strideX + 1;
  output->height = (input->height - kernel->height + 2 * paddingY) / strideY + 1;

  MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
  MTL::ComputeCommandEncoder* pComputeCommandEncoder = pCommandBuffer->computeCommandEncoder();
  pComputeCommandEncoder->setComputePipelineState(pComputePipelineStateConv2d);

  MTL::Buffer* inputBuffer = pDevice->newBuffer(input->data, sizeof(float) * input->width * input->height, MTL::ResourceStorageModeShared);
  MTL::Buffer* kernelBuffer = pDevice->newBuffer(kernel->data, sizeof(float) * kernel->width * kernel->height, MTL::ResourceStorageModeShared);
  MTL::Buffer* outputBuffer = pDevice->newBuffer(sizeof(float) * output->width * output->height, MTL::ResourceStorageModeShared);
  pComputeCommandEncoder->setBuffer(inputBuffer, 0, 0);
  pComputeCommandEncoder->setBytes(&input->width, sizeof(unsigned int), 1);
  pComputeCommandEncoder->setBytes(&input->height, sizeof(unsigned int), 2);

  pComputeCommandEncoder->setBuffer(kernelBuffer, 0, 3);
  pComputeCommandEncoder->setBytes(&kernel->width, sizeof(int), 4);
  pComputeCommandEncoder->setBytes(&kernel->height, sizeof(int), 5);

  pComputeCommandEncoder->setBuffer(outputBuffer, 0, 6);
  pComputeCommandEncoder->setBytes(&output->width, sizeof(unsigned int), 7);
  pComputeCommandEncoder->setBytes(&output->height, sizeof(unsigned int), 8);

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

  inputBuffer->release();
  kernelBuffer->release();
  outputBuffer->release();
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

  if (input->width < kernelWidth || input->height < kernelHeight) {
    std::cout << "Input size must be greater than kernel size" << std::endl;
    return;
  }

  if (strideX == 0 || strideY == 0) {
    std::cout << "Stride must be greater than 0" << std::endl;
    return;
  }

  output->width = (input->width - kernelWidth + 2 * paddingX) / strideX + 1;
  output->height = (input->height - kernelHeight + 2 * paddingY) / strideY + 1;

  MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
  MTL::ComputeCommandEncoder* pComputeCommandEncoder = pCommandBuffer->computeCommandEncoder();
  pComputeCommandEncoder->setComputePipelineState(pComputePipelineStateMaxPool);

  MTL::Buffer* inputBuffer = pDevice->newBuffer(input->data, sizeof(float) * input->width * input->height, MTL::ResourceStorageModeShared);
  MTL::Buffer* outputBuffer = pDevice->newBuffer(sizeof(float) * output->width * output->height, MTL::ResourceStorageModeShared);
  pComputeCommandEncoder->setBuffer(inputBuffer, 0, 0);
  pComputeCommandEncoder->setBytes(&input->width, sizeof(unsigned int), 1);
  pComputeCommandEncoder->setBytes(&input->height, sizeof(unsigned int), 2);

  pComputeCommandEncoder->setBytes(&kernelWidth, sizeof(int), 3);
  pComputeCommandEncoder->setBytes(&kernelHeight, sizeof(int), 4);

  pComputeCommandEncoder->setBuffer(outputBuffer, 0, 5);
  pComputeCommandEncoder->setBytes(&output->width, sizeof(unsigned int), 6);
  pComputeCommandEncoder->setBytes(&output->height, sizeof(unsigned int), 7);

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

  inputBuffer->release();
  outputBuffer->release();
}

void MetalConv::avgPool(
    const Mat2d<float>* input,
    const unsigned int kernelWidth,
    const unsigned int kernelHeight,
    Mat2d<float>* output,
    const unsigned int strideX,
    const unsigned int strideY,
    const unsigned int paddingX,
    const unsigned int paddingY) {

  if (input->width < kernelWidth || input->height < kernelHeight) {
    std::cout << "Input size must be greater than kernel size" << std::endl;
    return;
  }

  if (strideX == 0 || strideY == 0) {
    std::cout << "Stride must be greater than 0" << std::endl;
    return;
  }

  output->width = (input->width - kernelWidth + 2 * paddingX) / strideX + 1;
  output->height = (input->height - kernelHeight + 2 * paddingY) / strideY + 1;

  MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
  MTL::ComputeCommandEncoder* pComputeCommandEncoder = pCommandBuffer->computeCommandEncoder();
  pComputeCommandEncoder->setComputePipelineState(pComputePipelineStateAvgPool);

  MTL::Buffer* inputBuffer = pDevice->newBuffer(input->data, sizeof(float) * input->width * input->height, MTL::ResourceStorageModeShared);
  MTL::Buffer* outputBuffer = pDevice->newBuffer(sizeof(float) * output->width * output->height, MTL::ResourceStorageModeShared);
  pComputeCommandEncoder->setBuffer(inputBuffer, 0, 0);
  pComputeCommandEncoder->setBytes(&input->width, sizeof(unsigned int), 1);
  pComputeCommandEncoder->setBytes(&input->height, sizeof(unsigned int), 2);

  pComputeCommandEncoder->setBytes(&kernelWidth, sizeof(int), 3);
  pComputeCommandEncoder->setBytes(&kernelHeight, sizeof(int), 4);

  pComputeCommandEncoder->setBuffer(outputBuffer, 0, 5);
  pComputeCommandEncoder->setBytes(&output->width, sizeof(unsigned int), 6);
  pComputeCommandEncoder->setBytes(&output->height, sizeof(unsigned int), 7);

  pComputeCommandEncoder->setBytes(&strideX, sizeof(int), 8);
  pComputeCommandEncoder->setBytes(&strideY, sizeof(int), 9);

  pComputeCommandEncoder->setBytes(&paddingX, sizeof(int), 10);
  pComputeCommandEncoder->setBytes(&paddingY, sizeof(int), 11);

  MTL::Size gridSize = MTL::Size(output->width * output->height, 1, 1);
  // on M1 Pro Max
  NS::UInteger maxTotalThreadsPerThreadgroup = pComputePipelineStateAvgPool->maxTotalThreadsPerThreadgroup(); // 1024
  // NS::UInteger threadExecutionWidth = pComputePipelineStateAvgPool->threadExecutionWidth(); // 32
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

  inputBuffer->release();
  outputBuffer->release();
}

double MetalConv::reduceSum(
    const Mat2d<float>* input,
    const unsigned int width) {

  if (input->width < width) {
    std::cout << "Input size must be greater than width" << std::endl;
    return 0;
  }

  Mat2d<float> output;
  output.width = input->width % width == 0 ? input->width / width : input->width / width + 1;
  output.height = 1;

  if (input->width <= 32) {
    return reduceSumCPU(input);
  }

  MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
  MTL::ComputeCommandEncoder* pComputeCommandEncoder = pCommandBuffer->computeCommandEncoder();
  pComputeCommandEncoder->setComputePipelineState(pComputePipelineStateReduce);

  MTL::Buffer* inputBuffer = pDevice->newBuffer(input->data, sizeof(float) * input->width, MTL::ResourceStorageModeShared);
  MTL::Buffer* outputBuffer = pDevice->newBuffer(sizeof(float) * output.width, MTL::ResourceStorageModeShared);
  pComputeCommandEncoder->setBuffer(inputBuffer, 0, 0);
  pComputeCommandEncoder->setBytes(&input->width, sizeof(unsigned int), 1);
  pComputeCommandEncoder->setBytes(&width, sizeof(unsigned int), 2);

  pComputeCommandEncoder->setBuffer(outputBuffer, 0, 3);

  MTL::Size gridSize = MTL::Size(output.width, 1, 1);
  // on M1 Pro Max
  NS::UInteger maxTotalThreadsPerThreadgroup = pComputePipelineStateReduce->maxTotalThreadsPerThreadgroup(); // 1024
  // NS::UInteger threadExecutionWidth = pComputePipelineStateReduce->threadExecutionWidth(); // 32
  MTL::Size threadgroupSize(maxTotalThreadsPerThreadgroup, 1, 1);
  pComputeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
  pComputeCommandEncoder->endEncoding();

  pCommandBuffer->commit();
  pCommandBuffer->waitUntilCompleted();
  output.data = (float*)outputBuffer->contents();
  return reduceSum(&output, width);
}

void MetalConv::conv2dCPU(
    const Mat2d<float>* input,
    const Mat2d<float>* kernel,
    Mat2d<float>* output,
    const unsigned int strideX,
    const unsigned int strideY,
    const unsigned int paddingX,
    const unsigned int paddingY) {

  if (input->width < kernel->width || input->height < kernel->height) {
    std::cout << "Input size must be greater than kernel size" << std::endl;
    return;
  }

  if (strideX == 0 || strideY == 0) {
    std::cout << "Stride must be greater than 0" << std::endl;
    return;
  }

  output->width = (input->width - kernel->width + 2 * paddingX) / strideX + 1;
  output->height = (input->height - kernel->height + 2 * paddingY) / strideY + 1;
  output->data = new float[output->width * output->height];

  for (unsigned int oy = 0; oy < output->height; ++oy) {
    for (unsigned int ox = 0; ox < output->width; ++ox) {
      float sum = 0.0f;
      for (unsigned int ky = 0; ky < kernel->height; ++ky) {
        for (unsigned int kx = 0; kx < kernel->width; ++kx) {
          unsigned int ix = ox * strideX + kx - paddingX;
          unsigned int iy = oy * strideY + ky - paddingY;
          if (ix >= 0 && iy >= 0 && ix < input->width && iy < input->height) {
            sum += input->data[iy * input->width + ix] * kernel->data[ky * kernel->width + kx];
          }
        }
      }
      output->data[oy * output->width + ox] = sum;
    }
  }
}

void MetalConv::maxPoolCPU(
    const Mat2d<float>* input,
    const unsigned int kernelWidth,
    const unsigned int kernelHeight,
    Mat2d<float>* output,
    const unsigned int strideX,
    const unsigned int strideY,
    const unsigned int paddingX,
    const unsigned int paddingY) {
  if (input->width < kernelWidth || input->height < kernelHeight) {
    std::cout << "Input size must be greater than kernel size" << std::endl;
    return;
  }

  if (strideX == 0 || strideY == 0) {
    std::cout << "Stride must be greater than 0" << std::endl;
    return;
  }

  output->width = (input->width - kernelWidth + 2 * paddingX) / strideX + 1;
  output->height = (input->height - kernelHeight + 2 * paddingY) / strideY + 1;
  output->data = new float[output->width * output->height];

  for (unsigned int oy = 0; oy < output->height; ++oy) {
    for (unsigned int ox = 0; ox < output->width; ++ox) {
      float max = -FLT_MAX;
      for (unsigned int ky = 0; ky < kernelHeight; ++ky) {
        for (unsigned int kx = 0; kx < kernelWidth; ++kx) {
          unsigned int ix = ox * strideX + kx - paddingX;
          unsigned int iy = oy * strideY + ky - paddingY;
          if (ix >= 0 && iy >= 0 && ix < input->width && iy < input->height) {
            const float tmp = input->data[iy * input->width + ix];
            max = max > tmp ? max : tmp;
          }
        }
      }
      output->data[oy * output->width + ox] = max;
    }
  }
}

void MetalConv::avgPoolCPU(
    const Mat2d<float>* input,
    const unsigned int kernelWidth,
    const unsigned int kernelHeight,
    Mat2d<float>* output,
    const unsigned int strideX,
    const unsigned int strideY,
    const unsigned int paddingX,
    const unsigned int paddingY) {
  if (input->width < kernelWidth || input->height < kernelHeight) {
    std::cout << "Input size must be greater than kernel size" << std::endl;
    return;
  }

  if (strideX == 0 || strideY == 0) {
    std::cout << "Stride must be greater than 0" << std::endl;
    return;
  }

  output->width = (input->width - kernelWidth + 2 * paddingX) / strideX + 1;
  output->height = (input->height - kernelHeight + 2 * paddingY) / strideY + 1;
  output->data = new float[output->width * output->height];

  for (unsigned int oy = 0; oy < output->height; ++oy) {
    for (unsigned int ox = 0; ox < output->width; ++ox) {
      float sum = 0.0f;
      for (unsigned int ky = 0; ky < kernelHeight; ++ky) {
        for (unsigned int kx = 0; kx < kernelWidth; ++kx) {
          unsigned int ix = ox * strideX + kx - paddingX;
          unsigned int iy = oy * strideY + ky - paddingY;
          if (ix >= 0 && iy >= 0 && ix < input->width && iy < input->height) {
            sum += input->data[iy * input->width + ix];
          }
        }
      }
      output->data[oy * output->width + ox] = sum / (kernelWidth * kernelHeight);
    }
  }
}

double MetalConv::reduceSumCPU(const Mat2d<float>* input) {
  double sum = 0.0f;
  for (unsigned int i = 0; i < input->width; ++i) {
    sum += input->data[i];
  }
  return sum;
}

#endif // METAL_CONV_HPP
