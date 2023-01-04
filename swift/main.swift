import MetalKit

let count: Int = 10000

// let kernelSrc = """
//   #include <metal_stdlib>
//   using namespace metal;

//   kernel void add(device const float* arr1  [[ buffer(0) ]],
//                   device const float* arr2  [[ buffer(1) ]],
//                   device const float* arr3  [[ buffer(2) ]],
//                   device float* resultArray [[ buffer(3) ]],
//                   uint   index [[ thread_position_in_grid ]]) {
//     resultArray[index] = arr1[index] + arr2[index] + arr3[index];
//   }
// """;

// Create our random arrays
print("Start loading data...")
var arr1 = getRandomArray()
var arr2 = getRandomArray()
var arr3 = getRandomArray()
print("Finished loading data.")

// Begin the process
let startTime = CFAbsoluteTimeGetCurrent()

// The GPU we want to use
let device = MTLCreateSystemDefaultDevice()

// A library for getting our metal functions
let gpuFunctionLibrary = try! device?.makeLibrary(
  URL: URL(fileURLWithPath: "../metal/compute.metallib"))
// let gpuFunctionLibrary = try! device?.makeLibrary(source: kernelSrc, options: nil)

// Grab our gpu function
let additionGPUFunction = gpuFunctionLibrary?.makeFunction(name: "add")

var additionComputePipelineState: MTLComputePipelineState!
additionComputePipelineState = try! device?.makeComputePipelineState(function: additionGPUFunction!)

// A fifo queue for sending commands to the gpu
let commandQueue = device?.makeCommandQueue()

// Create a buffer to be sent to the command queue
let commandBuffer = commandQueue?.makeCommandBuffer()

// Create an encoder to set vaulues on the compute function
let commandEncoder = commandBuffer?.makeComputeCommandEncoder()
commandEncoder?.setComputePipelineState(additionComputePipelineState)

// Create the buffers to be sent to the gpu from our arrays
let arr1Buff = device?.makeBuffer(bytes: arr1, length: MemoryLayout<Float>.size * count, options: .storageModeShared)
let arr2Buff = device?.makeBuffer(bytes: arr2, length: MemoryLayout<Float>.size * count, options: .storageModeShared)
let arr3Buff = device?.makeBuffer(bytes: arr3, length: MemoryLayout<Float>.size * count, options: .storageModeShared)
let resultBuff = device?.makeBuffer(length: MemoryLayout<Float>.size * count, options: .storageModeShared)

// Set the parameters of our gpu function
commandEncoder?.setBuffer(arr1Buff, offset: 0, index: 0)
commandEncoder?.setBuffer(arr2Buff, offset: 0, index: 1)
commandEncoder?.setBuffer(arr3Buff, offset: 0, index: 2)
commandEncoder?.setBuffer(resultBuff, offset: 0, index: 3)

// Figure out how many threads we need to use for our operation
let threadsPerGrid = MTLSize(width: count, height: 1, depth: 1)
let maxThreadsPerThreadgroup = additionComputePipelineState.maxTotalThreadsPerThreadgroup  // 1024
let threadsPerThreadgroup = MTLSize(width: maxThreadsPerThreadgroup, height: 1, depth: 1)
commandEncoder?.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

// Tell the encoder that it is done encoding.  Now we can send this off to the gpu.
commandEncoder?.endEncoding()

// Push this command to the command queue for processing
commandBuffer?.commit()

// Wait until the gpu function completes before working with any of the data
commandBuffer?.waitUntilCompleted()

// Get the pointer to the beginning of our data
var resultBufferPointer = resultBuff?.contents().bindMemory(
  to: Float.self, capacity: MemoryLayout<Float>.size * count)

// Print out all of our new added together array information
for i in 0..<3 {
  print("\(arr1[i]) + \(arr2[i]) + \(arr3[i]) = \(Float(resultBufferPointer!.pointee) as Any)")
  resultBufferPointer = resultBufferPointer?.advanced(by: 1)
}

// Helper function
func getRandomArray() -> [Float] {
  var result = [Float].init(repeating: 0.0, count: count)
  for i in 0..<count {
    result[i] = Float(arc4random_uniform(10))
  }
  return result
}