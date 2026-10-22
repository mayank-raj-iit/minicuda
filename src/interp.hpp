// interp.hpp - run IR on the CPU, simulating a CUDA grid of thread blocks.
//
// This lets us test the compiler without a GPU: the same kernel can be run
// here and on a real GPU, and the results compared. It also checks every
// memory access (like NVIDIA's compute-sanitizer) and reports out-of-bounds
// reads/writes with the thread and block that did it.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ir.hpp"

struct DeviceMemory {
  struct Buffer {
    uint64_t base;
    std::vector<uint8_t> bytes;
  };
  std::vector<Buffer> buffers;
  uint64_t nextBase = 0x10000000;

  uint64_t alloc(size_t bytes);                  // returns a fake device address
  Buffer* find(uint64_t addr, size_t size);      // nullptr if out of bounds
};

struct KernelArg {
  IRType type = IRType::I32;   // I32, F32 or I64 (a device pointer)
  int32_t i = 0;
  float f = 0;
  uint64_t ptr = 0;
};

struct Dim3 { unsigned x = 1, y = 1, z = 1; };

struct RunError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct RunStats {
  uint64_t instructions = 0;     // dynamic instruction count, all threads
  uint64_t barriers = 0;
};

RunStats interpret(const Function& fn, DeviceMemory& mem, Dim3 grid, Dim3 block,
                   const std::vector<KernelArg>& args);
