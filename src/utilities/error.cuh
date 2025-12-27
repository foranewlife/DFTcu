#pragma once
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace dftcu {

inline void handle_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::string msg = "CUDA Error: ";
        msg += cudaGetErrorString(error);
        msg += " at " + std::string(file) + ":" + std::to_string(line);
        throw std::runtime_error(msg);
    }
}

}  // namespace dftcu

#define CHECK(condition) dftcu::handle_cuda_error(condition, __FILE__, __LINE__)

#define GPU_CHECK_KERNEL dftcu::handle_cuda_error(cudaGetLastError(), __FILE__, __LINE__)
