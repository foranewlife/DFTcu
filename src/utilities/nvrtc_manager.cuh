#pragma once

#include <cuda.h>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <nvrtc.h>

// Helper macros for error checking
#define NVRTC_SAFE_CALL(x)                                                      \
    do {                                                                        \
        nvrtcResult result = x;                                                 \
        if (result != NVRTC_SUCCESS) {                                          \
            throw std::runtime_error("NVRTC_SAFE_CALL failed: " +               \
                                     std::string(nvrtcGetErrorString(result))); \
        }                                                                       \
    } while (0)

#define CUDA_SAFE_CALL(x)             \
    do {                              \
        CUresult result = x;          \
        if (result != CUDA_SUCCESS) { \
            const char* msg;
cuGetErrorName(result, &msg);
throw std::runtime_error("CUDA_SAFE_CALL failed: " + std::string(msg));
}
}
while (0)

    namespace dftcu {

    class NVRTC_Manager {
      public:
        // Get the singleton instance
        static NVRTC_Manager& instance() {
            static NVRTC_Manager manager;
            return manager;
        }

        // Compile and retrieve a kernel function
        CUfunction get_kernel(const std::string& kernel_name, const std::string& kernel_source) {
            auto it = cache_.find(kernel_name);
            if (it != cache_.end()) {
                return it->second;
            }

            // Create and compile the program
            nvrtcProgram prog;
            NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernel_source.c_str(), kernel_name.c_str(), 0,
                                               NULL, NULL));

            // Get the compute capability of the current device
            int major = 0, minor = 0;
            cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
            cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);
            std::string arch =
                "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
            std::vector<const char*> opts = {arch.c_str(), "--fmad=true", "--use_fast_math"};

            nvrtcResult compile_res = nvrtcCompileProgram(prog, opts.size(), opts.data());
            if (compile_res != NVRTC_SUCCESS) {
                size_t log_size;
                NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));
                std::vector<char> log(log_size);
                NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.data()));
                std::cerr << "NVRTC compilation failed:\n" << log.data() << std::endl;
                throw std::runtime_error("NVRTC compilation failed.");
            }

            // Get PTX
            size_t ptx_size;
            NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptx_size));
            std::vector<char> ptx(ptx_size);
            NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));

            // Load module and get function
            CUmodule module;
            CUfunction kernel;
            CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
            CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, kernel_name.c_str()));

            // Store in cache
            cache_[kernel_name] = kernel;

            // Note: We don't destroy the program or module, as they are needed for the kernel
            // A more robust implementation would manage their lifetime, e.g., in a destructor
            return kernel;
        }

        // Disable copy and assignment
        NVRTC_Manager(const NVRTC_Manager&) = delete;
        void operator=(const NVRTC_Manager&) = delete;

      private:
        NVRTC_Manager() {
            // Initialize CUDA driver API
            CUDA_SAFE_CALL(cuInit(0));
        }
        ~NVRTC_Manager() = default;  // TODO: Add module/program cleanup

        std::map<std::string, CUfunction> cache_;
    };

    }  // namespace dftcu
