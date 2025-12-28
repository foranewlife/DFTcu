#pragma once

#include <stdexcept>
#include <string>

#include <cublas_v2.h>

// Helper macro for error checking
#define CUBLAS_SAFE_CALL(x)                                                        \
    do {                                                                           \
        cublasStatus_t result = x;                                                 \
        if (result != CUBLAS_STATUS_SUCCESS) {                                     \
            throw std::runtime_error("CUBLAS_SAFE_CALL failed with error code: " + \
                                     std::to_string(result));                      \
        }                                                                          \
    } while (0)

namespace dftcu {

class CublasManager {
  public:
    // Get the singleton instance
    static CublasManager& instance() {
        static CublasManager manager;
        return manager;
    }

    // Get the handle
    cublasHandle_t handle() { return handle_; }

    // Disable copy and assignment
    CublasManager(const CublasManager&) = delete;
    void operator=(const CublasManager&) = delete;

  private:
    CublasManager() { CUBLAS_SAFE_CALL(cublasCreate(&handle_)); }

    ~CublasManager() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    cublasHandle_t handle_;
};

// RAII helper to manage cuBLAS pointer mode
class CublasPointerModeGuard {
  public:
    CublasPointerModeGuard(cublasHandle_t handle, cublasPointerMode_t mode) : handle_(handle) {
        CUBLAS_SAFE_CALL(cublasSetPointerMode(handle_, mode));
    }
    ~CublasPointerModeGuard() {
        cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_DEVICE);  // Always reset to default
    }

    // Disable copy and assignment
    CublasPointerModeGuard(const CublasPointerModeGuard&) = delete;
    void operator=(const CublasPointerModeGuard&) = delete;

  private:
    cublasHandle_t handle_;
};

}  // namespace dftcu
