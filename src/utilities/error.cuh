#pragma once
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include <cusolverDn.h>

namespace dftcu {

// ════════════════════════════════════════════════════════════════════════
// DFTcu 异常类型层次结构
// ════════════════════════════════════════════════════════════════════════

/**
 * @brief DFTcu 基础异常类
 *
 * 所有 DFTcu 特定的异常都继承自此类。
 * Python 绑定时可以统一捕获此类型。
 */
class DFTcuException : public std::runtime_error {
  public:
    explicit DFTcuException(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief 配置错误异常
 *
 * 用于参数验证失败、配置不合理等情况。
 * 例如：ecutrho < 4*ecutwfc, nbands > ngw
 */
class ConfigurationError : public DFTcuException {
  public:
    explicit ConfigurationError(const std::string& message) : DFTcuException(message) {}
};

/**
 * @brief 物理错误异常
 *
 * 用于物理约束违反、数值不合理等情况。
 * 例如：负密度、负能量（不合理的情况）
 */
class PhysicsError : public DFTcuException {
  public:
    explicit PhysicsError(const std::string& message) : DFTcuException(message) {}
};

/**
 * @brief 收敛错误异常
 *
 * 用于 SCF/Davidson 迭代不收敛的情况。
 */
class ConvergenceError : public DFTcuException {
  public:
    explicit ConvergenceError(const std::string& message) : DFTcuException(message) {}
};

/**
 * @brief 文件 I/O 错误异常
 *
 * 用于文件读写失败、格式错误等情况。
 */
class FileIOError : public DFTcuException {
  public:
    explicit FileIOError(const std::string& message) : DFTcuException(message) {}
};

/**
 * @brief CUDA 错误异常
 *
 * 用于 CUDA API 调用失败的情况。
 */
class CUDAError : public DFTcuException {
  public:
    explicit CUDAError(const std::string& message) : DFTcuException(message) {}
};

// ════════════════════════════════════════════════════════════════════════
// CUDA 错误处理
// ════════════════════════════════════════════════════════════════════════

inline void handle_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::string msg = "CUDA Error: ";
        msg += cudaGetErrorString(error);
        msg += " at " + std::string(file) + ":" + std::to_string(line);
        throw CUDAError(msg);
    }
}

inline void handle_cusolver_error(cusolverStatus_t status, const char* file, int line) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw CUDAError("cuSOLVER Error at " + std::string(file) + ":" + std::to_string(line));
    }
}

}  // namespace dftcu

#define CHECK(condition) dftcu::handle_cuda_error(condition, __FILE__, __LINE__)

#define CHECK_CUSOLVER(condition) dftcu::handle_cusolver_error(condition, __FILE__, __LINE__)

#define GPU_CHECK_KERNEL dftcu::handle_cuda_error(cudaGetLastError(), __FILE__, __LINE__)
