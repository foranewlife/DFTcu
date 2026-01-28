/**
 * @file test_kernels.cuh
 * @brief 测试辅助：暴露内部 kernel 供单元测试使用
 *
 * 注意：这些 kernel 仅用于测试，不应在生产代码中直接调用
 */

#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

namespace dftcu {
namespace test {

// 类型别名：与 DFTcu 内部一致
using gpufftComplex = cufftDoubleComplex;

/**
 * @brief 实空间乘法 kernel：V(r) * ψ(r)
 *
 * 公式：(V·ψ)(r) = V(r) · ψ(r)
 *
 * @param n 数组大小
 * @param v_loc 局域势 V(r) [n]
 * @param psi_r 波函数 ψ(r) [n]，原地修改
 */
__global__ void apply_vloc_kernel_test(size_t n, const double* v_loc, gpufftComplex* psi_r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        psi_r[i].x *= v_loc[i];
        psi_r[i].y *= v_loc[i];
    }
}

/**
 * @brief 缩放 kernel：data *= scale
 *
 * @param n 数组大小
 * @param data 复数数组 [n]，原地修改
 * @param scale 缩放因子
 */
__global__ void scale_complex_kernel_test(size_t n, gpufftComplex* data, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

}  // namespace test
}  // namespace dftcu
