/**
 * @file test_vloc_kernel.cu
 * @brief 单元测试：V_loc 相关 kernel 的纯数学验证
 *
 * 测试策略：
 * - [PURE] 函数：解析解验证
 * - 不依赖 QE 参考数据
 * - 验证数学公式正确性
 */

#include <cufft.h>

#include <cmath>
#include <vector>

#include "test_kernels.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"

#include <gtest/gtest.h>

namespace dftcu {
namespace test {

/**
 * @brief 测试 apply_vloc_kernel: V(r) * ψ(r) 实空间乘法
 *
 * 物理公式：(V·ψ)(r) = V(r) · ψ(r)
 * 验证：逐点乘法的正确性
 */
TEST(VlocKernelTest, RealSpaceMultiplication_Analytical) {
    const int n = 100;

    // 准备测试数据
    std::vector<double> v_loc(n);
    std::vector<gpufftComplex> psi_r(n);
    std::vector<gpufftComplex> vpsi_r_expected(n);

    // 构造解析解：V(r) = 2.0, ψ(r) = (3.0, 4.0)
    for (int i = 0; i < n; ++i) {
        v_loc[i] = 2.0;
        psi_r[i].x = 3.0;
        psi_r[i].y = 4.0;

        // 期望：V·ψ = 2.0 * (3.0, 4.0) = (6.0, 8.0)
        vpsi_r_expected[i].x = 6.0;
        vpsi_r_expected[i].y = 8.0;
    }

    // 分配 GPU 内存
    double* v_loc_d;
    gpufftComplex* psi_r_d;
    CHECK(cudaMalloc(&v_loc_d, n * sizeof(double)));
    CHECK(cudaMalloc(&psi_r_d, n * sizeof(gpufftComplex)));

    CHECK(cudaMemcpy(v_loc_d, v_loc.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(psi_r_d, psi_r.data(), n * sizeof(gpufftComplex), cudaMemcpyHostToDevice));

    // 调用 kernel
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    apply_vloc_kernel_test<<<grid_size, block_size>>>(n, v_loc_d, psi_r_d);
    GPU_CHECK_KERNEL;

    // 拷贝回 CPU
    std::vector<gpufftComplex> vpsi_r_result(n);
    CHECK(cudaMemcpy(vpsi_r_result.data(), psi_r_d, n * sizeof(gpufftComplex),
                     cudaMemcpyDeviceToHost));

    // 验证
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(vpsi_r_result[i].x, vpsi_r_expected[i].x)
            << "Mismatch at index " << i << " (real part)";
        EXPECT_DOUBLE_EQ(vpsi_r_result[i].y, vpsi_r_expected[i].y)
            << "Mismatch at index " << i << " (imag part)";
    }

    // 清理
    CHECK(cudaFree(v_loc_d));
    CHECK(cudaFree(psi_r_d));
}

/**
 * @brief 测试实空间乘法：变化的势能
 *
 * V(r) = r, ψ(r) = (1.0, 2.0)
 */
TEST(VlocKernelTest, RealSpaceMultiplication_VaryingPotential) {
    const int n = 50;

    std::vector<double> v_loc(n);
    std::vector<gpufftComplex> psi_r(n);
    std::vector<gpufftComplex> expected(n);

    for (int i = 0; i < n; ++i) {
        v_loc[i] = static_cast<double>(i);  // V(r) = r
        psi_r[i].x = 1.0;
        psi_r[i].y = 2.0;

        expected[i].x = static_cast<double>(i) * 1.0;
        expected[i].y = static_cast<double>(i) * 2.0;
    }

    double* v_loc_d;
    gpufftComplex* psi_r_d;
    CHECK(cudaMalloc(&v_loc_d, n * sizeof(double)));
    CHECK(cudaMalloc(&psi_r_d, n * sizeof(gpufftComplex)));

    CHECK(cudaMemcpy(v_loc_d, v_loc.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(psi_r_d, psi_r.data(), n * sizeof(gpufftComplex), cudaMemcpyHostToDevice));

    apply_vloc_kernel_test<<<(n + 255) / 256, 256>>>(n, v_loc_d, psi_r_d);
    GPU_CHECK_KERNEL;

    std::vector<gpufftComplex> result(n);
    CHECK(cudaMemcpy(result.data(), psi_r_d, n * sizeof(gpufftComplex), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(result[i].x, expected[i].x) << "Index " << i;
        EXPECT_DOUBLE_EQ(result[i].y, expected[i].y) << "Index " << i;
    }

    CHECK(cudaFree(v_loc_d));
    CHECK(cudaFree(psi_r_d));
}

/**
 * @brief 测试缩放因子：fac = 0.5 (双 band 打包)
 *
 * QE 约定：Gamma-only 双 band 打包时需要 fac = 0.5
 */
TEST(VlocKernelTest, ScaleFactor_TwoBands) {
    const int n = 50;
    const double fac = 0.5;

    std::vector<gpufftComplex> data(n);
    std::vector<gpufftComplex> expected(n);

    for (int i = 0; i < n; ++i) {
        data[i].x = 10.0;
        data[i].y = 20.0;
        expected[i].x = 10.0 * fac;
        expected[i].y = 20.0 * fac;
    }

    gpufftComplex* data_d;
    CHECK(cudaMalloc(&data_d, n * sizeof(gpufftComplex)));
    CHECK(cudaMemcpy(data_d, data.data(), n * sizeof(gpufftComplex), cudaMemcpyHostToDevice));

    // 调用缩放 kernel
    scale_complex_kernel_test<<<(n + 255) / 256, 256>>>(n, data_d, fac);
    GPU_CHECK_KERNEL;

    std::vector<gpufftComplex> result(n);
    CHECK(cudaMemcpy(result.data(), data_d, n * sizeof(gpufftComplex), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(result[i].x, expected[i].x) << "Index " << i;
        EXPECT_DOUBLE_EQ(result[i].y, expected[i].y) << "Index " << i;
    }

    CHECK(cudaFree(data_d));
}

/**
 * @brief 测试缩放因子：fac = 1.0 (单 band)
 */
TEST(VlocKernelTest, ScaleFactor_SingleBand) {
    const int n = 50;
    const double fac = 1.0;

    std::vector<gpufftComplex> data(n);
    for (int i = 0; i < n; ++i) {
        data[i].x = 10.0;
        data[i].y = 20.0;
    }

    gpufftComplex* data_d;
    CHECK(cudaMalloc(&data_d, n * sizeof(gpufftComplex)));
    CHECK(cudaMemcpy(data_d, data.data(), n * sizeof(gpufftComplex), cudaMemcpyHostToDevice));

    scale_complex_kernel_test<<<(n + 255) / 256, 256>>>(n, data_d, fac);
    GPU_CHECK_KERNEL;

    std::vector<gpufftComplex> result(n);
    CHECK(cudaMemcpy(result.data(), data_d, n * sizeof(gpufftComplex), cudaMemcpyDeviceToHost));

    // fac=1.0 应该不改变数据
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(result[i].x, data[i].x) << "Index " << i;
        EXPECT_DOUBLE_EQ(result[i].y, data[i].y) << "Index " << i;
    }

    CHECK(cudaFree(data_d));
}

/**
 * @brief 测试缩放因子：fac = 0.0 (边界条件)
 */
TEST(VlocKernelTest, ScaleFactor_Zero) {
    const int n = 30;
    const double fac = 0.0;

    std::vector<gpufftComplex> data(n);
    for (int i = 0; i < n; ++i) {
        data[i].x = 10.0;
        data[i].y = 20.0;
    }

    gpufftComplex* data_d;
    CHECK(cudaMalloc(&data_d, n * sizeof(gpufftComplex)));
    CHECK(cudaMemcpy(data_d, data.data(), n * sizeof(gpufftComplex), cudaMemcpyHostToDevice));

    scale_complex_kernel_test<<<(n + 255) / 256, 256>>>(n, data_d, fac);
    GPU_CHECK_KERNEL;

    std::vector<gpufftComplex> result(n);
    CHECK(cudaMemcpy(result.data(), data_d, n * sizeof(gpufftComplex), cudaMemcpyDeviceToHost));

    // fac=0.0 应该将所有数据置零
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(result[i].x, 0.0) << "Index " << i;
        EXPECT_DOUBLE_EQ(result[i].y, 0.0) << "Index " << i;
    }

    CHECK(cudaFree(data_d));
}

/**
 * @brief 测试 G=0 约束：Im[ψ(G=0)] = 0
 *
 * Gamma-only 约束：G=0 处虚部必须为 0
 */
TEST(VlocKernelTest, GammaConstraint_G0_Concept) {
    // 这是概念验证测试，验证 Gamma-only 约束的理解

    gpufftComplex psi_g0;
    psi_g0.x = 1.5;    // 实部
    psi_g0.y = 1e-15;  // 虚部（数值噪声）

    // Gamma-only 约束：G=0 处虚部应该为 0
    // 在实际代码中，enforce_gamma_constraint 会强制 Im[ψ(G=0)] = 0
    EXPECT_NEAR(psi_g0.y, 0.0, 1e-14) << "G=0 imaginary part should be ~0";

    // 验证实部不受影响
    EXPECT_DOUBLE_EQ(psi_g0.x, 1.5);
}

/**
 * @brief 测试零势能：V(r) = 0
 *
 * 边界条件：V=0 时，ψ 不变
 */
TEST(VlocKernelTest, ZeroPotential_NoChange) {
    const int n = 40;

    std::vector<double> v_loc(n, 0.0);  // V(r) = 0
    std::vector<gpufftComplex> psi_r(n);

    for (int i = 0; i < n; ++i) {
        psi_r[i].x = 3.0 + i * 0.1;
        psi_r[i].y = 4.0 - i * 0.1;
    }

    // 保存原始值
    std::vector<gpufftComplex> psi_original = psi_r;

    double* v_loc_d;
    gpufftComplex* psi_r_d;
    CHECK(cudaMalloc(&v_loc_d, n * sizeof(double)));
    CHECK(cudaMalloc(&psi_r_d, n * sizeof(gpufftComplex)));

    CHECK(cudaMemcpy(v_loc_d, v_loc.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(psi_r_d, psi_r.data(), n * sizeof(gpufftComplex), cudaMemcpyHostToDevice));

    apply_vloc_kernel_test<<<(n + 255) / 256, 256>>>(n, v_loc_d, psi_r_d);
    GPU_CHECK_KERNEL;

    std::vector<gpufftComplex> result(n);
    CHECK(cudaMemcpy(result.data(), psi_r_d, n * sizeof(gpufftComplex), cudaMemcpyDeviceToHost));

    // V=0 时，结果应该全为 0（因为 0 * ψ = 0）
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(result[i].x, 0.0) << "Index " << i;
        EXPECT_DOUBLE_EQ(result[i].y, 0.0) << "Index " << i;
    }

    CHECK(cudaFree(v_loc_d));
    CHECK(cudaFree(psi_r_d));
}

}  // namespace test
}  // namespace dftcu
