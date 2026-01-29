/**
 * @file test_kinetic_energy.cu
 * @brief 单元测试：动能算符 T|ψ⟩ = ½|G|² |ψ⟩
 *
 * 测试策略：
 * - 使用解析解验证核心 kernel 的正确性
 * - 不依赖 QE 参考数据
 * - 验证数学公式的精确实现
 */

#include <cmath>
#include <complex>
#include <vector>

#include "fixtures/test_fixtures.cuh"
#include "model/grid.cuh"
#include "model/wavefunction.cuh"
#include "test_kernels.cuh"
#include "utilities/constants.cuh"
#include "utilities/error.cuh"

#include <gtest/gtest.h>

namespace dftcu {
namespace test {

/**
 * @brief 动能算符单元测试 Fixture
 */
class KineticEnergyTest : public SiCMinimalFixture {
  protected:
    void SetUp() override {
        SiCMinimalFixture::SetUp();

        // 获取网格参数
        nnr_ = grid_->nnr();

        // 创建临时波函数以获取 npw
        Wavefunction temp_psi(*grid_, 1, 10.0);
        npw_ = temp_psi.num_pw();

        std::cout << "  [Fixture] Grid parameters:" << std::endl;
        std::cout << "    - npw (G-vectors): " << npw_ << std::endl;
        std::cout << "    - nnr (FFT grid): " << nnr_ << std::endl;
    }

    int npw_;
    size_t nnr_;
};

/**
 * @brief 测试 1：平面波精确性 - 单个 G-vector
 *
 * 对于平面波 ψ(G) = δ(G - G₀)，动能算符应该精确给出：
 * T|ψ⟩ = ½|G₀|² |ψ⟩
 */
TEST_F(KineticEnergyTest, PlaneWave_SingleGVector_Exact) {
    int num_bands = 1;
    double ecutwfc_ha = 10.0;

    // 创建波函数
    Wavefunction psi(*grid_, num_bands, ecutwfc_ha);
    Wavefunction h_psi(*grid_, num_bands, ecutwfc_ha);

    // 获取 Miller 指数和 g2kin
    std::vector<int> h = grid_->get_miller_h();
    std::vector<int> k = grid_->get_miller_k();
    std::vector<int> l = grid_->get_miller_l();
    std::vector<double> g2kin_host(npw_);
    CHECK(cudaMemcpy(g2kin_host.data(), grid_->g2kin(), npw_ * sizeof(double),
                     cudaMemcpyDeviceToHost));

    std::cout << "  Testing single plane wave for each G-vector..." << std::endl;

    // 获取 nl 映射
    std::vector<int> nl = grid_->get_nl_d();

    // 测试前 10 个 G-vector（包括 G=0）
    int num_test = std::min(10, npw_);
    for (int ig_test = 0; ig_test < num_test; ++ig_test) {
        // 设置单个平面波：ψ(G) = δ(G - G_test)
        // 使用 set_coefficients() 直接设置 FFT grid 数据
        std::vector<std::complex<double>> psi_coeffs(nnr_, std::complex<double>(0.0, 0.0));
        int ifft_test = nl[ig_test];
        psi_coeffs[ifft_test] = std::complex<double>(1.0, 0.0);

        psi.set_coefficients(psi_coeffs, 0);

        // 清零输出
        std::vector<std::complex<double>> zeros(nnr_, std::complex<double>(0.0, 0.0));
        h_psi.copy_from_host(zeros.data());

        // 调用 kernel
        const int block_size = 256;
        const int grid_size = (npw_ * num_bands + block_size - 1) / block_size;

        apply_kinetic_kernel_test<<<grid_size, block_size>>>(
            npw_, num_bands, nnr_, grid_->nl_d(), grid_->g2kin(), psi.data(), h_psi.data());
        GPU_CHECK_KERNEL;
        CHECK(cudaDeviceSynchronize());

        // 提取结果
        std::vector<std::complex<double>> result = h_psi.get_coefficients(0);

        double expected_real = g2kin_host[ig_test] * 1.0;  // g2kin * Re(ψ)
        double expected_imag = g2kin_host[ig_test] * 0.0;  // g2kin * Im(ψ)

        double actual_real = result[ifft_test].real();
        double actual_imag = result[ifft_test].imag();

        // 对于 G=0，动能应该为 0
        if (h[ig_test] == 0 && k[ig_test] == 0 && l[ig_test] == 0) {
            EXPECT_NEAR(actual_real, 0.0, 1e-15) << "G=0: kinetic energy should be zero";
            EXPECT_NEAR(actual_imag, 0.0, 1e-15) << "G=0: kinetic energy should be zero";
        } else {
            EXPECT_NEAR(actual_real, expected_real, 1e-15)
                << "G=(" << h[ig_test] << "," << k[ig_test] << "," << l[ig_test]
                << "): Real part mismatch";
            EXPECT_NEAR(actual_imag, expected_imag, 1e-15)
                << "G=(" << h[ig_test] << "," << k[ig_test] << "," << l[ig_test]
                << "): Imag part mismatch";
        }

        // 验证其他位置为零（允许一定的数值误差）
        int num_nonzero = 0;
        for (size_t i = 0; i < nnr_; ++i) {
            if (std::abs(result[i]) > 1e-15) {
                num_nonzero++;
            }
        }

        // 对于 G=0，所有位置都应该是零
        if (h[ig_test] == 0 && k[ig_test] == 0 && l[ig_test] == 0) {
            EXPECT_EQ(num_nonzero, 0) << "G=0: all positions should be zero";
        } else {
            EXPECT_EQ(num_nonzero, 1) << "Only one G-vector should be non-zero";
        }
    }

    std::cout << "  ✓ Plane wave exactness verified for " << num_test << " G-vectors" << std::endl;
}

/**
 * @brief 测试 2：G=0 边界条件
 *
 * 对于 G=0，动能应该为 0：T|ψ(G=0)⟩ = 0
 */
TEST_F(KineticEnergyTest, GZero_KineticEnergy_IsZero) {
    int num_bands = 1;
    double ecutwfc_ha = 10.0;

    Wavefunction psi(*grid_, num_bands, ecutwfc_ha);
    Wavefunction h_psi(*grid_, num_bands, ecutwfc_ha);

    // 设置 G=0 平面波
    std::vector<int> h_g0 = {0};
    std::vector<int> k_g0 = {0};
    std::vector<int> l_g0 = {0};
    std::vector<std::complex<double>> coeff = {std::complex<double>(1.0, 0.0)};

    psi.set_coefficients_miller(h_g0, k_g0, l_g0, coeff, false);

    // 清零输出
    std::vector<std::complex<double>> zeros(nnr_, std::complex<double>(0.0, 0.0));
    h_psi.copy_from_host(zeros.data());

    // 调用 kernel
    const int block_size = 256;
    const int grid_size = (npw_ * num_bands + block_size - 1) / block_size;

    apply_kinetic_kernel_test<<<grid_size, block_size>>>(npw_, num_bands, nnr_, grid_->nl_d(),
                                                         grid_->g2kin(), psi.data(), h_psi.data());
    GPU_CHECK_KERNEL;
    CHECK(cudaDeviceSynchronize());

    // 提取结果
    std::vector<std::complex<double>> result = h_psi.get_coefficients(0);

    // 验证 G=0 位置为零
    std::vector<int> nl = grid_->get_nl_d();
    int ifft_g0 = nl[0];  // 假设 G=0 在 ig=0

    EXPECT_NEAR(result[ifft_g0].real(), 0.0, 1e-15) << "G=0: Real part should be zero";
    EXPECT_NEAR(result[ifft_g0].imag(), 0.0, 1e-15) << "G=0: Imag part should be zero";

    std::cout << "  ✓ G=0 kinetic energy is zero" << std::endl;
}

/**
 * @brief 测试 3：多 band 独立性
 *
 * 每个 band 的动能计算应该独立，互不影响
 */
TEST_F(KineticEnergyTest, MultiBand_Independence) {
    int num_bands = 4;
    double ecutwfc_ha = 10.0;

    Wavefunction psi(*grid_, num_bands, ecutwfc_ha);
    Wavefunction h_psi(*grid_, num_bands, ecutwfc_ha);

    // 获取 Miller 指数和 g2kin
    std::vector<int> h = grid_->get_miller_h();
    std::vector<int> k = grid_->get_miller_k();
    std::vector<int> l = grid_->get_miller_l();
    std::vector<int> nl = grid_->get_nl_d();
    std::vector<double> g2kin_host(npw_);
    CHECK(cudaMemcpy(g2kin_host.data(), grid_->g2kin(), npw_ * sizeof(double),
                     cudaMemcpyDeviceToHost));

    // 为每个 band 设置不同的单平面波
    // 使用 set_coefficients() 方法逐个 band 设置
    std::vector<int> test_ig = {0, 1, 2, 3};  // 测试前 4 个 G-vector

    for (int band = 0; band < num_bands; ++band) {
        int ig = test_ig[band];
        int ifft = nl[ig];

        // 构造单个 band 的系数数组（FFT grid 大小）
        std::vector<std::complex<double>> band_coeffs(nnr_, std::complex<double>(0.0, 0.0));
        band_coeffs[ifft] = std::complex<double>(1.0 + band * 0.1, 0.5 + band * 0.1);

        // 设置单个 band
        psi.set_coefficients(band_coeffs, band);
    }

    // 清零输出
    std::vector<std::complex<double>> zeros(nnr_ * num_bands, std::complex<double>(0.0, 0.0));
    h_psi.copy_from_host(zeros.data());

    // 调用 kernel
    const int block_size = 256;
    const int grid_size = (npw_ * num_bands + block_size - 1) / block_size;

    apply_kinetic_kernel_test<<<grid_size, block_size>>>(npw_, num_bands, nnr_, grid_->nl_d(),
                                                         grid_->g2kin(), psi.data(), h_psi.data());
    GPU_CHECK_KERNEL;
    CHECK(cudaDeviceSynchronize());

    // 验证每个 band 独立计算
    std::cout << "  Verifying multi-band independence..." << std::endl;

    for (int band = 0; band < num_bands; ++band) {
        int ig = test_ig[band];
        int ifft = nl[ig];

        std::vector<std::complex<double>> result = h_psi.get_coefficients(band);

        double psi_real = 1.0 + band * 0.1;
        double psi_imag = 0.5 + band * 0.1;
        double g2 = g2kin_host[ig];

        double expected_real = g2 * psi_real;
        double expected_imag = g2 * psi_imag;

        EXPECT_NEAR(result[ifft].real(), expected_real, 1e-14)
            << "Band " << band << ": Real part mismatch";
        EXPECT_NEAR(result[ifft].imag(), expected_imag, 1e-14)
            << "Band " << band << ": Imag part mismatch";

        std::cout << "    Band " << band << ": ψ=(" << psi_real << "," << psi_imag << "), T|ψ⟩=("
                  << result[ifft].real() << "," << result[ifft].imag() << "), expected=("
                  << expected_real << "," << expected_imag << ")" << std::endl;
    }

    std::cout << "  ✓ Multi-band independence verified" << std::endl;
}

/**
 * @brief 测试 4：线性叠加性
 *
 * T(c₁ψ₁ + c₂ψ₂) = c₁T|ψ₁⟩ + c₂T|ψ₂⟩
 */
TEST_F(KineticEnergyTest, Linearity_Superposition) {
    int num_bands = 1;
    double ecutwfc_ha = 10.0;

    // 获取 Miller 指数和 g2kin
    std::vector<int> h = grid_->get_miller_h();
    std::vector<int> k = grid_->get_miller_k();
    std::vector<int> l = grid_->get_miller_l();
    std::vector<int> nl = grid_->get_nl_d();
    std::vector<double> g2kin_host(npw_);
    CHECK(cudaMemcpy(g2kin_host.data(), grid_->g2kin(), npw_ * sizeof(double),
                     cudaMemcpyDeviceToHost));

    // 选择两个 G-vector
    int ig1 = 1, ig2 = 2;
    std::complex<double> c1(2.0, 1.0);
    std::complex<double> c2(1.5, -0.5);

    // 构造叠加态：ψ = c₁|G₁⟩ + c₂|G₂⟩
    Wavefunction psi(*grid_, num_bands, ecutwfc_ha);
    Wavefunction h_psi(*grid_, num_bands, ecutwfc_ha);

    std::vector<int> h_super = {h[ig1], h[ig2]};
    std::vector<int> k_super = {k[ig1], k[ig2]};
    std::vector<int> l_super = {l[ig1], l[ig2]};
    std::vector<std::complex<double>> coeffs = {c1, c2};

    psi.set_coefficients_miller(h_super, k_super, l_super, coeffs, false);

    // 清零输出
    std::vector<std::complex<double>> zeros(nnr_, std::complex<double>(0.0, 0.0));
    h_psi.copy_from_host(zeros.data());

    // 调用 kernel
    const int block_size = 256;
    const int grid_size = (npw_ * num_bands + block_size - 1) / block_size;

    apply_kinetic_kernel_test<<<grid_size, block_size>>>(npw_, num_bands, nnr_, grid_->nl_d(),
                                                         grid_->g2kin(), psi.data(), h_psi.data());
    GPU_CHECK_KERNEL;
    CHECK(cudaDeviceSynchronize());

    // 提取结果
    std::vector<std::complex<double>> result = h_psi.get_coefficients(0);

    // 验证线性性：T|ψ⟩ = c₁·g2kin[ig1]·|G₁⟩ + c₂·g2kin[ig2]·|G₂⟩
    int ifft1 = nl[ig1];
    int ifft2 = nl[ig2];

    std::complex<double> expected1 = c1 * g2kin_host[ig1];
    std::complex<double> expected2 = c2 * g2kin_host[ig2];

    EXPECT_NEAR(result[ifft1].real(), expected1.real(), 1e-14) << "G₁: Real part linearity failed";
    EXPECT_NEAR(result[ifft1].imag(), expected1.imag(), 1e-14) << "G₁: Imag part linearity failed";
    EXPECT_NEAR(result[ifft2].real(), expected2.real(), 1e-14) << "G₂: Real part linearity failed";
    EXPECT_NEAR(result[ifft2].imag(), expected2.imag(), 1e-14) << "G₂: Imag part linearity failed";

    std::cout << "  ✓ Linearity (superposition) verified" << std::endl;
}

/**
 * @brief 测试 5：能量期望值
 *
 * ⟨ψ|T|ψ⟩ = Σ_G ½|G|² |ψ(G)|²
 */
TEST_F(KineticEnergyTest, EnergyExpectation_Formula) {
    int num_bands = 1;
    double ecutwfc_ha = 10.0;

    Wavefunction psi(*grid_, num_bands, ecutwfc_ha);
    Wavefunction h_psi(*grid_, num_bands, ecutwfc_ha);

    // 获取 Miller 指数和 g2kin
    std::vector<int> h = grid_->get_miller_h();
    std::vector<int> k = grid_->get_miller_k();
    std::vector<int> l = grid_->get_miller_l();
    std::vector<int> nl = grid_->get_nl_d();
    std::vector<double> g2kin_host(npw_);
    CHECK(cudaMemcpy(g2kin_host.data(), grid_->g2kin(), npw_ * sizeof(double),
                     cudaMemcpyDeviceToHost));

    // 构造随机波函数（前 10 个 G-vector）
    int num_g = std::min(10, npw_);
    std::vector<int> h_test, k_test, l_test;
    std::vector<std::complex<double>> coeffs;

    for (int ig = 0; ig < num_g; ++ig) {
        h_test.push_back(h[ig]);
        k_test.push_back(k[ig]);
        l_test.push_back(l[ig]);
        coeffs.push_back(std::complex<double>(0.1 * (ig + 1), 0.05 * ig));
    }

    psi.set_coefficients_miller(h_test, k_test, l_test, coeffs, false);

    // 清零输出
    std::vector<std::complex<double>> zeros(nnr_, std::complex<double>(0.0, 0.0));
    h_psi.copy_from_host(zeros.data());

    // 调用 kernel
    const int block_size = 256;
    const int grid_size = (npw_ * num_bands + block_size - 1) / block_size;

    apply_kinetic_kernel_test<<<grid_size, block_size>>>(npw_, num_bands, nnr_, grid_->nl_d(),
                                                         grid_->g2kin(), psi.data(), h_psi.data());
    GPU_CHECK_KERNEL;
    CHECK(cudaDeviceSynchronize());

    // 提取结果
    std::vector<std::complex<double>> psi_result = psi.get_coefficients(0);
    std::vector<std::complex<double>> h_psi_result = h_psi.get_coefficients(0);

    // 计算能量期望值：⟨ψ|T|ψ⟩ = Σ ψ*(G) · T|ψ⟩(G)
    double energy_computed = 0.0;
    double energy_expected = 0.0;

    for (int ig = 0; ig < num_g; ++ig) {
        int ifft = nl[ig];
        std::complex<double> psi_val = psi_result[ifft];
        std::complex<double> tpsi_val = h_psi_result[ifft];

        // ⟨ψ|T|ψ⟩ += ψ*(G) · T|ψ⟩(G)
        energy_computed += (std::conj(psi_val) * tpsi_val).real();

        // 直接公式：Σ ½|G|² |ψ(G)|²
        energy_expected += g2kin_host[ig] * std::norm(psi_val);
    }

    std::cout << "  Energy expectation:" << std::endl;
    std::cout << "    - Computed: " << energy_computed << " Ha" << std::endl;
    std::cout << "    - Expected: " << energy_expected << " Ha" << std::endl;
    std::cout << "    - Difference: " << std::abs(energy_computed - energy_expected) << " Ha"
              << std::endl;

    EXPECT_NEAR(energy_computed, energy_expected, 1e-14) << "Energy expectation value mismatch";

    std::cout << "  ✓ Energy expectation formula verified" << std::endl;
}

/**
 * @brief 测试 6：零波函数边界条件
 *
 * T|0⟩ = 0
 */
TEST_F(KineticEnergyTest, ZeroWavefunction_ReturnsZero) {
    int num_bands = 1;
    double ecutwfc_ha = 10.0;

    Wavefunction psi(*grid_, num_bands, ecutwfc_ha);
    Wavefunction h_psi(*grid_, num_bands, ecutwfc_ha);

    // 设置零波函数（默认已经是零）
    std::vector<std::complex<double>> zeros(nnr_, std::complex<double>(0.0, 0.0));
    psi.copy_from_host(zeros.data());
    h_psi.copy_from_host(zeros.data());

    // 调用 kernel
    const int block_size = 256;
    const int grid_size = (npw_ * num_bands + block_size - 1) / block_size;

    apply_kinetic_kernel_test<<<grid_size, block_size>>>(npw_, num_bands, nnr_, grid_->nl_d(),
                                                         grid_->g2kin(), psi.data(), h_psi.data());
    GPU_CHECK_KERNEL;
    CHECK(cudaDeviceSynchronize());

    // 提取结果
    std::vector<std::complex<double>> result = h_psi.get_coefficients(0);

    // 验证所有位置为零
    for (size_t i = 0; i < nnr_; ++i) {
        EXPECT_NEAR(result[i].real(), 0.0, 1e-15) << "Index " << i << ": Real part not zero";
        EXPECT_NEAR(result[i].imag(), 0.0, 1e-15) << "Index " << i << ": Imag part not zero";
    }

    std::cout << "  ✓ Zero wavefunction returns zero" << std::endl;
}

}  // namespace test
}  // namespace dftcu
