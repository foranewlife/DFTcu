#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <sstream>

#include "fixtures/test_data_loader.cuh"
#include "fixtures/test_fixtures.cuh"
#include "functional/xc/lda_pz.cuh"
#include "utilities/constants.cuh"

#include <gtest/gtest.h>

namespace dftcu {
namespace test {

// ════════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ════════════════════════════════════════════════════════════════════════════════

/**
 * @brief 加载标量能量值 (e.g., E_xc, E_H)
 * @param filename 文件路径
 * @return 能量值
 */
static double load_scalar_energy(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::stringstream ss(line);
        double value;
        if (ss >> value) {
            return value;
        }
    }
    throw std::runtime_error("No valid data in file: " + filename);
}

// ════════════════════════════════════════════════════════════════════════════════
// Integration Test: LDA_PZ::compute() vs QE Reference
// ════════════════════════════════════════════════════════════════════════════════

class LdaPzIntegrationTest : public SiCFixture {
  protected:
    void SetUp() override {
        SiCFixture::SetUp();

        // 加载 QE 参考数据
        rho_ref_ = StandardDataLoader::load_real("tests/data/qe_reference/sic_minimal/rho_r.dat");
        v_xc_ref_ = StandardDataLoader::load_real("tests/data/qe_reference/sic_minimal/v_xc_r.dat");

        ASSERT_GT(rho_ref_.size(), 0) << "Failed to load rho_r.dat";
        ASSERT_GT(v_xc_ref_.size(), 0) << "Failed to load v_xc_r.dat";
        ASSERT_EQ(rho_ref_.size(), 5832) << "Expected 18³ = 5832 points";

        // 加载参考能量
        e_xc_ref_ry_ = load_scalar_energy("tests/data/qe_reference/sic_minimal/e_xc.dat");
        e_xc_ref_ha_ = e_xc_ref_ry_ * 0.5;  // Ry → Ha
    }

    std::vector<IndexedRealValue> rho_ref_;
    std::vector<IndexedRealValue> v_xc_ref_;
    double e_xc_ref_ry_;
    double e_xc_ref_ha_;
};

// ────────────────────────────────────────────────────────────────────────────────
// Integration Test 1: QE Reference Data Validity
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(LdaPzIntegrationTest, QEReference_DataValidity) {
    // 验证 rho 参考数据物理合理性
    double rho_min = 1e30, rho_max = -1e30, rho_sum = 0.0;
    for (const auto& v : rho_ref_) {
        rho_min = std::min(rho_min, v.value);
        rho_max = std::max(rho_max, v.value);
        rho_sum += v.value;
    }

    std::cout << "[INFO] rho range: [" << rho_min << ", " << rho_max << "]" << std::endl;
    std::cout << "[INFO] rho sum: " << rho_sum << std::endl;

    // 电子密度应该非负（QE first SCF 可能有小的负值）
    EXPECT_GT(rho_min, -0.01) << "Density should be approximately non-negative";
    EXPECT_GT(rho_max, 0.0) << "Max density should be positive";

    // 积分应该约等于电子数 (SiC: Si 4e + C 4e = 8e)
    double dv = grid_->dv_bohr();  // Bohr³
    double n_electrons = rho_sum * dv;
    std::cout << "[INFO] Integrated electrons: " << n_electrons << std::endl;
    EXPECT_NEAR(n_electrons, 8.0, 0.5);  // 允许 0.5 电子误差（first SCF）
}

// ────────────────────────────────────────────────────────────────────────────────
// Integration Test 2: V_xc QE Reference Data Validity
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(LdaPzIntegrationTest, QEReference_VxcDataValidity) {
    double v_min = 1e30, v_max = -1e30;
    for (const auto& v : v_xc_ref_) {
        v_min = std::min(v_min, v.value);
        v_max = std::max(v_max, v.value);
    }

    std::cout << "[INFO] V_xc range (Ry): [" << v_min << ", " << v_max << "]" << std::endl;

    // XC 势应该为负
    EXPECT_LT(v_max, 0.0) << "V_xc should be negative everywhere";
    // 典型范围 -0.5 到 -2.0 Ry
    EXPECT_GT(v_min, -5.0) << "V_xc should not be too negative";
}

// ────────────────────────────────────────────────────────────────────────────────
// Integration Test 3: E_xc Energy QE Comparison
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(LdaPzIntegrationTest, ExcEnergy_QEComparison) {
    // 从 QE 参考密度计算 E_xc

    // 将 QE 密度放入 GPU
    std::vector<double> rho_host(grid_->nnr(), 0.0);
    for (const auto& v : rho_ref_) {
        // 将 (ix, iy, iz) 转换为线性索引
        int idx = v.ix + v.iy * grid_->nr()[0] + v.iz * grid_->nr()[0] * grid_->nr()[1];
        rho_host[idx] = v.value;
    }

    RealField rho_field(*grid_);
    RealField v_xc_field(*grid_);
    cudaMemcpy(rho_field.data(), rho_host.data(), rho_host.size() * sizeof(double),
               cudaMemcpyHostToDevice);

    // 计算 XC
    LDA_PZ lda_pz;
    double e_xc_ha = lda_pz.compute(rho_field, v_xc_field);

    std::cout << "[INFO] E_xc (DFTcu): " << e_xc_ha << " Ha" << std::endl;
    std::cout << "[INFO] E_xc (QE ref): " << e_xc_ref_ha_ << " Ha (" << e_xc_ref_ry_ << " Ry)"
              << std::endl;

    double diff_ha = std::abs(e_xc_ha - e_xc_ref_ha_);
    double diff_mev = diff_ha * 27211.4;  // 1 Ha = 27.2114 eV = 27211.4 meV

    std::cout << "[INFO] |E_xc diff|: " << diff_ha << " Ha (" << diff_mev << " meV)" << std::endl;

    // 目标精度: < 1 meV
    EXPECT_LT(diff_mev, 1.0) << "E_xc difference should be < 1 meV";
}

// ────────────────────────────────────────────────────────────────────────────────
// Integration Test 4: V_xc Point-by-Point Comparison
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(LdaPzIntegrationTest, Vxc_PointByPoint_QEComparison) {
    // 将 QE 密度放入 GPU
    std::vector<double> rho_host(grid_->nnr(), 0.0);
    for (const auto& v : rho_ref_) {
        int idx = v.ix + v.iy * grid_->nr()[0] + v.iz * grid_->nr()[0] * grid_->nr()[1];
        rho_host[idx] = v.value;
    }

    RealField rho_field(*grid_);
    RealField v_xc_field(*grid_);
    cudaMemcpy(rho_field.data(), rho_host.data(), rho_host.size() * sizeof(double),
               cudaMemcpyHostToDevice);

    // 计算 XC
    LDA_PZ lda_pz;
    lda_pz.compute(rho_field, v_xc_field);

    // 拷贝回 CPU
    std::vector<double> v_xc_host(grid_->nnr());
    cudaMemcpy(v_xc_host.data(), v_xc_field.data(), v_xc_host.size() * sizeof(double),
               cudaMemcpyDeviceToHost);

    // 比较每个点 (DFTcu 输出 Ha, QE 参考 Ry)
    double max_diff_ry = 0.0;
    int max_diff_ix = 0, max_diff_iy = 0, max_diff_iz = 0;
    double sum_sq_diff = 0.0;

    for (const auto& v : v_xc_ref_) {
        int idx = v.ix + v.iy * grid_->nr()[0] + v.iz * grid_->nr()[0] * grid_->nr()[1];

        // DFTcu 输出 Ha, 转换为 Ry 比较
        double v_xc_dftcu_ry = v_xc_host[idx] * 2.0;  // Ha → Ry
        double v_xc_qe_ry = v.value;

        double diff = std::abs(v_xc_dftcu_ry - v_xc_qe_ry);
        if (diff > max_diff_ry) {
            max_diff_ry = diff;
            max_diff_ix = v.ix;
            max_diff_iy = v.iy;
            max_diff_iz = v.iz;
        }
        sum_sq_diff += diff * diff;
    }

    double rms_diff_ry = std::sqrt(sum_sq_diff / v_xc_ref_.size());
    double max_diff_mev = max_diff_ry * 0.5 * 27211.4;  // Ry → Ha → meV

    std::cout << "[INFO] V_xc max diff: " << max_diff_ry << " Ry at (" << max_diff_ix << ", "
              << max_diff_iy << ", " << max_diff_iz << ")" << std::endl;
    std::cout << "[INFO] V_xc RMS diff: " << rms_diff_ry << " Ry" << std::endl;
    std::cout << "[INFO] V_xc max diff: " << max_diff_mev << " meV" << std::endl;

    // 目标精度: 最大误差 < 10 meV
    EXPECT_LT(max_diff_mev, 10.0) << "V_xc max difference should be < 10 meV";
}

}  // namespace test
}  // namespace dftcu
