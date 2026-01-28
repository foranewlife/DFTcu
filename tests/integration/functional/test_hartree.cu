#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <sstream>

#include "fixtures/test_data_loader.cuh"
#include "fixtures/test_fixtures.cuh"
#include "functional/hartree.cuh"
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
// Integration Test: Hartree::compute() vs QE Reference
// ════════════════════════════════════════════════════════════════════════════════

class HartreeIntegrationTest : public SiCFixture {
  protected:
    void SetUp() override {
        SiCFixture::SetUp();

        // 加载 QE 参考数据
        rho_ref_ = StandardDataLoader::load_real("tests/data/qe_reference/sic_minimal/rho_r.dat");
        v_h_ref_ = StandardDataLoader::load_real("tests/data/qe_reference/sic_minimal/v_h_r.dat");

        ASSERT_GT(rho_ref_.size(), 0) << "Failed to load rho_r.dat";
        ASSERT_GT(v_h_ref_.size(), 0) << "Failed to load v_h_r.dat";
        ASSERT_EQ(rho_ref_.size(), 5832) << "Expected 18³ = 5832 points";

        // 加载参考能量
        e_h_ref_ry_ = load_scalar_energy("tests/data/qe_reference/sic_minimal/e_hartree.dat");
        e_h_ref_ha_ = e_h_ref_ry_ * 0.5;  // Ry → Ha
    }

    std::vector<IndexedRealValue> rho_ref_;
    std::vector<IndexedRealValue> v_h_ref_;
    double e_h_ref_ry_;
    double e_h_ref_ha_;
};

// ────────────────────────────────────────────────────────────────────────────────
// Integration Test 1: QE Reference Data Validity
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(HartreeIntegrationTest, QEReference_DataValidity) {
    // 验证 V_H 参考数据物理合理性
    double v_min = 1e30, v_max = -1e30;
    for (const auto& v : v_h_ref_) {
        v_min = std::min(v_min, v.value);
        v_max = std::max(v_max, v.value);
    }

    std::cout << "[INFO] V_H range (Ry): [" << v_min << ", " << v_max << "]" << std::endl;
    std::cout << "[INFO] E_H reference: " << e_h_ref_ry_ << " Ry (" << e_h_ref_ha_ << " Ha)"
              << std::endl;

    // Hartree 势可以有负值（取决于参考点选择）
    // 但能量应该为正
    EXPECT_GT(e_h_ref_ry_, 0.0) << "E_H should be positive";
}

// ────────────────────────────────────────────────────────────────────────────────
// Integration Test 2: E_H Energy QE Comparison
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(HartreeIntegrationTest, EhEnergy_QEComparison) {
    // 从 QE 参考密度计算 E_H

    // 将 QE 密度放入 GPU
    std::vector<double> rho_host(grid_->nnr(), 0.0);
    for (const auto& v : rho_ref_) {
        int idx = v.ix + v.iy * grid_->nr()[0] + v.iz * grid_->nr()[0] * grid_->nr()[1];
        rho_host[idx] = v.value;
    }

    RealField rho_field(*grid_);
    RealField v_h_field(*grid_);
    cudaMemcpy(rho_field.data(), rho_host.data(), rho_host.size() * sizeof(double),
               cudaMemcpyHostToDevice);

    // 计算 Hartree
    Hartree hartree;
    double e_h_ha = hartree.compute(rho_field, v_h_field);

    std::cout << "[INFO] E_H (DFTcu): " << e_h_ha << " Ha (" << e_h_ha * 2.0 << " Ry)" << std::endl;
    std::cout << "[INFO] E_H (QE ref): " << e_h_ref_ha_ << " Ha (" << e_h_ref_ry_ << " Ry)"
              << std::endl;

    double diff_ha = std::abs(e_h_ha - e_h_ref_ha_);
    double diff_mev = diff_ha * 27211.4;  // 1 Ha = 27211.4 meV

    std::cout << "[INFO] |E_H diff|: " << diff_ha << " Ha (" << diff_mev << " meV)" << std::endl;

    // 目标精度: < 1 meV
    EXPECT_LT(diff_mev, 1.0) << "E_H difference should be < 1 meV";
}

// ────────────────────────────────────────────────────────────────────────────────
// Integration Test 3: V_H Point-by-Point Comparison
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(HartreeIntegrationTest, Vh_PointByPoint_QEComparison) {
    // 将 QE 密度放入 GPU
    std::vector<double> rho_host(grid_->nnr(), 0.0);
    for (const auto& v : rho_ref_) {
        int idx = v.ix + v.iy * grid_->nr()[0] + v.iz * grid_->nr()[0] * grid_->nr()[1];
        rho_host[idx] = v.value;
    }

    RealField rho_field(*grid_);
    RealField v_h_field(*grid_);
    cudaMemcpy(rho_field.data(), rho_host.data(), rho_host.size() * sizeof(double),
               cudaMemcpyHostToDevice);

    // 计算 Hartree
    Hartree hartree;
    hartree.compute(rho_field, v_h_field);

    // 拷贝回 CPU
    std::vector<double> v_h_host(grid_->nnr());
    cudaMemcpy(v_h_host.data(), v_h_field.data(), v_h_host.size() * sizeof(double),
               cudaMemcpyDeviceToHost);

    // 比较每个点 (DFTcu 输出 Ha, QE 参考 Ry)
    // DFTcu Hartree 势输出 Ha，需要 *2 转为 Ry
    double max_diff_ry = 0.0;
    int max_diff_ix = 0, max_diff_iy = 0, max_diff_iz = 0;
    double sum_sq_diff = 0.0;

    for (const auto& v : v_h_ref_) {
        int idx = v.ix + v.iy * grid_->nr()[0] + v.iz * grid_->nr()[0] * grid_->nr()[1];

        // DFTcu Hartree::compute() 输出 Ha，转换为 Ry
        double v_h_dftcu_ry = v_h_host[idx] * 2.0;  // Ha → Ry
        double v_h_qe_ry = v.value;

        double diff = std::abs(v_h_dftcu_ry - v_h_qe_ry);
        if (diff > max_diff_ry) {
            max_diff_ry = diff;
            max_diff_ix = v.ix;
            max_diff_iy = v.iy;
            max_diff_iz = v.iz;
        }
        sum_sq_diff += diff * diff;
    }

    double rms_diff_ry = std::sqrt(sum_sq_diff / v_h_ref_.size());
    double max_diff_mev = max_diff_ry * 0.5 * 27211.4;  // Ry → Ha → meV

    std::cout << "[INFO] V_H max diff: " << max_diff_ry << " Ry at (" << max_diff_ix << ", "
              << max_diff_iy << ", " << max_diff_iz << ")" << std::endl;
    std::cout << "[INFO] V_H RMS diff: " << rms_diff_ry << " Ry" << std::endl;
    std::cout << "[INFO] V_H max diff: " << max_diff_mev << " meV" << std::endl;

    // 目标精度: 最大误差 < 10 meV
    EXPECT_LT(max_diff_mev, 10.0) << "V_H max difference should be < 10 meV";
}

// ────────────────────────────────────────────────────────────────────────────────
// Integration Test 4: V_H Sample Points Comparison
// ────────────────────────────────────────────────────────────────────────────────
TEST_F(HartreeIntegrationTest, Vh_SamplePoints_Comparison) {
    // 将 QE 密度放入 GPU
    std::vector<double> rho_host(grid_->nnr(), 0.0);
    for (const auto& v : rho_ref_) {
        int idx = v.ix + v.iy * grid_->nr()[0] + v.iz * grid_->nr()[0] * grid_->nr()[1];
        rho_host[idx] = v.value;
    }

    RealField rho_field(*grid_);
    RealField v_h_field(*grid_);
    cudaMemcpy(rho_field.data(), rho_host.data(), rho_host.size() * sizeof(double),
               cudaMemcpyHostToDevice);

    // 计算 Hartree
    Hartree hartree;
    hartree.compute(rho_field, v_h_field);

    // 拷贝回 CPU
    std::vector<double> v_h_host(grid_->nnr());
    cudaMemcpy(v_h_host.data(), v_h_field.data(), v_h_host.size() * sizeof(double),
               cudaMemcpyDeviceToHost);

    // 采样关键点进行详细对比
    // 注意: DFTcu 输出 Ha, 需要 *2 转为 Ry 与 QE 比较
    std::cout << "\n[INFO] Sample V_H comparisons (DFTcu*2 vs QE):" << std::endl;
    std::cout << "  (ix, iy, iz)  |  DFTcu*2 (Ry)  |  QE (Ry)  |  Diff (Ry)" << std::endl;
    std::cout << "  --------------------------------------------------------" << std::endl;

    // 采样 10 个点
    int sample_count = 0;
    for (const auto& v : v_h_ref_) {
        if (sample_count >= 10)
            break;
        if (v.ix % 4 == 0 && v.iy == 0 && v.iz == 0) {
            int idx = v.ix + v.iy * grid_->nr()[0] + v.iz * grid_->nr()[0] * grid_->nr()[1];
            double dftcu_val = v_h_host[idx] * 2.0;  // Ha → Ry
            double qe_val = v.value;
            double diff = dftcu_val - qe_val;

            std::cout << "  (" << v.ix << ", " << v.iy << ", " << v.iz << ")     |  " << dftcu_val
                      << "  |  " << qe_val << "  |  " << diff << std::endl;
            sample_count++;
        }
    }
}

}  // namespace test
}  // namespace dftcu
