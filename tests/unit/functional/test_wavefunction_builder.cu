#include <memory>

#include "fixtures/test_data_loader.cuh"
#include "fixtures/test_fixtures.cuh"
#include "functional/wavefunction_builder.cuh"
#include "model/atoms.cuh"
#include "model/grid.cuh"

#include <gtest/gtest.h>

using namespace dftcu;
using namespace dftcu::test;

/**
 * @brief WavefunctionBuilder 单元测试
 *
 * 测试策略：
 * - Test 1: num_bands() 计算验证（无需 QE 数据）
 *
 * 注意：chi_q 表和最终波函数的验证在集成测试中进行
 */
class WavefunctionBuilderTest : public SiCFixture {};

/**
 * Test 1: num_bands() 计算验证
 *
 * 验证内容：
 * - 对于 SiC 体系（Si: s+p, C: s+p）
 * - 应该返回 8 bands（Si: 1+3=4, C: 1+3=4）
 */
TEST_F(WavefunctionBuilderTest, NumBands_Calculation) {
    // 创建 WavefunctionBuilder
    WavefunctionBuilder builder(*grid_, atoms_);

    // 添加 Si 轨道（s + p）
    std::vector<double> r_si(100), chi_si_s(100), chi_si_p(100), rab_si(100);
    for (int i = 0; i < 100; ++i) {
        r_si[i] = i * 0.01;
        chi_si_s[i] = exp(-r_si[i]);
        chi_si_p[i] = r_si[i] * exp(-r_si[i]);
        rab_si[i] = 0.01;
    }
    builder.add_atomic_orbital(0, 0, r_si, chi_si_s, rab_si);  // Si s
    builder.add_atomic_orbital(0, 1, r_si, chi_si_p, rab_si);  // Si p

    // 添加 C 轨道（s + p）
    std::vector<double> r_c(100), chi_c_s(100), chi_c_p(100), rab_c(100);
    for (int i = 0; i < 100; ++i) {
        r_c[i] = i * 0.01;
        chi_c_s[i] = exp(-r_c[i] * 1.5);
        chi_c_p[i] = r_c[i] * exp(-r_c[i] * 1.5);
        rab_c[i] = 0.01;
    }
    builder.add_atomic_orbital(1, 0, r_c, chi_c_s, rab_c);  // C s
    builder.add_atomic_orbital(1, 1, r_c, chi_c_p, rab_c);  // C p

    // 验证 band 数量
    int num_bands = builder.num_bands();

    // SiC 体系：
    // - Si 原子：1 个 s 轨道 (2l+1=1) + 1 个 p 轨道 (2l+1=3) = 4 bands
    // - C 原子：1 个 s 轨道 (2l+1=1) + 1 个 p 轨道 (2l+1=3) = 4 bands
    // 总计：8 bands
    EXPECT_EQ(num_bands, 8) << "SiC 体系应该有 8 个 bands";

    std::cout << "✓ num_bands() 计算正确: " << num_bands << " bands" << std::endl;
}

/**
 * Test 2: build() 基本功能验证
 *
 * 验证内容：
 * - build() 能够成功创建 Wavefunction 对象
 * - 返回的波函数有正确的 band 数量
 */
TEST_F(WavefunctionBuilderTest, Build_BasicFunctionality) {
    // 创建 WavefunctionBuilder
    WavefunctionBuilder builder(*grid_, atoms_);

    // 添加轨道（简化版本）
    std::vector<double> r(100), chi_s(100), chi_p(100), rab(100);
    for (int i = 0; i < 100; ++i) {
        r[i] = i * 0.01;
        chi_s[i] = exp(-r[i]);
        chi_p[i] = r[i] * exp(-r[i]);
        rab[i] = 0.01;
    }

    builder.add_atomic_orbital(0, 0, r, chi_s, rab);  // Si s
    builder.add_atomic_orbital(0, 1, r, chi_p, rab);  // Si p
    builder.add_atomic_orbital(1, 0, r, chi_s, rab);  // C s
    builder.add_atomic_orbital(1, 1, r, chi_p, rab);  // C p

    // 构建波函数
    auto psi = builder.build();

    // 验证
    ASSERT_NE(psi, nullptr) << "build() 应该返回非空指针";
    EXPECT_EQ(psi->num_bands(), 8) << "波函数应该有 8 个 bands";

    std::cout << "✓ build() 成功创建波函数: " << psi->num_bands() << " bands" << std::endl;
}
