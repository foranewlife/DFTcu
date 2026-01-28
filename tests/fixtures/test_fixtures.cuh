#pragma once
#include <memory>
#include <vector>

#include "fixtures/test_data_loader.cuh"
#include "model/atoms.cuh"
#include "model/atoms_factory.cuh"
#include "model/grid.cuh"
#include "model/grid_factory.cuh"
#include "model/pseudopotential_data.cuh"

#include <gtest/gtest.h>

namespace dftcu {
namespace test {

/**
 * @brief 从 UPFRadialData 创建 PseudopotentialData（用于 mock UPF）
 *
 * 这个函数允许在 C++ 测试中直接构建 PseudopotentialData，
 * 而无需依赖 Python UPF 解析器。
 *
 * @param radial_data 从 QE 捕获的径向网格数据
 * @return PseudopotentialData 对象
 */
inline PseudopotentialData create_pseudo_from_radial(const UPFRadialData& radial_data) {
    PseudopotentialData pseudo;

    // 设置 header
    PseudopotentialHeader header;
    header.element = radial_data.element;
    header.pseudo_type = "NC";  // Norm-conserving
    header.functional = "LDA";
    header.z_valence = radial_data.zp;
    header.mesh_size = radial_data.msh;
    header.l_max = 0;
    header.l_local = 0;
    header.number_of_proj = 0;
    header.is_ultrasoft = false;
    header.is_paw = false;
    header.core_correction = false;
    pseudo.set_header(header);

    // 设置径向网格
    RadialMesh mesh;
    mesh.r = radial_data.r;
    mesh.rab = radial_data.rab;
    mesh.mesh = static_cast<int>(radial_data.r.size());
    mesh.msh = radial_data.msh;
    if (!radial_data.r.empty()) {
        mesh.rmax = radial_data.r.back();
    }
    pseudo.set_mesh(mesh);

    // 设置局域势
    LocalPotential local;
    local.vloc_r = radial_data.vloc_r;
    pseudo.set_local(local);

    return pseudo;
}

/**
 * @brief Base fixture for DFTcu tests.
 * Sets up 18x18x18 grid for SiC zinc-blende matching QE sic_minimal reference.
 *
 * QE Parameters (from manifest.json):
 * - celldm(1) = 8.22 Bohr
 * - ecutwfc = 20 Ry, ecutrho = 80 Ry
 * - nr1 = nr2 = nr3 = 18
 * - omega = 138.853062 Bohr³
 * - ibrav = 2 (FCC)
 */
class SiCFixture : public ::testing::Test {
  protected:
    void SetUp() override {
        // SiC lattice matching QE sic_minimal reference
        // celldm(1) = 8.22 Bohr, ibrav = 2 (FCC)
        double alat_bohr = 8.22;
        std::vector<std::vector<double>> lattice_bohr = {{-alat_bohr / 2.0, 0.0, alat_bohr / 2.0},
                                                         {0.0, alat_bohr / 2.0, alat_bohr / 2.0},
                                                         {-alat_bohr / 2.0, alat_bohr / 2.0, 0.0}};

        // Convert to Angstrom for factory (which expects Angstrom input)
        double bohr_to_ang = 0.529177;
        std::vector<std::vector<double>> lattice_ang;
        for (const auto& v : lattice_bohr) {
            lattice_ang.push_back({v[0] * bohr_to_ang, v[1] * bohr_to_ang, v[2] * bohr_to_ang});
        }

        // QE reference: 18³ grid, 20 Ry wfc cutoff, 80 Ry rho cutoff
        std::vector<int> nr = {18, 18, 18};
        double ecutwfc_ry = 20.0;
        double ecutrho_ry = 80.0;

        // Create Grid using factory
        grid_ = create_grid_from_qe(lattice_ang, nr, ecutwfc_ry, ecutrho_ry, true);

        // Create SiC atoms (Si at origin, C at 1/4,1/4,1/4)
        std::vector<std::string> elements = {"Si", "C"};
        std::vector<std::vector<double>> positions = {
            {0.0, 0.0, 0.0},    // Si at origin
            {0.25, 0.25, 0.25}  // C at (1/4, 1/4, 1/4)
        };
        atoms_ = create_atoms_from_structure(elements, positions, lattice_ang, false, {"Si", "C"},
                                             {{"Si", 4.0}, {"C", 4.0}});
    }

    std::unique_ptr<Grid> grid_;
    std::shared_ptr<Atoms> atoms_;
};

// SiCMinimalFixture is now an alias for SiCFixture (they use the same parameters)
using SiCMinimalFixture = SiCFixture;

}  // namespace test
}  // namespace dftcu
