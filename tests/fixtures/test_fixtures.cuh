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
 * Sets up a standard 16x16x16 grid for SiC zinc-blende.
 */
class SiCFixture : public ::testing::Test {
  protected:
    void SetUp() override {
        // Standard SiC lattice (Zinc-blende)
        // alat = 4.36 Angstrom (approx)
        double alat = 4.3596;
        std::vector<std::vector<double>> lattice = {{0.0, alat / 2.0, alat / 2.0},
                                                    {alat / 2.0, 0.0, alat / 2.0},
                                                    {alat / 2.0, alat / 2.0, 0.0}};

        std::vector<int> nr = {16, 16, 16};
        double ecutwfc_ry = 30.0;
        double ecutrho_ry = 120.0;

        // Create Grid using factory
        grid = create_grid_from_qe(lattice, nr, ecutwfc_ry, ecutrho_ry, true);

        // Create SiC atoms
        std::vector<std::string> elements = {"Si", "C"};
        std::vector<std::vector<double>> positions = {
            {0.0, 0.0, 0.0},    // Si at origin
            {0.25, 0.25, 0.25}  // C at (1/4, 1/4, 1/4)
        };
        // POSCAR-style fractional to Cartesian handled by factory if cartesian=false
        atoms = create_atoms_from_structure(elements, positions, lattice, false, {"Si", "C"},
                                            {{"Si", 4.0}, {"C", 4.0}});
    }

    std::unique_ptr<Grid> grid;
    std::shared_ptr<Atoms> atoms;
};

}  // namespace test
}  // namespace dftcu
