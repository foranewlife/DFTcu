#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "solver/nscf.cuh"
#include "utilities/cublas_manager.cuh"
#include "utilities/error.cuh"

#include <sys/stat.h>
#include <sys/types.h>

namespace dftcu {

NonSCFSolver::NonSCFSolver(Grid& grid) : grid_(grid), subspace_solver_(grid) {}

void NonSCFSolver::enable_diagnostics(const NonSCFDiagnostics& diagnostics) {
    diagnostics_ = diagnostics;

    // 创建输出目录（如果不存在）
    if (diagnostics_.enabled && !diagnostics_.output_dir.empty()) {
        mkdir(diagnostics_.output_dir.c_str(), 0755);
        std::cout << "     [Diagnostics] Output directory: " << diagnostics_.output_dir
                  << std::endl;
    }
}

EnergyBreakdown NonSCFSolver::solve(Hamiltonian& ham, Wavefunction& psi, double nelec,
                                    std::shared_ptr<Atoms> atoms, double ecutrho,
                                    const RealField* rho_scf, const RealField* rho_core,
                                    double alpha_energy) {
    std::cout << "     TRACE: entering NonSCFSolver::solve" << std::endl;
    std::cout << "     Band Structure Calculation" << std::endl;

    // ════════════════════════════════════════════════════════════════════════════════
    // NSCF 完整流程 (基于 QE non_scf.f90)
    // ════════════════════════════════════════════════════════════════════════════════
    //
    // QE NSCF 流程:
    //   non_scf() [PW/src/non_scf.f90]
    //     ├─> c_bands_nscf()  [对角化所有 k-points]
    //     ├─> poolrecover()   [收集本征值]
    //     ├─> weights()       [计算权重/占据数]
    //     └─> print_ks_energies_nonscf()
    //
    // DFTcu 实现:
    //   - 假设单个 k-point (Gamma-only)
    //   - 使用 SubspaceSolver 直接对角化
    //   - 计算权重和能量
    //
    // ════════════════════════════════════════════════════════════════════════════════

    // ────────────────────────────────────────────────────────────────────────────────
    // Phase 1: c_bands_nscf - 对角化
    // ────────────────────────────────────────────────────────────────────────────────
    //
    // 对应 QE:
    //   c_bands_nscf() [PW/src/c_bands.f90:1171]
    //     └─> diag_bands() → regterg() → Davidson 迭代
    //
    // DFTcu:
    //   SubspaceSolver::solve_direct() 封装了完整的 Davidson 迭代
    //   (或单次子空间对角化，取决于实现)
    //
    // CRITICAL:
    //   - Hamiltonian 已经在调用 solve() 之前构建好 (potinit 阶段)
    //   - vrs = V_ps + V_H[ρ_SCF] + V_xc[ρ_SCF] 已固定
    //   - 整个 NSCF 过程中 vrs 不变
    //
    // ────────────────────────────────────────────────────────────────────────────────

    // ────────────────────────────────────────────────────────────────────────────────
    // Diagnostic: 导出初始波函数 (Phase 0)
    // ────────────────────────────────────────────────────────────────────────────────
    if (diagnostics_.enabled && diagnostics_.dump_psi_initial) {
        std::string filename = diagnostics_.output_dir + "/dftcu_psi_initial.txt";
        dump_wavefunction(filename, psi);
        std::cout << "     [Diagnostics] Dumped initial wavefunction to " << filename << std::endl;
    }

    std::cout << "     TRACE: calling subspace_solver_.solve_direct (Davidson iteration)"
              << std::endl;
    std::vector<double> eigenvalues_all = subspace_solver_.solve_direct(ham, psi);

    // ✅ FIX (2026-01-26): 如果 psi 包含 n_starting_wfc > nbands 个原子波函数，
    // 只保留前 nbands 个本征值（对应占据态）
    int n_starting_wfc = psi.num_bands();
    int nbands_out = (int)(nelec / 2.0 + 0.5);  // 假设自旋简并，nelec/2 个占据态

    std::vector<double> eigenvalues;
    if (n_starting_wfc > nbands_out) {
        std::cout << "     TRACE: Subspace diagonalization: " << n_starting_wfc
                  << " atomic wfcs -> " << nbands_out << " eigenstates" << std::endl;
        eigenvalues.assign(eigenvalues_all.begin(), eigenvalues_all.begin() + nbands_out);

        // TODO: 需要用本征矢量旋转波函数，将 psi 从 n_starting_wfc bands 降维到 nbands_out bands
        // 当前暂时使用所有 n_starting_wfc 个本征值进行后续计算
        std::cout << "     WARNING: Wavefunction rotation not implemented yet. "
                  << "Using all " << n_starting_wfc << " bands for now." << std::endl;
        eigenvalues = eigenvalues_all;  // 临时：使用所有本征值
    } else {
        eigenvalues = eigenvalues_all;
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Diagnostic: 导出本征值 (Phase 1 结果)
    // ────────────────────────────────────────────────────────────────────────────────
    if (diagnostics_.enabled && diagnostics_.dump_eigenvalues) {
        std::string filename = diagnostics_.output_dir + "/dftcu_eigenvalues.txt";
        dump_eigenvalues_to_file(filename, eigenvalues);
        std::cout << "     [Diagnostics] Dumped eigenvalues to " << filename << std::endl;
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Phase 2: weights - 计算占据数和费米能级
    // ────────────────────────────────────────────────────────────────────────────────
    //
    // 对应 QE:
    //   weights() or weights_only() [PW/src/weights.f90]
    //
    // 绝缘体 (insulator):
    //   - 占据能带: occupation = 2.0 (自旋简并)
    //   - 空带: occupation = 0.0
    //   - 费米能级 ef = HOMO 能量
    //
    // ────────────────────────────────────────────────────────────────────────────────

    int nbands = psi.num_bands();
    std::vector<double> occupations(nbands, 0.0);
    double ef = 0.0;

    std::cout << "     TRACE: calling compute_weights_insulator" << std::endl;
    compute_weights_insulator(nbands, nelec, eigenvalues, occupations, ef);

    // ────────────────────────────────────────────────────────────────────────────────
    // Diagnostic: 导出占据数 (Phase 2 结果)
    // ────────────────────────────────────────────────────────────────────────────────
    if (diagnostics_.enabled && diagnostics_.dump_occupations) {
        std::string filename = diagnostics_.output_dir + "/dftcu_occupations.txt";
        dump_occupations_to_file(filename, occupations);
        std::cout << "     [Diagnostics] Dumped occupations to " << filename << std::endl;
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Phase 3: 选择用于能量计算的密度
    // ────────────────────────────────────────────────────────────────────────────────
    //
    // QE non_scf.f90:91-94:
    //   IF (lxdm .OR. ANY(...)) THEN
    //     ! 混合泛函: 从 NSCF 波函数重算密度
    //     CALL sum_band()
    //   ELSE
    //     ! 标准 DFT (LDA/GGA): 使用 SCF 密度
    //     ! (不需要重算)
    //   END IF
    //
    // DFTcu 实现:
    //   - 如果提供了 rho_scf (标准DFT): 使用 SCF 密度
    //   - 否则 (混合泛函): 从 NSCF 波函数计算密度
    //
    // ────────────────────────────────────────────────────────────────────────────────

    RealField rho_for_energy(grid_, 1);

    if (rho_scf != nullptr) {
        // 标准 DFT 路径: LDA, GGA, etc.
        std::cout << "     TRACE: using SCF density for energy calculation" << std::endl;
        rho_for_energy.copy_from(*rho_scf);
    } else {
        // 混合泛函路径: 从 NSCF 波函数重算密度
        std::cout
            << "     TRACE: computing density from NSCF wavefunctions (hybrid functional mode)"
            << std::endl;
        psi.compute_density(occupations, rho_for_energy);
        grid_.synchronize();
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Phase 4: 能量计算和输出
    // ────────────────────────────────────────────────────────────────────────────────
    //
    // 对应 QE:
    //   print_ks_energies_nonscf() [PW/src/output.f90]
    //     输出本征值、费米能级、能量分解
    //
    // DFTcu:
    //   使用 SCFSolver 的 compute_energy_breakdown() 计算能量
    //
    // 能量公式 (QE electrons.f90:1060-1080):
    //   E_tot = E_band + deband + E_H + E_XC + E_Ewald + alpha
    //
    //   其中:
    //     E_band  = Σ_i f_i * ε_i  (能带能量)
    //     deband  = -∫ ρ * V_eff dr (双重计数修正)
    //     E_H     = Hartree 能量
    //     E_XC    = 交换关联能量
    //     E_Ewald = Ewald 能量 (离子-离子库仑能)
    //     alpha   = G=0 修正项 = Nelec * v_of_0
    //
    // ────────────────────────────────────────────────────────────────────────────────

    SCFSolver::Options options;
    options.verbose = false;
    SCFSolver scf_helper(grid_, options);

    // Alpha 能量修正 (QE style): Nelec * v_of_0
    // v_of_0 = G=0 处的总局域势 (V_ps + V_H + V_xc)
    // 注意: SubspaceSolver::solve_direct() 已经在本征值中包含了 v_of_0
    double v0 = ham.get_v_of_0();
    double total_alpha_energy = nelec * v0;

    scf_helper.set_alpha_energy(total_alpha_energy);
    scf_helper.set_atoms(atoms);
    scf_helper.set_ecutrho(ecutrho);

    std::cout << "     End of band structure calculation" << std::endl;

    // 计算完整的能量分解
    EnergyBreakdown breakdown = scf_helper.compute_energy_breakdown(eigenvalues, occupations, ham,
                                                                    psi, rho_for_energy, rho_core);

    // 存储本征值方便访问
    breakdown.eigenvalues = eigenvalues;

    // ────────────────────────────────────────────────────────────────────────────────
    // Diagnostic: 导出能量分解 (Phase 4 结果)
    // ────────────────────────────────────────────────────────────────────────────────
    if (diagnostics_.enabled && diagnostics_.dump_energy_breakdown) {
        std::string filename = diagnostics_.output_dir + "/dftcu_energy_breakdown.txt";
        dump_energy(filename, breakdown);
        std::cout << "     [Diagnostics] Dumped energy breakdown to " << filename << std::endl;
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Diagnostic: 导出有效势能 (Phase 5)
    // ────────────────────────────────────────────────────────────────────────────────
    if (diagnostics_.enabled && diagnostics_.dump_v_eff) {
        std::string filename = diagnostics_.output_dir + "/dftcu_v_eff.txt";
        dump_v_eff(filename, ham.v_loc());
        std::cout << "     [Diagnostics] Dumped v_eff to " << filename << std::endl;
    }

    return breakdown;
}

void NonSCFSolver::compute_weights_insulator(int nbands, double nelec,
                                             const std::vector<double>& eigenvalues,
                                             std::vector<double>& occupations, double& ef) {
    double degspin = 2.0;  // Assume nspin=1 and not noncollinear
    int n_filled = static_cast<int>(std::round(nelec / degspin));

    for (int i = 0; i < nbands; ++i) {
        if (i < n_filled) {
            occupations[i] = degspin;
        } else {
            occupations[i] = 0.0;
        }
    }

    if (n_filled > 0) {
        ef = eigenvalues[n_filled - 1];
    } else {
        ef = -1e10;
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Diagnostic Helper Functions - 导出中间变量到文件（QE 格式兼容）
// ════════════════════════════════════════════════════════════════════════════════

void NonSCFSolver::dump_wavefunction(const std::string& filename, const Wavefunction& psi) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "     [Diagnostics] WARNING: Failed to open " << filename << std::endl;
        return;
    }

    ofs << std::scientific << std::setprecision(16);

    int npw = psi.num_pw();
    int nbands = psi.num_bands();
    int nnr = grid_.nnr();  // Internal storage size (FFT grid)

    // QE 格式: 每行一个 G-vector，所有能带的系数
    // Format: ig  Re[ψ(G,band1)]  Im[ψ(G,band1)]  Re[ψ(G,band2)]  Im[ψ(G,band2)]  ...
    ofs << "# Wavefunction: npw=" << npw << " nbands=" << nbands << "\n";
    ofs << "# Format: ig  Re[psi(:,1)]  Im[psi(:,1)]  Re[psi(:,2)]  Im[psi(:,2)]  ...\n";

    // 从 GPU 复制数据到 CPU（完整 FFT grid 大小）
    std::vector<std::complex<double>> psi_host(nnr * nbands);
    psi.copy_to_host(psi_host.data());

    // 获取 nl_d 映射（G-vector 到 FFT grid 的索引）
    std::vector<int> nl_d_host(npw);
    CHECK(cudaMemcpy(nl_d_host.data(), grid_.nl_d(), npw * sizeof(int), cudaMemcpyDeviceToHost));

    for (int ig = 0; ig < npw; ++ig) {
        ofs << std::setw(8) << ig + 1;  // Fortran 1-based indexing
        for (int ib = 0; ib < nbands; ++ib) {
            int ifft = nl_d_host[ig];   // Map G-vector to FFT grid index
            int idx = ib * nnr + ifft;  // Use nnr as leading dimension
            ofs << "  " << std::setw(24) << psi_host[idx].real() << "  " << std::setw(24)
                << psi_host[idx].imag();
        }
        ofs << "\n";
    }

    ofs.close();
}

void NonSCFSolver::dump_eigenvalues_to_file(const std::string& filename,
                                            const std::vector<double>& evals) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "     [Diagnostics] WARNING: Failed to open " << filename << std::endl;
        return;
    }

    ofs << std::scientific << std::setprecision(16);
    ofs << "# Eigenvalues (Hartree)\n";
    ofs << "# nbands = " << evals.size() << "\n";

    for (size_t i = 0; i < evals.size(); ++i) {
        ofs << std::setw(5) << i + 1 << "  " << std::setw(24) << evals[i] << "\n";
    }

    ofs.close();
}

void NonSCFSolver::dump_real_matrix(const std::string& filename, int n, const double* matrix) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "     [Diagnostics] WARNING: Failed to open " << filename << std::endl;
        return;
    }

    ofs << std::scientific << std::setprecision(16);
    ofs << "# Real symmetric matrix: n = " << n << "\n";

    // 行优先输出（每行所有列）
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ofs << std::setw(24) << matrix[i * n + j];
            if (j < n - 1)
                ofs << "  ";
        }
        ofs << "\n";
    }

    ofs.close();
}

void NonSCFSolver::dump_occupations_to_file(const std::string& filename,
                                            const std::vector<double>& occs) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "     [Diagnostics] WARNING: Failed to open " << filename << std::endl;
        return;
    }

    ofs << std::scientific << std::setprecision(16);
    ofs << "# Occupation numbers\n";
    ofs << "# nbands = " << occs.size() << "\n";

    for (size_t i = 0; i < occs.size(); ++i) {
        ofs << std::setw(5) << i + 1 << "  " << std::setw(24) << occs[i] << "\n";
    }

    ofs.close();
}

void NonSCFSolver::dump_energy(const std::string& filename, const EnergyBreakdown& energy) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "     [Diagnostics] WARNING: Failed to open " << filename << std::endl;
        return;
    }

    ofs << std::scientific << std::setprecision(16);
    ofs << "# Energy Breakdown (Hartree)\n";
    ofs << "eband   " << std::setw(24) << energy.eband << "\n";
    ofs << "deband  " << std::setw(24) << energy.deband << "\n";
    ofs << "ehart   " << std::setw(24) << energy.ehart << "\n";
    ofs << "etxc    " << std::setw(24) << energy.etxc << "\n";
    ofs << "eewld   " << std::setw(24) << energy.eewld << "\n";
    ofs << "alpha   " << std::setw(24) << energy.alpha << "\n";
    ofs << "etot    " << std::setw(24) << energy.etot << "\n";

    ofs.close();
}

void NonSCFSolver::dump_v_eff(const std::string& filename, const RealField& v_eff) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "     [Diagnostics] WARNING: Failed to open " << filename << std::endl;
        return;
    }

    ofs << std::scientific << std::setprecision(16);

    // 获取实空间网格点数
    size_t nnr = v_eff.size();

    // 复制数据到 host
    std::vector<double> v_eff_host(nnr);
    CHECK(
        cudaMemcpy(v_eff_host.data(), v_eff.data(), nnr * sizeof(double), cudaMemcpyDeviceToHost));

    // QE 格式：第一行是点数，后面每行一个值
    ofs << nnr << "\n";
    for (size_t i = 0; i < nnr; ++i) {
        ofs << "  " << v_eff_host[i] << "\n";
    }

    ofs.close();
}

}  // namespace dftcu
