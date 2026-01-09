#include "fft/fft_solver.cuh"
#include "fft/gamma_fft_solver.cuh"
#include "functional/density_functional_potential.cuh"
#include "functional/ewald.cuh"
#include "functional/hartree.cuh"
#include "functional/kedf/hc.cuh"
#include "functional/kedf/tf.cuh"
#include "functional/kedf/vw.cuh"
#include "functional/kedf/wt.cuh"
#include "functional/nonlocal_pseudo.cuh"
#include "functional/pseudo.cuh"
#include "functional/pseudopotential_data.cuh"
#include "functional/upf_parser.cuh"
#include "functional/xc/lda_pz.cuh"
#include "functional/xc/pbe.cuh"
#include "math/bessel.cuh"
#include "math/ylm.cuh"
#include "model/atoms.cuh"
#include "model/atoms_factory.cuh"
#include "model/density_builder.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"
#include "model/grid_factory.cuh"
#include "model/wavefunction.cuh"
#include "model/wavefunction_builder.cuh"
#include "solver/davidson.cuh"
#include "solver/evaluator.cuh"  // Keep for backward compatibility
#include "solver/hamiltonian.cuh"
#include "solver/nscf.cuh"
#include "solver/phase0_verifier.cuh"
#include "solver/scf.cuh"
#include "solver/subspace_solver.cuh"
#include "utilities/gpu_vector.cuh"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_dftcu, m) {
    m.doc() = "DFTcu: A GPU-accelerated orbital-free DFT code";

    // --- Constants ---
    py::module m_const = m.def_submodule("constants", "Physical constants");
    m_const.attr("BOHR_TO_ANGSTROM") = dftcu::constants::BOHR_TO_ANGSTROM;
    m_const.attr("ANGSTROM_TO_BOHR") = dftcu::constants::ANGSTROM_TO_BOHR;
    m_const.attr("HA_TO_EV") = dftcu::constants::HA_TO_EV;
    m_const.attr("HA_TO_RY") = dftcu::constants::HA_TO_RY;
    m_const.attr("RY_TO_HA") = dftcu::constants::RY_TO_HA;
    m_const.attr("D_PI") = dftcu::constants::D_PI;

    m.def("spherical_bessel_jl", &dftcu::spherical_bessel_jl, py::arg("l"), py::arg("x"));
    m.def("ylm", &dftcu::get_ylm, py::arg("l"), py::arg("m_idx"), py::arg("gx"), py::arg("gy"),
          py::arg("gz"), py::arg("gmod"));

    py::class_<dftcu::FFTSolver>(m, "FFTSolver")
        .def(py::init<dftcu::Grid&>())
        .def("forward", &dftcu::FFTSolver::forward)
        .def("backward", &dftcu::FFTSolver::backward);

    py::class_<dftcu::GammaFFTSolver>(m, "GammaFFTSolver")
        .def(py::init<dftcu::Grid&>())
        .def("wave_g2r_single", &dftcu::GammaFFTSolver::wave_g2r_single, py::arg("psi_g_half"),
             py::arg("miller_h"), py::arg("miller_k"), py::arg("miller_l"), py::arg("psi_r"))
        .def("wave_r2g_single", &dftcu::GammaFFTSolver::wave_r2g_single, py::arg("psi_r"),
             py::arg("miller_h"), py::arg("miller_k"), py::arg("miller_l"), py::arg("psi_g_half"))
        .def("wave_g2r_pair", &dftcu::GammaFFTSolver::wave_g2r_pair, py::arg("psi1_g"),
             py::arg("psi2_g"), py::arg("miller_h"), py::arg("miller_k"), py::arg("miller_l"),
             py::arg("psi_r_packed"))
        .def("wave_r2g_pair", &dftcu::GammaFFTSolver::wave_r2g_pair, py::arg("psi_r_packed"),
             py::arg("miller_h"), py::arg("miller_k"), py::arg("miller_l"), py::arg("psi1_g"),
             py::arg("psi2_g"));

    // Factory functions for Grid creation (明确单位)
    m.def(
        "create_grid_from_qe",
        [](py::array_t<double> lattice_ang, const std::vector<int>& nr, double ecutwfc_ry,
           double ecutrho_ry, bool is_gamma) {
            // Convert NumPy array to std::vector<std::vector<double>>
            auto buf = lattice_ang.request();
            if (buf.ndim != 2 || buf.shape[0] != 3 || buf.shape[1] != 3) {
                throw std::runtime_error("lattice_ang must be 3×3 array");
            }
            double* ptr = static_cast<double*>(buf.ptr);
            std::vector<std::vector<double>> lattice(3, std::vector<double>(3));
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    lattice[i][j] = ptr[i * 3 + j];
                }
            }
            return dftcu::create_grid_from_qe(lattice, nr, ecutwfc_ry, ecutrho_ry, is_gamma);
        },
        "Create Grid from QE units (Angstrom + Rydberg)", py::arg("lattice_ang"), py::arg("nr"),
        py::arg("ecutwfc_ry"), py::arg("ecutrho_ry") = -1.0, py::arg("is_gamma") = false);

    m.def(
        "create_grid_from_atomic_units",
        [](py::array_t<double> lattice_bohr, const std::vector<int>& nr, double ecutwfc_ha,
           double ecutrho_ha, bool is_gamma) {
            // Convert NumPy array to std::vector<std::vector<double>>
            auto buf = lattice_bohr.request();
            if (buf.ndim != 2 || buf.shape[0] != 3 || buf.shape[1] != 3) {
                throw std::runtime_error("lattice_bohr must be 3×3 array");
            }
            double* ptr = static_cast<double*>(buf.ptr);
            std::vector<std::vector<double>> lattice(3, std::vector<double>(3));
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    lattice[i][j] = ptr[i * 3 + j];
                }
            }
            return dftcu::create_grid_from_atomic_units(lattice, nr, ecutwfc_ha, ecutrho_ha,
                                                        is_gamma);
        },
        "Create Grid from atomic units (Bohr + Hartree)", py::arg("lattice_bohr"), py::arg("nr"),
        py::arg("ecutwfc_ha"), py::arg("ecutrho_ha") = -1.0, py::arg("is_gamma") = false);

    py::class_<dftcu::Grid>(m, "Grid")
        .def(py::init<const std::vector<double>&, const std::vector<int>&, double, double, bool>(),
             "Internal constructor - use factory functions instead", py::arg("lattice_bohr"),
             py::arg("nr"), py::arg("ecutwfc_ha"), py::arg("ecutrho_ha") = -1.0,
             py::arg("is_gamma") = false)
        .def("volume", &dftcu::Grid::volume)
        .def("volume_bohr", &dftcu::Grid::volume_bohr)
        .def("dv", &dftcu::Grid::dv)
        .def("dv_bohr", &dftcu::Grid::dv_bohr)
        .def("nnr", &dftcu::Grid::nnr)
        .def("nr",
             [](dftcu::Grid& self) {
                 return std::vector<int>{self.nr()[0], self.nr()[1], self.nr()[2]};
             })
        .def(
            "rec_lattice",
            [](dftcu::Grid& self) {
                // Return reciprocal lattice as flat array (row-major)
                std::vector<double> rec_flat(9);
                const double(*rec)[3] = self.rec_lattice();
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        rec_flat[i * 3 + j] = rec[i][j];
                    }
                }
                return rec_flat;
            },
            "Get reciprocal lattice vectors (row-major, 9 elements)")
        .def("g2max", &dftcu::Grid::g2max)
        .def("synchronize", &dftcu::Grid::synchronize)
        .def("gg",
             [](dftcu::Grid& self) {
                 std::vector<double> h_gg(self.nnr());
                 CHECK(cudaMemcpy(h_gg.data(), self.gg(), self.nnr() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
                 return h_gg;
             })
        .def("gx",
             [](dftcu::Grid& self) {
                 std::vector<double> h_gx(self.nnr());
                 CHECK(cudaMemcpy(h_gx.data(), self.gx(), self.nnr() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
                 return h_gx;
             })
        .def("gy",
             [](dftcu::Grid& self) {
                 std::vector<double> h_gy(self.nnr());
                 CHECK(cudaMemcpy(h_gy.data(), self.gy(), self.nnr() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
                 return h_gy;
             })
        .def("gz",
             [](dftcu::Grid& self) {
                 std::vector<double> h_gz(self.nnr());
                 CHECK(cudaMemcpy(h_gz.data(), self.gz(), self.nnr() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
                 return h_gz;
             })
        .def("is_gamma", &dftcu::Grid::is_gamma, "Check if Gamma-only calculation")
        // G-vector management (Phase 0c)
        .def("generate_gvectors", &dftcu::Grid::generate_gvectors,
             "Generate Smooth and Dense G-vectors based on cutoff energies")
        .def("load_gvectors_from_qe", &dftcu::Grid::load_gvectors_from_qe, py::arg("data_dir"),
             "Load G-vector data from QE output files (TEST ONLY)")
        .def("load_nl_mapping_from_file", &dftcu::Grid::load_nl_mapping_from_file,
             py::arg("filename"), "Load nl_d/nlm_d mapping from file (TEST ONLY)")
        // Cutoff energies - read-only (set at construction)
        .def("ecutwfc", &dftcu::Grid::ecutwfc, "Get wavefunction cutoff energy (Hartree)")
        .def("ecutrho", &dftcu::Grid::ecutrho, "Get density cutoff energy (Hartree)")

        // Smooth grid (ecutwfc) - for wavefunctions and beta projectors
        .def("ngw", &dftcu::Grid::ngw, "Number of G-vectors in Smooth grid (ecutwfc)")
        .def("ngm", &dftcu::Grid::ngm, "Alias for ngw() for backward compatibility")
        .def(
            "get_gg_smooth", [](dftcu::Grid& self) { return self.get_gg_smooth(); },
            "Get |G|^2 for Smooth grid (Angstrom^-2)")
        .def(
            "get_g2kin",
            [](dftcu::Grid& self) {
                std::vector<double> g2kin_host(self.ngw());
                CHECK(cudaMemcpy(g2kin_host.data(), self.g2kin(), self.ngw() * sizeof(double),
                                 cudaMemcpyDeviceToHost));
                return g2kin_host;
            },
            "Get kinetic energy coefficients for Smooth grid")

        // Dense grid (ecutrho) - for density and V_loc
        .def("ngm_dense", &dftcu::Grid::ngm_dense, "Number of G-vectors in Dense grid (ecutrho)")
        .def("ngl", &dftcu::Grid::ngl, "Number of G-shells in Dense grid")
        .def(
            "get_gl_shells", [](dftcu::Grid& self) { return self.get_gl_shells(); },
            "Get |G|^2 for each G-shell in Dense grid (Angstrom^-2)")
        .def(
            "get_igtongl", [](dftcu::Grid& self) { return self.get_igtongl(); },
            "Get G-vector to G-shell mapping for Dense grid")

        // Smooth -> Dense mapping
        .def(
            "get_igk", [](dftcu::Grid& self) { return self.get_igk(); },
            "Get Smooth grid G-vector to Dense grid G-vector mapping")
        .def(
            "get_nl_d",
            [](dftcu::Grid& self) {
                std::vector<int> nl_d_host(self.ngw());  // Smooth grid size
                CHECK(cudaMemcpy(nl_d_host.data(), self.nl_d(), self.ngw() * sizeof(int),
                                 cudaMemcpyDeviceToHost));
                return nl_d_host;
            },
            "Get FFT grid indices for Smooth grid G-vectors")
        .def(
            "get_nlm_d",
            [](dftcu::Grid& self) {
                std::vector<int> nlm_d_host(self.ngw());  // Smooth grid size
                CHECK(cudaMemcpy(nlm_d_host.data(), self.nlm_d(), self.ngw() * sizeof(int),
                                 cudaMemcpyDeviceToHost));
                return nlm_d_host;
            },
            "Get FFT grid indices for Smooth grid -G-vectors")
        .def(
            "get_miller_h",
            [](dftcu::Grid& self) {
                std::vector<int> h_host(self.ngw());  // Smooth grid size
                CHECK(cudaMemcpy(h_host.data(), self.miller_h(), self.ngw() * sizeof(int),
                                 cudaMemcpyDeviceToHost));
                return h_host;
            },
            "Get Miller index h for Smooth grid G-vectors")
        .def(
            "get_miller_k",
            [](dftcu::Grid& self) {
                std::vector<int> k_host(self.ngw());  // Smooth grid size
                CHECK(cudaMemcpy(k_host.data(), self.miller_k(), self.ngw() * sizeof(int),
                                 cudaMemcpyDeviceToHost));
                return k_host;
            },
            "Get Miller index k for Smooth grid G-vectors")
        .def(
            "get_miller_l",
            [](dftcu::Grid& self) {
                std::vector<int> l_host(self.ngw());  // Smooth grid size
                CHECK(cudaMemcpy(l_host.data(), self.miller_l(), self.ngw() * sizeof(int),
                                 cudaMemcpyDeviceToHost));
                return l_host;
            },
            "Get Miller index l for Smooth grid G-vectors")
        // Dense Grid accessors (Phase 0c)
        .def("get_gg_dense", &dftcu::Grid::get_gg_dense,
             "Get gg_dense (Dense grid |G|^2) as host vector")
        .def("get_gl_shells", &dftcu::Grid::get_gl_shells, "Get gl (G-shell |G|^2) as host vector")
        .def("get_igtongl", &dftcu::Grid::get_igtongl,
             "Get igtongl (Dense G → shell mapping) as host vector")
        .def("get_igk", &dftcu::Grid::get_igk,
             "Get igk (Smooth G → Dense G mapping) as host vector");

    py::class_<dftcu::RealField>(m, "RealField")
        .def(py::init<dftcu::Grid&, int>(), py::arg("grid"), py::arg("rank") = 1)
        .def("integral", &dftcu::RealField::integral)
        .def("fill", &dftcu::RealField::fill)
        .def("dot", &dftcu::RealField::dot)
        .def("copy_from_host",
             [](dftcu::RealField& self, py::array_t<double> arr) {
                 py::buffer_info buf = arr.request();
                 if (buf.size != self.size())
                     throw std::runtime_error("Size mismatch");
                 self.copy_from_host((double*)buf.ptr);
             })
        .def("copy_to_host", [](dftcu::RealField& self, py::array_t<double> arr) {
            py::buffer_info buf = arr.request();
            if (buf.size != self.size())
                throw std::runtime_error("Size mismatch");
            self.copy_to_host((double*)buf.ptr);
        });

    py::class_<dftcu::ComplexField>(m, "ComplexField")
        .def(py::init<dftcu::Grid&, int>(), py::arg("grid"), py::arg("rank") = 1)
        .def("fill", &dftcu::ComplexField::fill)
        .def("copy_from_host",
             [](dftcu::ComplexField& self, py::array_t<std::complex<double>> arr) {
                 py::buffer_info buf = arr.request();
                 if (buf.size != self.size())
                     throw std::runtime_error("Size mismatch");
                 self.copy_from_host((gpufftComplex*)buf.ptr);
             })
        .def("copy_to_host", [](dftcu::ComplexField& self, py::array_t<std::complex<double>> arr) {
            py::buffer_info buf = arr.request();
            if (buf.size != self.size())
                throw std::runtime_error("Size mismatch");
            self.copy_to_host((gpufftComplex*)buf.ptr);
        });

    py::class_<dftcu::Atom>(m, "Atom")
        .def(py::init<>())
        .def(py::init<double, double, double, double, int>(), py::arg("x"), py::arg("y"),
             py::arg("z"), py::arg("charge"), py::arg("type"));

    py::class_<dftcu::Atoms, std::shared_ptr<dftcu::Atoms>>(m, "Atoms")
        .def(py::init<const std::vector<dftcu::Atom>&>(),
             "Internal constructor - use factory functions instead. Positions must be in BOHR.",
             py::arg("atoms_bohr"))
        .def("nat", &dftcu::Atoms::nat)
        .def("h_type", &dftcu::Atoms::h_type);

    // Atoms factory functions
    m.def("create_atoms_from_angstrom", &dftcu::create_atoms_from_angstrom, py::arg("atoms_ang"),
          "Create Atoms from positions in Angstrom (recommended for user input)");
    m.def("create_atoms_from_bohr", &dftcu::create_atoms_from_bohr, py::arg("atoms_bohr"),
          "Create Atoms from positions in Bohr (atomic units)");

    py::class_<dftcu::Wavefunction, std::shared_ptr<dftcu::Wavefunction>>(m, "Wavefunction")
        .def(py::init<dftcu::Grid&, int, double>(), py::arg("grid"), py::arg("num_bands"),
             py::arg("encut"))
        .def("num_pw", &dftcu::Wavefunction::num_pw)
        .def("num_bands", &dftcu::Wavefunction::num_bands)
        .def("compute_density", &dftcu::Wavefunction::compute_density)
        .def("randomize", &dftcu::Wavefunction::randomize, py::arg("seed") = 42U)
        .def("get_pw_indices", &dftcu::Wavefunction::get_pw_indices)
        .def("get_g2kin", &dftcu::Wavefunction::get_g2kin)
        .def("orthonormalize", &dftcu::Wavefunction::orthonormalize)
        .def("dot", &dftcu::Wavefunction::dot, py::arg("band_a"), py::arg("band_b"))
        .def("copy_from_host",
             [](dftcu::Wavefunction& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> arr) {
                 py::buffer_info buf = arr.request();
                 size_t expected = static_cast<size_t>(self.grid().nnr()) * self.num_bands();
                 if (buf.size != expected) {
                     throw std::runtime_error("Wavefunction size mismatch in copy_from_host");
                 }
                 self.copy_from_host(static_cast<std::complex<double>*>(buf.ptr));
             })
        .def("copy_to_host",
             [](const dftcu::Wavefunction& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> arr) {
                 py::buffer_info buf = arr.request();
                 size_t expected = static_cast<size_t>(self.grid().nnr()) * self.num_bands();
                 if (buf.size != expected) {
                     throw std::runtime_error("Wavefunction size mismatch in copy_to_host");
                 }
                 self.copy_to_host(static_cast<std::complex<double>*>(buf.ptr));
             })
        .def("get_coefficients", &dftcu::Wavefunction::get_coefficients, py::arg("band"))
        .def("set_coefficients", &dftcu::Wavefunction::set_coefficients, py::arg("coeffs"),
             py::arg("band"))
        .def(
            "set_coefficients_miller",
            [](dftcu::Wavefunction& self, const std::vector<int>& h, const std::vector<int>& k,
               const std::vector<int>& l, py::array_t<std::complex<double>> values,
               bool expand_hermitian = true) {
                py::buffer_info buf = values.request();
                const std::complex<double>* ptr = static_cast<const std::complex<double>*>(buf.ptr);
                std::vector<std::complex<double>> vec(ptr, ptr + buf.size);
                self.set_coefficients_miller(h, k, l, vec, expand_hermitian);
            },
            py::arg("h"), py::arg("k"), py::arg("l"), py::arg("values"),
            py::arg("expand_hermitian") = true)
        .def("compute_kinetic_energy", &dftcu::Wavefunction::compute_kinetic_energy,
             py::arg("occupations"));

    py::class_<dftcu::WavefunctionBuilder>(m, "WavefunctionBuilder")
        .def(py::init<dftcu::Grid&, std::shared_ptr<dftcu::Atoms>>())
        .def("add_atomic_orbital", &dftcu::WavefunctionBuilder::add_atomic_orbital)
        .def("build_atomic_wavefunctions", &dftcu::WavefunctionBuilder::build_atomic_wavefunctions,
             py::arg("psi"), py::arg("randomize_phase") = false);

    // DensityFunctionalPotential (new name for Evaluator)
    // Also register as "Evaluator" for backward compatibility
    py::class_<dftcu::DensityFunctionalPotential,
               std::shared_ptr<dftcu::DensityFunctionalPotential>>(m, "DensityFunctionalPotential")
        .def(py::init<dftcu::Grid&>())
        .def("add_functional", &dftcu::DensityFunctionalPotential::add_functional)
        .def("add_functional",
             [](dftcu::DensityFunctionalPotential& self, std::shared_ptr<dftcu::ThomasFermi> f) {
                 self.add_functional(f);
             })
        .def("add_functional",
             [](dftcu::DensityFunctionalPotential& self, std::shared_ptr<dftcu::vonWeizsacker> f) {
                 self.add_functional(f);
             })
        .def("add_functional", [](dftcu::DensityFunctionalPotential& self,
                                  std::shared_ptr<dftcu::WangTeter> f) { self.add_functional(f); })
        .def("add_functional", [](dftcu::DensityFunctionalPotential& self,
                                  std::shared_ptr<dftcu::revHC> f) { self.add_functional(f); })
        .def("add_functional", [](dftcu::DensityFunctionalPotential& self,
                                  std::shared_ptr<dftcu::Hartree> f) { self.add_functional(f); })
        .def("add_functional", [](dftcu::DensityFunctionalPotential& self,
                                  std::shared_ptr<dftcu::LDA_PZ> f) { self.add_functional(f); })
        .def("add_functional", [](dftcu::DensityFunctionalPotential& self,
                                  std::shared_ptr<dftcu::PBE> f) { self.add_functional(f); })
        .def("add_functional", [](dftcu::DensityFunctionalPotential& self,
                                  std::shared_ptr<dftcu::Ewald> f) { self.add_functional(f); })
        .def("add_functional",
             [](dftcu::DensityFunctionalPotential& self, std::shared_ptr<dftcu::LocalPseudo> f) {
                 self.add_functional(f);
             })
        .def("compute", &dftcu::DensityFunctionalPotential::compute)
        .def("clear", &dftcu::DensityFunctionalPotential::clear);

    // Backward compatibility alias (same type, different name)
    m.attr("Evaluator") = m.attr("DensityFunctionalPotential");

    py::class_<dftcu::Hartree, std::shared_ptr<dftcu::Hartree>>(m, "Hartree")
        .def(py::init<>())
        .def("set_gcut", &dftcu::Hartree::set_gcut, py::arg("gcut"))
        .def(
            "compute",
            [](dftcu::Hartree& self, const dftcu::RealField& rho, dftcu::RealField& vh) {
                double energy = 0.0;
                self.compute(rho, vh, energy);
                return energy;
            },
            py::arg("rho"), py::arg("vh"));

    py::class_<dftcu::LDA_PZ, std::shared_ptr<dftcu::LDA_PZ>>(m, "LDA_PZ")
        .def(py::init<>())
        .def("set_rho_threshold", &dftcu::LDA_PZ::set_rho_threshold, py::arg("threshold"))
        .def(
            "compute",
            [](dftcu::LDA_PZ& self, const dftcu::RealField& rho, dftcu::RealField& v_out) {
                return self.compute(rho, v_out);
            },
            py::arg("rho"), py::arg("potential"));

    py::class_<dftcu::PBE, std::shared_ptr<dftcu::PBE>>(m, "PBE")
        .def(py::init<dftcu::Grid&>())
        .def(
            "compute",
            [](dftcu::PBE& self, const dftcu::RealField& rho, dftcu::RealField& v_out) {
                return self.compute(rho, v_out);
            },
            py::arg("rho"), py::arg("potential"));

    py::class_<dftcu::ThomasFermi, std::shared_ptr<dftcu::ThomasFermi>>(m, "ThomasFermi")
        .def(py::init<double>(), py::arg("coeff") = 1.0);
    py::class_<dftcu::vonWeizsacker, std::shared_ptr<dftcu::vonWeizsacker>>(m, "vonWeizsacker")
        .def(py::init<double>(), py::arg("coeff") = 1.0);
    py::class_<dftcu::WangTeter, std::shared_ptr<dftcu::WangTeter>>(m, "WangTeter")
        .def(py::init<double, double, double>(), py::arg("coeff") = 1.0,
             py::arg("alpha") = 5.0 / 6.0, py::arg("beta") = 5.0 / 6.0);
    py::class_<dftcu::revHC, std::shared_ptr<dftcu::revHC>>(m, "revHC")
        .def(py::init<double, double>(), py::arg("alpha") = 2.0, py::arg("beta") = 2.0 / 3.0);

    py::class_<dftcu::Ewald, std::shared_ptr<dftcu::Ewald>>(m, "Ewald")
        .def(py::init<dftcu::Grid&, std::shared_ptr<dftcu::Atoms>, double, double>(),
             py::arg("grid"), py::arg("atoms"), py::arg("precision") = 1e-8,
             py::arg("gcut_hint") = -1.0)
        .def("set_eta", &dftcu::Ewald::set_eta)
        .def("set_pme", &dftcu::Ewald::set_pme)
        .def("compute", (double(dftcu::Ewald::*)(bool, double)) & dftcu::Ewald::compute,
             py::arg("use_pme") = false, py::arg("gcut") = -1.0)
        .def("compute_legacy", &dftcu::Ewald::compute_legacy)
        .def("compute",
             (double(dftcu::Ewald::*)(const dftcu::RealField&, dftcu::RealField&)) &
                 dftcu::Ewald::compute,
             py::arg("rho"), py::arg("v_out"));

    // --- Pseudopotential Data Structures ---
    py::class_<dftcu::BetaProjector>(m, "BetaProjector")
        .def(py::init<>())
        .def_readwrite("index", &dftcu::BetaProjector::index)
        .def_readwrite("label", &dftcu::BetaProjector::label)
        .def_readwrite("angular_momentum", &dftcu::BetaProjector::angular_momentum)
        .def_readwrite("cutoff_radius_index", &dftcu::BetaProjector::cutoff_radius_index)
        .def_readwrite("beta_r", &dftcu::BetaProjector::beta_r);

    py::class_<dftcu::NonlocalPotential>(m, "NonlocalPotential")
        .def(py::init<>())
        .def_readwrite("beta_functions", &dftcu::NonlocalPotential::beta_functions)
        .def_readwrite("dij", &dftcu::NonlocalPotential::dij)
        .def_readwrite("nbeta", &dftcu::NonlocalPotential::nbeta);

    py::class_<dftcu::RadialMesh>(m, "RadialMesh")
        .def(py::init<>())
        .def_readwrite("r", &dftcu::RadialMesh::r)
        .def_readwrite("rab", &dftcu::RadialMesh::rab)
        .def_readwrite("dx", &dftcu::RadialMesh::dx)
        .def_readwrite("xmin", &dftcu::RadialMesh::xmin)
        .def_readwrite("rmax", &dftcu::RadialMesh::rmax)
        .def_readwrite("mesh", &dftcu::RadialMesh::mesh)
        .def_readwrite("zmesh", &dftcu::RadialMesh::zmesh);

    py::class_<dftcu::LocalPotential>(m, "LocalPotential")
        .def(py::init<>())
        .def_readwrite("vloc_r", &dftcu::LocalPotential::vloc_r);

    py::class_<dftcu::PseudopotentialHeader>(m, "PseudopotentialHeader")
        .def(py::init<>())
        .def_readwrite("element", &dftcu::PseudopotentialHeader::element)
        .def_readwrite("pseudo_type", &dftcu::PseudopotentialHeader::pseudo_type)
        .def_readwrite("functional", &dftcu::PseudopotentialHeader::functional)
        .def_readwrite("z_valence", &dftcu::PseudopotentialHeader::z_valence)
        .def_readwrite("wfc_cutoff", &dftcu::PseudopotentialHeader::wfc_cutoff)
        .def_readwrite("rho_cutoff", &dftcu::PseudopotentialHeader::rho_cutoff)
        .def_readwrite("l_max", &dftcu::PseudopotentialHeader::l_max)
        .def_readwrite("l_local", &dftcu::PseudopotentialHeader::l_local)
        .def_readwrite("mesh_size", &dftcu::PseudopotentialHeader::mesh_size)
        .def_readwrite("number_of_proj", &dftcu::PseudopotentialHeader::number_of_proj)
        .def_readwrite("is_ultrasoft", &dftcu::PseudopotentialHeader::is_ultrasoft)
        .def_readwrite("is_paw", &dftcu::PseudopotentialHeader::is_paw)
        .def_readwrite("core_correction", &dftcu::PseudopotentialHeader::core_correction);

    py::class_<dftcu::PseudopotentialData>(m, "PseudopotentialData")
        .def(py::init<>())
        .def("header", &dftcu::PseudopotentialData::header,
             py::return_value_policy::reference_internal)
        .def("mesh", &dftcu::PseudopotentialData::mesh, py::return_value_policy::reference_internal)
        .def("local", &dftcu::PseudopotentialData::local,
             py::return_value_policy::reference_internal)
        .def("get_nonlocal", &dftcu::PseudopotentialData::nonlocal,
             py::return_value_policy::reference_internal)
        .def("element", &dftcu::PseudopotentialData::element)
        .def("z_valence", &dftcu::PseudopotentialData::z_valence)
        .def("mesh_size", &dftcu::PseudopotentialData::mesh_size)
        .def("number_of_proj", &dftcu::PseudopotentialData::number_of_proj)
        .def("pseudo_type", &dftcu::PseudopotentialData::pseudo_type)
        .def("functional", &dftcu::PseudopotentialData::functional)
        .def("is_valid", &dftcu::PseudopotentialData::is_valid);

    py::class_<dftcu::UPFParser>(m, "UPFParser")
        .def(py::init<>())
        .def("parse", &dftcu::UPFParser::parse, py::arg("filename"))
        .def_static("detect_version", &dftcu::UPFParser::detect_version, py::arg("filename"));

    py::class_<dftcu::LocalPseudo, std::shared_ptr<dftcu::LocalPseudo>>(m, "LocalPseudo")
        .def(py::init<dftcu::Grid&, std::shared_ptr<dftcu::Atoms>>(), py::arg("grid"),
             py::arg("atoms"))
        .def_static("from_upf", &dftcu::LocalPseudo::from_upf, py::arg("grid"), py::arg("atoms"),
                    py::arg("upf_data"), py::arg("atom_type") = 0,
                    "Create LocalPseudo from UPF data")
        .def("init_tab_vloc", &dftcu::LocalPseudo::init_tab_vloc, py::arg("type"),
             py::arg("r_grid"), py::arg("vloc_r"), py::arg("rab"), py::arg("zp"), py::arg("omega"),
             py::arg("mesh_cutoff") = -1)
        .def("set_valence_charge", &dftcu::LocalPseudo::set_valence_charge, py::arg("type"),
             py::arg("zp"))
        .def("set_gcut", &dftcu::LocalPseudo::set_gcut, py::arg("gcut"))
        .def("compute_potential",
             [](dftcu::LocalPseudo& self, dftcu::RealField& vloc) { self.compute(vloc); })
        .def("compute", [](dftcu::LocalPseudo& self, const dftcu::RealField& rho,
                           dftcu::RealField& v_out) { return self.compute(rho, v_out); })
        .def("get_tab_vloc", &dftcu::LocalPseudo::get_tab_vloc, py::arg("type"))
        .def("set_tab_vloc", &dftcu::LocalPseudo::set_tab_vloc, py::arg("type"), py::arg("tab"))
        .def("get_alpha", &dftcu::LocalPseudo::get_alpha, py::arg("type"))
        .def("get_vloc_g_shells", &dftcu::LocalPseudo::get_vloc_g_shells, py::arg("type"),
             py::arg("g_shells"))
        .def("get_dq", &dftcu::LocalPseudo::get_dq)
        .def("set_dq", &dftcu::LocalPseudo::set_dq, py::arg("dq"))
        .def("get_nqx", &dftcu::LocalPseudo::get_nqx)
        .def("get_omega", &dftcu::LocalPseudo::get_omega);

    py::class_<dftcu::NonLocalPseudo, std::shared_ptr<dftcu::NonLocalPseudo>>(m, "NonLocalPseudo")
        .def(py::init<dftcu::Grid&>())
        .def_static("from_upf", &dftcu::NonLocalPseudo::from_upf, py::arg("grid"), py::arg("atoms"),
                    py::arg("upf_data"), py::arg("atom_type") = 0,
                    "Create NonLocalPseudo from UPF data")
        .def("apply", &dftcu::NonLocalPseudo::apply, py::arg("psi_in"), py::arg("h_psi_out"))
        .def("add_projector", &dftcu::NonLocalPseudo::add_projector, py::arg("beta_g"),
             py::arg("coupling_constant"))
        .def("init_tab_beta", &dftcu::NonLocalPseudo::init_tab_beta, py::arg("type"),
             py::arg("r_grid"), py::arg("beta_r"), py::arg("rab"), py::arg("l_list"),
             py::arg("kkbeta_list"), py::arg("omega_angstrom"))
        .def("set_tab_beta", &dftcu::NonLocalPseudo::set_tab_beta, py::arg("type"), py::arg("nb"),
             py::arg("tab"))
        .def("init_dij", &dftcu::NonLocalPseudo::init_dij, py::arg("type"), py::arg("dij"))
        .def("update_projectors", &dftcu::NonLocalPseudo::update_projectors, py::arg("atoms"))
        .def("set_projectors",
             [](dftcu::NonLocalPseudo& self, py::array_t<std::complex<double>> arr) {
                 py::buffer_info buf = arr.request();
                 std::vector<std::complex<double>> host_vec(
                     static_cast<std::complex<double>*>(buf.ptr),
                     static_cast<std::complex<double>*>(buf.ptr) + buf.size);
                 self.set_projectors(host_vec);
             })
        .def("calculate_energy", &dftcu::NonLocalPseudo::calculate_energy, py::arg("psi"),
             py::arg("occupations"))
        .def("clear", &dftcu::NonLocalPseudo::clear)
        .def("num_projectors", &dftcu::NonLocalPseudo::num_projectors)
        .def("get_tab_beta", &dftcu::NonLocalPseudo::get_tab_beta, py::arg("type"), py::arg("nb"))
        .def("get_projector", &dftcu::NonLocalPseudo::get_projector, py::arg("idx"))
        .def("get_projections", &dftcu::NonLocalPseudo::get_projections)
        .def("get_coupling", &dftcu::NonLocalPseudo::get_coupling)
        .def("get_d_projections", &dftcu::NonLocalPseudo::get_d_projections)
        .def("debug_projections",
             [](dftcu::NonLocalPseudo& self, const dftcu::Wavefunction& psi,
                py::array_t<double> qe_becp, py::array_t<std::complex<double>> qe_vkb,
                py::array_t<std::complex<double>> qe_evc, std::vector<std::vector<int>> miller) {
                 py::buffer_info becp_buf = qe_becp.request();
                 std::vector<double> becp_vec(static_cast<double*>(becp_buf.ptr),
                                              static_cast<double*>(becp_buf.ptr) + becp_buf.size);

                 py::buffer_info vkb_buf = qe_vkb.request();
                 std::vector<std::complex<double>> vkb_vec(
                     static_cast<std::complex<double>*>(vkb_buf.ptr),
                     static_cast<std::complex<double>*>(vkb_buf.ptr) + vkb_buf.size);

                 py::buffer_info evc_buf = qe_evc.request();
                 std::vector<std::complex<double>> evc_vec(
                     static_cast<std::complex<double>*>(evc_buf.ptr),
                     static_cast<std::complex<double>*>(evc_buf.ptr) + evc_buf.size);

                 self.debug_projections(psi, becp_vec, vkb_vec, evc_vec, miller);
             });

    py::class_<dftcu::Hamiltonian>(m, "Hamiltonian")
        // New base constructor
        .def(py::init<dftcu::Grid&>())
        // Backward compatibility constructor
        .def(py::init<dftcu::Grid&, std::shared_ptr<dftcu::DensityFunctionalPotential>,
                      std::shared_ptr<dftcu::NonLocalPseudo>>(),
             py::arg("grid"), py::arg("dfp"), py::arg("nl_pseudo") = nullptr)
        .def("set_density_functional_potential",
             &dftcu::Hamiltonian::set_density_functional_potential)
        .def("update_potentials", &dftcu::Hamiltonian::update_potentials)
        .def("apply", &dftcu::Hamiltonian::apply)
        .def("set_nonlocal", &dftcu::Hamiltonian::set_nonlocal)
        .def("has_nonlocal", &dftcu::Hamiltonian::has_nonlocal)
        .def(
            "get_nonlocal",
            [](dftcu::Hamiltonian& self) -> dftcu::NonLocalPseudo& { return self.get_nonlocal(); },
            py::return_value_policy::reference)
        .def("get_v_of_0", &dftcu::Hamiltonian::get_v_of_0)
        .def("v_loc", (dftcu::RealField & (dftcu::Hamiltonian::*)()) & dftcu::Hamiltonian::v_loc,
             py::return_value_policy::reference)

        .def("set_ecutrho", &dftcu::Hamiltonian::set_ecutrho, py::arg("ecutrho"));

    py::class_<dftcu::SubspaceSolver>(m, "SubspaceSolver")
        .def(py::init<dftcu::Grid&>())
        .def("solve_direct", &dftcu::SubspaceSolver::solve_direct, py::arg("ham"), py::arg("psi"))
        .def("solve_generalized",
             [](dftcu::SubspaceSolver& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> h,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> s) {
                 int nbands = h.shape(0);
                 if (h.ndim() != 2 || h.shape(1) != nbands || s.ndim() != 2 ||
                     s.shape(0) != nbands || s.shape(1) != nbands)
                     throw std::runtime_error("Matrix dimensions mismatch");

                 dftcu::GPU_Vector<gpufftComplex> d_h(nbands * nbands);
                 d_h.copy_from_host((gpufftComplex*)h.data());
                 d_h.copy_from_host((gpufftComplex*)h.data());  // Wait, this is just for safety
                 dftcu::GPU_Vector<gpufftComplex> d_s(nbands * nbands);
                 d_s.copy_from_host((gpufftComplex*)s.data());
                 dftcu::GPU_Vector<double> d_e(nbands);

                 self.solve_generalized(nbands, d_h.data(), d_s.data(), d_e.data(), nullptr);

                 std::vector<double> h_e(nbands);
                 d_e.copy_to_host(h_e.data());
                 return h_e;
             });

    py::class_<dftcu::NonSCFSolver>(m, "NonSCFSolver")
        .def(py::init<dftcu::Grid&>())
        .def("solve", &dftcu::NonSCFSolver::solve, py::arg("ham"), py::arg("psi"), py::arg("nelec"),
             py::arg("atoms"), py::arg("ecutrho"), py::arg("rho_core") = nullptr,
             py::arg("alpha_energy") = 0.0);

    py::class_<dftcu::DavidsonSolver>(m, "DavidsonSolver")
        .def(py::init<dftcu::Grid&, int, double>(), py::arg("grid"), py::arg("max_iter") = 50,
             py::arg("tol") = 1e-6)
        .def("solve", &dftcu::DavidsonSolver::solve);

    // SCF Solver
    py::class_<dftcu::EnergyBreakdown>(m, "EnergyBreakdown")
        .def(py::init<>())
        .def_readwrite("etot", &dftcu::EnergyBreakdown::etot)
        .def_readwrite("eband", &dftcu::EnergyBreakdown::eband)
        .def_readwrite("deband", &dftcu::EnergyBreakdown::deband)
        .def_readwrite("ehart", &dftcu::EnergyBreakdown::ehart)
        .def_readwrite("etxc", &dftcu::EnergyBreakdown::etxc)
        .def_readwrite("eewld", &dftcu::EnergyBreakdown::eewld)
        .def_readwrite("alpha", &dftcu::EnergyBreakdown::alpha)
        .def("__repr__", [](const dftcu::EnergyBreakdown& eb) {
            return "<EnergyBreakdown: etot=" + std::to_string(eb.etot) +
                   " eband=" + std::to_string(eb.eband) + " deband=" + std::to_string(eb.deband) +
                   " ehart=" + std::to_string(eb.ehart) + " etxc=" + std::to_string(eb.etxc) +
                   " eewld=" + std::to_string(eb.eewld) + " alpha=" + std::to_string(eb.alpha) +
                   ">";
        });

    py::enum_<dftcu::SCFSolver::MixingType>(m, "MixingType")
        .value("Linear", dftcu::SCFSolver::MixingType::Linear)
        .value("Broyden", dftcu::SCFSolver::MixingType::Broyden)
        .export_values();

    py::class_<dftcu::SCFSolver::Options>(m, "SCFOptions")
        .def(py::init<>())
        .def_readwrite("max_iter", &dftcu::SCFSolver::Options::max_iter)
        .def_readwrite("e_conv", &dftcu::SCFSolver::Options::e_conv)
        .def_readwrite("rho_conv", &dftcu::SCFSolver::Options::rho_conv)
        .def_readwrite("mixing_type", &dftcu::SCFSolver::Options::mixing_type)
        .def_readwrite("mixing_beta", &dftcu::SCFSolver::Options::mixing_beta)
        .def_readwrite("mixing_history", &dftcu::SCFSolver::Options::mixing_history)
        .def_readwrite("verbose", &dftcu::SCFSolver::Options::verbose);

    py::class_<dftcu::SCFSolver>(m, "SCFSolver")
        .def(py::init([](dftcu::Grid& grid) {
                 return new dftcu::SCFSolver(grid, dftcu::SCFSolver::Options());
             }),
             py::arg("grid"))
        .def(py::init<dftcu::Grid&, const dftcu::SCFSolver::Options&>(), py::arg("grid"),
             py::arg("options"))
        .def("solve", &dftcu::SCFSolver::solve, py::arg("ham"), py::arg("psi"),
             py::arg("occupations"), py::arg("rho_init"), py::arg("atoms"), py::arg("ecutrho"),
             py::arg("rho_core") = nullptr, py::arg("alpha_energy") = 0.0)
        .def("is_converged", &dftcu::SCFSolver::is_converged)
        .def("num_iterations", &dftcu::SCFSolver::num_iterations)
        .def("get_history",
             [](const dftcu::SCFSolver& self) {
                 auto hist = self.get_history();
                 py::array_t<double> result({(py::ssize_t)hist.size(), (py::ssize_t)4});
                 auto r = result.mutable_unchecked<2>();
                 for (size_t i = 0; i < hist.size(); ++i) {
                     for (size_t j = 0; j < 4; ++j) {
                         r(i, j) = hist[i][j];
                     }
                 }
                 return result;
             })
        .def("set_alpha_energy", &dftcu::SCFSolver::set_alpha_energy)
        .def("set_atoms", &dftcu::SCFSolver::set_atoms)
        .def("set_ecutrho", &dftcu::SCFSolver::set_ecutrho)
        .def("compute_energy_breakdown", &dftcu::SCFSolver::compute_energy_breakdown,
             py::arg("eigenvalues"), py::arg("occupations"), py::arg("ham"), py::arg("psi"),
             py::arg("rho_val"), py::arg("rho_core") = nullptr);

    py::class_<dftcu::DensityBuilder>(m, "DensityBuilder")
        .def(py::init<dftcu::Grid&, std::shared_ptr<dftcu::Atoms>>())
        .def("set_atomic_rho_g", &dftcu::DensityBuilder::set_atomic_rho_g)
        .def("set_atomic_rho_r", &dftcu::DensityBuilder::set_atomic_rho_r)
        .def("build_density", &dftcu::DensityBuilder::build_density)
        .def("set_gcut", &dftcu::DensityBuilder::set_gcut, py::arg("gcut"));

    // Phase 0 Verifier
    py::class_<dftcu::VerificationResult>(m, "VerificationResult")
        .def(py::init<>())
        .def_readwrite("success", &dftcu::VerificationResult::success)
        .def_readwrite("h_sub_error", &dftcu::VerificationResult::h_sub_error)
        .def_readwrite("s_sub_error", &dftcu::VerificationResult::s_sub_error);

    py::class_<dftcu::Phase0Verifier>(m, "Phase0Verifier")
        .def(py::init<dftcu::Grid&>())
        .def("verify", &dftcu::Phase0Verifier::verify, py::arg("wfc_file"), py::arg("s_ref_file"),
             py::arg("nbands"), py::arg("ecutwfc"));
}
