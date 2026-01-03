#include "fft/fft_solver.cuh"
#include "functional/ewald.cuh"
#include "functional/hartree.cuh"
#include "functional/kedf/hc.cuh"
#include "functional/kedf/tf.cuh"
#include "functional/kedf/vw.cuh"
#include "functional/kedf/wt.cuh"
#include "functional/nonlocal_pseudo.cuh"
#include "functional/pseudo.cuh"
#include "functional/xc/lda_pz.cuh"
#include "functional/xc/pbe.cuh"
#include "math/bessel.cuh"
#include "math/ylm.cuh"
#include "model/atoms.cuh"
#include "model/density_builder.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"
#include "model/wavefunction.cuh"
#include "model/wavefunction_builder.cuh"
#include "solver/davidson.cuh"
#include "solver/evaluator.cuh"
#include "solver/hamiltonian.cuh"
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
    m_const.attr("D_PI") = dftcu::constants::D_PI;

    m.def("spherical_bessel_jl", &dftcu::spherical_bessel_jl, py::arg("l"), py::arg("x"));
    m.def("ylm", &dftcu::get_ylm, py::arg("l"), py::arg("m_idx"), py::arg("gx"), py::arg("gy"),
          py::arg("gz"), py::arg("gmod"));

    py::class_<dftcu::Grid>(m, "Grid")
        .def(py::init<const std::vector<double>&, const std::vector<int>&>())
        .def("volume", &dftcu::Grid::volume)
        .def("volume_bohr", &dftcu::Grid::volume_bohr)
        .def("dv", &dftcu::Grid::dv)
        .def("dv_bohr", &dftcu::Grid::dv_bohr)
        .def("nnr", &dftcu::Grid::nnr)
        .def("nr",
             [](dftcu::Grid& self) {
                 return std::vector<int>{self.nr()[0], self.nr()[1], self.nr()[2]};
             })
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
        .def("gz", [](dftcu::Grid& self) {
            std::vector<double> h_gz(self.nnr());
            CHECK(cudaMemcpy(h_gz.data(), self.gz(), self.nnr() * sizeof(double),
                             cudaMemcpyDeviceToHost));
            return h_gz;
        });

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
        .def(py::init<const std::vector<dftcu::Atom>&>())
        .def("nat", &dftcu::Atoms::nat)
        .def("h_type", &dftcu::Atoms::h_type);

    py::class_<dftcu::Wavefunction, std::shared_ptr<dftcu::Wavefunction>>(m, "Wavefunction")
        .def(py::init<dftcu::Grid&, int, double>(), py::arg("grid"), py::arg("num_bands"),
             py::arg("encut"))
        .def("num_pw", &dftcu::Wavefunction::num_pw)
        .def("num_bands", &dftcu::Wavefunction::num_bands)
        .def("compute_density", &dftcu::Wavefunction::compute_density)
        .def("randomize", &dftcu::Wavefunction::randomize, py::arg("seed") = 42U)
        .def("get_pw_indices", &dftcu::Wavefunction::get_pw_indices)
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
        .def("set_coefficients_miller", &dftcu::Wavefunction::set_coefficients_miller, py::arg("h"),
             py::arg("k"), py::arg("l"), py::arg("values"), py::arg("expand_hermitian") = true)
        .def("compute_kinetic_energy", &dftcu::Wavefunction::compute_kinetic_energy,
             py::arg("occupations"));

    py::class_<dftcu::WavefunctionBuilder>(m, "WavefunctionBuilder")
        .def(py::init<dftcu::Grid&, std::shared_ptr<dftcu::Atoms>>())
        .def("add_atomic_orbital", &dftcu::WavefunctionBuilder::add_atomic_orbital)
        .def("build_atomic_wavefunctions", &dftcu::WavefunctionBuilder::build_atomic_wavefunctions,
             py::arg("psi"), py::arg("randomize_phase") = false);

    py::class_<dftcu::Evaluator>(m, "Evaluator")
        .def(py::init<dftcu::Grid&>())
        .def("add_functional", &dftcu::Evaluator::add_functional)
        .def("add_functional",
             [](dftcu::Evaluator& self, std::shared_ptr<dftcu::ThomasFermi> f) {
                 self.add_functional(f);
             })
        .def("add_functional",
             [](dftcu::Evaluator& self, std::shared_ptr<dftcu::vonWeizsacker> f) {
                 self.add_functional(f);
             })
        .def("add_functional", [](dftcu::Evaluator& self,
                                  std::shared_ptr<dftcu::WangTeter> f) { self.add_functional(f); })
        .def("add_functional", [](dftcu::Evaluator& self,
                                  std::shared_ptr<dftcu::revHC> f) { self.add_functional(f); })
        .def("add_functional", [](dftcu::Evaluator& self,
                                  std::shared_ptr<dftcu::Hartree> f) { self.add_functional(f); })
        .def("add_functional", [](dftcu::Evaluator& self,
                                  std::shared_ptr<dftcu::LDA_PZ> f) { self.add_functional(f); })
        .def("add_functional",
             [](dftcu::Evaluator& self, std::shared_ptr<dftcu::PBE> f) { self.add_functional(f); })
        .def("add_functional", [](dftcu::Evaluator& self,
                                  std::shared_ptr<dftcu::Ewald> f) { self.add_functional(f); })
        .def("add_functional",
             [](dftcu::Evaluator& self, std::shared_ptr<dftcu::LocalPseudo> f) {
                 self.add_functional(f);
             })
        .def("compute", &dftcu::Evaluator::compute);

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

    py::class_<dftcu::LocalPseudo, std::shared_ptr<dftcu::LocalPseudo>>(m, "LocalPseudo")
        .def(py::init<dftcu::Grid&, std::shared_ptr<dftcu::Atoms>>(), py::arg("grid"),
             py::arg("atoms"))
        .def("init_tab_vloc", &dftcu::LocalPseudo::init_tab_vloc, py::arg("type"),
             py::arg("r_grid"), py::arg("vloc_r"), py::arg("rab"), py::arg("zp"), py::arg("omega"))
        .def("set_valence_charge", &dftcu::LocalPseudo::set_valence_charge, py::arg("type"),
             py::arg("zp"))
        .def("set_gcut", &dftcu::LocalPseudo::set_gcut, py::arg("gcut"))
        .def("compute_potential",
             [](dftcu::LocalPseudo& self, dftcu::RealField& vloc) { self.compute(vloc); })
        .def("compute", [](dftcu::LocalPseudo& self, const dftcu::RealField& rho,
                           dftcu::RealField& v_out) { return self.compute(rho, v_out); })
        .def("get_tab_vloc", &dftcu::LocalPseudo::get_tab_vloc, py::arg("type"))
        .def("get_alpha", &dftcu::LocalPseudo::get_alpha, py::arg("type"))
        .def("get_vloc_g_shells", &dftcu::LocalPseudo::get_vloc_g_shells, py::arg("type"),
             py::arg("g_shells"))
        .def("get_dq", &dftcu::LocalPseudo::get_dq)
        .def("get_nqx", &dftcu::LocalPseudo::get_nqx)
        .def("get_omega", &dftcu::LocalPseudo::get_omega);

    py::class_<dftcu::NonLocalPseudo, std::shared_ptr<dftcu::NonLocalPseudo>>(m, "NonLocalPseudo")
        .def(py::init<dftcu::Grid&>())
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
        .def("calculate_energy", &dftcu::NonLocalPseudo::calculate_energy, py::arg("psi"),
             py::arg("occupations"))
        .def("clear", &dftcu::NonLocalPseudo::clear)
        .def("num_projectors", &dftcu::NonLocalPseudo::num_projectors)
        .def("get_tab_beta", &dftcu::NonLocalPseudo::get_tab_beta, py::arg("type"), py::arg("nb"))
        .def("get_projector", &dftcu::NonLocalPseudo::get_projector, py::arg("idx"))
        .def("get_projections", &dftcu::NonLocalPseudo::get_projections);

    py::class_<dftcu::Hamiltonian>(m, "Hamiltonian")
        .def(py::init<dftcu::Grid&, dftcu::Evaluator&>())
        .def("update_potentials", &dftcu::Hamiltonian::update_potentials)
        .def("apply", &dftcu::Hamiltonian::apply)
        .def("set_nonlocal", &dftcu::Hamiltonian::set_nonlocal)
        .def("has_nonlocal", &dftcu::Hamiltonian::has_nonlocal)
        .def(
            "get_nonlocal",
            [](dftcu::Hamiltonian& self) -> dftcu::NonLocalPseudo& { return self.get_nonlocal(); },
            py::return_value_policy::reference)
        .def("v_loc", (dftcu::RealField & (dftcu::Hamiltonian::*)()) & dftcu::Hamiltonian::v_loc,
             py::return_value_policy::reference)
        .def("set_ecutrho", &dftcu::Hamiltonian::set_ecutrho, py::arg("ecutrho"));

    py::class_<dftcu::SubspaceSolver>(m, "SubspaceSolver")
        .def(py::init<dftcu::Grid&>())
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
        .def_readwrite("davidson_max_iter", &dftcu::SCFSolver::Options::davidson_max_iter)
        .def_readwrite("davidson_tol", &dftcu::SCFSolver::Options::davidson_tol)
        .def_readwrite("verbose", &dftcu::SCFSolver::Options::verbose);

    py::class_<dftcu::SCFSolver>(m, "SCFSolver")
        .def(py::init([](dftcu::Grid& grid) {
                 return new dftcu::SCFSolver(grid, dftcu::SCFSolver::Options());
             }),
             py::arg("grid"))
        .def(py::init<dftcu::Grid&, const dftcu::SCFSolver::Options&>(), py::arg("grid"),
             py::arg("options"))
        .def("solve", &dftcu::SCFSolver::solve, py::arg("ham"), py::arg("psi"),
             py::arg("occupations"), py::arg("rho_init"), py::arg("atoms"), py::arg("ecutrho"))
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
        .def("compute_energy_breakdown", &dftcu::SCFSolver::compute_energy_breakdown,
             py::arg("eigenvalues"), py::arg("occupations"), py::arg("ham"), py::arg("psi"),
             py::arg("rho"));

    py::class_<dftcu::DensityBuilder>(m, "DensityBuilder")
        .def(py::init<dftcu::Grid&, std::shared_ptr<dftcu::Atoms>>())
        .def("set_atomic_rho_g", &dftcu::DensityBuilder::set_atomic_rho_g)
        .def("set_atomic_rho_r", &dftcu::DensityBuilder::set_atomic_rho_r)
        .def("build_density", &dftcu::DensityBuilder::build_density)
        .def("set_gcut", &dftcu::DensityBuilder::set_gcut, py::arg("gcut"));
}
