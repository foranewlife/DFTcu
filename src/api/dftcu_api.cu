#include "fft/fft_solver.cuh"
#include "functional/ewald.cuh"
#include "functional/hartree.cuh"
#include "functional/kedf/hc.cuh"
#include "functional/kedf/tf.cuh"
#include "functional/kedf/vw.cuh"
#include "functional/kedf/wt.cuh"
#include "functional/pseudo.cuh"
#include "functional/xc/lda_pz.cuh"
#include "functional/xc/pbe.cuh"
#include "math/linesearch.cuh"
#include "model/atoms.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"
#include "solver/evaluator.cuh"
#include "solver/optimizer.cuh"
#include "solver/tn_optimizer_legacy.cuh"
#include "utilities/kernels.cuh"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(dftcu, m) {
    py::class_<dftcu::Grid, std::shared_ptr<dftcu::Grid>>(m, "Grid")
        .def(py::init([](const std::vector<double>& lattice, const std::vector<int>& nr) {
            return std::make_shared<dftcu::Grid>(lattice, nr);
        }))
        .def("nnr", &dftcu::Grid::nnr)
        .def("dv", &dftcu::Grid::dv)
        .def("volume", &dftcu::Grid::volume)
        .def("g2max", &dftcu::Grid::g2max)
        .def("lattice",
             [](dftcu::Grid& self) {
                 std::vector<std::vector<double>> lat(3, std::vector<double>(3));
                 for (int i = 0; i < 3; ++i)
                     for (int j = 0; j < 3; ++j)
                         lat[i][j] = self.lattice()[i][j];
                 return lat;
             })
        .def("rec_lattice", [](dftcu::Grid& self) {
            std::vector<std::vector<double>> lat(3, std::vector<double>(3));
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    lat[i][j] = self.rec_lattice()[i][j];
            return lat;
        });

    py::class_<dftcu::Atom>(m, "Atom")
        .def(py::init<double, double, double, double, int>())
        .def_readwrite("x", &dftcu::Atom::x)
        .def_readwrite("y", &dftcu::Atom::y)
        .def_readwrite("z", &dftcu::Atom::z)
        .def_readwrite("charge", &dftcu::Atom::charge)
        .def_readwrite("type", &dftcu::Atom::type);

    py::class_<dftcu::Atoms, std::shared_ptr<dftcu::Atoms>>(m, "Atoms")
        .def(py::init<const std::vector<dftcu::Atom>&>())
        .def("nat", &dftcu::Atoms::nat);

    py::class_<dftcu::RealField>(m, "RealField")
        .def(py::init<dftcu::Grid&, int>(), py::arg("grid"), py::arg("rank") = 1)
        .def("fill", &dftcu::RealField::fill)
        .def("copy_from_host",
             [](dftcu::RealField& self, py::array_t<double> array) {
                 auto buf = array.request();
                 if (buf.size != self.size())
                     throw std::runtime_error("Size mismatch");
                 self.copy_from_host((double*)buf.ptr);
             })
        .def("copy_to_host",
             [](dftcu::RealField& self, py::array_t<double> array) {
                 auto buf = array.request();
                 if (buf.size != self.size())
                     throw std::runtime_error("Size mismatch");
                 self.copy_to_host((double*)buf.ptr);
             })
        .def("dot", [](const dftcu::RealField& self,
                       const dftcu::RealField& other) { return self.dot(other); })
        .def("integral", &dftcu::RealField::integral)
        .def("size", &dftcu::RealField::size);

    py::class_<dftcu::ComplexField>(m, "ComplexField")
        .def(py::init<dftcu::Grid&, int>(), py::arg("grid"), py::arg("rank") = 1)
        .def("fill",
             [](dftcu::ComplexField& self, std::complex<double> val) {
                 gpufftComplex gval;
                 gval.x = val.real();
                 gval.y = val.imag();
                 self.fill(gval);
             })
        .def("copy_from_host",
             [](dftcu::ComplexField& self, py::array_t<std::complex<double>> array) {
                 auto buf = array.request();
                 if (buf.size != self.size())
                     throw std::runtime_error("Size mismatch");
                 self.copy_from_host((gpufftComplex*)buf.ptr);
             })
        .def("copy_to_host",
             [](dftcu::ComplexField& self, py::array_t<std::complex<double>> array) {
                 auto buf = array.request();
                 if (buf.size != self.size())
                     throw std::runtime_error("Size mismatch");
                 self.copy_to_host((gpufftComplex*)buf.ptr);
             })
        .def("size", &dftcu::ComplexField::size);

    py::class_<dftcu::FFTSolver>(m, "FFTSolver")
        .def(py::init<dftcu::Grid&>())
        .def("forward", &dftcu::FFTSolver::forward)
        .def("backward", &dftcu::FFTSolver::backward);

    py::class_<dftcu::Hartree, std::shared_ptr<dftcu::Hartree>>(m, "Hartree")
        .def(py::init<dftcu::Grid&>())
        .def("compute",
             [](dftcu::Hartree& self, const dftcu::RealField& rho, dftcu::RealField& vh) {
                 double energy = 0.0;
                 self.compute(rho, vh, energy);
                 return energy;
             });

    py::class_<dftcu::Ewald, std::shared_ptr<dftcu::Ewald>>(m, "Ewald")
        .def(py::init<dftcu::Grid&, std::shared_ptr<dftcu::Atoms>, double, int>(), py::arg("grid"),
             py::arg("atoms"), py::arg("precision") = 1e-8, py::arg("bspline_order") = 10)
        .def(
            "compute", [](dftcu::Ewald& self, bool use_pme) { return self.compute(use_pme); },
            py::arg("use_pme") = false)
        .def(
            "compute",
            [](dftcu::Ewald& self, const dftcu::RealField& rho, dftcu::RealField& v_out) {
                return self.compute(rho, v_out);
            },
            py::arg("rho"), py::arg("v_out"))
        .def("set_eta", &dftcu::Ewald::set_eta);

    py::class_<dftcu::LocalPseudo, std::shared_ptr<dftcu::LocalPseudo>>(m, "LocalPseudo")
        .def(py::init<dftcu::Grid&, std::shared_ptr<dftcu::Atoms>>())
        .def("set_vloc", &dftcu::LocalPseudo::set_vloc)
        .def("set_vloc_radial", &dftcu::LocalPseudo::set_vloc_radial)
        .def("compute", [](dftcu::LocalPseudo& self, dftcu::RealField& v) { self.compute(v); })
        .def("compute", [](dftcu::LocalPseudo& self, const dftcu::RealField& rho,
                           dftcu::RealField& v_out) { return self.compute(rho, v_out); });

    // KEDF functionals
    py::class_<dftcu::KEDF_Base, std::shared_ptr<dftcu::KEDF_Base>>(m, "KEDF_Base");

    py::class_<dftcu::ThomasFermi, dftcu::KEDF_Base, std::shared_ptr<dftcu::ThomasFermi>>(
        m, "ThomasFermi")
        .def(py::init<double>(), py::arg("coeff") = 1.0)
        .def("compute", &dftcu::ThomasFermi::compute, py::arg("rho"), py::arg("v_kedf"));

    py::class_<dftcu::vonWeizsacker, dftcu::KEDF_Base, std::shared_ptr<dftcu::vonWeizsacker>>(
        m, "vonWeizsacker")
        .def(py::init<double>(), py::arg("coeff") = 1.0)
        .def("compute", &dftcu::vonWeizsacker::compute, py::arg("rho"), py::arg("v_kedf"));

    py::class_<dftcu::WangTeter, dftcu::KEDF_Base, std::shared_ptr<dftcu::WangTeter>>(m,
                                                                                      "WangTeter")
        .def(py::init<double, double, double>(), py::arg("coeff") = 1.0,
             py::arg("alpha") = 5.0 / 6.0, py::arg("beta") = 5.0 / 6.0)
        .def("compute", &dftcu::WangTeter::compute, py::arg("rho"), py::arg("v_kedf"));

    py::class_<dftcu::revHC, dftcu::KEDF_Base, std::shared_ptr<dftcu::revHC>>(m, "revHC")
        .def(py::init<dftcu::Grid&, double, double>(), py::arg("grid"), py::arg("alpha") = 2.0,
             py::arg("beta") = 2.0 / 3.0)
        .def("compute", &dftcu::revHC::compute, py::arg("rho"), py::arg("v_kedf"));

    py::class_<dftcu::LDA_PZ, std::shared_ptr<dftcu::LDA_PZ>>(m, "LDA_PZ")
        .def(py::init<>())
        .def("compute", &dftcu::LDA_PZ::compute, py::arg("rho"), py::arg("v_xc"));

    py::class_<dftcu::PBE, std::shared_ptr<dftcu::PBE>>(m, "PBE")
        .def(py::init<dftcu::Grid&>())
        .def("compute", &dftcu::PBE::compute, py::arg("rho"), py::arg("v_xc"));

    py::class_<dftcu::Evaluator>(m, "Evaluator")
        .def(py::init<dftcu::Grid&>())
        .def("add_functional",
             [](dftcu::Evaluator& self, std::shared_ptr<dftcu::ThomasFermi> f) {
                 self.add_functional(dftcu::Functional(f));
             })
        .def("add_functional",
             [](dftcu::Evaluator& self, std::shared_ptr<dftcu::vonWeizsacker> f) {
                 self.add_functional(dftcu::Functional(f));
             })
        .def("add_functional",
             [](dftcu::Evaluator& self, std::shared_ptr<dftcu::WangTeter> f) {
                 self.add_functional(dftcu::Functional(f));
             })
        .def("add_functional",
             [](dftcu::Evaluator& self, std::shared_ptr<dftcu::revHC> f) {
                 self.add_functional(dftcu::Functional(f));
             })
        .def("add_functional",
             [](dftcu::Evaluator& self, std::shared_ptr<dftcu::LDA_PZ> f) {
                 self.add_functional(dftcu::Functional(f));
             })
        .def("add_functional",
             [](dftcu::Evaluator& self, std::shared_ptr<dftcu::PBE> f) {
                 self.add_functional(dftcu::Functional(f));
             })
        .def("add_functional",
             [](dftcu::Evaluator& self, std::shared_ptr<dftcu::Hartree> f) {
                 self.add_functional(dftcu::Functional(f));
             })
        .def("add_functional",
             [](dftcu::Evaluator& self, std::shared_ptr<dftcu::LocalPseudo> f) {
                 self.add_functional(dftcu::Functional(f));
             })
        .def("add_functional",
             [](dftcu::Evaluator& self, std::shared_ptr<dftcu::Ewald> f) {
                 self.add_functional(dftcu::Functional(f));
             })
        .def("clear", &dftcu::Evaluator::clear)
        .def("compute", &dftcu::Evaluator::compute);

    py::class_<dftcu::OptimizationOptions>(m, "OptimizationOptions")
        .def(py::init<>())
        .def_readwrite("max_iter", &dftcu::OptimizationOptions::max_iter)
        .def_readwrite("econv", &dftcu::OptimizationOptions::econv)
        .def_readwrite("ncheck", &dftcu::OptimizationOptions::ncheck)
        .def_readwrite("step_size", &dftcu::OptimizationOptions::step_size);

    py::class_<dftcu::SimpleOptimizer>(m, "SimpleOptimizer")
        .def(py::init<dftcu::Grid&, dftcu::OptimizationOptions>(), py::arg("grid"),
             py::arg("options") = dftcu::OptimizationOptions())
        .def("solve", &dftcu::SimpleOptimizer::solve);

    py::class_<dftcu::CGOptimizer>(m, "CGOptimizer")
        .def(py::init<dftcu::Grid&, dftcu::OptimizationOptions>(), py::arg("grid"),
             py::arg("options") = dftcu::OptimizationOptions())
        .def("solve", &dftcu::CGOptimizer::solve);

    py::class_<dftcu::TNOptimizer>(m, "TNOptimizer")
        .def(py::init<dftcu::Grid&, dftcu::OptimizationOptions>(), py::arg("grid"),
             py::arg("options") = dftcu::OptimizationOptions())
        .def("solve", &dftcu::TNOptimizer::solve);

    py::class_<dftcu::TNOptimizerLegacy, std::shared_ptr<dftcu::TNOptimizerLegacy>>(
        m, "TNOptimizerLegacy")
        .def(py::init<dftcu::Grid&, dftcu::OptimizationOptions>(), py::arg("grid"),
             py::arg("options") = dftcu::OptimizationOptions())
        .def("solve", &dftcu::TNOptimizerLegacy::solve);

    m.def(
        "scalar_search_wolfe1",
        [](std::function<double(double)> phi, std::function<double(double)> derphi, double phi0,
           double derphi0, double phi0_old, double c1, double c2, double amax, double amin,
           double xtol) {
            double alpha_star = 0, phi_star = 0;
            bool conv = dftcu::scalar_search_wolfe1(phi, derphi, phi0, derphi0, phi0_old, c1, c2,
                                                    amax, amin, xtol, alpha_star, phi_star);
            return std::make_tuple(conv, alpha_star, phi_star);
        },
        py::arg("phi"), py::arg("derphi"), py::arg("phi0"), py::arg("derphi0"),
        py::arg("phi0_old") = 1e10, py::arg("c1") = 1e-4, py::arg("c2") = 0.9,
        py::arg("amax") = 100.0, py::arg("amin") = 1e-8, py::arg("xtol") = 1e-14);
}
