#include "functional/hartree.cuh"
#include "functional/kedf/tf.cuh"
#include "functional/kedf/vw.cuh"
#include "functional/kedf/wt.cuh"
#include "functional/pseudo.cuh"
#include "functional/xc/lda_pz.cuh"
#include "model/atoms.cuh"
#include "model/field.cuh"
#include "model/grid.cuh"
#include "utilities/kernels.cuh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(dftcu, m) {
    py::class_<dftcu::Grid>(m, "Grid")
        .def(py::init<const std::vector<double>&, const std::vector<int>&>())
        .def("nnr", &dftcu::Grid::nnr)
        .def("dv", &dftcu::Grid::dv)
        .def("volume", &dftcu::Grid::volume);

    py::class_<dftcu::Atom>(m, "Atom")
        .def(py::init<double, double, double, double, int>())
        .def_readwrite("x", &dftcu::Atom::x)
        .def_readwrite("y", &dftcu::Atom::y)
        .def_readwrite("z", &dftcu::Atom::z)
        .def_readwrite("charge", &dftcu::Atom::charge)
        .def_readwrite("type", &dftcu::Atom::type);

    py::class_<dftcu::Atoms>(m, "Atoms").def(py::init<const std::vector<dftcu::Atom>&>());

    py::class_<dftcu::RealField>(m, "RealField")
        .def(py::init<const dftcu::Grid&, int>())
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
        .def("dot",
             [](const dftcu::RealField& self, const dftcu::RealField& other) {
                 return dftcu::dot_product(self.size(), self.data(), other.data());
             })
        .def("integral", [](const dftcu::RealField& self) {
            dftcu::RealField ones(self.grid());
            ones.fill(1.0);
            return dftcu::dot_product(self.size(), self.data(), ones.data()) * self.grid().dv();
        });

    py::class_<dftcu::Hartree>(m, "Hartree")
        .def(py::init<const dftcu::Grid&>())
        .def("compute", [](dftcu::Hartree& self, const dftcu::RealField& rho,
                           dftcu::RealField& vh) {
            double energy;
            self.compute(rho, vh, energy);
            // Hartree energy in DFTpy: e_h = np.einsum("ijk, ijk->", v_h_of_r, rho) *
            // density.grid.dV / 2.0
            energy = 0.5 * dftcu::dot_product(rho.size(), rho.data(), vh.data()) * rho.grid().dv();
            return energy;
        });

    py::class_<dftcu::LocalPseudo>(m, "LocalPseudo")
        .def(py::init<const dftcu::Grid&, const dftcu::Atoms&>())
        .def("set_vloc", &dftcu::LocalPseudo::set_vloc)
        .def("compute", &dftcu::LocalPseudo::compute);

    // KEDF functionals
    py::class_<dftcu::ThomasFermi>(m, "ThomasFermi")
        .def(py::init<double>(), py::arg("coeff") = 1.0)
        .def(
            "compute",
            [](dftcu::ThomasFermi& self, const dftcu::RealField& rho, dftcu::RealField& v_kedf) {
                return self.compute(rho, v_kedf);
            },
            py::arg("rho"), py::arg("v_kedf"),
            "Compute Thomas-Fermi kinetic energy and potential\n\n"
            "Parameters:\n"
            "  rho: Input density field\n"
            "  v_kedf: Output potential field (δE/δρ)\n\n"
            "Returns:\n"
            "  energy: Total Thomas-Fermi kinetic energy");

    py::class_<dftcu::vonWeizsacker>(m, "vonWeizsacker")
        .def(py::init<double>(), py::arg("coeff") = 1.0)
        .def(
            "compute",
            [](dftcu::vonWeizsacker& self, const dftcu::RealField& rho, dftcu::RealField& v_kedf) {
                return self.compute(rho, v_kedf);
            },
            py::arg("rho"), py::arg("v_kedf"),
            "Compute von Weizsacker kinetic energy and potential\n\n"
            "Parameters:\n"
            "  rho: Input density field\n"
            "  v_kedf: Output potential field (δE/δρ)\n\n"
            "Returns:\n"
            "  energy: Total von Weizsacker kinetic energy");

    py::class_<dftcu::WangTeter>(m, "WangTeter")
        .def(py::init<double, double, double>(), 
             py::arg("coeff") = 1.0, py::arg("alpha") = 5.0/6.0, py::arg("beta") = 5.0/6.0)
        .def(
            "compute",
            [](dftcu::WangTeter& self, const dftcu::RealField& rho, dftcu::RealField& v_kedf) {
                return self.compute(rho, v_kedf);
            },
            py::arg("rho"), py::arg("v_kedf"),
            "Compute Wang-Teter non-local kinetic energy and potential\n\n"
            "Parameters:\n"
            "  rho: Input density field\n"
            "  v_kedf: Output potential field (δE/δρ)\n\n"
            "Returns:\n"
            "  energy: Total Wang-Teter non-local kinetic energy");

    py::class_<dftcu::LDA_PZ>(m, "LDA_PZ")
        .def(py::init<>())
        .def(
            "compute",
            [](dftcu::LDA_PZ& self, const dftcu::RealField& rho, dftcu::RealField& v_xc) {
                return self.compute(rho, v_xc);
            },
            py::arg("rho"), py::arg("v_xc"),
            "Compute LDA Perdew-Zunger exchange-correlation energy and potential\n\n"
            "Parameters:\n"
            "  rho: Input density field\n"
            "  v_xc: Output potential field (δE/δρ)\n\n"
            "Returns:\n"
            "  energy: Total XC energy");
}
