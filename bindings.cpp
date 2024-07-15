
#include <pybind11/pybind11.h>

#include "mlp.h"
#include "autograd.h"

namespace py = pybind11;

PYBIND11_MODULE(mlp, m) {
    m.doc() = "MLP module"; // Optional module docstring

    // Expose Data class if necessary
    py::class_<Data>(m, "Data")
        .def(py::init<>())
        .def("getGrad", &Data::getGrad)
        .def("setGrad", &Data::setGrad)
        .def("sigmoid", &Data::sigmoid)
        .def("backward", &Data::backward);

    // Expose Neuron class
    py::class_<Neuron>(m, "Neuron")
        .def(py::init<int>())
        .def("forward", &Neuron::forward)
        .def("backward", &Neuron::backward);

    // Expose Layer class
    py::class_<Layer>(m, "Layer")
        .def(py::init<int, int>())
        .def("forward", &Layer::forward)
        .def("backward", &Layer::backward);

    // Expose MLP class
    py::class_<MLP>(m, "MLP")
        .def(py::init<const std::vector<int>&>())
        .def("forward", &MLP::forward)
        .def("backward", &MLP::backward)
        .def("summary", &MLP::summary);
}
