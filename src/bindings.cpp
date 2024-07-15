#include <pybind11/pybind11.h>
#include "mlp.h"
#include "autograd.h"

namespace py = pybind11;

PYBIND11_MODULE(mlp, m) {
    m.doc() = "MLP module"; 
    
   py::class_<Data>(m, "Data")
        .def(py::init<const double&>())
        .def(py::init<const double&, const std::vector<Data*>&>())
        .def("getGrad", &Data::getGrad)
        .def("setGrad", &Data::setGrad)
        .def("sigmoid", (Data (Data::*)() const) &Data::sigmoid) 
        .def("backward", &Data::backward);
    py::class_<Neuron>(m, "Neuron")
        .def(py::init<int>())
        .def("forward", &Neuron::forward)
        .def("backward", &Neuron::backward);
 
    py::class_<Layer>(m, "Layer")
        .def(py::init<int, int>())
        .def("forward", &Layer::forward)
        .def("backward", &Layer::backward);

    py::class_<MLP>(m, "MLP")
        .def(py::init<const std::vector<int>&>())
        .def("forward", &MLP::forward)
        .def("backward", &MLP::backward)
        .def("summary", &MLP::summary);
}
