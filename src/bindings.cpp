#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "mlp.h"
#include <memory>
#include "autograd.h"

namespace py = pybind11;

PYBIND11_MODULE(CPPgrad, m) {
    m.doc() = "MLP module"; 

    py::class_<Data,std::shared_ptr<Data>>(m, "Data")
        .def(py::init<const double&>())
        .def(py::init<const double&, const std::vector<Data*>&>())
        .def("getGrad", &Data::getGrad)
        .def("setGrad", &Data::setGrad)
        .def("getData", &Data::getData)
        .def("setData", &Data::setData)
        .def("sigmoid", (Data (Data::*)() const) &Data::sigmoid)
        .def("backward", &Data::backward)
        .def(py::self + double())
        .def(py::self + py::self)
        .def(py::self * double())
        .def(py::self * py::self)
        .def(py::self - double())
        .def(py::self - py::self)
        .def(py::self / double())
        .def(py::self / py::self)
        .def(double() + py::self)
        .def(double() * py::self)
        .def(double() - py::self)
        .def(double() / py::self)
        .def(py::self ^ double())
        .def(py::self ^ py::self)
        .def(double() ^ py::self)
        .def(py::self += py::self)
        .def(py::self *= py::self);

    py::class_<Neuron>(m, "Neuron")
        .def(py::init<int>())
        .def("forward", &Neuron::forward)
        .def("backward", &Neuron::backward)
        .def("getWeights", &Neuron::getWeights);
 
    py::class_<Layer>(m, "Layer")
        .def(py::init<int, int>())
        .def("forward", &Layer::forward)
        .def("backward", &Layer::backward)
        .def("getNeurons", &Layer::getNeurons);

    py::class_<MLP>(m, "MLP")
        .def(py::init<const std::vector<int>&>())
        .def("forward", &MLP::forward)
        .def("backward", &MLP::backward)
        .def("summary", &MLP::summary)
        .def("getLayers", &MLP::getLayers);
}
