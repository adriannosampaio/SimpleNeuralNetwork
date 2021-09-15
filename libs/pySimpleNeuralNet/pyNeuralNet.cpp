#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "NeuralNet.hpp"

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(pyNeuralNet, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    auto nn_class = py::class_<NeuralNetwork>(m, "NeuralNetwork").def(py::init<const std::string&>());
    nn_class.def("feed_forward", &NeuralNetwork::feed_forward);
    m.def("add", &add, "A function which adds two numbers");
}