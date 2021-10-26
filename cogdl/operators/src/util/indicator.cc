#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

void startProfile();
void endProfile();

PYBIND11_MODULE(ind, m) {
  m.doc() = "indicator. ";
  m.def("start", &startProfile, "indicator");
  m.def("end", &endProfile, "Gaussian");
}