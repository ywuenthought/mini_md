// src/mini_md/_core.cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

void demo() {};

PYBIND11_MODULE(mini_md_core, m)
{
    m.def("demo", &demo);
}
