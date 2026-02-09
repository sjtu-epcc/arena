/* !/usr/bin/env c++
 * -*- coding:utf-8 -*-
 * Author: Chunyu Xue
 */

/* Refs:
 *  - Pybind11 Doc: https://pybind11.readthedocs.io/en/stable/reference.html
 */

#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "profiler.h"

namespace py = pybind11;


PYBIND11_MODULE(crius_cupti, m) {
    // Define module properties
    crius::define_properties(m);

    // Entrypoint of runnable profiling
    m.def("profile_runnable", [](py::function python_runnable, const std::string& metric) {
        // Cpp runnable
        std::function<void()> runnable =[python_runnable]() {
            python_runnable();
        };

        // Profile
        if (metric.size() == 0) {
            return crius::profile(std::move(runnable));
        } else {
            return crius::profile(std::move(runnable), metric);
        }

        // https://pybind11-jagerman.readthedocs.io/en/stable/advanced.html
        // Use std::move to move the return value contents into a new instance 
        // that will be owned by Python. 
    }, py::arg("runnable"), py::arg("metric") = "", py::return_value_policy::move);

    // Set metric cache
    m.def("set_cache_metrics", [](bool should_cache) {
        crius::setCacheMetrics(should_cache);
    }, py::arg("should_cache"));

}
