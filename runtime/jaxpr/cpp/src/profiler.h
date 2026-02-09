/* !/usr/bin/env c++ header
 * -*- coding:utf-8 -*-
 * Author: Chunyu Xue
 */

#pragma once

#include <functional>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>

#include "../external/habitat-cu116/cpp/src/cuda/kernel.h"

using habitat::cuda::KernelInstance;

namespace crius {

void define_properties(pybind11::module& m);

void setCacheMetrics(bool should_cache);

std::vector<KernelInstance> profile(std::function<void()> runnable);

std::vector<KernelInstance> profile(std::function<void()> runnable, const std::string& metric);

}
