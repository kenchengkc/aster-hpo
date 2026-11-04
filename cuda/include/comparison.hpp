#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

namespace aster_cuda {
inline constexpr double atol = 1e-12;
inline constexpr double rtol = 1e-10;

struct Comparison {
    std::size_t values_checked = 0;
    double max_absolute_error = 0;
    double max_normalized_error = 0;
};

inline void compare(std::span<const double> expected, std::span<const double> actual,
                    const std::string& label, Comparison& stats) {
    if (expected.size() != actual.size()) throw std::runtime_error(label + ": output size mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
            throw std::runtime_error(label + ": nonfinite result at path " + std::to_string(i));
        }
        const double error = std::abs(actual[i] - expected[i]);
        const double tolerance = atol + rtol * std::abs(expected[i]);
        ++stats.values_checked;
        stats.max_absolute_error = std::max(stats.max_absolute_error, error);
        stats.max_normalized_error = std::max(stats.max_normalized_error, error / tolerance);
        if (error > tolerance) {
            throw std::runtime_error(label + ": mismatch at path " + std::to_string(i) +
                                     ", expected " + std::to_string(expected[i]) +
                                     ", actual " + std::to_string(actual[i]));
        }
    }
}
}  // namespace aster_cuda
