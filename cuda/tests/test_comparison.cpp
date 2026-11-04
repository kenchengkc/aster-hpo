#include "comparison.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    try {
        aster_cuda::Comparison stats;
        const std::array expected{0.089, 0.011, 1.1};
        auto actual = expected;
        aster_cuda::compare(expected, actual, "literal", stats);
        if (stats.values_checked != 3) throw std::runtime_error("comparison did not inspect all outputs");
        // Any individual P&L, cost, or turnover error must fail, even with other fields correct.
        for (std::size_t i = 0; i < actual.size(); ++i) {
            actual = expected;
            actual[i] += 1e-5;
            bool rejected = false;
            try { aster_cuda::compare(expected, actual, "corrupted output", stats); }
            catch (const std::runtime_error&) { rejected = true; }
            if (!rejected) throw std::runtime_error("corrupted accounting output was accepted");
        }
        const std::array zero{0.0};
        const std::array near_zero{5e-13};
        aster_cuda::compare(zero, near_zero, "absolute tolerance", stats);
        for (const double invalid : {std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::infinity(), 2e-12}) {
            bool rejected = false;
            try { aster_cuda::compare(zero, std::array{invalid}, "invalid", stats); }
            catch (const std::runtime_error&) { rejected = true; }
            if (!rejected) throw std::runtime_error("nonfinite or excessive zero error was accepted");
        }
        bool rejected = false;
        try { aster_cuda::compare(expected, zero, "shape", stats); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("output size mismatch was accepted");
        std::cout << "Comparator checks passed, including each corrupted output.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
