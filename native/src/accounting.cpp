#include "aster/accounting.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace aster {
namespace {
void finite(double value) {
    if (!std::isfinite(value)) throw std::invalid_argument("Accounting inputs must be finite");
}
void valid_rate(double rate) {
    finite(rate);
    if (rate < 0) throw std::invalid_argument("Cost rate must be nonnegative");
}
Accounting checked_result(double gain, double turnover, double rate) {
    Accounting result{gain - rate * turnover, rate * turnover, turnover};
    if (!std::isfinite(gain) || !std::isfinite(result.pnl) ||
        !std::isfinite(result.costs) || !std::isfinite(turnover)) {
        throw std::overflow_error("Accounting produced a nonfinite result");
    }
    return result;
}
std::size_t product(std::size_t first, std::size_t second) {
    if (first && second > std::numeric_limits<std::size_t>::max() / first)
        throw std::overflow_error("Accounting dimensions overflow");
    return first * second;
}
bool overlap(std::span<const double> first, std::span<const double> second) {
    if (first.empty() || second.empty()) return false;
    const auto a = reinterpret_cast<std::uintptr_t>(first.data());
    const auto b = reinterpret_cast<std::uintptr_t>(second.data());
    return a < b + second.size_bytes() && b < a + first.size_bytes();
}
Accounting row(const double* prices, const double* holdings, std::size_t intervals, double rate) {
    double gain = 0, turnover = 0, previous = 0;
    for (std::size_t time = 0; time < intervals; ++time) {
        gain += holdings[time] * (prices[time + 1] - prices[time]);
        turnover += std::abs(holdings[time] - previous) * prices[time];
        previous = holdings[time];
    }
    turnover += std::abs(previous) * prices[intervals];
    return {gain - rate * turnover, rate * turnover, turnover};
}
}

Ledger::Ledger(double initial_price, double cost_rate) { reset(initial_price, cost_rate); }
void Ledger::reset(double initial_price, double cost_rate) {
    finite(initial_price);
    valid_rate(cost_rate);
    price_ = initial_price;
    cost_rate_ = cost_rate;
    holding_ = gain_ = turnover_ = 0;
    closed_ = false;
}
Accounting Ledger::result() const { return checked_result(gain_, turnover_, cost_rate_); }
Accounting Ledger::update(double price, double target_holding) {
    if (closed_) throw std::logic_error("Ledger is closed; reset before reuse");
    finite(price);
    finite(target_holding);
    const double gain = gain_ + holding_ * (price - price_);
    const double turnover = turnover_ + std::abs(target_holding - holding_) * price;
    const auto next = checked_result(gain, turnover, cost_rate_);
    price_ = price;
    holding_ = target_holding;
    gain_ = gain;
    turnover_ = turnover;
    return next;
}
Accounting Ledger::finish(double terminal_price) {
    auto final = update(terminal_price, 0);
    closed_ = true;
    return final;
}

void hedge_pnl_unchecked(std::span<const double> prices, std::span<const double> holdings,
                         std::size_t paths, std::size_t intervals, double cost_rate,
                         std::span<double> pnl, std::span<double> costs, std::span<double> turnover) {
    for (std::size_t path = 0; path < paths; ++path) {
        const auto value = row(prices.data() + path * (intervals + 1),
                               holdings.data() + path * intervals, intervals, cost_rate);
        pnl[path] = value.pnl;
        costs[path] = value.costs;
        turnover[path] = value.turnover;
    }
}

void hedge_pnl(std::span<const double> prices, std::span<const double> holdings,
               std::size_t paths, std::size_t intervals, double cost_rate,
               std::span<double> pnl, std::span<double> costs, std::span<double> turnover) {
    valid_rate(cost_rate);
    if (!intervals) throw std::invalid_argument("At least one interval is required");
    if (intervals == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Accounting dimensions overflow");
    const auto price_size = product(paths, intervals + 1);
    const auto holding_size = product(paths, intervals);
    product(price_size, sizeof(double));
    product(holding_size, sizeof(double));
    if (prices.size() != price_size || holdings.size() != holding_size ||
        pnl.size() != paths || costs.size() != paths || turnover.size() != paths)
        throw std::invalid_argument("Accounting buffer sizes do not match dimensions");
    const std::array<std::span<const double>, 5> buffers{prices, holdings, pnl, costs, turnover};
    for (const auto buffer : buffers) {
        const auto begin = reinterpret_cast<std::uintptr_t>(buffer.data());
        if (buffer.size_bytes() > std::numeric_limits<std::uintptr_t>::max() - begin)
            throw std::overflow_error("Accounting buffer address overflow");
        if (!buffer.empty() && buffer.data() == nullptr)
            throw std::invalid_argument("Accounting buffer is null");
    }
    for (std::size_t output = 2; output < buffers.size(); ++output)
        for (std::size_t other = 0; other < output; ++other)
            if (overlap(buffers[output], buffers[other]))
                throw std::invalid_argument("Accounting output buffers must not overlap");
    for (double value : prices) finite(value);
    for (double value : holdings) finite(value);
    for (std::size_t path = 0; path < paths; ++path) {
        const auto value = row(prices.data() + path * (intervals + 1),
                               holdings.data() + path * intervals, intervals, cost_rate);
        if (!std::isfinite(value.pnl) || !std::isfinite(value.costs) || !std::isfinite(value.turnover))
            throw std::overflow_error("Accounting produced a nonfinite result");
        pnl[path] = value.pnl;
        costs[path] = value.costs;
        turnover[path] = value.turnover;
    }
}
} // namespace aster
