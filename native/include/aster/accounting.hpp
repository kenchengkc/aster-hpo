#pragma once

#include <cstddef>
#include <span>

namespace aster {

struct Accounting {
    double pnl;
    double costs;
    double turnover;
};

class Ledger {
public:
    Ledger(double initial_price, double cost_rate);
    Accounting update(double price, double target_holding);
    Accounting finish(double terminal_price);
    void reset(double initial_price, double cost_rate);
    Accounting result() const;

private:
    double price_ = 0;
    double holding_ = 0;
    double gain_ = 0;
    double turnover_ = 0;
    double cost_rate_ = 0;
    bool closed_ = false;
};

void hedge_pnl(std::span<const double> prices, std::span<const double> holdings,
               std::size_t paths, std::size_t intervals, double cost_rate,
               std::span<double> pnl, std::span<double> costs, std::span<double> turnover);

// Prepared-input measurement primitive. The caller guarantees valid dimensions,
// nonoverlapping buffers, finite data, and arithmetic producing finite outputs.
void hedge_pnl_unchecked(std::span<const double> prices, std::span<const double> holdings,
                         std::size_t paths, std::size_t intervals, double cost_rate,
                         std::span<double> pnl, std::span<double> costs, std::span<double> turnover);

} // namespace aster
