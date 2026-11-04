#include "aster/accounting.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>

namespace {
std::atomic<std::size_t> allocations{0};
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void close(double actual, double expected) {
    check(std::abs(actual - expected) <= 1e-12 + 1e-10 * std::abs(expected), "wrong accounting value");
}
template <class Exception, class Function> void rejects(Function&& function) {
    bool caught = false;
    try { function(); } catch (const Exception&) { caught = true; }
    check(caught, "expected exception was not raised");
}
void same(aster::Accounting first, aster::Accounting second) {
    check(first.pnl == second.pnl && first.costs == second.costs && first.turnover == second.turnover,
          "failed event changed ledger state");
}
}

void* operator new(std::size_t bytes) {
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* memory = std::malloc(bytes == 0 ? 1 : bytes)) return memory;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

int main() {
    try {
        std::array<double, 3> prices{1.0, 1.1, 1.2};
        std::array<double, 2> holdings{0.5, 0.5};
        std::array<double, 1> pnl{}, costs{}, turnover{};
        aster::hedge_pnl(prices, holdings, 1, 2, 0.01, pnl, costs, turnover);
        close(pnl[0], 0.089); close(costs[0], 0.011); close(turnover[0], 1.1);

        aster::Ledger ledger(1.0, 0.01);
        close(ledger.update(1.0, 0.5).pnl, -0.005);
        close(ledger.update(1.1, 0.5).pnl, 0.045);
        auto finished = ledger.finish(1.2);
        close(finished.pnl, pnl[0]); close(finished.costs, costs[0]); close(finished.turnover, turnover[0]);
        rejects<std::logic_error>([&] { ledger.finish(1.2); });
        rejects<std::logic_error>([&] { ledger.update(1.2, 0.0); });
        same(finished, ledger.result());
        ledger.reset(1.0, 0.0);
        ledger.update(1.0, -0.5); ledger.update(1.1, -0.5);
        close(ledger.finish(1.2).pnl, -0.1);

        holdings = {-0.5, -0.5};
        aster::hedge_pnl(prices, holdings, 1, 2, 0.01, pnl, costs, turnover);
        close(pnl[0], -0.111); close(costs[0], 0.011);
        holdings = {0.0, 0.0};
        aster::hedge_pnl(prices, holdings, 1, 2, 0.01, pnl, costs, turnover);
        close(pnl[0], 0); close(costs[0], 0); close(turnover[0], 0);
        std::array<double, 2> single_prices{1, 2};
        std::array<double, 1> single_holding{0.5};
        aster::hedge_pnl(single_prices, single_holding, 1, 1, 0.01, pnl, costs, turnover);
        close(pnl[0], 0.485); close(costs[0], 0.015); close(turnover[0], 1.5);
        aster::hedge_pnl({}, {}, 0, 1, 0.0, {}, {}, {});

        rejects<std::invalid_argument>([&] { aster::hedge_pnl(prices, holdings, 1, 0, 0.01, pnl, costs, turnover); });
        rejects<std::invalid_argument>([&] { aster::hedge_pnl(prices, holdings, 2, 2, 0.01, pnl, costs, turnover); });
        rejects<std::invalid_argument>([&] { aster::hedge_pnl(prices, holdings, 1, 2, -0.01, pnl, costs, turnover); });
        rejects<std::invalid_argument>([&] { aster::hedge_pnl(prices, holdings, 1, 2, 0.01, {}, costs, turnover); });
        rejects<std::invalid_argument>([&] { aster::hedge_pnl(prices, holdings, 1, 2, 0.01, pnl, pnl, turnover); });
        rejects<std::invalid_argument>([&] { aster::hedge_pnl(prices, holdings, 1, 2, 0.01, {prices.data(), 1}, costs, turnover); });
        rejects<std::overflow_error>([&] { aster::hedge_pnl({}, {}, std::numeric_limits<std::size_t>::max(), 1, 0.01, {}, {}, {}); });
        rejects<std::overflow_error>([&] { aster::hedge_pnl({}, {}, 1, std::numeric_limits<std::size_t>::max(), 0.01, {}, {}, {}); });
        holdings[0] = std::numeric_limits<double>::quiet_NaN();
        pnl[0] = 42;
        rejects<std::invalid_argument>([&] { aster::hedge_pnl(prices, holdings, 1, 2, 0.01, pnl, costs, turnover); });
        check(pnl[0] == 42, "invalid input wrote outputs");
        holdings = {0.5, 0.5};
        prices[0] = std::numeric_limits<double>::infinity();
        rejects<std::invalid_argument>([&] { aster::hedge_pnl(prices, holdings, 1, 2, 0.01, pnl, costs, turnover); });
        prices = {1, 1.1, 1.2};

        ledger.reset(1.0, 0.01);
        ledger.update(1.0, 0.5);
        auto saved = ledger.result();
        rejects<std::invalid_argument>([&] { ledger.update(std::numeric_limits<double>::quiet_NaN(), 0); });
        rejects<std::invalid_argument>([&] { ledger.finish(std::numeric_limits<double>::infinity()); });
        rejects<std::invalid_argument>([&] { ledger.reset(1, -1); });
        same(saved, ledger.result());
        ledger.reset(1.0, 1.0);
        ledger.update(1.0, 1e308);
        saved = ledger.result();
        rejects<std::overflow_error>([&] { ledger.update(1e308, 1e308); });
        same(saved, ledger.result());
        std::array<double, 2> huge_prices{1, 1e308};
        std::array<double, 1> huge_holdings{1e308};
        rejects<std::overflow_error>([&] { aster::hedge_pnl(huge_prices, huge_holdings, 1, 1, 0, pnl, costs, turnover); });

        ledger.reset(1, 0.01);
        auto count = allocations.load();
        ledger.update(1, 0.5); ledger.update(1.1, 0.5); ledger.finish(1.2);
        aster::hedge_pnl_unchecked(prices, holdings, 1, 2, 0.01, pnl, costs, turnover);
        aster::hedge_pnl(prices, holdings, 1, 2, 0.01, pnl, costs, turnover);
        check(allocations.load() == count, "successful accounting allocated from the heap");
        std::cout << "Native accounting, validation, streaming, and allocation checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
