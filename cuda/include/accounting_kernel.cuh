#pragma once

// This kernel is compiled by both the standalone runner and CuPy NVRTC.
// Unsigned 64-bit dimensions avoid depending on host standard-library headers.
extern "C" __global__ void accounting_kernel(
    const double* prices, const double* holdings,
    unsigned long long paths, unsigned long long intervals, double rate,
    double* pnl, double* costs, double* turnover) {
    const auto path = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= paths) return;
    const auto price_offset = path * (intervals + 1);
    const auto holding_offset = path * intervals;
    double gain = 0, traded = 0, previous = 0;
    for (unsigned long long t = 0; t < intervals; ++t) {
        const double holding = holdings[holding_offset + t];
        const double price = prices[price_offset + t];
        gain += holding * (prices[price_offset + t + 1] - price);
        traded += fabs(holding - previous) * price;
        previous = holding;
    }
    traded += fabs(previous) * prices[price_offset + intervals];
    const double cost = rate * traded;
    pnl[path] = gain - cost;
    costs[path] = cost;
    turnover[path] = traded;
}
