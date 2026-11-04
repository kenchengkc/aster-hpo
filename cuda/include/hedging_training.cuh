#pragma once
#ifndef __CUDACC_RTC__
#include <math.h>
#endif

// All launches use 128 threads. Reductions have a fixed tree and no FP atomics.
// Dimensions/indices are 64 bit; Python checks array shapes before the hot loop.
__device__ __forceinline__ double hedging_sign(double value) {
    return isnan(value) ? value : static_cast<double>((value > 0) - (value < 0));
}

extern "C" __global__ void hedging_forward(
    const double* prices, const double* w1, const double* b1,
    const double* w2, const double* b2,
    unsigned long long paths, unsigned long long steps, unsigned long long width,
    double strike, double scale, double* feature0, double* hidden, double* holdings) {
    const auto point = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (point >= paths * steps) return;
    const auto path = point / steps;
    const auto t = point % steps;
    const double money = log(prices[path * (steps + 1) + t] / strike) / scale;
    const double tau = static_cast<double>(steps - t) / static_cast<double>(steps);
    feature0[point] = money;
    double logit = 0;
    for (unsigned long long j = 0; j < width; ++j) {
        const double value = tanh((money * w1[j] + tau * w1[width + j]) + b1[j]);
        hidden[point * width + j] = value;
        logit += value * w2[j];
    }
    logit += b2[0];
    const double exponent = exp(-fabs(logit));
    holdings[point] = logit >= 0 ? 1 / (1 + exponent) : exponent / (1 + exponent);
}

extern "C" __global__ void hedging_error(
    const double* prices, const double* holdings,
    unsigned long long paths, unsigned long long steps,
    double cost_rate, double premium, double strike, double* errors) {
    const auto path = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= paths) return;
    const auto price_offset = path * (steps + 1);
    const auto holding_offset = path * steps;
    double gain = 0, traded = 0, previous = 0;
    for (unsigned long long t = 0; t < steps; ++t) {
        const double holding = holdings[holding_offset + t];
        const double price = prices[price_offset + t];
        gain += holding * (prices[price_offset + t + 1] - price);
        traded += fabs(holding - previous) * price;
        previous = holding;
    }
    const double terminal = prices[price_offset + steps];
    traded += fabs(previous) * terminal;
    const double payoff = terminal - strike;
    const double positive_payoff = isnan(payoff) ? payoff : (payoff > 0 ? payoff : 0);
    errors[path] = (premium + (gain - cost_rate * traded)) - positive_payoff;
}

extern "C" __global__ void hedging_gradient(
    const double* prices, const double* w1, const double* w2,
    const double* feature0, const double* hidden, const double* holdings, const double* errors,
    unsigned long long paths, unsigned long long steps, unsigned long long width,
    double cost_rate, double regularization,
    double* gradient_w1, double* gradient_b1, double* gradient_w2, double* gradient_b2,
    double* loss) {
    const auto parameter = static_cast<unsigned long long>(blockIdx.x);
    const unsigned tid = threadIdx.x;
    __shared__ double partial[128];
    double value = 0;
    if (parameter == 4 * width + 1) {
        for (auto path = static_cast<unsigned long long>(tid); path < paths; path += blockDim.x) {
            value += errors[path] * errors[path];
        }
    } else {
        for (auto point = static_cast<unsigned long long>(tid); point < paths * steps; point += blockDim.x) {
            const auto path = point / steps;
            const auto t = point % steps;
            const auto price_offset = path * (steps + 1) + t;
            const double holding = holdings[point];
            const double previous = t == 0 ? 0 : holdings[point - 1];
            const double next = t + 1 == steps ? 0 : holdings[point + 1];
            const double cost_gradient = cost_rate * (
                prices[price_offset] * hedging_sign(holding - previous)
                - prices[price_offset + 1] * hedging_sign(next - holding));
            const double gradient_h = (2 * errors[path]) * (
                (prices[price_offset + 1] - prices[price_offset]) - cost_gradient)
                / static_cast<double>(paths);
            const double gradient_z = (gradient_h * holding) * (1 - holding);
            if (parameter == 4 * width) {
                value += gradient_z;
            } else if (parameter >= 3 * width) {
                value += hidden[point * width + parameter - 3 * width] * gradient_z;
            } else {
                const auto j = parameter < 2 * width ? parameter % width : parameter - 2 * width;
                const double activation = hidden[point * width + j];
                const double gradient_hidden = (gradient_z * w2[j]) * (1 - activation * activation);
                const double feature = parameter < width ? feature0[point]
                    : (parameter < 2 * width ? static_cast<double>(steps - t) / steps : 1);
                value += feature * gradient_hidden;
            }
        }
    }
    partial[tid] = value;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2; offset; offset /= 2) {
        if (tid < offset) partial[tid] += partial[tid + offset];
        __syncthreads();
    }
    if (tid != 0) return;
    if (parameter < 2 * width) {
        gradient_w1[parameter] = partial[0] + (2 * regularization) * w1[parameter];
    } else if (parameter < 3 * width) {
        gradient_b1[parameter - 2 * width] = partial[0];
    } else if (parameter < 4 * width) {
        const auto j = parameter - 3 * width;
        gradient_w2[j] = partial[0] + (2 * regularization) * w2[j];
    } else if (parameter == 4 * width) {
        gradient_b2[0] = partial[0];
    } else {
        double sum_w1 = 0, sum_w2 = 0;
        for (unsigned long long j = 0; j < 2 * width; ++j) sum_w1 += w1[j] * w1[j];
        for (unsigned long long j = 0; j < width; ++j) sum_w2 += w2[j] * w2[j];
        loss[0] = (partial[0] / static_cast<double>(paths)
            + regularization * sum_w1) + regularization * sum_w2;
    }
}

// Return the array group and index of a packed parameter without copying arrays.
__device__ __forceinline__ unsigned hedging_group(unsigned long long& index, unsigned long long width) {
    if (index < 2 * width) return 0;
    index -= 2 * width;
    if (index < width) return 1;
    index -= width;
    if (index < width) return 2;
    index -= width;
    return 3;
}

extern "C" __global__ void hedging_adam(
    double* w1, double* b1, double* w2, double* b2,
    const double* gradient_w1, const double* gradient_b1, const double* gradient_w2, const double* gradient_b2,
    double* first_w1, double* first_b1, double* first_w2, double* first_b2,
    double* second_w1, double* second_b1, double* second_w2, double* second_b2,
    const double* loss, bool* finite, unsigned long long width,
    double learning_rate, double first_correction, double second_correction) {
    const unsigned tid = threadIdx.x;
    const double* gradients[4] = {gradient_w1, gradient_b1, gradient_w2, gradient_b2};
    double* parameters[4] = {w1, b1, w2, b2};
    double* first[4] = {first_w1, first_b1, first_w2, first_b2};
    double* second[4] = {second_w1, second_b1, second_w2, second_b2};
    __shared__ double partial[128];
    __shared__ unsigned valid[128];
    __shared__ double norm;
    double squares = 0;
    unsigned all_finite = 1;
    for (auto parameter = static_cast<unsigned long long>(tid); parameter < 4 * width + 1; parameter += blockDim.x) {
        auto index = parameter;
        const auto group = hedging_group(index, width);
        const double gradient = gradients[group][index];
        squares += gradient * gradient;
        all_finite &= isfinite(gradient);
    }
    partial[tid] = squares;
    valid[tid] = all_finite;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2; offset; offset /= 2) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
            valid[tid] &= valid[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        norm = sqrt(partial[0]);
        finite[0] = finite[0] && valid[0] && isfinite(loss[0]);
    }
    __syncthreads();
    // Infinite squared norm with finite gradients intentionally clips to zero.
    const double denominator = norm < 1 ? 1 : norm;
    for (auto parameter = static_cast<unsigned long long>(tid); parameter < 4 * width + 1; parameter += blockDim.x) {
        auto index = parameter;
        const auto group = hedging_group(index, width);
        const double gradient = gradients[group][index] / denominator;
        const double m = .9 * first[group][index] + .1 * gradient;
        const double v = .999 * second[group][index] + .001 * (gradient * gradient);
        first[group][index] = m;
        second[group][index] = v;
        parameters[group][index] -= (learning_rate * (m / first_correction))
            / (sqrt(v / second_correction) + 1e-8);
    }
}
