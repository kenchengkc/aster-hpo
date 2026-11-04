#include <cuda_runtime.h>

#include "accounting_kernel.cuh"
#include "aster/accounting.hpp"
#include "build_metadata.hpp"
#include "comparison.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t max_samples = 1000000;

void cuda_check(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
    }
}
#define CUDA_CHECK(expression) cuda_check((expression), #expression)

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::size_t checked_product(std::size_t left, std::size_t right) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::invalid_argument("workload size overflow");
    }
    return left * right;
}

std::size_t checked_sum(std::size_t left, std::size_t right) {
    if (left > std::numeric_limits<std::size_t>::max() - right) throw std::invalid_argument("workload size overflow");
    return left + right;
}

std::size_t number(std::string_view value) {
    if (value.empty()) throw std::invalid_argument("empty numeric argument");
    std::size_t result = 0;
    for (const char digit : value) {
        if (digit < '0' || digit > '9') throw std::invalid_argument("numeric arguments require unsigned decimal integers");
        const auto next = static_cast<std::size_t>(digit - '0');
        if (result > (std::numeric_limits<std::size_t>::max() - next) / 10) {
            throw std::invalid_argument("numeric argument overflow");
        }
        result = result * 10 + next;
    }
    return result;
}

struct Options {
    std::string mode = "smoke";
    std::string output;
    std::size_t paths = 65536, intervals = 12, block = 128, warmups = 5, repeats = 20;
    std::size_t operations = 8;
    std::uint64_t seed = 1729;
    int device = 0;
    bool help = false;
};

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help" || option == "-h") { options.help = true; continue; }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + option);
        const std::string value = argv[++i];
        if (option == "--mode") options.mode = value;
        else if (option == "--output") options.output = value;
        else if (option == "--paths") options.paths = number(value);
        else if (option == "--intervals") options.intervals = number(value);
        else if (option == "--block-size") options.block = number(value);
        else if (option == "--warmups") options.warmups = number(value);
        else if (option == "--repeats") options.repeats = number(value);
        else if (option == "--seed") options.seed = number(value);
        else if (option == "--operations") options.operations = number(value);
        else if (option == "--device") {
            const auto ordinal = number(value);
            if (ordinal > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::invalid_argument("device ordinal overflow");
            options.device = static_cast<int>(ordinal);
        } else throw std::invalid_argument("unknown option: " + option);
    }
    if (options.mode != "smoke" && options.mode != "correctness" && options.mode != "benchmark" && options.mode != "pipeline") throw std::invalid_argument("--mode must be smoke, correctness, benchmark, or pipeline");
    if (options.operations == 0 || options.operations > 256) throw std::invalid_argument("operations must be 1..256");
    if (options.paths == 0 || options.intervals == 0 || options.intervals > 256 || options.block == 0 || options.block > 1024 || options.repeats == 0 || options.repeats > max_samples || options.warmups > max_samples) {
        throw std::invalid_argument("paths/repeats must be positive; intervals 1..256, block-size 1..1024, warmups/repeats <=1000000");
    }
    checked_product(checked_product(options.paths, options.intervals + 1), sizeof(double));
    checked_product(checked_product(options.paths, options.intervals), sizeof(double));
    checked_product(options.paths, 3 * sizeof(double));
    if (options.mode == "pipeline") {
        const auto input_bytes = checked_product(checked_product(options.paths, 2 * options.intervals + 1), sizeof(double));
        const auto output_bytes = checked_product(checked_product(options.paths, options.operations), 3 * sizeof(double));
        checked_sum(input_bytes, output_bytes);
        checked_sum(input_bytes, checked_product(output_bytes, 5));
        checked_product(input_bytes, options.operations);
    }
    return options;
}

// Reserve the destination atomically before CUDA initialization. Failed runs leave an
// empty file rather than a success report; reruns require a different destination.
class Output {
public:
    explicit Output(const std::string& path) {
        if (!path.empty()) {
            fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
            if (fd_ < 0) {
                if (errno == EEXIST) throw std::runtime_error("output already exists: " + path);
                throw std::runtime_error("cannot create output: " + path + ": " + std::strerror(errno));
            }
        }
    }
    ~Output() { if (fd_ >= 0) ::close(fd_); }
    Output(const Output&) = delete;
    Output& operator=(const Output&) = delete;
    void write(const std::string& data) {
        if (fd_ < 0) { std::cout << data; if (!std::cout) throw std::runtime_error("stdout write failed"); return; }
        std::size_t position = 0;
        while (position < data.size()) {
            const auto written = ::write(fd_, data.data() + position, data.size() - position);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) throw std::runtime_error("output write failed");
            position += static_cast<std::size_t>(written);
        }
        if (::fsync(fd_) != 0) throw std::runtime_error("output sync failed");
    }
private:
    int fd_ = -1;
};

struct Data {
    std::string name;
    std::size_t paths, intervals;
    double rate;
    std::vector<double> prices, holdings;
};
std::size_t output_count(std::size_t paths, std::size_t operations) {
    const auto count = checked_product(paths, operations);
    checked_product(count, 3 * sizeof(double));
    return count;
}
struct ResultView { std::span<double> pnl, costs, turnover; };
struct Results {
    std::vector<double> pnl, costs, turnover;
    explicit Results(std::size_t paths, std::size_t operations = 1)
        : pnl(output_count(paths, operations)), costs(pnl.size()), turnover(pnl.size()) {}
    ResultView view() { return {pnl, costs, turnover}; }
    ResultView slot(std::size_t paths, std::size_t operation) {
        const auto offset = checked_product(paths, operation);
        if (offset > pnl.size() || paths > pnl.size() - offset) throw std::invalid_argument("host output slot out of range");
        return {std::span<double>(pnl).subspan(offset, paths),
                std::span<double>(costs).subspan(offset, paths),
                std::span<double>(turnover).subspan(offset, paths)};
    }
};

Data random_data(std::string name, std::size_t paths, std::size_t intervals,
                 double rate, std::uint64_t seed) {
    Data data{std::move(name), paths, intervals, rate,
              std::vector<double>(checked_product(paths, intervals + 1)),
              std::vector<double>(checked_product(paths, intervals))};
    std::mt19937_64 engine(seed);
    const auto unit = [&engine] { return static_cast<double>(engine() >> 11) * 0x1.0p-53; };
    for (auto& price : data.prices) price = 0.5 + 1.5 * unit();
    for (auto& holding : data.holdings) holding = -1.0 + 2.0 * unit();
    return data;
}

Results cpu_reference(const Data& data) {
    Results result(data.paths);
    aster::hedge_pnl(data.prices, data.holdings, data.paths, data.intervals,
                     data.rate, result.pnl, result.costs, result.turnover);
    return result;
}

struct Sample { double kernel_event_ms = 0, completed_host_ms = 0, native_cpu_prepared_ms = 0; };

struct Transfers {
    std::size_t h2d_bytes = 0, d2h_bytes = 0, h2d_calls = 0, d2h_calls = 0;
};

// Inputs are explicitly uploaded. Each upload invalidates every retained output,
// including same-shape reloads. Output arrays use compact [operation][path] order.
class DeviceBuffers {
public:
    DeviceBuffers(std::size_t price_count, std::size_t holding_count, std::size_t paths,
                  std::size_t block, const cudaDeviceProp& device, std::size_t slots = 1)
        : block_(block) {
        if (paths == 0 || block == 0 || slots == 0 || slots > 256) throw std::invalid_argument("invalid device buffer dimensions");
        if (block > static_cast<std::size_t>(device.maxThreadsPerBlock) || block > static_cast<std::size_t>(device.maxThreadsDim[0])) {
            throw std::invalid_argument("block size exceeds device launch limits");
        }
        const auto blocks = (paths - 1) / block + 1;
        if (blocks > static_cast<std::size_t>(device.maxGridSize[0])) throw std::invalid_argument("grid size exceeds device launch limits");
        price_capacity_ = price_count; holding_capacity_ = holding_count; path_capacity_ = paths;
        const auto price_bytes = checked_product(price_count, sizeof(double));
        const auto holding_bytes = checked_product(holding_count, sizeof(double));
        const auto count = output_count(paths, slots);
        const auto output_bytes = checked_product(count, 3 * sizeof(double));
        if (price_bytes > std::numeric_limits<std::size_t>::max() - holding_bytes ||
            price_bytes + holding_bytes > std::numeric_limits<std::size_t>::max() - output_bytes) {
            throw std::invalid_argument("aggregate device allocation overflow");
        }
        slot_generation_.assign(slots, 0);
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        if (price_bytes + holding_bytes + output_bytes > free) throw std::runtime_error("workload exceeds available device memory");
        try {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreate(&begin_));
            CUDA_CHECK(cudaEventCreate(&end_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&prices_), price_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&holdings_), holding_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pnl_), count * sizeof(double)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&costs_), count * sizeof(double)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&turnover_), count * sizeof(double)));
        } catch (...) { release(); throw; }
    }
    ~DeviceBuffers() { release(); }
    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;

    void upload(const Data& data) {
        if (!stream_) throw std::logic_error("device buffers are closed");
        if (data.intervals == 0 || data.intervals == std::numeric_limits<std::size_t>::max() ||
            data.paths > path_capacity_ || data.prices.size() != checked_product(data.paths, data.intervals + 1) ||
            data.holdings.size() != checked_product(data.paths, data.intervals) ||
            data.prices.size() > price_capacity_ || data.holdings.size() > holding_capacity_) {
            throw std::invalid_argument("device input shape/capacity mismatch");
        }
        // Invalidate before issuing copies, so a partial upload cannot expose old outputs.
        loaded_ = false;
        if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
            std::fill(slot_generation_.begin(), slot_generation_.end(), 0);
            generation_ = 0;
        }
        ++generation_;
        active_paths_ = data.paths; active_intervals_ = data.intervals;
        if (data.paths) {
            copy(prices_, data.prices.data(), checked_product(data.prices.size(), sizeof(double)), cudaMemcpyHostToDevice);
            copy(holdings_, data.holdings.data(), checked_product(data.holdings.size(), sizeof(double)), cudaMemcpyHostToDevice);
        }
        loaded_ = true;
    }
    void launch(double rate, std::size_t slot) {
        require_loaded();
        if (slot >= slot_generation_.size()) throw std::invalid_argument("device output slot out of range");
        validate_rate(rate);
        if (active_paths_) {
            const auto blocks = (active_paths_ - 1) / block_ + 1;
            const auto offset = checked_product(active_paths_, slot);
            accounting_kernel<<<static_cast<unsigned>(blocks), static_cast<unsigned>(block_), 0, stream_>>>(
                prices_, holdings_, active_paths_, active_intervals_, rate, pnl_ + offset, costs_ + offset, turnover_ + offset);
            CUDA_CHECK(cudaGetLastError());
        }
        slot_generation_[slot] = generation_;
    }
    void readback(ResultView result, std::size_t first_slot, std::size_t slots) {
        require_loaded();
        if (slots == 0 || first_slot >= slot_generation_.size() || slots > slot_generation_.size() - first_slot) {
            throw std::invalid_argument("device output slot out of range");
        }
        const auto count = checked_product(active_paths_, slots);
        if (result.pnl.size() != count || result.costs.size() != count || result.turnover.size() != count) {
            throw std::invalid_argument("device output shape mismatch");
        }
        for (std::size_t slot = first_slot; slot < first_slot + slots; ++slot) {
            if (slot_generation_[slot] != generation_) throw std::logic_error("resident output is incomplete or stale");
        }
        const auto offset = checked_product(active_paths_, first_slot);
        const auto bytes = checked_product(count, sizeof(double));
        if (count) {
            copy(result.pnl.data(), pnl_ + offset, bytes, cudaMemcpyDeviceToHost);
            copy(result.costs.data(), costs_ + offset, bytes, cudaMemcpyDeviceToHost);
            copy(result.turnover.data(), turnover_ + offset, bytes, cudaMemcpyDeviceToHost);
        }
        // Each readback consumes its slots; a later read requires another launch.
        for (std::size_t slot = first_slot; slot < first_slot + slots; ++slot) slot_generation_[slot] = 0;
    }
    void synchronize() {
        require_loaded();
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    void reset_transfers() { transfers_ = {}; }
    Transfers transfers() const { return transfers_; }
    void copy_per_call(const Data& data, double rate, ResultView result) {
        validate_result(data.paths, result);
        validate_rate(rate);
        upload(data);
        launch(rate, 0);
        readback(result, 0, 1);
        synchronize();
    }
    Sample execute(const Data& data, Results& result) {
        validate_result(data.paths, result.view());
        validate_rate(data.rate);
        if (data.paths == 0) { upload(data); return {}; }
        const auto start = Clock::now();
        upload(data);
        CUDA_CHECK(cudaEventRecord(begin_, stream_));
        launch(data.rate, 0);
        CUDA_CHECK(cudaEventRecord(end_, stream_));
        readback(result.view(), 0, 1);
        synchronize();
        const auto finish = Clock::now();
        float elapsed = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin_, end_));
        return {static_cast<double>(elapsed), milliseconds(start, finish), 0};
    }
    void close() {
        loaded_ = false;
        CUDA_CHECK(cudaFree(prices_)); prices_ = nullptr;
        CUDA_CHECK(cudaFree(holdings_)); holdings_ = nullptr;
        CUDA_CHECK(cudaFree(pnl_)); pnl_ = nullptr;
        CUDA_CHECK(cudaFree(costs_)); costs_ = nullptr;
        CUDA_CHECK(cudaFree(turnover_)); turnover_ = nullptr;
        CUDA_CHECK(cudaEventDestroy(begin_)); begin_ = nullptr;
        CUDA_CHECK(cudaEventDestroy(end_)); end_ = nullptr;
        CUDA_CHECK(cudaStreamDestroy(stream_)); stream_ = nullptr;
    }
private:
    static void validate_rate(double rate) {
        if (!std::isfinite(rate) || rate < 0 || rate > 0.01) throw std::invalid_argument("resident rate must be finite and in [0,0.01]");
    }
    static void validate_result(std::size_t paths, ResultView result) {
        if (result.pnl.size() != paths || result.costs.size() != paths || result.turnover.size() != paths) {
            throw std::invalid_argument("device output shape mismatch");
        }
    }
    void require_loaded() const {
        if (!stream_ || !loaded_) throw std::logic_error("resident inputs are not loaded");
    }
    void copy(void* destination, const void* source, std::size_t bytes, cudaMemcpyKind kind) {
        CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, kind, stream_));
        if (kind == cudaMemcpyHostToDevice) { transfers_.h2d_bytes += bytes; ++transfers_.h2d_calls; }
        else { transfers_.d2h_bytes += bytes; ++transfers_.d2h_calls; }
    }
    void release() noexcept {
        if (prices_) cudaFree(prices_);
        if (holdings_) cudaFree(holdings_);
        if (pnl_) cudaFree(pnl_);
        if (costs_) cudaFree(costs_);
        if (turnover_) cudaFree(turnover_);
        if (begin_) cudaEventDestroy(begin_);
        if (end_) cudaEventDestroy(end_);
        if (stream_) cudaStreamDestroy(stream_);
        prices_ = holdings_ = pnl_ = costs_ = turnover_ = nullptr;
        begin_ = end_ = nullptr; stream_ = nullptr; loaded_ = false;
    }
    double *prices_ = nullptr, *holdings_ = nullptr, *pnl_ = nullptr, *costs_ = nullptr, *turnover_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t begin_ = nullptr, end_ = nullptr;
    std::size_t price_capacity_ = 0, holding_capacity_ = 0, path_capacity_ = 0, block_;
    std::size_t active_paths_ = 0, active_intervals_ = 0;
    std::vector<std::uint64_t> slot_generation_;
    std::uint64_t generation_ = 0;
    bool loaded_ = false;
    Transfers transfers_;
};

void compare_all(const Results& expected, const Results& actual, const std::string& name,
                 aster_cuda::Comparison& comparison) {
    aster_cuda::compare(expected.pnl, actual.pnl, name + "/pnl", comparison);
    aster_cuda::compare(expected.costs, actual.costs, name + "/costs", comparison);
    aster_cuda::compare(expected.turnover, actual.turnover, name + "/turnover", comparison);
}

std::vector<Data> cases(const Options& options) {
    std::vector<Data> fixtures;
    fixtures.push_back({"hand_checked", 1, 2, 0.01, {1.0, 1.1, 1.2}, {0.5, 0.5}});
    fixtures.push_back({"short_single_interval", 1, 1, 0.01, {1.0, 1.25}, {-1.0}});
    fixtures.push_back({"empty_paths", 0, 1, 0.01, {}, {}});
    fixtures.push_back({"zero_holdings", 3, 1, 0.01, {1.0, 2.0, 0.5, 1.0, 2.0, 0.5}, {0.0, 0.0, 0.0}});
    if (options.mode == "smoke") return fixtures;
    for (const auto paths : std::array<std::size_t, 9>{1, 2, 31, 32, 33, 127, 128, 129, 257}) {
        for (const auto intervals : std::array<std::size_t, 5>{1, 2, 12, 63, 256}) {
            for (const double rate : {0.0, 0.01}) fixtures.push_back(random_data("random_" + std::to_string(paths) + "x" + std::to_string(intervals) + "_" + std::to_string(rate), paths, intervals, rate, options.seed + paths * 257 + intervals));
        }
    }
    // Always include a partial block, including when a caller selects a nonstandard block size.
    fixtures.push_back(random_data("selected_block_partial", options.block + 1, 13, 0.007, options.seed));
    for (const double epsilon : {0.0, 1e-13}) {
        auto data = random_data(epsilon == 0 ? "exact_cancelling_gain" : "near_cancelling_gain", 5, 256, 0, options.seed);
        std::fill(data.holdings.begin(), data.holdings.end(), 0.75);
        for (std::size_t p = 0; p < data.paths; ++p) {
            for (std::size_t t = 0; t <= data.intervals; ++t) data.prices[p * (data.intervals + 1) + t] = t % 2 ? 1.5 : 1.0;
            data.prices[p * (data.intervals + 1) + data.intervals] += epsilon;
        }
        fixtures.push_back(std::move(data));
    }
    // Nonzero gains/costs nearly cancel in final P&L at the lower price bound.
    fixtures.push_back({"gain_minus_cost_cancellation", 1, 1, 0.01,
                        {0.5, 0.5 * 1.01 / 0.99}, {0.75}});
    return fixtures;
}

std::string escaped(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c) << std::dec;
        else out << c;
    }
    out << '"';
    return out.str();
}

std::string hash_input(const Data& data) {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto add = [&hash](const void* address, std::size_t bytes) {
        const auto* input = static_cast<const unsigned char*>(address);
        for (std::size_t i = 0; i < bytes; ++i) { hash ^= input[i]; hash *= 1099511628211ULL; }
    };
    add(data.prices.data(), data.prices.size() * sizeof(double));
    add(data.holdings.data(), data.holdings.size() * sizeof(double));
    add(&data.rate, sizeof(data.rate));
    std::ostringstream out; out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

std::array<double, 3> checksum(const Results& result) {
    std::array<double, 3> value{};
    for (const double x : result.pnl) value[0] += x;
    for (const double x : result.costs) value[1] += x;
    for (const double x : result.turnover) value[2] += x;
    for (const double x : value) if (!std::isfinite(x)) throw std::runtime_error("nonfinite checksum");
    return value;
}

std::vector<double> sweep_rates(std::size_t operations) {
    std::vector<double> rates(operations);
    for (std::size_t i = 0; i < operations; ++i) {
        rates[i] = (operations == 1 || i + 1 == operations) ? 0.01
            : 0.01 * static_cast<double>(i) / static_cast<double>(operations - 1);
    }
    return rates;
}

void cpu_sequence(const Data& data, const std::vector<double>& rates, Results& result, bool checked) {
    for (std::size_t operation = 0; operation < rates.size(); ++operation) {
        const auto slot = result.slot(data.paths, operation);
        if (checked) aster::hedge_pnl(data.prices, data.holdings, data.paths, data.intervals,
                                    rates[operation], slot.pnl, slot.costs, slot.turnover);
        else aster::hedge_pnl_unchecked(data.prices, data.holdings, data.paths, data.intervals,
                                       rates[operation], slot.pnl, slot.costs, slot.turnover);
    }
}

void resident_sequence(DeviceBuffers& buffers, const Data& data, const std::vector<double>& rates,
                       Results& result, bool upload) {
    if (upload) buffers.upload(data);
    for (std::size_t operation = 0; operation < rates.size(); ++operation) buffers.launch(rates[operation], operation);
    buffers.readback(result.view(), 0, rates.size());
    buffers.synchronize();
}

void copy_sequence(DeviceBuffers& buffers, const Data& data, const std::vector<double>& rates, Results& result) {
    for (std::size_t operation = 0; operation < rates.size(); ++operation) {
        buffers.copy_per_call(data, rates[operation], result.slot(data.paths, operation));
    }
}

template<class Function>
void expect_rejected(Function operation, const std::string& message) {
    try { operation(); }
    catch (const std::exception& error) {
        if (std::string(error.what()).find(message) != std::string::npos) return;
        throw std::runtime_error("wrong lifecycle rejection: " + std::string(error.what()));
    }
    throw std::runtime_error("missing lifecycle rejection: " + message);
}

void resident_correctness(const Options& options, const cudaDeviceProp& device,
                          const std::vector<Data>& fixtures, aster_cuda::Comparison& comparison,
                          std::vector<std::string>& checked_cases) {
    std::size_t prices = 1, holdings = 1, paths = 1;
    for (const auto& data : fixtures) {
        prices = std::max(prices, data.prices.size()); holdings = std::max(holdings, data.holdings.size()); paths = std::max(paths, data.paths);
    }
    DeviceBuffers buffers(prices, holdings, paths, options.block, device, 3);
    const auto rates = sweep_rates(3);
    const auto one_rate = sweep_rates(1);
    Results one(1), three(1, 3), wrong_shape(2);
    expect_rejected([&] { buffers.launch(0.01, 0); }, "inputs are not loaded");
    expect_rejected([&] { buffers.readback(one.view(), 0, 1); }, "inputs are not loaded");
    const auto& hand = fixtures.front();
    buffers.upload(hand);
    expect_rejected([&] { buffers.launch(0.01, 3); }, "slot out of range");
    for (const double rate : {-0.01, 0.02, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        expect_rejected([&] { buffers.launch(rate, 0); }, "rate must be finite");
    }
    expect_rejected([&] { buffers.readback(one.view(), 3, 1); }, "slot out of range");
    expect_rejected([&] { buffers.readback(wrong_shape.view(), 0, 1); }, "output shape mismatch");
    auto malformed = hand;
    malformed.holdings.pop_back();
    expect_rejected([&] { buffers.upload(malformed); }, "input shape/capacity mismatch");
    expect_rejected([&] { buffers.readback(one.view(), 0, 1); }, "incomplete or stale");
    buffers.launch(rates[0], 0);
    expect_rejected([&] { buffers.readback(three.view(), 0, 3); }, "incomplete or stale");
    buffers.readback(one.view(), 0, 1); buffers.synchronize();
    expect_rejected([&] { buffers.readback(one.view(), 0, 1); }, "incomplete or stale");
    checked_cases.push_back("resident_lifecycle_invalid_launch_rate_slot_incomplete_and_consumed_outputs");

    for (const auto& data : fixtures) {
        if (data.name != "hand_checked" && data.name != "short_single_interval" &&
            data.name != "empty_paths" && data.name != "selected_block_partial" &&
            data.name != "exact_cancelling_gain" && data.name != "near_cancelling_gain" &&
            data.name != "gain_minus_cost_cancellation" && data.name != "random_257x256_0.010000") continue;
        for (const auto* selected : {&one_rate, &rates}) {
            Results reference(data.paths, selected->size()), copied(data.paths, selected->size()), resident(data.paths, selected->size());
            cpu_sequence(data, *selected, reference, true);
            copy_sequence(buffers, data, *selected, copied);
            resident_sequence(buffers, data, *selected, resident, true);
            compare_all(reference, copied, data.name + "/copy_sequence", comparison);
            compare_all(reference, resident, data.name + "/resident_sequence", comparison);
            if (data.name == "hand_checked") {
                Results literal(1, selected->size());
                for (std::size_t i = 0; i < selected->size(); ++i) {
                    literal.pnl[i] = 0.1 - (*selected)[i] * 1.1;
                    literal.costs[i] = (*selected)[i] * 1.1;
                    literal.turnover[i] = 1.1;
                }
                compare_all(literal, copied, "resident_sweep_literal_copy", comparison);
                compare_all(literal, resident, "resident_sweep_literal_resident", comparison);
            }
            checked_cases.push_back("resident_" + data.name + "_operations_" + std::to_string(selected->size()));
        }
    }

    // A changed shape and a same-shape changed input must each invalidate all slots.
    for (const bool same_shape : {false, true}) {
        Data initial = hand;
        Results previous(1, 3);
        resident_sequence(buffers, initial, rates, previous, true);
        for (std::size_t i = 0; i < rates.size(); ++i) buffers.launch(rates[i], i);
        Data changed = same_shape ? initial : fixtures[3];
        changed.prices.back() += 0.125;
        changed.holdings.front() = -0.25;
        buffers.upload(changed);
        Results actual(changed.paths, 3), copied(changed.paths, 3), reference(changed.paths, 3);
        expect_rejected([&] { buffers.readback(actual.view(), 0, 3); }, "incomplete or stale");
        resident_sequence(buffers, changed, rates, actual, false);
        copy_sequence(buffers, changed, rates, copied);
        cpu_sequence(changed, rates, reference, true);
        compare_all(reference, actual, "resident_reloaded_inputs", comparison);
        compare_all(reference, copied, "copy_reloaded_inputs", comparison);
        if (actual.pnl == previous.pnl) throw std::runtime_error("reload regression did not change outputs");
        checked_cases.push_back(same_shape ? "resident_same_shape_changed_input_reload" : "resident_changed_shape_reload");
    }
    buffers.close();
}

enum class Method : std::size_t { copy_per_call, resident_session, resident_preloaded, native_cpu_sequence };
constexpr std::array<const char*, 4> method_names{"copy_per_call", "resident_session", "resident_preloaded", "native_cpu_sequence"};
struct PipelineSample {
    std::array<Method, 4> order{};
    std::array<double, 4> elapsed{};
    std::array<Transfers, 3> transfers{};
};

void write_transfers(std::ostream& json, const Transfers& transfers) {
    json << "{\"h2d_bytes\":" << transfers.h2d_bytes << ",\"d2h_bytes\":" << transfers.d2h_bytes
         << ",\"h2d_calls\":" << transfers.h2d_calls << ",\"d2h_calls\":" << transfers.d2h_calls << '}';
}

void validate_transfers(const Transfers& actual, const Data& data, std::size_t operations, Method method) {
    const auto input_bytes = checked_product(checked_sum(data.prices.size(), data.holdings.size()), sizeof(double));
    const auto output_bytes = checked_product(checked_product(data.paths, operations), 3 * sizeof(double));
    const auto batches = method == Method::copy_per_call ? operations : method == Method::resident_session ? 1 : 0;
    if (actual.h2d_bytes != checked_product(input_bytes, batches) || actual.d2h_bytes != output_bytes ||
        actual.h2d_calls != 2 * batches || actual.d2h_calls != (method == Method::copy_per_call ? 3 * operations : 3)) {
        throw std::runtime_error("pipeline transfer counter mismatch");
    }
}

PipelineSample pipeline_iteration(DeviceBuffers& buffers, const Data& data, const std::vector<double>& rates,
                                  std::array<Results, 4>& results, std::mt19937_64& order_engine) {
    PipelineSample sample;
    sample.order = {Method::copy_per_call, Method::resident_session, Method::resident_preloaded, Method::native_cpu_sequence};
    std::shuffle(sample.order.begin(), sample.order.end(), order_engine);
    for (const auto method : sample.order) {
        const auto index = static_cast<std::size_t>(method);
        if (method == Method::resident_preloaded) { buffers.upload(data); buffers.synchronize(); }
        buffers.reset_transfers();
        const auto start = Clock::now();
        switch (method) {
            case Method::copy_per_call: copy_sequence(buffers, data, rates, results[index]); break;
            case Method::resident_session: resident_sequence(buffers, data, rates, results[index], true); break;
            case Method::resident_preloaded: resident_sequence(buffers, data, rates, results[index], false); break;
            case Method::native_cpu_sequence: cpu_sequence(data, rates, results[index], false); break;
        }
        sample.elapsed[index] = milliseconds(start, Clock::now());
        if (!std::isfinite(sample.elapsed[index]) || sample.elapsed[index] < 0 || (index < 3 && sample.elapsed[index] == 0)) {
            throw std::runtime_error("invalid completed pipeline timing sample");
        }
        if (index < 3) {
            sample.transfers[index] = buffers.transfers();
            validate_transfers(sample.transfers[index], data, rates.size(), method);
        }
    }
    return sample;
}

std::string hash_results(const Results& results) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto* values : {&results.pnl, &results.costs, &results.turnover}) {
        for (const double value : *values) {
            const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
            for (std::size_t i = 0; i < sizeof(double); ++i) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
        }
    }
    std::ostringstream out; out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

int run_pipeline(const Options& options, Output& output, const cudaDeviceProp& device,
                 int driver, int runtime, double runtime_ms) {
    const auto preparation_start = Clock::now();
    const auto data = random_data("pipeline", options.paths, options.intervals, 0.01, options.seed);
    const auto rates = sweep_rates(options.operations);
    Results reference(data.paths, rates.size());
    std::array<Results, 4> results{Results(data.paths, rates.size()), Results(data.paths, rates.size()),
                                  Results(data.paths, rates.size()), Results(data.paths, rates.size())};
    std::vector<PipelineSample> samples(options.repeats);
    cpu_sequence(data, rates, reference, true);
    const double preparation_ms = milliseconds(preparation_start, Clock::now());
    const auto allocation_start = Clock::now();
    DeviceBuffers buffers(data.prices.size(), data.holdings.size(), data.paths, options.block, device, rates.size());
    const double allocation_ms = milliseconds(allocation_start, Clock::now());
    const auto first_start = Clock::now();
    buffers.copy_per_call(data, 0.01, results[0].slot(data.paths, 0));
    const double first_host_ms = milliseconds(first_start, Clock::now());
    const std::uint64_t order_seed = options.seed ^ 0x9e3779b97f4a7c15ULL;
    std::mt19937_64 order_engine(order_seed);
    for (std::size_t i = 0; i < options.warmups; ++i) pipeline_iteration(buffers, data, rates, results, order_engine);
    aster_cuda::Comparison comparison;
    for (auto& sample : samples) {
        sample = pipeline_iteration(buffers, data, rates, results, order_engine);
        for (std::size_t i = 0; i < results.size(); ++i) compare_all(reference, results[i], method_names[i], comparison);
    }
    buffers.close();
    std::ostringstream json; json << std::setprecision(17);
    json << "{\n\"schema_version\":2,\"passed\":true,\"mode\":\"pipeline\",\"gpu_owning_processes_in_runner\":1,\n"
         << "\"device\":{\"ordinal\":" << options.device << ",\"name\":" << escaped(device.name)
         << ",\"compute_capability\":" << escaped(std::to_string(device.major) + "." + std::to_string(device.minor))
         << ",\"total_memory_bytes\":" << device.totalGlobalMem << ",\"driver_version\":" << driver << ",\"runtime_version\":" << runtime << "},\n"
         << "\"build\":{\"cuda_compiler\":" << escaped(build_cuda_compiler) << ",\"host_compiler\":" << escaped(build_host_compiler)
         << ",\"cuda_architectures\":" << escaped(build_cuda_architectures) << ",\"cuda_flags\":" << escaped(build_cuda_flags)
         << ",\"cuda_user_flags\":" << escaped(build_cuda_user_flags) << ",\"host_user_flags\":" << escaped(build_host_user_flags)
         << ",\"build_type\":" << escaped(build_type) << ",\"float_type\":\"float64\",\"layout\":\"path_major\",\"output_layout\":\"operation_major_path_minor\"},\n"
         << "\"cold\":{\"runtime_setup_ms\":" << runtime_ms << ",\"host_preparation_ms\":" << preparation_ms
         << ",\"device_allocation_ms\":" << allocation_ms << ",\"first_completed_host_ms\":" << first_host_ms << "},\n"
         << "\"comparison\":{\"passed\":true,\"all_operations_checked\":true,\"atol\":" << aster_cuda::atol << ",\"rtol\":" << aster_cuda::rtol
         << ",\"outputs\":[\"pnl\",\"costs\",\"turnover\"],\"values_checked\":" << comparison.values_checked
         << ",\"max_absolute_error\":" << comparison.max_absolute_error << ",\"max_normalized_error\":" << comparison.max_normalized_error << "},\n"
         << "\"workload\":{\"paths\":" << options.paths << ",\"intervals\":" << options.intervals << ",\"operations\":" << rates.size()
         << ",\"block_size\":" << options.block << ",\"seed\":" << options.seed << ",\"warmups\":" << options.warmups << ",\"repeats\":" << samples.size()
         << ",\"input_hash_fnv1a64\":" << escaped(hash_input(data)) << ",\"method_order_seed\":" << order_seed
         << ",\"output_values_per_method\":" << output_count(data.paths, rates.size()) * 3 << ",\"rates\":[";
    for (std::size_t i = 0; i < rates.size(); ++i) { if (i) json << ','; json << rates[i]; }
    json << "]},\n\"timing_policy\":{"
         << "\"copy_per_call_ms\":\"K calls, each H2D inputs + unchanged kernel + D2H three per-path arrays + stream synchronization\","
         << "\"resident_session_ms\":\"one H2D input batch + K unchanged kernels in retained slots + three bulk D2H copies of all K operations + stream synchronization\","
         << "\"resident_preloaded_ms\":\"input upload and its synchronization excluded; K unchanged kernels + all output D2H copies + stream synchronization\","
         << "\"native_cpu_sequence_ms\":\"K prepared unchecked native calls writing all outputs, one host thread; no cached sensitivity algorithm\","
         << "\"capacity\":\"all host/device capacity allocated before cold first call, warmups, and measured sessions; pageable host vectors\","
         << "\"transfers\":\"successful requested cudaMemcpyAsync bytes and calls, not physical PCIe traffic; preloaded upload excluded\","
         << "\"scope\":\"standalone accounting sensitivity sweep; excludes binding costs, process startup, full training, and complete HPO runtime\","
         << "\"method_order\":\"seeded shuffle per warmup and repeat, recorded per measured sample\","
         << "\"checksum_order\":\"pnl[operation][path], then costs[operation][path], then turnover[operation][path], host-native double bytes\","
         << "\"quantiles\":\"none; raw completed samples retained\"},\n\"checksum\":{";
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i) json << ',';
        const auto sums = checksum(results[i]);
        json << escaped(method_names[i]) << ":{\"pnl\":" << sums[0] << ",\"costs\":" << sums[1] << ",\"turnover\":" << sums[2]
             << ",\"ordered_hash_fnv1a64\":" << escaped(hash_results(results[i])) << '}';
    }
    json << "},\n\"pipeline_samples\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i) json << ',';
        const auto& sample = samples[i];
        json << "{\"method_order\":[";
        for (std::size_t method = 0; method < sample.order.size(); ++method) { if (method) json << ','; json << escaped(method_names[static_cast<std::size_t>(sample.order[method])]); }
        json << ']';
        for (std::size_t method = 0; method < method_names.size(); ++method) json << ',' << escaped(std::string(method_names[method]) + "_ms") << ':' << sample.elapsed[method];
        json << ",\"transfers\":{";
        for (std::size_t method = 0; method < 3; ++method) { if (method) json << ','; json << escaped(method_names[method]) << ':'; write_transfers(json, sample.transfers[method]); }
        json << "}}";
    }
    json << "]\n}\n";
    output.write(json.str());
    return 0;
}

int run(const Options& options, Output& output) {
    int device_count = 0, driver = 0, runtime = 0;
    cudaDeviceProp device{};
    const auto runtime_start = Clock::now();
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (options.device >= device_count) throw std::invalid_argument("requested device ordinal is unavailable");
    CUDA_CHECK(cudaSetDevice(options.device));
    CUDA_CHECK(cudaFree(nullptr));
    CUDA_CHECK(cudaGetDeviceProperties(&device, options.device));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    const double runtime_ms = milliseconds(runtime_start, Clock::now());
    if (options.mode == "pipeline") return run_pipeline(options, output, device, driver, runtime, runtime_ms);
    auto fixtures = cases(options);
    std::size_t prices = 1, holdings = 1, paths = 1;
    for (const auto& fixture : fixtures) { prices = std::max(prices, fixture.prices.size()); holdings = std::max(holdings, fixture.holdings.size()); paths = std::max(paths, fixture.paths); }
    if (options.mode == "benchmark") { prices = std::max(prices, checked_product(options.paths, options.intervals + 1)); holdings = std::max(holdings, checked_product(options.paths, options.intervals)); paths = std::max(paths, options.paths); }
    const auto allocation_start = Clock::now();
    DeviceBuffers buffers(prices, holdings, paths, options.block, device);
    const double allocation_ms = milliseconds(allocation_start, Clock::now());
    Data benchmark{};
    if (options.mode == "benchmark") benchmark = random_data("benchmark", options.paths, options.intervals, 0.01, options.seed);
    aster_cuda::Comparison comparison;
    std::vector<std::string> checked_cases;
    double first_host_ms = 0;
    for (const auto& fixture : fixtures) {
        const auto reference = cpu_reference(fixture);
        Results actual(fixture.paths);
        const auto sample = buffers.execute(fixture, actual);
        if (first_host_ms == 0 && fixture.paths != 0) first_host_ms = sample.completed_host_ms;
        compare_all(reference, actual, fixture.name, comparison);
        if (fixture.name == "hand_checked") {
            Results literal(1); literal.pnl[0] = 0.089; literal.costs[0] = 0.011; literal.turnover[0] = 1.1;
            compare_all(literal, actual, "hand_checked_literal", comparison);
        }
        if (fixture.name == "short_single_interval") {
            Results literal(1); literal.pnl[0] = -0.2725; literal.costs[0] = 0.0225; literal.turnover[0] = 2.25;
            compare_all(literal, actual, "short_literal", comparison);
        }
        checked_cases.push_back(fixture.name);
    }
    if (options.mode == "correctness") resident_correctness(options, device, fixtures, comparison, checked_cases);
    std::vector<Sample> samples;
    std::array<double, 3> sums{};
    if (options.mode == "benchmark") {
        const auto reference = cpu_reference(benchmark);
        Results actual(benchmark.paths), cpu(benchmark.paths);
        for (std::size_t i = 0; i < options.warmups; ++i) {
            aster::hedge_pnl_unchecked(benchmark.prices, benchmark.holdings, benchmark.paths, benchmark.intervals, benchmark.rate, cpu.pnl, cpu.costs, cpu.turnover);
            buffers.execute(benchmark, actual);
        }
        samples.reserve(options.repeats);
        for (std::size_t i = 0; i < options.repeats; ++i) {
            const auto start = Clock::now();
            aster::hedge_pnl_unchecked(benchmark.prices, benchmark.holdings, benchmark.paths, benchmark.intervals, benchmark.rate, cpu.pnl, cpu.costs, cpu.turnover);
            const double cpu_ms = milliseconds(start, Clock::now());
            auto sample = buffers.execute(benchmark, actual);
            sample.native_cpu_prepared_ms = cpu_ms;
            compare_all(reference, actual, "benchmark_gpu", comparison);
            compare_all(reference, cpu, "benchmark_cpu_prepared", comparison);
            if (!(sample.kernel_event_ms > 0) || !(sample.completed_host_ms > 0) || !(cpu_ms > 0)) throw std::runtime_error("nonpositive completed timing sample");
            samples.push_back(sample);
        }
        sums = checksum(actual);
        checked_cases.push_back("benchmark_identical_inputs_every_repeat");
    }
    buffers.close();
    std::ostringstream json; json << std::setprecision(17);
    json << "{\n\"schema_version\":1,\"passed\":true,\"mode\":" << escaped(options.mode)
         << ",\"gpu_owning_processes_in_runner\":1,\n\"device\":{\"ordinal\":" << options.device
         << ",\"name\":" << escaped(device.name) << ",\"compute_capability\":" << escaped(std::to_string(device.major) + "." + std::to_string(device.minor))
         << ",\"total_memory_bytes\":" << device.totalGlobalMem << ",\"driver_version\":" << driver << ",\"runtime_version\":" << runtime << "},\n"
         << "\"build\":{\"cuda_compiler\":" << escaped(build_cuda_compiler) << ",\"host_compiler\":" << escaped(build_host_compiler)
         << ",\"cuda_architectures\":" << escaped(build_cuda_architectures) << ",\"cuda_flags\":" << escaped(build_cuda_flags)
         << ",\"cuda_user_flags\":" << escaped(build_cuda_user_flags) << ",\"host_user_flags\":" << escaped(build_host_user_flags)
         << ",\"build_type\":" << escaped(build_type) << ",\"float_type\":\"float64\",\"layout\":\"path_major\"},\n"
         << "\"cold\":{\"runtime_setup_ms\":" << runtime_ms << ",\"device_allocation_ms\":" << allocation_ms << ",\"first_completed_host_ms\":" << first_host_ms << "},\n"
         << "\"comparison\":{\"atol\":" << aster_cuda::atol << ",\"rtol\":" << aster_cuda::rtol << ",\"outputs\":[\"pnl\",\"costs\",\"turnover\"],\"values_checked\":" << comparison.values_checked
         << ",\"max_absolute_error\":" << comparison.max_absolute_error << ",\"max_normalized_error\":" << comparison.max_normalized_error << "},\n"
         << "\"cases\":[";
    for (std::size_t i = 0; i < checked_cases.size(); ++i) { if (i) json << ','; json << escaped(checked_cases[i]); }
    json << "],\n\"workload\":{\"paths\":" << options.paths << ",\"intervals\":" << options.intervals << ",\"block_size\":" << options.block
         << ",\"seed\":" << options.seed << ",\"warmups\":" << (options.mode == "benchmark" ? options.warmups : 0) << ",\"repeats\":" << samples.size()
         << ",\"input_hash_fnv1a64\":" << escaped(options.mode == "benchmark" ? hash_input(benchmark) : "see correctness cases") << "},\n"
         << "\"timing_policy\":{\"kernel_event_ms\":\"same-stream CUDA events, completed before reading\",\"completed_host_ms\":\"H2D inputs + kernel + D2H all outputs + stream synchronization; reused device buffers; pageable host buffers\",\"native_cpu_prepared_ms\":\"prevalidated native arithmetic and output writes only, one host thread; validation and allocation excluded\",\"quantiles\":\"none; raw completed samples retained\"},\n"
         << "\"checksum\":{\"pnl\":" << sums[0] << ",\"costs\":" << sums[1] << ",\"turnover\":" << sums[2] << "},\n\"samples\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i) json << ',';
        const auto& sample = samples[i];
        json << "{\"kernel_event_ms\":" << sample.kernel_event_ms << ",\"completed_host_ms\":" << sample.completed_host_ms << ",\"native_cpu_prepared_ms\":" << sample.native_cpu_prepared_ms << '}';
    }
    json << "]\n}\n";
    output.write(json.str());
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.help) {
            std::cout << "Usage: aster_cuda [--mode smoke|correctness|benchmark|pipeline] [--paths N] [--intervals 1..256]\n"
                      << "  [--operations 1..256 (pipeline default: 8)]\n"
                      << "  [--block-size 1..1024] [--warmups N] [--repeats N] [--seed N] [--device N]\n"
                      << "  [--output NEW_JSON_FILE]  Existing output files are never overwritten.\n";
            return 0;
        }
        Output output(options.output);
        return run(options, output);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
