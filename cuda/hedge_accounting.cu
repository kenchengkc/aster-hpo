#include <cuda_runtime.h>

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
        else if (option == "--device") {
            const auto ordinal = number(value);
            if (ordinal > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::invalid_argument("device ordinal overflow");
            options.device = static_cast<int>(ordinal);
        } else throw std::invalid_argument("unknown option: " + option);
    }
    if (options.mode != "smoke" && options.mode != "correctness" && options.mode != "benchmark") throw std::invalid_argument("--mode must be smoke, correctness, or benchmark");
    if (options.paths == 0 || options.intervals == 0 || options.intervals > 256 || options.block == 0 || options.block > 1024 || options.repeats == 0 || options.repeats > max_samples || options.warmups > max_samples) {
        throw std::invalid_argument("paths/repeats must be positive; intervals 1..256, block-size 1..1024, warmups/repeats <=1000000");
    }
    checked_product(checked_product(options.paths, options.intervals + 1), sizeof(double));
    checked_product(checked_product(options.paths, options.intervals), sizeof(double));
    checked_product(options.paths, 3 * sizeof(double));
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
struct Results {
    std::vector<double> pnl, costs, turnover;
    explicit Results(std::size_t paths) : pnl(paths), costs(paths), turnover(paths) {}
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

__global__ void accounting_kernel(const double* prices, const double* holdings,
                                  std::size_t paths, std::size_t intervals, double rate,
                                  double* pnl, double* costs, double* turnover) {
    const std::size_t path = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= paths) return;
    const auto price_offset = path * (intervals + 1);
    const auto holding_offset = path * intervals;
    double gain = 0, traded = 0, previous = 0;
    for (std::size_t t = 0; t < intervals; ++t) {
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

struct Sample { double kernel_event_ms = 0, completed_host_ms = 0, native_cpu_prepared_ms = 0; };

// Device allocations and timing events are reused across correctness cases and every
// warm/measured call. Pageable host vectors are intentional; transfer staging is in host time.
class DeviceBuffers {
public:
    DeviceBuffers(std::size_t price_count, std::size_t holding_count, std::size_t paths,
                  std::size_t block, const cudaDeviceProp& device) : block_(block) {
        if (block > static_cast<std::size_t>(device.maxThreadsPerBlock) || block > static_cast<std::size_t>(device.maxThreadsDim[0])) {
            throw std::invalid_argument("block size exceeds device launch limits");
        }
        const auto blocks = paths == 0 ? 0 : (paths - 1) / block + 1;
        if (blocks > static_cast<std::size_t>(device.maxGridSize[0])) throw std::invalid_argument("grid size exceeds device launch limits");
        price_capacity_ = price_count; holding_capacity_ = holding_count; path_capacity_ = paths;
        // Account for each buffer without overflowing the aggregate byte count.
        const auto price_bytes = checked_product(price_count, sizeof(double));
        const auto holding_bytes = checked_product(holding_count, sizeof(double));
        const auto output_bytes = checked_product(paths, 3 * sizeof(double));
        if (price_bytes > std::numeric_limits<std::size_t>::max() - holding_bytes ||
            price_bytes + holding_bytes > std::numeric_limits<std::size_t>::max() - output_bytes) {
            throw std::invalid_argument("aggregate device allocation overflow");
        }
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        if (price_bytes + holding_bytes + output_bytes > free) throw std::runtime_error("workload exceeds available device memory");
        try {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreate(&begin_));
            CUDA_CHECK(cudaEventCreate(&end_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&prices_), price_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&holdings_), holding_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pnl_), paths * sizeof(double)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&costs_), paths * sizeof(double)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&turnover_), paths * sizeof(double)));
        } catch (...) { release(); throw; }
    }
    ~DeviceBuffers() { release(); }
    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;
    Sample execute(const Data& data, Results& result) {
        if (data.prices.size() > price_capacity_ || data.holdings.size() > holding_capacity_ || data.paths > path_capacity_ ||
            result.pnl.size() != data.paths || result.costs.size() != data.paths || result.turnover.size() != data.paths) {
            throw std::runtime_error("device buffer capacity/output mismatch");
        }
        if (data.paths == 0) return {};
        const auto blocks = (data.paths - 1) / block_ + 1;
        const auto start = Clock::now();
        CUDA_CHECK(cudaMemcpyAsync(prices_, data.prices.data(), data.prices.size() * sizeof(double), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(holdings_, data.holdings.data(), data.holdings.size() * sizeof(double), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaEventRecord(begin_, stream_));
        accounting_kernel<<<static_cast<unsigned>(blocks), static_cast<unsigned>(block_), 0, stream_>>>(prices_, holdings_, data.paths, data.intervals, data.rate, pnl_, costs_, turnover_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(end_, stream_));
        CUDA_CHECK(cudaMemcpyAsync(result.pnl.data(), pnl_, data.paths * sizeof(double), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(result.costs.data(), costs_, data.paths * sizeof(double), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(result.turnover.data(), turnover_, data.paths * sizeof(double), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        const auto finish = Clock::now();
        float elapsed = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin_, end_));
        return {static_cast<double>(elapsed), milliseconds(start, finish), 0};
    }
    void close() {
        // All work is already complete. Check teardown errors on the successful path.
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
        begin_ = end_ = nullptr; stream_ = nullptr;
    }
    double *prices_ = nullptr, *holdings_ = nullptr, *pnl_ = nullptr, *costs_ = nullptr, *turnover_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t begin_ = nullptr, end_ = nullptr;
    std::size_t price_capacity_ = 0, holding_capacity_ = 0, path_capacity_ = 0, block_;
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
            std::cout << "Usage: aster_cuda [--mode smoke|correctness|benchmark] [--paths N] [--intervals 1..256]\n"
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
