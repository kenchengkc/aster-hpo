// Standalone accounting service-time measurements. No Python runtime is required.
#include "aster/accounting.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <sys/sysctl.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <sys/utsname.h>
#endif

#ifndef ASTER_BUILD_TYPE
#define ASTER_BUILD_TYPE "unrecorded"
#endif
#ifndef ASTER_BUILD_FLAGS
#define ASTER_BUILD_FLAGS "unrecorded"
#endif

namespace {
using Clock = std::chrono::steady_clock;
using Samples = std::vector<std::int64_t>;
namespace fs = std::filesystem;
constexpr double rate = 0.001;
constexpr double atol = 1e-12;
constexpr double rtol = 1e-10;

struct Settings {
    std::size_t paths = 2048;
    std::size_t intervals = 12;
    std::size_t samples = 100;
    std::size_t warmup = 10;
    std::size_t events = 10000;
    fs::path output;
    fs::path sources = fs::path(__FILE__).parent_path().parent_path();
    std::string host_notes = "host load/power not controlled or recorded by operator";
};

std::string quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 32) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned>(c) << std::dec;
            } else {
                out << c;
            }
        }
    }
    return out.str() + '"';
}

std::size_t positive(const std::string& text) {
    if (text.empty() || text[0] == '-') throw std::invalid_argument("expected positive integer");
    std::size_t end = 0;
    auto parsed = std::stoull(text, &end);
    if (end != text.size() || parsed == 0 || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("expected positive integer");
    }
    return static_cast<std::size_t>(parsed);
}

Settings parse(int argc, char** argv) {
    Settings settings;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") {
            std::cout << "accounting_bench [--paths N] [--intervals N] [--samples N] "
                         "[--warmup N] [--events N] [--output NEW_DIRECTORY] "
                         "[--source-root PATH] [--host-notes TEXT]\n";
            std::exit(0);
        }
        if (++i == argc) throw std::invalid_argument("missing value for " + key);
        std::string value = argv[i];
        if (key == "--paths") settings.paths = positive(value);
        else if (key == "--intervals") settings.intervals = positive(value);
        else if (key == "--samples") settings.samples = positive(value);
        else if (key == "--warmup") settings.warmup = positive(value);
        else if (key == "--events") settings.events = positive(value);
        else if (key == "--output") settings.output = value;
        else if (key == "--source-root") settings.sources = value;
        else if (key == "--host-notes") settings.host_notes = value;
        else throw std::invalid_argument("unknown argument " + key);
    }
    if (settings.intervals == std::numeric_limits<std::size_t>::max() ||
        settings.paths > std::numeric_limits<std::size_t>::max() / (settings.intervals + 1)) {
        throw std::invalid_argument("workload size overflow");
    }
    if (fs::is_directory(settings.sources / "native")) settings.sources /= "native";
    if (!settings.output.empty()) {
        if (!settings.output.parent_path().empty()) fs::create_directories(settings.output.parent_path());
        if (!fs::create_directory(settings.output)) {
            throw std::runtime_error("output directory already exists: " + settings.output.string());
        }
    }
    return settings;
}

std::int64_t elapsed(Clock::time_point start) {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    if (duration < 0) {
        throw std::runtime_error("steady clock moved backwards");
    }
    return duration;
}

double quantile(Samples values, double q) {
    std::sort(values.begin(), values.end());
    double index = static_cast<double>(values.size() - 1) * q;
    auto low = static_cast<std::size_t>(index);
    auto high = std::min(low + 1, values.size() - 1);
    return static_cast<double>(values[low]) +
           (static_cast<double>(values[high]) - static_cast<double>(values[low])) * (index - low);
}

void quantiles(std::ostream& out, const Samples& values) {
    out << "{\"p50\":" << quantile(values, 0.5) << ",\"p95\":" << quantile(values, 0.95)
        << ",\"p99\":" << quantile(values, 0.99) << '}';
}

void samples_json(std::ostream& out, const Samples& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

struct Fnv {
    std::uint64_t value = 14695981039346656037ULL;
    void add(const void* bytes, std::size_t size) {
        auto data = static_cast<const unsigned char*>(bytes);
        for (std::size_t i = 0; i < size; ++i) {
            value ^= data[i];
            value *= 1099511628211ULL;
        }
    }
    void add(const std::string& text) { add(text.data(), text.size()); }
    void file(const fs::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot hash file " + path.string());
        std::array<char, 8192> bytes{};
        while (input) {
            input.read(bytes.data(), bytes.size());
            add(bytes.data(), static_cast<std::size_t>(input.gcount()));
        }
        if (!input.eof()) throw std::runtime_error("failed reading " + path.string());
    }
    std::string hex() const {
        std::ostringstream out;
        out << std::hex << std::setfill('0') << std::setw(16) << value;
        return out.str();
    }
};

std::vector<fs::path> source_files(const fs::path& root) {
    std::vector<fs::path> files;
    if (!fs::is_directory(root)) throw std::runtime_error("source root does not exist");
    for (auto iter = fs::recursive_directory_iterator(root); iter != fs::recursive_directory_iterator(); ++iter) {
        auto name = iter->path().filename().string();
        if (iter->is_directory()) {
            if (name.starts_with(".") || name.starts_with("build") || name == "dist" || name == "__pycache__") {
                iter.disable_recursion_pending();
            }
            continue;
        }
        auto extension = iter->path().extension().string();
        if (iter->is_regular_file() && (extension == ".cpp" || extension == ".hpp" || extension == ".py" ||
                                       name == "CMakeLists.txt" || name == "pyproject.toml")) {
            files.push_back(iter->path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) throw std::runtime_error("source root contains no native inputs");
    return files;
}

fs::path executable_path(const char* argv0) {
#if defined(__APPLE__)
    std::uint32_t length = 0;
    _NSGetExecutablePath(nullptr, &length);
    std::vector<char> path(length);
    if (_NSGetExecutablePath(path.data(), &length) == 0) return fs::weakly_canonical(path.data());
#elif defined(__linux__)
    return fs::read_symlink("/proc/self/exe");
#endif
    return fs::absolute(argv0);
}

std::string cpu_model() {
#if defined(__APPLE__)
    std::size_t size = 0;
    if (sysctlbyname("machdep.cpu.brand_string", nullptr, &size, nullptr, 0) == 0) {
        std::string value(size, '\0');
        if (sysctlbyname("machdep.cpu.brand_string", value.data(), &size, nullptr, 0) == 0) {
            if (!value.empty() && value.back() == '\0') value.pop_back();
            return value;
        }
    }
#elif defined(__linux__)
    std::ifstream info("/proc/cpuinfo");
    for (std::string line; std::getline(info, line);) {
        if (line.starts_with("model name")) return line.substr(line.find(':') + 2);
    }
#endif
    return "unavailable";
}

void provenance(std::ostream& out, const Settings& settings, const char* argv0) {
    auto files = source_files(settings.sources);
    Fnv sources;
    for (const auto& path : files) {
        sources.add(path.lexically_relative(settings.sources).generic_string());
        sources.add("\0", 1);
        sources.file(path);
        sources.add("\0", 1);
    }
    Fnv binary;
    binary.file(executable_path(argv0));
    std::string platform = "unavailable";
#if defined(__unix__) || defined(__APPLE__)
    struct utsname system{};
    if (uname(&system) == 0) platform = std::string(system.sysname) + " " + system.release + " " + system.machine;
#endif
    out << "{\"source_fnv1a64\":" << quote(sources.hex()) << ",\"source_files\":[";
    for (std::size_t i = 0; i < files.size(); ++i) {
        if (i) out << ',';
        out << quote(files[i].lexically_relative(settings.sources).generic_string());
    }
    out << "],\"binary_fnv1a64\":" << quote(binary.hex())
        << ",\"hash_note\":\"FNV-1a 64-bit content identifier, not a cryptographic hash\""
        << ",\"compiler\":" << quote(__VERSION__)
        << ",\"cplusplus\":" << __cplusplus
        << ",\"build_type\":" << quote(ASTER_BUILD_TYPE)
        << ",\"compiler_flags\":" << quote(ASTER_BUILD_FLAGS)
        << ",\"platform\":" << quote(platform)
        << ",\"cpu\":" << quote(cpu_model())
        << ",\"hardware_threads\":" << std::thread::hardware_concurrency()
        << ",\"processes\":1,\"native_threads\":1,\"instrumentation\":\"none in benchmark harness\""
        << ",\"host_power_load\":" << quote(settings.host_notes)
        << ",\"floating_point\":\"float64; build must disable fast-math and FMA contraction\""
        << ",\"thread_environment\":{";
    const std::array<const char*, 3> envs{"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"};
    for (std::size_t i = 0; i < envs.size(); ++i) {
        if (i) out << ',';
        const char* value = std::getenv(envs[i]);
        out << quote(envs[i]) << ':' << (value ? quote(value) : "null");
    }
    out << "}}";
}

// Fixed integer generator, so the workload does not depend on library RNG implementations.
double uniform(std::uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<double>(state >> 11) * 0x1.0p-53;
}

void assert_close(double actual, long double expected, const char* label) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(static_cast<long double>(actual) - expected) > atol + rtol * std::abs(expected)) {
        throw std::runtime_error(std::string("correctness check failed: ") + label);
    }
}

struct Event { bool finish; std::int64_t ns; };

int benchmark(int argc, char** argv) {
    auto settings = parse(argc, argv);
    std::vector<double> prices(settings.paths * (settings.intervals + 1));
    std::vector<double> holdings(settings.paths * settings.intervals);
    std::uint64_t rng = 7;
    for (auto& price : prices) price = 0.5 + 1.5 * uniform(rng);
    for (auto& holding : holdings) holding = -1.0 + 2.0 * uniform(rng);
    Fnv workload;
    workload.add(prices.data(), prices.size() * sizeof(double));
    workload.add(holdings.data(), holdings.size() * sizeof(double));
    std::vector<double> pnl(settings.paths), costs(settings.paths), turnover(settings.paths);
    auto checked = [&] {
        aster::hedge_pnl(prices, holdings, settings.paths, settings.intervals, rate, pnl, costs, turnover);
    };
    auto prepared = [&] {
        aster::hedge_pnl_unchecked(prices, holdings, settings.paths, settings.intervals, rate, pnl, costs, turnover);
    };
    checked();
    for (std::size_t i = 0; i < settings.paths; ++i) {
        const auto* p = prices.data() + i * (settings.intervals + 1);
        const auto* h = holdings.data() + i * settings.intervals;
        long double gain = 0, trades = 0, previous = 0;
        aster::Ledger ledger(p[0], rate);
        for (std::size_t j = 0; j < settings.intervals; ++j) {
            gain += static_cast<long double>(h[j]) * (static_cast<long double>(p[j + 1]) - p[j]);
            trades += std::abs(static_cast<long double>(h[j]) - previous) * p[j];
            previous = h[j];
            ledger.update(p[j], h[j]);
        }
        trades += std::abs(previous) * p[settings.intervals];
        auto state = ledger.finish(p[settings.intervals]);
        assert_close(pnl[i], gain - rate * trades, "batch pnl");
        assert_close(costs[i], rate * trades, "batch costs");
        assert_close(turnover[i], trades, "batch turnover");
        assert_close(state.pnl, pnl[i], "streaming pnl");
        assert_close(state.costs, costs[i], "streaming costs");
        assert_close(state.turnover, turnover[i], "streaming turnover");
    }
    const auto expected_pnl = pnl, expected_costs = costs, expected_turnover = turnover;
    prepared();
    for (std::size_t i = 0; i < settings.paths; ++i) {
        assert_close(pnl[i], expected_pnl[i], "prepared pnl");
        assert_close(costs[i], expected_costs[i], "prepared costs");
        assert_close(turnover[i], expected_turnover[i], "prepared turnover");
    }
    auto consume = [&] {
        double sum = 0;
        for (std::size_t i = 0; i < settings.paths; ++i) sum += pnl[i] + costs[i] + turnover[i];
        return sum;
    };
    Samples timer;
    timer.reserve(1000);
    for (int i = 0; i < 1000; ++i) { auto start = Clock::now(); timer.push_back(elapsed(start)); }
    Samples prepared_samples, checked_samples;
    prepared_samples.reserve(settings.samples);
    checked_samples.reserve(settings.samples);
    double prepared_checksum = 0, checked_checksum = 0;
    for (std::size_t i = 0; i < settings.warmup; ++i) { checked(); prepared(); }
    // Alternate order, preserving all completed call samples without trimming outliers.
    for (std::size_t i = 0; i < settings.samples; ++i) {
        auto sample = [&](auto& call, Samples& values, double& checksum) {
            auto start = Clock::now();
            call();
            values.push_back(elapsed(start));
            checksum += consume();
        };
        if (i % 2 == 0) {
            sample(prepared, prepared_samples, prepared_checksum);
            sample(checked, checked_samples, checked_checksum);
        } else {
            sample(checked, checked_samples, checked_checksum);
            sample(prepared, prepared_samples, prepared_checksum);
        }
    }
    aster::Ledger ledger(prices[0], rate);
    for (std::size_t i = 0; i < settings.warmup; ++i) {
        ledger.update(prices[0], holdings[0]);
        ledger.finish(prices[1]);
        ledger.reset(prices[0], rate);
    }
    Samples update_samples, finish_samples;
    std::vector<Event> events;
    update_samples.reserve(settings.events);
    finish_samples.reserve(settings.events / settings.intervals + 1);
    events.reserve(settings.events + settings.events / settings.intervals + 1);
    std::size_t path = 0, interval = 0;
    double stream_checksum = 0;
    for (std::size_t i = 0; i < settings.events; ++i) {
        std::size_t po = path * (settings.intervals + 1), ho = path * settings.intervals;
        auto start = Clock::now();
        auto state = ledger.update(prices[po + interval], holdings[ho + interval]);
        auto duration = elapsed(start);
        update_samples.push_back(duration);
        events.push_back({false, duration});
        stream_checksum += state.pnl + state.costs + state.turnover;
        if (++interval == settings.intervals || i + 1 == settings.events) {
            start = Clock::now();
            state = ledger.finish(prices[po + interval]);
            duration = elapsed(start);
            finish_samples.push_back(duration);
            events.push_back({true, duration});
            stream_checksum += state.pnl + state.costs + state.turnover;
            path = (path + 1) % settings.paths;
            interval = 0;
            ledger.reset(prices[path * (settings.intervals + 1)], rate);
        }
    }
    if (!std::isfinite(prepared_checksum + checked_checksum + stream_checksum)) {
        throw std::runtime_error("nonfinite consumed checksum");
    }
    for (const auto* values : {&prepared_samples, &checked_samples, &update_samples, &finish_samples}) {
        if (*std::max_element(values->begin(), values->end()) <= 0) {
            throw std::runtime_error("clock could not resolve any positive samples; collect more events");
        }
    }
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\"schema_version\":1,\"interpretation\":\"single-threaded local accounting service time\",\"provenance\":";
    provenance(out, settings, argv[0]);
    out << ",\"settings\":{\"paths\":" << settings.paths << ",\"intervals\":" << settings.intervals
        << ",\"samples\":" << settings.samples << ",\"warmup\":" << settings.warmup
        << ",\"streaming_updates\":" << settings.events << ",\"cost_rate\":" << rate
        << ",\"independent_process_sessions\":1,\"rng\":\"xorshift64, seed 7\"}"
        << ",\"workload_fnv1a64\":" << quote(workload.hex())
        << ",\"correctness\":{\"passed\":true,\"oracle\":\"long double scalar accounting and batch/streaming comparison\","
           "\"atol\":1e-12,\"rtol\":1e-10,\"long_double_digits\":" << std::numeric_limits<long double>::digits << '}'
        << ",\"timer\":{\"clock\":\"std::chrono::steady_clock\",\"subtracted\":false,"
           "\"zero_note\":\"Raw zero samples are retained when events complete within one clock increment\","
           "\"nominal_period_ns\":" << 1e9 * Clock::period::num / Clock::period::den
        << ",\"samples_ns\":";
    samples_json(out, timer);
    out << ",\"quantiles_ns\":";
    quantiles(out, timer);
    out << "},\"quantile_method\":\"linear interpolation at (n-1)*q\",\"batch\":[";
    auto batch_json = [&](const char* name, const char* boundary, const Samples& values, double checksum) {
        out << "{\"backend\":" << quote(name) << ",\"boundary\":" << quote(boundary)
            << ",\"samples_ns\":";
        samples_json(out, values);
        out << ",\"call_quantiles_ns\":";
        quantiles(out, values);
        const double median_ns = quantile(values, 0.5);
        if (median_ns > 0) {
            out << ",\"median_ns_per_path\":" << median_ns / settings.paths
                << ",\"median_paths_per_second\":" << settings.paths * 1e9 / median_ns
                << ",\"median_rate_note\":null";
        } else {
            out << ",\"median_ns_per_path\":null,\"median_paths_per_second\":null,"
                   "\"median_rate_note\":\"unavailable: median completed call is below clock resolution\"";
        }
        out << ",\"allocation_count\":null,\"allocation_note\":\"not measured\",\"checksum\":" << checksum << '}';
    };
    batch_json("native_prepared", "prevalidated inputs, arithmetic and caller-owned output writes; validation excluded",
               prepared_samples, prepared_checksum);
    out << ',';
    batch_json("native_checked", "checked native call including validation and caller-owned output writes; output allocation excluded",
               checked_samples, checked_checksum);
    out << "],\"streaming\":{\"boundary\":\"individual sequential replay events including native validation; input generation/reset/checksum excluded\","
           "\"quantile_method\":\"linear interpolation at (n-1)*q\",\"allocation_count\":null,"
           "\"allocation_note\":\"not measured\",\"zero_update_samples\":"
        << std::count(update_samples.begin(), update_samples.end(), 0)
        << ",\"update_sample_count\":" << update_samples.size()
        << ",\"update_samples_ns\":";
    samples_json(out, update_samples);
    out << ",\"update_quantiles_ns\":";
    quantiles(out, update_samples);
    out << ",\"finish_sample_count\":" << finish_samples.size() << ",\"finish_samples_ns\":";
    samples_json(out, finish_samples);
    out << ",\"finish_quantiles_ns\":";
    quantiles(out, finish_samples);
    out << ",\"checksum\":" << stream_checksum << ",\"events\":[";
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (i) out << ',';
        out << "{\"kind\":" << quote(events[i].finish ? "finish" : "update")
            << ",\"duration_ns\":" << events[i].ns << '}';
    }
    out << "]}}\n";
    if (!settings.output.empty()) {
        std::ofstream file(settings.output / "report.json");
        file << out.str();
        if (!file) throw std::runtime_error("could not write report.json");
    }
    std::cout << out.str();
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try { return benchmark(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "measurement failed: " << error.what() << '\n';
        return 1;
    }
}
