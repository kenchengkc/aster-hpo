# Native Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the CPU implementation. Independent CUDA/package and benchmark work may run in parallel against the interfaces below. Steps use checkbox syntax for tracking.

**Goal:** Deliver the approved C++20 batch/streaming foundation and an A100-ready CUDA correctness/benchmark package.
**Architecture:** A standalone checked C++ library implements accounting; an optional nanobind companion exposes NumPy batch functions and a ledger. A separate CUDA runner compares identical inputs with the CPU library, records completed device/host time, and exports results.
**Tech Stack:** C++20, CMake, nanobind, scikit-build-core, NumPy, CUDA C++.
**Spec:** ../specs/2026-11-04-native-performance-design.md

## Global Constraints

- Keep the Python core dependency-free and its existing numerical/runtime behavior compatible.
- Float64, chronological accounting, opening/rebalance/liquidation costs, no fast-math or FMA contraction.
- Accuracy study: prices `[0.5,2]`, holdings `[-1,1]`, rates `[0,0.01]`, 1–256 intervals; atol `1e-12`, rtol `1e-10`, including cancellation cases.
- CPU/native, Python API, completed GPU work, and complete objective timings are distinct measurements.
- No heap allocations in the validated native arithmetic loop or successful native ledger update.
- No private keys, credentials, build products, or machine access details enter source control.
- Retain branch/worktree; no publishing or merging. Execute continuously under the user's explicit go-ahead; do not request routine plan approval again.

## Review Focus

- Invalid buffers, sizes, overlap, or nonfinite arithmetic must fail rather than corrupt memory or silently return a plausible loss.
- Failed ledger events must preserve the previous state; terminal liquidation must occur once.
- Read-only input arrays and writable outputs must retain ownership and reject hidden copies.
- CPU/GPU comparison must use identical data and completed work, including adverse shapes and invalid launch settings.
- Test runners must return failure on wrong answers, record actual hardware/settings, and preserve existing result directories.

## Shared interfaces

`native/include/aster/accounting.hpp` defines namespace `aster`:

- `struct Accounting { double pnl, costs, turnover; };`
- `Ledger(double initial_price, double cost_rate)`; `Accounting update(double price, double target_holding)`; `Accounting finish(double terminal_price)`; `void reset(double initial_price, double cost_rate)`; `Accounting result() const`.
- `void hedge_pnl(std::span<const double> prices, std::span<const double> holdings, std::size_t paths, std::size_t intervals, double cost_rate, std::span<double> pnl, std::span<double> costs, std::span<double> turnover)`.
- `void hedge_pnl_unchecked(...)` has the same argument list; callers supply already validated nonoverlapping buffers and finite safe arithmetic. It is used only for explicitly labeled prepared-input measurements.
- CMake target `aster_accounting`; default standalone configuration requires no Python. `ASTER_BUILD_PYTHON` enables bindings, `ASTER_BUILD_TESTS` and `ASTER_BUILD_BENCHMARKS` enable local tools, `ASTER_ENABLE_SANITIZERS` enables address/undefined checks.
- Python `aster_hpo_native.hedge_pnl(prices, holdings, *, cost_rate)` returns `(pnl,costs,turnover)` NumPy-owned arrays; `hedge_pnl_into(prices, holdings, pnl, costs, turnover, *, cost_rate)` writes supplied arrays. `Ledger`/`Accounting` mirror the native interface.

## Task 1: Checked native accounting and streaming ledger

**Files:** `native/include/aster/accounting.hpp`, `native/src/accounting.cpp`, `native/tests/test_accounting.cpp`, `native/CMakeLists.txt`.
**Produces:** all shared C++ interfaces and target above.

- [x] Write native tests for literal P&L `0.089`, cost `0.011`, turnover `1.1`; streaming equivalence; shorts/zero costs; empty batches; invalid sizes, overflow/overlap, nonfinite arithmetic, immutable failed updates, terminal/reset behavior, and no allocations on successful updates.
- [x] Compile the tests before implementation; confirm the missing accounting header/API is the failure.
- [x] Implement checked entry points and a fused scalar loop with fixed-size ledger state. Validate before mutation and detect nonfinite results.
- [x] Build Release with CMake and run CTest; build ASan/UBSan and rerun native tests.
- [x] Commit the verified standalone component.

## Task 2: Optional Python package and numerical parity

**Files:** `native/pyproject.toml`, `native/src/bindings.cpp`, `native/python/aster_hpo_native/__init__.py`, `native/tests/test_python.py`, `native/README.md`.
**Consumes:** Task 1 core/target.
**Produces:** companion wheel `aster-hpo-native`, version `0.1.0`, import/API above.

- [x] Write tests before the package exists: hand-calculated outputs, identical-input NumPy parity across sizes, cancellation fixtures, strict dtype/layout/shape/writable checks, overlap, read-only inputs, owned outputs, ledger errors, nonfinite input/output, and preallocated operation.
- [x] Run the tests; confirm the missing companion import is the failure.
- [x] Add nanobind bindings with explicit array constraints and GIL release only during pure native work; keep Python validation and ownership explicit.
- [x] Build/install the wheel and run the Python tests, core suite, and standalone native tests.
- [x] Check fresh installed core-only and companion environments outside the checkout; commit the package.

## Task 3: Measurement tools and CUDA test package

**Files:** `native/benchmarks/accounting_bench.cpp`, `benchmarks/native_performance.py`, `cuda/CMakeLists.txt`, `cuda/hedge_accounting.cu`, `cuda/run_tests.sh`, `cuda/README.md`, `docs/NATIVE.md`, README, CI.
**Consumes:** Task 1 core/target and Task 2 Python interface. Independent benchmark/CUDA work may begin once these signatures are fixed.
**Produces:** standalone CPU benchmark; NumPy/native API and whole-objective report; A100 (`sm_80`) correctness/benchmark executable and repeatable remote command.

- [x] Add real smoke checks for benchmark JSON, positive sample counts/durations, numerical checksums, no-overwrite behavior, and failing correctness checks. Observe missing tools before implementation.
- [x] Implement streaming event samples and batch timings with warmup/validation boundaries, consumed results, compiler/source metadata, and raw records.
- [x] Prepare a standalone CUDA float64 path-major kernel with reusable device buffers, CPU-reference comparisons, adverse dimensions, checked launch settings, CUDA event and completed host timings; record cold setup separately. A small smoke mode precedes larger benchmarks.
- [x] Add a shell runner for environment inventory, build, correctness, optional Compute Sanitizer, and exported JSON. Preserve nonzero failures and existing output directories. Do not claim GPU validation before running it on NVIDIA hardware.
- [ ] Run CPU-side tests and packaging locally; run the CUDA suite once the user supplies SSH access. Use the dedicated SSH key outside the repository and preserve host-key checking.
- [x] Add focused native CI and honest installation/measurement instructions. Request one fresh whole-branch review, resolve important findings with regressions, and commit tested artifacts.

## Authorization and execution

On 2026-11-04 the owner explicitly approved the design and said to go ahead with
the C++ foundation and prepare the CUDA test package. This also authorizes creating
the requested dedicated SSH key. The owner has rented an A100 PCIe 40GB instance;
connection details are pending. No additional instance or paid resource will be
created. The proposed CUDA launch-autotuning study remains a later experiment.

## Verification checkpoint

CPU foundation and CUDA package preparation are complete. Local combined tests:
164 passed; native Release and UBSan CTests passed; both wheel/sdist builds passed;
core-only and companion installations passed outside the checkout. A short native/API/
whole-objective report was generated with raw inputs and no new speedup claim.
A fresh final reviewer found no Critical or Important defects. Its documentation
accuracy note was corrected: method order alternates forward/reverse.

The remaining hardware-dependent check is CUDA compilation/execution/Compute
Sanitizer and Linux ASan on the rented A100 host after the owner supplies SSH
connection details. AppleClang 15's local ASan runtime aborts even for an empty
program before user code executes; this is recorded as unavailable, not passing.
