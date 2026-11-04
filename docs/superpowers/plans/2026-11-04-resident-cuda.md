# Resident CUDA Implementation Plan

> Use superpowers:executing-plans. The owner explicitly said to go ahead; implement continuously. CUDA implementation and the independent process benchmark collector can run in parallel against the JSON contract below.

**Goal:** Keep accounting data resident across operations and measure complete sessions on the current A100.
**Spec:** ../specs/2026-11-04-resident-cuda-design.md
**Architecture:** Existing device buffers gain upload/launch/readback phases and retained operation outputs. A Python collector drives fresh executable processes and validates records before summarizing.
**Tech stack:** C++20/CUDA, standard-library Python, existing native reference.

## Global constraints and review focus

- Preserve numerical arithmetic, float64, tolerances, old modes, and immutable evidence.
- Equal workloads/outputs across variants; retain every operation's results.
- Validate state, same-shape reloads, output-slot bounds, size overflow, and all requested transfer counters.
- Time completed work; include initial upload and final all-output download in the primary session metric.
- Keep keys and SSH connection details outside committed artifacts; no extra paid resources.

## JSON interface

Pipeline mode keeps device/build/comparison identity and reports workload operations
and explicit `rates`. `pipeline_samples` contains method order and completed durations
`copy_per_call_ms`, `resident_session_ms`, `resident_preloaded_ms`,
`native_cpu_sequence_ms`. Each sample records `transfers` for the three GPU methods,
with integer `h2d_bytes`, `d2h_bytes`, `h2d_calls`, `d2h_calls`. `checksum` retains
each method's all-output checksums. The implementation may add detailed fields but
must explain their timing boundaries. All records must pass numerical comparisons.

## Task 1: Resident execution and hardware correctness

Files: `cuda/hedge_accounting.cu`, CLI/result checks and CUDA README.

- [x] Show the old executable rejecting pipeline mode/operations before implementation.
- [x] Add resident state, upload-once execution, K retained output slots and transfer counters.
- [x] Add real correctness cases for all-output equivalence, changed inputs/shapes and invalid lifecycle; extend CLI checks.
- [x] Build and run on A100; run Compute Sanitizer with the new resident paths exercised.

## Task 2: Session benchmark and reproducible summaries

Files: `cuda/benchmark_pipeline.py`, `cuda/tests/test_pipeline_report.py`.

- [x] Write failing tests for corrupted/missing/nonfinite records and wrong transfer counts; preserve existing result directories.
- [x] Implement fresh-process collector, source/binary hashes, raw records, method order checks and validated summaries.
- [x] Run the fixed protocol from the spec; compare per-process medians/paired ratios without claiming independent samples within one process.

## Task 3: Verification and evidence

- [x] Run local tests/lint, native regression checks, and actual GPU correctness/memcheck.
- [x] Request one fresh whole-change review; fix substantive findings with regressions.
- [x] Archive raw records/settings/checksums, update docs with precise session boundaries, commit verified changes and retain branch.

## Completion record

- Executed CUDA source: `6cb9805`; collector validation hardening: `3f2da2b`.
- A100: 117 correctness cases, default and explicit pipeline memcheck with zero
  errors/leaked bytes, and the K=256 operation boundary passed.
- Fixed protocol: 36 fresh processes / 540 measured sample sets. For 65,536 paths
  and K=32, complete sessions fell from 26.408633ms to 4.421862ms; median paired
  process speedup 5.972288×, range 5.824863–6.435166×.
- One independent code review and data audit completed. A comparison-count
  validator finding was fixed after three failing regressions; all 36 archived
  records pass the stronger validator and regenerate the original summary exactly.
- Local verification: 182 tests passed; lint and formatting passed.
- [Report, raw records, hashes, and reproduction commands](../../evidence/cuda-resident-2026-11-04/README.md).
  Retain `feat/native-performance`; no merge or push requested.
