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

- [ ] Show the old executable rejecting pipeline mode/operations before implementation.
- [ ] Add resident state, upload-once execution, K retained output slots and transfer counters.
- [ ] Add real correctness cases for all-output equivalence, changed inputs/shapes and invalid lifecycle; extend CLI checks.
- [ ] Build and run on A100; run Compute Sanitizer with the new resident paths exercised.

## Task 2: Session benchmark and reproducible summaries

Files: `cuda/benchmark_pipeline.py`, `cuda/tests/test_pipeline_report.py`.

- [ ] Write failing tests for corrupted/missing/nonfinite records and wrong transfer counts; preserve existing result directories.
- [ ] Implement fresh-process collector, source/binary hashes, raw records, method order checks and validated summaries.
- [ ] Run the fixed protocol from the spec; compare per-process medians/paired ratios without claiming independent samples within one process.

## Task 3: Verification and evidence

- [ ] Run local tests/lint, native regression checks, and actual GPU correctness/memcheck.
- [ ] Request one fresh whole-change review; fix substantive findings with regressions.
- [ ] Archive raw records/settings/checksums, update docs with precise session boundaries, commit verified changes and retain branch.
