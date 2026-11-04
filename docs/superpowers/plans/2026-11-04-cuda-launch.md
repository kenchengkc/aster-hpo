# CUDA launch-overhead Implementation Plan

> Execute continuously under the owner's request. Parallelize independent fused
> kernels and benchmark collector; one final independent review and data audit.

**Goal:** Reduce launched work in resident training and rebenchmark full search plus audits.
**Spec:** ../specs/2026-11-04-cuda-launch-design.md
**Architecture:** Per-call GPU workspaces, four small-batch training kernels, fused
optimizer with existing gradients for larger shapes, preserved eager backend.
**Tech stack:** Python/CuPy, CUDA C++, Nsight Systems, existing Study/SQLite collector.

## Constraints and review focus

- Preserve seeds, float64, loss/gradients/optimizer conventions; NumPy default.
- No per-epoch host data or scalar transfers, unsafe cached addresses, or floating atomics.
- Same current stream/device; complete gradients before updating parameters.
- Include all first-use allocation/compilation and objective/workflow work in comparisons.
- Keep tiny/scaled settings fixed, controls contemporaneous, quality/failures checked.

## Task 1: Diagnose and implement launch fusion

Files: benchmarks/hedging_cuda_fused.py, cuda/include/hedging_training.cuh,
benchmarks/hedging_cuda.py, tests/test_cuda_fused.py.

- [x] Run baseline local suite and real small/scaled Nsight traces before implementation.
- [x] Observe failing kernel/module tests; implement fused forward, accounting/error,
  deterministic gradient/loss reductions, finite/norm/clipping/Adam update.
- [x] Preserve train_policy_eager, route optimized train_policy through small/fallback paths.
- [x] Run CPU host checks and A100 numerical/parity/stream/lifecycle checks; run memcheck.

## Task 2: Wire explicit comparison and collect evidence

Files: benchmarks/hedging.py, benchmarks/study.py, benchmarks/protocol.py,
benchmarks/financial_report.py, examples/hedging.py, benchmarks/cuda_workflow.py,
tests/test_hedging.py, tests/test_study.py, tests/test_cuda_workflow.py.

- [x] Test before adding explicit cuda_eager selection/provenance and 42-job protocol.
- [x] Add paired optimized/eager summaries without changing legacy 30-job behavior.
- [x] Profile final optimized/eager representative objectives and preserve raw reports.
- [x] Freeze verified sources and execute all42 campaigns, with unchanged work/quality gates.

## Task 3: Review and publish

- [x] Full local regression suite/lint, real A100 checks, independent code review.
- [x] Audit42jobs/sourceidentities/rawjournals/metrics/summary and profile counts.
- [x] Archive checksummed evidence/report/plots/reproduction; update docs, commit, retain branch.

## Completion record

- Executed source `0d2e12b`; all37frozenhashesmatch, eagerbodyunchangedfrom76da272.
- Local257passed/40GPUskips; A100127passed/twoCPU-onlyskips; memcheck42passed,
  explicitAPIreporting, zeroerrors/leakedbytes; GPUtrainingfilesmatchcommit.
- FinalNsight:small10,985->361kernels(96.7%fewer);scaled11,126->5,174(53.5%fewer).
  Scaled summedGPUkernel durationincreased6.8%; profiledtimingsarenotbenchmarks.
- All42freshcampaignscompleted:594search+126audits,0failures; smallfullworkflow
  4.625s->0.923s,paired4.955×;scaled5.264s->3.604s,paired1.461×. Smallparent
  processlatency~singleCPUparity;fourCPUworkersremainfaster.
- Independentreview's2collectorfindingstest-firstfixed, whole-suitegreen.
  Independentaudit1017finiteJSON/84journals/720metrics,all7metricsclose,exactsummary.
- [Report,traces,rawresults,checksums](../../evidence/cuda-launch-2026-11-04/README.md).
  OpaqueNsightcaptureskeptprivately; publishedfullCSVtraceexportsomitenvironmentmetadata.
  Retainfeaturebranch/worktree;instanceleft running,allrequiredresultsdownloaded.
