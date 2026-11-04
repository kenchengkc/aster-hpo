# Resident CUDA accounting pipeline

The owner explicitly requested optimizing CUDA data residency and benchmarking
end-to-end on the already rented A100. Extend the existing standalone accounting
tool; retain the approved float64 accounting and numerical tolerances. Execute
under that go-ahead in the existing feature worktree. No new instance is required.

## Work and output contract

Use a transaction-cost sensitivity sweep: identical prices and holdings, K distinct
cost rates spanning `[0,0.01]` (K=1 uses 0.01). Every operation recomputes the same
existing accounting kernel. Return **all K operations' three per-path arrays**;
never compare a full-output baseline against final-only or scalar-only results.
This is a systems workload, not financial hyperparameter optimization or GPU neural
training. Data reuse is applicable when paths/holdings stay fixed; changed inputs
must be explicitly uploaded again.

Compare these boundaries in one binary using preallocated host/device capacity:

1. Copy-per-call: upload the inputs, run one kernel, download three arrays, and
   synchronize for every operation. This retains the existing behavior.
2. Resident session: upload inputs once, run K kernels into separate retained output
   slots, bulk-download all outputs, and synchronize. Measure from before upload
   to completed download. This is the primary end-to-end optimized measurement.
3. Preloaded resident calls: already uploaded inputs, K kernels, bulk-download all
   outputs, and synchronize. Report separately; it excludes initial upload.
4. Prepared native CPU sequence: K equivalent calls with the same rates and outputs.
   This is context, not a claim against an optimally cached CPU sensitivity engine.

The optimization combines input residency, batched output copies, and fewer host
synchronizations. Do not attribute gains solely to transfer elimination. Preserve
the old benchmark mode/schema and archived evidence.

## Implementation and checks

Split device-buffer upload, checked resident launch, and readback; allocate K output
slots once. Reject launch before upload, invalid output slots, invalid rates, and
incomplete/stale readback. Validate dimensions and memory products before allocation.
Check all outputs against the existing checked CPU reference outside timing,
including K=1, different rates, shorts, cancellation, partial blocks, changed shapes,
and reloaded changed inputs of the same shape. Keep the original arithmetic and
strict floating-point compiler settings.

Count actual requested H2D/D2H bytes and copy calls at the transfer helper. For N
paths, T intervals, K operations, input bytes are N*(2*T+1)*8 and output bytes are
K*N*3*8. Copy-per-call uploads K input batches; a resident session uploads one;
preloaded timing uploads zero. All three return equal output bytes. These are API
copy requests, not physical PCIe traffic measurements.

Add `--mode pipeline` and `--operations 1..256`. Keep allocation/context startup
separate from reusable-capacity session timing. Record all raw completed samples,
comparison results, ordered checksums, transfer counters, exact rates, input/build
identity, and method order. Randomize timed method order reproducibly per sample.
Any device-event span across launches includes inter-launch gaps and must be named
accordingly rather than called exclusive kernel time.

## Benchmark protocol

On the same A100, test path counts 1, 256, 2048, 65536 with 12 intervals and K=1,
8,32. Use five warmups and 15 samples per fresh process; run three processes per
setting, interleaving settings deterministically. The principal comparison is
resident-session versus copy-per-call, including initial upload and all outputs.
Report per-process medians and their paired ratios, with ranges across processes.
Do not relabel pooled dependent samples as independent runs or claim p99.

Run correctness and Compute Sanitizer before collecting timings. Download raw
records, verify sources/binary hashes, and publish reproduction commands plus the
observed scope. Cold setup, Python binding costs, full neural training, and complete
HPO runtime are outside this standalone end-to-end session measurement.
