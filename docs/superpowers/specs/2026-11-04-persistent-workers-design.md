# Reusable managed workers and repeated GPU HPO

The owner requests persistent GPU workers and repeated HPO benchmarks versus CPU
on the rented A100. Continue in the existing isolated feat/native-performance
worktree. This introduces explicit reusable managed processes; it does not claim
that the existing direct one-worker API initializes CUDA on every study. Existing
one-worker unbounded calls in one interpreter already reuse CuPy/context caches.

## Public contract

Export dependency-free `WorkerPool(workers=1, threads_per_worker=1)` from aster_hpo.
Use as a context manager, or explicitly close in a finally block. Add keyword-only
`worker_pool=None` to Study.optimize and audit; omitted keeps current behavior.
Supplied pool selects managed spawned execution even when Study.workers==1.
GPU-specific imports stay inside the objective and evaluating child.

Pool settings are immutable positive integers. A study may request at most the
pool's capacity, with exactly matching threads_per_worker. This permits a four-worker
search followed by a one-worker audit in the same pool. Reject incompatible,
closed, foreign-process or concurrent/nested use before touching storage. Acquire
an exclusive run lease; never pickle the pool. Spawn lazily on admitted work so
first-use startup remains inside measured dispatch and wall budgets.

Retain healthy workers on successful completion or drained attempt allowance.
Rebind a freshly serialized objective once per child for each optimize/audit
invocation, even if its identity/caller object is unchanged. Diagnostic directory,
market and arbitrary callable instance state must not leak from the prior run.
Arbitrary module globals intentionally remain process-local, including CUDA caches.
Each invocation has a unique token distinct from SQLite session/attempt IDs;
validate request/response tokens as well as attempts. Discard stale responses,
retire dead idle children, and never reuse terminated pipes.

Preserve dispatch-time timeout accounting and mailbox receipt timestamps. Timeout
reaps/replaces its culprit; infrastructure death retries identical seeds under the
existing cap. Numerical failures are accepted failed observations without retry.
Wall/caller cancellation and transport/coordinator exceptions reap owned processes
before bookkeeping and leave an empty reusable pool. No active work survives a
session ending. Healthy idle processes remain only until explicit pool close.
Close is idempotent, bounded using existing shared group deadlines, and rejects
closing a borrowed pool rather than racing its coordinator. Apply fixed native
thread limits inside children too. Record persistent_processes backend, pool and
run identities, capacity and child PID evidence in metadata without changing
statistical seeds or the durable study manifest between pooled/unpooled resumes.

## Implementation approach

Extend the current owned Worker transport with run-token/fresh-binding messages,
reuse its exchange thread and bounded shutdown, and allow run_parallel to borrow
a pool-owned worker list/event. Keep the existing scheduler/journal protocol.
A proxy objective or background web service would duplicate timeout ownership or
require a service protocol; an explicit pool integrates with current supervision.
CUDA math/workspaces are unchanged and remain allocated per objective invocation.

## Fixed repeated-workflow benchmark

Two markets: existing tiny128/evaluation2048 and scaled4096/evaluation32768;
64epochs,12steps,four candidates,FullBudget4 and three independent audits per winner.
Use three sequence roots41000,42000,43000. Each sequence performs three fresh HPO
workflows with study seeds root,root+1,root+2, fresh SQLite/metric paths and fresh
callables. Freeze the four-candidate pool using the sequence root for all three
positions; change replication seeds with root+position. This holds width mix fixed
while ensuring new simulations, and keeps paired variants on identical workloads. First plus two subsequent runs are reported separately; the three
sequence processes are the independent timing units, not the six warm positions.

Eight variants, each with a fresh coordinator process per workload/sequence:
NumPy/direct1,CUDA/direct1; fresh-pool NumPy1,CUDA1,NumPy4; reused-pool
NumPy1,CUDA1,NumPy4. Fresh-pool creates/closes a pool around each search+audit;
reused-pool keeps it across all three workflows and closes at sequence end.
Direct GPU is the already persistent same-interpreter control. CPU pooling receives
the same opportunity to amortize startup. Audits request one worker from any pool.
This yields48 sequence processes and144 workflows/2736 expected objective evaluations.
No objective warmup or synthetic initialization before first-run timing.

Time first run from before numerical/backend imports/pool creation through search,
locking, audit and artifact writes. Later runs use the same complete workflow
boundary; fresh-pool run latency includes its close. Record final reused-pool close
separately and include every setup/shutdown in full sequence and parent process
elapsed. Also report amortized sequence cost per workflow, so warm-only speedups
cannot hide lifecycle overhead. Shared compiler disk cache is retained/disclosed.
Verify PIDs plus process tokens for actual reuse/restart, no surviving workers,
thread settings, all attempts/observations, fresh journals and no skipped work.
All matched seven metrics must satisfy rtol1e-6,atol1e-8. Match search identities;
record valid winner/audit differences without rejecting their different config IDs.
Independent code review, data audit, exact source hashes and summary regeneration
precede reporting. No profiling captures or unrestricted environment dumps required.
Archive results locally; leave rented instance and feature branch intact.
