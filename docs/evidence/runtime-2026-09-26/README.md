# Budget and resume demonstration

The standard-library Monte Carlo example ran with two workers, a 0.300-second
wall allowance, and zero infrastructure retries. It stopped at 0.315 seconds,
with 30 accepted observations and two deliberately cancelled attempts. A new
invocation reused committed work and completed all 36 observations. Each accepted
value was recomputed from its saved seeds; values and committed observations match.

This demonstrates runtime controls, not algorithmic speedup. The two interrupted
attempts have unknown CPU cost; no zero-cost assumption or quality inference is made.
Observed stopping latency is one run on the recorded machine, not a guaranteed bound.

The JSON exports preserve session events, budgets, stop reasons, attempts, configurations,
seeds, and observations. The SQLite execution database is not included. See
[the runtime contract](../../RUNTIME.md#resource-limits-in-v03) and run
`python -m examples.budgeted_search --output results/new-budget-demo` to reproduce
the procedure; exact timings and the number of accepted observations can differ.

Execution revision: `6eec3d2`. Source changes after this snapshot do not change these
historical records. `SHA256SUMS` verifies the accompanying artifact bytes.
