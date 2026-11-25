# Budget and resume demonstration

The Monte Carlo example ran on two workers with a 0.300-second time limit and
retries disabled. It stopped after 0.315 seconds with 30 results saved and two
attempts cancelled. A second run picked up where the first left off and finished
all 36 observations. Recomputing each result from its saved seeds gave the same value.

The 0.315 seconds is the observed stopping time for one run on this machine.
The CPU time spent on the two cancelled attempts is unknown. This example
tests whether work can stop and resume. Measuring speedup or model quality
would require a separate comparison.

The JSON files record events from both runs, including their budgets, stop
reasons, attempts, configurations, seeds, and observations. The archive omits
the SQLite database.

Run `python -m examples.budgeted_search --output results/new-budget-demo` to
repeat the example using only the Python standard library. The stopping time
and number of observations saved before the stop can vary. [Time limits and retries](../../RUNTIME.md#resource-limits-in-v03)

The example ran at commit `6eec3d2`. The checksums for the saved files are in
`SHA256SUMS`.
