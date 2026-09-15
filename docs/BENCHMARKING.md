# Benchmarking protocol

No comparative performance results are established yet. The synthetic example is a smoke
test, not a benchmark of practical speedups.

Before collecting final results, freeze the objective, search space, candidate budget,
replication checkpoints, hardware allocation, seed schedule, selection budget, and analysis.
Choose the quality margin using a disjoint pilot. Keep pilot output separate from final data.

## Required comparisons

1. Serial full-budget random search.
2. Parallel full-budget random search.
3. Parallel random search with ASHA.
4. Parallel random search with the experimental replication scheduler.
5. Optuna TPE and Ray Tune baselines with documented settings and versions.

Reuse the candidate pool for allocation-only comparisons. TPE chooses its own candidates
and must be labeled as a whole-method comparison. Match workers and numerical-library
thread limits, and report both equal-wall-time and equal-compute comparisons.

## Quality and uncertainty

Lock the selected configuration before independent audit. Predeclare a non-inferiority
margin in objective units and an analysis appropriate for the sampling structure. Account
for variability across optimizer runs as well as nested training replications. The current
`audit()` helper does not implement this complete analysis.

Report confidence intervals, failed runs, target-attainment rates, and cases where pruning
loses. Failure to detect a difference is not proof of equivalence. Do not use repeatedly
inspected ordinary confidence intervals as a sequential selection guarantee.

## Measurements and artifacts

- Wall time, separately identifying setup, optimization, selection, and audit.
- Allocated CPU-hours and objective CPU time, with their different meanings made explicit.
- Number of completed, failed, and pruned evaluations; retry and wasted-work cost.
- Quality curves, paired differences, and scaling on supported worker counts.
- Code revision, dependency lock/snapshot, machine details, thread limits, and all seeds.
- Raw observations and configurations sufficient to regenerate figures.

Changing algorithmic work while adding workers conflates two sources of improvement.
Measure fixed-workload scaling separately from savings due to adaptive allocation.
For target-attainment curves, predefine evaluation times and monitoring data and account
for monitoring costs. Runs that never reach the target must not be omitted.

The full rationale and planned financial workload are described in [DESIGN.md](../DESIGN.md).
