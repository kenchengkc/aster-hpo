# Aster upgrade: credible scheduler evidence

Date: 2026-09-23
Status: approved for implementation on 2026-09-23.
Inspected revision: `4026c29`.

## Intended outcome

Build a parallel hyperparameter optimization library that reduces the cost of
stochastic simulation campaigns while retaining independently measured solution
quality. The intended audience is hedge fund recruiters, quantitative researchers,
and research engineers.

The proposed first milestone makes the scheduler's behavior explainable and
reproducible. Later milestones establish practical speed and quality on a financial
workload. Assume a laptop or desktop CPU for the first release; GPU and cluster
execution remain optional extensions. Hardware and delivery timing are not yet
confirmed by the project owner.

The contribution should be a measured approach to allocating noisy replications.
Parallel HPO and common random numbers are established techniques, so their use
alone is not a novelty claim.

## What the current repository establishes

- The source-tree test command `PYTHONPATH=src python3 -m pytest -q` passes all
  37 tests on Python 3.11.5. Ruff lint and formatting checks also pass.
- A bare test invocation in the inspected interpreter cannot import `aster_hpo`
  because the package is not installed there. This is an environment setup issue;
  the README already documents an editable installation.
- The implementation has process workers, stable seeds, replication racing,
  atomic completed-result export, and an independent audit namespace.
- The saved website example evaluates 432 replications for full-budget search and
  72 for racing, selecting the same candidate. These are synthetic smoke results,
  not evidence of practical speedup or general quality preservation.
- `DESIGN.md` and `docs/BENCHMARKING.md` already describe an appropriate longer-term
  study. The priority is to implement and test that direction.

## Concrete gaps identified in the code

1. **Comparison opportunities depend on candidate order.** In
   `src/aster_hpo/core.py`, `_pruning_decision` runs only when a candidate itself
   reaches a checkpoint. Missing or subsequently improved comparison data does
   not trigger reconsideration of that checkpoint.

   With the existing `tests.test_core.noisy` objective, `workers=1`, root seed 0,
   and checkpoints `(3, 9)`, configurations `x = [0, 5, 10]` use 15 observations;
   configurations `x = [10, 5, 0]` use 27. Both select `x = 0`. This is a diagnostic
   of avoidable work in this case, not an assertion that every permutation should
   produce an identical asynchronous search trajectory.

2. **The primary pruning test cancels the noise exactly.** Its loss is
   `x**2 + shared_shock`. Paired differences therefore have zero random variation.
   This tests the mechanics of pruning but does not test uncertain rankings,
   unequal variances, or accidentally eliminating the best candidate.

3. **Concurrency shrinks with the surviving candidate count.** `Study` permits
   only one in-flight replication per candidate. With one survivor, at most one
   worker can be evaluating an objective. This is a code-level concurrency bound;
   its wall-time impact still needs measurement on an expensive workload.

4. **Experiment evidence is incomplete.** `audit()` reports a mean and standard
   error for a locked winner, but does not compare algorithms across independent
   optimization runs or implement the planned non-inferiority analysis.

5. **Long experiments are not recoverable.** Results are exported after
   optimization. There is no durable progress store, restart protocol, bounded
   infrastructure retry, or complete attempt-cost accounting.

## Approaches considered

| Approach | Value | Trade-off |
| --- | --- | --- |
| Scheduler diagnostics first — recommended | Directly investigates the project's central speed/quality claim and a reproducible inefficiency | Initial results are synthetic; a financial experiment must follow |
| Durable runtime first | Strong engineering demonstration and safer long experiments | Does not establish that the allocation rule is useful |
| Distributed execution first | Enables larger deployments | Adds operational complexity before there is evidence that more machines address the limiting factor |

## First milestone: scheduler diagnostics and missed comparisons

### Scope and compatibility

Preserve the public `Study`, `FullBudget`, `ReplicationRace`, and `audit` interfaces,
scalar-loss contract, search/audit seed separation, and maximum replication budgets.
Keep the runtime dependency-free. Benchmark and plotting dependencies, if needed,
belong in an optional extra.

Extract checkpoint decision logic from the executor into a small internal policy
module so the same observation sequence can be tested without process timing.
The coordinator owns trial state and calls the policy after accepted observations;
workers continue to evaluate objectives and return observations.

Reconsider unresolved comparisons when peers acquire the matching checkpoint
prefix. A candidate must be able to use a better eligible peer that arrived after
its own checkpoint. Use losses from the same replication indices, never compare
unequal prefixes, and exclude failed or pruned peers. Each decision records the
checkpoint, candidate, comparator, supporting indices, and statistics. Suppress
identical decisions when their supporting observations have not changed.

A pruning decision prevents subsequent dispatches. Work already submitted may
finish and must remain in the observation and cost record; it must not reactivate
a pruned candidate. Completed candidates remain terminal. The implementation plan
must make decision application and in-flight state transitions explicit.

Reconsideration changes adaptive testing behavior, so retain the heuristic label.
Do not reinterpret the standard-error margin as a confidence guarantee.

### Reproducible diagnostic runner

Provide a command that runs full-budget search and replication racing on the same
candidate pool and seed schedule. The initial suite contains:

- The existing additive shared-noise case as an easy control.
- Unequal noise variances and configuration-specific residual noise, so paired
  comparisons remain genuinely random.
- Near ties, where aggressive pruning can discard a competitive candidate.
- A finite-variance heavy-tail case with analytically known expected loss.
- Candidate-order permutations, plus shared versus independent scenarios.

Keep expected losses known analytically. For every selected candidate, compute
regret against the best member of the frozen pool, separately from regret against
a continuous optimum where one exists. Record if a true pool optimum was pruned,
with ties handled explicitly. A final winner can have low regret even when one of
several equivalent optima was pruned; retain both diagnostics.

Use a fixed default schedule of 20 independent optimizer seeds for development.
This is a diagnostic default, not a power calculation for a financial claim.
Methods are paired by seed and pool. Treat order permutations within a seed as
sensitivity checks, not additional independent experimental runs. Publish the
per-run records and summaries even when racing loses.

The generated report contains replication use versus true regret, the distribution
of paired regret differences, and order sensitivity. Store observed wall time and
objective CPU time, but clearly label this cheap suite as unsuitable for practical
speedup claims. No artificial delays appear in performance evidence.

### Artifacts and failure behavior

Each run writes an immutable settings manifest and an individual result file;
the aggregate report is regenerated from those files. Include code revision, dirty
working-tree status, dependency versions, worker count, seed schedule, workload
parameters, and scheduler settings. Prevent accidental overwrite of a different
experiment under the same output identifier.

Represent failed runs explicitly. A method without a completed winner has no
regret observation; report its failure count rather than dropping it silently or
substituting a favorable score. Existing objective exceptions remain recorded
trial failures. Infrastructure failures remain explicit until the recovery
milestone is implemented.

Likely implementation locations:

- `src/aster_hpo/core.py`: coordinator integration and cost/state handling.
- `src/aster_hpo/schedulers.py`: internal checkpoint policy and decision records.
- `benchmarks/`: deterministic workload definitions, runner, and report generation.
- `tests/test_core.py` and new policy/benchmark tests: invariants and diagnostics.
- `docs/BENCHMARKING.md`: runnable commands and interpretation of generated output.

### Acceptance criteria

1. Existing public API tests continue to pass, including full-budget equality
   between serial and parallel objective observations.
2. Both serial orders in the three-candidate diagnostic prune the clearly worse
   candidates at the first usable checkpoint and use 15 observations.
3. A deterministic completion-sequence test demonstrates that newly available
   comparator data can trigger pruning; parallel tests check budgets, no new work
   after pruning, and accounting for already submitted evaluations.
4. Tests cover actual variance in paired differences, no-peer checkpoints,
   better-peer arrival, duplicate decision suppression, and peer failure.
5. Serial reruns reproduce configurations, seeds, losses, decisions, and selected
   candidates. Timing fields are excluded from equality checks. Parallel runs
   retain documented completion-order sensitivity.
6. One documented command generates all per-run evidence and a readable report
   for the default diagnostic suite. Report calculations are checked against
   hand-computable cases, including failures and tied optima.
7. The report displays adverse outcomes and makes no quality-preservation or
   practical-speed claim unsupported by these experiments.

### Explicit boundaries

This first milestone does not add a cluster executor, alter the financial
simulation model, or establish a formal sequential guarantee. It also does not
remove the one-in-flight-replication limit. Changing that limit needs a separate
design for checkpoint boundaries, out-of-order observations, and wasted work.

## Subsequent milestones

### Reliable, observable execution

Implement coordinator-owned SQLite storage with unique replication and attempt
identities, transactional result acceptance, and compatible-manifest checks on
resume. Retrying an infrastructure failure must reuse the original seed; numerical
failure must follow a declared policy. Demonstrate coordinator interruption,
resume, duplicate rejection, and bounded retry with meaningful integration tests.

Record submission/start/completion events and numerical-library thread settings.
Measure worker utilization and scheduler overhead. Then evaluate bounded multiple
replications per surviving candidate, without dispatching beyond the next decision
boundary. Report both saved time and any extra evaluations.

### CPU financial pilot and established baselines

Implement the small train-and-evaluate hedging workload already described in
`DESIGN.md`: one underlying, a European call, proportional transaction costs,
fixed simulation fidelity, and a small trainable policy. Tune training
hyperparameters rather than changing the market assumptions. Check cash accounting,
transaction costs, seed alignment, and a delta-hedge reference before using the
simulator for HPO evidence.

Profile one complete training replication before choosing campaign size. Add an
ASHA comparator with documented rung and promotion semantics, and validate it
against an established implementation on controlled cases. Use Optuna TPE as a
separately labeled whole-method comparison. Freeze baseline versions and settings.

### Controlled study and portfolio release

Freeze the final protocol after disjoint pilots. Compare serial full budget,
parallel full budget, ASHA, and replication racing using equal candidate pools
where applicable. Evaluate scaling at 1, 2, 4, and 8 workers where supported, and
separate fixed-workload parallel speedup from adaptive compute savings.

Lock winners before auditing on fresh training and evaluation randomness. Analyze
paired differences across independent optimization runs, respecting nested
replications. Prespecify the quality margin and fixed-sample uncertainty analysis.
Preservation is supported only when the one-sided upper bound on excess expected
loss is below that margin; an inconclusive result remains inconclusive.

Publish raw evidence, a short technical report, and three central figures: quality
versus time, compute versus quality, and scaling versus worker count. Include a
worker timeline and a demonstrated interrupted-run recovery for engineering depth.
Populate resume claims only with measured results and their named baseline.

## Research context

- [Optuna parallelization documentation](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html): established support for parallel and multi-node optimization.
- [A System for Massively Parallel Hyperparameter Tuning](https://arxiv.org/abs/1810.05934): the asynchronous successive-halving comparator.
- [Bayesian Optimization Allowing for Common Random Numbers](https://arxiv.org/abs/1910.09259): prior work on reusing scenarios in noisy optimization.
- [Deep Hedging](https://arxiv.org/abs/1802.03042): motivation for a financial workload with market frictions.

These sources motivate the comparisons and workload; this proposal does not claim
their algorithms as original contributions.
