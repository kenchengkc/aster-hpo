# Reproducing the experiments

Run research commands from the repository checkout. The wheel contains `aster_hpo`;
`benchmarks` and `examples` are repository research modules.
Raw evidence archives are versioned in the repository and excluded from package
distributions to keep installation small.

```bash
python -m pip install -e '.[dev,benchmark]'
# Optional: match the archived numerical/reporting versions on supported Python/hardware.
python -m pip install -r requirements-research.txt
```

The [September CPU report](RESULTS.md) contains measured findings, the exact frozen
protocol, and compressed raw records. Numerical seeds are reproducible; elapsed times
and asynchronous decision order depend on the machine and scheduling conditions.

## Regenerate the archived report without rerunning optimization

```bash
mkdir -p results/reproduced
# Verify the downloaded/checked-out evidence before extracting it:
(cd docs/evidence/cpu-2026-09-24 && shasum -a 256 -c SHA256SUMS)
tar -xzf docs/evidence/cpu-2026-09-24/financial-artifacts.tar.gz -C results/reproduced
python -m benchmarks.study report --output results/reproduced/financial-study --plots
```

The archive contains the frozen protocol, all search/audit observations and seeds,
locked configurations, per-replication financial diagnostics, fixed-workload scaling,
the disjoint pilot, and the original pre-correction analysis. SQLite databases and
lock files are excluded; exported observations, attempts, and events are sufficient
for report regeneration, but the archive is not a resumable execution database.

The report command validates required run identities, audit identities, winner locks,
and protocol digests. Missing data fail reporting instead of silently reducing the
sample. Checksums detect changed archive bytes; they are not third-party attestation.

## Run a new frozen financial comparison

```bash
# Disjoint pilot: size the workload and choose the quality margin before final runs.
python -m examples.hedging --output results/new-pilot --seed 10000 \
  --workers 2 --configs 6 --epochs 256 --batch-paths 256
python -m benchmarks.study freeze --pilot results/new-pilot \
  --protocol results/new-protocol.json --seeds 20 --workers 4 --configs 6 --audits 8
python -m benchmarks.study run --protocol results/new-protocol.json \
  --output results/new-study
python -m benchmarks.study report --output results/new-study --plots
```

Execution refuses an existing output directory by default. To continue an interrupted
comparison on the same source and compatible environment:

```bash
python -m benchmarks.study run --protocol results/new-protocol.json \
  --output results/new-study --resume
```

Freeze records a hash of runtime and benchmark source, market settings, seeds,
candidate count, checkpoints, worker/thread allocation, audit budget, pilot-derived
margin, and analysis. Changing source requires a new protocol. Resume also checks
Python/platform/CPU and NumPy/SciPy/threadpoolctl versions. Reporting archived data
is allowed from newer analysis code; record and disclose any change in interpretation.

The archived study was executed at `bef29bf`; reporting was corrected at `89e75c6`
for zero sample variance. Running that original protocol against later source is
intentionally rejected. To repeat its original execution code, use that revision in
a separate checkout; for current behavior, freeze a new protocol with the same
settings and retain both datasets.

## Comparison and inference contract

The four methods are serial full budget, parallel full budget, parallel stopping-ASHA,
and parallel replication racing. Each optimizer seed supplies the same candidate
pool to all methods; method execution order is randomized within seed. Parallel
methods use equal worker and native-thread limits. The winner is locked before
an audit that uses a disjoint seed namespace. All audit costs are recorded separately.

There is no extra finalist-selection phase: choose the lowest observed mean among
candidates that reach the common cap. Each audit replication trains a fresh policy
and evaluates it on fresh paths. Matched audit seeds across methods permit pairing;
nested training runs and market paths are not extra independent optimizer runs.

For the primary comparison, form one mean audit difference per optimizer seed and
calculate a fixed-sample, one-sided 95% Student t bound. The margin is 1% of the
disjoint pilot reference MSE. The approximation requires independent optimizer runs
and an adequate distributional approximation for their mean differences. Missing
campaigns, fewer than two independent runs, and zero sample variance cannot support
a positive non-inferiority conclusion. A numerical failure is recorded, not replaced
with zero or excluded from the campaign denominator. Other method comparisons are
descriptive and have no multiplicity adjustment.

Search-time speedup is a paired ratio within each seed, then summarized by its
median. The 95% percentile bootstrap resamples whole pairs 2,000 times with a fixed
analysis seed. It describes variability across this finite seed set, not machines
or market families. These are fixed-budget endpoints, not time-to-target curves,
equal-wall-budget experiments, or evidence of a sequential pruning guarantee.

## Synthetic allocation diagnostics

```bash
python -m benchmarks.diagnostics --output results/new-diagnostics --seeds 20 --plots
# Or regenerate the archived 640-run diagnostic:
tar -xzf docs/evidence/cpu-2026-09-24/diagnostic-artifacts.tar.gz -C results/reproduced
python -m benchmarks.diagnostics --output results/reproduced/diagnostics-v02 \
  --report-only --plots
```

Four noise models × 20 seeds × shared/independent scenarios × two candidate orders ×
two methods produce 640 campaigns. Each candidate pool has 12 configurations and
checkpoints `(3, 9, 27)`. Reports include analytical pool regret, continuous regret,
replication counts, pruned optima, failures, and paired differences. Order permutations
are separate sensitivity groups, not additional independent seeds. These cheap
objectives diagnose allocation behavior; their timings do not establish practical speedup.

## Costs, artifacts, and remaining work

Search wall time includes process startup, journal writes, and idle capacity. Returned
objective CPU time excludes coordinator/startup overhead and cannot recover CPU cost
from killed attempts. Allocated worker-seconds equal active-session wall time times
workers; they are capacity, not measured CPU consumption. Audit wall/CPU costs are
separate. Worker timelines expose start/finish times; gaps do not by themselves explain
why a worker was idle.

Fixed-workload scaling uses a separate seed set, 12 candidates, three replications,
and no pruning, so increasing workers does not change algorithmic work. The archived
sweep has only two seeds and uses increasing worker counts within seed; it is a
small scaling demonstration, not a randomized hardware study.

Full Ray Tune and Optuna TPE comparisons, equal-budget studies, target-attainment
curves, and larger financial stress tests remain on the [roadmap](../ROADMAP.md).
The [baseline note](BASELINES.md) defines exactly what the current Ray compatibility
test establishes.
