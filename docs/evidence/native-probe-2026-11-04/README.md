# Hedging performance before the C++ implementation

This profile shows where the Python/NumPy hedging objective spent its time at
revision `065020f`, before the C++ work began.

The run used one numerical thread on an Apple M5 (arm64), with 256 training epochs,
256 paths per training batch, 2,048 evaluation paths, and 12 trading intervals.
The three networks had widths of 4, 8, and 16. All used a learning rate of 0.03
and regularization of 0.0001. Each width used its own replication with root seed
1000, index 0, phase `pilot`, and shared scenarios. The timed objective wrote no
diagnostics. This run measured execution time only and included no independent
audit of financial results.

After one warm-up per width, each network ran three times on identical inputs.
Median times were about 79 ms for width 4, 89 ms for width 8, and 129 ms for
width 16. All three runs at each width returned the same loss. `summary.json`
contains every duration and the replication details. Three samples per width are
too few to estimate 99th percentile latency.

In a separate cProfile run at width 8, `hedge_pnl` accounted for about 6.3% of
cumulative time and the forward pass for 38.7%. Computing loss and gradients
took 79.7%, including time in the forward pass and accounting. Adding the three
percentages would count some work twice. The profiler adds overhead.
`profile.txt` lists the calls with directory names stripped. The run did not count
memory allocations.

To repeat the profile, install the benchmark dependencies and run this from the
repository root.

```bash
python - <<'PY'
import cProfile
import pstats
import statistics
import time
from threadpoolctl import threadpool_limits
from aster_hpo import Replication
from benchmarks.hedging import HedgingObjective, Market

objective = HedgingObjective(Market(epochs=256, batch_paths=256, evaluation_paths=2048))
with threadpool_limits(limits=1):
    for width in (4, 8, 16):
        config = {"width": width, "learning_rate": 0.03, "regularization": 0.0001}
        rep = Replication.create(1000, config, 0, "pilot", True)
        objective(config, rep)
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            objective(config, rep)
            durations.append(time.perf_counter() - start)
        print(width, durations, statistics.median(durations))
    config = {"width": 8, "learning_rate": 0.03, "regularization": 0.0001}
    rep = Replication.create(1000, config, 0, "pilot", True)
    profile = cProfile.Profile()
    profile.runcall(objective, config, rep)
pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(25)
PY
```

`SHA256SUMS` lists the checksums for the saved records. The design describes
which parts of the workload the later CPU and GPU benchmarks measure.
[Native design](../../superpowers/specs/2026-11-04-native-performance-design.md)
