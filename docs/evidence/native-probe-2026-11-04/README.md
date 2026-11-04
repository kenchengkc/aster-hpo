# Native foundation: development profiling probe

This is a small probe of the existing Python/NumPy objective at source revision
`065020f`, before any native implementation. It helps choose the next component;
it is not a speedup comparison or a tail-latency study.

Hardware: Apple M5, arm64. One numerical thread. Workload: 256 training epochs,
256 paths per training batch, 2,048 evaluation paths, and 12 trading intervals.
Configurations use learning rate 0.03, regularization 0.0001, and widths 4/8/16.
Each width has its own deterministic replication from root 1000, index 0,
phase `pilot`, and shared scenarios. No financial comparison or independent audit
was performed. No diagnostics were written inside the measured objective.

After one warm-up per width, three identical-input runs produced median wall times
of about 79, 89, and 129 milliseconds respectively. Repeated losses matched within
each width. The saved `summary.json` contains every duration and replication identity.
These few samples do not support general performance or p99 claims.

A separate width-8 cProfile run produced `profile.txt`, with directory names stripped
for portability. Accounting (`hedge_pnl`) accounts for about 6.3% of cumulative
profile time. Forward computation accounts for 38.7%; loss/gradient computation
accounts for 79.7%, including forward and accounting work. Cumulative entries overlap
and cannot be added. Profiler overhead changes timings. No memory-allocation counts
were measured.

To repeat the procedure from a checkout with the benchmark dependencies installed:

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

Reproduction timings will vary. `SHA256SUMS` verifies the accompanying records.
See the [proposed native design](../../superpowers/specs/2026-11-04-native-performance-design.md)
for the interpretation and separate CPU/GPU measurement boundaries.
