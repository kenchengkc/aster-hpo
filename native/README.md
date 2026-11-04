# Optional native accounting library

A C++20 float64 batch kernel and streaming ledger. The Python core remains usable
without this companion package or a compiler.

```bash
python -m pip install ./native
python -m pytest native/tests/test_python.py -q
```

```python
import numpy as np
from aster_hpo_native import Ledger, hedge_pnl

prices = np.array([[1.0, 1.1, 1.2]])
holdings = np.array([[0.5, 0.5]])
pnl, costs, turnover = hedge_pnl(prices, holdings, cost_rate=0.01)
# P&L 0.089; costs 0.011; turnover 1.1, including opening and liquidation.

ledger = Ledger(initial_price=1.0, cost_rate=0.01)
ledger.update(1.0, 0.5)
ledger.update(1.1, 0.5)
result = ledger.finish(1.2)
```

Inputs must be aligned, C-contiguous NumPy float64 arrays with finite values. Prices
have shape `(paths, intervals + 1)` and holdings `(paths, intervals)`, with at least
one interval. Empty path batches are supported. Read-only inputs are accepted;
implicit dtype/layout conversion is rejected. Cost rates must be finite and
nonnegative. Long and short holdings are supported.

`hedge_pnl` allocates three NumPy-owned output arrays. `hedge_pnl_into` accepts
preallocated writable arrays; outputs must not overlap any other supplied buffer.
Inputs are not modified. Do not mutate arrays concurrently with a native call;
the implementation releases the interpreter lock during native work. Discard
outputs after arithmetic overflow because earlier rows may have been written.

The native ledger performs no heap allocation on successful updates. Python calls
and returned Python objects have their own overhead. Invalid events leave ledger
state intact. `finish` closes the ledger and charges terminal liquidation once;
call `reset(initial_price, cost_rate)` before reuse.

## Standalone build

Requires a C++20 compiler and CMake 3.20+. Python is optional.

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j
ctest --test-dir build/native --output-on-failure
```

Add `-DASTER_ENABLE_SANITIZERS=ON` in a separate build for address/undefined behavior
checks. Profiling and sanitizer builds are distinct from performance measurements.
Both builds disable fast-math and floating-point contraction. `build_info()` exposes
compiler, build type, and flags for the Python extension. No whole-search speedup
or trading latency claim follows from a kernel benchmark.

See [the design](../docs/superpowers/specs/2026-11-04-native-performance-design.md)
and [CUDA test package](../cuda/README.md) for measurement boundaries.
