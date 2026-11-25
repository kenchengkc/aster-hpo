# Optional native accounting library

This C++20 library calculates profit and loss, trading costs, and turnover in
64-bit floating point. Use `hedge_pnl` for a batch of price paths or `Ledger`
for one trade at a time. The core Python library runs without a compiler.

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
one interval. You can pass empty batches and read-only inputs. The library
rejects arrays with the wrong data type or memory layout. Cost rates must be
finite and nonnegative. Holdings can be long or short.

`hedge_pnl` allocates three NumPy-owned output arrays. `hedge_pnl_into` accepts
preallocated writable arrays. Each output needs its own storage, separate from
the inputs and other outputs. Both functions leave the inputs unchanged and
release the Python interpreter lock during the calculation. Wait for the call to
finish before changing any array it uses. If arithmetic overflows, discard the
outputs because the function may have written some rows.

Successful C++ ledger updates allocate no heap memory. Calling the ledger from
Python adds interpreter overhead and creates Python objects for return values. An invalid event
leaves the ledger unchanged. `finish` closes the position and charges the final
trading cost once. Call `reset(initial_price, cost_rate)` before reusing the ledger.

## Standalone build

Requires a C++20 compiler and CMake 3.20+. Python is optional.

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j
ctest --test-dir build/native --output-on-failure
```

Add `-DASTER_ENABLE_SANITIZERS=ON` in a separate build to check for memory
errors and undefined behavior. Use a release build for timings and run the
profiler and sanitizers separately. The builds disable fast-math and floating-point
contraction. In Python, `build_info()` reports the compiler, build type, and flags.

The benchmark times the accounting calculation in isolation. The guides describe
what the CPU and GPU timers include. [Native design](../docs/superpowers/specs/2026-11-04-native-performance-design.md) · [CUDA test package](../cuda/README.md)
