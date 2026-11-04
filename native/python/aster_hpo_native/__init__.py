"""Strict NumPy interfaces to the optional C++ accounting library."""

import math
from numbers import Real

import numpy as np

from . import _core
from ._core import Accounting, Ledger, build_info

__version__ = _core.__version__
__all__ = ["Accounting", "Ledger", "build_info", "hedge_pnl", "hedge_pnl_into"]


def _array(value, dimensions, name, *, writable=False):
    if not isinstance(value, np.ndarray) or value.dtype != np.float64:
        raise TypeError(f"{name} must be a NumPy float64 array")
    if value.ndim != dimensions or not value.flags.c_contiguous or not value.flags.aligned:
        raise ValueError(f"{name} must be aligned, C-contiguous, and {dimensions}-dimensional")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _inputs(prices, holdings, cost_rate):
    prices = _array(prices, 2, "prices")
    holdings = _array(holdings, 2, "holdings")
    if holdings.shape[1] < 1 or prices.shape != (holdings.shape[0], holdings.shape[1] + 1):
        raise ValueError(
            "prices must have shape (paths, intervals + 1), holdings (paths, intervals)"
        )
    if isinstance(cost_rate, (bool, np.bool_)) or not isinstance(cost_rate, Real):
        raise TypeError("cost_rate must be a real number")
    cost_rate = float(cost_rate)
    if not math.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError("cost_rate must be finite and nonnegative")
    return prices, holdings, cost_rate


def hedge_pnl(prices, holdings, *, cost_rate):
    """Return NumPy-owned (pnl, costs, turnover) arrays for finite float64 inputs."""
    prices, holdings, cost_rate = _inputs(prices, holdings, cost_rate)
    outputs = tuple(np.empty(prices.shape[0], dtype=np.float64) for _ in range(3))
    _core._hedge_pnl_into(prices, holdings, *outputs, cost_rate)
    return outputs


def hedge_pnl_into(prices, holdings, pnl, costs, turnover, *, cost_rate):
    """Write preallocated outputs. All outputs must be disjoint from other buffers.

    Inputs remain unchanged. Do not mutate any buffer concurrently with this call.
    Discard outputs after an arithmetic error: earlier rows may have been written.
    """
    prices, holdings, cost_rate = _inputs(prices, holdings, cost_rate)
    outputs = tuple(
        _array(value, 1, name, writable=True)
        for value, name in zip((pnl, costs, turnover), ("pnl", "costs", "turnover"), strict=True)
    )
    if any(value.shape != (prices.shape[0],) for value in outputs):
        raise ValueError("Each output must have one element per path")
    _core._hedge_pnl_into(prices, holdings, *outputs, cost_rate)
