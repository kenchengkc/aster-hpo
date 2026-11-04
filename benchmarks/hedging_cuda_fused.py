"""Launch-efficient float64 CUDA training with per-invocation scratch storage.

Importing this module does not import CuPy or create a CUDA context. Kernels are
cached by device; live input/output pointers are owned only by caller/workspace.
"""

import math
from functools import cache
from pathlib import Path

import numpy as np

_HEADER = Path(__file__).resolve().parents[1] / "cuda/include/hedging_training.cuh"
_NAMES = ("w1", "b1", "w2", "b2")
_THREADS = 128


@cache
def _kernels(device_id):
    from .hedging_cuda import _cupy

    cp = _cupy()
    module = cp.RawModule(
        code=_HEADER.read_text(),
        options=(
            "--std=c++17",
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ),
    )
    return tuple(
        module.get_function(name)
        for name in (
            "hedging_forward",
            "hedging_error",
            "hedging_gradient",
            "hedging_adam",
        )
    )


class TrainingWorkspace:
    """Reusable buffers for exactly one training shape on the current device.

    ``compute`` is the validated training hot path: it performs no host transfers.
    Returned arrays alias this workspace and are overwritten by the next call.
    """

    def __init__(self, cp, paths_count, steps, width):
        for name, value in (("paths_count", paths_count), ("steps", steps), ("width", width)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.cp = cp
        self.device_id = cp.cuda.runtime.getDevice()
        self.paths_count, self.steps, self.width = paths_count, steps, width
        self.hidden = cp.empty((paths_count, steps, width), dtype=cp.float64)
        self.holdings = cp.empty((paths_count, steps), dtype=cp.float64)
        self.errors = cp.empty(paths_count, dtype=cp.float64)
        self.features0 = cp.empty((paths_count, steps), dtype=cp.float64)
        self.gradients = {
            "w1": cp.empty((2, width), dtype=cp.float64),
            "b1": cp.empty(width, dtype=cp.float64),
            "w2": cp.empty(width, dtype=cp.float64),
            "b2": cp.empty(1, dtype=cp.float64),
        }
        self.loss = cp.empty((), dtype=cp.float64)
        self._functions = _kernels(self.device_id)

    def compute(self, parameters, paths, market, regularization):
        from .hedging import call_price

        cp = self.cp
        forward, error, gradient, _ = self._functions
        count, steps, width = self.paths_count, self.steps, self.width
        dimensions = (np.uint64(count), np.uint64(steps), np.uint64(width))
        stream = cp.cuda.get_current_stream()
        forward(
            ((count * steps + _THREADS - 1) // _THREADS,),
            (_THREADS,),
            (
                paths,
                *(parameters[name] for name in _NAMES),
                *dimensions,
                np.float64(market.strike),
                np.float64(max(market.volatility * math.sqrt(market.maturity), 1e-3)),
                self.features0,
                self.hidden,
                self.holdings,
            ),
            stream=stream,
        )
        error(
            ((count + _THREADS - 1) // _THREADS,),
            (_THREADS,),
            (
                paths,
                self.holdings,
                *dimensions[:2],
                np.float64(market.transaction_cost),
                np.float64(call_price(market)),
                np.float64(market.strike),
                self.errors,
            ),
            stream=stream,
        )
        gradient(
            (4 * width + 2,),
            (_THREADS,),
            (
                paths,
                parameters["w1"],
                parameters["w2"],
                self.features0,
                self.hidden,
                self.holdings,
                self.errors,
                *dimensions,
                np.float64(market.transaction_cost),
                np.float64(regularization),
                *(self.gradients[name] for name in _NAMES),
                self.loss,
            ),
            stream=stream,
        )
        return self.loss, self.gradients


def loss_and_gradient(parameters, paths, market, *, regularization, workspace=None):
    """Checked public loss/gradient operation; validation reads one device flag."""
    from .hedging_cuda import _cupy, _validate_policy

    cp = _cupy()
    _validate_policy(cp, parameters, paths, market)
    if not math.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    width = parameters["w2"].shape[0]
    if workspace is None:
        workspace = TrainingWorkspace(cp, paths.shape[0], market.steps, width)
    if (
        not isinstance(workspace, TrainingWorkspace)
        or workspace.device_id != cp.cuda.runtime.getDevice()
        or (workspace.paths_count, workspace.steps, workspace.width)
        != (paths.shape[0], market.steps, width)
    ):
        raise ValueError("Workspace must match policy shape and current CUDA device")
    return workspace.compute(parameters, paths, market, regularization)


def update(parameters, gradients, first, second, loss, finite, *, learning_rate, epoch):
    """Update validated arrays in place with one norm/finite/clipping/Adam launch."""
    from .hedging_cuda import _cupy

    cp = _cupy()
    kernel = _kernels(cp.cuda.runtime.getDevice())[3]
    kernel(
        (1,),
        (_THREADS,),
        (
            *(parameters[name] for name in _NAMES),
            *(gradients[name] for name in _NAMES),
            *(first[name] for name in _NAMES),
            *(second[name] for name in _NAMES),
            loss,
            finite,
            np.uint64(parameters["w2"].shape[0]),
            np.float64(learning_rate),
            np.float64(1 - 0.9**epoch),
            np.float64(1 - 0.999**epoch),
        ),
        stream=cp.cuda.get_current_stream(),
    )
