"""Optional float64 resident CUDA implementation of the NumPy hedging objective.

NumPy prepares identical initialization and random shocks once per training call.
Simulation, explicit backpropagation, Adam, and accounting then run on device.
Importing this module does not import CuPy or initialize a CUDA context.
"""

import math
from functools import cache, lru_cache
from pathlib import Path

import numpy as np

_HEADER = Path(__file__).resolve().parents[1] / "cuda/include/accounting_kernel.cuh"


@lru_cache(maxsize=1)
def _cupy():
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA devices were found")
    except Exception as exc:
        raise RuntimeError(
            "CUDA hedging requires a compatible CuPy installation and an available CUDA device"
        ) from exc
    return cp


@cache
def _kernel(device_id):
    # RawKernel uses the current device and stream; cache separately per device.
    cp = _cupy()
    return cp.RawKernel(
        _HEADER.read_text(),
        "accounting_kernel",
        options=(
            "--std=c++17",
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ),
    )


def synchronize():
    """Complete work on the current CuPy stream before a host timing boundary."""
    _cupy().cuda.get_current_stream().synchronize()


def device_info():
    """Return stable backend identity, without changing memory or clock statistics."""
    cp = _cupy()
    device = cp.cuda.runtime.getDevice()
    properties = cp.cuda.runtime.getDeviceProperties(device)
    name = properties["name"]
    return {
        "backend": "cuda",
        "cupy_version": cp.__version__,
        "numpy_version": np.__version__,
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
        "device_id": device,
        "device_name": name.decode() if isinstance(name, bytes) else name,
        "compute_capability": f"{properties['major']}.{properties['minor']}",
    }


def _array(cp, value, name, *, ndim):
    if not isinstance(value, cp.ndarray):
        raise TypeError(f"{name} must be a CuPy device array")
    if value.dtype != cp.float64:
        raise TypeError(f"{name} must have float64 dtype")
    if value.ndim != ndim or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be {ndim}-dimensional and C contiguous")
    if value.device.id != cp.cuda.runtime.getDevice():
        raise ValueError(f"{name} must reside on the current CUDA device")


def _finite(cp, values):
    flag = cp.ones((), dtype=cp.bool_)
    for value in values:
        flag &= cp.isfinite(value).all()
    return flag


def _accounting(cp, paths, holdings, cost_rate):
    count, columns = paths.shape
    outputs = tuple(cp.empty(count, dtype=cp.float64) for _ in range(3))
    if count:
        _kernel(cp.cuda.runtime.getDevice())(
            ((count + 127) // 128,),
            (128,),
            (
                paths,
                holdings,
                np.uint64(count),
                np.uint64(columns - 1),
                np.float64(cost_rate),
                *outputs,
            ),
            stream=cp.cuda.get_current_stream(),
        )
    return outputs


def accounting(paths, holdings, *, cost_rate):
    """Account for opening, rebalancing, and liquidation on resident device arrays."""
    cp = _cupy()
    _array(cp, paths, "paths", ndim=2)
    _array(cp, holdings, "holdings", ndim=2)
    if paths.shape[1] < 2 or holdings.shape != (paths.shape[0], paths.shape[1] - 1):
        raise ValueError("Holdings must have one value per path and trading interval")
    if not math.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError("cost_rate must be finite and nonnegative")
    if not bool(_finite(cp, (paths, holdings))):
        raise ValueError("Accounting inputs must be finite")
    return _accounting(cp, paths, holdings, cost_rate)


def _validate_policy(cp, parameters, paths, market):
    _array(cp, paths, "paths", ndim=2)
    if paths.shape[0] < 1 or paths.shape[1] != market.steps + 1:
        raise ValueError("Policy paths must be nonempty with market.steps + 1 prices")
    if set(parameters) != {"w1", "b1", "w2", "b2"}:
        raise ValueError("Policy parameters must contain w1, b1, w2, and b2")
    for name, value in parameters.items():
        _array(cp, value, name, ndim=2 if name == "w1" else 1)
    width = parameters["w2"].shape[0]
    shapes = {"w1": (2, width), "b1": (width,), "w2": (width,), "b2": (1,)}
    if width < 1 or any(parameters[name].shape != shape for name, shape in shapes.items()):
        raise ValueError("Policy parameter shapes are inconsistent")
    if not bool(_finite(cp, (paths, *parameters.values()))):
        raise ValueError("Policy inputs must be finite")


def _paths_from_shocks(cp, shocks, market):
    dt = market.maturity / market.steps
    returns = -0.5 * market.volatility**2 * dt + market.volatility * math.sqrt(dt) * cp.asarray(
        shocks
    )
    initial = cp.zeros((*returns.shape[:-1], 1), dtype=cp.float64)
    return market.spot * cp.exp(cp.concatenate((initial, returns.cumsum(axis=-1)), axis=-1))


def _forward(cp, parameters, paths, market):
    moneyness = cp.log(paths[:, :-1] / market.strike)
    moneyness /= max(market.volatility * math.sqrt(market.maturity), 1e-3)
    tau = cp.broadcast_to(
        (market.steps - cp.arange(market.steps)) / market.steps,
        moneyness.shape,
    )
    features = cp.stack((moneyness, tau), axis=-1)
    hidden = cp.tanh(features @ parameters["w1"] + parameters["b1"])
    logits = hidden @ parameters["w2"] + parameters["b2"]
    # No boolean gather: its data-dependent allocation could synchronize each epoch.
    exponent = cp.exp(-cp.abs(logits))
    holdings = cp.where(logits >= 0, 1 / (1 + exponent), exponent / (1 + exponent))
    return features, hidden, holdings


def positions(parameters, paths, market):
    cp = _cupy()
    _validate_policy(cp, parameters, paths, market)
    return _forward(cp, parameters, paths, market)[2]


def _loss_and_gradient(cp, parameters, paths, market, regularization):
    from .hedging import call_price

    features, hidden, holdings = _forward(cp, parameters, paths, market)
    pnl, _, _ = _accounting(cp, paths, holdings, market.transaction_cost)
    error = call_price(market) + pnl - cp.maximum(paths[:, -1] - market.strike, 0)
    zeros = cp.zeros((len(paths), 1), dtype=cp.float64)
    trades = cp.diff(cp.concatenate((zeros, holdings, zeros), axis=1), axis=1)
    cost_gradient = market.transaction_cost * (
        paths[:, :-1] * cp.sign(trades[:, :-1]) - paths[:, 1:] * cp.sign(trades[:, 1:])
    )
    gradient_h = 2 * error[:, None] * (cp.diff(paths, axis=1) - cost_gradient) / len(paths)
    gradient_z = gradient_h * holdings * (1 - holdings)
    gradient_hidden = gradient_z[:, :, None] * parameters["w2"] * (1 - hidden**2)
    gradients = {
        "w2": cp.sum(hidden * gradient_z[:, :, None], axis=(0, 1)),
        "b2": gradient_z.sum().reshape(1),
        "w1": features.reshape(-1, 2).T @ gradient_hidden.reshape(-1, hidden.shape[-1]),
        "b1": gradient_hidden.sum(axis=(0, 1)),
    }
    loss = cp.mean(error**2)
    for name in ("w1", "w2"):
        loss += regularization * cp.sum(parameters[name] ** 2)
        gradients[name] += 2 * regularization * parameters[name]
    return loss, gradients


def loss_and_gradient(parameters, paths, market, *, regularization):
    """Return a device scalar loss and device gradients; no autograd is required."""
    cp = _cupy()
    _validate_policy(cp, parameters, paths, market)
    if not math.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    return _loss_and_gradient(cp, parameters, paths, market, regularization)


def train_policy(config, replication, market):
    from .hedging import initialize, validate_config

    width, learning_rate, regularization = validate_config(config)
    cp = _cupy()
    initialization, training = np.random.SeedSequence(replication.training_seed).spawn(2)
    parameters = {
        name: cp.asarray(value)
        for name, value in initialize(width, np.random.default_rng(initialization)).items()
    }
    # Bulk host RNG preparation preserves the reference stream and is timed.
    shocks = np.random.default_rng(training).standard_normal(
        (market.epochs, market.batch_paths, market.steps),
    )
    all_paths = _paths_from_shocks(cp, shocks, market)
    first = {name: cp.zeros_like(value) for name, value in parameters.items()}
    second = {name: cp.zeros_like(value) for name, value in parameters.items()}
    finite = cp.ones((), dtype=cp.bool_)
    for epoch in range(1, market.epochs + 1):
        loss, gradients = _loss_and_gradient(
            cp,
            parameters,
            all_paths[epoch - 1],
            market,
            regularization,
        )
        finite &= _finite(cp, (loss, *gradients.values()))
        norm = cp.sqrt(sum(cp.sum(gradient**2) for gradient in gradients.values()))
        for name, gradient in gradients.items():
            gradient = gradient / cp.maximum(norm, 1)
            first[name] = 0.9 * first[name] + 0.1 * gradient
            second[name] = 0.999 * second[name] + 0.001 * gradient**2
            m = first[name] / (1 - 0.9**epoch)
            v = second[name] / (1 - 0.999**epoch)
            parameters[name] -= learning_rate * m / (cp.sqrt(v) + 1e-8)
    finite &= _finite(cp, parameters.values())
    # The only training scalar transfer occurs after all epochs finish.
    if not bool(finite):
        raise FloatingPointError("Non-finite training outcome")
    return parameters


def evaluate_policy(parameters, paths, market):
    from .hedging import call_price

    cp = _cupy()
    _validate_policy(cp, parameters, paths, market)
    holdings = _forward(cp, parameters, paths, market)[2]
    pnl, costs, turnover = _accounting(cp, paths, holdings, market.transaction_cost)
    error = call_price(market) + pnl - cp.maximum(paths[:, -1] - market.strike, 0)
    quantiles = cp.quantile(error, cp.asarray([0.05, 0.5, 0.95]))
    names = (
        "mse",
        "bias",
        "error_q05",
        "error_q50",
        "error_q95",
        "mean_transaction_cost",
        "mean_turnover",
    )
    values = cp.asnumpy(
        cp.stack((cp.mean(error**2), cp.mean(error), *quantiles, cp.mean(costs), cp.mean(turnover)))
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite evaluation outcome")
    return dict(zip(names, map(float, values), strict=True))


def evaluate(config, replication, market):
    parameters = train_policy(config, replication, market)
    cp = _cupy()
    shocks = np.random.default_rng(replication.scenario_seed).standard_normal(
        (market.evaluation_paths, market.steps),
    )
    return evaluate_policy(parameters, _paths_from_shocks(cp, shocks, market), market)
