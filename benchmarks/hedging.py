"""Neural hedger in a fixed synthetic market with optional resident CUDA execution.

Zero interest and drift; long European call liability; fixed initial capital;
bounded positions; costs on opening, rebalancing, and terminal liquidation.
This workload is inspired by Deep Hedging, not a reproduction of its full model.
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aster_hpo import Choice, Float, sample_configs

from .common import write_json


@dataclass(frozen=True)
class Market:
    spot: float = 1.0
    strike: float = 1.0
    volatility: float = 0.2
    maturity: float = 1.0
    steps: int = 12
    transaction_cost: float = 0.001
    epochs: int = 64
    batch_paths: int = 128
    evaluation_paths: int = 2048

    def __post_init__(self):
        for name in ("spot", "strike", "maturity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("volatility", "transaction_cost"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("steps", "epochs", "batch_paths", "evaluation_paths"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def simulate_paths(market, count, rng):
    dt = market.maturity / market.steps
    shocks = rng.standard_normal((count, market.steps))
    returns = -0.5 * market.volatility**2 * dt + market.volatility * math.sqrt(dt) * shocks
    initial = np.zeros((count, 1))
    return market.spot * np.exp(np.concatenate([initial, returns.cumsum(axis=1)], axis=1))


def _cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def call_price(market):
    if market.volatility == 0:
        return max(market.spot - market.strike, 0)
    scale = market.volatility * math.sqrt(market.maturity)
    d1 = math.log(market.spot / market.strike) / scale + 0.5 * scale
    return market.spot * _cdf(d1) - market.strike * _cdf(d1 - scale)


def delta_positions(paths, market):
    if market.volatility == 0:
        return (paths[:, :-1] > market.strike).astype(float)
    tau = market.maturity * (market.steps - np.arange(market.steps)) / market.steps
    scale = market.volatility * np.sqrt(tau)
    d1 = np.log(paths[:, :-1] / market.strike) / scale + 0.5 * scale
    return np.fromiter((_cdf(x) for x in d1.flat), float).reshape(d1.shape)


def hedge_pnl(paths, holdings, *, cost_rate):
    if paths.ndim != 2 or holdings.shape != (paths.shape[0], paths.shape[1] - 1):
        raise ValueError("Holdings must have one value per path and trading interval")
    if not math.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError("cost_rate must be finite and nonnegative")
    zeros = np.zeros((len(paths), 1))
    trades = np.diff(np.concatenate([zeros, holdings, zeros], axis=1), axis=1)
    turnover = np.sum(np.abs(trades) * paths, axis=1)
    costs = cost_rate * turnover
    pnl = np.sum(holdings * np.diff(paths, axis=1), axis=1) - costs
    return pnl, costs, turnover


def initialize(width, rng):
    return {
        "w1": rng.normal(0, 1 / math.sqrt(2), (2, width)),
        "b1": np.zeros(width),
        "w2": rng.normal(0, 0.1 / math.sqrt(width), width),
        "b2": np.zeros(1),
    }


def _forward(parameters, paths, market):
    moneyness = np.log(paths[:, :-1] / market.strike)
    moneyness /= max(market.volatility * math.sqrt(market.maturity), 1e-3)
    tau = np.broadcast_to((market.steps - np.arange(market.steps)) / market.steps, moneyness.shape)
    features = np.stack([moneyness, tau], axis=-1)
    hidden = np.tanh(features @ parameters["w1"] + parameters["b1"])
    logits = hidden @ parameters["w2"] + parameters["b2"]
    # Stable logistic without clipping, so analytic gradients remain consistent.
    holdings = np.empty_like(logits)
    positive = logits >= 0
    holdings[positive] = 1 / (1 + np.exp(-logits[positive]))
    negative_exp = np.exp(logits[~positive])
    holdings[~positive] = negative_exp / (1 + negative_exp)
    return features, hidden, holdings


def positions(parameters, paths, market):
    return _forward(parameters, paths, market)[2]


def loss_and_gradient(parameters, paths, market, *, regularization):
    features, hidden, holdings = _forward(parameters, paths, market)
    pnl, _, _ = hedge_pnl(paths, holdings, cost_rate=market.transaction_cost)
    error = call_price(market) + pnl - np.maximum(paths[:, -1] - market.strike, 0)
    zeros = np.zeros((len(paths), 1))
    trades = np.diff(np.concatenate([zeros, holdings, zeros], axis=1), axis=1)
    cost_gradient = market.transaction_cost * (
        paths[:, :-1] * np.sign(trades[:, :-1]) - paths[:, 1:] * np.sign(trades[:, 1:])
    )
    gradient_h = 2 * error[:, None] * (np.diff(paths, axis=1) - cost_gradient) / len(paths)
    gradient_z = gradient_h * holdings * (1 - holdings)
    gradient_hidden = gradient_z[:, :, None] * parameters["w2"] * (1 - hidden**2)
    gradients = {
        "w2": np.sum(hidden * gradient_z[:, :, None], axis=(0, 1)),
        "b2": np.array([gradient_z.sum()]),
        "w1": features.reshape(-1, 2).T @ gradient_hidden.reshape(-1, hidden.shape[-1]),
        "b1": gradient_hidden.sum(axis=(0, 1)),
    }
    loss = float(np.mean(error**2))
    for name in ("w1", "w2"):
        loss += regularization * float(np.sum(parameters[name] ** 2))
        gradients[name] += 2 * regularization * parameters[name]
    return loss, gradients


def validate_backend(backend):
    if not isinstance(backend, str) or backend not in ("numpy", "cuda"):
        raise ValueError("backend must be 'numpy' or 'cuda'")
    return backend


def validate_config(config):
    if set(config) != {"width", "learning_rate", "regularization"}:
        raise ValueError("Tune only width, learning_rate, and regularization")
    width, learning_rate, regularization = (
        config["width"],
        config["learning_rate"],
        config["regularization"],
    )
    if type(width) is not int or width < 1:
        raise ValueError("width must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    return width, learning_rate, regularization


def train_policy(config, replication, market):
    width, learning_rate, regularization = validate_config(config)
    initialization, training = np.random.SeedSequence(replication.training_seed).spawn(2)
    parameters = initialize(width, np.random.default_rng(initialization))
    rng = np.random.default_rng(training)
    first = {name: np.zeros_like(value) for name, value in parameters.items()}
    second = {name: np.zeros_like(value) for name, value in parameters.items()}
    for epoch in range(1, market.epochs + 1):
        paths = simulate_paths(market, market.batch_paths, rng)
        loss, gradients = loss_and_gradient(
            parameters, paths, market, regularization=regularization
        )
        if not math.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients.values()):
            raise FloatingPointError("Non-finite training outcome")
        norm = math.sqrt(sum(float(np.sum(g**2)) for g in gradients.values()))
        for name, gradient in gradients.items():
            gradient = gradient / max(norm, 1)
            first[name] = 0.9 * first[name] + 0.1 * gradient
            second[name] = 0.999 * second[name] + 0.001 * gradient**2
            m = first[name] / (1 - 0.9**epoch)
            v = second[name] / (1 - 0.999**epoch)
            parameters[name] -= learning_rate * m / (np.sqrt(v) + 1e-8)
    return parameters


def evaluate_policy(parameters, paths, market):
    holdings = positions(parameters, paths, market)
    pnl, costs, turnover = hedge_pnl(paths, holdings, cost_rate=market.transaction_cost)
    error = call_price(market) + pnl - np.maximum(paths[:, -1] - market.strike, 0)
    quantiles = np.quantile(error, [0.05, 0.5, 0.95])
    return {
        "mse": float(np.mean(error**2)),
        "bias": float(np.mean(error)),
        "error_q05": float(quantiles[0]),
        "error_q50": float(quantiles[1]),
        "error_q95": float(quantiles[2]),
        "mean_transaction_cost": float(np.mean(costs)),
        "mean_turnover": float(np.mean(turnover)),
    }


@dataclass(frozen=True)
class HedgingObjective:
    market: Market = Market()
    diagnostic_dir: Path | None = None
    backend: str = "numpy"

    def __post_init__(self):
        validate_backend(self.backend)

    def evaluate(self, config, replication):
        if self.backend == "cuda":
            from .hedging_cuda import evaluate

            return evaluate(config, replication, self.market)
        parameters = train_policy(config, replication, self.market)
        paths = simulate_paths(
            self.market,
            self.market.evaluation_paths,
            np.random.default_rng(replication.scenario_seed),
        )
        return evaluate_policy(parameters, paths, self.market)

    def __call__(self, config, replication):
        metrics = self.evaluate(config, replication)
        if self.diagnostic_dir is not None:
            name = (
                f"{replication.phase}-{replication.training_seed:016x}-"
                f"{replication.scenario_seed:016x}-{replication.index}.json"
            )
            write_json(
                Path(self.diagnostic_dir) / name,
                {
                    "config": config,
                    "replication": asdict(replication),
                    "market": asdict(self.market),
                    "backend": self.backend,
                    "metrics": metrics,
                },
            )
        return metrics["mse"]


def hedging_pool(seed, count):
    return sample_configs(
        {
            "learning_rate": Float(0.003, 0.1, log=True),
            "width": Choice((4, 8, 16)),
            "regularization": Float(1e-6, 1e-3, log=True),
        },
        count=count,
        seed=seed,
    )
