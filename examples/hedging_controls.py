"""Compare three European call hedges on shared simulated paths with NumPy."""

import argparse

import numpy as np

from aster_hpo import Replication
from benchmarks.hedging import (
    Market,
    call_price,
    delta_positions,
    hedge_pnl,
    positions,
    simulate_paths,
    train_policy,
)

# Fixed before evaluating the held-out paths.
NEURAL_CONFIG = {"width": 8, "learning_rate": 0.03, "regularization": 0.0001}


def compare_hedges(parameters, paths, market):
    """Score terminal wealth after liquidation against the call payoff on each path."""
    capital = call_price(market)
    payoff = np.maximum(paths[:, -1] - market.strike, 0)
    strategies = {
        "Fixed neural hedge": positions(parameters, paths, market),
        "Black-Scholes delta": delta_positions(paths, market),
        "No hedge": np.zeros_like(paths[:, :-1]),
    }
    rows = {}
    for strategy, holdings in strategies.items():
        pnl, costs, turnover = hedge_pnl(paths, holdings, cost_rate=market.transaction_cost)
        rows[strategy] = {
            "mse": float(np.mean((capital + pnl - payoff) ** 2)),
            "mean_transaction_cost": float(np.mean(costs)),
            "mean_turnover": float(np.mean(turnover)),
        }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="root for training and evaluation")
    parser.add_argument("--paths", type=int, default=2048, help="number of held-out paths")
    args = parser.parse_args()
    if args.paths < 1:
        parser.error("--paths must be positive")

    market = Market(evaluation_paths=args.paths)
    rep = Replication.create(args.seed, NEURAL_CONFIG, 0, "hedging_controls", True)
    parameters = train_policy(NEURAL_CONFIG, rep, market)
    paths = simulate_paths(
        market, market.evaluation_paths, np.random.default_rng(rep.scenario_seed)
    )
    results = compare_hedges(parameters, paths, market)

    print(
        f"European call | spot {market.spot:g} | strike {market.strike:g} | "
        f"maturity {market.maturity:g}y | volatility {market.volatility:.0%} | "
        f"{market.steps} steps | zero drift and interest"
    )
    print(
        f"Initial capital {call_price(market):.8f} | cost {market.transaction_cost:.2%} "
        "of traded value, including opening and liquidation"
    )
    print(f"Fixed neural configuration {NEURAL_CONFIG}")
    print(
        f"Root seed {args.seed} | training seed {rep.training_seed} "
        f"({market.epochs} batches of {market.batch_paths}) | "
        f"evaluation seed {rep.scenario_seed} ({len(paths)} shared paths)"
    )
    print(f"\n{'Strategy':<21} {'Hedging MSE':>12} {'Mean cost':>12} {'Mean turnover':>15}")
    for name, row in results.items():
        print(
            f"{name:<21} {row['mse']:12.8f} {row['mean_transaction_cost']:12.8f} "
            f"{row['mean_turnover']:15.8f}"
        )


if __name__ == "__main__":
    main()
