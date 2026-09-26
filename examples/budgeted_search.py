"""Demonstrate a wall-limited search and resume: python -m examples.budgeted_search."""

import argparse
import json
import random
from pathlib import Path

from aster_hpo import FullBudget, RunBudget, Study


def monte_carlo_loss(config, replication):
    # Actual CPU work; this example demonstrates control semantics, not speedup.
    rng = random.Random(replication.scenario_seed)
    noise = sum(rng.gauss(0, 1) for _ in range(100_000)) / 100_000
    return (config["x"] - 0.25) ** 2 + noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/budgeted-search"))
    parser.add_argument("--seconds", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    allowance = RunBudget(max_wall_seconds=args.seconds)
    args.output.mkdir(parents=True, exist_ok=False)
    study = Study(
        [{"x": i / 4 - 1} for i in range(12)],
        scheduler=FullBudget(3),
        workers=args.workers,
        seed=41,
        storage=args.output / "study.sqlite",
        objective_id="mc-quadratic-v1",
        max_retries=0,
    )
    stopped = study.optimize(monte_carlo_loss, budget=allowance)
    stopped.save(args.output / "stopped.json")
    resumed = study.optimize(monte_carlo_loss)
    resumed.save(args.output / "resumed.json")
    summary = {
        "interpretation": "runtime control demonstration; not a performance benchmark",
        "first_stop_reason": stopped.metadata["termination_reason"],
        "wall_allowance_seconds": args.seconds,
        "first_session_wall_seconds": stopped.metadata["session_wall_seconds"],
        "first_session_attempts": stopped.metadata["session_attempts"],
        "first_accepted_observations": sum(len(t.observations) for t in stopped.trials),
        "resume_stop_reason": resumed.metadata["termination_reason"],
        "total_accepted_observations": sum(len(t.observations) for t in resumed.trials),
        "unmeasured_attempts": resumed.metadata["unmeasured_attempts"],
        "winner": resumed.best_trial.config if resumed.best_trial else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
