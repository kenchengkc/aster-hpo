"""Run a functional smoke comparison; this cheap objective is not a speed benchmark."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from aster_hpo import Float, FullBudget, Replication, ReplicationRace, Study, audit, sample_configs


def objective(config: dict, replication: Replication) -> float:
    # Consume the same exogenous shock for each config; the true optimum is x = 0.25.
    shock = random.Random(replication.scenario_seed).gauss(0.0, 0.1)
    x = config["x"]
    return (x - 0.25) ** 2 + (1.0 + abs(x)) * shock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--configs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("results/smoke"))
    args = parser.parse_args()
    configs = sample_configs({"x": Float(-2, 2)}, count=args.configs, seed=args.seed)
    summary = {}
    for name, scheduler in [("full_budget", FullBudget(27)), ("race", ReplicationRace())]:
        result = Study(configs, scheduler=scheduler, workers=args.workers, seed=args.seed).optimize(
            objective
        )
        result.save(args.output / f"{name}.json")
        winner = result.best_trial
        if winner is None:
            raise RuntimeError("No configuration completed")
        check = audit(objective, winner.config, seed=args.seed, replications=100)
        audit_path = args.output / f"{name}_audit.json"
        audit_path.write_text(json.dumps(asdict(check), indent=2) + "\n", encoding="utf-8")
        summary[name] = {
            "best_config": winner.config,
            "true_loss": (winner.config["x"] - 0.25) ** 2,
            "audit_mean": check.mean,
            "audit_standard_error": check.standard_error,
            "search_replications": sum(len(t.observations) for t in result.trials),
            "search_wall_seconds": result.wall_seconds,
            "objective_cpu_seconds": result.cpu_seconds,
            "pruned_trials": sum(t.status == "pruned" for t in result.trials),
        }
    args.output.joinpath("summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Smoke comparison only: process startup dominates this cheap objective.")


if __name__ == "__main__":
    main()
