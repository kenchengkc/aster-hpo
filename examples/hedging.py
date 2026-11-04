"""Financial pilot: python -m examples.hedging --output results/hedging-pilot."""

import argparse
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from aster_hpo import FullBudget, ReplicationRace, Study, audit
from aster_hpo.runtime import thread_limits
from benchmarks.common import create_experiment, write_json
from benchmarks.hedging import HedgingObjective, Market, hedging_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--configs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-paths", type=int, default=128)
    parser.add_argument("--evaluation-paths", type=int, default=2048)
    parser.add_argument("--backend", choices=("numpy", "cuda"), default="numpy")
    args = parser.parse_args()
    if args.backend == "cuda" and args.workers != 1:
        parser.error("CUDA on one GPU requires --workers 1")
    market = Market(
        epochs=args.epochs, batch_paths=args.batch_paths, evaluation_paths=args.evaluation_paths
    )
    output = create_experiment(
        args.output,
        {
            "schema_version": 1,
            "kind": "financial pilot",
            "market": asdict(market),
            "backend": args.backend,
            "seed": args.seed,
            "workers": args.workers,
            "count": args.configs,
            "checkpoints": [3, 9],
            "audit_replications": 8,
        },
    )
    configs = hedging_pool(args.seed, args.configs)
    summaries = {}
    for name, scheduler in (("full_budget", FullBudget(9)), ("race", ReplicationRace((3, 9)))):
        objective = HedgingObjective(market, output / name / "metrics", args.backend)
        result = Study(
            configs,
            scheduler=scheduler,
            workers=args.workers,
            seed=args.seed,
            storage=output / name / "study.sqlite",
            objective_id=f"gbm-hedging-pilot-v1:{args.backend}",
        ).optimize(objective)
        result.save(output / name / "result.json")
        if result.best_trial is None:
            raise RuntimeError("Pilot has no completed candidate")
        # Lock the selected configuration before the independent pilot audit.
        write_json(output / name / "winner.json", result.best_trial.config)
        start = time.perf_counter()
        with thread_limits(1):
            checked = audit(objective, result.best_trial.config, replications=8, seed=args.seed)
        write_json(output / name / "audit.json", asdict(checked))
        summaries[name] = {
            "winner": result.best_trial.config,
            "audit_mean": checked.mean,
            "audit_standard_error": checked.standard_error,
            "audit_wall_seconds": time.perf_counter() - start,
            "search_wall_seconds": result.wall_seconds,
            "objective_cpu_seconds": result.cpu_seconds,
            "replications": sum(len(t.observations) for t in result.trials),
            "median_replication_seconds": statistics.median(
                o.wall_seconds for t in result.trials for o in t.observations
            ),
        }
    write_json(output / "summary.json", summaries)
    print(summaries)
    print("Disjoint pilot only. Freeze a separate protocol before the comparative study.")


if __name__ == "__main__":
    main()
