"""Freeze, execute, resume, and report controlled hedging comparisons."""

import argparse
import json
import random
import sqlite3
from dataclasses import asdict
from pathlib import Path

from aster_hpo import ASHA, FullBudget, ReplicationRace, Study, audit
from aster_hpo.storage import SQLiteStore

from .common import create_experiment, environment, write_json
from .financial_report import regenerate
from .hedging import HedgingObjective, Market, hedging_pool, validate_backend
from .protocol import digest, freeze_protocol, source_fingerprint


def _partial(database):
    if not database.exists():
        return {"observations": [], "attempts": [], "wall_seconds": 0.0}
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        observations = [
            json.loads(row[0]) for row in db.execute("SELECT payload FROM observations")
        ]
        attempts = [dict(row) for row in db.execute("SELECT * FROM attempts")]
        saved = db.execute("SELECT payload FROM state WHERE id=1").fetchone()
        wall = json.loads(saved[0])["wall_seconds"] if saved else 0.0
        return {"observations": observations, "attempts": attempts, "wall_seconds": wall}


def _campaign(folder, spec, protocol):
    folder.mkdir(parents=True, exist_ok=True)
    market = Market(**protocol["market"])
    objective = HedgingObjective(market, folder / "metrics", protocol.get("backend", "numpy"))
    settings = {
        "seed": spec["seed"],
        "threads_per_worker": protocol["threads_per_worker"],
        "objective_id": "hedging-v1:" + digest(protocol),
    }
    scheduler = {
        "serial_full": FullBudget(protocol["checkpoints"][-1]),
        "parallel_full": FullBudget(protocol["checkpoints"][-1]),
        "race": ReplicationRace(tuple(protocol["checkpoints"])),
        "asha": ASHA(tuple(protocol["checkpoints"]), protocol["reduction_factor"]),
    }[spec["method"]]
    try:
        result = Study(
            hedging_pool(spec["seed"], protocol["count"]),
            scheduler=scheduler,
            workers=spec["workers"],
            storage=folder / "search.sqlite",
            **settings,
        ).optimize(objective)
        result.save(folder / "search.json")
        if result.best_trial is None:
            raise RuntimeError("No completed candidate")
        winner = result.best_trial.config
        lock = folder / "winner.json"
        if lock.exists() and json.loads(lock.read_text()) != winner:
            raise ValueError("Locked winner differs from search result")
        write_json(lock, winner)
        checked = audit(
            objective,
            winner,
            replications=protocol["audit_replications"],
            workers=1,
            storage=folder / "audit.sqlite",
            **settings,
        )
        write_json(folder / "audit.json", asdict(checked))
    except Exception as exc:
        write_json(
            folder / "failure.json",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "search": _partial(folder / "search.sqlite"),
                "audit": _partial(folder / "audit.sqlite"),
            },
        )
    write_json(folder / "run.json", spec)


def run_study(protocol_path, output, *, resume=False, progress=False):
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol["schema_version"] != 1 or protocol["source_sha256"] != source_fingerprint():
        raise ValueError("Protocol source fingerprint differs; freeze a new protocol")
    output = Path(output)
    checksum = digest(protocol)
    backend = validate_backend(protocol.get("backend", "numpy"))
    backend_identity = None
    if backend.startswith("cuda"):
        from .hedging_cuda import device_info

        backend_identity = {**device_info(), "implementation": backend}
    if resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["protocol_sha256"] != checksum:
            raise ValueError("Saved protocol differs from requested protocol")
        current = environment()
        previous = manifest["environment"]
        if manifest.get("backend_identity") != backend_identity:
            raise ValueError("Resume CUDA backend identity differs")
        for field in ("python", "platform", "machine", "cpu_model"):
            if current[field] != previous[field]:
                raise ValueError(f"Resume environment differs: {field}")
        for package in ("numpy", "scipy", "threadpoolctl"):
            if current["dependencies"].get(package) != previous["dependencies"].get(package):
                raise ValueError(f"Resume dependency differs: {package}")
    else:
        campaigns = []
        for seed in protocol["seeds"]:
            methods = list(protocol["methods"])
            random.Random(seed).shuffle(methods)
            campaigns.extend(
                {
                    "id": f"{seed}-{method}",
                    "seed": seed,
                    "method": method,
                    "workers": 1 if method == "serial_full" else protocol["workers"],
                }
                for method in methods
            )
        scaling = [
            {"id": f"{seed}-{workers}", "seed": seed, "workers": workers}
            for seed in protocol["scaling_seeds"]
            for workers in protocol["scaling_workers"]
        ]
        manifest = {
            "schema_version": 1,
            "kind": "financial comparison",
            "protocol": protocol,
            "protocol_sha256": checksum,
            "campaigns": campaigns,
            "scaling": scaling,
            "backend_identity": backend_identity,
        }
        create_experiment(output, manifest)
        manifest = json.loads((output / "manifest.json").read_text())
    # Prevent two resume commands from competing over experiment artifacts.
    with SQLiteStore(output / "runner.sqlite", {"protocol_sha256": checksum}):
        for index, spec in enumerate(manifest["campaigns"], 1):
            folder = output / "campaigns" / spec["id"]
            if not (folder / "run.json").exists():
                _campaign(folder, spec, protocol)
            if progress:
                print(f"Campaign {index}/{len(manifest['campaigns'])}: {spec['id']}", flush=True)
        for spec in manifest["scaling"]:
            folder = output / "scaling" / spec["id"]
            if not (folder / "result.json").exists():
                result = Study(
                    hedging_pool(spec["seed"], protocol["scaling_count"]),
                    scheduler=FullBudget(protocol["scaling_replications"]),
                    workers=spec["workers"],
                    seed=spec["seed"],
                    storage=folder / "study.sqlite",
                    objective_id="hedging-scale:" + checksum,
                ).optimize(HedgingObjective(Market(**protocol["market"]), backend=backend))
                result.save(folder / "result.json")
            if progress:
                print(f"Scaling: {spec['id']}", flush=True)
    return regenerate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--pilot", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--seeds", type=int, default=20)
    freeze.add_argument("--workers", type=int, default=4)
    freeze.add_argument("--configs", type=int, default=6)
    freeze.add_argument("--audits", type=int, default=8)
    freeze.add_argument("--backend", choices=("numpy", "cuda", "cuda_eager"), default="numpy")
    execute = commands.add_parser("run")
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--resume", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_protocol(
            args.pilot,
            args.protocol,
            seeds=tuple(range(args.seeds)),
            workers=args.workers,
            count=args.configs,
            audit_replications=args.audits,
            backend=args.backend,
        )
        print(f"Protocol frozen: {args.protocol}")
    elif args.command == "run":
        report = run_study(args.protocol, args.output, resume=args.resume, progress=True)
        print(json.dumps(report["comparisons"]["race"], indent=2))
    else:
        regenerate(args.output)
        if args.plots:
            from .study_plots import plot_study

            plot_study(args.output)
        print(f"Report: {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
