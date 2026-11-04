"""Cold, fresh-process hedging search and independent-audit development comparison.

Run ``python -m benchmarks.cuda_workflow run --output NEW_DIRECTORY`` from the
source checkout. Add ``--include-eager`` to compare the preserved CUDA trainer.
No objective warmup is performed. The 30 campaigns (42 with eager) use the real
Study runtime and retain their SQLite journals, locked winner, and every metric.
CUDA is optional until an explicitly selected CUDA campaign initializes it.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .common import environment, write_json

ROOT = Path(__file__).resolve().parents[1]
RTOL = 1e-6
ATOL = 1e-8
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
METRICS = {
    "mse",
    "bias",
    "error_q05",
    "error_q50",
    "error_q95",
    "mean_transaction_cost",
    "mean_turnover",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(), parse_constant=reject)


def source_identity():
    """Fingerprint Python and native sources, including the shared CUDA header."""
    paths = set(ROOT.glob("src/aster_hpo/*.py")) | set(ROOT.glob("benchmarks/*.py"))
    paths.add(ROOT / "pyproject.toml")
    for directory in (ROOT / "cuda", ROOT / "native"):
        paths.update(
            path
            for path in directory.rglob("*")
            if (
                path.suffix in {".cu", ".cuh", ".cpp", ".hpp", ".h", ".in"}
                or path.name == "CMakeLists.txt"
            )
            and not any(part in {"build", ".venv"} for part in path.relative_to(ROOT).parts)
        )
    files = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    return {"sha256": _digest(files), "files": files}


def default_protocol(*, include_eager=False):
    # The numerical module is imported only when freezing or inside the timed child.
    from .hedging import Market

    tiny = asdict(Market())
    protocol = {
        "schema_version": 1,
        "kind": "CUDA whole-workflow development comparison",
        "markets": {
            "tiny": tiny,
            "scaled": {**tiny, "batch_paths": 4096, "evaluation_paths": 32768},
        },
        "seeds": [31000, 31001, 31002],
        "count": 4,
        "checkpoints": [2, 4],
        "audit_replications": 3,
        "threads_per_worker": 1,
        "closeness": {"rtol": RTOL, "atol": ATOL},
        "warmup_objective_calls": 0,
        "cache_scope": (
            "Fresh interpreter and CUDA context per campaign; ordinary shared disk/compiler "
            "caches are retained, including kernels populated by prior correctness validation. "
            "Fresh process does not mean empty JIT cache. No explicit warmup or cache purge."
        ),
        "interpretation": "Descriptive paired development comparison; no p99 or quality guarantee",
    }
    if include_eager:
        protocol["include_eager"] = True
    return protocol


def build_jobs(protocol):
    """Deterministically interleave workload and backend within each optimizer seed."""
    variants = [
        ("numpy", "full", 1),
        ("cuda", "full", 1),
        ("numpy", "race", 1),
        ("cuda", "race", 1),
        ("numpy", "full", 4),
    ]
    if protocol.get("include_eager", False):
        variants.extend([("cuda_eager", "full", 1), ("cuda_eager", "race", 1)])
    jobs = []
    for seed in protocol["seeds"]:
        order = list(variants)
        random.Random(seed).shuffle(order)
        workloads = list(protocol["markets"])
        if seed % 2:
            workloads.reverse()
        for backend, scheduler, workers in order:
            for workload in workloads:
                jobs.append(
                    {
                        "id": f"{workload}-{seed}-{backend}-{scheduler}-w{workers}",
                        "workload": workload,
                        "seed": seed,
                        "backend": backend,
                        "scheduler": scheduler,
                        "workers": workers,
                    }
                )
    return jobs


def _backend_environment():
    result = environment()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if name.lower().startswith(("cupy", "nvidia-")):
            result["dependencies"][name] = distribution.version
    result["backend_environment"] = {
        key: os.environ.get(key)
        for key in (
            *THREAD_VARIABLES,
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "CUDA_HOME",
            "CUDA_PATH",
            "CUPY_CACHE_DIR",
            "CUPY_ACCELERATORS",
            "CUPY_CUDA_COMPILE_WITH_DEBUG",
            "CUPY_CACHE_IN_MEMORY",
            "CUDA_CACHE_PATH",
            "CUDA_CACHE_DISABLE",
        )
    }
    cache = Path(os.environ.get("CUPY_CACHE_DIR", Path.home() / ".cupy/kernel_cache"))
    result["cupy_disk_cache"] = {
        "path": str(cache),
        "exists": cache.is_dir(),
        "files_at_snapshot": sum(path.is_file() for path in cache.rglob("*"))
        if cache.is_dir()
        else 0,
        "scope": "Inherited shared disk cache, possibly populated by correctness validation",
    }
    result["executable"] = sys.executable
    return result


def _finite(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Nonfinite or invalid {label}")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid {label}")


def _validate_observations(observations, config, seed, phase):
    from aster_hpo import Replication

    for index, observation in enumerate(observations):
        if observation["error"] is not None or observation["loss"] is None:
            raise ValueError(f"Failed {phase} observation at index {index}")
        if not math.isfinite(observation["loss"]):
            raise ValueError(f"Nonfinite {phase} loss")
        expected = asdict(Replication.create(seed, config, index, phase, True))
        if observation["replication"] != expected:
            raise ValueError(f"Mismatched {phase} observation identity")
        _finite(observation["wall_seconds"], f"{phase} observation wall time")
        _finite(observation["cpu_seconds"], f"{phase} observation host CPU time")


def _validate_attempts(metadata, expected):
    if metadata["termination_reason"] != "completed":
        raise ValueError("Incomplete campaign termination")
    attempts = metadata["attempts"]
    if len(attempts) != len(expected) or any(a["status"] != "accepted" for a in attempts):
        raise ValueError("Missing or unsuccessful campaign attempt")
    identities = set()
    for attempt in attempts:
        key = attempt["trial_id"], attempt["replication_index"]
        if (
            key in identities
            or key not in expected
            or attempt["replication"] != expected[key]
            or attempt["error"] is not None
        ):
            raise ValueError("Accepted attempt identity differs from observation")
        identities.add(key)
    if metadata.get("unmeasured_attempts", 0):
        raise ValueError("Unmeasured campaign attempts")


def _validate_search(search, job, protocol):
    from .hedging import hedging_pool

    trials = search["trials"]
    configs = hedging_pool(job["seed"], protocol["count"])
    if len(trials) != len(configs):
        raise ValueError("Missing search trial")
    total = 0
    maximum = protocol["checkpoints"][-1]
    for index, (trial, config) in enumerate(zip(trials, configs, strict=True)):
        if trial["id"] != index or trial["config"] != config:
            raise ValueError("Mismatched search configuration identity")
        status = trial["status"]
        observations = trial["observations"]
        n = len(observations)
        if status == "complete":
            if n != maximum:
                raise ValueError("Incomplete full-budget trial observation count")
        elif job["scheduler"] == "race" and status == "pruned":
            if not protocol["checkpoints"][0] <= n <= maximum:
                raise ValueError("Invalid pruned trial observation count")
        else:
            raise ValueError(f"Failed or incomplete search trial: {status}")
        _validate_observations(observations, config, job["seed"], "search")
        total += n
    _validate_attempts(
        search["metadata"],
        {
            (trial["id"], o["replication"]["index"]): o["replication"]
            for trial in trials
            for o in trial["observations"]
        },
    )
    complete = [t for t in trials if t["status"] == "complete"]
    if not complete:
        raise ValueError("No completed search candidate")
    return min(
        complete, key=lambda t: (statistics.mean(o["loss"] for o in t["observations"]), t["id"])
    ), total


def _observations(record):
    for trial in record["search"]["trials"]:
        for observation in trial["observations"]:
            yield trial["config"], observation
    for observation in record["audit"]["observations"]:
        yield record["winner"], observation


def _metric_name(replication):
    return (
        f"{replication['phase']}-{replication['training_seed']:016x}-"
        f"{replication['scenario_seed']:016x}-{replication['index']}.json"
    )


def _collect_metrics(folder, record):
    rows = []
    for config, observation in _observations(record):
        filename = "metrics/" + _metric_name(observation["replication"])
        path = folder / filename
        if not path.is_file():
            raise ValueError(f"Missing metric file: {filename}")
        payload = _read(path)
        if payload["config"] != config or payload["replication"] != observation["replication"]:
            raise ValueError(f"Mismatched metric identity: {filename}")
        rows.append(
            {
                "file": filename,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                **payload,
            }
        )
    return rows


def validate_record(record, protocol, *, folder=None):
    """Reject incomplete work, failed replications and corrupt/missing metric evidence."""
    try:
        job = record["job"]
        if record["protocol_sha256"] != _digest(protocol):
            raise ValueError("Campaign protocol identity differs")
        if record["backend"]["name"] != job["backend"]:
            raise ValueError("Campaign backend identity differs")
        if job["backend"] in ("cuda", "cuda_eager"):
            implementation = "eager" if job["backend"] == "cuda_eager" else "optimized"
            if record["backend"].get("implementation") != implementation:
                raise ValueError("Campaign CUDA implementation identity differs")
        winner, count = _validate_search(record["search"], job, protocol)
        if record["winner"] != winner["config"] or record["winner_trial_id"] != winner["id"]:
            raise ValueError("Locked winner differs from completed search")
        observations = record["audit"]["observations"]
        if len(observations) != protocol["audit_replications"]:
            raise ValueError("Incomplete independent audit observation count")
        _validate_observations(observations, record["winner"], job["seed"], "audit")
        _validate_attempts(
            record["audit"]["metadata"],
            {(0, o["replication"]["index"]): o["replication"] for o in observations},
        )
        expected_counts = {
            "search_observations": count,
            "audit_observations": len(observations),
            "objective_failures": 0,
            "unsuccessful_attempts": 0,
        }
        if (
            any(type(value) is not int for value in record["counts"].values())
            or record["counts"] != expected_counts
        ):
            raise ValueError("Campaign observation/failure counts differ")
        metric_rows = record["metrics"]
        if len(metric_rows) != count + len(observations):
            raise ValueError("Missing metric records")
        for row, (config, observation) in zip(metric_rows, _observations(record), strict=True):
            if row["config"] != config or row["replication"] != observation["replication"]:
                raise ValueError("Metric observation identity differs")
            if row["backend"] != job["backend"]:
                raise ValueError("Metric backend differs")
            if row["market"] != protocol["markets"][job["workload"]]:
                raise ValueError("Metric market differs")
            if not METRICS.issubset(row["metrics"]):
                raise ValueError("Missing financial metric")
            if not all(math.isfinite(value) for value in row["metrics"].values()):
                raise ValueError("Nonfinite financial metric")
            if row["metrics"]["mse"] != observation["loss"]:
                raise ValueError("Metric loss differs from accepted observation")
        if len({row["file"] for row in metric_rows}) != len(metric_rows):
            raise ValueError("Duplicate metric identity")
        if folder is not None:
            actual = _collect_metrics(Path(folder), record)
            if actual != metric_rows:
                raise ValueError("Metric files differ from saved records")
            if len(list((Path(folder) / "metrics").glob("*.json"))) != len(actual):
                raise ValueError("Unexpected metric files")
        if "timings" in record:
            times = record["timings"]
            parts = (
                "backend_setup_seconds",
                "search_seconds",
                "audit_seconds",
                "result_write_seconds",
            )
            for name in (*parts, "workflow_seconds"):
                _finite(times[name], name, positive=True)
            _finite(times["host_cpu_seconds"], "host CPU seconds")
            if times["workflow_seconds"] < sum(times[name] for name in parts):
                raise ValueError("Workflow timing omits measured stages")
        if "process_seconds" in record:
            _finite(record["process_seconds"], "whole process seconds", positive=True)
            if record["process_seconds"] < record["timings"]["workflow_seconds"]:
                raise ValueError("Whole process time is shorter than workflow")
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError(f"Incomplete or invalid campaign record: {error}") from error
    return record


def _partial_journal(path):
    if not path.is_file():
        return {"attempts": [], "observations": []}
    with sqlite3.connect(path) as database:
        database.row_factory = sqlite3.Row
        return {
            "attempts": [dict(row) for row in database.execute("SELECT * FROM attempts")],
            "observations": [
                json.loads(row[0]) for row in database.execute("SELECT payload FROM observations")
            ],
        }


def campaign(request):
    """Internal child entry point: the clock starts before numerical/backend imports."""
    job, protocol = request["job"], request["protocol"]
    if source_identity() != request["source"]:
        raise ValueError("Frozen source identity differs before campaign")
    folder = Path(request["folder"])
    folder.mkdir(parents=True, exist_ok=False)
    timings = {}
    preloaded = [
        name for name in ("numpy", "cupy", "benchmarks.hedging_cuda") if name in sys.modules
    ]
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    try:
        # Import and CUDA context creation are inside the cold workflow boundary.
        from aster_hpo import FullBudget, ReplicationRace, Study, audit

        from .hedging import HedgingObjective, Market, hedging_pool

        def synchronize():
            pass

        backend = {"name": job["backend"]}
        if job["backend"] in ("cuda", "cuda_eager"):
            from . import hedging_cuda

            backend["device"] = hedging_cuda.device_info()
            backend["implementation"] = "eager" if job["backend"] == "cuda_eager" else "optimized"
            synchronize = hedging_cuda.synchronize
        elif job["backend"] != "numpy":
            raise ValueError(f"Unknown backend: {job['backend']}")
        market = Market(**protocol["markets"][job["workload"]])
        objective = HedgingObjective(market, folder / "metrics", backend=job["backend"])
        backend["environment"] = _backend_environment()
        synchronize()
        timings["backend_setup_seconds"] = time.perf_counter() - wall_start
        settings = {
            "seed": job["seed"],
            "threads_per_worker": protocol["threads_per_worker"],
            "objective_id": f"hedging-cuda-workflow:{job['backend']}:" + _digest(protocol),
            "max_retries": 0,
        }
        start = time.perf_counter()
        scheduler = (
            FullBudget(protocol["checkpoints"][-1])
            if job["scheduler"] == "full"
            else ReplicationRace(tuple(protocol["checkpoints"]))
        )
        result = Study(
            hedging_pool(job["seed"], protocol["count"]),
            scheduler=scheduler,
            workers=job["workers"],
            storage=folder / "search.sqlite",
            **settings,
        ).optimize(objective)
        synchronize()
        timings["search_seconds"] = time.perf_counter() - start
        result.save(folder / "search.json")
        search = _read(folder / "search.json")
        winner, count = _validate_search(search, job, protocol)
        # Exclusive creation makes the exact configuration reviewable before audit.
        with (folder / "winner.json").open("x") as handle:
            json.dump(winner["config"], handle, indent=2, allow_nan=False)
            handle.write("\n")
        start = time.perf_counter()
        checked = audit(
            objective,
            winner["config"],
            replications=protocol["audit_replications"],
            workers=1,
            storage=folder / "audit.sqlite",
            **settings,
        )
        synchronize()
        timings["audit_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        audited = asdict(checked)
        write_json(folder / "audit.json", audited)
        record = {
            "schema_version": 1,
            "job": job,
            "protocol_sha256": _digest(protocol),
            "campaign_pid": os.getpid(),
            "preloaded_numerical_modules": preloaded,
            "source_sha256": request["source"]["sha256"],
            "backend": backend,
            "winner": winner["config"],
            "winner_trial_id": winner["id"],
            "search": search,
            "audit": audited,
            "host_objective_cpu_seconds": {
                "search": result.cpu_seconds,
                "audit": sum(o.cpu_seconds for o in checked.observations),
            },
            "counts": {
                "search_observations": count,
                "audit_observations": len(checked.observations),
                "objective_failures": 0,
                "unsuccessful_attempts": 0,
            },
            "timing_scope": {
                "workflow": (
                    "Before backend imports/setup through search, winner lock, independent "
                    "audit, device completion and result writes; timing trailer excluded"
                ),
                "process": "Parent clock: interpreter startup, imports, workflow, trailer and exit",
                "cpu": (
                    "host process CPU seconds; excludes child-worker CPU; not GPU compute cost"
                ),
                "objective_cpu": (
                    "Sum of observation host process CPU seconds; excludes worker startup"
                ),
                "warmup_objective_calls": 0,
            },
        }
        record["metrics"] = _collect_metrics(folder, record)
        validate_record(record, protocol, folder=folder)
        synchronize()
        if source_identity() != request["source"]:
            raise ValueError("Frozen source changed during campaign")
        write_json(folder / "record.json", record)
        timings["result_write_seconds"] = time.perf_counter() - start
        timings["workflow_seconds"] = time.perf_counter() - wall_start
        timings["host_cpu_seconds"] = time.process_time() - cpu_start
        write_json(folder / "timing.json", timings)
        return record
    except BaseException as error:
        partial = {
            phase: _partial_journal(folder / f"{phase}.sqlite") for phase in ("search", "audit")
        }
        counts = {
            "search_observations": len(partial["search"]["observations"]),
            "audit_observations": len(partial["audit"]["observations"]),
            "objective_failures": sum(
                o["error"] is not None for phase in partial.values() for o in phase["observations"]
            ),
            "unsuccessful_attempts": sum(
                a["status"] != "accepted" for phase in partial.values() for a in phase["attempts"]
            ),
        }
        write_json(
            folder / "failure.json",
            {
                "error": f"{type(error).__name__}: {error}",
                "job": job,
                "elapsed_workflow_seconds": time.perf_counter() - wall_start,
                "host_cpu_seconds": time.process_time() - cpu_start,
                **partial,
                "counts": counts,
            },
        )
        raise


def _identity(config, replication):
    return (
        _digest(config),
        *(replication[key] for key in ("index", "scenario_seed", "training_seed", "phase")),
    )


def _losses(record):
    return {
        _identity(trial["config"], observation["replication"]): observation["loss"]
        for trial in record["search"]["trials"]
        for observation in trial["observations"]
    }


def _range(values):
    return {"median": statistics.median(values), "minimum": min(values), "maximum": max(values)}


def _close(a, b, protocol):
    return (
        math.isfinite(a)
        and math.isfinite(b)
        and abs(a - b) <= protocol["closeness"]["atol"] + protocol["closeness"]["rtol"] * abs(a)
    )


def _compare_pairs(
    pairs, protocol, *, adaptive, reference_label="numpy", candidate_label="cuda", metrics=False
):
    rows, mismatches, matched = [], [], 0
    metric_mismatches, metric_count, metric_observations = [], 0, 0
    for reference, candidate in pairs:
        a, b = _losses(reference), _losses(candidate)
        keys = a.keys() & b.keys()
        if not adaptive and a.keys() != b.keys():
            raise ValueError("Full-budget matched observation identities differ")
        failures = [
            {
                "identity": list(key),
                f"{reference_label}_loss": a[key],
                f"{candidate_label}_loss": b[key],
                "absolute_difference": abs(a[key] - b[key]),
            }
            for key in sorted(keys)
            if not _close(a[key], b[key], protocol)
        ]
        mismatches.extend({"seed": reference["job"]["seed"], **failure} for failure in failures)
        matched += len(keys)
        audit_a = reference["audit"]["observations"]
        audit_b = candidate["audit"]["observations"]
        if [(o["replication"]["index"], o["replication"]["scenario_seed"]) for o in audit_a] != [
            (o["replication"]["index"], o["replication"]["scenario_seed"]) for o in audit_b
        ]:
            raise ValueError("Paired audit scenario identities differ")
        audit_mean_a = statistics.mean(o["loss"] for o in audit_a)
        audit_mean_b = statistics.mean(o["loss"] for o in audit_b)
        row = {
            "seed": reference["job"]["seed"],
            f"{reference_label}_job": reference["job"]["id"],
            f"{candidate_label}_job": candidate["job"]["id"],
            "workflow_speedup": (
                reference["timings"]["workflow_seconds"] / candidate["timings"]["workflow_seconds"]
            ),
            "process_speedup": reference["process_seconds"] / candidate["process_seconds"]
            if "process_seconds" in reference and "process_seconds" in candidate
            else None,
            "search_speedup": reference["timings"]["search_seconds"]
            / candidate["timings"]["search_seconds"],
            "audit_speedup": reference["timings"]["audit_seconds"]
            / candidate["timings"]["audit_seconds"],
            f"{reference_label}_observations": len(a),
            f"{candidate_label}_observations": len(b),
            "matched_observations": len(keys),
            f"{reference_label}_unmatched_observations": len(a.keys() - b.keys()),
            f"{candidate_label}_unmatched_observations": len(b.keys() - a.keys()),
            "winner_agreement": reference["winner"] == candidate["winner"],
            f"{reference_label}_winner": reference["winner"],
            f"{candidate_label}_winner": candidate["winner"],
            f"{reference_label}_audit_mean": audit_mean_a,
            f"{candidate_label}_audit_mean": audit_mean_b,
            "audit_mean_difference": audit_mean_b - audit_mean_a,
            "audit_loss_differences": [
                b["loss"] - a["loss"] for a, b in zip(audit_a, audit_b, strict=True)
            ],
            "maximum_matched_loss_absolute_difference": max(
                (abs(a[key] - b[key]) for key in keys), default=None
            ),
        }
        if metrics:
            metric_a, metric_b = (
                {
                    _identity(item["config"], item["replication"]): item["metrics"]
                    for item in record["metrics"]
                }
                for record in (reference, candidate)
            )
            metric_keys = metric_a.keys() & metric_b.keys()
            if not adaptive:
                # Each campaign needs complete metric evidence. Distinct winners can
                # legitimately have different audit training seeds and identities.
                for record, saved in ((reference, metric_a), (candidate, metric_b)):
                    expected = {
                        _identity(config, observation["replication"])
                        for config, observation in _observations(record)
                    }
                    if saved.keys() != expected:
                        raise ValueError(
                            "Full-budget financial metric observation identities differ"
                        )
            metric_mismatches.extend(
                {
                    "seed": reference["job"]["seed"],
                    "identity": list(key),
                    "metric": metric,
                    f"{reference_label}_value": metric_a[key][metric],
                    f"{candidate_label}_value": metric_b[key][metric],
                    "absolute_difference": abs(metric_a[key][metric] - metric_b[key][metric]),
                }
                for key in sorted(metric_keys)
                for metric in sorted(METRICS)
                if not _close(metric_a[key][metric], metric_b[key][metric], protocol)
            )
            metric_observations += len(metric_keys)
            metric_count += len(metric_keys) * len(METRICS)
            row.update(
                allocation_agreement=a.keys() == b.keys(),
                matched_metric_observations=len(metric_keys),
                **{
                    f"{reference_label}_audit_observations": len(audit_a),
                    f"{candidate_label}_audit_observations": len(audit_b),
                    f"{reference_label}_unmatched_metric_observations": len(
                        metric_a.keys() - metric_b.keys()
                    ),
                    f"{candidate_label}_unmatched_metric_observations": len(
                        metric_b.keys() - metric_a.keys()
                    ),
                },
            )
        rows.append(row)
    result = {
        "workload": pairs[0][0]["job"]["workload"],
        "scheduler": pairs[0][0]["job"]["scheduler"],
        f"{reference_label}_workers": pairs[0][0]["job"]["workers"],
        f"{candidate_label}_workers": pairs[0][1]["job"]["workers"],
        "pairs": rows,
        "paired_seeds": [row["seed"] for row in rows],
        "workflow_speedup": _range([row["workflow_speedup"] for row in rows]),
        "search_speedup": _range([row["search_speedup"] for row in rows]),
        "audit_speedup": _range([row["audit_speedup"] for row in rows]),
        "winner_agreement_count": sum(row["winner_agreement"] for row in rows),
        "audit_mean_difference": _range([row["audit_mean_difference"] for row in rows]),
        "matched_losses": {
            "count": matched,
            "mismatches": len(mismatches),
            "all_close": not mismatches,
            "details": mismatches,
            **protocol["closeness"],
        },
    }
    process = [row["process_speedup"] for row in rows if row["process_speedup"] is not None]
    result["process_speedup"] = _range(process) if process else None
    if metrics:
        result["matched_metrics"] = {
            "count": metric_count,
            "observations": metric_observations,
            "metrics": sorted(METRICS),
            "mismatches": len(metric_mismatches),
            "all_close": not metric_mismatches,
            "details": metric_mismatches,
            **protocol["closeness"],
        }
        result["allocation_agreement_count"] = sum(row["allocation_agreement"] for row in rows)
        result["speedup_definition"] = f"{reference_label}_seconds / {candidate_label}_seconds"
        for position, label in enumerate((reference_label, candidate_label)):
            result[f"{label}_backend"] = pairs[0][position]["job"]["backend"]
            for timing in pairs[0][position]["timings"]:
                result[f"{label}_{timing}"] = _range(
                    [pair[position]["timings"][timing] for pair in pairs]
                )
            durations = [
                pair[position]["process_seconds"]
                for pair in pairs
                if "process_seconds" in pair[position]
            ]
            result[f"{label}_process_seconds"] = _range(durations) if durations else None
    return result


def summarize(records, protocol):
    """Summarize validated campaigns by optimizer seed; adaptive work stays separate."""
    index = {}
    for record in records:
        job = record["job"]
        key = (job["workload"], job["scheduler"], job["workers"], job["backend"], job["seed"])
        if key in index:
            raise ValueError("duplicate campaign identity")
        index[key] = record
    result = {
        "full_budget": [],
        "adaptive": [],
        "cpu_four_worker_control": [],
        "cuda_optimization": [],
    }
    groups = sorted({(key[0], key[1], key[2]) for key in index if key[3] == "numpy"})
    used = set()
    for workload, scheduler, workers in groups:
        seeds = sorted(
            key[4] for key in index if key[:4] == (workload, scheduler, workers, "numpy")
        )
        pairs = []
        for seed in seeds:
            cpu_key = workload, scheduler, workers, "numpy", seed
            gpu_key = workload, scheduler, 1, "cuda", seed
            if gpu_key not in index:
                raise ValueError(f"Missing paired CUDA campaign: {gpu_key}")
            pairs.append((index[cpu_key], index[gpu_key]))
            used.update((cpu_key, gpu_key))
        category = (
            "adaptive"
            if scheduler == "race"
            else "cpu_four_worker_control"
            if workers == 4
            else "full_budget"
        )
        result[category].append(
            _compare_pairs(
                pairs,
                protocol,
                adaptive=scheduler == "race",
                metrics=protocol.get("include_eager", False),
            )
        )
    if protocol.get("include_eager", False) or any(key[3] == "cuda_eager" for key in index):
        groups = sorted({key[:3] for key in index if key[3] in ("cuda", "cuda_eager")})
        for workload, scheduler, workers in groups:
            if workers != 1:
                raise ValueError("CUDA optimization comparison requires one worker")
            seeds = sorted(
                {
                    key[4]
                    for key in index
                    if key[:3] == (workload, scheduler, workers)
                    and key[3] in ("cuda", "cuda_eager")
                }
            )
            pairs = []
            for seed in seeds:
                eager_key = workload, scheduler, workers, "cuda_eager", seed
                optimized_key = workload, scheduler, workers, "cuda", seed
                if eager_key not in index or optimized_key not in index:
                    raise ValueError(f"Missing paired eager/optimized CUDA campaign: {eager_key}")
                pairs.append((index[eager_key], index[optimized_key]))
                used.update((eager_key, optimized_key))
            result["cuda_optimization"].append(
                _compare_pairs(
                    pairs,
                    protocol,
                    adaptive=scheduler == "race",
                    reference_label="eager",
                    candidate_label="optimized",
                    metrics=True,
                )
            )
    if used != index.keys():
        raise ValueError("Unpaired campaign records")
    result["interpretation"] = (
        "Ratios are computed within optimizer seed, then summarized by median and range. "
        "Adaptive counts may differ; full-budget matched losses are the equal-work check. "
        "Audit differences are CUDA minus NumPy for CPU controls, and optimized minus eager "
        "for CUDA optimization, using independently evaluated locked winners. "
        "Eager-enabled comparisons also check all seven financial metrics on matched identities."
    )
    return result


def validate_summary(summary):
    """Require fixed-budget CPU and all matched CUDA-implementation numerical agreement."""
    failed = [
        group
        for category in ("full_budget", "cpu_four_worker_control")
        for group in summary[category]
        if not group["matched_losses"]["all_close"]
        or ("matched_metrics" in group and not group["matched_metrics"]["all_close"])
    ]
    failed.extend(
        group
        for group in summary.get("cuda_optimization", [])
        if not group["matched_losses"]["all_close"] or not group["matched_metrics"]["all_close"]
    )
    if failed:
        raise ValueError("Matched numerics failed the frozen development closeness check")
    return summary


def run(output, *, progress=False, include_eager=False):
    output = Path(output).resolve()
    # Preserve even an empty existing destination before imports or any subprocess.
    output.mkdir(parents=True, exist_ok=False)
    protocol = default_protocol(include_eager=include_eager)
    source = source_identity()
    jobs = build_jobs(protocol)
    manifest = {
        "schema_version": 1,
        "frozen_at": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "protocol_sha256": _digest(protocol),
        "source": source,
        "environment": _backend_environment(),
        "jobs": jobs,
        "timing": (
            "Manifest, protocol, source hashes, environment and order frozen before campaigns"
        ),
        "child_thread_environment": {name: "1" for name in THREAD_VARIABLES},
    }
    write_json(output / "manifest.json", manifest)
    for job in jobs:
        folder = output / "campaigns" / job["id"]
        write_json(
            output / "jobs" / f"{job['id']}.json",
            {"job": job, "protocol": protocol, "source": source, "folder": str(folder)},
        )
    child_env = {**os.environ, **{name: "1" for name in THREAD_VARIABLES}}
    records = []
    for index, job in enumerate(jobs, 1):
        folder = output / "campaigns" / job["id"]
        request_path = output / "jobs" / f"{job['id']}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.cuda_workflow",
            "campaign",
            "--job",
            str(request_path),
        ]
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, env=child_env, text=True, capture_output=True)
        process_seconds = time.perf_counter() - start
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "stdout.txt").write_text(completed.stdout)
        (folder / "stderr.txt").write_text(completed.stderr)
        write_json(
            folder / "process.json",
            {
                "seconds": process_seconds,
                "returncode": completed.returncode,
                "command": command,
            },
        )
        if completed.returncode:
            write_json(
                output / "failure.json",
                {
                    "job": job,
                    "returncode": completed.returncode,
                    "process_seconds": process_seconds,
                },
            )
            raise RuntimeError(f"Campaign {job['id']} failed; inspect {folder / 'failure.json'}")
        record = _read(folder / "record.json")
        record["timings"] = _read(folder / "timing.json")
        record["process_seconds"] = process_seconds
        if record["source_sha256"] != source["sha256"]:
            raise ValueError("Campaign source identity differs from frozen manifest")
        validate_record(record, protocol, folder=folder)
        records.append(record)
        if progress:
            print(f"Campaign {index}/{len(jobs)}: {job['id']}", flush=True)
    write_json(output / "records.json", records)
    summary = summarize(records, protocol)
    write_json(output / "summary.json", summary)
    validate_summary(summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument(
        "--include-eager", action="store_true", help="Compare preserved CUDA trainer"
    )
    child = commands.add_parser("campaign", help=argparse.SUPPRESS)
    child.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "campaign":
            campaign(_read(args.job))
        else:
            run(args.output, progress=True, include_eager=args.include_eager)
            print(f"Saved workflow comparison: {args.output.resolve()}")
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        parser.exit(1, f"Workflow comparison failed: {error}\n")


if __name__ == "__main__":
    main()
