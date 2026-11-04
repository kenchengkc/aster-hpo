"""Repeated complete HPO workflows with direct, fresh and reused worker controls.

Run ``python -m benchmarks.persistent_workflow run --output NEW_DIRECTORY``.
The fixed protocol uses 48 fresh coordinator processes and 144 new workflows.
Each coordinator runs three searches and independent winner audits without warmup.
CUDA is initialized only by the actual evaluating process inside the timed work.
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .common import write_json
from .cuda_workflow import (
    ATOL,
    METRICS,
    ROOT,
    RTOL,
    THREAD_VARIABLES,
    _backend_environment,
    _close,
    _collect_metrics,
    _digest,
    _finite,
    _identity,
    _observations,
    _partial_journal,
    _range,
    _read,
    _validate_search,
    source_identity,
    validate_record,
)

# A fresh import in each spawned interpreter makes PID reuse distinguishable.
_PROCESS_NONCE = uuid.uuid4().hex


def _numerical_modules():
    return [name for name in ("numpy", "cupy", "benchmarks.hedging_cuda") if name in sys.modules]


def default_protocol():
    """Freeze equal CPU persistence controls and the already persistent direct GPU."""
    # This runs in the untimed parent; child imports remain inside its first timer.
    from .hedging import Market

    tiny = asdict(Market())
    return {
        "schema_version": 1,
        "kind": "Repeated complete hedging HPO workflows",
        "markets": {
            "tiny": tiny,
            "scaled": {**tiny, "batch_paths": 4096, "evaluation_paths": 32768},
        },
        "sequence_roots": [41000, 42000, 43000],
        "workflow_offsets": [0, 1, 2],
        "candidate_pool_seed_policy": (
            "Fixed candidate pool from sequence root; each fresh workflow uses study seed "
            "root plus offset for new training/search/audit seeds and fresh journals"
        ),
        "variants": [
            {"mode": "direct", "backend": "numpy", "workers": 1},
            {"mode": "direct", "backend": "cuda", "workers": 1},
            {"mode": "fresh_pool", "backend": "numpy", "workers": 1},
            {"mode": "fresh_pool", "backend": "cuda", "workers": 1},
            {"mode": "fresh_pool", "backend": "numpy", "workers": 4},
            {"mode": "reused_pool", "backend": "numpy", "workers": 1},
            {"mode": "reused_pool", "backend": "cuda", "workers": 1},
            {"mode": "reused_pool", "backend": "numpy", "workers": 4},
        ],
        "count": 4,
        "checkpoints": [4],
        "audit_replications": 3,
        "threads_per_worker": 1,
        "closeness": {"rtol": RTOL, "atol": ATOL},
        "warmup_objective_calls": 0,
        "cache_scope": (
            "No objective warmup or cache purge. Shared compiler and disk caches are retained; "
            "a fresh process does not imply an empty compiler cache. Direct CUDA in one "
            "interpreter already retains its CUDA context and process caches."
        ),
        "interpretation": (
            "Three outer sequence processes are independent timing units. First-run, "
            "subsequent-run and complete lifecycle costs are descriptive development evidence. "
            "Reusable managed workers do not imply a universal improvement over direct CUDA."
        ),
    }


def _variant(job):
    return f"{job['mode']}-{job['backend']}-w{job['workers']}"


def build_jobs(protocol):
    """Deterministic interleaving, frozen before any sequence timing starts."""
    jobs = []
    for root in protocol["sequence_roots"]:
        variants = list(protocol["variants"])
        random.Random(root).shuffle(variants)
        workloads = list(protocol["markets"])
        if root % 2:
            workloads.reverse()
        for variant in variants:
            for workload in workloads:
                jobs.append(
                    {
                        "id": f"{workload}-{root}-{_variant(variant)}",
                        "workload": workload,
                        "root_seed": root,
                        "scheduler": "full",
                        **variant,
                    }
                )
    return jobs


@dataclass(frozen=True)
class ObservedObjective:
    """Picklable objective with one safe, worker-side identity file per run/PID."""

    objective: object
    identity_dir: Path

    def __call__(self, config, replication):
        path = Path(self.identity_dir) / f"worker-{os.getpid()}.json"
        if not path.exists():
            backend = {"name": self.objective.backend}
            if self.objective.backend == "cuda":
                from .hedging_cuda import device_info

                backend.update(device=device_info(), implementation="optimized")
            payload = {
                "pid": os.getpid(),
                "parent_pid": os.getppid(),
                "process_nonce": f"{_PROCESS_NONCE}:{os.getpid()}",
                "backend": backend,
                "numerical_modules": _numerical_modules(),
                "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as handle:
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.write("\n")
        return self.objective(config, replication)


def _worker_identities(folder):
    return [_read(path) for path in sorted((folder / "workers").glob("worker-*.json"))]


def _process_keys(run):
    return {(row["pid"], row["process_nonce"]) for row in run["worker_identities"]}


def _reuse(runs):
    keys = [_process_keys(run) for run in runs]
    return {
        "observed_processes": [
            [{"pid": pid, "process_nonce": nonce} for pid, nonce in sorted(values)]
            for values in keys
        ],
        "adjacent_reused_processes": [len(a & b) for a, b in zip(keys, keys[1:], strict=False)],
        "adjacent_new_processes": [len(b - a) for a, b in zip(keys, keys[1:], strict=False)],
        "unique_processes": len(set().union(*keys)),
    }


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _workflow(request, index, pool, wall_start, cpu_start):
    # On the first workflow these imports and pool creation are inside wall_start.
    from aster_hpo import FullBudget, Study, audit

    from .hedging import HedgingObjective, Market, hedging_pool

    protocol, sequence_job = request["protocol"], request["job"]
    folder = Path(request["folder"]) / f"workflow-{index}"
    folder.mkdir(parents=True, exist_ok=False)
    seed = sequence_job["root_seed"] + protocol["workflow_offsets"][index]
    job = {
        **sequence_job,
        "seed": seed,
        "pool_seed": sequence_job["root_seed"],
        "id": f"{sequence_job['id']}-workflow-{index}",
    }
    objective = ObservedObjective(
        HedgingObjective(
            Market(**protocol["markets"][job["workload"]]),
            folder / "metrics",
            backend=job["backend"],
        ),
        folder / "workers",
    )
    settings = {
        "seed": seed,
        "threads_per_worker": protocol["threads_per_worker"],
        "objective_id": f"hedging-repeated-workflow:{job['backend']}:" + _digest(protocol),
        "max_retries": 0,
    }
    pooled = {"worker_pool": pool} if pool is not None else {}
    timings = {"backend_setup_seconds": time.perf_counter() - wall_start}
    start = time.perf_counter()
    result = Study(
        hedging_pool(job["pool_seed"], protocol["count"]),
        scheduler=FullBudget(protocol["checkpoints"][-1]),
        workers=job["workers"],
        storage=folder / "search.sqlite",
        **settings,
    ).optimize(objective, **pooled)
    timings["search_seconds"] = time.perf_counter() - start
    result.save(folder / "search.json")
    search = _read(folder / "search.json")
    winner, count = _validate_search(search, job, protocol)
    with (folder / "winner.json").open("x") as handle:
        json.dump(winner["config"], handle, indent=2, allow_nan=False)
        handle.write("\n")
    start = time.perf_counter()
    audited = audit(
        objective,
        winner["config"],
        replications=protocol["audit_replications"],
        workers=1,
        storage=folder / "audit.sqlite",
        **settings,
        **pooled,
    )
    timings["audit_seconds"] = time.perf_counter() - start
    timings["pool_close_seconds"] = 0.0
    if job["mode"] == "fresh_pool":
        start = time.perf_counter()
        pool.close()
        timings["pool_close_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    audited = asdict(audited)
    write_json(folder / "audit.json", audited)
    record = {
        "schema_version": 1,
        "job": job,
        "workflow_index": index,
        "folder": f"workflow-{index}",
        "protocol_sha256": _digest(protocol),
        "source_sha256": request["source"]["sha256"],
        "campaign_pid": os.getpid(),
        "backend": {
            "name": job["backend"],
            **({"implementation": "optimized"} if job["backend"] == "cuda" else {}),
        },
        "winner": winner["config"],
        "winner_trial_id": winner["id"],
        "search": search,
        "audit": audited,
        "counts": {
            "search_observations": count,
            "audit_observations": len(audited["observations"]),
            "objective_failures": 0,
            "unsuccessful_attempts": 0,
        },
        "worker_identities": _worker_identities(folder),
    }
    record["metrics"] = _collect_metrics(folder, record)
    validate_record(record, protocol, folder=folder)
    write_json(folder / "record.json", record)
    timings["result_write_seconds"] = time.perf_counter() - start
    timings["workflow_seconds"] = time.perf_counter() - wall_start
    timings["host_cpu_seconds"] = time.process_time() - cpu_start
    write_json(folder / "timing.json", timings)
    record["timings"] = timings
    return record


def sequence(request):
    """Internal fresh coordinator entry point; no hidden initialization or warmup."""
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    preloaded = _numerical_modules()
    protocol, job = request["protocol"], request["job"]
    if source_identity() != request["source"]:
        raise ValueError("Frozen source identity differs before sequence")
    if job not in build_jobs(protocol):
        raise ValueError("Sequence job differs from frozen protocol")
    folder = Path(request["folder"])
    folder.mkdir(parents=True, exist_ok=False)
    pool, runs = None, []
    try:
        for index, _ in enumerate(protocol["workflow_offsets"]):
            run_wall, run_cpu = (
                (wall_start, cpu_start)
                if index == 0
                else (time.perf_counter(), time.process_time())
            )
            if job["mode"] != "direct" and (pool is None or job["mode"] == "fresh_pool"):
                from aster_hpo import WorkerPool

                pool = WorkerPool(
                    workers=job["workers"], threads_per_worker=protocol["threads_per_worker"]
                )
            runs.append(_workflow(request, index, pool, run_wall, run_cpu))
        start = time.perf_counter()
        if pool is not None:
            pool.close()
        final_close = time.perf_counter() - start if job["mode"] == "reused_pool" else 0.0
        managed_pids = sorted(
            {
                row["pid"]
                for run in runs
                for row in run["worker_identities"]
                if row["pid"] != os.getpid()
            }
        )
        surviving = [pid for pid in managed_pids if _alive(pid)]
        if surviving:
            raise ValueError(f"Managed workers survived close: {surviving}")
        if source_identity() != request["source"]:
            raise ValueError("Frozen source changed during sequence")
        record = {
            "schema_version": 1,
            "job": job,
            "protocol_sha256": _digest(protocol),
            "source_sha256": request["source"]["sha256"],
            "coordinator_pid": os.getpid(),
            "preloaded_numerical_modules": preloaded,
            "runs": runs,
            "reuse": _reuse(runs),
            "cleanup": {
                "managed_worker_pids": managed_pids,
                "surviving_worker_pids": surviving,
                "pool_closed": pool is None or pool.closed,
            },
            "timing_scope": {
                "first_workflow": (
                    "Before numerical imports and pool creation through result writes"
                ),
                "subsequent_workflow": (
                    "Fresh objective and paths through search, winner lock, independent audit, "
                    "result writes and fresh-pool close"
                ),
                "whole_sequence": (
                    "All setup, three workflows, per-run timing writes and final pool close; "
                    "only the final timing trailer is excluded"
                ),
                "process": "Parent clock includes interpreter startup, all writes and exit",
                "cpu": "Coordinator host CPU; excludes child worker CPU and GPU compute",
                "warmup_objective_calls": 0,
            },
        }
        write_json(folder / "sequence.json", record)
        durations = [run["timings"]["workflow_seconds"] for run in runs]
        total = time.perf_counter() - wall_start
        timings = {
            "first_workflow_seconds": durations[0],
            "subsequent_workflow_seconds": durations[1:],
            "subsequent_median_seconds": statistics.median(durations[1:]),
            "final_pool_close_seconds": final_close,
            "whole_sequence_seconds": total,
            "amortized_workflow_seconds": total / len(runs),
            "host_cpu_seconds": time.process_time() - cpu_start,
        }
        write_json(folder / "sequence-timing.json", timings)
        record["timings"] = timings
        validate_sequence(record, protocol, folder=folder)
        return record
    except BaseException as error:
        if pool is not None and not pool.closed:
            pool.close()
        write_json(
            folder / "failure.json",
            {
                "job": job,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_sequence_seconds": time.perf_counter() - wall_start,
                "completed_workflows": len(runs),
                "journals": {
                    path.name: {
                        phase: _partial_journal(path / f"{phase}.sqlite")
                        for phase in ("search", "audit")
                    }
                    for path in sorted(folder.glob("workflow-*"))
                    if path.is_dir()
                },
            },
        )
        raise


def validate_sequence(record, protocol, *, folder=None):
    """Reject partial work, false reuse, missing diagnostics and omitted lifecycle costs."""
    try:
        job, runs = record["job"], record["runs"]
        if job not in build_jobs(protocol) or record["protocol_sha256"] != _digest(protocol):
            raise ValueError("Sequence job/protocol identity differs")
        if record["preloaded_numerical_modules"]:
            raise ValueError("Numerical modules initialized before first-run timer")
        if len(runs) != len(protocol["workflow_offsets"]) or not runs:
            raise ValueError("Missing workflow records")
        pool_ids, lease_ids = [], set()
        for index, run in enumerate(runs):
            if (
                run["workflow_index"] != index
                or run["folder"] != f"workflow-{index}"
                or run["job"]["seed"] != job["root_seed"] + protocol["workflow_offsets"][index]
                or run["job"]["pool_seed"] != job["root_seed"]
                or any(
                    run["job"][key] != job[key]
                    for key in ("workload", "mode", "backend", "workers", "scheduler", "root_seed")
                )
                or run["campaign_pid"] != record["coordinator_pid"]
                or run["source_sha256"] != record["source_sha256"]
            ):
                raise ValueError("Workflow identity differs from sequence")
            path = Path(folder) / run["folder"] if folder is not None else None
            validate_record(run, protocol, folder=path)
            identities = run["worker_identities"]
            if not identities or len(_process_keys(run)) != len(identities):
                raise ValueError("Missing or duplicate worker identity")
            observed_pids = {}
            for phase in ("search", "audit"):
                observations = (
                    [obs for trial in run["search"]["trials"] for obs in trial["observations"]]
                    if phase == "search"
                    else run["audit"]["observations"]
                )
                if any(
                    type(obs["worker_pid"]) is not int or obs["worker_pid"] <= 0
                    for obs in observations
                ):
                    raise ValueError("Invalid accepted observation worker PID")
                observed_pids[phase] = {obs["worker_pid"] for obs in observations}
            for identity in identities:
                pid, nonce = identity["pid"], identity["process_nonce"]
                if (
                    type(pid) is not int
                    or pid <= 0
                    or not isinstance(nonce, str)
                    or not nonce
                    or identity["backend"]["name"] != job["backend"]
                    or identity["thread_environment"]
                    != {name: str(protocol["threads_per_worker"]) for name in THREAD_VARIABLES}
                ):
                    raise ValueError("Invalid worker PID, nonce, backend or thread evidence")
                if job["backend"] == "cuda" and not identity["backend"].get("device"):
                    raise ValueError("Missing worker CUDA device identity")
                if job["backend"] == "numpy" and identity["numerical_modules"] != ["numpy"]:
                    raise ValueError("CPU worker imported a GPU module")
                if job["mode"] == "direct" and pid != record["coordinator_pid"]:
                    raise ValueError("Direct objective PID differs from coordinator")
                if job["mode"] != "direct" and pid == record["coordinator_pid"]:
                    raise ValueError("Managed objective executed in coordinator")
            if set().union(*observed_pids.values()) != {row["pid"] for row in identities}:
                raise ValueError("Accepted observation PIDs differ from worker identity evidence")
            if path is not None:
                if _worker_identities(path) != identities:
                    raise ValueError("Worker identity files differ from saved evidence")
                if not (path / "search.sqlite").is_file() or not (path / "audit.sqlite").is_file():
                    raise ValueError("Missing fresh workflow journals")
                if _read(path / "winner.json") != run["winner"]:
                    raise ValueError("Locked winner artifact differs")
            if job["mode"] != "direct":
                phase_pool_ids, phase_pids = set(), set()
                for phase in ("search", "audit"):
                    metadata = run[phase]["metadata"]
                    if metadata["execution_backend"] != "persistent_processes":
                        raise ValueError("Missing persistent execution backend evidence")
                    pool_id, lease_id = metadata["worker_pool_id"], metadata["worker_pool_run_id"]
                    pids = metadata["worker_pool_pids"]
                    expected_workers = job["workers"] if phase == "search" else 1
                    if (
                        not isinstance(pool_id, str)
                        or not pool_id
                        or not isinstance(lease_id, str)
                        or not lease_id
                        or lease_id in lease_ids
                        or metadata["worker_pool_capacity"] != job["workers"]
                        or metadata["workers"] != expected_workers
                        or metadata["threads_per_worker"] != protocol["threads_per_worker"]
                        or not pids
                        or any(type(pid) is not int or pid <= 0 for pid in pids)
                        or len(pids) != len(set(pids))
                        or len(pids) > expected_workers
                        or set(pids) != observed_pids[phase]
                        or not set(pids).issubset({row["pid"] for row in identities})
                    ):
                        raise ValueError("Invalid pool lease, capacity or worker PID evidence")
                    phase_pool_ids.add(pool_id)
                    lease_ids.add(lease_id)
                    phase_pids.update(pids)
                if len(phase_pool_ids) != 1 or phase_pids != {row["pid"] for row in identities}:
                    raise ValueError("Search/audit pool or worker PID evidence differs")
                pool_ids.append(next(iter(phase_pool_ids)))
            else:
                for phase in ("search", "audit"):
                    if run[phase]["metadata"]["execution_backend"] != "in_process":
                        raise ValueError("Direct workflow execution backend differs")
                    if observed_pids[phase] != {record["coordinator_pid"]}:
                        raise ValueError("Direct observation PID differs from coordinator")
            close = run["timings"]["pool_close_seconds"]
            _finite(close, "workflow pool close", positive=job["mode"] == "fresh_pool")
            if job["mode"] != "fresh_pool" and close != 0:
                raise ValueError("Unexpected per-workflow pool close")
            measured = sum(
                run["timings"][part]
                for part in (
                    "backend_setup_seconds",
                    "search_seconds",
                    "audit_seconds",
                    "result_write_seconds",
                    "pool_close_seconds",
                )
            )
            if run["timings"]["workflow_seconds"] < measured:
                raise ValueError("Workflow timing omits pool lifecycle")
        evidence = _reuse(runs)
        if record["reuse"] != evidence:
            raise ValueError("Saved process reuse differs from PID and nonce evidence")
        reused = evidence["adjacent_reused_processes"]
        if job["mode"] == "fresh_pool" and any(reused):
            raise ValueError("Fresh pools reused processes across workflows")
        if job["mode"] in ("direct", "reused_pool") and not all(reused):
            raise ValueError("Expected process reuse was not observed")
        if job["mode"] == "reused_pool" and len(set(pool_ids)) != 1:
            raise ValueError("Reused workflows changed worker pool identity")
        if job["mode"] == "fresh_pool" and len(set(pool_ids)) != len(runs):
            raise ValueError("Fresh workflows reused worker pool identity")
        managed = sorted(
            {
                row["pid"]
                for run in runs
                for row in run["worker_identities"]
                if row["pid"] != record["coordinator_pid"]
            }
        )
        if record["cleanup"] != {
            "managed_worker_pids": managed,
            "surviving_worker_pids": [],
            "pool_closed": True,
        }:
            raise ValueError("Incomplete worker cleanup evidence")
        times = record["timings"]
        durations = [run["timings"]["workflow_seconds"] for run in runs]
        for name in (
            "first_workflow_seconds",
            "subsequent_median_seconds",
            "whole_sequence_seconds",
            "amortized_workflow_seconds",
        ):
            _finite(times[name], name, positive=True)
        _finite(
            times["final_pool_close_seconds"],
            "final pool close",
            positive=job["mode"] == "reused_pool",
        )
        if (
            times["first_workflow_seconds"] != durations[0]
            or times["subsequent_workflow_seconds"] != durations[1:]
            or times["subsequent_median_seconds"] != statistics.median(durations[1:])
            or times["whole_sequence_seconds"]
            < (sum(durations) + times["final_pool_close_seconds"])
            or times["amortized_workflow_seconds"] != (times["whole_sequence_seconds"] / len(runs))
        ):
            raise ValueError("Sequence timing omits complete workflow or lifecycle costs")
        if "process_seconds" in record:
            _finite(record["process_seconds"], "process seconds", positive=True)
            if record["process_seconds"] < times["whole_sequence_seconds"]:
                raise ValueError("Parent process clock is shorter than sequence")
    except (KeyError, TypeError, OverflowError, statistics.StatisticsError) as error:
        raise ValueError(f"Incomplete or invalid sequence record: {error}") from error
    return record


def _quality(reference, candidate, protocol):
    matched, mismatches, winners, differences = 0, [], 0, []
    for a, b in zip(reference["runs"], candidate["runs"], strict=True):
        rows = []
        for run in (a, b):
            metrics = {
                _identity(row["config"], row["replication"]): row["metrics"]
                for row in run["metrics"]
            }
            expected = {_identity(config, obs["replication"]) for config, obs in _observations(run)}
            if metrics.keys() != expected or len(metrics) != len(run["metrics"]):
                raise ValueError("Missing or duplicate complete financial metric evidence")
            rows.append(metrics)
        first, second = rows
        search_a = {key for key in first if key[-1] == "search"}
        search_b = {key for key in second if key[-1] == "search"}
        if search_a != search_b:
            raise ValueError("Matched search identities differ")
        keys = first.keys() & second.keys()
        for key in sorted(keys):
            for metric in sorted(METRICS):
                matched += 1
                if not _close(first[key][metric], second[key][metric], protocol):
                    mismatches.append(
                        {
                            "seed": a["job"]["seed"],
                            "identity": list(key),
                            "metric": metric,
                            "reference": first[key][metric],
                            "candidate": second[key][metric],
                        }
                    )
        winners += a["winner"] == b["winner"]
        differences.append(
            {
                "seed": a["job"]["seed"],
                "winner_agreement": a["winner"] == b["winner"],
                "reference_winner": a["winner"],
                "candidate_winner": b["winner"],
                "reference_audit_mean": statistics.mean(
                    o["loss"] for o in a["audit"]["observations"]
                ),
                "candidate_audit_mean": statistics.mean(
                    o["loss"] for o in b["audit"]["observations"]
                ),
                "unmatched_audit_metrics": [
                    len(first.keys() - second.keys()),
                    len(second.keys() - first.keys()),
                ],
            }
        )
    return {
        "count": matched,
        "mismatches": len(mismatches),
        "all_close": not mismatches,
        "details": mismatches,
        "metrics": sorted(METRICS),
        **protocol["closeness"],
    }, {
        "winner_agreement_count": winners,
        "audits": differences,
    }


def summarize(records, protocol):
    """Pair outer sequences before ratios; never pool the six subsequent positions."""
    index = {}
    for record in records:
        job = record["job"]
        key = job["workload"], job["root_seed"], _variant(job)
        if key in index:
            raise ValueError("duplicate sequence identity")
        validate_sequence(record, protocol)
        index[key] = record
    expected = {(job["workload"], job["root_seed"], _variant(job)) for job in build_jobs(protocol)}
    if index.keys() != expected:
        raise ValueError("Missing or unpaired frozen sequence records")
    comparisons = []
    variants = [_variant(variant) for variant in protocol["variants"]]
    reference_name = "direct-numpy-w1"
    comparisons_to_make = [(reference_name, name) for name in variants if name != reference_name]
    # Compare CPU and GPU after granting both the same cross-workflow persistence.
    reused_cuda = "reused_pool-cuda-w1"
    for workers in (1, 4):
        reused_cpu = f"reused_pool-numpy-w{workers}"
        if reused_cpu in variants and reused_cuda in variants:
            comparisons_to_make.append((reused_cpu, reused_cuda))
    # Isolate startup amortization at the same backend and worker count as well.
    for variant in protocol["variants"]:
        if variant["mode"] != "reused_pool":
            continue
        candidate = _variant(variant)
        fresh = _variant({**variant, "mode": "fresh_pool"})
        direct = _variant({**variant, "mode": "direct"})
        comparisons_to_make.extend(
            (name, candidate)
            for name in (fresh, direct)
            if name in variants and (name, candidate) not in comparisons_to_make
        )
    timing_names = ("first_workflow", "subsequent_median", "whole_sequence", "amortized_workflow")
    for workload in protocol["markets"]:
        for reference_name, candidate_name in comparisons_to_make:
            pairs, metric_count, failures, audits = [], 0, [], []
            for root in protocol["sequence_roots"]:
                reference = index[workload, root, reference_name]
                candidate = index[workload, root, candidate_name]
                quality, audit = _quality(reference, candidate, protocol)
                metric_count += quality["count"]
                failures.extend(quality["details"])
                audits.append({"root_seed": root, **audit})
                pairs.append(
                    {
                        "root_seed": root,
                        "reference_job": reference["job"]["id"],
                        "candidate_job": candidate["job"]["id"],
                        **{
                            f"{name}_speedup": reference["timings"][f"{name}_seconds"]
                            / candidate["timings"][f"{name}_seconds"]
                            for name in timing_names
                        },
                        "process_speedup": reference["process_seconds"]
                        / candidate["process_seconds"]
                        if "process_seconds" in reference and "process_seconds" in candidate
                        else None,
                    }
                )
            process = [
                pair["process_speedup"] for pair in pairs if pair["process_speedup"] is not None
            ]
            comparisons.append(
                {
                    "workload": workload,
                    "reference": reference_name,
                    "candidate": candidate_name,
                    "independent_sequences": len(pairs),
                    "pairs": pairs,
                    **{
                        f"{name}_speedup": _range([pair[f"{name}_speedup"] for pair in pairs])
                        for name in timing_names
                    },
                    "process_speedup": _range(process) if process else None,
                    "matched_metrics": {
                        "count": metric_count,
                        "mismatches": len(failures),
                        "all_close": not failures,
                        "details": failures,
                        "metrics": sorted(METRICS),
                        **protocol["closeness"],
                    },
                    "winner_audits": audits,
                }
            )
    return {
        "schema_version": 1,
        "sequence_count": len(records),
        "workflow_count": sum(len(record["runs"]) for record in records),
        "objective_evaluations": sum(
            sum(run["counts"][name] for name in ("search_observations", "audit_observations"))
            for record in records
            for run in record["runs"]
        ),
        "comparisons": comparisons,
        "interpretation": (
            "Reference seconds / candidate seconds, paired by workload and outer sequence root. "
            "Each sequence contributes one median of its two subsequent workflows. The three "
            "outer processes, rather than six warm positions, are the independent units. "
            "Different valid locked winners retain descriptive independent audit outcomes. "
            "Direct CUDA already persists in its coordinator interpreter. First-use, full "
            "sequence, amortized and parent process times retain setup and shutdown costs."
        ),
    }


def validate_summary(summary):
    if any(not comparison["matched_metrics"]["all_close"] for comparison in summary["comparisons"]):
        raise ValueError("Matched financial metrics failed the frozen closeness check")
    return summary


def run(output, *, progress=False):
    # Do not resolve a symlink away before the exclusive destination check.
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    protocol, source = default_protocol(), source_identity()
    jobs = build_jobs(protocol)
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "frozen_at": datetime.now(UTC).isoformat(),
            "protocol": protocol,
            "protocol_sha256": _digest(protocol),
            "source": source,
            "environment": _backend_environment(),
            "jobs": jobs,
            "child_thread_environment": {
                name: str(protocol["threads_per_worker"]) for name in THREAD_VARIABLES
            },
            "timing": "Protocol, source hashes, job order and environment frozen before sequences",
        },
    )
    for job in jobs:
        write_json(
            output / "jobs" / f"{job['id']}.json",
            {
                "job": job,
                "protocol": protocol,
                "source": source,
                "folder": str(output / "sequences" / job["id"]),
            },
        )
    child_env = {
        **os.environ,
        **{name: str(protocol["threads_per_worker"]) for name in THREAD_VARIABLES},
    }
    records = []
    for index, job in enumerate(jobs, 1):
        folder = output / "sequences" / job["id"]
        command = [
            sys.executable,
            "-m",
            "benchmarks.persistent_workflow",
            "sequence",
            "--job",
            str(output / "jobs" / f"{job['id']}.json"),
        ]
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, env=child_env, text=True, capture_output=True)
        seconds = time.perf_counter() - start
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "stdout.txt").write_text(completed.stdout)
        (folder / "stderr.txt").write_text(completed.stderr)
        write_json(
            folder / "process.json",
            {"seconds": seconds, "returncode": completed.returncode, "command": command},
        )
        if completed.returncode:
            write_json(
                output / "failure.json",
                {"job": job, "returncode": completed.returncode, "process_seconds": seconds},
            )
            raise RuntimeError(f"Sequence {job['id']} failed; inspect {folder}")
        record = _read(folder / "sequence.json")
        record["timings"] = _read(folder / "sequence-timing.json")
        record["process_seconds"] = seconds
        if record["source_sha256"] != source["sha256"]:
            raise ValueError("Sequence source identity differs from frozen manifest")
        validate_sequence(record, protocol, folder=folder)
        records.append(record)
        if progress:
            print(f"Sequence {index}/{len(jobs)}: {job['id']}", flush=True)
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
    child = commands.add_parser("sequence", help=argparse.SUPPRESS)
    child.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "sequence":
            sequence(_read(args.job))
        else:
            run(args.output, progress=True)
            print(f"Saved repeated workflow comparison: {args.output.absolute()}")
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        parser.exit(1, f"Repeated workflow comparison failed: {error}\n")


if __name__ == "__main__":
    main()
