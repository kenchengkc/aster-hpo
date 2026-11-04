"""Reproducible accounting measurements; timings are never correctness evidence.

Run ``python -m benchmarks.native_performance OUTPUT --standalone PATH`` after
installing the optional companion. Use --numpy-only to measure the reference alone.
Every invocation reserves a new directory, saves exact inputs, and checks all
native outputs against NumPy before timing. --objective is a separate, serial,
in-process experiment with the accounting function restored after each call.
"""

import argparse
import contextlib
import hashlib
import importlib
import importlib.machinery
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import hedging
from .common import environment, write_json

ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-12
RTOL = 1e-10


def compare_outputs(expected, actual):
    """Fail on incorrect accounting, including NaN, before any report is written."""
    if len(actual) != 3:
        raise AssertionError("accounting must return pnl, costs, and turnover")
    errors = {}
    for name, reference, result in zip(("pnl", "costs", "turnover"), expected, actual, strict=True):
        result = np.asarray(result)
        if result.shape != reference.shape or not np.isfinite(result).all():
            raise AssertionError(f"{name}: wrong shape or nonfinite output")
        np.testing.assert_allclose(result, reference, atol=ATOL, rtol=RTOL, err_msg=name)
        errors[name] = float(np.max(np.abs(result - reference), initial=0.0))
    return errors


def _checksum(outputs):
    if np.isscalar(outputs):
        return float(outputs)
    return sum(float(np.sum(value, dtype=np.float64)) for value in outputs)


def _file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _native_sources():
    native = ROOT / "native"
    return sorted(
        path
        for path in native.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(("build", ".")) or part in ("__pycache__", "dist")
            for part in path.relative_to(native).parts[:-1]
        )
        and (
            path.suffix in (".cpp", ".hpp", ".py")
            or path.name in ("CMakeLists.txt", "pyproject.toml")
        )
    )


def _provenance(native, standalone):
    sources = _native_sources()
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    binaries = {}
    build_info = None
    if native is not None:
        package_dir = Path(native.__file__).resolve().parent
        for path in package_dir.rglob("*"):
            if path.is_file() and any(
                path.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
            ):
                binaries[path.name] = _file_hash(path)
        build_info = getattr(native, "build_info", None)
        if callable(build_info):
            build_info = build_info()
    if standalone is not None:
        binaries["standalone"] = _file_hash(standalone)
    return {
        "native_source_sha256": digest.hexdigest(),
        "native_source_files": [path.relative_to(ROOT).as_posix() for path in sources],
        "benchmark_source_sha256": _file_hash(__file__),
        "binary_sha256": binaries,
        "native_version": getattr(native, "__version__", None),
        "native_build_info": build_info,
        "native_build_info_note": "unavailable" if build_info is None else "extension supplied",
        "python_compiler": platform.python_compiler(),
        "environment": environment(),
    }


def _workload(paths, intervals, seed):
    # A separate shape-derived RNG means cold child processes reproduce exact bytes.
    rng = np.random.default_rng(np.random.SeedSequence([seed, paths, intervals]))
    prices = rng.uniform(0.5, 2.0, (paths, intervals + 1))
    holdings = rng.uniform(-1.0, 1.0, (paths, intervals))
    digest = hashlib.sha256()
    for value in (prices, holdings):
        digest.update(str(value.shape).encode() + b"\0")
        digest.update(value.tobytes())
    return prices, holdings, digest.hexdigest()


def _backends(native, prices, holdings, rate):
    calls = {"numpy": lambda: hedging.hedge_pnl(prices, holdings, cost_rate=rate)}
    if native is not None:
        outputs = tuple(np.empty(len(prices), dtype=np.float64) for _ in range(3))

        def into():
            native.hedge_pnl_into(prices, holdings, *outputs, cost_rate=rate)
            return outputs

        calls["native_allocating"] = lambda: native.hedge_pnl(prices, holdings, cost_rate=rate)
        calls["native_preallocated"] = into
    return calls


def _quantiles(samples):
    values = np.quantile(samples, (0.5, 0.95, 0.99), method="linear")
    return {name: float(value) for name, value in zip(("p50", "p95", "p99"), values, strict=True)}


def _measure(call, samples):
    durations = []
    cpu_durations = []
    checksum = 0.0
    for _ in range(samples):
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        result = call()
        elapsed = time.perf_counter_ns() - start
        cpu_elapsed = time.process_time_ns() - cpu_start
        if elapsed <= 0 or cpu_elapsed <= 0:
            raise RuntimeError("timer resolution did not produce a positive duration")
        durations.append(elapsed)
        cpu_durations.append(cpu_elapsed)
        checksum += _checksum(result)
    return durations, cpu_durations, checksum


def _load_native(numpy_only):
    if numpy_only:
        return None
    try:
        return importlib.import_module("aster_hpo_native")
    except ImportError as error:
        raise RuntimeError(
            "install the optional aster-hpo-native companion or explicitly pass --numpy-only"
        ) from error


def _cold_child(args):
    import_start = time.perf_counter_ns()
    native = _load_native(args._cold_backend == "numpy")
    import_ns = time.perf_counter_ns() - import_start
    records = []
    # Each process measures one backend and shape: every first call is actually first.
    paths = args.sizes[0]
    prices, holdings, digest = _workload(paths, args.intervals, args.seed)
    call = _backends(native, prices, holdings, args.cost_rate)[args._cold_backend]
    cpu_start = time.process_time_ns()
    start = time.perf_counter_ns()
    outputs = call()
    elapsed = time.perf_counter_ns() - start
    cpu_elapsed = time.process_time_ns() - cpu_start
    errors = compare_outputs(hedging.hedge_pnl(prices, holdings, cost_rate=args.cost_rate), outputs)
    records.append(
        {
            "backend": args._cold_backend,
            "paths": paths,
            "intervals": args.intervals,
            "workload_sha256": digest,
            "first_call_ns": elapsed,
            "first_call_cpu_ns": cpu_elapsed,
            "native_import_ns": import_ns if native is not None else None,
            "checksum": _checksum(outputs),
            "max_absolute_error": errors,
        }
    )
    print(json.dumps(records, allow_nan=False))


def _cold_measurements(args, names):
    records = []
    for repeat in range(args.cold_samples):
        for paths in args.sizes:
            for name in names:
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.native_performance",
                    "--_cold-backend",
                    name,
                    "--sizes",
                    str(paths),
                    "--intervals",
                    str(args.intervals),
                    "--seed",
                    str(args.seed),
                    "--cost-rate",
                    str(args.cost_rate),
                ]
                start = time.perf_counter_ns()
                result = subprocess.run(
                    command, check=True, text=True, capture_output=True, cwd=ROOT
                )
                wall = time.perf_counter_ns() - start
                for record in json.loads(result.stdout):
                    records.append({**record, "repeat": repeat, "process_wall_ns": wall})
    return records


@contextlib.contextmanager
def _accounting_backend(call):
    original = hedging.hedge_pnl
    hedging.hedge_pnl = call
    try:
        yield
    finally:
        hedging.hedge_pnl = original


def _objective_measurements(args, native):
    from aster_hpo import Replication

    market = hedging.Market(
        transaction_cost=args.cost_rate,
        epochs=args.objective_epochs,
        batch_paths=args.objective_batch_paths,
        evaluation_paths=args.objective_evaluation_paths,
        steps=args.intervals,
    )
    config = {"width": 8, "learning_rate": 0.01, "regularization": 1e-4}
    replication = Replication.create(args.seed, config, 0, "search", True)
    calls = {"numpy": hedging.hedge_pnl}
    if native is not None:
        calls["native_allocating"] = native.hedge_pnl
    objective = hedging.HedgingObjective(market)
    records = []
    expected = objective(config, replication)
    for session in range(args.sessions):
        # Interleave backends in each session; all runs use the same replication.
        names = list(calls)
        if session % 2:
            names.reverse()
        for name in names:
            with _accounting_backend(calls[name]):
                for _ in range(args.warmup):
                    objective(config, replication)
                samples, cpu_samples, checksum = _measure(
                    lambda: objective(config, replication), args.samples
                )
                observed = objective(config, replication)
                if not math.isclose(observed, expected, abs_tol=ATOL, rel_tol=RTOL):
                    raise AssertionError(f"whole objective mismatch for {name}")
            records.append(
                {
                    "backend": name,
                    "objective_id": f"hedging-native-comparison-v1:{name}",
                    "session": session,
                    "samples_ns": samples,
                    "cpu_samples_ns": cpu_samples,
                    "call_quantiles_ns": _quantiles(samples),
                    "checksum": checksum,
                    "loss": observed,
                }
            )
    return {
        "boundary": "complete serial HedgingObjective call; RNG, training and evaluation included",
        "backend_selection": "temporary module replacement, restored after each backend",
        "market": asdict(market),
        "config": config,
        "replication": asdict(replication),
        "records": records,
    }


def run(args):
    if args.output is None:
        raise ValueError("an output directory is required")
    output = Path(args.output)
    # Reserve the destination before imports, subprocesses, or expensive measurements.
    output.mkdir(parents=True, exist_ok=False)
    native = _load_native(args.numpy_only)
    standalone_path = Path(args.standalone).resolve() if args.standalone else None
    settings = {
        "sizes": args.sizes,
        "intervals": args.intervals,
        "cost_rate": args.cost_rate,
        "seed": args.seed,
        "samples_per_session": args.samples,
        "independent_process_sessions": False,
        "sessions": args.sessions,
        "session_note": "rounds in one process; separate invocations are independent sessions",
        "warmup_calls_per_backend_per_session": args.warmup,
        "cold_samples_per_shape_backend": args.cold_samples,
        "processes": 1,
        "thread_policy": "inherited host configuration, recorded without changing thread pools",
        "host_power_load": args.host_notes,
        "atol": ATOL,
        "rtol": RTOL,
    }
    provenance = _provenance(native, standalone_path)
    write_json(output / "manifest.json", {"settings": settings, "provenance": provenance})
    timer = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        timer.append(time.perf_counter_ns() - start)
    records = []
    workload_hashes = {}
    backend_names = ["numpy"]
    if native is not None:
        backend_names.extend(("native_allocating", "native_preallocated"))
    for paths in args.sizes:
        prices, holdings, digest = _workload(paths, args.intervals, args.seed)
        workload_hashes[paths] = digest
        filename = f"inputs-{paths}x{args.intervals}.npz"
        np.savez(output / filename, prices=prices, holdings=holdings)
        expected = hedging.hedge_pnl(prices, holdings, cost_rate=args.cost_rate)
        calls = _backends(native, prices, holdings, args.cost_rate)
        errors = {name: compare_outputs(expected, call()) for name, call in calls.items()}
        for session in range(args.sessions):
            names = backend_names if session % 2 == 0 else backend_names[::-1]
            for name in names:
                call = calls[name]
                for _ in range(args.warmup):
                    call()
                samples, cpu_samples, checksum = _measure(call, args.samples)
                records.append(
                    {
                        "backend": name,
                        "boundary": (
                            "checked Python call including output allocation"
                            if name != "native_preallocated"
                            else "checked Python call; caller-owned outputs; allocation excluded"
                        ),
                        "session": session,
                        "paths": paths,
                        "intervals": args.intervals,
                        "workload_sha256": digest,
                        "input_file": filename,
                        "samples_ns": samples,
                        "cpu_samples_ns": cpu_samples,
                        "call_quantiles_ns": _quantiles(samples),
                        "checksum": checksum,
                        "max_absolute_error": errors[name],
                    }
                )
    cold = _cold_measurements(args, backend_names)
    for record in cold:
        if record["workload_sha256"] != workload_hashes[record["paths"]]:
            raise AssertionError("cold process used different input arrays")
    standalone = None
    if standalone_path:
        completed = subprocess.run(
            [
                str(standalone_path),
                "--samples",
                str(args.samples),
                "--warmup",
                str(args.warmup),
                "--intervals",
                str(args.intervals),
                "--paths",
                str(max(args.sizes)),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        standalone = json.loads(completed.stdout)
    report = {
        "schema_version": 1,
        "interpretation": "local accounting service time and Python API cost",
        "settings": settings,
        "provenance": provenance,
        "timer": {
            "clock": "perf_counter_ns",
            "samples_ns": timer,
            "quantiles_ns": _quantiles(timer),
            "subtracted": False,
        },
        "quantile_method": "linear interpolation (NumPy method='linear')",
        "api": records,
        "cold": cold,
        "cold_boundary": (
            "process_wall_ns includes interpreter startup, imports, exact input generation, first "
            "completed call, correctness check, serialization, and exit; first_call_ns excludes "
            "imports/input generation and includes API validation/allocation as applicable"
        ),
        "standalone": standalone,
        "objective": _objective_measurements(args, native) if args.objective else None,
    }
    try:
        from threadpoolctl import threadpool_info

        report["thread_pools"] = threadpool_info()
    except ImportError:
        report["thread_pools"] = "threadpoolctl unavailable"
    write_json(output / "report.json", report)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", nargs="?", help="new output directory; existing paths are rejected"
    )
    parser.add_argument("--numpy-only", action="store_true", help="explicitly omit native backends")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 256, 2048, 65536])
    parser.add_argument("--intervals", type=int, default=12)
    parser.add_argument("--cost-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--cold-samples", type=int, default=1)
    parser.add_argument("--standalone", help="optional uninstrumented accounting_bench executable")
    parser.add_argument(
        "--objective", action="store_true", help="also compare complete objective calls"
    )
    parser.add_argument("--objective-epochs", type=int, default=256)
    parser.add_argument("--objective-batch-paths", type=int, default=256)
    parser.add_argument("--objective-evaluation-paths", type=int, default=2048)
    parser.add_argument(
        "--host-notes", default="host load/power not controlled or recorded by operator"
    )
    parser.add_argument(
        "--_cold-backend",
        choices=("numpy", "native_allocating", "native_preallocated"),
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    for name in (
        "intervals",
        "samples",
        "warmup",
        "sessions",
        "cold_samples",
        "objective_epochs",
        "objective_batch_paths",
        "objective_evaluation_paths",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if (
        not args.sizes
        or any(value < 1 for value in args.sizes)
        or len(set(args.sizes)) != len(args.sizes)
    ):
        parser.error("sizes must be distinct positive integers")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if not math.isfinite(args.cost_rate) or args.cost_rate < 0:
        parser.error("cost-rate must be finite and nonnegative")
    try:
        if args._cold_backend:
            _cold_child(args)
        else:
            run(args)
            print(f"Wrote measurements to {Path(args.output).resolve()}")
    except (
        OSError,
        ValueError,
        RuntimeError,
        AssertionError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"measurement failed: {error}\n")


if __name__ == "__main__":
    main()
