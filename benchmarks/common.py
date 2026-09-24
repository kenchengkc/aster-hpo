"""Experiment manifests and atomic artifacts shared by benchmark commands."""

import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def environment():
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    versions = {}
    for name in ("aster-hpo", "numpy", "scipy", "matplotlib", "optuna", "ray"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    dirty = git("status", "--porcelain")
    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty": None if dirty is None else bool(dirty),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "dependencies": versions,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def create_experiment(output, manifest):
    output = Path(output)
    # Refuse any existing directory, even if its settings happen to match.
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    write_json(output / "manifest.json", {**manifest, "environment": environment()})
    return output


def validate_seeds(seeds):
    seeds = tuple(seeds)
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty sequence of distinct integers")
    return seeds
