"""Freeze financial study settings before collecting comparative results."""

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

from aster_hpo import ASHA

from .common import environment, validate_seeds
from .hedging import Market, hedging_pool

METHODS = ("serial_full", "parallel_full", "asha", "race")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    sources = [*root.glob("src/aster_hpo/*.py"), *root.glob("benchmarks/*.py")]
    return digest({str(path.relative_to(root)): path.read_text() for path in sorted(sources)})


def freeze_protocol(
    pilot,
    destination,
    *,
    seeds=None,
    count=6,
    workers=4,
    checkpoints=(3, 9),
    audit_replications=8,
    margin_fraction=0.01,
    scaling_workers=None,
):
    seeds = validate_seeds(range(20) if seeds is None else seeds)
    pilot = Path(pilot)
    pilot_manifest = json.loads((pilot / "manifest.json").read_text())
    pilot_summary = json.loads((pilot / "summary.json").read_text())
    if pilot_manifest["kind"] != "financial pilot" or pilot_manifest["seed"] in seeds:
        raise ValueError("Final seeds must be disjoint from the financial pilot")
    market = Market(**pilot_manifest["market"])
    hedging_pool(seeds[0], count)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if type(audit_replications) is not int or audit_replications < 2:
        raise ValueError("At least two audit replications are required")
    if len(checkpoints) < 2:
        raise ValueError("Specify at least two geometric checkpoints")
    factor = checkpoints[1] // checkpoints[0]
    ASHA(checkpoints, reduction_factor=factor)
    reference = pilot_summary["full_budget"]["audit_mean"]
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Pilot reference loss must be finite and positive")
    if not math.isfinite(margin_fraction) or margin_fraction <= 0:
        raise ValueError("margin_fraction must be finite and positive")
    if scaling_workers is None:
        scaling_workers = tuple(w for w in (1, 2, 4, 8) if w <= (os.cpu_count() or 1))
    if any(type(w) is not int or w < 1 for w in scaling_workers):
        raise ValueError("Scaling workers must be positive integers")
    protocol = {
        "schema_version": 1,
        "kind": "frozen financial comparison",
        "frozen_at": datetime.now(UTC).isoformat(),
        "source_sha256": source_fingerprint(),
        "market": vars(market),
        "seeds": list(seeds),
        "count": count,
        "workers": workers,
        "checkpoints": list(checkpoints),
        "reduction_factor": factor,
        "audit_replications": audit_replications,
        "threads_per_worker": 1,
        "quality_margin": reference * margin_fraction,
        "margin_fraction": margin_fraction,
        "pilot_seed": pilot_manifest["seed"],
        "pilot_reference_loss": reference,
        "pilot_manifest": pilot_manifest,
        "pilot_summary": pilot_summary,
        "methods": list(METHODS),
        "confidence": 0.95,
        "primary_comparison": "race versus parallel_full",
        "failure_policy": "failed campaigns preclude a blanket non-inferiority claim",
        "analysis": "paired one-sided Student t bound across optimizer seeds",
        "scaling_workers": sorted(set(scaling_workers)),
        "scaling_seeds": [20000, 20001],
        "scaling_count": max(12, count),
        "scaling_replications": 3,
        "environment": environment(),
        "baseline_reference": "Ray 2.58.0 single-bracket stopping ASHA; finite scalar losses",
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        json.dump(protocol, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return protocol
