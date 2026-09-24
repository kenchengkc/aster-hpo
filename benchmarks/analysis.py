"""Fixed-sample, run-level analysis; uncertainty is conditional on stated assumptions."""

import math
import statistics
from collections import defaultdict


def paired_analysis(differences, *, margin, confidence=0.95):
    from scipy.stats import t

    if not math.isfinite(margin) or margin < 0 or not 0.5 < confidence < 1:
        raise ValueError("Need a finite nonnegative margin and confidence in (0.5, 1)")
    values = [float(x) for x in differences if x is not None]
    if any(not math.isfinite(x) for x in values):
        raise ValueError("Run differences must be finite or explicitly missing")
    n = len(values)
    mean = statistics.mean(values) if n else None
    se = statistics.stdev(values) / math.sqrt(n) if n >= 2 else None
    upper = mean + float(t.ppf(confidence, n - 1)) * se if n >= 2 else None
    if n != len(differences):
        status = "incomplete"
    elif n < 2:
        status = "insufficient"
    else:
        status = "noninferior" if upper < margin else "inconclusive"
    return {
        "differences": list(differences),
        "n": n,
        "missing": len(differences) - n,
        "mean_difference": mean,
        "standard_error": se,
        "upper_bound": upper,
        "margin": margin,
        "confidence": confidence,
        "status": status,
        "analysis": "one-sided Student t bound across independent optimizer seeds",
    }


def _median_interval(values):
    if not values:
        return None
    import numpy as np

    rng = np.random.default_rng(20260924)
    samples = rng.choice(values, size=(2000, len(values)), replace=True)
    return [float(x) for x in np.quantile(np.median(samples, axis=1), [0.025, 0.975])]


def analyze_campaigns(records, *, margin, reference="parallel_full"):
    by_seed = defaultdict(dict)
    for record in records:
        if record["method"] in by_seed[record["seed"]]:
            raise ValueError("Duplicate campaign identity")
        by_seed[record["seed"]][record["method"]] = record
    methods = sorted({record["method"] for record in records})
    if reference not in methods:
        raise ValueError("Missing reference method")
    if any(set(group) != set(methods) for group in by_seed.values()):
        raise ValueError("Missing campaigns; failures must be represented explicitly")
    comparisons, summaries = {}, {}
    for method in methods:
        differences, speedups, audit_means = [], [], []
        values = [group[method] for _, group in sorted(by_seed.items())]
        for _, group in sorted(by_seed.items()):
            current, baseline = group[method], group[reference]
            a, b = current["audit_losses"], baseline["audit_losses"]
            if a is None or b is None:
                differences.append(None)
            else:
                if not a or len(a) != len(b):
                    raise ValueError("Audit replication counts must match")
                # Replications are nested inside one independent optimizer run.
                differences.append(statistics.mean(x - y for x, y in zip(a, b, strict=True)))
                speedups.append(baseline["search_wall_seconds"] / current["search_wall_seconds"])
            if a is not None:
                audit_means.append(statistics.mean(a))
        if method != reference:
            comparisons[method] = paired_analysis(differences, margin=margin)
        summaries[method] = {
            "runs": len(values),
            "failed_runs": sum(v["audit_losses"] is None for v in values),
            "mean_audited_loss": statistics.mean(audit_means) if audit_means else None,
            "median_search_wall_seconds": statistics.median(
                v["search_wall_seconds"] for v in values
            ),
            "mean_objective_cpu_seconds": statistics.mean(
                v["objective_cpu_seconds"] for v in values
            ),
            "mean_audit_cpu_seconds": statistics.mean(v["audit_cpu_seconds"] for v in values),
            "mean_audit_wall_seconds": statistics.mean(v["audit_wall_seconds"] for v in values),
            "mean_allocated_worker_seconds": statistics.mean(
                v["allocated_worker_seconds"] for v in values
            ),
            "mean_replications": statistics.mean(v["replications"] for v in values),
            "unmeasured_attempts": sum(v["unknown_attempts"] for v in values),
            "median_speedup": statistics.median(speedups) if speedups else None,
            "speedup_bootstrap_95_interval": _median_interval(speedups),
        }
    return {
        "reference": reference,
        "methods": summaries,
        "comparisons": comparisons,
        "primary_comparison": "race versus parallel_full",
        "interpretation": "fixed candidate budget; endpoint quality and search time",
    }
