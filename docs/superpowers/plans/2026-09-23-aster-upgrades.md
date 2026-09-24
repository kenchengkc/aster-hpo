# Aster Research Runtime Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Deliver the approved upgrades in order: scheduler evidence, recoverable execution, a CPU financial workload, and honest comparative evaluation.

**Architecture:** A coordinator accepts observations, invokes isolated scheduler policies, and optionally persists state through a single-writer SQLite store. Reproducible experiment modules consume the public API and generate raw evidence and reports. Numerical dependencies remain optional.

**Tech stack:** Python 3.11+, standard-library processes/SQLite; NumPy for the financial workload; SciPy and Matplotlib for offline analysis.

**Spec:** ../specs/2026-09-23-scheduler-evidence-design.md

## Global constraints

- Preserve existing public entry points and scalar-loss objectives.
- Keep the runtime dependency-free; financial/analysis packages are optional extras.
- Preserve stable per-replication seeds and disjoint search/audit namespaces.
- Never claim practical speed or preserved quality from a cheap diagnostic.
- Use descriptive branch names and neutral project authorship.
- Implement inline, progressing through the user's approved order.

## Review focus

- Already dispatched work finishing after pruning: retain costs and terminal status.
- Recovered sessions and retries: reject duplicate/stale results and incompatible manifests.
- All-failed campaigns: no winner and no fabricated quality value.
- Finance: no look-ahead, correct self-financing cash flows, training/audit isolation.
- Statistics: independent optimizer runs are the unit of inference, not nested paths or order permutations.

## Task 1: Reconsider checkpoint comparisons

**Files:** create src/aster_hpo/schedulers.py and tests/test_schedulers.py; modify src/aster_hpo/core.py.

**Interfaces:** internal RacePolicy.decisions(trials) yields decision dictionaries with trial_id, peer_id, checkpoint indices and paired statistics; consumes existing Trial and ReplicationRace objects by structural typing.

- [ ] Write the regression first:

  ```python
  @pytest.mark.parametrize("xs", [[0, 5, 10], [10, 5, 0]])
  def test_candidate_order_uses_available_checkpoint_peers(xs):
      result = Study([{"x": x} for x in xs],
                     scheduler=ReplicationRace((3, 9))).optimize(noisy)
      assert sum(len(t.observations) for t in result.trials) == 15
      assert result.best_trial.config == {"x": 0}
  ```

- [ ] Run .venv/bin/python -m pytest tests/test_schedulers.py -q.
  Expected: reverse-order case fails with 27 observations.
- [ ] Add deterministic completion-sequence tests: a late best peer prunes earlier candidates; repeated identical prefixes do not repeat decisions; failed/pruned peers are excluded; completed trials stay terminal; nonzero residual noise produces a nonzero standard error.
- [ ] Implement RacePolicy with seen (candidate, peer, checkpoint) identities. On every acceptance, reconsider eligible prefixes and apply pruning before future dispatch. Discard stale ready-queue entries and retain observations from already running pruned work.

  ```python
  for decision in policy.decisions(trials):
      decisions.append(decision)
      if decision["action"] == "prune":
          trials[decision["trial_id"]].status = "pruned"
  ```

- [ ] Run the complete suite and Ruff. Expected: all pass; both orderings use 15 evaluations.
- [ ] Commit: fix: reconsider replication checkpoints when peers become available.

## Task 2: Reproducible synthetic diagnostics

**Files:** create benchmarks/__init__.py, benchmarks/synthetic.py, benchmarks/common.py, benchmarks/diagnostics.py, tests/test_benchmarks.py; update docs/BENCHMARKING.md.

**Interfaces:** synthetic.pool(seed, count) -> list[dict]; SyntheticObjective(case)(config, replication) -> float; expected_loss(case, config) -> float; diagnostics.run_suite(output, seeds, workers, count, checkpoints) -> dict.

- [ ] Test the reporting contract before implementation:

  ```python
  report = run_suite(tmp_path / "run", seeds=(7,), workers=1,
                     count=3, checkpoints=(3, 9))
  assert report["interpretation"] == "synthetic diagnostic"
  assert report["runs"] > 0
  assert (tmp_path / "run" / "report.md").exists()
  ```

- [ ] Run new tests. Expected: missing benchmark module/API.
- [ ] Add hand-calculated regret and tied-optimum fixtures, all-failed runs, identical serial reruns excluding timing, missing/corrupt artifacts, and collision protection.
- [ ] Implement additive, heteroscedastic, near-tie, and finite-variance heavy-tail objectives; shared/independent scenarios; forward/reversed candidate pools. Store one manifest and raw result per run; generate per-run and paired summaries without treating permutations as independent seeds.
- [ ] Add optional report figures with Matplotlib and a report-only command consuming saved data.
- [ ] Run full suite and a default 20-seed diagnostic. Expected: explicit adverse outcomes, complete artifact inventory, no practical-speed claim.
- [ ] Commit: feat: add reproducible noisy scheduler diagnostics.

## Task 3: Recoverable and observable local execution

**Files:** create src/aster_hpo/storage.py, src/aster_hpo/runtime.py, tests/test_storage.py, tests/test_recovery.py; modify core.py.

**Interfaces:** Study gains optional storage, objective_id, max_retries, threads_per_worker parameters. SQLiteStore manages a manifest, durable coordinator snapshots, attempt identities, and an event journal. Existing no-storage behavior stays valid.

- [ ] Write integration tests that interrupt a real serial run using KeyboardInterrupt, resume with the same manifest, and compare accepted observations with uninterrupted evaluation. Assert a changed config/objective_id fails before evaluation.

  ```python
  with pytest.raises(KeyboardInterrupt):
      Study(configs, storage=db, objective_id="fixture-v1").optimize(interrupt_once)
  result = Study(configs, storage=db, objective_id="fixture-v1").optimize(interrupt_once)
  assert [t.losses for t in result.trials] == reference_losses
  ```

- [ ] Run tests. Expected: unsupported storage argument.
- [ ] Test duplicate/stale acceptance, database ownership, worker death and bounded pool retries with unchanged seeds, retry exhaustion, and retention of abandoned attempts with unknown CPU cost.
- [ ] Implement transactional snapshots after each accepted event; persist attempts before dispatch; acquire an exclusive coordinator lease; mark orphaned attempts interrupted on resume. Retry only infrastructure failures, never objective numerical failure. Validate objective identity and all scheduling inputs before resume.
- [ ] Record submit/start/finish and recovery events. Limit common native thread pools in worker initialization and record requested/effective settings; explain limits for already imported serial numerical runtimes.
- [ ] Run full suite, a real subprocess kill/resume demonstration, and regression diagnostics. Expected: no duplicate accepted replication, bounded retries, cost metadata distinguishes measured and unknown work.
- [ ] Commit: feat: persist experiments and recover interrupted workers.

## Task 4: CPU hedging workload

**Files:** create benchmarks/hedging.py, examples/hedging.py, tests/test_hedging.py; extend optional dependencies.

**Interfaces:** Market dataclass freezes financial/grid/path/training budgets; HedgingObjective(market)(config, replication) -> float; evaluate_policy returns MSE, quantiles, turnover, and cost metrics.

- [ ] Write deterministic financial-accounting tests before implementation: zero-position wealth, exact turnover cost, zero-volatility paths, no look-ahead, delta hedge sanity, seed isolation, and finite-difference gradient checks.

  ```python
  paths = np.array([[1.0, 1.1, 1.2]])
  positions = np.array([[0.5, 0.5]])
  pnl, costs, turnover = hedge_pnl(paths, positions, cost_rate=0.01)
  assert pnl[0] == pytest.approx(0.089)
  assert costs[0] == pytest.approx(0.011)
  ```

- [ ] Run tests. Expected: missing hedging module.
- [ ] Implement fixed-fidelity GBM paths and a bounded small two-layer policy trained by gradient descent with NumPy. The policy observes current price/time only; train using independent paths and evaluate on shared exogenous paths. Initial capital is fixed across policies. Include liquidation costs.
- [ ] Tune learning rate, hidden width, and regularization with fixed training budgets. One replication means one independently trained policy. Record both scalar objective and diagnostic artifacts without changing the core objective protocol.
- [ ] Run full suite and a disjoint pilot to measure actual replication cost. Expected: repeatable finite outcomes and validated gradients/accounting; no live-profit claim.
- [ ] Commit: feat: add reproducible CPU neural hedging benchmark.

## Task 5: Established allocation baseline and paired quality study

**Files:** extend schedulers.py and exports; create benchmarks/study.py, benchmarks/analysis.py, tests/test_analysis.py, tests/test_asha.py.

**Interfaces:** ASHA(checkpoints, reduction_factor) schedules replication-count rungs; paired_analysis(differences, margin, confidence=0.95) returns fixed-sample one-sided bounds; study runner stores selected configurations before independent paired audits.

- [ ] Test literal rung/percentile decisions, ties, final cap, and compatibility against the published Ray stopping-ASHA rule; test a hand-computable mean/SE, insufficient independent runs, failed campaigns, and nested-data aggregation.

  ```python
  result = paired_analysis([1.0, 2.0, 3.0], margin=5.0)
  assert result["mean_difference"] == 2.0
  assert result["standard_error"] == pytest.approx(1 / math.sqrt(3))
  ```

- [ ] Run tests. Expected: missing ASHA/analysis API.
- [ ] Implement documented asynchronous stopping-ASHA with a frozen candidate pool. Pin the external comparator source/version used for verification and keep differences explicit.
- [ ] Freeze a pilot-derived financial protocol before final data: seeds, candidate pool, checkpoints, workers, audit count, quality margin, and failure policy. Execute serial full budget, parallel full budget, parallel ASHA, and racing with paired seeds; lock winners before audits.
- [ ] Compute one mean audited difference per independent optimizer seed. Use a fixed-sample t upper bound with explicit assumptions; a failed method prevents a blanket non-inferiority claim. Keep pilot and final artifacts separate.
- [ ] Generate endpoint quality/time, compute/quality, scaling, and worker-timeline plots from raw artifacts. Do not label endpoint measurements time-to-target curves.
- [ ] Run full suite and the frozen modest CPU study. Expected: complete reproducible evidence, including inconclusive/negative results.
- [ ] Commit: feat: compare allocation methods with independent quality audits.

## Task 6: Package, documentation, and final review

**Files:** README.md, ROADMAP.md, CHANGELOG.md, docs/BENCHMARKING.md, docs/RESULTS.md, CI, generated compact evidence bundle.

**Interfaces:** documentation commands use the installed public package and benchmark modules; evidence links identify their manifest and raw records.

- [ ] Update version, runtime guarantees, known limitations, examples, reproducibility commands, and measured findings. Keep diagnostic/development/final labels distinct.
- [ ] Run .venv/bin/python -m pytest -q; ruff check .; ruff format --check .; .venv/bin/python -m build; installed-package examples and report regeneration.
  Expected: all pass, valid wheel and sdist, reproducible analysis from saved artifacts.
- [ ] Request one fresh whole-branch review; address substantive findings with reproducing tests and rerun the suite.
- [ ] Commit verified changes; leave a reviewable feature branch without publishing or merging.
