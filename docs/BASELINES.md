# Allocation baselines

All four methods in the archived financial study consume the same six configurations
within each optimizer seed. They share objective code, replication identities,
training fidelity, loss definition, and the nine-replication cap.

| Method | Allocation | Workers |
| --- | --- | ---: |
| Serial full budget | All candidates to the cap | 1 |
| Parallel full budget | All candidates to the cap | 4 |
| ASHA | Rank-based stopping at replication rungs 3 and 9 | 4 |
| Replication race | Paired mean-difference heuristic at checkpoints 3 and 9 | 4 |

This isolates allocation changes inside one runtime. Random search supplies the
candidate pool; no Bayesian proposer is used.

## ASHA compatibility

Aster implements a **single-bracket stopping** variant inspired by
[ASHA](https://arxiv.org/abs/1810.05934). At each nonterminal rung, compare the current
candidate's mean loss with the linearly interpolated `1 / reduction_factor` quantile
of **previously recorded** rung losses. A strictly worse value stops; ties survive.
Record the candidate after the comparison, including a candidate that stops.
Candidates with no prior rung peers continue. The terminal rung completes the trial.

Checkpoints must be geometric and the reduction factor an integer of at least two.
The financial protocol uses `(3, 9)` and factor 3. Results depend on arrival order;
this is not synchronous successive halving or a pause/promotion implementation.

The compatibility test imports `_Bracket` from **Ray 2.58.0** and checks a fixed
arrival trace, quantiles, and ties against the upstream rule. The reference is
[Ray's pinned async_hyperband.py](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/tune/schedulers/async_hyperband.py).
The optional dependency is pinned because this is a private upstream interface.

```bash
python -m pip install -e '.[dev,benchmark,baseline]'
python -m pytest tests/test_asha.py -q
```

No Ray cluster is launched in the reported comparison. Matching these scheduler
decisions does not benchmark Ray's executor, storage, searchers, multi-bracket
behavior, or end-to-end performance. Those comparisons and Optuna TPE remain future work.
