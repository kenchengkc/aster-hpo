# CPU hedging allocation study

Fixed candidate and replication caps; endpoint quality versus elapsed search time.
These are synthetic-market results, not live trading performance.

Independent optimizer seeds: 20. Audit replications per locked winner: 8.
Prespecified excess-MSE margin: 5.0903946e-06 (1% of disjoint pilot reference).

| Method | Failed | Mean audited MSE | Median search seconds | Mean objective CPU seconds | Mean replications | Median speedup vs parallel full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| asha | 0 | 0.000433374 | 2.27505 | 7.05189 | 38.4 | 1.2784 |
| parallel_full | 0 | 0.000433374 | 2.80221 | 10.4252 | 54 | 1 |
| race | 0 | 0.000433374 | 1.89592 | 4.92845 | 25.75 | 1.43159 |
| serial_full | 0 | 0.000433374 | 5.7708 | 5.92449 | 54 | 0.504635 |

## Independent quality comparison

- asha: mean excess MSE 0; one-sided 95% upper bound unavailable; status: **degenerate**.
- race: mean excess MSE 0; one-sided 95% upper bound unavailable; status: **degenerate**.
- serial_full: mean excess MSE 0; one-sided 95% upper bound unavailable; status: **degenerate**.

The primary comparison is racing versus parallel full budget. Other method
comparisons are descriptive, with nominal intervals and no multiplicity correction.
The t analysis assumes independent optimizer runs and an adequate approximation for
their mean differences; it is not a sequential pruning guarantee.
Audit paths and training replications are nested within optimizer runs, not counted
as extra independent runs. Inconclusive intervals do not establish equivalence.
A degenerate comparison has zero sample variance: its observed difference is
reported, but no inferential bound or non-inferiority conclusion is supplied.

## Cost and scope

Search time includes process startup, journal writes, and idle capacity. Audit costs
are reported separately below. Returned objective CPU time excludes coordinator and
startup overhead. Missing attempt cost is explicitly counted in summary.json.
The speedup bootstrap interval resamples whole paired optimizer seeds.
These endpoints are not time-to-target curves or equal-wall-budget comparisons.

- asha: mean audit wall 1.06s; mean audit CPU 0.9808s; unmeasured attempts 0; speedup 95% bootstrap interval [1.0508909231164814, 1.6238741792020401].
- parallel_full: mean audit wall 0.9475s; mean audit CPU 0.9237s; unmeasured attempts 0; speedup 95% bootstrap interval [1.0, 1.0].
- race: mean audit wall 0.8836s; mean audit CPU 0.8724s; unmeasured attempts 0; speedup 95% bootstrap interval [1.270728346788104, 1.8278703279726742].
- serial_full: mean audit wall 0.8349s; mean audit CPU 0.8286s; unmeasured attempts 0; speedup 95% bootstrap interval [0.48029611701881664, 0.5243262440422706].

## Fixed-workload scaling

12 candidates × 3 replications, 2 seeds; no pruning.
- 1 workers: median 3.569s.
- 2 workers: median 1.905s.
- 4 workers: median 1.016s.
- 8 workers: median 0.8336s.
