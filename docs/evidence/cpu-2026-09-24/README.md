# Archived CPU evidence

See the [technical report](../../RESULTS.md) for interpretation and
[commands](../../BENCHMARKING.md) for extraction and regeneration.

- `financial-artifacts.tar.gz`: all 80 campaigns, eight scaling runs, per-replication
  financial diagnostics, the frozen protocol, the large disjoint pilot, and original
  analysis before the zero-variance correction. JSON/Markdown only; no SQLite files.
- `diagnostic-artifacts.tar.gz`: all 640 diagnostic runs with manifest and summaries.
- `protocol.json`, `manifest.json`, `summary.json`, `report.md`: readable copies of
  the frozen settings, execution environment, and corrected financial analysis.
- `diagnostic-summary.json`: all diagnostic scenario/order slices, including losses.
- PNGs: standalone figures generated from those raw artifacts.
- `provenance.json`: execution/reporting revisions and the disclosed correction.
- `SHA256SUMS`: checksums for the archives and accompanying evidence files.

Financial execution: `bef29bf`. Reporting and diagnostic execution: `89e75c6`.
The original analysis in `financial-study/original-analysis/` is retained for audit
history and is superseded by the corrected report; its non-inferiority labels should
not be used as a supported claim.

Archive members use relative paths under `financial-study/`, `hedging-pilot-large/`,
and `diagnostics-v02/`, plus `financial-protocol.json`. Extract into a fresh directory.
The archives preserve observations and telemetry needed for analysis; they do not
contain execution databases for resuming optimization.
