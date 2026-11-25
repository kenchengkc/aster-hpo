# Archived CPU evidence

This directory contains the recorded runs behind the CPU study. The report
explains the results. The benchmark guide shows how to unpack the data and
regenerate the analysis. [Technical report](../../RESULTS.md) · [Benchmark guide](../../BENCHMARKING.md)

- `financial-artifacts.tar.gz` contains all 80 searches, eight worker-scaling runs,
  the financial measurements for each simulation, the protocol set before the study,
  the separate large pilot, and the original analysis. The files are JSON and Markdown.
- `diagnostic-artifacts.tar.gz` contains all 640 diagnostic runs with their manifest
  and summaries.
- `protocol.json` records the study settings. `manifest.json` describes the machine
  and software. `summary.json` and `report.md` contain the corrected results.
- `diagnostic-summary.json` lists results and losses for each scenario and candidate order.
- The PNG files are plots of the recorded data.
- `provenance.json` lists the code revisions and explains the correction.
- `SHA256SUMS` lists checksums for the archives and the files in this directory.

The financial searches ran at commit `bef29bf`. The reports and diagnostic runs
used `89e75c6`. Across all 20 seeds, the methods picked the same settings and
the paired differences in audit error were zero. The original analysis incorrectly
reported non-inferiority. With zero variance, the planned test could not give
a statistical bound for this comparison. The archive keeps the original analysis
in `financial-study/original-analysis/`.

Extract the archives into a fresh directory. They contain `financial-study/`,
`hedging-pilot-large/`, `diagnostics-v02/`, and `financial-protocol.json`.
You can rebuild the analysis from these files. Resuming a search requires the
original SQLite database, which the archives omit.
