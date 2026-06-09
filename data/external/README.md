# External Source Data

This directory contains the source extracts used by the selection-bias,
method-sweep, and scope-stability workflows.

- `opgg_mid_*`: OP.GG champion summaries and matchup aggregates by patch/rank.
- `lolalytics_mid_*`: LoLalytics pick-rate and player-depth extracts.
- `raw/`: dated OP.GG snapshots retained for source comparisons.
- `enemy_freq_*`: legacy frequency exports corresponding to prepared patches.

The main CLI discovers this directory automatically. Override it with
`--selection-bias-extra-data-dir` or `--method-sweep-extra-data-dir` when
testing another extract set.

Data can be refreshed with the utilities in `scripts/extractors/`.
