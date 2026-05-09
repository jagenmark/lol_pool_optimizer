# League of Legends Champion Pool Optimizer

This project now supports patch-based loading for the single-patch optimizer workflow.

Current scope:
- load cleaned matchup and enemy-frequency files from `data/<patch>/`
- validate required columns before running
- brute-force the best pool for a selected patch
- keep the scoring and optimizer modules unchanged
- leave room to build train-patch vs eval-patch workflows later

## Patch-Based Data Layout

Place files under patch folders like this:

```text
data/
|-- 16.05/
|   |-- opgg_mid_matchups_clean.csv
|   `-- enemy_freq_df.csv
|-- 16.06/
|   |-- opgg_mid_matchups_clean.csv
|   `-- enemy_freq_df.csv
`-- 16.07/
    |-- opgg_mid_matchups_clean.csv
    `-- enemy_freq_df.csv
```

## Main Entry Point

Run the single-patch optimizer with a patch label:

```powershell
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\src\main.py" --patch 16.05
```

Optional examples:

```powershell
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\src\main.py" --patch 16.06 --pool-size 2
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\src\main.py" --patch 16.07 --candidates ahri syndra orianna viktor
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\src\main.py" --patch 16.05 --top-k 10
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\src\main.py" --patch 16.05 --lowest-pickrate 1
```

`--lowest-pickrate` is optional and is interpreted as a percentage.
Example:
- `--lowest-pickrate 1` keeps only candidate champions with `pickrate >= 1%`

This filter uses:
- `data/<patch>/opgg_mid_champion_summary.csv`

It applies only to the candidate champion set before brute-force pool search.
It does not remove low-pickrate champions from the enemy side.
Enemy frequency logic and enemy champion coverage remain unchanged.

## Streamlit GUI

The GUI is a local prototype for visualizing the existing optimizer. It reuses the
same patch data, scoring, and brute-force pool ranking as the CLI, then adds
interactive tables and charts for understanding the recommendation.

```powershell
pip install -r requirements.txt
streamlit run app.py
```

From the sidebar you can choose patch `16.05`, `16.06`, or `16.07`, select
candidate champions, choose pool size, and run the optimizer.

## Champion Icons

Download local champion icons from Riot Data Dragon for the GUI:

```powershell
python scripts/download_champion_icons.py
```

Use `--force` to redownload icons that already exist. The script writes icons to
`assets/champion_icons/` and updates `assets/champion_icons/champion_icon_manifest.csv`.

## Loader Design

- `src/data_loader.py` now includes `load_patch_data(patch, data_dir)`
- patch loading is isolated behind one function so later train-patch vs eval-patch workflows can compose multiple patch loads cleanly
- required columns are validated for matchup, enemy-frequency, and summary files

## Future Cross-Patch Work

The earlier experiment-oriented entrypoint has been preserved in:
- [cross_patch_experiment.py](C:\Users\gosee\Documents\codex\lol_pool_optimizer\src\cross_patch_experiment.py)

That keeps the project easy to extend later without mixing the single-patch CLI and cross-patch experimentation concerns.

## Results Workflow

Generate school-project result CSVs and plots with:

```powershell
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\scripts\generate_results.py" --patch 16.05 --patches 16.05 16.06 16.07 --pool-size 3 --max-pool-size 8
```

Optional candidate filtering works the same way as the main optimizer:

```powershell
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\scripts\generate_results.py" --patch 16.05 --lowest-pickrate 1
```

Example output modes:

```powershell
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\scripts\generate_results.py" --patch 16.07 --max-pool-size 8 --output-dir results/unrestricted
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\scripts\generate_results.py" --patch 16.07 --max-pool-size 8 --lowest-pickrate 1.0 --output-dir results/min_pickrate_1
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\scripts\generate_results.py" --patch 16.07 --max-pool-size 8 --candidates-file candidates_mid.txt --output-dir results/custom_candidates
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\scripts\generate_results.py" --patch 16.07 --pool-size 3 --force-champion Ahri --lowest-pickrate 1 --output-dir results/forced_ahri
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\scripts\generate_results.py" --patches 16.05 16.06 16.07 --max-pool-size 8 --force-champion Ahri --lowest-pickrate 1 --output-dir results/patch_validation_forced_ahri
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\scripts\generate_results.py" --patch 16.05 --pool-size 3 --force-champion-batch top_pickrate:10 --lowest-pickrate 1 --output-dir results/forced_batch_top10
```

The results script also accepts patch aliases like `26.7` when the local data folder is named `16.07`.

The workflow saves outputs under the chosen `--output-dir`, including:
- `recommended_pool.csv`
- `baseline_comparison.csv`
- `baseline_comparison.png`, `baseline_comparison_fullscale.png`, and `baseline_comparison_zoomed.png`
- `marginal_utility.csv`
- `marginal_utility.png`
- `patch_validation.csv`
- `patch_validation_absolute.png`, `patch_validation_absolute_fullscale.png`, and `patch_validation_absolute_zoomed.png`
- `patch_validation_delta.csv`
- `patch_validation_delta.png`
- `matchup_coverage.csv`
- `matchup_heatmap.png`
- `matchup_heatmap_values.csv`
- `matchup_games_histogram.csv`
- `matchup_games_histogram.png`
- `matchup_games_histogram_log.png`
- `forced_champion_comparison.csv` and `forced_champion_comparison.png` when `--force-champion` is used
- `forced_champion_by_pool_size.csv` and `forced_champion_by_pool_size.png` when `--force-champion` is used
- `forced_champion_batch.csv` and batch summary files when `--force-champion-batch` is used

The patch-validation output selects pools on each training patch and evaluates the same selected pool on the following patch in the `--patches` list.

CSV/plot meanings:
- `baseline_comparison`: optimized pool score compared to highest-winrate, highest-blindscore, and highest-pickrate baselines across pool sizes.
- `marginal_utility`: how much extra score is gained when increasing the optimized pool size.
- `patch_validation`: train on one patch, evaluate the same selected pool on the next patch.
- `patch_validation_delta`: test-score difference versus the highest-winrate baseline.
- `matchup_coverage`: which pool champion covers each common enemy champion.
- `matchup_heatmap_values`: W(i,j) values used by the matchup heatmap.
- `matchup_games_histogram`: histogram bins and counts for matchup sample sizes.
- `forced_champion_comparison`: compares an optimized complement against forced winrate, blindscore, and pickrate baselines for one pool size.
- `forced_champion_by_pool_size`: repeats the forced-complement comparison across pool sizes.
- comparison plots with poolscores also save `_fullscale` and `_zoomed` PNG versions for report readability.
- `forced_champion_batch`: repeats forced-complement testing for several anchor champions and summarizes average scores, average deltas, and optimized-model win counts.
