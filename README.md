# League of Legends Champion Pool Optimizer

This project now supports patch-based loading for the single-patch optimizer workflow.

Current scope:
- load cleaned matchup and enemy-frequency files from `data/<patch>/`
- validate required columns before running
- rank the best pool exactly for small searches and with bounded beam search for large ones
- keep matchup scoring modular and reusable
- leave room to build train-patch vs eval-patch workflows later
- optionally simulate beta-posterior matchup and pool-score uncertainty

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
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\src\main.py" --patch 16.07 --estimator raw
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\src\main.py" --patch 16.07 --estimator eb --eb-alpha 100
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\src\main.py" --patch 16.07 --estimator eb --eb-alpha 100 --eb-mu 0.5
python src/main.py --patch 16.07 --estimator eb --uncertainty
python src/main.py --patch 16.07 --uncertainty --simulation-mode fixed-policy --posterior-samples 5000 --posterior-seed 42 --simulate-top-pools 100
python src/main.py --patch 16.07 --compare-ranks --compare-top-n 10
```

`raw` remains the default estimator. The `eb` estimator uses:

```text
(wins + alpha * mu) / (games + alpha)
```

When `--eb-mu` is omitted, `mu` is estimated as the games-weighted global
winrate for the loaded patch. The cleaned OP.GG files contain matchup games and
a rounded aggregate winrate, not exact wins, so the loader infers
fractional wins as `games * raw_winrate`. The CLI writes
`outputs/matchup_shrinkage_comparison.csv`, ordered by the largest absolute
adjustment, with raw winrate, shrinked winrate, games, inferred wins, and
shrinkage amount.

## Posterior Matchup Uncertainty

`--uncertainty` builds an independent Beta posterior for every matchup and
simulates score distributions for the top point-estimate pools. Candidate pools
are ranked first using the selected `--estimator`; only the top
`--simulate-top-pools` are simulated. Small search spaces use exact brute force.
When the number of combinations exceeds 250,000, candidate generation uses a
bounded beam search so large pool sizes do not enumerate every possible pool.

The posterior uses the games-weighted global winrate as its prior mean, or
`--eb-mu` when supplied. `--prior-strength` controls the prior pseudo-game
count and defaults to `--eb-alpha`. The posterior mean is the same empirical
Bayes formula as the `eb` point estimator when their prior settings match.
Unlike raw winrate, it shrinks low-game observations toward the prior mean.

`--simulation-mode fixed-policy` is the default and recommended practical
interpretation. It chooses each enemy's best response once using posterior
means, then evaluates that locked policy across posterior draws.
`--simulation-mode oracle` reselects the best response after seeing every draw;
it is optimistic and is retained as an upper-bound diagnostic.

Default outputs under `--output-dir`:

- `posterior_matchups.csv`: games, inferred wins/losses, raw winrate, posterior
  mean and standard deviation, and 5th/95th posterior percentiles.
- `pool_score_simulation.csv`: one sampled score per pool and simulation.
- `pool_score_simulation_summary.csv`: mean, median, standard deviation,
  5th/95th score percentiles, and probability of being best.

Use `--output-posterior-matchups PATH` or `--output-pool-simulation PATH` to
override the corresponding paths. The summary is written beside the detailed
simulation file.

`lower_5_score` is a downside-oriented score: 95% of posterior simulations were
above it. `probability_of_being_best` is the fraction of draws where a pool had
the highest score among the simulated top-N candidate pools. This can reveal a
stable pool with a slightly lower mean but a stronger lower tail.

These intervals describe sampling uncertainty under the beta-binomial model.
They do not correct selection bias, patch drift, dependence between matchup
estimates, player skill differences, or other data-quality issues.

## Pool Rank Comparison

Use `--compare-ranks` to produce `outputs/pool_rank_comparison.csv`. Comparison
mode automatically runs posterior simulation and compares:

- raw ranking: pool scores from unshrunk point estimates
- EB ranking: pool scores after empirical Bayes shrinkage
- posterior mean ranking: average performance across posterior simulations
- lower-5 ranking: conservative performance at the fifth percentile
- probability-of-being-best ranking: how often each pool wins among the
  simulated candidate pools

Pool champion names are sorted before joining, so differently ordered labels
refer to the same pool. The report includes the union of each metric's top
`--compare-top-n` pools. Positive rank-change values mean the pool rose relative
to its raw rank; negative values mean it fell.

```powershell
python src/main.py --patch 16.07 --estimator eb --compare-ranks `
  --compare-top-n 10 `
  --simulate-top-pools 100 `
  --posterior-samples 5000 `
  --comparison-output outputs/pool_rank_comparison.csv
```

For comparison mode, posterior candidates are the normalized union of the top
`--simulate-top-pools` raw pools and top `--simulate-top-pools` EB pools.

Comparison mode also writes `outputs/pool_contributions.csv`. It contains one
row per included top pool and enemy champion, with:

- the enemy's normalized frequency
- the pool champion with the highest posterior mean into that enemy
- raw winrate, posterior mean, 5th/95th posterior percentiles, and games
- `weighted_contribution = enemy_frequency * posterior_mean`

Frequencies are renormalized over the enemies that can be scored for each pool,
matching the optimizer. Consequently, `weighted_contribution` sums to that
pool's posterior-mean point score. This differs slightly from the simulation
`mean_score`: fixed-policy simulation averages draws from the locked
posterior-mean response, while oracle simulation averages the maximum sampled
matchup in every draw.

Use `--pool-contribution-output PATH` to override the contribution CSV path.

## Selection-Bias Diagnostics

Run diagnostic outputs without changing the optimizer's objective:

```powershell
python src/main.py --patch 16.07 --estimator eb --eb-alpha 100 `
  --pool-size 3 `
  --selection-bias-diagnostics `
  --selection-bias-top-pools 100 `
  --selection-bias-output-dir outputs/selection_bias_16_07 `
  --selection-bias-extra-data-dir ../data
```

The workflow writes:

- `selection_bias_champion_summary.csv`
- `selection_bias_matchup_enrichment.csv`
- `selection_bias_favorable_selection.csv`
- `selection_bias_pool_dependency.csv`
- `selection_bias_source_stability.csv`
- `selection_bias_patch_rank_stability.csv`
- `selection_bias_sources.csv`
- `selection_bias_report.md`

The diagnostics measure top-pool dependence, exact exclusion loss, importance
relative to candidate pick-rate share, matchup enrichment, favorable opponent
selection, contribution concentration, posterior matchup reliability, and
patch/rank stability. When the workspace-level `data/` directory is available,
the workflow also uses the dated LoLalytics breadth/depth extract and the OP.GG
Emerald+ validation extract.

`selection_advantage` compares the observed opponent distribution for a
champion with the general enemy distribution over the same recorded matchup
universe. It is descriptive: a positive value does not prove that the observed
matchup win rates are causally biased.

LoLalytics breadth/depth is used only as a heuristic. Its recorded scope is
preserved in the source output because depth may cover all ranks, regions, and
roles rather than the exact historical mid-lane cohort.

Use `--exclude-champions Sion Pantheon` to rerun the primary optimizer with
specific candidates removed.

## Aggregate Method Sweep

Run the broader non-Riot-API robustness workflow:

```powershell
python src/main.py --patch 16.07 --estimator eb --eb-alpha 100 `
  --method-sweep `
  --method-sweep-output-dir outputs/method_sweep_16_07 `
  --method-sweep-extra-data-dir ../data
```

The sweep uses only prepared OP.GG aggregates and existing local LoLalytics
extracts. It compares fixed-policy and oracle posterior simulation, EB alpha
sensitivity, Dirichlet enemy-frequency perturbations, local patch/rank scopes,
offmeta penalty stress tests, contribution concentration, a two-way logit
residual diagnostic, and conservative fixed-policy and worst-scope objectives.

Outputs:

- `method_sweep_report.md`
- `method_sweep_summary.csv`
- `fixed_policy_simulation_summary.csv`
- `alpha_sensitivity.csv`
- `enemy_frequency_sensitivity.csv`
- `scope_stability.csv`
- `offmeta_penalty_sensitivity.csv`
- `contribution_concentration.csv`
- `residual_model_summary.csv`
- `robust_objective_comparison.csv`

The deterministic best-pool score printed by the optimizer is the selected
point estimator's score. For example, with `--estimator eb`, `54.23%` means
`sum_j f_j max_i EB(W_ij)`. It is not a posterior simulation mean or lower-5
score. Posterior rankings can differ because they evaluate uncertainty and,
in oracle mode, may also change the selected response inside each draw.

These methods are stress tests, not causal corrections for player familiarity,
draft context, or specialist selection.

## All-Scope Stability Diagnostics

Run the EB robustness workflow across every discovered cumulative patch/rank
scope:

```powershell
python scripts/run_scope_stability.py
```

The workflow discovers paired aggregate files such as
`opgg_mid_matchups__emerald_plus__16.11.csv` and prepared patch directories,
deduplicates equivalent scopes, and uses the common candidate intersection so
patch/rank comparisons are not driven by candidate availability.

Outputs under `outputs/scope_stability/`:

- `scope_summary.csv`
- `best_pools_by_scope.csv`
- `champion_inclusion_by_scope.csv`
- `exclusion_loss_by_scope.csv`
- `sion_pantheon_matchup_stability.csv`
- `scope_stability_report.md`

Rank labels are kept explicitly cumulative. The workflow only creates disjoint
rank buckets when exact additive wins/games and fully nested matchup keys are
available. Rounded aggregate win rates or threshold-censored matchup rows make
subtraction unsafe, so the workflow fails closed and records the reason.

These outputs are robustness/stability diagnostics. They do not causally
correct selection bias.

`--lowest-pickrate` is optional and is interpreted as a percentage.
Example:
- `--lowest-pickrate 1` keeps only candidate champions with `pickrate >= 1%`

This filter uses:
- `data/<patch>/opgg_mid_champion_summary.csv`

It applies only to the candidate champion set before brute-force pool search.
It does not remove low-pickrate champions from the enemy side.
Enemy frequency logic and enemy champion coverage remain unchanged.

## Browser GUI

The GUI is a static browser application in `site/`. It uses exported copies of
the existing patch data and reproduces the CLI scoring rules in the browser:
self and missing matchups are skipped, usable enemy frequencies are
renormalized, and each pool uses its best available answer for every enemy.

Regenerate the browser data after changing the patch CSVs:

```powershell
python scripts/export_web_data.py
```

Serve the site locally from the project root:

```powershell
python -m http.server 8501 --directory site
```

Then open `http://localhost:8501/`.

The static bundle can be published directly to here.now:

```bash
publish.sh site --client codex
```

The Python CLI remains the source workflow for command-line optimization and
research outputs. The browser GUI is a lightweight interactive view over the
same patch data and scoring model.

## Champion Icons

Download local champion icons from Riot Data Dragon for the GUI:

```powershell
python scripts/download_champion_icons.py
```

Use `--force` to redownload icons that already exist. The script writes icons to
`assets/champion_icons/` and updates `assets/champion_icons/champion_icon_manifest.csv`.
Run `python scripts/export_web_data.py` afterward to copy updated icons into the
static site bundle.

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
py "C:\Users\gosee\Documents\codex\lol_pool_optimizer\scripts\generate_results.py" --patch 16.07 --estimator eb --eb-alpha 100
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
- `matchup_shrinkage_comparison.csv`
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
