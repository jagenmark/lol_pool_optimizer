# Aggregate Method Sweep Report

## Executive Summary

- Fixed-policy posterior mean favors **Pantheon, Sion, Xerath** at **54.22%**.
- Oracle posterior mean favors **Pantheon, Sion, Vel'Koz** at **54.71%** and should be read as an upper-bound diagnostic.
- Residual-adjusted scoring favors **Mel, Naafiri, Sion** at **52.57%**.
- Worst-scope robust scoring favors **Katarina, Pantheon, Xerath (52.88%)**.
- Sion appears in 23/25 method-summary best pools. Pantheon appears in 19/25 method-summary best pools.
- The most defensible single-scope uncertainty recommendation is **Pantheon, Sion, Xerath**, with a fixed-policy simulated lower-5 score of **53.70%**.

## Score Definitions

- `deterministic_eb`: `sum_j f_j max_i EB(W_ij)`.
- `fixed-policy`: choose `argmax_i posterior_mean_ij` once for each enemy, then simulate that locked policy.
- `oracle`: resample every matchup and then take the max in each draw; this is optimistic and not a practical policy.
- `offmeta_penalty`: deterministic score minus a transparent aggregate penalty from low pickrate, importance/pickrate ratio, and LoLalytics breadth/depth flags.
- `residual_adjusted`: two-way logit diagnostic with champion main effect removed, preserving enemy and matchup residual terms.

## Data Sources

- No live data was fetched. Every source below was already present locally.
- `opgg_plat_plus_16.05`: opgg_patch_folder, patch 16.05, rank plat_plus, role mid, retrieved 2026-03-13; [representative URL](https://op.gg/lol/champions/ahri/build/mid?region=global&tier=platinum_plus). Local matchup file: `C:\Users\gosee\Documents\codex\lol_pool_optimizer\data\16.05\opgg_mid_matchups_clean.csv`.
- `opgg_plat_plus_16.06`: opgg_local_extract, patch 16.06, rank plat_plus, role mid, retrieved 2026-04-07; [representative URL](https://op.gg/lol/champions/ahri/build/mid?region=global&tier=platinum_plus&patch=16.06). Local matchup file: `..\data\opgg_mid_matchups__plat_plus__16.06.csv`.
- `opgg_plat_plus_16.07`: opgg_local_extract, patch 16.07, rank plat_plus, role mid, retrieved 2026-04-07; [representative URL](https://op.gg/lol/champions/malzahar/build/mid?region=global&tier=platinum_plus&patch=16.07). Local matchup file: `..\data\opgg_mid_matchups__plat_plus__16.07.csv`.
- `opgg_emerald_plus_16.07`: opgg_raw_extract, patch 16.07, rank emerald_plus, role mid, retrieved 2026-04-05; [representative URL](https://op.gg/lol/champions/ahri/build/mid?region=global&tier=emerald_plus). Local matchup file: `..\data\raw\opgg_mid_matchups__global__emerald_plus__2026-04-05.csv`.
- `opgg_plat_plus_16.10`: opgg_local_extract, patch 16.10, rank plat_plus, role mid, retrieved 2026-06-07; [representative URL](https://op.gg/lol/champions/fizz/build/mid?region=global&tier=platinum_plus&patch=16.10). Local matchup file: `..\data\opgg_mid_matchups__plat_plus__16.10.csv`.
- `opgg_emerald_plus_16.10`: opgg_local_extract, patch 16.10, rank emerald_plus, role mid, retrieved 2026-06-08; [representative URL](https://op.gg/lol/champions/ahri/build/mid?region=global&tier=emerald_plus&patch=16.10). Local matchup file: `..\data\opgg_mid_matchups__emerald_plus__16.10.csv`.
- `opgg_diamond_plus_16.10`: opgg_local_extract, patch 16.10, rank diamond_plus, role mid, retrieved 2026-06-08; [representative URL](https://op.gg/lol/champions/ahri/build/mid?region=global&tier=diamond_plus&patch=16.10). Local matchup file: `..\data\opgg_mid_matchups__diamond_plus__16.10.csv`.
- `opgg_master_plus_16.10`: opgg_local_extract, patch 16.10, rank master_plus, role mid, retrieved 2026-06-08; [representative URL](https://op.gg/lol/champions/viktor/build/mid?region=global&tier=master_plus&patch=16.10). Local matchup file: `..\data\opgg_mid_matchups__master_plus__16.10.csv`.
- `opgg_plat_plus_16.11`: opgg_local_extract, patch 16.11, rank plat_plus, role mid, retrieved 2026-06-07; [representative URL](https://op.gg/lol/champions/fizz/build/mid?region=global&tier=platinum_plus&patch=16.11). Local matchup file: `..\data\opgg_mid_matchups__plat_plus__16.11.csv`.
- `opgg_emerald_plus_16.11`: opgg_local_extract, patch 16.11, rank emerald_plus, role mid, retrieved 2026-06-08; [representative URL](https://op.gg/lol/champions/ahri/build/mid?region=global&tier=emerald_plus&patch=16.11). Local matchup file: `..\data\opgg_mid_matchups__emerald_plus__16.11.csv`.
- `opgg_diamond_plus_16.11`: opgg_local_extract, patch 16.11, rank diamond_plus, role mid, retrieved 2026-06-08; [representative URL](https://op.gg/lol/champions/ahri/build/mid?region=global&tier=diamond_plus&patch=16.11). Local matchup file: `..\data\opgg_mid_matchups__diamond_plus__16.11.csv`.
- `opgg_master_plus_16.11`: opgg_local_extract, patch 16.11, rank master_plus, role mid, retrieved 2026-06-08; [representative URL](https://op.gg/lol/champions/viktor/build/mid?region=global&tier=master_plus&patch=16.11). Local matchup file: `..\data\opgg_mid_matchups__master_plus__16.11.csv`.
- LoLalytics pickrate/breadth/depth extract: patch scope `global_platinum_plus_ranked_solo_duo_mid`, depth scope `all_regions_all_ranks_last_7_days`, retrieved 2026-04-07; [representative URL](https://lolalytics.com/lol/ahri/build/?lane=middle&patch=16.7&tier=platinum_plus).

## Main Findings

- Sion is in the best pool for 7/7 EB alpha settings; Pantheon is in 5/7.
- Under enemy-frequency perturbation, Sion is in the winning pool 100.0% of draws and Pantheon 100.0%.
- The residual-adjusted best pool is `Mel, Naafiri, Sion`. Sion is retained and Pantheon is removed.
- Enemy-frequency perturbation is mostly a meta-weight stress test; it cannot reveal player-selection bias in `W_ij`.
- The scope sweep used 12 local OP.GG aggregate scopes and no Riot API calls.
- Neither focus champion leaves the best pool at tested penalty strengths up to `lambda=0.01`; larger values are deliberately stronger stress tests.
- First tested offmeta lambda removing Sion from the best pool: `0.1`. For Pantheon: `0.05`.

## Focus Champion Scope Stability

- **Sion:** best-pool member in 8/12 scope rows; median top-pool share 71.00%.
- **Pantheon:** best-pool member in 6/12 scope rows; median top-pool share 15.50%.

## Residual Model
- **Sion:** aggregate champion main effect +0.106 log-odds (odds ratio 1.111). This term is removed in residual-adjusted scoring.
- **Pantheon:** aggregate champion main effect +0.073 log-odds (odds ratio 1.076). This term is removed in residual-adjusted scoring.

## Contribution Concentration

- Pool rank 1, **Pantheon** in `Pantheon, Sion, Xerath`: 12 enemies, 22.56% enemy mass, effective matchups 6.1, top-5 lift share 81.56%.
- Pool rank 1, **Sion** in `Pantheon, Sion, Xerath`: 21 enemies, 46.56% enemy mass, effective matchups 7.9, top-5 lift share 59.10%.
- Pool rank 2, **Pantheon** in `Aurelion Sol, Pantheon, Sion`: 10 enemies, 21.28% enemy mass, effective matchups 5.3, top-5 lift share 80.91%.
- Pool rank 2, **Sion** in `Aurelion Sol, Pantheon, Sion`: 21 enemies, 45.99% enemy mass, effective matchups 8.0, top-5 lift share 57.22%.
- Pool rank 3, **Pantheon** in `Pantheon, Sion, Vel'Koz`: 11 enemies, 20.93% enemy mass, effective matchups 6.8, top-5 lift share 76.57%.
- Pool rank 3, **Sion** in `Pantheon, Sion, Vel'Koz`: 21 enemies, 46.33% enemy mass, effective matchups 8.6, top-5 lift share 59.97%.
- Pool rank 4, **Pantheon** in `Fizz, Pantheon, Sion`: 12 enemies, 21.53% enemy mass, effective matchups 6.2, top-5 lift share 78.53%.
- Pool rank 4, **Sion** in `Fizz, Pantheon, Sion`: 25 enemies, 53.70% enemy mass, effective matchups 10.8, top-5 lift share 52.35%.

## Limitations

- No Riot Match-V5 or live Riot API data was collected.
- These methods do not observe player identity, champion familiarity, pick order, team composition, or repeated-player effects.
- LoLalytics breadth/depth is a heuristic with its own recorded population scope; it is not a causal selection-bias correction.
- The residual model is an aggregate decomposition, not a replacement for the optimizer and not a causal adjustment.
- The enemy-frequency perturbation uses a Dirichlet model and treats the configured effective sample size as a sensitivity parameter, not a known sampling design.
- Fixed-policy lower-5 analytic objectives use an independence and normal approximation; the CSV also includes direct posterior simulation results.

## Next Non-Riot-API Step

Fetch or locally archive a second aggregate matchup source with W_ij by patch/rank, then rerun this sweep with true cross-source matchup agreement rather than only cross-source pickrate/depth heuristics.
