# Selection Bias Diagnostics for the Champion Pool Optimizer

## Executive Summary

- Patch **16.07**, estimator **eb**, pool size **3**, and top **100** pools were analyzed.
- The best pool is **Pantheon, Sion, Xerath** with score **54.23%**.
- That score is the deterministic **EB point-estimate pool score**: `sum_j f_j max_i EB(W_ij)`. It is not a raw score, posterior simulation mean, lower-5 score, or probability-of-being-best.
- Excluding Sion and Pantheon together changes the best score by **0.71%**.
- **Conclusion:** The evidence remains ambiguous: the aggregate signal is material, but selection diagnostics add meaningful generalizability concerns.

## Why Posterior Simulation Can Differ

Posterior simulation samples uncertain matchup values instead of scoring one fixed matrix. Fixed-policy simulation locks each enemy's response using posterior means, while oracle simulation reselects the maximum after every draw and is therefore optimistic. Those different estimands can change both scores and pool ordering.

## Why Sion and Pantheon Rank Highly

The optimizer rewards complementary matchup coverage, not overall win rate alone. A champion dominates when it is the best available answer to many high-frequency enemies or supplies large marginal gains where the rest of the pool is weak.

- **Sion:** top-pool share 99.00%, slot-share/pick-rate ratio 59.05x, and exclusion loss 0.47%.
  Favorable-selection score 0.52%; best-pool coverage spans 21 enemies with median 266 games. The top five matchups supply 59.10% of its marginal lift, equivalent to about 7.9 equally weighted matchups.
  LoLalytics specialist heuristic: not flagged; best-pool member in 4/4 patch/rank scopes.
- **Pantheon:** top-pool share 31.00%, slot-share/pick-rate ratio 22.36x, and exclusion loss 0.24%.
  Favorable-selection score 0.69%; best-pool coverage spans 12 enemies with median 210 games. The top five matchups supply 81.56% of its marginal lift, equivalent to about 6.1 equally weighted matchups.
  LoLalytics specialist heuristic: not flagged; best-pool member in 2/4 patch/rank scopes.

## Within-Matchup Reliability

- **Sion largest marginal matchups:** Malzahar (1240 games, posterior 60.5%, 90% interval 58.3%-62.7%); Katarina (523 games, posterior 52.5%, 90% interval 49.2%-55.8%); Xerath (447 games, posterior 54.8%, 90% interval 51.3%-58.3%); Yasuo (587 games, posterior 56.9%, 90% interval 53.8%-60.0%); Galio (293 games, posterior 56.2%, 90% interval 52.1%-60.3%).
  3 of 21 selected matchups have fewer than 100 games; these should be treated as tail-risk evidence rather than primary support.
- **Pantheon largest marginal matchups:** Sylas (541 games, posterior 57.4%, 90% interval 54.2%-60.6%); Yone (491 games, posterior 56.5%, 90% interval 53.1%-59.9%); Akali (561 games, posterior 55.1%, 90% interval 51.9%-58.2%); Ekko (192 games, posterior 54.8%, 90% interval 50.0%-59.6%); Irelia (209 games, posterior 57.0%, 90% interval 52.3%-61.6%).
  2 of 12 selected matchups have fewer than 100 games; these should be treated as tail-risk evidence rather than primary support.

## Evidence Consistent With Selection Bias

- Low mid pick rate makes both champions less representative of the ordinary mid-player population, even when matchup game counts are adequate.
- Positive favorable-selection scores indicate that the recorded opponent mix is tilted toward matchups where the champion performs better than its own matchup baseline. This is descriptive and does not prove the matchup win rates themselves are biased.
- Matchup records cover only the source's recorded mid-opponent universe. Missing offrole opponents and unobserved draft context can still affect generalizability.

## Evidence Consistent With a Robust Aggregate Signal

- The report checks whether contributions are spread over many enemies, and reports median/minimum games plus posterior intervals for every selected matchup.
- Patch/rank stability is available for 4 scopes. Repeated top-pool appearance is harder to explain as one isolated noisy matchup.
- OP.GG and LoLalytics pick rates can be compared at the same patch and rank. Agreement supports the low-popularity diagnosis, though it does not validate W_ij.

## Cross-Source Limitation

A true matchup-level cross-source stability test was not available in this run. The LoLalytics extract contains pick rate and breadth/depth, not a second W_ij matrix.

## What Cannot Be Identified From Aggregates

- Player familiarity, one-trick status, and repeated-player weighting.
- Whether the champion was selected before or after the lane opponent.
- Team composition, bans, autofill, role swaps, and premade context.
- Within-player performance on the same matchup with and without specialization.

## Best Next Data Collection Step

Use Riot Match-V5 to build a stratified match-level sample across regions and ranks. Derive actual role, patch, champion matchup, pick-order proxy where available, player champion-game history, and repeated-player identifiers. Then fit a hierarchical model with matchup effects plus player familiarity and rank controls, and compare adjusted matchup estimates with the current aggregates.

## Data Sources

- OP.GG aggregate champion and matchup files, URLs and retrieval dates recorded in `selection_bias_sources.csv`.
- LoLalytics pick rate and breadth/depth extract, with its mixed population scopes explicitly retained.
- Riot Match-V5 documentation: https://developer.riotgames.com/apis#match-v5 (proposed, not used).

## Interpretation Guardrail

These diagnostics do not causally correct selection bias and do not replace the optimizer. They identify dependence, concentration, instability, and generalizability warnings around its inputs.
