# Aggregate Method Sweep Report

## Executive Summary

- Fixed-policy posterior mean favors **Pantheon, Sion, Xerath** at **54.14%**.
- Oracle posterior mean favors **Pantheon, Sion, Vel'Koz** at **54.70%** and should be read as an upper-bound diagnostic.
- Residual-adjusted scoring favors **Mel, Naafiri, Sion** at **52.57%**.
- Worst-scope robust scoring favors **Katarina, Pantheon, Xerath (52.88%)**.
- Sion appears in 21/22 method-summary best pools. Pantheon appears in 18/22 method-summary best pools.

## Score Definitions

- `deterministic_eb`: `sum_j f_j max_i EB(W_ij)`.
- `fixed-policy`: choose `argmax_i posterior_mean_ij` once for each enemy, then simulate that locked policy.
- `oracle`: resample every matchup and then take the max in each draw; this is optimistic and not a practical policy.
- `offmeta_penalty`: deterministic score minus a transparent aggregate penalty from low pickrate, importance/pickrate ratio, and LoLalytics breadth/depth flags.
- `residual_adjusted`: two-way logit diagnostic with champion main effect removed, preserving enemy and matchup residual terms.

## Main Findings

- Sion is the most robust aggregate signal in the sweep when the method still rewards matchup coverage.
- Pantheon remains important in several methods but is more fragile than Sion under penalties, residual adjustment, and some cross-scope views.
- Enemy-frequency perturbation is mostly a meta-weight stress test; it cannot reveal player-selection bias in `W_ij`.
- The scope sweep used 12 local OP.GG aggregate scopes and no Riot API calls.

## Focus Champion Scope Stability

- **Sion:** best-pool member in 8/12 scope rows; median top-pool share 100.00%.
- **Pantheon:** best-pool member in 6/12 scope rows; median top-pool share 20.00%.

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

## Next Non-Riot-API Step

Fetch or locally archive a second aggregate matchup source with W_ij by patch/rank, then rerun this sweep with true cross-source matchup agreement rather than only cross-source pickrate/depth heuristics.
