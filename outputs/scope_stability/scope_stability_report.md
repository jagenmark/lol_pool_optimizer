# Scope Stability Report

## Technical Summary

- Discovered and validated **12 cumulative patch/rank scopes**.
- All scopes use the EB estimator with a fixed common candidate universe of **49 champions** and deterministic exact search.
- Mean pairwise Jaccard similarity of best pools is **0.215**; the exact best-pool match rate across scope pairs is **1.5%**.
- Mean best-pool Jaccard is **0.246 across ranks within the same patch** and **0.347 across patches within the same cumulative rank label**.
- The results strengthen the selection-bias concern as a robustness issue because recommendations change materially with patch/rank scope. They do not establish that selection bias caused the instability.
- These are descriptive robustness/stability diagnostics. They do not causally correct selection bias.

## Best EB Pools Vary Across Cumulative Scopes

- `16.05__plat_plus__cumulative`: **Sion, Vex, Xerath** (54.065%).
- `16.06__plat_plus__cumulative`: **Sion, Swain, Xerath** (53.973%).
- `16.07__plat_plus__cumulative`: **Pantheon, Sion, Xerath** (54.231%).
- `16.07__emerald_plus__cumulative`: **Ekko, Pantheon, Sion** (53.779%).
- `16.10__plat_plus__cumulative`: **Naafiri, Sion, Xerath** (54.002%).
- `16.10__emerald_plus__cumulative`: **Katarina, Pantheon, Sion** (53.471%).
- `16.10__diamond_plus__cumulative`: **Fizz, Pantheon, Veigar** (53.530%).
- `16.10__master_plus__cumulative`: **Fizz, Pantheon, Twisted Fate** (53.670%).
- `16.11__plat_plus__cumulative`: **Fizz, Sion, Vex** (53.792%).
- `16.11__emerald_plus__cumulative`: **Fizz, Sion, Vex** (53.509%).
- `16.11__diamond_plus__cumulative`: **Diana, Pantheon, Talon** (53.374%).
- `16.11__master_plus__cumulative`: **Katarina, Talon, Vex** (53.601%).

## Sion And Pantheon Dependence

- **Sion:** best-pool member in 8/12 scopes; median top-100 inclusion 71.000%; median exclusion loss 0.219%; top-matchup stability `unstable` (mean pairwise Jaccard 0.294).
- **Pantheon:** best-pool member in 6/12 scopes; median top-100 inclusion 15.500%; median exclusion loss 0.008%; top-matchup stability `unstable` (mean pairwise Jaccard 0.282).

Exclusion loss is the deterministic EB score difference between the unrestricted best pool and the best pool after removing the named champion(s). A zero loss means the unrestricted optimum remains feasible.

## High-Rank Scopes Carry More Shrinkage Dependence

- `16.07__emerald_plus__cumulative`: **moderate** warning; median matchup games 236, 25.3% below alpha.
- `16.10__diamond_plus__cumulative`: **moderate** warning; median matchup games 369, 13.5% below alpha.
- `16.10__master_plus__cumulative`: **high** warning; median matchup games 131, 39.7% below alpha.
- `16.11__diamond_plus__cumulative`: **moderate** warning; median matchup games 233, 23.8% below alpha.
- `16.11__master_plus__cumulative`: **high** warning; median matchup games 105, 48.0% below alpha.

Warnings compare observed matchup games with the EB prior strength. They describe sampling support and shrinkage dependence, not total model error.

## Cumulative Scopes Are Not Disjoint Rank Buckets

- Disjoint subtraction status: **not safe**.
- Unsafe: matchup files contain rounded win rates and matchup games but no exact win counts, so inferred wins are not exactly additive; 16.10 master_plus contains 147 matchup keys absent from diamond_plus, consistent with minimum-game row censoring; 16.11 emerald_plus contains 99 matchup keys absent from plat_plus, consistent with minimum-game row censoring; 16.11 master_plus contains 85 matchup keys absent from diamond_plus, consistent with minimum-game row censoring.
- Therefore `Plat+`, `Emerald+`, `Diamond+`, and `Master+` are reported as overlapping cumulative populations. No Plat-only, Emerald-only, or Diamond-only estimates were generated.

## Top 10 Pools Per Scope

- `16.05__plat_plus__cumulative`: 1. Sion, Vex, Xerath (54.065%); 2. Aurelion Sol, Sion, Vex (53.819%); 3. Sion, Veigar, Xerath (53.805%); 4. Annie, Sion, Xerath (53.796%); 5. Naafiri, Sion, Xerath (53.788%); 6. Pantheon, Sion, Xerath (53.775%); 7. Malzahar, Sion, Xerath (53.743%); 8. Aurelion Sol, Sion, Xerath (53.680%); 9. Anivia, Sion, Xerath (53.667%); 10. Lux, Sion, Xerath (53.663%)
- `16.06__plat_plus__cumulative`: 1. Sion, Swain, Xerath (53.973%); 2. Sion, Vex, Xerath (53.973%); 3. Fizz, Sion, Swain (53.920%); 4. Naafiri, Sion, Xerath (53.906%); 5. Pantheon, Sion, Xerath (53.899%); 6. Aurelion Sol, Sion, Vex (53.884%); 7. Aurelion Sol, Naafiri, Sion (53.874%); 8. Aurelion Sol, Fizz, Sion (53.857%); 9. Annie, Sion, Xerath (53.846%); 10. Naafiri, Pantheon, Sion (53.834%)
- `16.07__plat_plus__cumulative`: 1. Pantheon, Sion, Xerath (54.231%); 2. Aurelion Sol, Pantheon, Sion (54.183%); 3. Pantheon, Sion, Vel'Koz (54.110%); 4. Fizz, Pantheon, Sion (54.101%); 5. Katarina, Pantheon, Sion (54.050%); 6. Akshan, Pantheon, Sion (54.036%); 7. Aurelion Sol, Sion, Vex (53.995%); 8. Akshan, Sion, Xerath (53.966%); 9. Sion, Vex, Xerath (53.964%); 10. Naafiri, Pantheon, Sion (53.962%)
- `16.07__emerald_plus__cumulative`: 1. Ekko, Pantheon, Sion (53.779%); 2. Fizz, Malzahar, Sion (53.762%); 3. Fizz, Pantheon, Sion (53.759%); 4. Katarina, Malzahar, Sion (53.737%); 5. Fizz, Sion, Veigar (53.728%); 6. Katarina, Pantheon, Sion (53.722%); 7. Akshan, Aurelion Sol, Sion (53.693%); 8. Aurelion Sol, Katarina, Sion (53.675%); 9. Pantheon, Sion, Ziggs (53.667%); 10. Akshan, Sion, Veigar (53.654%)
- `16.10__plat_plus__cumulative`: 1. Naafiri, Sion, Xerath (54.002%); 2. Sion, Vex, Xerath (53.884%); 3. Malphite, Sion, Xerath (53.825%); 4. Pantheon, Sion, Xerath (53.791%); 5. Malzahar, Sion, Xerath (53.766%); 6. Fizz, Sion, Xerath (53.741%); 7. Fizz, Naafiri, Sion (53.721%); 8. Annie, Sion, Xerath (53.717%); 9. Sion, Swain, Xerath (53.695%); 10. Fizz, Sion, Vex (53.687%)
- `16.10__emerald_plus__cumulative`: 1. Katarina, Pantheon, Sion (53.471%); 2. Pantheon, Sion, Xerath (53.449%); 3. Fizz, Pantheon, Sion (53.419%); 4. Sion, Vex, Xerath (53.417%); 5. Annie, Sion, Xerath (53.417%); 6. Fizz, Sion, Vex (53.415%); 7. Naafiri, Sion, Xerath (53.388%); 8. Katarina, Sion, Xerath (53.387%); 9. Fizz, Sion, Swain (53.369%); 10. Sion, Twisted Fate, Xerath (53.340%)
- `16.10__diamond_plus__cumulative`: 1. Fizz, Pantheon, Veigar (53.530%); 2. Aurelion Sol, Fizz, Pantheon (53.514%); 3. Katarina, Pantheon, Veigar (53.509%); 4. Brand, Fizz, Pantheon (53.487%); 5. Fizz, Pantheon, Twisted Fate (53.446%); 6. Katarina, Pantheon, Xerath (53.442%); 7. Katarina, Pantheon, Twisted Fate (53.434%); 8. Cassiopeia, Fizz, Pantheon (53.422%); 9. Katarina, Pantheon, Zoe (53.422%); 10. Fizz, Katarina, Taliyah (53.416%)
- `16.10__master_plus__cumulative`: 1. Fizz, Pantheon, Twisted Fate (53.670%); 2. Ahri, Fizz, Pantheon (53.549%); 3. Ahri, Fizz, Katarina (53.514%); 4. Fizz, Twisted Fate, Vex (53.510%); 5. Fizz, Katarina, Twisted Fate (53.493%); 6. Katarina, Twisted Fate, Vex (53.484%); 7. Fizz, Katarina, Pantheon (53.482%); 8. Katarina, Twisted Fate, Zoe (53.481%); 9. Annie, Katarina, Twisted Fate (53.461%); 10. Fizz, Katarina, Vladimir (53.421%)
- `16.11__plat_plus__cumulative`: 1. Fizz, Sion, Vex (53.792%); 2. Naafiri, Sion, Vex (53.773%); 3. Naafiri, Sion, Vladimir (53.770%); 4. Fizz, Naafiri, Sion (53.748%); 5. Malphite, Naafiri, Sion (53.718%); 6. Sion, Vex, Xerath (53.683%); 7. Naafiri, Sion, Xerath (53.646%); 8. Fizz, Malzahar, Sion (53.632%); 9. Fizz, Pantheon, Sion (53.622%); 10. Annie, Naafiri, Sion (53.615%)
- `16.11__emerald_plus__cumulative`: 1. Fizz, Sion, Vex (53.509%); 2. Naafiri, Sion, Vex (53.507%); 3. Naafiri, Sion, Vladimir (53.482%); 4. Fizz, Naafiri, Sion (53.475%); 5. Naafiri, Pantheon, Sion (53.444%); 6. Lissandra, Naafiri, Sion (53.406%); 7. Malphite, Naafiri, Sion (53.399%); 8. Naafiri, Sion, Xerath (53.352%); 9. Katarina, Naafiri, Sion (53.351%); 10. Fizz, Sion, Vladimir (53.348%)
- `16.11__diamond_plus__cumulative`: 1. Diana, Pantheon, Talon (53.374%); 2. Diana, Naafiri, Pantheon (53.325%); 3. Diana, Pantheon, Vladimir (53.296%); 4. Diana, Katarina, Pantheon (53.296%); 5. Diana, Pantheon, Vel'Koz (53.235%); 6. Pantheon, Talon, Vladimir (53.209%); 7. Diana, Pantheon, Swain (53.195%); 8. Diana, Galio, Pantheon (53.172%); 9. Naafiri, Pantheon, Vladimir (53.168%); 10. Naafiri, Pantheon, Talon (53.166%)
- `16.11__master_plus__cumulative`: 1. Katarina, Talon, Vex (53.601%); 2. Galio, Katarina, Talon (53.556%); 3. Talon, Vel'Koz, Vex (53.547%); 4. Talon, Vex, Zoe (53.472%); 5. Katarina, Vex, Zoe (53.458%); 6. Katarina, Talon, Vladimir (53.456%); 7. Diana, Katarina, Talon (53.455%); 8. Diana, Talon, Vex (53.423%); 9. Katarina, Talon, Zoe (53.420%); 10. Aurelion Sol, Talon, Vex (53.414%)

## Method

- Scope discovery pairs matchup and champion-summary files by patch and rank, deduplicating prepared and aggregate representations.
- The 49-champion intersection is used in every scope so recommendation changes are not caused by candidate availability.
- Enemy weights come from prepared frequency files when available and otherwise from normalized aggregate opponent matchup counts.
- Matchup values use empirical Bayes shrinkage `(wins + alpha * mu) / (games + alpha)` with fractional wins inferred from the published aggregate win rate.
- Pool stability uses Jaccard similarity of deterministic best-pool sets. Focus-matchup stability uses Jaccard similarity of each champion's top marginal-contribution enemy sets while that champion is in the best pool.

## Limitations And Interpretation

- OP.GG aggregate rows are observational summaries and may reflect player specialization, pick timing, counterpick behavior, team composition, survivorship, and source-specific filtering.
- Rounded win rates imply fractional inferred wins. This is acceptable for EB robustness diagnostics but is not exact event-level reconstruction.
- Higher-rank scopes have smaller matchup samples and can be more sensitive to the prior even after shrinkage.
- Stability across overlapping cumulative scopes is not independent replication, because higher ranks are contained in lower rank scopes.
- No Riot Match-V5 or live Riot API data was used.

## Recommended Next Steps

- Treat pools that remain near the top across scopes as robust candidates, while inspecting close score gaps rather than over-reading rank order.
- Preserve cumulative-scope labels in downstream reporting and avoid describing them as rank-specific disjoint samples.
- If exact wins/losses and uncensored nested matchup keys become available, rerun the guarded subtraction path before considering disjoint buckets.

## Further Questions

- Are the same recommendations stable under alternative EB prior strengths?
- Do fixed candidate restrictions based on practical champion ownership or role suitability change Sion/Pantheon dependence?
