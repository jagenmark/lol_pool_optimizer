# League of Legends Midlane Champion Pool Optimizer — v1 Spec

## Goal
Build a prototype that recommends an optimal champion pool for midlane in League of Legends.

The tool should:
- take a user-selected set of candidate champions
- take a desired pool size `n`
- use matchup data and enemy matchup frequencies
- return the best pool according to a clearly defined score

## Scope for v1
Included:
- Midlane only
- One fixed dataset
- One rank interval
- One patch window
- User selects candidate champions
- User selects pool size `n`
- Matchup-based scoring
- Blind pick recommendation
- Pool recommendation
- Simple counterpick mapping
- Top alternative pools

Excluded:
- Live client integration
- Scraping / API automation
- Personal mastery or user history model
- Full draft / team composition modeling
- Multi-lane support
- Advanced UI in the first step
- Mandatory banrate adjustment in v1
- Advanced shrinkage in v1 implementation

## Data Inputs

### 1. matchup_data.csv
Expected columns:
- `champion_i`: champion being evaluated
- `champion_j`: enemy champion
- `games_ij`: number of observed games for matchup i vs j
- `wins_i`: number of wins for champion i in matchup i vs j
- `winrate_ij`: observed winrate for i vs j

Interpretation:
- `W(i,j)` is the matchup performance of champion `i` into enemy champion `j`
- in v1, this can be taken from `winrate_ij` directly or from a prepared/preprocessed value

### 2. enemy_frequency.csv
Expected columns:
- `champion_j`
- `count_j`
- `freq_j`

Interpretation:
- `freq_j` should represent enemy champion `j`'s relative frequency among all relevant observed midlane matchups
- if frequencies do not sum to 1 exactly, normalize them

### 3. optional banrates.csv
Not required in v1, but reserved for later use

## Core Mathematical Definitions

### Raw matchup winrate
For matchup `(i,j)`:
- `wins_i = w_ij`
- `games_ij = n_ij`

Raw observed matchup winrate:
`p_hat_ij = w_ij / n_ij`

In v1, we may use raw or preprocessed matchup values.
Shrinkage / Empirical Bayes is a future extension.

### Matchup matrix
Define:
- `W(i,j)` = performance of champion `i` into enemy champion `j`

This is represented internally as a lookup or matrix built from the matchup dataset.

### Enemy matchup frequencies
Let:
- `N_j` = number of observed occurrences of enemy champion `j`
- `N = sum_j N_j`

Then:
`f(j) = N_j / N`

This means enemy champions are weighted by their share of all relevant observed midlane matchups.

### Blind pick score
For a single champion `i`:
`BlindScore(i) = sum_j f(j) * W(i,j)`

Interpretation:
- a weighted average of champion `i`'s performance against the relevant enemy population

### Pool score
For a pool `S`:
`Score(S) = sum_j f(j) * max_{i in S} W(i,j)`

Interpretation:
- for each enemy champion `j`, use the champion in the pool that performs best into that matchup
- then weight by how common that enemy champion is

### Optimization problem
Let:
- `C` = user-selected candidate set
- `n` = desired pool size

Then the goal is:
- choose `S` such that `S subseteq C` and `|S| = n`
- maximize `Score(S)`

Formally:
`S* = argmax_{S subseteq C, |S| = n} Score(S)`

## Required Outputs
The prototype should output:
1. Best pool of size `n`
2. Best blind pick among candidates
3. Top 5 pools by score
4. Counterpick table:
   - for each enemy champion `j`
   - show which champion in the chosen pool maximizes `W(i,j)`

## Functional Requirements
1. Load CSV files
2. Validate required columns
3. Normalize enemy frequencies if needed
4. Validate that selected candidates exist in the data
5. Validate that `n <= len(C)`
6. Compute blind scores
7. Generate all candidate pools of size `n` using brute force
8. Compute scores for all pools
9. Sort and display results clearly

## Architecture Requirements
The code should be modular so that the following can be added later:
- shrinkage / Empirical Bayes preprocessing
- banrate adjustment
- risk-adjusted blind pick scoring
- Streamlit or GUI frontend

## Notes for Future Extensions
Possible future additions:
- statistical shrinkage of matchup values
- Empirical Bayes / Beta-Binomial preprocessing
- banrate-adjusted matchup frequencies
- robust blind-pick score
- stability analysis
- automated data ingestion
