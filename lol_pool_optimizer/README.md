# League of Legends Midlane Champion Pool Optimizer

This project is a Python v1 prototype for recommending a midlane champion pool from a user-selected candidate set.

The prototype uses:
- prepared matchup values `W(i, j)` from the cleaned real matchup file
- enemy champion frequencies `f(j)` from the prepared clean enemy frequency file
- brute-force search over all pools of size `n`

It is intentionally simple, readable, and modular so future preprocessing and UI layers can be added without rewriting the optimizer core.

## Project Structure

```text
lol_pool_optimizer/
|-- data/
|   |-- clean/
|   |   |-- enemy_freq_df.csv
|   |   |-- opgg_mid_champion_summary.csv
|   |   `-- opgg_mid_matchups_clean.csv
|   |-- matchup_data.csv
|   |-- enemy_frequency.csv
|   `-- banrates.csv
|-- src/
|   |-- data_loader.py
|   |-- scoring.py
|   |-- optimizer.py
|   |-- utils.py
|   `-- main.py
|-- outputs/
|-- spec.md
|-- README.md
`-- requirements.txt
```

## v1 Formulas

`W(i, j)` is the matchup performance of champion `i` into enemy champion `j`.

`f(j)` is the normalized frequency of enemy champion `j` in the target midlane population.

Blind pick score:

```text
BlindScore(i) = sum_j f(j) * W(i, j)
```

Pool score:

```text
Score(S) = sum_j f(j) * max_{i in S} W(i, j)
```

Optimization objective:

```text
Choose S such that S is a subset of C and |S| = n, maximizing Score(S).
```

## How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the prototype from the project root:

```bash
python src/main.py
```

3. Optionally provide your own candidate list and pool size:

```bash
python src/main.py --candidates Ahri Syndra Orianna Viktor Yone --pool-size 3
```

You can also pass comma-separated candidates:

```bash
python src/main.py --candidates "Ahri,Syndra,Orianna,Viktor,Yone" --pool-size 2
```

4. Use the synthetic fallback dataset only when you explicitly want it:

```bash
python src/main.py --dataset synthetic
```

## What The Prototype Returns

The CLI prints:
- dataset mode
- number of champions loaded
- number of matchup rows loaded
- whether enemy frequencies were present or derived
- normalized enemy frequencies
- blind scores for each candidate
- total number of brute-force pools considered
- best pool
- best blind pick
- top 5 pools by score
- a counterpick table for the best pool

## Default Input Files

Normal execution now uses the cleaned real files:
- [opgg_mid_matchups_clean.csv](C:\Users\gosee\Documents\codex\lol_pool_optimizer\data\clean\opgg_mid_matchups_clean.csv)
- [enemy_freq_df.csv](C:\Users\gosee\Documents\codex\lol_pool_optimizer\data\clean\enemy_freq_df.csv)

The loader adapts those file schemas internally instead of requiring the files to be renamed or rewritten.

Enemy frequencies are now loaded directly from the prepared frequency file and validated against the expected schema:
- `champion_j`
- `f_j`
- optional `enemy_total_games`

The cleaned matchup file may still have a few impossible or missing rows. Self-matchups are never scored, and any other singular missing matchup rows are skipped during scoring with frequency renormalization over the remaining scorable enemy rows. Champions are not excluded just because one matchup row is missing.

The synthetic sample files remain available only as an optional fallback for testing:
- [matchup_data.csv](C:\Users\gosee\Documents\codex\lol_pool_optimizer\data\matchup_data.csv)
- [enemy_frequency.csv](C:\Users\gosee\Documents\codex\lol_pool_optimizer\data\enemy_frequency.csv)

`banrates.csv` is still unused in v1.

## Design Notes

- `data_loader.py` handles schema validation, cleaned-file column mapping, frequency derivation, normalization, and matchup lookup creation.
- `data_loader.py` handles schema validation, cleaned-file column mapping, loading the prepared enemy frequency file, merging `f_j` into matchup rows, and matchup lookup creation.
- `scoring.py` contains blind score, pool score, and counterpick logic.
- `optimizer.py` handles brute-force pool generation and ranking.
- `main.py` is the CLI entry point and prints sanity-check output.

## Validation And Sanity Checks

The prototype checks:
- required CSV columns exist
- the prepared enemy frequency file contains `champion_j` and `f_j`
- `sum(f_j)` is already normalized in the prepared frequency file
- matchup rows are unique by `(champion_i, champion_j)`
- frequencies are non-negative and normalized to sum to 1
- cleaned percentage-style values are converted to 0-1 internally
- `winrate_ij` stays inside `[0, 1]`
- candidate champions exist in the matchup dataset
- `n` is positive and `n <= len(C)`
- every candidate has matchup coverage for every enemy champion in the frequency table

## Assumptions

- `winrate_ij` is already prepared and usable as `W(i, j)` for v1.
- The prepared enemy frequency file is the source of enemy champion weights.
- Only midlane data is modeled.
- One fixed patch window and one fixed rank interval are assumed.
- The brute-force search is appropriate only for relatively small candidate sets.

## Limitations

- No shrinkage or Empirical Bayes preprocessing yet.
- No banrate adjustment yet.
- No team composition or draft-sequence modeling.
- No personal player data or champion mastery model.
- No GUI or Streamlit interface yet.

## TODOs

- TODO: Add shrinkage / Empirical Bayes preprocessing as a separate module.
- TODO: Add optional banrate adjustment as a pluggable step before optimization.
- TODO: Add a Streamlit UI on top of the CLI workflow.
