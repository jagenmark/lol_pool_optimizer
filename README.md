# League Stats Extractors

This workspace contains League of Legends data extractors for both OP.GG and LoLalytics.

## Install

```powershell
py -m pip install -r requirements.txt
```

## LoLalytics midlane pick rate + main rate

Files:

- `inspect_lolalytics_mid_single.py`
- `lolalytics_mid_extractor.py`
- `data/lolalytics_mid_pickrate_mainrate.csv`

Run the inspection step first:

```powershell
py inspect_lolalytics_mid_single.py
```

Run the full extractor:

```powershell
py lolalytics_mid_extractor.py
```

What it extracts:

- `champion_name`
- `champion_name_normalized`
- `lane`
- `patch`
- `rank`
- `population_scope`
- `pick_rate`
- `main_rate`
- `source_url`
- `extraction_date`

Implementation notes:

- The LoLalytics tier list does not expose the full mid list cleanly in server HTML, so the extractor uses Playwright with the locally installed browser to read the rendered list.
- `pick_rate` comes from the rendered midlane tier list at `lane=middle&tier=platinum_plus`.
- The closest available specialization metric is LoLalytics `Depth (Games per player)`, extracted from the `Normalised Champion Ranked Player Base` chart tooltip on each champion page.
- LoLalytics serves that depth tooltip as `7 Days, All Ranks, All Regions` even when the page URL includes `tier=platinum_plus`. The extractor preserves that limitation explicitly in `population_scope`.
- Percentages are written as percentage points, not fractions. Example: `13.12` means `13.12%`.

## OP.GG mid Platinum+ extractor

Files:

- `opgg_mid_extractor.py`
- `data/opgg_mid_champion_summary.csv`
- `data/opgg_mid_matchups.csv`

Run:

```powershell
py opgg_mid_extractor.py
```

Implementation notes:

- The OP.GG extractor uses server-rendered HTML rather than browser automation.
- It targets `region=global`, `tier=platinum_plus`, and `position=mid`.
- Champion names are normalized to lowercase ASCII alphanumerics with punctuation removed and `&` converted to `and`.
